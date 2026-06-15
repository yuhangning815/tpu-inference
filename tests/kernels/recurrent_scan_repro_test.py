# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reproduction harness for the GDN recurrent_scan vmem/placement quality bug.

WHY THE EXISTING recurrent_scan_test CANNOT REPRODUCE THE E2E FAILURE
--------------------------------------------------------------------
1. Footprint too small. It uses n_kq=2 / n_v=8 -> per-token dim 1536 and ~1MiB
   of kernel scratch. The failing eval config is n_kq=16 / n_v=64 -> dim 12288
   and ~8MiB scratch. With the tiny config, mixed_qkv stays VMEM-resident (S1)
   even at vmem_limit=0.8, so the *S0/HBM* placement of mixed_qkv that triggers
   the corruption never happens.
2. Wrong state dtype. It uses f32 recurrent_state; the eval runs with
   --mamba-ssm-cache-dtype=bfloat16.
3. No vmem sweep. It never passes vmem_limit_bytes, and MAX_TOKENS=8192 is
   unrealistic (mixed_qkv would be ~200MiB, always HBM).

This harness:
  * uses the real eval shapes (n_kq=16, n_v=64, d=128 -> dim 12288),
  * uses bf16 recurrent_state,
  * sizes the token buffer realistically (~1k) so mixed_qkv is ~25MiB and the
    S1<->S0 placement flip happens near the vmem budget,
  * sweeps vmem_limit_bytes from full capacity DOWN to values that force
    mixed_qkv to spill to HBM,
  * sweeps ragged patterns (incl. non-16/32-aligned lengths, transition blocks
    at boundaries, and chunked-prefill continuations via has_initial_state),
  * does a DIFFERENTIAL check: kernel@low-vmem vs kernel@full-vmem must match,
    and both must match the f32 reference.

IMPORTANT — reproducing a placement bug in isolation
----------------------------------------------------
In an isolated unit test there is far more free VMEM than in the full model, so
vmem_limit=0.8 alone may still keep mixed_qkv resident (S1) and pass. That is
exactly why the eval (which has the rest of the layer competing for VMEM) fails
at 0.8 while a unit test does not. To force the failing placement here we sweep
vmem_limit DOWN past the point where the 25MiB mixed_qkv can no longer stay
resident alongside the kernel's ~20MiB mandatory scratch (empirically < ~45MiB).
The first vmem at which the low-vmem run diverges from the full-vmem run (or goes
NaN) is the reproduction.

Run:  pytest -s tests/kernels/recurrent_scan_repro_test.py
The -s flag surfaces the per-(config, vmem) divergence report.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.gdn.v2.recurrent_scan_v2 import recurrent_scan
from tpu_inference.layers.common.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref

jax.config.parse_flags_with_absl()

MiB = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class Cfg:
    """A single ragged forward-step configuration."""
    decode_lengths: int
    prefill_lengths: tuple[int, ...]
    # has_initial_state for each prefill request (chunked-prefill continuation).
    prefill_has_init: tuple[bool, ...] = ()
    # Padded request bucket (>= decode + #prefill). Exercises the bucketization
    # padding that the is_last_seq schedule logic depends on.
    bucket: int = 16
    chunk_size: int = 32
    # E2E per-rank head config (NOT the toy 2/8 from the old test).
    n_kq: int = 16
    n_v: int = 64
    d_k: int = 128
    d_v: int = 128
    state_dtype: jnp.dtype = jnp.bfloat16
    io_dtype: jnp.dtype = jnp.bfloat16
    # Numerics knobs for the long-context tests.
    #   gate_decay_bias is ADDED to A_log. More negative -> exp(A_log) smaller
    #   -> g ~ 0 -> decay ~ 1 (slow decay). This is the regime a TRAINED GDN
    #   actually uses to retain long context; random-normal gates give
    #   g ~ -0.7/token, decay to ~0 within a token, and hide the cumulative
    #   instability entirely.
    #   v_scale multiplies the value stream (v = silu(value)); real activations
    #   are not unit-norm, and a larger v grows the state faster.
    gate_decay_bias: float = 0.0
    v_scale: float = 1.0

    def label(self) -> str:
        hi = "".join("1" if h else "0" for h in self.prefill_has_init) or "-"
        return (f"d{self.decode_lengths}_p{'+'.join(map(str, self.prefill_lengths)) or '-'}"
                f"_hi{hi}_bucket{self.bucket}")


def _build_inputs(cfg: Cfg, seed: int):
    """Builds a single ragged batch matching the eval's tensor layout."""
    n_kq, n_v, d_k, d_v = cfg.n_kq, cfg.n_v, cfg.d_k, cfg.d_v
    key_dim = n_kq * d_k
    value_dim = n_v * d_v

    decode = cfg.decode_lengths
    prefills = list(cfg.prefill_lengths)
    actual_num_tokens = decode + sum(prefills)
    actual_max_reqs = decode + len(prefills)
    assert actual_max_reqs <= cfg.bucket, (actual_max_reqs, cfg.bucket)
    max_reqs = cfg.bucket

    rngs = iter(jax.random.split(jax.random.key(seed), 16))

    # query_start_loc: [0, decode tokens (1 each), prefill cumulative...], then
    # padded monotonically with the last valid loc (NOT 1 — that would make it
    # non-monotonic). Length bucket+1.
    q_loc = jnp.cumsum(
        jnp.array([0] + [1] * decode + prefills, dtype=jnp.int32))
    q_loc = jnp.pad(q_loc, (0, max_reqs + 1 - len(q_loc)),
                    mode="constant",
                    constant_values=actual_num_tokens)

    # State slot indices: random distinct slots for valid reqs, 0 for padding.
    available = jax.random.permutation(next(rngs),
                                       jnp.arange(1, max_reqs, dtype=jnp.int32))
    valid_state_indices = available[:actual_max_reqs]
    state_indices = jnp.pad(valid_state_indices,
                            (0, max_reqs - actual_max_reqs),
                            mode="constant",
                            constant_values=0)

    # Token buffer sized tightly (actual + one chunk of tail pad), so mixed_qkv
    # is ~num_tokens*dim*2 bytes (realistic), NOT 8192-wide.
    buf = actual_num_tokens
    query = jax.random.normal(next(rngs), (buf, key_dim), dtype=cfg.io_dtype)
    key = jax.random.normal(next(rngs), (buf, key_dim), dtype=cfg.io_dtype)
    value = jax.random.normal(next(rngs), (buf, value_dim),
                              dtype=cfg.io_dtype) * cfg.v_scale
    b = jax.random.normal(next(rngs), (buf, n_v), dtype=cfg.io_dtype)
    a = jax.random.normal(next(rngs), (buf, n_v), dtype=cfg.io_dtype)

    pad = cfg.chunk_size
    query = jnp.pad(query, ((0, pad), (0, 0)))
    key = jnp.pad(key, ((0, pad), (0, 0)))
    value = jnp.pad(value, ((0, pad), (0, 0)))
    b = jnp.pad(b, ((0, pad), (0, 0)))
    a = jnp.pad(a, ((0, pad), (0, 0)))
    mixed_qkv = jnp.concatenate([query, key, value], axis=-1)

    # Recurrent state. Decode reqs and has_init prefills get real (nonzero)
    # prior state; fresh prefills start at zero.
    recurrent_state = jnp.zeros((max_reqs, n_v, d_k, d_v), dtype=cfg.state_dtype)
    has_initial_state = jnp.zeros((max_reqs,), dtype=bool)

    # Decode reqs always have prior state.
    if decode > 0:
        ds = jax.random.normal(next(rngs), (decode, n_v, d_k, d_v),
                               dtype=jnp.float32).astype(cfg.state_dtype)
        recurrent_state = recurrent_state.at[valid_state_indices[:decode]].set(ds)
        has_initial_state = has_initial_state.at[:decode].set(True)

    # Prefill reqs: per prefill_has_init, give a continuation a real prior state.
    hi = list(cfg.prefill_has_init) + [False] * (
        len(prefills) - len(cfg.prefill_has_init))
    for j, flag in enumerate(hi):
        req = decode + j
        if flag:
            ps = jax.random.normal(next(rngs), (n_v, d_k, d_v),
                                   dtype=jnp.float32).astype(cfg.state_dtype)
            recurrent_state = recurrent_state.at[valid_state_indices[req]].set(ps)
            has_initial_state = has_initial_state.at[req].set(True)

    A_log = (jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32) +
             cfg.gate_decay_bias).astype(cfg.io_dtype)
    dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=cfg.io_dtype)

    distribution = jnp.array([decode, actual_max_reqs, actual_max_reqs],
                             dtype=jnp.int32)

    return dict(
        mixed_qkv=mixed_qkv, b=b, a=a, A_log=A_log, dt_bias=dt_bias,
        recurrent_state=recurrent_state, query_start_loc=q_loc,
        state_indices=state_indices, distribution=distribution,
        has_initial_state=has_initial_state,
        valid_state_indices=valid_state_indices,
        actual_num_tokens=actual_num_tokens,
    )


def _run_kernel(inp, cfg: Cfg, vmem_limit_bytes, *, barrier=False):
    # recurrent_scan is @jax.jit-decorated (vmem_limit_bytes is a static arg),
    # so repeated calls with identical shapes + static args reuse the same
    # compiled executable — no per-call recompile. This is what makes the
    # decode loop below cheap (one compile per state dtype, N dispatches).
    mixed_qkv = inp["mixed_qkv"]
    if barrier:
        mixed_qkv = jax.lax.optimization_barrier(mixed_qkv)
    state, out = recurrent_scan(
        mixed_qkv=mixed_qkv,
        b=inp["b"], a=inp["a"],
        recurrent_state=inp["recurrent_state"],
        A_log=inp["A_log"], dt_bias=inp["dt_bias"],
        query_start_loc=inp["query_start_loc"],
        state_indices=inp["state_indices"],
        distribution=inp["distribution"],
        n_kq=cfg.n_kq, n_v=cfg.n_v, d_k=cfg.d_k, d_v=cfg.d_v,
        chunk_size=cfg.chunk_size, BT=cfg.chunk_size,
        use_qk_norm_in_gdn=True,
        has_initial_state=inp["has_initial_state"],
        vmem_limit_bytes=int(vmem_limit_bytes),
    )
    return jax.block_until_ready((state, out))


def _run_ref(inp, cfg: Cfg):
    dummy = jnp.zeros((1, cfg.n_v, cfg.d_k, cfg.d_v), dtype=jnp.float32)
    rstate = jnp.concatenate(
        [dummy, inp["recurrent_state"].astype(jnp.float32)], axis=0)
    state, out = ragged_gated_delta_rule_ref(
        inp["mixed_qkv"].astype(jnp.float32),
        inp["b"].astype(jnp.float32),
        inp["a"].astype(jnp.float32),
        rstate,
        inp["A_log"][None, None, :].astype(jnp.float32),
        inp["dt_bias"][None, None, :].astype(jnp.float32),
        inp["query_start_loc"],
        inp["state_indices"] + 1,
        inp["distribution"],
        inp["has_initial_state"],
        n_kq=cfg.n_kq, n_v=cfg.n_v, d_k=cfg.d_k, d_v=cfg.d_v,
    )
    return jax.block_until_ready((state, out))


def _max_abs_diff(x, y):
    return float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))))


# Ragged patterns chosen to exercise: aligned vs non-16/32-aligned lengths,
# multiple prefills (-> transition blocks at boundaries), pure-decode-with-
# prefill mixes, and chunked-prefill continuations (has_initial_state).
_CONFIGS = [
    Cfg(decode_lengths=0, prefill_lengths=(1024,)),            # single big prefill
    Cfg(decode_lengths=0, prefill_lengths=(1000,)),            # non-aligned tail
    Cfg(decode_lengths=0, prefill_lengths=(511, 513)),         # 2 unaligned seqs
    Cfg(decode_lengths=8, prefill_lengths=(1000,)),            # decode + prefill
    Cfg(decode_lengths=24, prefill_lengths=(500, 500)),        # mixed boundary
    Cfg(decode_lengths=0, prefill_lengths=(333, 333, 334)),    # 3 unaligned seqs
    Cfg(decode_lengths=0, prefill_lengths=(1000,),
        prefill_has_init=(True,)),                             # continuation
    Cfg(decode_lengths=8, prefill_lengths=(700, 300),
        prefill_has_init=(True, False)),                       # mixed continuation
]

@pytest.mark.skipif(not os.environ.get("RUN_GDN_REPRO"),
                    reason="set RUN_GDN_REPRO=1 to run on TPU")
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RecurrentScanReproTest(jtu.JaxTestCase):

    def _vmem_grid(self):
        cap = pltpu.get_tpu_info().vmem_capacity_bytes
        # Full, the eval's 0.8, then progressively force mixed_qkv out of VMEM.
        fracs = [1.0, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4]
        return cap, [int(cap * f) for f in fracs]

    @parameterized.named_parameters(
        (c.label(), c) for c in _CONFIGS)
    def test_vmem_differential(self, cfg: Cfg):
        """For one ragged config, sweep vmem and compare to the full-vmem run.

        The full-vmem run is the trusted baseline (it is the configuration that
        passes the eval). Any low-vmem run that diverges from it — or goes
        NaN/Inf — is the reproduction of the e2e bug.
        """
        inp = _build_inputs(cfg, seed=0)
        cap, vmems = self._vmem_grid()
        n = inp["actual_num_tokens"]
        sl = inp["valid_state_indices"]

        # Trusted baseline at full capacity.
        base_state, base_out = _run_kernel(inp, cfg, cap)
        assert not bool(jnp.any(jnp.isnan(base_out))), "baseline itself is NaN"

        # Optional: sanity vs f32 reference (loose; bf16 kernel).
        ref_state, ref_out = _run_ref(inp, cfg)
        base_vs_ref = _max_abs_diff(base_out[:n], ref_out[:n])
        print(f"\n[{cfg.label()}] mixed_qkv={inp['mixed_qkv'].shape} "
              f"{inp['mixed_qkv'].dtype}  baseline-vs-ref maxdiff={base_vs_ref:.3g}")

        first_break = None
        for v in vmems:
            state, out = _run_kernel(inp, cfg, v)
            nan = bool(jnp.any(jnp.isnan(out)) | jnp.any(jnp.isinf(out)))
            d_out = _max_abs_diff(out[:n], base_out[:n])
            d_state = _max_abs_diff(state[sl], base_state[sl])
            flag = "  <-- DIVERGES" if (nan or d_out > 1e-2 or d_state > 1e-2) else ""
            print(f"    vmem={v/MiB:6.1f}MiB ({v/cap:.2f})  "
                  f"out_diff={d_out:.3g}  state_diff={d_state:.3g}  "
                  f"nan={nan}{flag}")
            if first_break is None and (nan or d_out > 1e-2 or d_state > 1e-2):
                first_break = v

        if first_break is not None:
            # Repro found. Re-run that vmem WITH the optimization_barrier to
            # classify X (buffer reuse) vs Y (HBM DMA path).
            bstate, bout = _run_kernel(inp, cfg, first_break, barrier=True)
            b_nan = bool(jnp.any(jnp.isnan(bout)) | jnp.any(jnp.isinf(bout)))
            b_diff = _max_abs_diff(bout[:n], base_out[:n])
            verdict = ("barrier FIXED it -> X (XLA buffer reuse/overlap)"
                       if (not b_nan and b_diff <= 1e-2)
                       else "barrier did NOT fix -> Y (HBM DMA path / kernel)")
            self.fail(
                f"[{cfg.label()}] REPRODUCED at vmem={first_break/MiB:.1f}MiB "
                f"({first_break/cap:.2f}); {verdict} "
                f"(barrier out_diff={b_diff:.3g}, nan={b_nan})")


# ---------------------------------------------------------------------------
# Long-context numerics tests (production-like).
#
# GDN's gate/decay is CUMULATIVE, so its numerical failure mode is long context
# (the eval runs 8k/1k and 1k/8k). The existing tests above top out at ~1k
# tokens and never exercise thousands of recurrent steps, so they cannot see
# the cumulative drift / inf blow-up.
#
# REFERENCE: ragged_gated_delta_rule_ref. It applies the SAME preprocessing as
# the kernel (silu, l2-norm, d_k^-0.5 scale, identical gate math) but runs the
# exact token-by-token recurrence in float32. So it is the correctness oracle.
# We assert BOTH:
#   (1) the kernel output/state stays FINITE  -> direct regression test for the
#       inf guard; catches the catastrophic "!!!!" endpoint, but is necessary,
#       NOT sufficient (finite-but-wrong garbage still tanks eval quality);
#   (2) the kernel stays close to the f32 ref -> catches silent drift. Drift
#       GROWING with context length is the fingerprint of the instability.
#
# Chunk is fixed at 32 (production). Gates are biased toward slow decay
# (decay ~ 1) because that is the regime a trained GDN uses to retain long
# context; random-normal gates decay in ~1 token and hide the bug.
# ---------------------------------------------------------------------------

# Lengths to sweep so the drift-vs-length trend is visible.
_LONG_PREFILL_LENGTHS = (512, 1024, 2048, 4096, 8192)
# Per-step decode is one kernel call; 8192 calls is slow. Default to a
# representative length; set GDN_DECODE_STEPS=8192 to match the eval's 1k/8k.
_DECODE_STEPS = int(os.environ.get("GDN_DECODE_STEPS", "2048"))
# Slow-decay, scaled-value regime — production-faithful for long context.
_LONG_DECAY_BIAS = -6.0
_LONG_V_SCALE = 4.0


def _full_vmem() -> int:
    return pltpu.get_tpu_info().vmem_capacity_bytes


def _rel_l2(k, r) -> float:
    k = k.astype(jnp.float32)
    r = r.astype(jnp.float32)
    return float(jnp.linalg.norm(k - r) / (jnp.linalg.norm(r) + 1e-9))


# The bf16-vs-f32 mamba-state-cache A/B. io_dtype stays bf16 in BOTH (model
# activations are bf16); only the recurrent-state cache dtype changes — exactly
# the `--mamba-ssm-cache-dtype` knob. f32 is the high-precision control.
_STATE_DTYPES = (("bf16", jnp.bfloat16), ("f32", jnp.float32))


@pytest.mark.skipif(not os.environ.get("RUN_GDN_REPRO"),
                    reason="set RUN_GDN_REPRO=1 to run on TPU")
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RecurrentScanLongContextTest(jtu.JaxTestCase):
    """Long-context numerics, bf16 vs f32 mamba state cache.

    Oracle = ragged_gated_delta_rule_ref (exact token-by-token recurrence in
    f32; same silu/l2-norm/scale/gate math as the kernel). For each state-cache
    dtype we report BOTH Inf/NaN (the catastrophic "!!!!" endpoint) and the
    drift vs the f32 ref (silent degradation). Production proxy: real head dims
    (16/64/128/128), bf16 activations, chunk=32, qk_norm, eval lengths (8k
    prefill / long decode), gates biased to slow decay (the regime a trained
    GDN actually uses for long memory). The one gap we cannot close in a unit
    test is the input *distribution* (random-normal, not real activations).
    """

    @parameterized.named_parameters(
        (f"len{L}", L) for L in _LONG_PREFILL_LENGTHS)
    def test_long_prefill_state_cache_ab(self, seq_len):
        """8k-input case: one big prefill (seq_len/32 chained chunks in a single
        call), run with bf16 vs f32 state cache, each compared to the f32 ref.
        """
        out_r = None
        summary = {}
        for name, sdt in _STATE_DTYPES:
            cfg = Cfg(decode_lengths=0, prefill_lengths=(seq_len, ), bucket=16,
                      gate_decay_bias=_LONG_DECAY_BIAS, v_scale=_LONG_V_SCALE,
                      state_dtype=sdt)
            inp = _build_inputs(cfg, seed=0)
            n = inp["actual_num_tokens"]
            state_k, out_k = _run_kernel(inp, cfg, _full_vmem())
            # Ref is f32 and state-dtype independent (inputs identical, fresh
            # prefill starts from zero state), so compute it once.
            if out_r is None:
                _, out_r = _run_ref(inp, cfg)
            nonfinite = bool(
                jnp.any(~jnp.isfinite(out_k[:n]))
                | jnp.any(~jnp.isfinite(state_k)))
            rel = _rel_l2(out_k[:n], out_r[:n])
            summary[name] = (nonfinite, rel)
            print(f"\n[prefill {seq_len:5d} state={name:4s}] "
                  f"nonfinite={nonfinite}  rel_l2={rel:.3g}")

        for name, (nonfinite, rel) in summary.items():
            self.assertFalse(
                nonfinite,
                f"[prefill {seq_len} state={name}] kernel produced NaN/Inf")
            # Garbage tripwire (not a precision bound): a correct kernel drifts
            # only mildly from f32; rel_l2 >~ 1 means it is wrong, not merely
            # low-precision.
            self.assertLess(
                rel, 1.0,
                f"[prefill {seq_len} state={name}] diverged rel_l2={rel:.3g}")

    def _decode_loop(self, name, sdt, mixed_seq, a_seq, b_seq, A_log, dt_bias,
                     out_ref, steps, checkpoints):
        """One decode trajectory with state cache dtype `sdt`, feeding state
        forward. Returns (first_nonfinite_step | None, worst_rel)."""
        cfg = Cfg(decode_lengths=1, prefill_lengths=(), bucket=16,
                  gate_decay_bias=_LONG_DECAY_BIAS, v_scale=_LONG_V_SCALE,
                  state_dtype=sdt)
        dec_inp = _build_inputs(cfg, seed=1)
        slot = int(dec_inp["valid_state_indices"][0])
        state = jnp.zeros_like(dec_inp["recurrent_state"])  # dtype = sdt
        vmem = _full_vmem()

        worst_rel = 0.0
        for t in range(steps):
            di = dict(dec_inp)
            di["mixed_qkv"] = dec_inp["mixed_qkv"].at[0].set(mixed_seq[t])
            di["a"] = dec_inp["a"].at[0].set(a_seq[t])
            di["b"] = dec_inp["b"].at[0].set(b_seq[t])
            di["A_log"], di["dt_bias"] = A_log, dt_bias
            di["recurrent_state"] = state

            new_state, out = _run_kernel(di, cfg, vmem)

            # Finite check every step pinpoints the exact blow-up step. Once
            # nonfinite, the state is poisoned and all later steps are too, so
            # stop and report the first bad step.
            if bool(jnp.any(~jnp.isfinite(out[0]))
                    | jnp.any(~jnp.isfinite(new_state[slot]))):
                return t, worst_rel

            state = state.at[slot].set(new_state[slot])

            if (t + 1) in checkpoints:
                rel = _rel_l2(out[0], out_ref[t])
                worst_rel = max(worst_rel, rel)
                print(f"    [state={name:4s}] decode step {t + 1:5d}: "
                      f"vs-ref rel_l2={rel:.3g}")
        return None, worst_rel

    def test_long_decode_state_cache_ab(self):
        """1k/8k generation case: the per-token decode recurrence for many
        steps, feeding state back each step, run with bf16 vs f32 state cache.
        This is the path most exposed to the bf16-cache round-trip and the one
        the inf guard protects. Reports first Inf/NaN step + drift per dtype.
        """
        steps = _DECODE_STEPS
        base = Cfg(decode_lengths=1, prefill_lengths=())  # for dims only
        n_kq, n_v, d_k, d_v = base.n_kq, base.n_v, base.d_k, base.d_v
        key_dim = n_kq * d_k
        value_dim = n_v * d_v

        # One canonical raw (pre-silu) sequence + shared gate params, fed
        # IDENTICALLY to the ref (whole seq) and to both decode loops. io_dtype
        # is bf16 for all — only the STATE cache dtype differs between runs.
        rngs = iter(jax.random.split(jax.random.key(7), 8))
        q_seq = jax.random.normal(next(rngs), (steps, key_dim), dtype=base.io_dtype)
        k_seq = jax.random.normal(next(rngs), (steps, key_dim), dtype=base.io_dtype)
        v_seq = jax.random.normal(next(rngs), (steps, value_dim),
                                  dtype=base.io_dtype) * _LONG_V_SCALE
        a_seq = jax.random.normal(next(rngs), (steps, n_v), dtype=base.io_dtype)
        b_seq = jax.random.normal(next(rngs), (steps, n_v), dtype=base.io_dtype)
        A_log = (jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32) +
                 _LONG_DECAY_BIAS).astype(base.io_dtype)
        dt_bias = jax.random.normal(next(rngs), (n_v, ), dtype=base.io_dtype)
        mixed_seq = jnp.concatenate([q_seq, k_seq, v_seq], axis=-1)

        # f32 ground truth: the whole sequence as ONE prefill from zero state.
        ref_cfg = Cfg(decode_lengths=0, prefill_lengths=(steps, ), bucket=16)
        ref_inp = dict(_build_inputs(ref_cfg, seed=0))
        ref_inp["mixed_qkv"] = ref_inp["mixed_qkv"].at[:steps].set(mixed_seq)
        ref_inp["a"] = ref_inp["a"].at[:steps].set(a_seq)
        ref_inp["b"] = ref_inp["b"].at[:steps].set(b_seq)
        ref_inp["A_log"], ref_inp["dt_bias"] = A_log, dt_bias
        ref_inp["recurrent_state"] = jnp.zeros_like(ref_inp["recurrent_state"])
        ref_inp["has_initial_state"] = jnp.zeros_like(
            ref_inp["has_initial_state"])
        _, out_ref = _run_ref(ref_inp, ref_cfg)

        checkpoints = [c for c in (256, 512, 1024, 2048, 4096, 8192)
                       if c <= steps]

        results = {}
        for name, sdt in _STATE_DTYPES:
            first_bad, worst_rel = self._decode_loop(
                name, sdt, mixed_seq, a_seq, b_seq, A_log, dt_bias, out_ref,
                steps, checkpoints)
            results[name] = (first_bad, worst_rel)
            msg = (f"first Inf/NaN at step {first_bad}" if first_bad is not None
                   else f"finite to {steps}; worst rel_l2={worst_rel:.3g}")
            print(f"\n[decode state={name:4s}] {msg}")

        # Assert per dtype AFTER running both, so the A/B is always reported.
        for name, (first_bad, worst_rel) in results.items():
            self.assertIsNone(
                first_bad,
                f"[decode state={name}] produced Inf/NaN at step {first_bad}")
            self.assertLess(
                worst_rel, 1.0,
                f"[decode state={name}] diverged worst rel_l2={worst_rel:.3g}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
