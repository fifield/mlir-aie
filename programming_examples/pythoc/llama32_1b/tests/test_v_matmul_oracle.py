#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Runtime oracle for the placed-IRON v_matmul_seg GEMM emit.

Compiles two versions of `rms_gemms_rope` and runs them on NPU with identical
synthetic inputs:

  - "cached"   : reference_mlir/rms_gemms_rope.npu.air.mlir as-is (V/K/Q GEMMs
                 are the inline vector.contract chain emitted by aircc).
  - "placed-v" : same cached MLIR but with `aie.device @v_matmul_seg`
                 replaced by the placed-IRON emit (uses bf16_gemm_kernel_bf16out
                 link), all other 5 inner devices + dispatcher cached.

Diffs the V output buffer (arg8) and a CPU reference. Used to classify the
runtime-garbage bug documented in beads PythoC-8ns.13.

Run with:   python3 tests/test_v_matmul_oracle.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

# Llama 3.2 1B prefill dims (rms_gemms_rope is fixed to these)
SEQ_LEN = 2048
EMB_DIM = 2048
KV_DIM = 512
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64


# ---------------------------------------------------------------------------
# Module builders
# ---------------------------------------------------------------------------

def _build_modules(mode: str = "v_only"):
    """Return (cached_text, placed_text).

    Modes:
      "v_only" -- cached rms_gemms_rope with only @v_matmul_seg placed.
      "vkq"    -- cached rms_gemms_rope with v/k/q_matmul_seg placed.
      "full"   -- full rms_gemms_rope placed-IRON build (all 7 devices).
      "og"     -- cached o_ffn with only @og_matmul_seg placed.
      "dg"     -- cached o_ffn with only @dg_matmul_seg placed.
      "gg"     -- cached o_ffn with only @gg_matmul_seg placed.
      "ug"     -- cached o_ffn with only @ug_matmul_seg placed.
    """
    from aie.extras.context import mlir_mod_ctx

    from builders.rms_gemms_rope import (
        _emit_v_matmul_seg,
        _emit_k_matmul_seg,
        _emit_q_matmul_seg,
        _extract_single_device,
        _splice_device,
        build_rms_gemms_rope_module,
    )
    from builders._emit import attach_loop_annotation_to_all_scf_for

    O_FFN_DEVICES = {"og", "dg", "gg", "ug"}

    if mode in O_FFN_DEVICES:
        from builders import o_ffn as _ofn
        emit_name = f"_emit_{mode}_matmul_seg"
        if not hasattr(_ofn, emit_name):
            raise NotImplementedError(
                f"builders/o_ffn.py does not yet expose {emit_name}; "
                "see the README's Deferred section."
            )
        emit_fn = getattr(_ofn, emit_name)
        cached_path = PROJECT_DIR / "reference_mlir" / "o_ffn.npu.air.mlir"
        cached_text = cached_path.read_text()
        with mlir_mod_ctx() as ctx:
            emit_fn()
            module = ctx.module
            attach_loop_annotation_to_all_scf_for(module)
        placed_text = module.operation.get_asm(assume_verified=True)
        placed_dev = _extract_single_device(placed_text, f"{mode}_matmul_seg")
        spliced = _splice_device(cached_text, f"{mode}_matmul_seg", placed_dev)
        return cached_text, spliced

    cached_path = PROJECT_DIR / "reference_mlir" / "rms_gemms_rope.npu.air.mlir"
    cached_text = cached_path.read_text()

    if mode == "full":
        return cached_text, build_rms_gemms_rope_module()

    with mlir_mod_ctx() as ctx:
        _emit_v_matmul_seg()
        if mode == "vkq":
            _emit_k_matmul_seg()
            _emit_q_matmul_seg()
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    placed_module_text = str(module)
    placed_v = _extract_single_device(placed_module_text, "v_matmul_seg")
    spliced = _splice_device(cached_text, "v_matmul_seg", placed_v)
    if mode == "vkq":
        placed_k = _extract_single_device(placed_module_text, "k_matmul_seg")
        placed_q = _extract_single_device(placed_module_text, "q_matmul_seg")
        spliced = _splice_device(spliced, "k_matmul_seg", placed_k)
        spliced = _splice_device(spliced, "q_matmul_seg", placed_q)
    return cached_text, spliced


# ---------------------------------------------------------------------------
# Synthetic inputs (host-arg layout matches llama32_1b_prefill.py:212-225)
# ---------------------------------------------------------------------------

def _synth_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)

    def rand_bf16(*shape, scale=0.02):
        a = (rng.standard_normal(size=shape) * scale).astype(np.float32)
        return a.astype(bfloat16)

    return [
        rand_bf16(SEQ_LEN, EMB_DIM),                                # 0 x_in
        rand_bf16(EMB_DIM, scale=1.0),                              # 1 rms gamma
        np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16),               # 2 normed_buf (out of rms_norm)
        rand_bf16(EMB_DIM, EMB_DIM),                                # 3 Wq
        np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16),               # 4 q_buf
        rand_bf16(EMB_DIM, KV_DIM),                                 # 5 Wk
        np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16),                # 6 k_buf
        rand_bf16(EMB_DIM, KV_DIM),                                 # 7 Wv
        np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16),                # 8 V OUTPUT
        rand_bf16(SEQ_LEN * N_HEADS * HEAD_DIM),                    # 9 rope_q_lut
        rand_bf16(SEQ_LEN * N_KV_HEADS * HEAD_DIM),                 # 10 rope_k_lut
        np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16),               # 11 q_roped_buf
        np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16),                # 12 k_roped_buf
    ]


# ---------------------------------------------------------------------------
# Compile + run a single rms_gemms_rope module
# ---------------------------------------------------------------------------

def _run(name, ir_text, inputs, cache_dir, verbose=False):
    from kernel_builder.cache import KernelCache

    cache = KernelCache(cache_dir=cache_dir, verbose=verbose)
    cache.compile_and_cache(name, ir_text, instance_name="rms_gemms_rope")

    results = cache.load_and_run(
        name,
        {},
        *inputs,
        output_indices=[4, 6, 8, 11, 12],
    )
    return {
        "q_pre": results[4].reshape(SEQ_LEN, EMB_DIM).copy(),
        "k_pre": results[6].reshape(SEQ_LEN, KV_DIM).copy(),
        "v":     results[8].reshape(SEQ_LEN, KV_DIM).copy(),
        "q_rop": results[11].reshape(SEQ_LEN, EMB_DIM).copy(),
        "k_rop": results[12].reshape(SEQ_LEN, KV_DIM).copy(),
    }


# ---------------------------------------------------------------------------
# CPU reference (V only)
# ---------------------------------------------------------------------------

def _cpu_v_reference(x_bf16, gamma_bf16, wv_bf16):
    """Compute V_ref = (RMSNorm(x) * gamma) @ Wv in f32, return as bf16-as-f32."""
    x_f32 = x_bf16.astype(np.float32)
    g_f32 = gamma_bf16.astype(np.float32)
    wv_f32 = wv_bf16.astype(np.float32)
    # RMSNorm with eps = 1e-6
    var = np.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    normed = x_f32 / np.sqrt(var + 1e-6) * g_f32[None, :]
    # bf16-cast normed to match what flows into the GEMM in cached
    normed_bf16 = normed.astype(bfloat16).astype(np.float32)
    v = normed_bf16 @ wv_f32
    return v.astype(bfloat16).astype(np.float32)


# ---------------------------------------------------------------------------
# Diff reporting
# ---------------------------------------------------------------------------

def _stats(label, a, b):
    a_f32 = a.astype(np.float32).flatten()
    b_f32 = b.astype(np.float32).flatten()
    diff = a_f32 - b_f32
    abs_diff = np.abs(diff)
    n = a_f32.size
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    n_zero_diff = int((abs_diff == 0).sum())
    n_a_zero = int((a_f32 == 0).sum())
    n_b_zero = int((b_f32 == 0).sum())
    # Pearson correlation
    if a_f32.std() > 0 and b_f32.std() > 0:
        corr = float(np.corrcoef(a_f32, b_f32)[0, 1])
    else:
        corr = float("nan")
    # Top mismatch sample (first 5 indices by abs_diff)
    if max_abs > 0:
        top_idx = np.argpartition(-abs_diff, kth=min(5, n - 1))[:5]
        top_idx = top_idx[np.argsort(-abs_diff[top_idx])]
        samples = [(int(i), float(a_f32[i]), float(b_f32[i])) for i in top_idx]
    else:
        samples = []

    print(f"\n  --- {label} ---")
    print(f"    elements        : {n}")
    print(f"    identical       : {n_zero_diff}")
    print(f"    max |a-b|       : {max_abs:.6f}")
    print(f"    mean |a-b|      : {mean_abs:.6f}")
    print(f"    corr(a, b)      : {corr:.6f}")
    print(f"    a==0 count      : {n_a_zero}")
    print(f"    b==0 count      : {n_b_zero}")
    if samples:
        print(f"    top mismatches (idx, a, b):")
        for idx, av, bv in samples:
            print(f"      [{idx:>8d}]  {av:+.6f}  vs  {bv:+.6f}")
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "corr": corr,
        "n_zero_diff": n_zero_diff,
        "n_a_zero": n_a_zero,
        "n_b_zero": n_b_zero,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# o_ffn synthetic inputs + runner (15-arg host signature)
# ---------------------------------------------------------------------------

O_FFN_HIDDEN_DIM = 8192


def _synth_o_ffn_inputs(seed: int = 0):
    """Synthetic 15-arg input list matching _o_ffn_host_arg_types.

    Arg layout verified against llama32_1b_prefill.py:306-326:
      arg0  X  (attention output, og input)            -- random
      arg1  W  (wo, O-projection weight)               -- random
      arg2  Y  (og output / ra_add input1)             -- zeros (written by og)
      arg3  X  (residual base, ra_add input2)          -- random
      arg4  Y  (ra_add output, rms_norm input)         -- zeros (written by ra_add)
      arg5  W  (ffn_norm gamma)                        -- random
      arg6  Y  (rms_norm output, gg/ug X-input)        -- zeros (written by rms_norm)
      arg7  W  (w_gate)                                -- random
      arg8  Y  (gate output, silu_mul input1)          -- zeros (written by gg)
      arg9  W  (w_up)                                  -- random (was zeros — bug!)
      arg10 Y  (up output, silu_mul input2)            -- zeros (written by ug)
      arg11 Y  (silu_mul output, dg X-input)           -- zeros (written by sw_silu_mul)
      arg12 W  (w_down)                                -- random
      arg13 Y  (dg output, fa_add input1)              -- zeros (written by dg)
      arg14 ?  (4M-elt scratch / flat work buffer)     -- zeros
    """
    rng = np.random.default_rng(seed)

    def rand_bf16(*shape, scale=0.02):
        a = (rng.standard_normal(size=shape) * scale).astype(np.float32)
        return a.astype(bfloat16)

    return [
        rand_bf16(EMB_DIM, EMB_DIM),                     # arg0  X (og input)
        rand_bf16(EMB_DIM, EMB_DIM),                     # arg1  Wo (og weight)
        np.zeros((EMB_DIM, EMB_DIM), dtype=bfloat16),    # arg2  og output
        rand_bf16(EMB_DIM, EMB_DIM),                     # arg3  residual base
        np.zeros((EMB_DIM, EMB_DIM), dtype=bfloat16),    # arg4  ra_add output
        rand_bf16(EMB_DIM, scale=1.0),                   # arg5  ffn_norm gamma
        np.zeros((EMB_DIM, EMB_DIM), dtype=bfloat16),    # arg6  rms_norm output
        rand_bf16(EMB_DIM, O_FFN_HIDDEN_DIM),             # arg7  W_gate
        np.zeros((EMB_DIM, O_FFN_HIDDEN_DIM), dtype=bfloat16),  # arg8  gate output
        rand_bf16(EMB_DIM, O_FFN_HIDDEN_DIM),             # arg9  W_up
        np.zeros((EMB_DIM, O_FFN_HIDDEN_DIM), dtype=bfloat16),  # arg10 up output
        np.zeros((EMB_DIM, O_FFN_HIDDEN_DIM), dtype=bfloat16),  # arg11 silu_mul output
        rand_bf16(O_FFN_HIDDEN_DIM, EMB_DIM),             # arg12 W_down
        np.zeros((EMB_DIM, EMB_DIM), dtype=bfloat16),    # arg13 dg output
        np.zeros((SEQ_LEN * 2048,), dtype=bfloat16),     # arg14 scratch
    ]


def _run_o_ffn(name, ir_text, inputs, cache_dir, verbose=False,
               output_indices=(2,)):
    from kernel_builder.cache import KernelCache

    cache = KernelCache(cache_dir=cache_dir, verbose=verbose)
    cache.compile_and_cache(name, ir_text, instance_name="o_ffn")
    results = cache.load_and_run(
        name,
        {},
        *inputs,
        output_indices=list(output_indices),
    )
    return {idx: results[idx].copy() for idx in output_indices}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default=str(PROJECT_DIR / "build_peano" / "oracle_cache"))
    p.add_argument("--skip-cached", action="store_true",
                   help="Reuse a previously-compiled cached run (skip recompile + rerun)")
    p.add_argument("--skip-placed", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--mode",
                   choices=["v_only", "vkq", "full", "og", "gg", "ug", "dg"],
                   default="v_only",
                   help="v_only: splice only @v_matmul_seg; "
                        "vkq: splice v/k/q_matmul_seg; "
                        "full: full placed-IRON rms_gemms_rope (all 7 devices); "
                        "og: splice only @og_matmul_seg into o_ffn cached MLIR; "
                        "gg: splice only @gg_matmul_seg into o_ffn cached MLIR; "
                        "ug: splice only @ug_matmul_seg into o_ffn cached MLIR; "
                        "dg: splice only @dg_matmul_seg into o_ffn cached MLIR")
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[oracle] workdir = {workdir}")
    print(f"[oracle] seed    = {args.seed}")

    print(f"[oracle] Building modules (mode={args.mode})...")
    cached_text, placed_text = _build_modules(mode=args.mode)
    print(f"[oracle]  cached      : {len(cached_text)} bytes")
    print(f"[oracle]  placed-{args.mode:8s}: {len(placed_text)} bytes "
          f"(delta {len(placed_text) - len(cached_text):+d})")

    # ---------------------------------------------------------------
    # o_ffn matmul modes (og, gg, ug, dg): diff only the matmul's
    # output buffer.  Host-arg layout (matches _o_ffn_host_arg_types):
    #     og_matmul_seg: X=arg0, W=arg1,  Y=arg2  (2048x2048)
    #     gg_matmul_seg: X=arg6, W=arg7,  Y=arg8  (2048x8192)
    #     ug_matmul_seg: X=arg6, W=arg9,  Y=arg10 (2048x8192)
    #     dg_matmul_seg: X=arg11,W=arg12, Y=arg13 (2048x2048)
    # ---------------------------------------------------------------
    if args.mode in ("og", "gg", "ug", "dg"):
        # Per-mode output arg index.
        out_arg_by_mode = {"og": 2, "gg": 8, "ug": 10, "dg": 13}
        out_arg = out_arg_by_mode[args.mode]

        print("[oracle] Generating synthetic o_ffn inputs...")
        inputs_cached = _synth_o_ffn_inputs(seed=args.seed)
        inputs_placed = _synth_o_ffn_inputs(seed=args.seed)
        assert np.array_equal(
            inputs_cached[0].view(np.int16), inputs_placed[0].view(np.int16))

        cached_dir = workdir / f"cached_{args.mode}"
        placed_dir = workdir / f"placed_{args.mode}"
        cached_dir.mkdir(parents=True, exist_ok=True)
        placed_dir.mkdir(parents=True, exist_ok=True)

        cached_results = None
        placed_results = None

        # Track the matmul output + a few neighbours for sanity.
        output_indices = tuple(sorted({2, 4, 6, 8, 10, 11, 13, 14, out_arg}))

        if not args.skip_cached:
            print(f"\n[oracle] === Run 1: o_ffn CACHED reference ===")
            cached_results = _run_o_ffn(
                f"o_ffn_cached", cached_text, inputs_cached,
                cache_dir=cached_dir, verbose=args.verbose,
                output_indices=output_indices,
            )
        if not args.skip_placed:
            print(f"\n[oracle] === Run 2: o_ffn PLACED-{args.mode} ===")
            placed_results = _run_o_ffn(
                f"o_ffn_placed_{args.mode}", placed_text, inputs_placed,
                cache_dir=placed_dir, verbose=args.verbose,
                output_indices=output_indices,
            )

        if cached_results is None or placed_results is None:
            print("[oracle] skipping diffs (a run was skipped)")
            return 0

        print(f"\n[oracle] === Diff: placed-{args.mode} vs cached "
              f"({args.mode} output / arg{out_arg}) ===")
        mm_stats = _stats(f"{args.mode} (placed vs cached)",
                          placed_results[out_arg], cached_results[out_arg])

        # Classification
        print("\n[oracle] === Classification ===")
        mm_corr = mm_stats["corr"]
        mm_max = mm_stats["max_abs"]
        if mm_max == 0:
            verdict = f"MATCH: placed {args.mode} == cached {args.mode}"
        elif mm_corr > 0.99:
            verdict = (f"CLOSE: placed {args.mode} correlates {mm_corr:.4f} "
                       "(likely numeric noise)")
        elif placed_results[out_arg].astype(np.float32).std() < 1e-6:
            verdict = (f"PLACED {args.mode} IS ZERO/CONSTANT: kernel may "
                       "not be writing")
        elif mm_corr > 0:
            verdict = (f"PARTIAL: corr={mm_corr:.4f}, max_abs={mm_max:.4f} "
                       "(possible operand/stride bug)")
        else:
            verdict = (f"DIVERGENT: corr={mm_corr:.4f} -- output "
                       "structurally wrong (operand swap, sign error, scale)")
        print(f"  {verdict}")
        return 0

    # ---------------------------------------------------------------
    # rms_gemms_rope mode (v_only / vkq / full)
    # ---------------------------------------------------------------
    print("[oracle] Generating synthetic inputs...")
    inputs_cached = _synth_inputs(seed=args.seed)
    inputs_placed = _synth_inputs(seed=args.seed)
    # Sanity: same seed → bit-identical inputs.
    assert np.array_equal(inputs_cached[0].view(np.int16), inputs_placed[0].view(np.int16))

    # Compute CPU V reference up front.
    print("[oracle] Computing CPU V reference...")
    v_cpu_ref = _cpu_v_reference(inputs_cached[0], inputs_cached[1], inputs_cached[7])

    cached_dir = workdir / "cached"
    placed_dir = workdir / f"placed_{args.mode}"
    cached_dir.mkdir(parents=True, exist_ok=True)
    placed_dir.mkdir(parents=True, exist_ok=True)

    cached_results = None
    placed_results = None

    if not args.skip_cached:
        print("\n[oracle] === Run 1: CACHED reference ===")
        cached_results = _run("rms_gemms_rope_cached", cached_text, inputs_cached,
                              cache_dir=cached_dir, verbose=args.verbose)
    if not args.skip_placed:
        print(f"\n[oracle] === Run 2: PLACED ({args.mode}) ===")
        placed_results = _run(f"rms_gemms_rope_placed_{args.mode}", placed_text, inputs_placed,
                              cache_dir=placed_dir, verbose=args.verbose)

    if cached_results is None or placed_results is None:
        print("[oracle] skipping diffs (a run was skipped)")
        return 0

    print("\n[oracle] === Diff: placed-V vs cached (V output / arg8) ===")
    v_stats = _stats("V (placed vs cached)",
                     placed_results["v"], cached_results["v"])

    print("\n[oracle] === Diff: cached vs CPU ref (V output) ===")
    _stats("V (cached vs CPU)", cached_results["v"], v_cpu_ref)

    print("\n[oracle] === Diff: placed-V vs CPU ref (V output) ===")
    _stats("V (placed vs CPU)", placed_results["v"], v_cpu_ref)

    print("\n[oracle] === Sanity diffs (everything else SHOULD match) ===")
    _stats("Q_pre  (placed vs cached)", placed_results["q_pre"], cached_results["q_pre"])
    _stats("K_pre  (placed vs cached)", placed_results["k_pre"], cached_results["k_pre"])
    _stats("Q_rop  (placed vs cached)", placed_results["q_rop"], cached_results["q_rop"])
    _stats("K_rop  (placed vs cached)", placed_results["k_rop"], cached_results["k_rop"])

    # Classify the bug
    print("\n[oracle] === Classification ===")
    v_corr = v_stats["corr"]
    v_max = v_stats["max_abs"]
    if v_max == 0:
        verdict = "MATCH: placed V == cached V"
    elif v_corr > 0.99:
        verdict = f"CLOSE: placed V correlates {v_corr:.4f} (likely numeric noise)"
    elif placed_results["v"].astype(np.float32).std() < 1e-6:
        verdict = "PLACED V IS ZERO/CONSTANT: kernel may not be writing"
    elif v_corr > 0:
        verdict = f"PARTIAL: corr={v_corr:.4f}, max_abs={v_max:.4f} (possible operand/stride bug)"
    else:
        verdict = (f"DIVERGENT: corr={v_corr:.4f} -- output structurally wrong "
                   "(operand swap, sign error, scale)")
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
