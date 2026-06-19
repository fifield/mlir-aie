#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bench + correctness for the resident c2_attn runtime-L mask fix.

Measures, for the resident c2_attn device (ONE PDI, runtime-L mask):

  1. RESIDENT c2_attn latency (kernel-only NPU exec, start()->wait2(), median
     over N iters) at seq 64/128/256.  This is the number the runtime-mask fix
     targets (the old scalar mask dominated at ~15.6 ms/tok).

  2. KERNEL-ONLY NPU attention compute vs CPU attention wall-time at the same
     seqs -> the explicit more/less/same verdict (the clean number we never had:
     "ignoring dispatch, is NPU attention compute more/less/same vs CPU").
     NOTE: the resident device fuses attention WAVE-0 with the full O/add/FFN
     pipeline in one PDI, so the kernel_ms is attn+FFN.  We ALSO time the
     standalone batched attention device (attention ONLY) for the clean
     attn-vs-CPU comparison.

  3. MASK CORRECTNESS (independent of the known O-wave cross-token hazard that
     corrupts the FINAL FFN output res[14]): we read the WIDE attn scratch
     res[1] (the per-group context the attention wave writes BEFORE the O wave),
     repack rows 0..3 of each group head-major, and compare to CPU attention
     (tol 1e-3 / 5.2e-4).  Determinism: same input twice -> bit-exact res[1].

Run:
    source ~/npu-dev-pythoc/env.sh
    cd build_peano
    python ../tools/bench_c2_attn_mask.py --iters 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from test_decode_attn_npu import (  # noqa: E402
    N_HEADS, N_KV_HEADS, HEAD_DIM, EMB_DIM, GROUP_SIZE, TILE_ROWS,
    decode_attention_ref, decode_attention_npu_batched, _make_inputs,
)
from test_c2_attn import (  # noqa: E402
    _make_layer_weights, _o_gemv_ffn_args, _pack_attn_inputs_resident,
    _build_c2_merged, TILE_SIZE,
)

HIDDEN_DIM = 8192


def _resident_run(cache, q, k_cache, v_cache, current_pos, lw, *, kernel,
                  n_groups=N_KV_HEADS):
    """One resident c2_attn dispatch.  Returns (res1_scratch, out14)."""
    q_all, k_all, v_all = _pack_attn_inputs_resident(
        q, k_cache, v_cache, current_pos, n_groups)
    args = _o_gemv_ffn_args(np.zeros(EMB_DIM, dtype=bfloat16), lw)
    args[1] = np.zeros(n_groups * TILE_SIZE, dtype=bfloat16)
    args = args + [q_all, k_all, v_all]
    res = cache.load_and_run(
        kernel, None, *args,
        output_indices=[1, 14],
        static_input_indices={0, 7, 9, 12},
        intermediate_indices={1, 2, 4, 6, 8, 10, 11, 13, 14},
    )
    return (np.asarray(res[1], dtype=bfloat16),
            np.asarray(res[14], dtype=bfloat16))


def _attn_from_scratch(scratch, n_groups=N_KV_HEADS):
    """Repack the wide attn scratch (n_groups*4096, group g at g*4096 as a
    (64,64) untiled tile, rows 0..3 = the 4 real GQA heads) to head-major
    (n_heads, head_dim)."""
    s = scratch.reshape(n_groups, TILE_ROWS, HEAD_DIM)
    out = np.zeros((N_HEADS, HEAD_DIM), dtype=bfloat16)
    for g in range(n_groups):
        out[g * GROUP_SIZE:(g + 1) * GROUP_SIZE] = s[g, :GROUP_SIZE]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--positions", type=int, nargs="*",
                    default=[63, 127, 255])
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    from kernel_builder.cache import KernelCache, Profiler
    from builders.c2_attn import (build_c2_attn_resident_module,
                                  c2_attn_resident_kernel_id,
                                  c2_attn_kernel_id)

    prof = Profiler(enabled=True)
    cache = KernelCache(cache_dir="c2_attn_cache", verbose=True, profiler=prof)

    kernel = "c2_attn_resident"
    if kernel not in cache.artifacts or args.rebuild:
        ir = build_c2_attn_resident_module(N_KV_HEADS, verbose=True)
        cache.compile_and_cache(kernel, ir, c2_attn_resident_kernel_id())

    # ---- MASK CORRECTNESS + DETERMINISM on the attn scratch (res[1]) ----
    print("\n=== MASK CORRECTNESS (attn scratch res[1] vs CPU attention) ===")
    mask_ok = True
    for pos in args.positions:
        seq_len = pos + 1
        q, k_cache, v_cache = _make_inputs(seq_len, seed=pos)
        lw = _make_layer_weights(seed=pos)
        attn_cpu = decode_attention_ref(q, k_cache, v_cache, pos)  # (32,64) f32

        s1, _ = _resident_run(cache, q, k_cache, v_cache, pos, lw, kernel=kernel)
        s2, _ = _resident_run(cache, q, k_cache, v_cache, pos, lw, kernel=kernel)
        det = bool(np.array_equal(s1, s2))
        attn_npu = _attn_from_scratch(s1).astype(np.float32)
        e = float(np.max(np.abs(attn_npu - attn_cpu)))
        good = e < 1e-3 and det
        mask_ok = mask_ok and good
        print(f"  [pos={pos:3d} seq={seq_len:3d}] attn_err={e:.3e} "
              f"det={det} {'PASS' if good else 'FAIL'}")

    # ---- RESIDENT c2_attn latency (kernel-only, attn+FFN fused, 1 PDI) ----
    print("\n=== RESIDENT c2_attn KERNEL-ONLY latency (attn+FFN, 1 PDI) ===")
    res_lat = {}
    for pos in args.positions:
        seq_len = pos + 1
        q, k_cache, v_cache = _make_inputs(seq_len, seed=pos)
        lw = _make_layer_weights(seed=pos)

        def run():
            return _resident_run(cache, q, k_cache, v_cache, pos, lw,
                                 kernel=kernel)
        for _ in range(args.warmup):
            run()
        prof.kernel_breakdowns.clear()
        host_t = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            run()
            host_t.append((time.perf_counter() - t0) * 1e3)
        ks = [e["kernel_ms"] for e in prof.kernel_breakdowns[kernel]]
        kmed = float(np.median(ks))
        hmed = float(np.median(host_t))
        res_lat[seq_len] = (kmed, hmed)
        print(f"  [seq={seq_len:3d}] kernel-only={kmed:7.4f} ms  "
              f"host-envelope={hmed:7.4f} ms")

    # ---- Standalone batched ATTENTION-ONLY device: NPU vs CPU compute ----
    print("\n=== ATTENTION-ONLY: NPU kernel compute vs CPU wall (clean) ===")
    print(f"{'seq':>5} {'NPU_attn_ms':>12} {'CPU_attn_ms':>12} {'verdict':>10}")
    verdicts = {}
    for pos in args.positions:
        seq_len = pos + 1
        n_chunks = (seq_len + TILE_ROWS - 1) // TILE_ROWS
        if n_chunks > 4:
            continue
        akern = f"decode_attn_b{seq_len}"
        if akern not in cache.artifacts:
            from builders.attn_decode import build_decode_attn_batched_module
            ir = build_decode_attn_batched_module(seq_len, N_KV_HEADS)
            cache.compile_and_cache(akern, ir, "decode_attn")
        q, k_cache, v_cache = _make_inputs(seq_len, seed=pos)

        # NPU attention-only (kernel_ms via profiler).
        def arun():
            return decode_attention_npu_batched(cache, q, k_cache, v_cache,
                                                pos, kernel=akern)
        for _ in range(args.warmup):
            arun()
        prof.kernel_breakdowns.clear()
        for _ in range(args.iters):
            arun()
        aks = [e["kernel_ms"] for e in prof.kernel_breakdowns[akern]]
        npu_attn_ms = float(np.median(aks))

        # CPU attention wall-time.
        for _ in range(3):
            decode_attention_ref(q, k_cache, v_cache, pos)
        cpu_t = []
        for _ in range(max(args.iters, 50)):
            t0 = time.perf_counter()
            decode_attention_ref(q, k_cache, v_cache, pos)
            cpu_t.append((time.perf_counter() - t0) * 1e3)
        cpu_attn_ms = float(np.median(cpu_t))

        ratio = npu_attn_ms / cpu_attn_ms
        verdict = ("SAME" if 0.8 <= ratio <= 1.25
                   else ("FASTER" if ratio < 0.8 else "SLOWER"))
        verdicts[seq_len] = (npu_attn_ms, cpu_attn_ms, verdict)
        print(f"{seq_len:>5} {npu_attn_ms:>12.4f} {cpu_attn_ms:>12.4f} "
              f"{verdict:>10}")

    print("\n=== SUMMARY ===")
    print(f"mask correctness/determinism: {'PASS' if mask_ok else 'FAIL'}")
    for seq, (k, h) in sorted(res_lat.items()):
        print(f"  resident c2_attn seq={seq}: kernel={k:.4f} ms "
              f"host={h:.4f} ms")
    for seq, (n, c, v) in sorted(verdicts.items()):
        print(f"  attn-only seq={seq}: NPU={n:.4f} ms CPU={c:.4f} ms -> {v}")
    return 0 if mask_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
