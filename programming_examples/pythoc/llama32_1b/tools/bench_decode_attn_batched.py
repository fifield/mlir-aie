#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Latency benchmark: BATCHED single-dispatch NPU decode attention vs CPU.

Times the batched NPU decode-attention path (builders/attn_decode.py
``build_decode_attn_batched_module`` -- ALL 8 GQA groups in ONE dispatch via
8 cores across 8 columns) against the production ``decode_attention_cpu``
(numpy) it would replace, for seq_len in {64,128,200,256} (n_chunks<=4).

This is the payoff measurement for the dispatch-collapse experiment: the
prior path did 8 dispatches/token (~0.25-0.33 ms each => ~2 ms); this does 1.

Usage:
    source ~/npu-dev-pythoc/env.sh
    python tools/bench_decode_attn_batched.py
    python tools/bench_decode_attn_batched.py --seq-lens 64 256 --iters 100
    python tools/bench_decode_attn_batched.py --n-groups 2 4 8   # rung sweep
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from test_decode_attn_npu import (  # noqa: E402
    N_HEADS, HEAD_DIM, N_KV_HEADS,
    decode_attention_npu_batched, decode_attention_ref, _make_inputs,
)


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return samples


def _stats(samples):
    return (statistics.median(samples), statistics.mean(samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[64, 128, 200, 256])
    ap.add_argument("--n-groups", type=int, nargs="+", default=[8],
                    help="GQA-group rungs to bench (2/4/8); 8 = full token")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    from kernel_builder.cache import KernelCache
    from builders.attn_decode import build_decode_attn_batched_module
    from llama32_1b_decode import decode_attention_cpu

    cache = KernelCache(cache_dir="decode_attn_batched_cache", verbose=False)

    for ng in args.n_groups:
        print(f"\n=== n_groups={ng} ({ng} cores, 1 dispatch) ===")
        print(f"{'seq':>5} | {'CPU ms (med/mean)':>22} | "
              f"{'NPU-batched ms (med/mean)':>26} | {'NPU/CPU':>8}")
        print("-" * 76)
        for seq_len in args.seq_lens:
            kernel = f"decode_attn_b{ng}_s{seq_len}"
            if kernel not in cache.artifacts:
                try:
                    ir = build_decode_attn_batched_module(seq_len, ng)
                except NotImplementedError as e:
                    print(f"{seq_len:>5} | SKIP -- {e}")
                    continue
                cache.compile_and_cache(kernel, ir, "decode_attn")

            q, k_cache, v_cache = _make_inputs(seq_len, seed=seq_len)
            pos = seq_len - 1

            # Correctness gate before timing.
            ref = decode_attention_ref(q, k_cache, v_cache, pos)
            npu0 = np.asarray(
                decode_attention_npu_batched(cache, q, k_cache, v_cache, pos,
                                             kernel=kernel, n_groups=ng),
                dtype=bfloat16).reshape(N_HEADS, HEAD_DIM).astype(np.float32)
            nh = ng * (N_HEADS // N_KV_HEADS)
            err = float(np.max(np.abs(npu0[:nh] - ref[:nh])))
            if err >= 2e-2:
                print(f"{seq_len:>5} | NUMERICS FAIL (err {err:.2e}) -- skip")
                continue

            # CPU reference times the FULL token (all 8 groups) regardless of
            # the NPU rung, since that is the thing the full path replaces.
            cpu_s = _time(
                lambda: decode_attention_cpu(
                    q.reshape(-1), k_cache, v_cache, pos,
                    N_HEADS, N_KV_HEADS, HEAD_DIM),
                args.iters, args.warmup)
            npu_s = _time(
                lambda: decode_attention_npu_batched(
                    cache, q, k_cache, v_cache, pos,
                    kernel=kernel, n_groups=ng),
                args.iters, args.warmup)

            c_med, c_mean = _stats(cpu_s)
            n_med, n_mean = _stats(npu_s)
            print(f"{seq_len:>5} | {c_med:9.3f} / {c_mean:9.3f}    | "
                  f"{n_med:11.3f} / {n_mean:11.3f}    | {n_med / c_med:7.2f}x")

    print("\nNotes:")
    print("  - NPU-batched = 1 dispatch/token (all groups, 1 core/column).")
    print("  - CPU = full 32-head decode_attention_cpu (the path replaced).")
    print("  - err gate: 2e-2 (bf16).")


if __name__ == "__main__":
    main()
