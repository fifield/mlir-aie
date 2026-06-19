#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Latency benchmark: NPU decode attention (<=256 path) vs CPU.

Times the validated NPU decode-attention path (builders/attn_decode.py,
8 dispatches/token -- one per GQA group) against the production
``decode_attention_cpu`` (numpy) it would replace, across seq_len in the
validated range (n_chunks<=4, i.e. seq_len<=256).

This measures the CURRENT unoptimized path: 8 separate dispatches per token.
The single-device batched variant (8 groups -> 1 dispatch) is a follow-on and
would change the NPU number substantially; this is the honest baseline.

Usage:
    source ~/npu-dev-pythoc/env.sh
    python tools/bench_decode_attn.py
    python tools/bench_decode_attn.py --seq-lens 64 256 --iters 100
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

from test_decode_attn_npu import (  # noqa: E402
    N_HEADS, HEAD_DIM, N_KV_HEADS,
    decode_attention_npu, decode_attention_ref, _make_inputs,
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
    return (statistics.median(samples), statistics.mean(samples),
            min(samples), max(samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[64, 128, 200, 256])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    from kernel_builder.cache import KernelCache
    from builders.attn_decode import build_decode_attn_module
    from llama32_1b_decode import decode_attention_cpu

    cache = KernelCache(cache_dir="decode_attn_test_cache", verbose=False)

    print(f"{'seq':>5} | {'CPU ms (med/mean)':>22} | "
          f"{'NPU ms (med/mean)':>22} | {'NPU/CPU':>8} | {'ms/dispatch':>11}")
    print("-" * 86)

    for seq_len in args.seq_lens:
        kernel = f"decode_attn_s{seq_len}"
        if kernel not in cache.artifacts:
            try:
                ir = build_decode_attn_module(seq_len, verbose=False)
            except NotImplementedError as e:
                print(f"{seq_len:>5} | SKIP -- {e}")
                continue
            cache.compile_and_cache(kernel, ir, "decode_attn")

        q, k_cache, v_cache = _make_inputs(seq_len, seed=seq_len)
        pos = seq_len - 1

        # Correctness gate before timing (don't benchmark a wrong kernel).
        ref = decode_attention_ref(q, k_cache, v_cache, pos)
        npu0 = np.asarray(
            decode_attention_npu(cache, q, k_cache, v_cache, pos, kernel=kernel),
            dtype=bfloat16).reshape(N_HEADS, HEAD_DIM).astype(np.float32)
        err = float(np.max(np.abs(npu0 - ref)))
        if err >= 2e-2:
            print(f"{seq_len:>5} | NUMERICS FAIL (err {err:.2e}) -- skipping timing")
            continue

        cpu_s = _time(
            lambda: decode_attention_cpu(
                q.reshape(-1), k_cache, v_cache, pos, N_HEADS, N_KV_HEADS, HEAD_DIM),
            args.iters, args.warmup)
        npu_s = _time(
            lambda: decode_attention_npu(cache, q, k_cache, v_cache, pos,
                                         kernel=kernel),
            args.iters, args.warmup)

        c_med, c_mean, _, _ = _stats(cpu_s)
        n_med, n_mean, _, _ = _stats(npu_s)
        print(f"{seq_len:>5} | {c_med:9.3f} / {c_mean:9.3f}    | "
              f"{n_med:9.3f} / {n_mean:9.3f}    | {n_med / c_med:7.2f}x | "
              f"{n_med / N_KV_HEADS:10.3f}")

    print("\nNotes:")
    print(f"  - NPU = {N_KV_HEADS} dispatches/token (one per GQA group); "
          "ms/dispatch = NPU median / 8.")
    print("  - Batched single-device variant (8->1 dispatch) not yet built.")
    print("  - err gate: 2e-2 (bf16); numerics already validated separately.")


if __name__ == "__main__":
    main()
