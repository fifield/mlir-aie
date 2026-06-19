#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ITERATION 2 driver: single GQA group, seq_len=64, decode attention on HW.

Compares NPU rows 0-3 (the 4 real q-heads) against the float32 reference for
ONE group, isolating the single-tile dataflow before the full 8-group sweep.
"""
import os
import sys
import numpy as np
from ml_dtypes import bfloat16

_ROOT = "/home/jfifield/npu-dev-pythoc/mlir-aie/programming_examples/pythoc/llama32_1b"
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from kernel_builder.cache import KernelCache
from builders.attn_decode import build_decode_attn_module, TILE_ROWS, HEAD_DIM, GROUP_SIZE

SEQ = 64


def main():
    cache = KernelCache(cache_dir="decode_attn_probe_cache", verbose=True)
    cache.compile_and_cache(
        "decode_attn", build_decode_attn_module(SEQ, verbose=True), "decode_attn"
    )

    rng = np.random.default_rng(1)
    # One group: 4 q heads, 64 KV positions, head_dim 64.
    q4 = (rng.standard_normal((GROUP_SIZE, HEAD_DIM)).astype(np.float32) * 0.1)
    k = (rng.standard_normal((SEQ, HEAD_DIM)).astype(np.float32) * 0.1)
    v = (rng.standard_normal((SEQ, HEAD_DIM)).astype(np.float32) * 0.1)

    # float32 reference for these 4 heads (scale folded as 1/sqrt(head_dim)).
    scale = 1.0 / np.sqrt(HEAD_DIM)
    ref = np.zeros((GROUP_SIZE, HEAD_DIM), np.float32)
    for h in range(GROUP_SIZE):
        s = (q4[h] @ k.T) * scale
        p = np.exp(s - s.max()); p /= p.sum()
        ref[h] = p @ v

    # Pack q into (64,64): 4 real rows + zero pad.
    q_pad = np.zeros((TILE_ROWS, HEAD_DIM), np.float32)
    q_pad[:GROUP_SIZE] = q4
    q_g = np.ascontiguousarray(q_pad.astype(bfloat16)).reshape(-1)
    k_g = np.ascontiguousarray(k.astype(bfloat16)).reshape(-1)
    v_g = np.ascontiguousarray(v.astype(bfloat16)).reshape(-1)
    out_g = np.zeros(TILE_ROWS * HEAD_DIM, dtype=bfloat16)

    results = cache.load_and_run(
        "decode_attn", None, q_g, k_g, v_g, out_g, output_indices=[3]
    )
    out = np.asarray(results[3], dtype=bfloat16).reshape(TILE_ROWS, HEAD_DIM)
    npu = out[:GROUP_SIZE].astype(np.float32)

    err = np.max(np.abs(npu - ref))
    print(f"[stage0] max-abs err (rows 0-3) = {err:.3e}  tol=2e-2  "
          f"{'PASS' if err < 2e-2 else 'FAIL'}")
    print("  ref[0,:6] =", ref[0, :6])
    print("  npu[0,:6] =", npu[0, :6])
    print("  per-head max-err:", [float(np.max(np.abs(npu[h]-ref[h]))) for h in range(GROUP_SIZE)])
    sys.exit(0 if err < 2e-2 else 1)


if __name__ == "__main__":
    main()
