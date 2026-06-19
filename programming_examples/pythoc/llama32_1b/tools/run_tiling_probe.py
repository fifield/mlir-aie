#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ITERATION 1 driver: compile + run the decode-attn tiling probe on HW.

Verifies host_out == host_in for a 64x64 bf16 tile round-tripped through the
column-block-major 8x8-tiled L1 (in-tile DMA) and back (un-tile DMA).
"""
import os
import sys
import numpy as np
from ml_dtypes import bfloat16

_ROOT = "/home/jfifield/npu-dev-pythoc/mlir-aie/programming_examples/pythoc/llama32_1b"
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from kernel_builder.cache import KernelCache
from builders.attn_decode import build_tiling_probe_module, TILE_SIZE


def main():
    cache = KernelCache(cache_dir="decode_attn_probe_cache", verbose=True)
    cache.compile_and_cache(
        "decode_attn", build_tiling_probe_module(verbose=True), "decode_attn"
    )

    rng = np.random.default_rng(0)
    host_in = (rng.standard_normal(TILE_SIZE).astype(np.float32) * 0.5).astype(bfloat16)
    host_out = np.zeros(TILE_SIZE, dtype=bfloat16)

    results = cache.load_and_run(
        "decode_attn", None, host_in, host_out, output_indices=[1]
    )
    out = np.asarray(results[1], dtype=bfloat16).reshape(-1)

    eq = np.array_equal(out.view(np.int16), host_in.view(np.int16))
    diff = np.max(np.abs(out.astype(np.float32) - host_in.astype(np.float32)))
    print(f"[tiling-probe] exact-eq={eq}  max-abs-diff={diff:.3e}")
    if not eq:
        # Diagnostic: how does out relate to a numpy tiling of in?
        nat = host_in.astype(np.float32).reshape(64, 64)
        print("  in[0,:8] =", nat[0, :8])
        print("  out[:8]  =", out.astype(np.float32)[:8])
        print("  out[512:520] =", out.astype(np.float32)[512:520])
    sys.exit(0 if eq else 1)


if __name__ == "__main__":
    main()
