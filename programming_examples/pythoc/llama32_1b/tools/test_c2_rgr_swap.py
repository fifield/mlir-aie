#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Minimal repro: o_gemv_ffn (c2 pack) dispatch followed by rms_gemv_rope.

Mimics the decode preload order that wedges inference (ogf layer0 -> rgr
layer1) without weights/prefill. Uses the production decode_kernel_cache.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

EMB, HID, KV = 2048, 8192, 512

os.chdir(PROJECT_DIR / "build_peano")
from kernel_builder.cache import KernelCache  # noqa: E402

cache = KernelCache(cache_dir=PROJECT_DIR / "build_peano" / "decode_kernel_cache",
                    verbose=False)
cache.load_manifest()

_rng = np.random.default_rng(3)
if os.environ.get("SWAP_RAND"):
    z = lambda n: (_rng.standard_normal(n) * 0.02).astype(bfloat16)
else:
    z = lambda n: np.zeros(n, dtype=bfloat16)
ogf_args = [z((EMB, EMB)), z(EMB), z(EMB), z(EMB), z(EMB), z(EMB), z(EMB),
            z((HID, EMB)), z(HID), z((HID, EMB)), z(HID), z(HID),
            z((EMB, HID)), z(EMB), z(EMB)]
rgr_args = [z(EMB), z(EMB), z(EMB), z((EMB, EMB)), z(EMB), z((KV, EMB)), z(KV),
            z((KV, EMB)), z(KV), z(64), z(64), z(EMB), z(KV)]

n = int(os.environ.get("SWAP_ITERS", "3"))
for i in range(n):
    t0 = time.perf_counter()
    cache.load_and_run("o_gemv_ffn", None, *ogf_args, output_indices=[14])
    t1 = time.perf_counter()
    cache.load_and_run("rms_gemv_rope", None, *rgr_args, output_indices=[11, 12])
    print(f"iter {i}: ogf {t1-t0:.2f}s rgr {time.perf_counter()-t1:.2f}s")
print("PASS")
