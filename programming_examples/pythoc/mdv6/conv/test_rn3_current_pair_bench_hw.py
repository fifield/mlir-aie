#!/usr/bin/env python3
"""Benchmark the current production mc_re6_rn3 two-conv path.

This is a timing/dispatch-count comparison scaffold for the rn3pair fusion work.
It runs the existing production 3x3 multicore layer twice on a 24x16 HWC input
(6 spatial tiles), matching the geometry used by test_rn3_pair_full_layer_hw.py.
It is not a fused-kernel correctness test; it uses random fused Conv+BN weights
and checks only that the two existing NPU dispatches complete and produce finite
output.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "../../../python"))

import run_tiled_mc as mcr  # noqa: E402


def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).cpu().numpy().reshape(-1).copy()


def make_fused_weights(rng: np.random.Generator, oc: int = 48, ic: int = 48) -> np.ndarray:
    w = rng.normal(0.0, 0.05, size=(oc, ic, 3, 3)).astype(np.float32)
    bn_w = rng.normal(1.0, 0.02, size=(oc,)).astype(np.float32)
    bn_b = rng.normal(0.0, 0.01, size=(oc,)).astype(np.float32)
    return f32_to_bf16_u16(np.concatenate([w.reshape(-1), bn_w, bn_b]))


class LaunchCounter:
    def __init__(self):
        self.n = 0
        self.ms = 0.0
        self.orig = None

    def __enter__(self):
        self.orig = mcr._xrt_run_kernel

        def wrapped(kernel, args):
            t0 = time.perf_counter()
            r = self.orig(kernel, args)
            dt = time.perf_counter() - t0
            self.n += 1
            self.ms += dt * 1000
            return r

        mcr._xrt_run_kernel = wrapped
        return self

    def __exit__(self, *exc):
        mcr._xrt_run_kernel = self.orig
        return False


def run_once(H: int, W: int, seed: int):
    rng = np.random.default_rng(seed)
    C = 48
    image = torch.from_numpy(rng.normal(0.0, 0.15, size=(H, W, C)).astype(np.float32)).to(torch.bfloat16)
    w1 = make_fused_weights(rng, 48, 48)
    w2 = make_fused_weights(rng, 48, 48)

    # Ensure the layer name resolves through the same merged-ELF path used by
    # full-model production. For re6_rn3, run_tiled_mc documents 25 full-model
    # tiles as one call per logical 3x3 conv.
    t0 = time.perf_counter()
    with LaunchCounter() as lc:
        y1 = mcr.run_tiled_fused_conv_mc(
            "mc_re6_rn3", "re6_rn_c3", image, w1,
            H, W, 48, 8, 8, 16, 1, 3, 1,
        )
        y2 = mcr.run_tiled_fused_conv_mc(
            "mc_re6_rn3", "re6_rn_c3", y1, w2,
            H, W, 48, 8, 8, 16, 1, 3, 1,
        )
    wall_ms = (time.perf_counter() - t0) * 1000
    finite = bool(torch.isfinite(y2.float()).all().item())
    return y2, finite, lc.n, lc.ms, wall_ms


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=24)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args(argv)

    n_tiles = (args.height // 8) * (args.width // 8)
    print(f"height={args.height} width={args.width} n_tiles={n_tiles} repeats={args.repeats}")
    npu_vals = []
    wall_vals = []
    for i in range(args.repeats):
        y2, finite, launches, npu_ms, wall_ms = run_once(args.height, args.width, args.seed + i)
        print(f"run={i} shape={tuple(y2.shape)} finite={finite} launches={launches} npu_ms={npu_ms:.2f} wall_ms={wall_ms:.2f}")
        if not finite:
            raise SystemExit("non-finite output")
        npu_vals.append(npu_ms)
        wall_vals.append(wall_ms)
    print(f"summary launches={launches} npu_ms_avg={np.mean(npu_vals):.2f} wall_ms_avg={np.mean(wall_vals):.2f}")


if __name__ == "__main__":
    main()
