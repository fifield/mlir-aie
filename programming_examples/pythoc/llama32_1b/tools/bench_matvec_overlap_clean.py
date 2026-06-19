#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Low-variance kernel-only BW comparison for the ingest modes (ADDITIVE).

The host-wrapped timing in bench_matvec_overlap.py is noisy (it includes BO
write+read and Python).  This isolates the *NPU-only* execution time
(run.start() -> wait2()) via the cache's Profiler breakdown, sweeps a few
LARGE weight sizes (so steady streaming dominates the launch floor), and runs
all modes back-to-back in ONE process so cross-run drift cancels.

Reports, per (mode, C): the kernel-only BW(C) fit and the per-size kernel ms.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLE_DIR))

from builders.matvec_bw_overlap import build_matvec_bw_overlap_module, K_FIXED, M_TILE
from kernel_builder.cache import KernelCache, Profiler


def _inputs(M, K, seed):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)).astype(np.float32) * 0.05).astype(bfloat16)
    x = (rng.standard_normal((K,)).astype(np.float32) * 0.5).astype(bfloat16)
    y0 = np.zeros((M,), dtype=bfloat16)
    ref = (W.astype(np.float32) @ x.astype(np.float32)).astype(bfloat16)
    return W, x, y0, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "nop", "pingpong"])
    ap.add_argument("--cols", type=int, nargs="+", default=[8])
    ap.add_argument("--ms", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    K = K_FIXED
    prof = Profiler(enabled=True)
    cache_dir = args.cache_dir or str(_SCRIPT_DIR / "bench_overlap_cache")
    cache = KernelCache(cache_dir=cache_dir, profiler=prof)

    # results[(mode,C,M)] = (kernel_ms_median, wbytes)
    results = {}
    for mode in args.modes:
        for C in args.cols:
            for M in args.ms:
                if M % (C * M_TILE) != 0:
                    continue
                if mode in ("pingpong", "pingpong_l2"):
                    if ((M // C) // 4) % 2 != 0:
                        continue
                name = f"mvbwo_{mode}_c{C}_m{M}_k{K}"
                if name not in cache.artifacts:
                    ir = build_matvec_bw_overlap_module(M, K, C, mode=mode)
                    cache.compile_and_cache(name, ir, "matvec_bw")
                W, x, y0, ref = _inputs(M, K, 7 + M + C)

                def run():
                    r = cache.load_and_run(
                        name, None, W.reshape(-1), x.reshape(-1), y0.reshape(-1),
                        output_indices=[2])
                    return np.asarray(r[2], dtype=bfloat16).reshape(-1)

                out = run()
                if mode != "nop":
                    err = float(np.max(np.abs(out.astype(np.float32)
                                              - ref.astype(np.float32))))
                    denom = float(np.max(np.abs(ref.astype(np.float32)))) + 1e-6
                    gate = "PASS" if err < 0.05 * denom + 0.02 else "FAIL"
                else:
                    err, gate = 0.0, "-"

                prof.kernel_breakdowns.clear()
                for _ in range(args.warmup):
                    run()
                prof.kernel_breakdowns.clear()
                for _ in range(args.iters):
                    run()
                ks = [e["kernel_ms"] for e in prof.kernel_breakdowns[name]]
                kmed = float(np.median(ks))
                results[(mode, C, M)] = (kmed, M * K * 2)
                print(f"  {mode:>12} C={C} M={M:>6}  kernel={kmed:7.4f} ms  "
                      f"gate={gate}", flush=True)

    print("\n" + "=" * 70)
    print("KERNEL-ONLY  time = floor + bytes / BW(C)   [GB/s]")
    print("=" * 70)
    for C in sorted(set(c for _m, c, _M in results)):
        print(f"-- C={C} --")
        for mode in args.modes:
            pts = [(wb, km) for (mm, cc, _M), (km, wb) in results.items()
                   if mm == mode and cc == C]
            if len(pts) < 2:
                continue
            b = np.array([p[0] for p in pts], float)
            t = np.array([p[1] for p in pts], float) / 1e3
            A = np.vstack([np.ones_like(b), b]).T
            (floor, slope), *_ = np.linalg.lstsq(A, t, rcond=None)
            bw = (1.0 / slope) / 1e9 if slope > 0 else float("nan")
            print(f"   {mode:>12}: BW={bw:6.1f} GB/s  floor={floor*1e3:6.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
