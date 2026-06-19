#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Columns -> weight-streaming-bandwidth, with prefetch-overlap variants.

ADDITIVE companion to tools/bench_matvec_columns.py.  Same fit methodology
(time(bytes) = floor + bytes / BW(C)), but sweeps several ingest MODES from
builders.matvec_bw_overlap to attribute the ~15 GB/s decode-GEMV ceiling:

  baseline     -- single-buffered L1 weight slot (reproduces the probe)
  nop          -- compute-free: DMA flows, MAC skipped => PURE ingest BW
  pingpong     -- 2 L1 weight slots, weight DMA fills N+1 while core MACs N
  pingpong_l2  -- pingpong + 2-slot memtile staging ring

Usage:
    source ~/npu-dev-pythoc/env.sh
    python tools/bench_matvec_overlap.py --modes baseline nop pingpong \
        --cols 1 4 8 --ms 1024 4096 16384
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLE_DIR))

from builders.matvec_bw_overlap import build_matvec_bw_overlap_module, K_FIXED, M_TILE
from kernel_builder.cache import KernelCache

DEFAULT_MS = [1024, 4096, 16384]
DEFAULT_COLS = [1, 4, 8]
DEFAULT_MODES = ["baseline", "nop", "pingpong"]
INSTANCE = "matvec_bw"


def _make_inputs(M, K, n_cols, *, seed):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)).astype(np.float32) * 0.05).astype(bfloat16)
    x = (rng.standard_normal((K,)).astype(np.float32) * 0.5).astype(bfloat16)
    y0 = np.zeros((M,), dtype=bfloat16)
    ref = (W.astype(np.float32) @ x.astype(np.float32)).astype(bfloat16)
    return W, x, y0, ref


def _run_once(cache, name, W, x, y0):
    res = cache.load_and_run(
        name, None, W.reshape(-1), x.reshape(-1), y0.reshape(-1),
        output_indices=[2])
    return np.asarray(res[2], dtype=bfloat16).reshape(-1)


def bench_config(cache, mode, n_cols, M, K, *, iters, warmup, gate, verbose):
    name = f"mvbwo_{mode}_c{n_cols}_m{M}_k{K}"
    if name not in cache.artifacts:
        ir = build_matvec_bw_overlap_module(M, K, n_cols, mode=mode, verbose=verbose)
        cache.compile_and_cache(name, ir, INSTANCE)

    W, x, y0, ref = _make_inputs(M, K, n_cols, seed=1234 + M + n_cols)

    ok, max_err = None, None
    out = _run_once(cache, name, W, x, y0)
    if gate and mode != "nop":  # nop produces zeros by design
        of = out.astype(np.float32)
        rf = ref.astype(np.float32)
        max_err = float(np.max(np.abs(of - rf)))
        denom = float(np.max(np.abs(rf))) + 1e-6
        ok = max_err < 0.05 * denom + 0.02
        if verbose:
            print(f"      gate {mode} C={n_cols} M={M}: max_abs_err={max_err:.4g} "
                  f"(ref_max={denom:.4g}) -> {'PASS' if ok else 'FAIL'}")

    for _ in range(warmup):
        _run_once(cache, name, W, x, y0)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _run_once(cache, name, W, x, y0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), ok, max_err


def fit_line(bytes_arr, ms_arr):
    b = np.asarray(bytes_arr, dtype=np.float64)
    t_s = np.asarray(ms_arr, dtype=np.float64) / 1e3
    A = np.vstack([np.ones_like(b), b]).T
    (floor_s, slope_s_per_byte), *_ = np.linalg.lstsq(A, t_s, rcond=None)
    bw = (1.0 / slope_s_per_byte) / 1e9 if slope_s_per_byte > 0 else float("nan")
    return floor_s * 1e3, bw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--cols", type=int, nargs="+", default=DEFAULT_COLS)
    ap.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    K = K_FIXED
    cache_dir = args.cache_dir or str(_SCRIPT_DIR / "bench_overlap_cache")
    cache = KernelCache(cache_dir=cache_dir, verbose=args.verbose)

    # results[(mode,C,M)] = (ms, bytes, ok, err)
    results = {}
    for mode in args.modes:
        for C in args.cols:
            for M in args.ms:
                if M % (C * M_TILE) != 0:
                    continue
                # pingpong needs even blocks_per_outer
                if mode in ("pingpong", "pingpong_l2"):
                    bpc = (M // C) // M_TILE
                    if bpc % 2 != 0:
                        print(f"[skip] {mode} C={C} M={M}: odd blocks_per_outer")
                        continue
                wbytes = M * K * 2
                print(f"[bench] {mode} C={C} M={M} weight={wbytes/1e6:.2f}MB ...",
                      flush=True)
                try:
                    med, ok, err = bench_config(
                        cache, mode, C, M, K,
                        iters=args.iters, warmup=args.warmup,
                        gate=(not args.no_gate), verbose=args.verbose)
                except Exception as e:
                    print(f"    ERROR: {type(e).__name__}: {e}")
                    results[(mode, C, M)] = (None, wbytes, False, None)
                    continue
                results[(mode, C, M)] = (med, wbytes, ok, err)
                gate_str = "" if ok is None else (" PASS" if ok else " FAIL")
                print(f"    median={med:.4f} ms{gate_str}", flush=True)

    print("\n" + "=" * 86)
    print("RAW time(bytes) points")
    print("=" * 86)
    print(f"{'mode':>12} {'C':>3} {'M':>7} {'weightMB':>9} {'median_ms':>10} {'gate':>6}")
    for (mode, C, M), (med, wb, ok, err) in sorted(results.items()):
        gate_str = "-" if ok is None else ("PASS" if ok else "FAIL")
        med_str = "ERR" if med is None else f"{med:.4f}"
        print(f"{mode:>12} {C:>3} {M:>7} {wb/1e6:>9.2f} {med_str:>10} {gate_str:>6}")

    print("\n" + "=" * 86)
    print("FITTED  time = floor + bytes / BW(C)   [GB/s]")
    print("=" * 86)
    fits = {}  # (mode,C) -> (bw, floor)
    modes = [m for m in args.modes]
    cols = sorted(set(c for _m, c, _M in results))
    header = f"{'C':>3}" + "".join(f"{m:>16}" for m in modes)
    print(header)
    for C in cols:
        row = f"{C:>3}"
        for mode in modes:
            pts = [(wb, med) for (mm, cc, _M), (med, wb, _o, _e) in results.items()
                   if mm == mode and cc == C and med is not None]
            if len(pts) < 2:
                row += f"{'(n/a)':>16}"
                continue
            floor_ms, bw = fit_line([p[0] for p in pts], [p[1] for p in pts])
            fits[(mode, C)] = (bw, floor_ms)
            row += f"{bw:>16.1f}"
        print(row)

    print("\n" + "=" * 86)
    print("VERDICT  (BW in GB/s; LPDDR5 peak ~200 GB/s)")
    print("=" * 86)
    for C in cols:
        base = fits.get(("baseline", C), (None,))[0]
        for mode in modes:
            bw = fits.get((mode, C), (None,))[0]
            if bw is None:
                continue
            rel = "" if (base is None or mode == "baseline") else \
                f"  ({bw/base:.2f}x vs baseline)"
            print(f"  C={C:>2} {mode:>12}: BW={bw:6.1f} GB/s{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
