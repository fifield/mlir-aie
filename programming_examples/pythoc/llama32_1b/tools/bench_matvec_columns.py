#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Columns -> weight-streaming-bandwidth curve for the decode GEMV matvec.

Single-token decode GEMV is M=1 (weights read once, no reuse) => it is
weight-DMA-bandwidth-bound.  This sweep answers: is effective weight-streaming
bandwidth LINEAR in the column count C (shim-channel-bound, more columns keep
helping) or does it PLATEAU (DRAM-bandwidth-bound)?

Method (separates launch floor from steady streaming rate):
  For each C in {1,2,4,8}, SWEEP the weight-stream size M (K fixed at 2048) and
  fit  time(bytes) = floor + bytes / BW(C).  The SLOPE gives steady-state
  bandwidth BW(C); the INTERCEPT gives the launch floor.

Each (C,M) uses one cached ELF + one XRT context with BO reuse, so we time
STEADY dispatch (not first-load).  At least one size per C is numerically
gated against the W @ x reference.

Usage:
    source ~/npu-dev-pythoc/env.sh
    python tools/bench_matvec_columns.py
    python tools/bench_matvec_columns.py --iters 50 --warmup 10
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

from builders.matvec_bw_probe import build_matvec_bw_module, K_FIXED, M_TILE
from kernel_builder.cache import KernelCache

# Default sweep: M sizes per column count.  M must be divisible by C*M_TILE.
# Weight bytes = M * K * 2.  At K=2048 these are 1/4/16/64 MB.
DEFAULT_MS = [256, 1024, 4096, 16384, 32768]
DEFAULT_COLS = [1, 2, 4, 8]
INSTANCE = "matvec_bw"


def _make_inputs(M: int, K: int, n_cols: int, *, seed: int):
    """Return (W, x, y0) and the reference proj (per-column-partitioned layout).

    The seg partitions W's rows across columns as contiguous slabs; the host W
    is plain (M,K) row-major, so the reference is simply proj = W @ x.
    """
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)).astype(np.float32) * 0.05).astype(bfloat16)
    x = (rng.standard_normal((K,)).astype(np.float32) * 0.5).astype(bfloat16)
    y0 = np.zeros((M,), dtype=bfloat16)
    ref = (W.astype(np.float32) @ x.astype(np.float32)).astype(bfloat16)
    return W, x, y0, ref


def _run_once(cache, name, W, x, y0):
    res = cache.load_and_run(
        name, None,
        W.reshape(-1), x.reshape(-1), y0.reshape(-1),
        output_indices=[2],
    )
    return np.asarray(res[2], dtype=bfloat16).reshape(-1)


def bench_config(cache, n_cols: int, M: int, K: int, *,
                 iters: int, warmup: int, gate: bool, verbose: bool):
    """Compile (if needed), optionally numeric-gate, time `iters` launches.

    Returns (median_ms, ok, max_abs_err) -- ok is None if gate skipped.
    """
    name = f"matvec_bw_c{n_cols}_m{M}_k{K}"
    if name not in cache.artifacts:
        ir = build_matvec_bw_module(M, K, n_cols, verbose=verbose)
        cache.compile_and_cache(name, ir, INSTANCE)

    W, x, y0, ref = _make_inputs(M, K, n_cols, seed=1234 + M + n_cols)

    ok, max_err = None, None
    # First call also loads the XRT context + allocates BOs.
    out = _run_once(cache, name, W, x, y0)
    if gate:
        of = out.astype(np.float32)
        rf = ref.astype(np.float32)
        max_err = float(np.max(np.abs(of - rf)))
        denom = float(np.max(np.abs(rf))) + 1e-6
        ok = max_err < 0.05 * denom + 0.02
        if verbose:
            print(f"      gate C={n_cols} M={M}: max_abs_err={max_err:.4g} "
                  f"(ref_max={denom:.4g}) -> {'PASS' if ok else 'FAIL'}")

    for _ in range(warmup):
        _run_once(cache, name, W, x, y0)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _run_once(cache, name, W, x, y0)
        times.append((time.perf_counter() - t0) * 1000.0)
    med = float(np.median(times))
    return med, ok, max_err


def fit_line(bytes_arr, ms_arr):
    """Least-squares fit ms = floor + bytes*slope_ms_per_byte.

    Returns (floor_ms, BW_GB_s).  BW = 1/slope, with slope in ms/byte =>
    bytes/ms = 1/slope; GB/s = (bytes/ms)*1e3/1e9 = 1e-6/slope... compute
    directly from bytes & seconds for clarity.
    """
    b = np.asarray(bytes_arr, dtype=np.float64)
    t_s = np.asarray(ms_arr, dtype=np.float64) / 1e3  # seconds
    A = np.vstack([np.ones_like(b), b]).T
    (floor_s, slope_s_per_byte), *_ = np.linalg.lstsq(A, t_s, rcond=None)
    bw_gb_s = (1.0 / slope_s_per_byte) / 1e9 if slope_s_per_byte > 0 else float("nan")
    return floor_s * 1e3, bw_gb_s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cols", type=int, nargs="+", default=DEFAULT_COLS)
    ap.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    K = K_FIXED
    cache_dir = args.cache_dir or str(_SCRIPT_DIR / "bench_matvec_cache")
    cache = KernelCache(cache_dir=cache_dir, verbose=args.verbose)

    # Per C, the valid M values (divisible by C*M_TILE).
    results = {}   # (C,M) -> (ms, bytes, ok, err)
    for C in args.cols:
        for M in args.ms:
            if M % (C * M_TILE) != 0:
                continue
            wbytes = M * K * 2
            print(f"[bench] C={C} M={M} weight={wbytes/1e6:.2f}MB ...",
                  flush=True)
            try:
                med, ok, err = bench_config(
                    cache, C, M, K,
                    iters=args.iters, warmup=args.warmup,
                    gate=(not args.no_gate), verbose=args.verbose)
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")
                results[(C, M)] = (None, wbytes, False, None)
                continue
            results[(C, M)] = (med, wbytes, ok, err)
            gate_str = "" if ok is None else (" PASS" if ok else " FAIL")
            print(f"    median={med:.4f} ms{gate_str}", flush=True)

    # Report.
    print("\n" + "=" * 78)
    print("RAW time(bytes) points")
    print("=" * 78)
    print(f"{'C':>3} {'M':>7} {'weightMB':>9} {'median_ms':>10} {'gate':>6}")
    for (C, M), (med, wb, ok, err) in sorted(results.items()):
        gate_str = "-" if ok is None else ("PASS" if ok else "FAIL")
        med_str = "ERR" if med is None else f"{med:.4f}"
        print(f"{C:>3} {M:>7} {wb/1e6:>9.2f} {med_str:>10} {gate_str:>6}")

    print("\n" + "=" * 78)
    print("FITTED  time = floor + bytes / BW(C)")
    print("=" * 78)
    print(f"{'C':>3} {'BW(C) GB/s':>12} {'floor ms':>10} {'per-chan GB/s':>14} "
          f"{'npts':>5}")
    fits = {}
    for C in sorted(set(c for c, _ in results)):
        pts = [(wb, med) for (cc, _M), (med, wb, _ok, _e) in results.items()
               if cc == C and med is not None]
        if len(pts) < 2:
            print(f"{C:>3} {'(need >=2 pts)':>12}")
            continue
        bs = [p[0] for p in pts]
        ts = [p[1] for p in pts]
        floor_ms, bw = fit_line(bs, ts)
        fits[C] = (bw, floor_ms)
        # 2 shim MM2S channels per column contribute weight bandwidth.
        per_chan = bw / (2 * C)
        print(f"{C:>3} {bw:>12.1f} {floor_ms:>10.4f} {per_chan:>14.2f} "
              f"{len(pts):>5}")

    if fits:
        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        cs = sorted(fits)
        for C in cs:
            print(f"  BW({C}) = {fits[C][0]:.1f} GB/s, "
                  f"GB/s-per-column = {fits[C][0]/C:.1f}")
        if len(cs) >= 2:
            c_lo, c_hi = cs[0], cs[-1]
            scale = fits[c_hi][0] / fits[c_lo][0]
            col_scale = c_hi / c_lo
            eff = scale / col_scale
            print(f"\n  BW({c_hi})/BW({c_lo}) = {scale:.2f}x for {col_scale:.0f}x "
                  f"columns -> scaling efficiency {eff*100:.0f}%")
            if eff > 0.7:
                print("  => LINEAR-in-C (shim-channel-bound); columns keep "
                      "helping, NOT DRAM-saturated.")
            else:
                print("  => SUB-LINEAR / PLATEAU (approaching DRAM-bandwidth "
                      "limit); columns beyond the knee give diminishing return.")
            if 8 in fits:
                print(f"\n  8-column model: BW(8)={fits[8][0]:.1f} GB/s over 16 "
                      f"shim weight channels = {fits[8][0]/16:.2f} GB/s/channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
