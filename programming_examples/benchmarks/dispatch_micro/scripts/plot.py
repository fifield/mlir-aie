#!/usr/bin/env python3
# plot.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Generate PNG charts from dispatch_micro results.jsonl."""
import argparse
import collections
import json
import os
import statistics
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("matplotlib not installed; install with `pip install matplotlib`\n")
    sys.exit(1)


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _us(ns):
    return ns / 1000.0


def plot_vs_tiles(rows, metric, outdir, bds_filter=None):
    """Plot per-mechanism median latency vs tiles. Excludes batched runs
    (they live in plot_batched). Optionally pin bds to one value."""
    series = collections.defaultdict(dict)
    for r in rows:
        if r.get("metric") != metric or "ns" not in r:
            continue
        if r.get("batched", False):
            continue
        if bds_filter is not None and r["bds"] != bds_filter:
            continue
        series[r["mechanism"]][r["tiles"]] = _us(r["ns"]["p50"])
    if not series:
        return
    fig, ax = plt.subplots()
    for mech, pts in sorted(series.items()):
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=mech)
    ax.set_xlabel("tiles configured")
    ax.set_ylabel("median latency (µs)")
    suffix = f" (bds={bds_filter})" if bds_filter else ""
    ax.set_title(f"{metric}: latency vs tiles{suffix}")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fname = f"{metric}_vs_tiles" + (f"_bds{bds_filter}" if bds_filter else "")
    fig.savefig(os.path.join(outdir, fname + ".png"), dpi=150)
    plt.close(fig)


def plot_vs_bds(rows, metric, outdir, tiles_filter=None):
    series = collections.defaultdict(dict)
    for r in rows:
        if r.get("metric") != metric or "ns" not in r:
            continue
        if r.get("batched", False):
            continue
        if tiles_filter is not None and r["tiles"] != tiles_filter:
            continue
        series[r["mechanism"]][r["bds"]] = _us(r["ns"]["p50"])
    if not series:
        return
    fig, ax = plt.subplots()
    for mech, pts in sorted(series.items()):
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=mech)
    ax.set_xlabel("BDs per tile")
    ax.set_ylabel("median latency (µs)")
    suffix = f" (tiles={tiles_filter})" if tiles_filter else ""
    ax.set_title(f"{metric}: latency vs BDs/tile{suffix}")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fname = f"{metric}_vs_bds" + (f"_t{tiles_filter}" if tiles_filter else "")
    fig.savefig(os.path.join(outdir, fname + ".png"), dpi=150)
    plt.close(fig)


def plot_batched(rows, outdir):
    """Per-dispatch latency (p50 ns / batch_size) vs batch size, per mech, per tile-count."""
    series = collections.defaultdict(dict)
    for r in rows:
        if r.get("metric") != "pure_dispatch" or "ns" not in r:
            continue
        bs = r.get("batch_size", 1)
        key = (r["mechanism"], r["tiles"])
        series[key][bs] = _us(r["ns"]["p50"]) / bs
    if not series:
        return
    fig, ax = plt.subplots()
    for (mech, t), pts in sorted(series.items()):
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=f"{mech} (t={t})")
    ax.set_xlabel("batch size (runlist runs per execute)")
    ax.set_ylabel("per-dispatch median latency (µs)")
    ax.set_title("pure_dispatch: per-dispatch latency vs batch size")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle=":", which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pure_dispatch_batched.png"), dpi=150)
    plt.close(fig)


def plot_vs_total_tiles(rows, outdir, bds_filter=2):
    """For the whole-array sweep: latency vs total compute tiles (cols × rows)."""
    series = collections.defaultdict(dict)
    for r in rows:
        if r.get("metric") != "pure_dispatch" or "ns" not in r:
            continue
        if r.get("batched", False):
            continue
        if "rows_per_col" not in r:
            continue
        if r["bds"] != bds_filter:
            continue
        total = r["tiles"] * r["rows_per_col"]
        # Multiple shapes may share the same total; keep the median.
        series[r["mechanism"]].setdefault(total, []).append(_us(r["ns"]["p50"]))
    if not series:
        return
    fig, ax = plt.subplots()
    for mech, pts in sorted(series.items()):
        xs = sorted(pts)
        ys = [sum(pts[x])/len(pts[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=mech)
    ax.set_xlabel("total compute tiles (cols × rows_per_col)")
    ax.set_ylabel("median latency (µs)")
    ax.set_title(f"pure_dispatch: latency vs total compute tiles (bds={bds_filter})")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, linestyle=":", which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pure_dispatch_vs_total_tiles.png"), dpi=150)
    plt.close(fig)


def plot_cold_breakdown(rows, outdir):
    # Aggregate cold_start phases per mechanism.
    phases = ["load_ns", "register_ns", "kernel_ns", "first_dispatch_ns"]
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("metric") != "cold_start" or "cold_phases" not in r:
            continue
        cp = r["cold_phases"]
        for p in phases:
            agg[r["mechanism"]][p].append(cp.get(p, 0))
    if not agg:
        return
    mechs = sorted(agg)
    width = 0.6
    bottoms = [0] * len(mechs)
    fig, ax = plt.subplots()
    for p in phases:
        vals = [_us(statistics.median(agg[m][p])) if agg[m][p] else 0 for m in mechs]
        ax.bar(mechs, vals, width, bottom=bottoms, label=p.replace("_ns", ""))
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("median latency (µs)")
    ax.set_title("cold_start: phase breakdown")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cold_start_breakdown.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="results.jsonl")
    ap.add_argument("--outdir", default="results/plots")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.input)
    if not rows:
        sys.stderr.write("no rows in " + args.input + "\n")
        sys.exit(1)
    for metric in ("pure_dispatch", "warm_reconfig"):
        plot_vs_tiles(rows, metric, args.outdir)
        plot_vs_bds(rows, metric, args.outdir)
        for bds in (2, 4, 8):
            plot_vs_tiles(rows, metric, args.outdir, bds_filter=bds)
        for t in (1, 4, 8):
            plot_vs_bds(rows, metric, args.outdir, tiles_filter=t)
    plot_batched(rows, args.outdir)
    plot_vs_total_tiles(rows, args.outdir, bds_filter=2)
    plot_cold_breakdown(rows, args.outdir)
    print(f"wrote plots to {args.outdir}/")


if __name__ == "__main__":
    main()
