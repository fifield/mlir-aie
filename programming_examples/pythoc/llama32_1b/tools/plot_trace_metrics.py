#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Plot per-kernel AIE trace cycle-state metrics as grouped bar charts.

Consumes the ``summary_metrics.csv`` files emitted by ``trace_summary.py``
(one per trace-sweep directory) and renders a PNG comparing the headline
cycle-state metrics across the traced sub-devices, one panel per sweep
(e.g. BF16 vs AWQ).

The story these charts tell: the decode matvecs are DMA/lock-bound
(vec_util a few %, lock_stall ~75%, high DMA starvation), so per-kernel
*compute* optimizations are largely hidden behind the weight-stream wait.

Usage:
    python3 tools/plot_trace_metrics.py \
        --csv BF16=/tmp/sweep_bf16/summary_metrics.csv \
        --csv AWQ=/tmp/sweep_awq/summary_metrics.csv \
        -o /tmp/trace_metrics.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (csv column, legend label, color) -- the headline cycle-state metrics.
METRICS = [
    ("vec_util_pct",     "vec_util",      "#2ca02c"),
    ("lock_stall_pct",   "lock_stall",    "#d62728"),
    ("starv0_pct",       "DMA starv0 (X/out)", "#ff7f0e"),
    ("starv1_pct",       "DMA starv1 (W)", "#9467bd"),
    ("dma_in1_eff_pct",  "dma_in1_eff (W)", "#1f77b4"),
]


def _load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # short label = sub_device with trailing _0 / _bf16_0 trimmed
    for r in rows:
        sd = r.get("sub_device", r.get("target", "?"))
        r["_label"] = sd.replace("_matvec", "").replace("_bf16_0", "").replace("_0", "")
    rows.sort(key=lambda r: r["_label"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", required=True,
                    help="LABEL=path/to/summary_metrics.csv (repeatable)")
    ap.add_argument("-o", "--output", default="trace_metrics.png")
    ap.add_argument("--title", default="Decode matvec trace metrics (cycles-in-state %)")
    args = ap.parse_args()

    panels = []
    for spec in args.csv:
        label, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            raise SystemExit(f"missing CSV for {label!r}: {path}")
        panels.append((label, _load(path)))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.2 * n), squeeze=False)
    for ax, (label, rows) in zip(axes[:, 0], panels):
        kernels = [r["_label"] for r in rows]
        x = np.arange(len(kernels))
        nm = len(METRICS)
        w = 0.8 / nm
        for i, (col, leg, color) in enumerate(METRICS):
            vals = [float(r.get(col, 0) or 0) for r in rows]
            bars = ax.bar(x + (i - nm / 2) * w + w / 2, vals, w,
                          label=leg, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{label}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(kernels, rotation=0)
        ax.set_ylabel("% of trace span")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=len(METRICS), fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
