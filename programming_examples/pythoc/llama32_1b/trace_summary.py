#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate one sweep dir of AIE trace.json files into a single summary.

For every per-target subdir produced by ``trace_sweep.py``:
  - Pair Chrome B/E events per (pid, tid, name) to recover per-event durations.
  - Sum durations per event name → cycles-in-state.
  - Compute the trace span = max(B.ts) - min(B.ts) on the core trace pid.
  - Derive headline metrics:
      vec_util      = INSTR_VECTOR / span
      lock_stall    = LOCK_STALL / span
      dma_in0_eff   = PORT_RUNNING_0 / (PORT_RUNNING_0 + DMA_S2MM_0_STREAM_STARVATION)
      dma_in1_eff   = PORT_RUNNING_1 / (PORT_RUNNING_1 + DMA_S2MM_1_STREAM_STARVATION)

Emits ``summary_metrics.csv`` (machine-readable) and ``summary_metrics.md``
(human-readable) next to the sweep dir's existing ``summary.csv``.

Usage::

  python3 trace_summary.py build_peano/trace_sweep/20260521T204319
  python3 trace_summary.py SWEEP_DIR --sort lock_stall --top 5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


_WIDTH_RE = re.compile(r"^[<>=^]?(\d+)")


def _width_of(fmt: str) -> int:
    """Extract the field-width prefix from a format spec (e.g. ``>12,d`` → 12)."""
    m = _WIDTH_RE.search(fmt)
    return int(m.group(1)) if m else 0

CORE_EVENTS = (
    "INSTR_EVENT_0", "INSTR_EVENT_1", "INSTR_VECTOR",
    "LOCK_STALL", "MEMORY_STALL",
    "PORT_RUNNING_0", "PORT_RUNNING_1", "PORT_RUNNING_2",
)
MEM_EVENTS = (
    "DMA_S2MM_0_START_TASK", "DMA_S2MM_1_START_TASK", "DMA_MM2S_0_START_TASK",
    "DMA_S2MM_0_FINISHED_TASK", "DMA_S2MM_1_FINISHED_TASK",
    "DMA_MM2S_0_FINISHED_TASK",
    "DMA_S2MM_0_STREAM_STARVATION", "DMA_S2MM_1_STREAM_STARVATION",
)

# Columns shown in the markdown table (others land in CSV only).
MD_COLUMNS = [
    ("target",       "target",           "<48s"),
    ("span",         "span_cycles",      ">12,d"),
    ("vec_util",     "vec_util_pct",     ">7.2f"),
    ("lock_stall",   "lock_stall_pct",   ">8.2f"),
    ("in0",          "port_running_0_pct", ">6.1f"),
    ("in1",          "port_running_1_pct", ">6.1f"),
    ("out",          "port_running_2_pct", ">6.1f"),
    ("starv0",       "starv0_pct",       ">7.1f"),
    ("starv1",       "starv1_pct",       ">7.1f"),
    ("dma_in0_eff",  "dma_in0_eff_pct",  ">10.1f"),
    ("dma_in1_eff",  "dma_in1_eff_pct",  ">10.1f"),
    ("launches",     "launches",         ">8d"),
]


def _pair_events(events) -> tuple[dict[str, int], int, int]:
    """Pair Chrome B/E events and return (durations_per_name, span_cycles,
    core_min_ts). Span is measured against core trace pid (assumed pid 0)."""
    open_ts: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    core_ts_min = None
    core_ts_max = None
    for e in events:
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        pid = e.get("pid")
        tid = e.get("tid")
        name = e.get("name")
        ts = int(e.get("ts", 0))
        if pid == 0 and ph == "B":
            if core_ts_min is None or ts < core_ts_min:
                core_ts_min = ts
            if core_ts_max is None or ts > core_ts_max:
                core_ts_max = ts
        key = (pid, tid, name)
        if ph == "B":
            open_ts[key].append(ts)
        else:  # E
            if open_ts[key]:
                t0 = open_ts[key].pop(0)
                totals[name] += ts - t0
    span = (core_ts_max - core_ts_min) if core_ts_min is not None else 0
    return dict(totals), span, core_ts_min or 0


def _row_for_target(sub: Path) -> dict | None:
    trace_path = sub / "trace.json"
    meta_path = sub / "meta.json"
    if not trace_path.exists() or not meta_path.exists():
        return None
    events = json.loads(trace_path.read_text())
    meta = json.loads(meta_path.read_text())

    totals, span, t0 = _pair_events(events)
    span_safe = span if span > 0 else 1
    target = meta.get("target", {})

    def pct(name: str) -> float:
        return 100.0 * totals.get(name, 0) / span_safe

    def eff(active: str, starv: str) -> float:
        a = totals.get(active, 0)
        s = totals.get(starv, 0)
        denom = a + s
        return (100.0 * a / denom) if denom > 0 else 0.0

    row: dict = {
        "target": (
            f"{target.get('kernel','?')}:"
            f"{target.get('sub_device','?')}:"
            f"{target.get('col','?')}:{target.get('row','?')}"
        ),
        "kernel": target.get("kernel"),
        "sub_device": target.get("sub_device"),
        "col": target.get("col"),
        "row": target.get("row"),
        "span_cycles": span,
        "launches": int(meta.get("launches", 0)),
        "nonzero_words": int(meta.get("nonzero_words", 0)),
        "total_words": int(meta.get("total_words", 0)),
        # Headline derived metrics (pct of span)
        "vec_util_pct": pct("INSTR_VECTOR"),
        "lock_stall_pct": pct("LOCK_STALL"),
        "port_running_0_pct": pct("PORT_RUNNING_0"),
        "port_running_1_pct": pct("PORT_RUNNING_1"),
        "port_running_2_pct": pct("PORT_RUNNING_2"),
        "starv0_pct": pct("DMA_S2MM_0_STREAM_STARVATION"),
        "starv1_pct": pct("DMA_S2MM_1_STREAM_STARVATION"),
        # Efficiency: active / (active + starved)
        "dma_in0_eff_pct": eff("PORT_RUNNING_0", "DMA_S2MM_0_STREAM_STARVATION"),
        "dma_in1_eff_pct": eff("PORT_RUNNING_1", "DMA_S2MM_1_STREAM_STARVATION"),
    }
    # All raw cycles-in-state (for postmortem).
    for name in CORE_EVENTS + MEM_EVENTS:
        row[f"cy_{name}"] = totals.get(name, 0)
    return row


def _render_markdown(rows: list[dict], sort_by: str, top: int | None) -> str:
    rows = list(rows)
    rows.sort(key=lambda r: r.get(sort_by, 0), reverse=True)
    if top is not None:
        rows = rows[:top]
    header_cells = [hdr for (hdr, _, _) in MD_COLUMNS]
    sep_cells = [":---" if i == 0 else "---:" for i in range(len(MD_COLUMNS))]
    out = ["| " + " | ".join(header_cells) + " |",
           "| " + " | ".join(sep_cells) + " |"]
    for r in rows:
        cells = []
        for (_, key, _) in MD_COLUMNS:
            v = r.get(key, "")
            if isinstance(v, int):
                cells.append(f"{v:,}")
            elif isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _render_text(rows: list[dict], sort_by: str, top: int | None) -> str:
    rows = list(rows)
    rows.sort(key=lambda r: r.get(sort_by, 0), reverse=True)
    if top is not None:
        rows = rows[:top]
    header = []
    for (hdr, key, fmt) in MD_COLUMNS:
        width = _width_of(fmt)
        align = ">" if ">" in fmt else "<"
        header.append(f"{hdr:{align}{width}s}")
    lines = ["  ".join(header)]
    lines.append("-" * len(lines[0]))
    for r in rows:
        cells = []
        for (_, key, fmt) in MD_COLUMNS:
            v = r.get(key, "")
            try:
                cells.append(format(v, fmt))
            except (TypeError, ValueError):
                width = _width_of(fmt)
                cells.append(str(v).rjust(width))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", type=Path,
                   help="Sweep dir produced by trace_sweep.py "
                        "(e.g. build_peano/trace_sweep/20260521T204319)")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="Override CSV output path (default: <sweep>/summary_metrics.csv)")
    p.add_argument("--md-out", type=Path, default=None,
                   help="Override Markdown output path (default: <sweep>/summary_metrics.md)")
    p.add_argument("--sort", default="span_cycles",
                   help="Column to sort by (default: span_cycles, descending). "
                        "Try lock_stall_pct, vec_util_pct, starv0_pct, …")
    p.add_argument("--top", type=int, default=None,
                   help="Only render the top N rows in the printed table "
                        "(CSV always carries all rows)")
    args = p.parse_args(argv)

    if not args.sweep_dir.is_dir():
        p.error(f"not a directory: {args.sweep_dir}")

    rows: list[dict] = []
    for sub in sorted(args.sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            r = _row_for_target(sub)
        except Exception as e:
            print(f"  [warn] skipping {sub.name}: {e}", file=sys.stderr)
            continue
        if r is None:
            continue
        rows.append(r)

    if not rows:
        p.error(f"no per-target trace data found under {args.sweep_dir}")

    csv_path = args.csv_out or (args.sweep_dir / "summary_metrics.csv")
    md_path = args.md_out or (args.sweep_dir / "summary_metrics.md")

    # CSV: all columns, alphabetically stable
    cols = sorted({k for r in rows for k in r})
    # Move the common ones to the front for readability
    front = ["target", "kernel", "sub_device", "col", "row",
             "span_cycles", "launches",
             "vec_util_pct", "lock_stall_pct",
             "port_running_0_pct", "port_running_1_pct", "port_running_2_pct",
             "starv0_pct", "starv1_pct",
             "dma_in0_eff_pct", "dma_in1_eff_pct",
             "nonzero_words", "total_words"]
    rest = [c for c in cols if c not in front]
    ordered_cols = [c for c in front if c in cols] + rest

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Files always carry the full set of targets; --top only affects what we
    # print to stdout.
    md_path.write_text(_render_markdown(rows, args.sort, top=None))

    print(_render_text(rows, args.sort, args.top))
    print()
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Targets:  {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
