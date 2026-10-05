#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Report event occupancy, tile periods and stalls from a parsed GEMM trace.

Input is trace.json from python/utils/trace/parse.py; --timeline writes a PNG.
"""

import argparse
import json
import statistics
from collections import defaultdict


def load(path):
    with open(path) as f:
        events = json.load(f)
    procs, threads = {}, {}
    open_b = {}
    spans = defaultdict(list)  # (pid, tid) -> [(start, end)]
    for e in events:
        ph = e.get("ph")
        if ph == "M":
            if e["name"] == "process_name":
                procs[e["pid"]] = e["args"]["name"]
            elif e["name"] == "thread_name":
                threads[(e["pid"], e["tid"])] = e["args"]["name"]
        elif ph == "B":
            open_b[(e["pid"], e["tid"])] = e["ts"]
        elif ph == "E":
            k = (e["pid"], e["tid"])
            if k in open_b:
                spans[k].append((open_b.pop(k), e["ts"]))
        elif ph == "i":
            spans[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + 1))
    # Close unfinished events at the capture's last timestamp.
    t_end = max((e["ts"] for e in events if "ts" in e), default=0)
    for k, t in open_b.items():
        spans[k].append((t, t_end))
    return procs, threads, spans


def overlap(spans, lo, hi):
    return sum(max(0, min(e, hi) - max(s, lo)) for s, e in spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_json")
    ap.add_argument("--clk-ghz", type=float, default=1.2)
    ap.add_argument("--gaps", type=int, default=8, help="largest gaps to list")
    ap.add_argument("--timeline", help="write a timeline PNG here")
    args = ap.parse_args()

    procs, threads, spans = load(args.trace_json)
    us = lambda cyc: cyc / (args.clk_ghz * 1e3)
    by_pid = defaultdict(dict)
    for (pid, tid), name in threads.items():
        by_pid[pid][name] = sorted(spans.get((pid, tid), []))

    all_t = [t for v in spans.values() for s in v for t in s]
    if not all_t:
        print("trace is empty")
        return
    t0, t1 = min(all_t), max(all_t)
    print(f"trace window: {t1 - t0} cycles = {us(t1 - t0):.1f} us\n")

    for pid in sorted(by_pid):
        evs = by_pid[pid]
        print(f"== {procs.get(pid, pid)}")
        for name, sp in sorted(evs.items()):
            if not sp:
                continue
            busy = overlap(sp, t0, t1)
            print(f"   {name:36s} {100 * busy / (t1 - t0):6.1f}%  ({len(sp)} spans)")

        starts = [s for s, _ in evs.get("INSTR_EVENT_0", [])]
        if len(starts) > 2:
            periods = [b - a for a, b in zip(starts, starts[1:])]
            med = statistics.median(periods)
            mm = evs.get("INSTR_MATRIX", [])
            first, last = starts[0], starts[-1]
            mm_busy = overlap(mm, first, last)
            print(
                f"   tiles: {len(starts)}, period median {med:.0f} cyc, "
                f"min {min(periods)}, max {max(periods)}; MMAC busy "
                f"{100 * mm_busy / max(1, last - first):.1f}% of "
                f"{us(last - first):.1f} us between first/last tile"
            )
            gaps = sorted(
                (
                    (b - a, a, b)
                    for a, b in zip(starts, starts[1:])
                    if b - a > 1.5 * med
                ),
                reverse=True,
            )
            lost = sum(g - med for g, _, _ in gaps)
            print(
                f"   {len(gaps)} long periods (>1.5x median) cost "
                f"{lost:.0f} cyc = {us(lost):.1f} us"
            )
            for g, a, b in gaps[: args.gaps]:
                why = ", ".join(
                    f"{n} {overlap(evs[n], a, b)}"
                    for n in ("STREAM_STALL", "LOCK_STALL", "MEMORY_STALL")
                    if n in evs and overlap(evs[n], a, b)
                )
                print(f"     @{a - t0:>8} +{g:>6} cyc  stalls: {why or 'none'}")
        print()

    if args.timeline:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [
            (procs.get(pid, pid), name, sp)
            for pid in sorted(by_pid)
            for name, sp in sorted(by_pid[pid].items())
            if sp
        ]
        fig, ax = plt.subplots(figsize=(16, 0.28 * len(rows) + 1.5))
        for i, (unit, name, sp) in enumerate(rows):
            ax.broken_barh(
                [(us(s - t0), max(us(e - s), 0.02)) for s, e in sp], (i, 0.8)
            )
        ax.set_yticks([i + 0.4 for i in range(len(rows))])
        ax.set_yticklabels(
            [f"{u.split(' for ')[-1]}: {n}" for u, n, _ in rows], fontsize=6
        )
        ax.invert_yaxis()
        ax.set_xlabel("us")
        ax.set_title("gemm_int8 event trace (column 0)")
        fig.tight_layout()
        fig.savefig(args.timeline, dpi=130)
        print(f"timeline -> {args.timeline}")


if __name__ == "__main__":
    main()
