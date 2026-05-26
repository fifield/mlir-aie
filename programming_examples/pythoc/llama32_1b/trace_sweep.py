#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Repeatable AIE-trace sweep across (kernel, sub_device, col, row) targets.

For each target this script:
  1. Invokes ``make trace KERNEL=… SUBDEVICE=… COL=… ROW=…`` with whatever
     run-time knobs the caller passes (weights, n_tokens, prompt, trace_size).
  2. Copies the resulting ``<cache>/trace/`` directory to
     ``<out_dir>/<target_slug>/`` so a later run doesn't overwrite it.
  3. Records the meta.json summary into ``<out_dir>/summary.csv``.

Targets are given on the command line (``--target K:S:C:R``, repeatable) or
read from a file (``--targets-file PATH``, one per line, ``#`` comments OK).
Example default config lives in ``trace_sweep.targets`` next to this script.

Usage::

  python3 trace_sweep.py \\
      --target o_gemv_ffn:gg_matvec_bf16_0:4:2 \\
      --target o_gemv_ffn:og_matvec_bf16_0:7:2 \\
      --weights synthetic --n-tokens 5 --trace-size 8388608
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Target:
    kernel: str
    sub_device: str
    col: int
    row: int

    @classmethod
    def parse(cls, spec: str) -> "Target":
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"trace target must be KERNEL:SUB_DEVICE:COL:ROW, got {spec!r}"
            )
        k, s, c, r = parts
        return cls(k, s, int(c), int(r))

    @property
    def slug(self) -> str:
        return f"{self.kernel}__{self.sub_device}__c{self.col}_r{self.row}"

    @property
    def spec(self) -> str:
        return f"{self.kernel}:{self.sub_device}:{self.col}:{self.row}"


def _read_targets_file(path: Path) -> list[Target]:
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(Target.parse(line))
    return out


def _which_cache(target: Target, quant: str) -> Path:
    """Where ``make trace`` will deposit ``trace/`` for this target."""
    suffix = "" if quant == "bf16" else f"_{quant}"
    if target.kernel.startswith(("flash_attn", "rms_gemms_rope", "o_ffn")):
        return REPO_ROOT / "build_peano" / f"prefill_kernel_cache{suffix}" / "trace"
    return REPO_ROOT / "build_peano" / f"decode_kernel_cache{suffix}" / "trace"


def _run_one(target: Target, args, env) -> tuple[bool, str]:
    cmd = [
        "make", "-C", str(REPO_ROOT), "trace",
        f"KERNEL={target.kernel}",
        f"SUBDEVICE={target.sub_device}",
        f"COL={target.col}",
        f"ROW={target.row}",
        f"N_TOKENS={args.n_tokens}",
        f"TRACE_SIZE={args.trace_size}",
        f"QUANT={args.quant}",
        f"PROMPT={args.prompt}",
    ]
    if args.weights:
        cmd.append(f"WEIGHTS={args.weights}")
    if args.model:
        cmd.append(f"MODEL={args.model}")
    if args.hf_model_id:
        cmd.append(f"HF_MODEL_ID={args.hf_model_id}")

    print(f"[sweep] >>> {target.spec}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        return False, f"make failed (rc={proc.returncode}, {dt:.1f}s)"
    return True, f"ok ({dt:.1f}s)"


def _archive_trace(target: Target, args, out_dir: Path) -> dict:
    src = _which_cache(target, args.quant)
    dst = out_dir / target.slug
    if not src.exists():
        return {"target": target.spec, "status": "missing_trace_dir"}
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    meta_path = dst / "meta.json"
    if not meta_path.exists():
        return {"target": target.spec, "status": "no_meta_json", "dst": str(dst)}
    meta = json.loads(meta_path.read_text())
    return {
        "target": target.spec,
        "kernel": target.kernel,
        "sub_device": target.sub_device,
        "col": target.col,
        "row": target.row,
        "launches": meta.get("launches"),
        "total_words": meta.get("total_words"),
        "nonzero_words": meta.get("nonzero_words"),
        "trace_size_bytes": meta.get("target", {}).get("trace_size_bytes"),
        "event_markers_inserted": meta.get("info", {}).get("event_markers_inserted"),
        "json": meta.get("json"),
        "status": "ok",
        "dst": str(dst),
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", action="append", default=[],
                   help="KERNEL:SUB_DEVICE:COL:ROW (repeatable)")
    p.add_argument("--targets-file", type=Path,
                   help="File with one target per line; '#' comments allowed")
    p.add_argument("--out-dir", type=Path,
                   help="Directory to deposit per-target trace copies + summary "
                        "(default: build_peano/trace_sweep/<utc-timestamp>)")
    p.add_argument("--n-tokens", type=int, default=1)
    p.add_argument("--trace-size", type=int, default=8 * 1024 * 1024)
    p.add_argument("--quant", default="bf16", choices=["bf16", "awq-emulate"])
    p.add_argument("--weights", default="synthetic",
                   help="WEIGHTS make var (default: synthetic — no HF needed)")
    p.add_argument("--model", default=None)
    p.add_argument("--hf-model-id", default=None)
    p.add_argument("--prompt", default="hi")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the sweep at the first failing target")
    args = p.parse_args(argv)

    targets: list[Target] = [Target.parse(s) for s in args.target]
    if args.targets_file:
        targets.extend(_read_targets_file(args.targets_file))
    if not targets:
        p.error("no targets — pass --target K:S:C:R or --targets-file FILE")

    if args.out_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        args.out_dir = REPO_ROOT / "build_peano" / "trace_sweep" / ts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] output dir: {args.out_dir}")

    env = os.environ.copy()
    results = []
    for t in targets:
        ok, msg = _run_one(t, args, env)
        print(f"[sweep]     run: {msg}")
        if ok:
            row = _archive_trace(t, args, args.out_dir)
        else:
            row = {"target": t.spec, "status": f"run_failed: {msg}"}
        results.append(row)
        if not ok and args.stop_on_error:
            break

    columns = [
        "target", "status", "launches", "total_words", "nonzero_words",
        "trace_size_bytes", "event_markers_inserted", "json", "dst",
    ]
    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n[sweep] summary CSV: {csv_path}")

    print(f"\n[sweep] {'target':<55s} {'launches':>9s} {'nonzero':>10s} "
          f"{'total':>10s}  status")
    for r in results:
        print(f"        {r['target']:<55s} "
              f"{str(r.get('launches','-')):>9s} "
              f"{str(r.get('nonzero_words','-')):>10s} "
              f"{str(r.get('total_words','-')):>10s}  "
              f"{r['status']}")

    failed = [r for r in results if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
