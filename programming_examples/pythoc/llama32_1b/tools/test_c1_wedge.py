#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bisect the post-run NPU wedge left by the c1_merged o_gemv_ffn device.

A completed c1_merged run leaves the NPU so the NEXT process's first command
times out (alternating fail/pass). This harness compiles the c1_merged device
ALONE (no D4) with PYTHOC_C1_STAGES stages, runs it once, exits; then a child
probe process loads it again and runs once. If the parent run wedged the
device, the probe times out.

Usage: python3 tools/test_c1_wedge.py --stages 6
Exit 0 = no wedge, 1 = probe timed out / wedge.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

EMB_DIM = 2048
HIDDEN_DIM = 8192

NAME = "c1_wedge_probe"  # suffixed per stage count


def _inputs():
    rng = np.random.default_rng(7)
    rand = lambda shape: rng.standard_normal(shape).astype(bfloat16) * 0.02
    return [
        rand((EMB_DIM, EMB_DIM)),                  # 0 wo
        rand((EMB_DIM,)),                          # 1 attn_out
        np.zeros((EMB_DIM,), dtype=bfloat16),      # 2 proj
        rand((EMB_DIM,)),                          # 3 x_resid
        np.zeros((EMB_DIM,), dtype=bfloat16),      # 4 res1
        rand((EMB_DIM,)),                          # 5 ffn_norm_w
        np.zeros((EMB_DIM,), dtype=bfloat16),      # 6 normed2
        rand((HIDDEN_DIM, EMB_DIM)),               # 7 wgate
        np.zeros((HIDDEN_DIM,), dtype=bfloat16),   # 8 gate
        rand((HIDDEN_DIM, EMB_DIM)),               # 9 wup
        np.zeros((HIDDEN_DIM,), dtype=bfloat16),   # 10 up
        np.zeros((HIDDEN_DIM,), dtype=bfloat16),   # 11 swiglu
        rand((EMB_DIM, HIDDEN_DIM)),               # 12 wdown
        np.zeros((EMB_DIM,), dtype=bfloat16),      # 13 down
        np.zeros((EMB_DIM,), dtype=bfloat16),      # 14 output
    ]


def run_once(workdir: Path, compile_: bool, name: str = NAME) -> None:
    from builders.o_gemv_ffn import build_o_gemv_ffn_module
    from kernel_builder.cache import KernelCache

    obj_dir = PROJECT_DIR / "build_peano"
    os.chdir(obj_dir)
    cache = KernelCache(cache_dir=workdir, verbose=False)
    if compile_:
        mode = os.environ.get("PYTHOC_WEDGE_MODE", "c1_merged")
        seq = (mode, "d4_dg_a2_pack") if os.environ.get(
            "PYTHOC_C1_WITH_D4") else (mode,)
        ir = build_o_gemv_ffn_module(pack_mode=mode, dispatch_sequence=seq)
        cache.compile_and_cache(name, ir, instance_name="o_gemv_ffn")
        cache._save_manifest()
    else:
        cache.load_manifest()
    iters = int(os.environ.get("PYTHOC_C1_ITERS", "1"))
    t0 = time.perf_counter()
    for _ in range(iters):
        cache.load_and_run(name, None, *_inputs(),
                           output_indices=[2, 4, 6, 8, 10, 11])
    print(f"run ok x{iters} in {time.perf_counter() - t0:.3f}s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default=os.environ.get("PYTHOC_C1_STAGES", "6"))
    p.add_argument("--probe", action="store_true", help="run once, no compile")
    p.add_argument("--workdir", type=Path,
                   default=PROJECT_DIR / "build_peano" / "c1_wedge_cache")
    args = p.parse_args()
    os.environ["PYTHOC_C1_STAGES"] = str(args.stages)
    os.environ["PYTHOC_C2_STAGES"] = str(args.stages)
    args.workdir.mkdir(parents=True, exist_ok=True)

    name = f"{NAME}_{os.environ.get('PYTHOC_WEDGE_MODE', 'c1_merged')}_s{args.stages}" + ("_d4" if os.environ.get("PYTHOC_C1_WITH_D4") else "")
    if args.probe:
        run_once(args.workdir, compile_=False, name=name)
        return 0

    # Sacrificial probe to clear any wedge inherited from a previous run
    # (a wedged device times out exactly one command and then recovers).
    if (args.workdir / "manifest.json").exists():
        subprocess.run(
            [sys.executable, __file__, "--probe", "--stages", str(args.stages),
             "--workdir", str(args.workdir)],
            timeout=180, capture_output=True, text=True)

    run_once(args.workdir, compile_=True, name=name)
    # Probe in a fresh process: it times out if we wedged the NPU.
    proc = subprocess.run(
        [sys.executable, __file__, "--probe", "--stages", str(args.stages),
         "--workdir", str(args.workdir)],
        timeout=180, capture_output=True, text=True, env=os.environ.copy())
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr[-800:])
    print(f"stages={args.stages}: {'NO WEDGE' if proc.returncode == 0 else 'WEDGED'}")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
