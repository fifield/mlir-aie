#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reproducer: c2 add1 wave starves on the X-broadcast fan's extreme columns.

C2 (`o_gemv_ffn` pack_mode c2_rms/c2_merged) deadlocks; this isolates *where*.
After the O matvec wave (proj) completes on all 8 columns, the add1 eltwise
wave (proj + x_resid -> res1) never produces output on specific columns, so
RMS/gate/up/swiglu downstream all stall.

Instrument: run rms_gemv_rope first (it completes from any device state and
provides a realistic predecessor PDI), then run o_gemv_ffn, then read res1
(arg4) back per column. A column with 0 non-zeros is a starved add1.

Key result — the starved columns track the X-broadcast SOURCE column
(env PYTHOC_C2_XCOL, default 0), NOT the packet ids and NOT cross-PDI state:

    XCOL=0 (east-going fan)  -> col 0 starves            res1 [0,256,256,256,256,256,256,256]
    XCOL=3 (fans both ways)  -> cols 0 and 7 starve      res1 [0,256,256,256,256,256,256,0]
    XCOL=7 (west-going fan)  -> add1 completes (but a later stage still stalls)

So add1's per-column in1 (this col's shim MM2S1) contends with the X-broadcast
that transits the shim row on the same MM2S1 lane; the fan's terminal columns
lose the arbitration and never receive in1. Deterministic (3/3 per XCOL).
Renumbering packet ids to distinct single bits (matvec=1/add=2/swiglu=4/down=8,
already applied) removes a separate mask-aliasing hazard but does NOT fix this.

Usage (from build_peano/, decode cache built for c2_rms):
    PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=c2_rms make compile   # once
    python3 tools/test_c2_add_starve.py            # sweeps XCOL unless one is set
Exit 0 = expected starvation pattern observed, 1 = no starvation, 2 = device dead.

NOTE: building each XCOL needs a recompile (the source column is baked into the
ELF). The harness only *reads* PYTHOC_C2_XCOL to label output; rebuild with the
matching env to change the routing. With no rebuild it reports the built ELF.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

EMB, HID, KV = 2048, 8192, 512


def main() -> int:
    os.chdir(PROJECT_DIR / "build_peano")
    from kernel_builder.cache import KernelCache
    import pyxrt as xrt

    rng = np.random.default_rng(7)
    rand = lambda s: (rng.standard_normal(s) * 0.05 + 0.05).astype(bfloat16)
    z = lambda n: np.zeros(n, dtype=bfloat16)
    ogf = [rand((EMB, EMB)), rand(EMB), z(EMB), rand(EMB), z(EMB), rand(EMB),
           z(EMB), rand((HID, EMB)), z(HID), rand((HID, EMB)), z(HID), z(HID),
           rand((EMB, HID)), z(EMB), z(EMB)]
    rgr = [z(EMB), z(EMB), z(EMB), z((EMB, EMB)), z(EMB), z((KV, EMB)), z(KV),
           z((KV, EMB)), z(KV), z(64), z(64), z(EMB), z(KV)]

    c = KernelCache(cache_dir=PROJECT_DIR / "build_peano" / "decode_kernel_cache",
                    verbose=False)
    if not c.load_manifest():
        print("decode_kernel_cache missing — run `make compile` first")
        return 2

    # rgr completes from any device state; gives a realistic predecessor PDI.
    cleared = False
    for i in range(4):
        try:
            c.load_and_run("rms_gemv_rope", None, *rgr, output_indices=[11, 12],
                           bo_key=f"clr{i}")
            cleared = True
            break
        except RuntimeError:
            print(f"rgr clear {i} consumed a wedge", flush=True)
    if not cleared:
        print("device unusable — inconclusive")
        return 2

    completed = True
    try:
        c.load_and_run("o_gemv_ffn", None, *ogf, output_indices=[2, 4], bo_key="o")
    except RuntimeError:
        completed = False

    b = c._cached_bos["o"][4]
    b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    a = np.frombuffer(b.map(), dtype=bfloat16, count=EMB)
    sz = EMB // 8
    cols = [int(np.count_nonzero(a[k * sz:(k + 1) * sz])) for k in range(8)]
    starved = [k for k, v in enumerate(cols) if v == 0]
    xcol = os.environ.get("PYTHOC_C2_XCOL", "0")
    print(f"XCOL={xcol}  ogf {'COMPLETED' if completed else 'TIMEOUT'}  "
          f"res1 per-col={cols}  starved add1 cols={starved}")
    if starved:
        print("REPRODUCED: add1 starves on broadcast-fan extreme column(s)")
        return 0
    print("no add1 starvation in res1 (downstream may still stall)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
