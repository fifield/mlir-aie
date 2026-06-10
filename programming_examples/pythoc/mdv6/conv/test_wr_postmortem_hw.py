#!/usr/bin/env python3
"""Post-mortem stall probe for the wt-replay chain.

Sentinel-fills the image BOs, runs wt-only (compute=2), and on timeout syncs
DDR back to count which (col, worker, round) tiles drained. Drained tiles
overwrite the sentinel, so the stall round/col is visible.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aie.utils.compile import compile_mlir_module
from aie2_rn3_chain_geo import rn3_chain_raster_wr, raster_params, TILE, PAD
from conv.resident_xclbin_runner import ResidentXCLBinRunner

GEO = "re4w"
N_ITERS = int(os.environ.get("N_ITERS", "1"))
SENT = 0x7777


def main():
    p = raster_params(GEO)
    ic, G, IMG = p["IC"], p["GBOUND"], p["IMG"]
    COLS, NWORK, TPR, GRID = p["COLS"], p["NWORK"], p["TPR"], p["GRID"]
    bd = Path(__file__).parent / "build_wr_nc"
    if not (bd / "final.xclbin").exists():
        (bd / "work").mkdir(parents=True, exist_ok=True)
        compile_mlir_module(mlir_module=rn3_chain_raster_wr(GEO, N_ITERS, compute=2),
                            insts_path=str(bd / "insts.bin"), xclbin_path=str(bd / "final.xclbin"),
                            work_dir=str(bd / "work"))
    wt = np.zeros(N_ITERS * 2 * p["N_BLK"] * p["WSLOT"], np.uint16)
    img = np.full(p["IMG_ELEMS"], SENT, np.uint16)
    r = ResidentXCLBinRunner(bd / "final.xclbin", bd / "insts.bin")
    try:
        r.run(img, wt, img.copy(), bo_key="pm", output_indices={0, 2}, inout_indices={0, 2})
        print("COMPLETED (no stall)")
        return
    except RuntimeError as e:
        print("TIMEOUT — post-mortem:", e)

    tens = r._bo_cache["pm"]
    for t in (tens[0], tens[2]):
        t._sync_from_device()
    dst = np.array(tens[2].data, copy=True)  # iter0 drains to BO2

    JUNK_ROW = GRID * TILE + 2 * PAD
    N_TILES = GRID * GRID
    print("round-major drained tiles per col (iter0 -> BO2):")
    for rnd in range(TPR):
        row = []
        for c in range(COLS):
            done = 0
            for w in range(NWORK):
                idx = (c * NWORK + w) * TPR + rnd
                if idx < N_TILES:
                    gr, gc = (idx // GRID) * TILE, (idx % GRID) * TILE
                    off = ((PAD + gr) * IMG + PAD + gc) * ic
                else:
                    k = idx - N_TILES
                    off = ((JUNK_ROW + (k // GRID) * TILE) * IMG + (k % GRID) * TILE) * ic
                if dst[off] != SENT:
                    done += 1
            row.append(done)
        print(f"  round {rnd}: {row}")


if __name__ == "__main__":
    main()
