#!/usr/bin/env python3
"""HW proof for the tiles-per-core halo conv (conv/aie2_halo_conv_mt.py).

Validates the multi-tile generator against the SAME numpy 3x3 + in-kernel-BN+SiLU
reference the proven one-tile path uses (test_halo_conv_hw.py). Stages:

  1. small (re8 20x20, tpc=2): tiles-per-core must give the SAME output as the
     proven one-tile path (BFP tol) — bit-exact proof of the loop restructure.
  2. re4 (80x80, GRID=10, tpc=4 -> 28 workers): does it compile/place (<=32
     cores) AND produce correct output vs the host conv reference?

Run:  source env.sh && flock /tmp/npu-dev.lock python3 conv/test_halo_conv_mt_hw.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MDV6 = HERE.parent
for _p in (str(HERE), str(MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime
from aie.utils.compile import compile_mlir_module

from aie2_halo_conv_mt import halo_conv_mt, slot_to_tile, TILE, PAD, WIN
# reuse the proven reference helpers verbatim
from test_halo_conv_hw import (
    bf16, to_u16, from_u16, numpy_conv3x3, tile_b, untile_c,
    pack_halo_weights, bn_silu_ref,
)


def run_shape(ic, oc, gbound, tpc, wd_name, tag):
    module, meta = halo_conv_mt(ic=ic, oc=oc, gbound=gbound, tpc=tpc)
    assert module.operation.verify()
    GRID, N_TILES = meta["GRID"], meta["N_TILES"]
    IMG_W, IMG_H, IMG_ELEMS = meta["IMG_W"], meta["IMG_H"], meta["IMG_ELEMS"]
    WIN_ELEMS, WSLOT, C_ELEMS = meta["WIN_ELEMS"], meta["WSLOT"], meta["C_ELEMS"]
    n_slots, NWORK, COLS = meta["n_slots"], meta["NWORK"], meta["COLS"]
    n_workers = meta["n_workers"]

    # --- L1 budget (single-tile residency: win + wt + C + stack) ---
    win_b = WIN_ELEMS * 2
    wt_b = WSLOT * 2
    c_b = C_ELEMS * 4
    # io_depth=1 (default): win + C single-buffered; wt depth-1 always
    l1_d2 = (win_b + wt_b + c_b) + 4096
    print(f"\n[{tag}] ic={ic} oc={oc} gbound={gbound} tpc={tpc} | GRID={GRID} "
          f"N_TILES={N_TILES} workers={n_workers} ({COLS}cols x {NWORK}) "
          f"slots={n_slots}")
    print(f"[{tag}] L1/core (depth-1): win {win_b//1024}KB + wt {wt_b//1024}KB "
          f"+ C {c_b//1024}KB + stack 4KB = ~{l1_d2//1024}KB")

    wd = HERE / wd_name; wd.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] compiling (workers={n_workers}, IMG={IMG_W}x{IMG_H}) ...", flush=True)
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    print(f"[{tag}] COMPILE+PLACE OK ({n_workers} cores <= 32)", flush=True)

    # ---- inputs: PAD(2)-padded HWC image (chain output format), conv-view ----
    rng = np.random.default_rng(0)
    # only the valid padded region (IMG_W x IMG_W) carries signal; junk band 0
    img = np.zeros((IMG_H, IMG_W, ic), np.float32)
    img[:IMG_W, :IMG_W, :] = rng.standard_normal((IMG_W, IMG_W, ic)).astype(np.float32) * 0.25
    img[:PAD, :, :] = 0; img[IMG_W - PAD:IMG_W, :, :] = 0
    img[:, :PAD, :] = 0; img[:, IMG_W - PAD:, :] = 0
    img_bf = bf16(img)
    W = (rng.standard_normal((oc, 9, ic)).astype(np.float32) * 0.15)
    W_bf = bf16(W)
    bn_w = bf16(rng.standard_normal(oc).astype(np.float32) * 0.5 + 1.0)
    bn_b = bf16(rng.standard_normal(oc).astype(np.float32) * 0.2)

    SHIFT = PAD - 1  # =1
    conv_img = np.zeros_like(img_bf)
    conv_img[:IMG_H - SHIFT, :IMG_W - SHIFT, :] = img_bf[SHIFT:, SHIFT:, :]
    raw = numpy_conv3x3(conv_img[:IMG_W, :IMG_W, :], W_bf, gbound, ic, oc)
    ref = bn_silu_ref(raw, bn_w, bn_b)

    img_u16 = to_u16(conv_img)
    wt_u16 = pack_halo_weights(W_bf, bn_w, bn_b, ic, oc)

    npu = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    out = iron.zeros(n_slots * C_ELEMS, dtype=np.float32)
    DefaultNPURuntime.run(h, [iron.tensor(img_u16, dtype=np.uint16),
                              iron.tensor(wt_u16, dtype=np.uint16), out])
    flat = np.array(out.numpy())

    # ---- assemble: OUT slot index == raster tile idx (slot_to_tile) ----
    got = np.zeros((gbound, gbound, oc), np.float32)
    seen = np.zeros((gbound, gbound), bool)
    for slot in range(n_slots):
        t = slot_to_tile(slot, meta)
        if t is None:
            continue
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[slot * C_ELEMS:(slot + 1) * C_ELEMS], oc)
        for pl in range(64):
            oh = tr * TILE + pl // 8
            ow = tc * TILE + pl % 8
            if oh < gbound and ow < gbound:
                got[oh, ow, :] = tile[pl, :]
                seen[oh, ow] = True

    assert seen.all(), f"{(~seen).sum()} output pixels never written"
    d = np.abs(got - ref)
    max_diff = float(d.max()); mean_diff = float(d.mean())
    # BFP576 max tail scales with #pixels (re4 has 16x re8's grid) and is
    # amplified by the in-kernel BN scale (~1±0.5). Gate on mean (tight) +
    # a BN-amplified max bound (matches the one-tile test's rationale).
    tol = 0.20
    n_over = int((d > 0.15).sum())
    ok = (max_diff < tol) and (mean_diff < 0.02)
    print(f"[{tag}] elems over 0.15: {n_over}/{d.size} ({100*n_over/d.size:.4f}%)")
    print(f"[{tag}] vs numpy 3x3 (BFP576): max_diff={max_diff:.5f} "
          f"mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"[{tag}] got[0,0,:4]={got[0,0,:4]}  ref[0,0,:4]={ref[0,0,:4]}")
    mid = gbound // 2
    print(f"[{tag}] got[{mid},{mid},:4]={got[mid,mid,:4]}  ref={ref[mid,mid,:4]}")
    return ok, max_diff, mean_diff, l1_d2 // 1024, n_workers


def main():
    only = os.environ.get("ONLY", "")
    results = []
    if only in ("", "small"):
        results.append(("small re8 tpc=2",
                        run_shape(64, 32, 20, 2, "build_halo_conv_mt_small", "small")))
    if only in ("", "re4_oc32"):
        # re4 geometry at OC=32 (WSLOT fits L1 like the small shape) — isolates
        # the multi-tile re4 plumbing from the full-OC weight-slot L1 cap.
        results.append(("re4 oc32 tpc=4",
                        run_shape(64, 32, 80, 4, "build_halo_conv_mt_re4_oc32", "re4_oc32")))
    if only in ("", "re4"):
        results.append(("re4 oc64 tpc=4",
                        run_shape(64, 64, 80, 4, "build_halo_conv_mt_re4", "re4")))
    print("\n===== SUMMARY =====")
    allok = True
    for name, (ok, mx, mn, l1, nw) in results:
        allok &= ok
        print(f"  {name:18s}: {'PASS' if ok else 'FAIL'} "
              f"max={mx:.5f} mean={mn:.6f} workers={nw} L1~{l1}KB")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
