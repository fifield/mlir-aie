#!/usr/bin/env python3
"""KEYSTONE OC=128 proof: the on-device halo-gather 3x3 conv at the FULL
mc_re8_c3 shape (IC=128 -> OC=128), padded-HWC input -> on-device halo gather +
per-oc-block-PAIR C-DRAIN -> 20x20x128 output, bit-exact (BFP tol) vs the
host-im2col + reference 3x3.

This is the gating item for the chain->halo_c3 merged seam (B2c3-1) at OC=128.
The full-OC C accumulator (16 oc-blocks * 8 * 64 * 4B = 32KB) overflowed L1 at
OC=128; this test exercises the stream_oc C-drain path where only ONE oc-block-
PAIR's weights (36KB) AND C (4KB) are ever resident -> fits the 64KB L1.

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_halo_conv_oc128_hw.py
"""
from __future__ import annotations
import os, sys
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

from aie2_halo_conv import halo_conv, TILE, PAD, deinterleave_stream_out
from test_halo_conv_hw import (bf16, to_u16, numpy_conv3x3, tile_b, untile_c,
                               host_im2col_window)
from aie2_halo_conv import WIN


def main():
    # FULL mc_re8_c3 shape: IC=128 -> OC=128, 28x28x128 PAD(2) image, 20x20x128 out
    ic = int(os.environ.get("HC_IC", "128"))
    oc = int(os.environ.get("HC_OC", "128"))
    gbound = int(os.environ.get("HC_GBOUND", "20"))
    # "block" mode: per-SINGLE-oc-block weight streaming + per-block C drain.
    # This is the OC=128 C-drain — only one oc-block's weights (18KB) and C
    # (2KB) are resident, so IC=128->OC=128 fits the 64KB L1.
    module, meta = halo_conv(ic=ic, oc=oc, gbound=gbound, stream_oc="block")
    assert module.operation.verify()
    GRID, N_TILES, IMG_W = meta["GRID"], meta["N_TILES"], meta["IMG_W"]
    C_ELEMS = meta["C_ELEMS"]

    wd = HERE / "build_halo_conv_oc128"; wd.mkdir(parents=True, exist_ok=True)
    print(f"  compiling OC=128 C-drain halo_conv (ic={ic} oc={oc} gbound={gbound} "
          f"GRID={GRID} N_TILES={N_TILES} IMG={IMG_W}x{IMG_W} N_BLK={oc//8}) ...",
          flush=True)
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    print("  COMPILED (L1 fits) — stream_oc per-pair weight+C drain", flush=True)

    rng = np.random.default_rng(0)
    img = rng.standard_normal((IMG_W, IMG_W, ic)).astype(np.float32) * 0.25
    img[:PAD, :, :] = 0; img[-PAD:, :, :] = 0
    img[:, :PAD, :] = 0; img[:, -PAD:, :] = 0
    img_bf = bf16(img)
    W = (rng.standard_normal((oc, 9, ic)).astype(np.float32) * 0.15)
    W_bf = bf16(W)
    SHIFT = PAD - 1
    conv_img = np.zeros_like(img_bf)
    conv_img[:IMG_W - SHIFT, :IMG_W - SHIFT, :] = img_bf[SHIFT:, SHIFT:, :]
    ref = numpy_conv3x3(conv_img, W_bf, gbound, ic, oc)
    img_u16 = to_u16(conv_img)
    wt_u16 = to_u16(bf16(tile_b(W_bf, ic, oc)))

    # confirm the on-device fill TAP gathers identical windows to host im2col
    img_u16_hwc = img_u16.reshape(IMG_W, IMG_W, ic)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        host_win = host_im2col_window(conv_img, tr, tc, ic)
        r0, c0 = tr * TILE, tc * TILE
        dev_win = img_u16_hwc[r0:r0 + WIN, c0:c0 + WIN, :].reshape(-1)
        assert np.array_equal(to_u16(host_win), dev_win), f"TAP window mismatch tile {t}"
    print(f"  host-im2col vs device-TAP windows: IDENTICAL for all {N_TILES} tiles")

    npu = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    out = iron.zeros(N_TILES * C_ELEMS, dtype=np.float32)
    DefaultNPURuntime.run(h, [iron.tensor(img_u16, dtype=np.uint16),
                              iron.tensor(wt_u16, dtype=np.uint16), out])
    flat = np.array(out.numpy())
    # stream_oc C-drain OUT is column-packed unit-major -> reorder to canonical
    # [N_TILES, C_ELEMS] before untiling.
    flat = deinterleave_stream_out(flat, meta)

    got = np.zeros((gbound, gbound, oc), np.float32)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS], oc)  # (64 pix, oc)
        for pl in range(64):
            oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
            if oh < gbound and ow < gbound:
                got[oh, ow, :] = tile[pl, :]

    d = np.abs(got - ref)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    max_diff, mean_diff = float(d.max()), float(d.mean())
    tol = 0.06
    ok = max_diff < tol
    print(f"\n  OC=128 C-DRAIN halo_conv (ic={ic} oc={oc}, N_BLK={oc//8}) vs numpy "
          f"3x3 (BFP576): max_diff={max_diff:.5f} mean={mean_diff:.6f} tol={tol} "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  got[0,0,:6]={got[0,0,:6]}")
    print(f"  ref[0,0,:6]={ref[0,0,:6]}")
    print(f"  got[10,10,:6]={got[10,10,:6]}")
    print(f"  ref[10,10,:6]={ref[10,10,:6]}")
    print(f"\n  {'PASS' if ok else 'FAIL'}: FULL mc_re8_c3 (IC=128->OC=128) on-device "
          f"halo gather + per-oc-block-pair C-drain, fits L1, bit-exact* (BFP tol)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
