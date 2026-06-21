#!/usr/bin/env python3
"""Validate the per-oc-block-PAIR streaming halo conv (plumbing #1) standalone.

Same as test_halo_conv_hw.py but stream_oc=True at OC=64 (and optionally
IC=128/OC=128 via env), where the full-OC weight slot would overflow L1. The
drained C must match the single-slot layout bit-for-bit (BFP tol).

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_halo_conv_stream_hw.py
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

from aie2_halo_conv import halo_conv, TILE, PAD
from test_halo_conv_hw import bf16, to_u16, numpy_conv3x3, tile_b, untile_c


def main():
    ic = int(os.environ.get("HC_IC", "64"))
    oc = int(os.environ.get("HC_OC", "64"))
    gbound = 20
    module, meta = halo_conv(ic=ic, oc=oc, gbound=gbound, stream_oc=True)
    assert module.operation.verify()
    GRID, N_TILES, IMG_W = meta["GRID"], meta["N_TILES"], meta["IMG_W"]
    C_ELEMS = meta["C_ELEMS"]

    wd = HERE / "build_halo_conv_stream"; wd.mkdir(parents=True, exist_ok=True)
    print(f"  compiling streaming halo_conv (ic={ic} oc={oc}) ...", flush=True)
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))

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

    npu = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    out = iron.zeros(N_TILES * C_ELEMS, dtype=np.float32)
    DefaultNPURuntime.run(h, [iron.tensor(img_u16, dtype=np.uint16),
                              iron.tensor(wt_u16, dtype=np.uint16), out])
    flat = np.array(out.numpy())

    got = np.zeros((gbound, gbound, oc), np.float32)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS], oc)
        for pl in range(64):
            oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
            if oh < gbound and ow < gbound:
                got[oh, ow, :] = tile[pl, :]

    d = np.abs(got - ref)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    max_diff, mean_diff = float(d.max()), float(d.mean())
    tol = 0.06
    ok = max_diff < tol
    print(f"\n  STREAMING halo_conv (ic={ic} oc={oc}, N_PAIRS={oc//16}) vs numpy 3x3: "
          f"max_diff={max_diff:.5f} mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"  got[0,0,:6]={got[0,0,:6]}")
    print(f"  ref[0,0,:6]={ref[0,0,:6]}")
    print(f"\n  {'PASS' if ok else 'FAIL'}: per-oc-block-pair weight streaming "
          f"(plumbing #1) — full OC fits L1")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
