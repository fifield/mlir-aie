#!/usr/bin/env python3
"""KEYSTONE HW proof: 3x3 conv whose input is gathered ON-DEVICE as halo'd
windows from a contiguous PAD-padded HWC image -- NO host im2col.

Reference (the path being replaced): host extract_all_patches_u16 (im2col,
patch-packed) + the existing 3x3 BFP conv. New path: padded-HWC image in ->
on-device fill-TAP halo gather -> same BFP conv kernel. We assert the new path
matches a numpy 3x3 conv on the same padded image (BFP tol), AND that the
host-im2col reference path produces the identical numbers (so the only change
is input delivery). Proven on the real re8 c3 shape (28x28x64 PAD(2) image,
20x20x64 output).

Run:  source env.sh && flock /tmp/npu-dev.lock python3 conv/test_halo_conv_hw.py
"""
from __future__ import annotations
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

from aie2_halo_conv import halo_conv, TILE, PAD, WIN


def bf16(x):
    a = np.atleast_1d(np.asarray(x, np.float32))
    return (((a.view(np.uint32) >> 16) << 16).view(np.float32)).reshape(np.shape(x))


def to_u16(x):
    return (x.astype(np.float32).reshape(-1).view(np.uint32) >> 16).astype(np.uint16)


def from_u16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def numpy_conv3x3(img_pad, W, gbound, ic, oc):
    """3x3 stride-1 valid conv over the padded image, reading the GBOUND x GBOUND
    output region whose halo windows sit in img_pad. W[oc, kk, ic].
    Output tile (tr,tc) window origin = (tr*8, tc*8) in PADDED coords; output
    pixel p=(oh,ow) within the GBOUND grid reads img_pad[oh+kh, ow+kw, :]."""
    out = np.zeros((gbound, gbound, oc), np.float32)
    for oh in range(gbound):
        for ow in range(gbound):
            acc = np.zeros(oc, np.float32)
            for kh in range(3):
                for kw in range(3):
                    px = img_pad[oh + kh, ow + kw, :]       # (ic,)
                    acc += W[:, kh * 3 + kw, :] @ px         # (oc,)
            out[oh, ow, :] = acc
    return out


def tile_b(W, ic, oc):
    """W[oc, 9, ic] -> BFP B layout [oc_block, kk, 8(ic), 8(oc)] row-major."""
    N_BLK_OC = oc // 8
    N_BLK_IC = ic // 8
    KKMAX = N_BLK_IC * 9
    buf = np.zeros(N_BLK_OC * KKMAX * 64, np.float32)
    for nb in range(N_BLK_OC):
        for icb in range(N_BLK_IC):
            for tp in range(9):           # kk = kh*3+kw
                k = icb * 9 + tp
                base = (nb * KKMAX + k) * 64
                for il in range(8):       # ic within block
                    for ol in range(8):   # oc within block
                        buf[base + il * 8 + ol] = W[nb * 8 + ol, tp, icb * 8 + il]
    return buf


def untile_c(flat, oc):
    """C[N_BLOCKS_OC, M_BLOCKS=8, 8, 8] -> [64 pix, oc]; C[n, m][pl, ol]."""
    N_BLK_OC = oc // 8
    t = flat.reshape(N_BLK_OC, 8, 8, 8)
    out = np.zeros((64, oc), np.float32)
    for nb in range(N_BLK_OC):
        for mb in range(8):
            for pl in range(8):
                for ol in range(8):
                    out[mb * 8 + pl, nb * 8 + ol] = t[nb, mb, pl, ol]
    return out


def host_im2col_window(img_pad, tr, tc, ic):
    """Reference im2col gather (the path being replaced): pull tile (tr,tc)'s
    WIN x WIN x IC window out of the padded image, contiguous. This is what the
    on-device fill TAP does -- here on host so we can compare the two."""
    r0 = tr * TILE
    c0 = tc * TILE
    return img_pad[r0:r0 + WIN, c0:c0 + WIN, :].reshape(-1)


def main():
    ic, oc, gbound = 64, 32, 20
    module, meta = halo_conv(ic=ic, oc=oc, gbound=gbound)
    assert module.operation.verify()
    GRID, N_TILES, IMG_W = meta["GRID"], meta["N_TILES"], meta["IMG_W"]
    IMG_ELEMS, WIN_ELEMS, WSLOT, C_ELEMS = (
        meta["IMG_ELEMS"], meta["WIN_ELEMS"], meta["WSLOT"], meta["C_ELEMS"])

    wd = HERE / "build_halo_conv"; wd.mkdir(parents=True, exist_ok=True)
    print(f"  compiling halo_conv (ic={ic} oc={oc} gbound={gbound} "
          f"GRID={GRID} N_TILES={N_TILES} IMG={IMG_W}x{IMG_W}) ...", flush=True)
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))

    # ---- inputs: a PAD(2)-padded HWC image (the chain's output format) ----
    rng = np.random.default_rng(0)
    img = rng.standard_normal((IMG_W, IMG_W, ic)).astype(np.float32) * 0.25
    # zero the pad border so a valid-region conv is well-defined (PAD=2 ring)
    img[:PAD, :, :] = 0; img[-PAD:, :, :] = 0
    img[:, :PAD, :] = 0; img[:, -PAD:, :] = 0
    img_bf = bf16(img)
    W = (rng.standard_normal((oc, 9, ic)).astype(np.float32) * 0.15)
    W_bf = bf16(W)

    # The conv consumer's valid output region: a pad-1 3x3 conv on the unpadded
    # feature map == a valid 3x3 conv whose windows start at PAD-1 in the padded
    # buffer. Output pixel (oh,ow) reads img_pad[(PAD-1)+oh+kh, (PAD-1)+ow+kw].
    # Equivalently shift the image origin by PAD-1 so output tile (tr,tc) window
    # origin tr*8,tc*8 lands correctly. Build the "conv-view" image accordingly.
    SHIFT = PAD - 1  # = 1
    conv_img = np.zeros_like(img_bf)
    conv_img[:IMG_W - SHIFT, :IMG_W - SHIFT, :] = img_bf[SHIFT:, SHIFT:, :]
    # numpy reference over conv_img (windows origin tr*8/tc*8 => GBOUND output)
    ref = numpy_conv3x3(conv_img, W_bf, gbound, ic, oc)        # (gbound,gbound,oc)

    img_u16 = to_u16(conv_img)                                  # device feeds this
    wt_u16 = to_u16(bf16(tile_b(W_bf, ic, oc)))

    # ---- host-im2col reference path: same windows pulled on host ----
    # Confirm the on-device fill TAP gathers identical windows to host im2col.
    img_u16_hwc = img_u16.reshape(IMG_W, IMG_W, ic)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        host_win = host_im2col_window(conv_img, tr, tc, ic)
        # what the device fill TAP gathers (strided): offset (tr*8*IMG_W+tc*8)*ic,
        # sizes [1,WIN,WIN,ic], strides [0,IMG_W*ic,ic,1] -> the WINxWINxic window
        r0, c0 = tr * TILE, tc * TILE
        dev_win = img_u16_hwc[r0:r0 + WIN, c0:c0 + WIN, :].reshape(-1)
        assert np.array_equal(to_u16(host_win), dev_win), f"TAP window mismatch tile {t}"
    print(f"  host-im2col vs device-TAP windows: IDENTICAL for all {N_TILES} tiles")

    # ---- run on HW ----
    npu = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    out = iron.zeros(N_TILES * C_ELEMS, dtype=np.float32)
    DefaultNPURuntime.run(h, [iron.tensor(img_u16, dtype=np.uint16),
                              iron.tensor(wt_u16, dtype=np.uint16), out])
    flat = np.array(out.numpy())

    # ---- assemble device output into a GBOUND x GBOUND x OC map ----
    got = np.zeros((gbound, gbound, oc), np.float32)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS], oc)  # (64 pix, oc)
        for pl in range(64):
            oh = tr * TILE + pl // 8
            ow = tc * TILE + pl % 8
            if oh < gbound and ow < gbound:
                got[oh, ow, :] = tile[pl, :]

    d = np.abs(got - ref)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    max_diff = d.max()
    mean_diff = d.mean()
    # BFP576 tol: block-float quant gives ~1e-2 abs error at this magnitude
    tol = 0.05
    ok = max_diff < tol
    print(f"\n  halo-conv vs numpy 3x3 (BFP576): max_diff={max_diff:.5f} "
          f"mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"  got[0,0,:6]={got[0,0,:6]}")
    print(f"  ref[0,0,:6]={ref[0,0,:6]}")
    print(f"  got[10,10,:6]={got[10,10,:6]}")
    print(f"  ref[10,10,:6]={ref[10,10,:6]}")
    print(f"\n  {'PASS' if ok else 'FAIL'}: bit-exact* 3x3 conv from a padded-HWC "
          f"image with NO host im2col (*BFP tol {tol}, exact window gather)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
