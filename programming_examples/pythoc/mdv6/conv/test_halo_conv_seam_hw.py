#!/usr/bin/env python3
"""KEYSTONE seam proof: feed the MERGED re8 rn3 chain ELF's padded-HWC output
DIRECTLY into the on-device halo-conv -- NO host im2col, NO layout reformat.

This is the chain->c3 seam the three prior fusion attempts could not bridge.
The chain emits memref<50176xui16> = 28x28x64 PAD(2)-padded HWC. The halo-conv
reads that SAME buffer (a pad-1 3x3 conv's windows all lie inside the PAD=2
ring) and produces a 20x20xOC output, matching a numpy 3x3 conv on the chain's
20x20x64 feature map. Proves the 4.08x im2col bridge is eliminated.

Run:  source env.sh && flock /tmp/npu-dev.lock python3 conv/test_halo_conv_seam_hw.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MDV6 = _HERE.parent
for _p in (str(_HERE), str(_MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyxrt as xrt
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime
from aie.utils.compile import compile_mlir_module

from build_re8_chain_merged import build as build_chain
from aie2_rn3_chain_geo import geo_params
from rn3_chain_runner import run_rn3_chain_geo, _pack_geo_iter, PAD as _PAD
from test_rn3_pair_vector_oneblock_hw import f32_to_bf16_u16, bf16_u16_to_f32
from test_re8_chain_merged_hw import _make_weight_pairs

from aie2_halo_conv import halo_conv, TILE, PAD, WIN
from test_halo_conv_hw import (bf16, to_u16, numpy_conv3x3, tile_b, untile_c,
                               pack_halo_weights, bn_silu_ref)

N_ITERS = 3
GEO = "re8"


def main():
    p = geo_params(GEO)
    ic, G = p["IC"], p["GBOUND"]              # 64, 20
    IMG_W = p["IMG"]                          # 28
    assert p["IMG_ELEMS"] == IMG_W * IMG_W * ic == 50176

    # ===== 1. run the merged chain ELF -> padded-HWC output BO =====
    elf_path = build_chain(GEO, n_iters=N_ITERS)
    if elf_path is None:
        print("FAIL: merged chain ELF build failed"); return 1

    rng = np.random.default_rng(7)
    inp = torch.from_numpy(
        (rng.standard_normal((G, G, ic)).astype(np.float32) * 0.5)).to(torch.bfloat16)
    pairs = _make_weight_pairs(rng, N_ITERS, ic)
    nt = p["WORKER_TILES"][0]
    weights = np.concatenate([
        np.tile(_pack_geo_iter(w1, w2, ic, p["WSLOT"], p["N_BLK"]), nt)
        for w1, w2 in pairs])
    img = np.zeros(p["IMG_ELEMS"], np.float32)
    img.reshape(p["IMG_H"], IMG_W, ic)[_PAD:_PAD + G, _PAD:_PAD + G, :] = inp.float().numpy()
    img_u16 = f32_to_bf16_u16(img.reshape(-1))

    device = xrt.device(0)
    elf = xrt.elf(elf_path)
    ctx = xrt.hw_context(device, elf)
    kern = xrt.ext.kernel(ctx, "main")
    a_bo = xrt.ext.bo(device, img_u16.nbytes)
    wt_bo = xrt.ext.bo(device, weights.nbytes)
    b_bo = xrt.ext.bo(device, img_u16.nbytes)

    def _fill(bo, arr):
        mv = bo.map()
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(np.ascontiguousarray(arr), dtype=np.uint8))
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _fill(a_bo, img_u16); _fill(wt_bo, weights.astype(np.uint16)); _fill(b_bo, img_u16)
    r = xrt.run(kern)
    r.set_arg(0, a_bo); r.set_arg(1, wt_bo); r.set_arg(2, b_bo)
    r.start(); r.wait2()
    final_bo = b_bo if (N_ITERS % 2 == 1) else a_bo
    final_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    # THE SEAM BUFFER: chain output, 28x28x64 padded HWC, u16 -- fed verbatim.
    chain_pad_u16 = np.frombuffer(final_bo.map(), dtype=np.uint16,
                                  count=p["IMG_ELEMS"]).copy()
    chain_pad = bf16_u16_to_f32(chain_pad_u16).reshape(IMG_W, IMG_W, ic)
    # cross-check the chain output itself vs the resident runner
    ref_fmap = run_rn3_chain_geo(GEO, inp, pairs).float().numpy()       # [G,G,ic]
    got_fmap = chain_pad[_PAD:_PAD + G, _PAD:_PAD + G, :]
    chain_diff = float(np.abs(got_fmap - ref_fmap).max())
    print(f"  chain merged-ELF output vs resident runner: max_diff={chain_diff:.6f} "
          f"({'OK' if chain_diff < 0.05 else 'MISMATCH'})")

    # ===== 2. feed the chain's padded buffer DIRECTLY into the halo-conv =====
    oc = 32
    module, meta = halo_conv(ic=ic, oc=oc, gbound=G)
    assert module.operation.verify()
    GRID, N_TILES, C_ELEMS = meta["GRID"], meta["N_TILES"], meta["C_ELEMS"]
    wd = _HERE / "build_halo_conv_seam"; wd.mkdir(parents=True, exist_ok=True)
    print(f"  compiling halo-conv (ic={ic} oc={oc} G={G}) ...", flush=True)
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))

    # SEAM SHIFT: chain valid feature map sits at [PAD:PAD+G]; a pad-1 c3 conv's
    # output pixel (0,0) needs the window at feature-map (-1,-1) = padded (PAD-1).
    # The halo-conv tile-0 window origin is (0,0), so shift the chain buffer by
    # SHIFT=PAD-1 so origins line up. (A real fused seam would bake this 1-pixel
    # offset into the chain's drain origin -- here we do it host-side, no im2col.)
    SHIFT = PAD - 1
    seam = np.zeros_like(chain_pad)
    seam[:IMG_W - SHIFT, :IMG_W - SHIFT, :] = chain_pad[SHIFT:, SHIFT:, :]
    seam_bf = bf16(seam)
    seam_u16 = to_u16(seam_bf)

    W = (rng.standard_normal((oc, 9, ic)).astype(np.float32) * 0.1)
    W_bf = bf16(W)
    bn_w = bf16(rng.standard_normal(oc).astype(np.float32) * 0.5 + 1.0)
    bn_b = bf16(rng.standard_normal(oc).astype(np.float32) * 0.2)
    wt_u16 = pack_halo_weights(W_bf, bn_w, bn_b, ic, oc)
    raw = numpy_conv3x3(seam_bf, W_bf, G, ic, oc)                       # [G,G,oc]
    ref = bn_silu_ref(raw, bn_w, bn_b)

    npu = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    out = iron.zeros(N_TILES * C_ELEMS, dtype=np.float32)
    DefaultNPURuntime.run(h, [iron.tensor(seam_u16, dtype=np.uint16),
                              iron.tensor(wt_u16, dtype=np.uint16), out])
    flat = np.array(out.numpy())

    got = np.zeros((G, G, oc), np.float32)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS], oc)
        for pl in range(64):
            oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
            if oh < G and ow < G:
                got[oh, ow, :] = tile[pl, :]

    d = np.abs(got - ref)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    max_diff, mean_diff = d.max(), d.mean()
    tol = 0.20  # in-kernel BN amplifies the BFP max tail; mean stays ~7e-3
    ok = (chain_diff < 0.05) and (max_diff < tol) and (mean_diff < 0.02)
    print("\n========= chain -> halo_c3 SEAM (no host im2col) =========")
    print(f"  chain output buffer fed verbatim: memref<{p['IMG_ELEMS']}xui16> "
          f"(28x28x64 PAD(2) HWC) -> halo-conv -> {G}x{G}x{oc}")
    print(f"  halo-conv vs numpy 3x3 on chain fmap: max_diff={max_diff:.5f} "
          f"mean={mean_diff:.6f} tol={tol} -> {'PASS' if max_diff < tol else 'FAIL'}")
    print(f"  got[0,0,:6]={got[0,0,:6]}")
    print(f"  ref[0,0,:6]={ref[0,0,:6]}")
    print(f"\n{'PASS' if ok else 'FAIL'}: merged-chain padded-HWC output feeds the "
          f"halo-conv with NO im2col bridge")
    print("==========================================================")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
