#!/usr/bin/env python3
"""C3 HW proof: re8 chain -> rnm GEMM -> halo_c3 (the FULL x3rn->x3 hop) in ONE ELF.

The entire re8 RepNCSP bottleneck -> conv3 hop, device-resident in ONE merged ELF
/ ONE xrt.hw_context. THREE sub-devices threaded producer->consumer with NO host
touch between them (chain_links=[(0,2,1,0),(1,2,2,0)]):

  chain (PAD2 28x28x64) -> [de-pad+concat x2] -> rnm 1x1 128->128 (PAD2 28x28x128
  seam) -> halo_c3 3x3 128->128 (shift=PAD-1 baked).

Reference (3-context host path the device replaces):
  chain(resident) -> host concat(depad(chain), x2) -> host rnm matmul+BN+SiLU ->
  host PAD(2) pad -> host PAD-1 shift -> host 3x3 conv (c3 BN-scale folded into
  weights) -> host c3 BN-bias + SiLU. The halo kernel does RAW scale*conv; the
  BN-bias + SiLU epilogue is applied on the readback (same split as run_rnm_c3).
Bit-exact within BFP tol.

CONTEXT: chain + rnm GEMM + c3 3x3 = 3 hw_contexts -> 1 (one xrt.hw_context),
both seams device-resident.

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_chain_rnm_halo_merged_hw.py
"""
import os
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
from build_chain_rnm_halo_merged import build
from aie2_rn3_chain_geo import geo_params, PAD
from rn3_chain_runner import run_rn3_chain_geo, _pack_geo_iter
from test_rn3_pair_vector_oneblock_hw import f32_to_bf16_u16, bf16_u16_to_f32
from test_gemm_truth import _pack_weights_blocked, _torch_reference, _silu_kernel
from test_halo_conv_hw import (bf16, to_u16, numpy_conv3x3, tile_b, untile_c,
                               pack_halo_weights, bn_silu_ref)
from aie2_halo_conv import deinterleave_stream_out

N_ITERS = 3
GEO = os.environ.get("CRH_GEO", "re8")
TILE = 8
# per-geo (oc, gbound, tpc) registry — re8 (20x20x128) + re6 (40x40x96) one-tile
# halo (tpc=1); re4 (80x80x64) tiles-per-core multi-tile halo (tpc=4, 28 workers).
_GEO_CFG = {"re8": dict(oc=128, gbound=20, tpc=1),
            "re6": dict(oc=96, gbound=40, tpc=1),
            "re4": dict(oc=64, gbound=80, tpc=4)}


def _make_weight_pairs(rng, n_iters, ic):
    pairs = []
    n = ic * ic * 9 + 2 * ic
    for _ in range(n_iters):
        w1 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
        w2 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
        pairs.append((w1, w2))
    return pairs


def main():
    cfg = _GEO_CFG[GEO]
    tpc = cfg["tpc"]
    p0 = geo_params(GEO)
    elf_path, dmeta, hmeta = build(geo=GEO, n_iters=N_ITERS,
                                   ic2=p0["IC"], oc=cfg["oc"], gbound=cfg["gbound"],
                                   tpc=tpc)
    if elf_path is None:
        print("FAIL: merged chain->rnm->halo ELF build failed")
        return 1

    p = geo_params(GEO)
    ic2, G = p["IC"], p["GBOUND"]
    oc = dmeta["oc"]; ic = dmeta["ic"]
    IMG = dmeta["IMG"]; HALF_ELEMS = dmeta["HALF_ELEMS"]; IN_ELEMS = dmeta["IN_ELEMS"]
    SEAM_ELEMS = dmeta["IMG_ELEMS"]
    GRID, N_TILES, C_ELEMS = hmeta["GRID"], hmeta["N_TILES"], hmeta["C_ELEMS"]
    nt = p["WORKER_TILES"][0]

    rng = np.random.default_rng(11)
    inp = torch.from_numpy(
        (rng.standard_normal((G, G, ic2)).astype(np.float32) * 0.5)).to(torch.bfloat16)
    pairs = _make_weight_pairs(rng, N_ITERS, ic2)
    x2 = torch.from_numpy(
        (rng.standard_normal((G, G, ic2)).astype(np.float32) * 0.25)).to(torch.bfloat16)
    # rnm GEMM weights (1x1 128->128 + BN)
    rnm_conv = torch.from_numpy(
        (rng.standard_normal((oc, ic)).astype(np.float32) * 0.1)).to(torch.bfloat16)
    rnm_bnw = (torch.ones(oc) + 0.05 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    rnm_bnb = (0.1 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    # c3 3x3 weights (128->128 + BN). conv [oc, ic, 3, 3].
    c3_conv = (rng.standard_normal((oc, ic, 3, 3)).astype(np.float32) * 0.08)
    c3_bnw = (np.ones(oc, np.float32) + 0.05 * rng.standard_normal(oc).astype(np.float32))
    c3_bnb = (0.1 * rng.standard_normal(oc).astype(np.float32))

    # ===== reference: chain(resident) -> host concat -> host rnm -> host c3 =====
    chain_ref = run_rn3_chain_geo(GEO, inp, pairs)               # [G,G,ic2]
    concat_hwc = torch.cat([chain_ref, x2], dim=2)              # [G,G,128]
    rnm_pix = _torch_reference(concat_hwc.reshape(G * G, ic), rnm_conv, rnm_bnw, rnm_bnb)
    rnm_out = rnm_pix.to(torch.float32).numpy().reshape(G, G, oc)  # [G,G,oc] (rnm BN+SiLU)
    # PAD(2) pad + PAD-1 shift (the halo's baked shift)
    seam_ref = np.zeros((IMG, IMG, oc), np.float32)
    seam_ref[PAD:PAD + G, PAD:PAD + G, :] = rnm_out
    SHIFT = PAD - 1
    conv_img = np.zeros_like(seam_ref)
    conv_img[:IMG - SHIFT, :IMG - SHIFT, :] = seam_ref[SHIFT:, SHIFT:, :]
    # c3: in-kernel BN+SiLU on the f32 accumulator -> UN-scaled BFP weights.
    W3 = c3_conv.reshape(oc, ic, 9).transpose(0, 2, 1)          # [oc, 9, ic]
    W3_bf = bf16(W3)
    c3_bnw_bf = bf16(c3_bnw); c3_bnb_bf = bf16(c3_bnb)
    raw = numpy_conv3x3(bf16(conv_img), W3_bf, G, oc, oc)        # [G,G,oc] raw conv
    ref = bn_silu_ref(raw, c3_bnw_bf, c3_bnb_bf).astype(np.float32)  # [G,G,oc] final x3

    # ===== merged ELF host buffers =====
    weights = np.concatenate([
        np.tile(_pack_geo_iter(w1, w2, ic2, p["WSLOT"], p["N_BLK"]), nt)
        for w1, w2 in pairs])
    img = np.zeros(IN_ELEMS, np.float32)
    img[:p["IMG_H"] * IMG * ic2].reshape(p["IMG_H"], IMG, ic2)[PAD:PAD + G, PAD:PAD + G, :] = inp.float().numpy()
    a_u16 = f32_to_bf16_u16(img)
    # x2 half stacks above the chain image at HALF_ELEMS (== chain IMG_ELEMS,
    # tall for re6). Pad x2 into the SAME tall (IMG_H, IMG) layout so the dcg
    # de-pad gather (valid rows [PAD:PAD+G], row stride IMG) reads it correctly.
    x2_padded = np.zeros((p["IMG_H"], IMG, ic2), np.uint16)
    x2_padded[PAD:PAD + G, PAD:PAD + G, :] = x2.view(torch.uint16).numpy()
    b_u16 = a_u16.copy(); b_u16[HALF_ELEMS:IN_ELEMS] = x2_padded.reshape(-1)
    gemm_wt_u16 = _pack_weights_blocked(rnm_conv, rnm_bnw, rnm_bnb)
    halo_wt_u16 = pack_halo_weights(W3_bf, c3_bnw_bf, c3_bnb_bf, oc, oc, stream_oc="block")
    # mt halo (tpc>1) drains n_slots (>= N_TILES; junk slots included) raster slots.
    out_slots = hmeta.get("n_slots", N_TILES)
    out_elems = out_slots * C_ELEMS

    device = xrt.device(0)
    elf = xrt.elf(elf_path)
    ctx = xrt.hw_context(device, elf)
    kern = xrt.ext.kernel(ctx, "main")
    one_context = isinstance(ctx, xrt.hw_context)

    # flat @main args (links (0,2,1,0),(1,2,2,0)):
    #   0=A(chain in) 1=WT(chain) 2=B(chain out=dcg in) 3=gemm_wt
    #   4=seam(dcg out=halo in) 5=halo_wt 6=halo_out
    a_bo = xrt.ext.bo(device, a_u16.nbytes)
    wt_bo = xrt.ext.bo(device, weights.nbytes)
    b_bo = xrt.ext.bo(device, b_u16.nbytes)
    gwt_bo = xrt.ext.bo(device, gemm_wt_u16.nbytes)
    seam_bo = xrt.ext.bo(device, SEAM_ELEMS * 2)
    hwt_bo = xrt.ext.bo(device, halo_wt_u16.nbytes)
    out_bo = xrt.ext.bo(device, out_elems * 4)

    def _fill(bo, arr):
        mv = bo.map()
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(np.ascontiguousarray(arr), dtype=np.uint8))
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _fill(a_bo, a_u16)
    _fill(wt_bo, weights.astype(np.uint16))
    _fill(b_bo, b_u16)
    _fill(gwt_bo, gemm_wt_u16)
    _fill(seam_bo, np.zeros(SEAM_ELEMS, np.uint16))   # poison PAD border
    _fill(hwt_bo, halo_wt_u16)

    r = xrt.run(kern)
    r.set_arg(0, a_bo); r.set_arg(1, wt_bo); r.set_arg(2, b_bo)
    r.set_arg(3, gwt_bo); r.set_arg(4, seam_bo); r.set_arg(5, hwt_bo)
    r.set_arg(6, out_bo)
    r.start(); r.wait2()
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    raw_out = np.frombuffer(out_bo.map(), dtype=np.float32, count=out_elems).copy()

    # kernel applied BN + SiLU in-kernel: out IS the final activation.
    got = np.zeros((G, G, oc), np.float32)
    if tpc > 1:
        # mt halo: raster slots (slot == raster tile idx). De-interleave the
        # column-round-packed block-major OUT, then de-raster each real slot.
        from aie2_halo_conv_mt import deinterleave_stream_mt, slot_to_tile
        flat = deinterleave_stream_mt(raw_out, hmeta)
        for slot in range(out_slots):
            t = slot_to_tile(slot, hmeta)
            if t is None:
                continue
            tr, tc = t // GRID, t % GRID
            tile = untile_c(flat[slot * C_ELEMS:(slot + 1) * C_ELEMS], oc)
            for pl in range(64):
                oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
                if oh < G and ow < G:
                    got[oh, ow, :] = tile[pl, :]
    else:
        flat = deinterleave_stream_out(raw_out, hmeta)
        for t in range(N_TILES):
            tr, tc = t // GRID, t % GRID
            tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS], oc)
            for pl in range(64):
                oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
                if oh < G and ow < G:
                    got[oh, ow, :] = tile[pl, :]

    d = np.abs(got - ref)
    max_diff, mean_diff = float(d.max()), float(d.mean())
    tol = 0.07
    ok = max_diff < tol

    print("\n========= C3: re8 chain -> rnm GEMM -> halo_c3 (FULL hop) in ONE merged ELF =========")
    print(f"  ELF: {Path(elf_path).name}")
    print(f"     sub0=chain, sub1=depad+concat rnm GEMM, sub2=halo_c3 3x3; "
          f"chain_links=[(0,2,1,0),(1,2,2,0)]")
    print(f"  full hop: chain(PAD2 28x28x{ic2}) -> [depad+concat x2] -> rnm 1x1 {ic}->{oc} "
          f"(PAD2 28x28x{oc} seam) -> c3 3x3 {oc}->{oc} (shift={SHIFT} baked)")
    print(f"  hw_context: chain + rnm + c3 = 3 -> 1 (one xrt.hw_context: "
          f"{'YES' if one_context else 'NO'})")
    print(f"  both seams device-resident (chain->rnm and rnm->c3), no host concat/repack/"
          f"im2col/shift")
    print(f"  merged FULL hop vs chain(resident)+host rnm+im2col+3x3: "
          f"max_diff={max_diff:.5f} mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"  got[0,0,:4]={got[0,0,:4]}  ref[0,0,:4]={ref[0,0,:4]}")
    print(f"  got[10,10,:4]={got[10,10,:4]}  ref[10,10,:4]={ref[10,10,:4]}")
    res = ok and one_context
    print(f"\n{'PASS' if res else 'FAIL'}: ENTIRE re8 x3rn->x3 hop device-resident in 1 ELF / "
          f"1 ctx (3 ops -> 1 context)")
    print("====================================================================================")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
