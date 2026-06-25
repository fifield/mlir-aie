#!/usr/bin/env python3
"""WB1 HW proof: rn1 (two 1x1 GEMMs) + chain + rnm GEMM + halo_c3, the WHOLE
RepNCSP `run_rn_mc->c3` hop INCLUDING the rn1 pair, device-resident in ONE merged
ELF / ONE xrt.hw_context.

Vs test_chain_rnm_halo_merged_hw.py (which is handed x1b/x2b host-padded), here
the rn1 pair runs ON-DEVICE as two gemm_pad_out sub-devices that drain PAD(2)-
padded HWC straight into the stacked chain BO halves:
  rn1a (conv1) -> chain A lower half (x1b = chain input)
  rn1b (conv2) -> chain B upper half (x2b = concat half, survives the ping-pong)

So the FULL fold is 5 sub-devices (rn1a, rn1b, chain, dcg, halo) in 1 hw_context.
The host only fills the rn1 input (inp) + weights; A/B borders are host-zeroed
ONCE (resident). This removes the model's gemm_pair(rn1) launch + the per-hop host
x1b/x2b pad/stack.

Reference (the host path the device replaces):
  rn1: x1b = SiLU(BN(conv1(inp))), x2b = SiLU(BN(conv2(inp)))  [1x1 GEMMs]
  then chain(x1b) -> concat(chain_out, x2b) -> rnm 1x1 -> c3 3x3 (same as the
  no-rn1 merged test). Bit-exact within BFP tol.

Run: source env.sh && CRH_GEO=re8 flock /tmp/npu-dev.lock python3 conv/test_chain_rnm_halo_rn1_hw.py
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
from test_gemm_truth import _pack_weights_blocked, _torch_reference
from test_halo_conv_hw import (bf16, to_u16, numpy_conv3x3, tile_b, untile_c,
                               bn_silu_ref)
from aie2_halo_conv import deinterleave_stream_out

N_ITERS = 3
GEO = os.environ.get("CRH_GEO", "re8")
TILE = 8
# (oc, gbound, tpc) + rn1 input channels (= proc; both hops in a geo share it).
_GEO_CFG = {"re8": dict(oc=128, gbound=20, tpc=1, rn1_ic=128),
            "re6": dict(oc=96, gbound=40, tpc=1, rn1_ic=96),
            "re4": dict(oc=64, gbound=80, tpc=4, rn1_ic=64)}


def _rn1_gemm_ref(inp_hwc, conv_w, bn_w, bn_b):
    """Host rn1: 1x1 GEMM (ic_in -> oc) + BN + SiLU -> [G,G,oc] (kernel-matching)."""
    G = inp_hwc.shape[0]
    ic_in = inp_hwc.shape[2]
    oc = conv_w.shape[0]
    out_pix = _torch_reference(inp_hwc.reshape(G * G, ic_in), conv_w, bn_w, bn_b)
    return out_pix.to(torch.bfloat16).reshape(G, G, oc)


def main():
    cfg = _GEO_CFG[GEO]
    tpc = cfg["tpc"]
    rn1_ic = cfg["rn1_ic"]
    p = geo_params(GEO)
    ic2, G = p["IC"], p["GBOUND"]
    nt = p["WORKER_TILES"][0]

    elf_path, dmeta, hmeta = build(geo=GEO, n_iters=N_ITERS, ic2=ic2,
                                   oc=cfg["oc"], gbound=cfg["gbound"], tpc=tpc,
                                   fold_rn1=True, rn1_ic=rn1_ic,
                                   build_dir=str(_HERE / f"build_wb1_{GEO}"))
    if elf_path is None:
        print("FAIL: WB1 merged rn1+chain+rnm+halo ELF build failed")
        return 1

    oc = dmeta["oc"]; ic = dmeta["ic"]
    IMG = dmeta["IMG"]; HALF_ELEMS = dmeta["HALF_ELEMS"]; IN_ELEMS = dmeta["IN_ELEMS"]
    SEAM_ELEMS = dmeta["IMG_ELEMS"]
    GRID, N_TILES, C_ELEMS = hmeta["GRID"], hmeta["N_TILES"], hmeta["C_ELEMS"]
    rmeta = dmeta["rn1"]
    n_cores_rn1 = rmeta["n_cores"]; in_tile = rmeta["full_in_tile"]
    rpc_rn1 = rmeta["rows_per_core"]

    rng = np.random.default_rng(7)
    # rn1 input (the c1 split half feeding repncsp.conv1/conv2)
    inp = torch.from_numpy(
        (rng.standard_normal((G, G, rn1_ic)).astype(np.float32) * 0.4)).to(torch.bfloat16)
    # rn1 conv1/conv2 (1x1 rn1_ic -> ic2) + BN
    rn1a_conv = torch.from_numpy((rng.standard_normal((ic2, rn1_ic)).astype(np.float32) * 0.12)).to(torch.bfloat16)
    rn1a_bnw = (torch.ones(ic2) + 0.05 * torch.from_numpy(rng.standard_normal(ic2).astype(np.float32))).to(torch.bfloat16)
    rn1a_bnb = (0.1 * torch.from_numpy(rng.standard_normal(ic2).astype(np.float32))).to(torch.bfloat16)
    rn1b_conv = torch.from_numpy((rng.standard_normal((ic2, rn1_ic)).astype(np.float32) * 0.12)).to(torch.bfloat16)
    rn1b_bnw = (torch.ones(ic2) + 0.05 * torch.from_numpy(rng.standard_normal(ic2).astype(np.float32))).to(torch.bfloat16)
    rn1b_bnb = (0.1 * torch.from_numpy(rng.standard_normal(ic2).astype(np.float32))).to(torch.bfloat16)

    # chain bottleneck weight pairs
    def _wp(rng, n_iters, ic):
        pairs = []
        n = ic * ic * 9 + 2 * ic
        for _ in range(n_iters):
            w1 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
            w2 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
            pairs.append((w1, w2))
        return pairs
    pairs = _wp(rng, N_ITERS, ic2)
    # rnm GEMM weights (1x1 2*ic2 -> oc)
    rnm_conv = torch.from_numpy((rng.standard_normal((oc, ic)).astype(np.float32) * 0.1)).to(torch.bfloat16)
    rnm_bnw = (torch.ones(oc) + 0.05 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    rnm_bnb = (0.1 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    # c3 3x3 weights
    c3_conv = (rng.standard_normal((oc, ic, 3, 3)).astype(np.float32) * 0.08)
    c3_bnw = (np.ones(oc, np.float32) + 0.05 * rng.standard_normal(oc).astype(np.float32))
    c3_bnb = (0.1 * rng.standard_normal(oc).astype(np.float32))

    # ===== reference: host rn1 -> chain -> concat -> rnm -> c3 =====
    x1b = _rn1_gemm_ref(inp, rn1a_conv, rn1a_bnw, rn1a_bnb)     # [G,G,ic2]
    x2b = _rn1_gemm_ref(inp, rn1b_conv, rn1b_bnw, rn1b_bnb)     # [G,G,ic2]
    chain_ref = run_rn3_chain_geo(GEO, x1b, pairs)             # [G,G,ic2]
    concat_hwc = torch.cat([chain_ref, x2b], dim=2)           # [G,G,2*ic2]
    rnm_pix = _torch_reference(concat_hwc.reshape(G * G, ic), rnm_conv, rnm_bnw, rnm_bnb)
    rnm_out = rnm_pix.to(torch.float32).numpy().reshape(G, G, oc)
    seam_ref = np.zeros((IMG, IMG, oc), np.float32)
    seam_ref[PAD:PAD + G, PAD:PAD + G, :] = rnm_out
    SHIFT = PAD - 1
    conv_img = np.zeros_like(seam_ref)
    conv_img[:IMG - SHIFT, :IMG - SHIFT, :] = seam_ref[SHIFT:, SHIFT:, :]
    W3 = c3_conv.reshape(oc, ic, 9).transpose(0, 2, 1)
    W3_bf = bf16(W3); c3_bnw_bf = bf16(c3_bnw); c3_bnb_bf = bf16(c3_bnb)
    raw = numpy_conv3x3(bf16(conv_img), W3_bf, G, oc, oc)
    ref = bn_silu_ref(raw, c3_bnw_bf, c3_bnb_bf).astype(np.float32)

    # ===== merged ELF host buffers =====
    # rn1 input: each core gets `rpc` consecutive valid rows [rpc*G, rn1_ic] HWC.
    inp_u16 = inp.view(torch.uint16).numpy()                  # [G,G,rn1_ic]
    rn1_in = np.zeros(n_cores_rn1 * in_tile, np.uint16)
    for c in range(n_cores_rn1):
        rows = inp_u16[c * rpc_rn1:(c + 1) * rpc_rn1]         # [rpc, G, rn1_ic]
        rn1_in[c * in_tile:(c + 1) * in_tile] = rows.reshape(-1)
    rn1a_wt = _pack_weights_blocked(rn1a_conv, rn1a_bnw, rn1a_bnb)
    rn1b_wt = _pack_weights_blocked(rn1b_conv, rn1b_bnw, rn1b_bnb)
    # A/B start zeroed (resident; on-device rn1 writes only the valid windows,
    # chain writes only the lower-half interior; PAD borders stay zero).
    a_u16 = np.zeros(IN_ELEMS, np.uint16)
    b_u16 = np.zeros(IN_ELEMS, np.uint16)
    weights = np.concatenate([
        np.tile(_pack_geo_iter(w1, w2, ic2, p["WSLOT"], p["N_BLK"]), nt)
        for w1, w2 in pairs])
    gemm_wt_u16 = _pack_weights_blocked(rnm_conv, rnm_bnw, rnm_bnb)
    from test_halo_conv_hw import pack_halo_weights
    halo_wt_u16 = pack_halo_weights(W3_bf, c3_bnw_bf, c3_bnb_bf, oc, oc, stream_oc="block")
    out_slots = hmeta.get("n_slots", N_TILES)
    out_elems = out_slots * C_ELEMS

    device = xrt.device(0)
    elf = xrt.elf(elf_path)
    ctx = xrt.hw_context(device, elf)
    kern = xrt.ext.kernel(ctx, "main")
    one_context = isinstance(ctx, xrt.hw_context)

    # flat @main args (5 subs; chain_links fold rn1 in + thread both seams):
    #   0=rn1_in 1=rn1a_wt 2=A(=rn1a.out=chain.A) 3=rn1b_wt 4=B(=rn1b.out=chain.B)
    #   5=chain_wt 6=gemm_wt 7=seam(=dcg.out=halo.in) 8=halo_wt 9=halo_out
    in_bo = xrt.ext.bo(device, rn1_in.nbytes)
    awt_bo = xrt.ext.bo(device, rn1a_wt.nbytes)
    a_bo = xrt.ext.bo(device, a_u16.nbytes)
    bwt_bo = xrt.ext.bo(device, rn1b_wt.nbytes)
    b_bo = xrt.ext.bo(device, b_u16.nbytes)
    wt_bo = xrt.ext.bo(device, weights.nbytes)
    gwt_bo = xrt.ext.bo(device, gemm_wt_u16.nbytes)
    seam_bo = xrt.ext.bo(device, SEAM_ELEMS * 2)
    hwt_bo = xrt.ext.bo(device, halo_wt_u16.nbytes)
    out_bo = xrt.ext.bo(device, out_elems * 4)

    def _fill(bo, arr):
        mv = bo.map()
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(np.ascontiguousarray(arr), dtype=np.uint8))
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _fill(in_bo, rn1_in)
    _fill(awt_bo, rn1a_wt.astype(np.uint16))
    _fill(a_bo, a_u16)
    _fill(bwt_bo, rn1b_wt.astype(np.uint16))
    _fill(b_bo, b_u16)
    _fill(wt_bo, weights.astype(np.uint16))
    _fill(gwt_bo, gemm_wt_u16)
    _fill(seam_bo, np.zeros(SEAM_ELEMS, np.uint16))
    _fill(hwt_bo, halo_wt_u16)

    r = xrt.run(kern)
    for i, bo in enumerate([in_bo, awt_bo, a_bo, bwt_bo, b_bo, wt_bo,
                            gwt_bo, seam_bo, hwt_bo, out_bo]):
        r.set_arg(i, bo)
    r.start(); r.wait2()
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    raw_out = np.frombuffer(out_bo.map(), dtype=np.float32, count=out_elems).copy()

    got = np.zeros((G, G, oc), np.float32)
    if tpc > 1:
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
    tol = 0.08
    ok = max_diff < tol and one_context

    print(f"\n===== WB1: rn1 + chain -> rnm GEMM -> halo_c3 (FULL hop incl rn1) in ONE merged ELF =====")
    print(f"  GEO={GEO}  ELF: {Path(elf_path).name}")
    print(f"  5 subs: rn1a, rn1b (on-device rn1 pair), chain, dcg(rnm GEMM), halo(c3 3x3)")
    print(f"  rn1: 1x1 {rn1_ic}->{ic2} x2 -> stacked chain BO halves (A lower / B upper)")
    print(f"  hw_context: rn1+chain+rnm+c3 -> 1 (one xrt.hw_context: {'YES' if one_context else 'NO'})")
    print(f"  merged FULL hop (incl rn1) vs host rn1+chain+rnm+im2col+3x3: "
          f"max_diff={max_diff:.5f} mean={mean_diff:.6f} tol={tol} -> {'PASS' if (max_diff < tol) else 'FAIL'}")
    print(f"  got[10,10,:4]={got[10,10,:4]}  ref[10,10,:4]={ref[10,10,:4]}")
    print(f"\n{'PASS' if ok else 'FAIL'}: ENTIRE RepNCSP rn_mc->c3 hop (incl rn1) device-resident in 1 ELF / 1 ctx")
    print("=" * 90)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
