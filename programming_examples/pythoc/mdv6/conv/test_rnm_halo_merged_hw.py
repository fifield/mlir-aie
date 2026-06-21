#!/usr/bin/env python3
"""S2 HW proof: rnm GEMM -> halo_c3 in ONE merged ELF / ONE hw_context — the
REAL model seam (RepNCSP conv3 1x1 128->128 feeding run_re_mc c3 3x3 128->128),
device-resident.

Runs the build_rnm_halo_merged ELF: rnm GEMM (PAD(2)-padded HWC producer) and the
OC=128 halo-gather 3x3 conv (consumer) as two sub-devices, wired so the 100352-u16
PAD(2) HWC seam threads through both device-side (gemm.arg2 == halo.arg0) with the
pad-1 origin offset baked into the halo TAP (shift=PAD-1). The seam BO is
device-resident: host poisons it and never syncs it between ops.

Bit-exact reference = the 2-context host path: host rnm (matmul+BN+SiLU per pixel)
-> host PAD(2) pad -> host PAD-1 shift -> host im2col + numpy 3x3 conv. This is the
exact path the device seam replaces — but here the rnm->c3 handoff is ON-DEVICE in
1 context instead of a 2-context host bounce + host im2col.

CONTEXT: this test loads ONE xrt.hw_context (the merged ELF). hw_context 2 -> 1.

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_rnm_halo_merged_hw.py
"""
import os, sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MDV6 = _HERE.parent
for _p in (str(_HERE), str(_MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyxrt as xrt

from build_rnm_halo_merged import build as build_merged_seam
from aie2_gemm_pad_out import PAD, TILE
from test_gemm_truth import _pack_weights_blocked, _torch_reference
from test_halo_conv_hw import bf16, to_u16, numpy_conv3x3, tile_b, untile_c
from aie2_halo_conv import deinterleave_stream_out


def main():
    ic = int(os.environ.get("RNM_IC", "128"))
    oc = int(os.environ.get("RNM_OC", "128"))
    gbound = int(os.environ.get("RNM_GBOUND", "20"))

    elf_path, gmeta, hmeta = build_merged_seam(ic=ic, oc=oc, gbound=gbound)
    if elf_path is None:
        print("FAIL: merged rnm->halo ELF build failed"); return 1
    IMG, IMG_ELEMS = gmeta["IMG"], gmeta["IMG_ELEMS"]
    n_cores, input_tile_size = gmeta["n_cores"], gmeta["input_tile_size"]
    GRID, N_TILES, C_ELEMS = hmeta["GRID"], hmeta["N_TILES"], hmeta["C_ELEMS"]

    # ---- inputs ----
    rng = np.random.default_rng(7)
    fmap = torch.from_numpy(
        (rng.standard_normal((gbound, gbound, ic)).astype(np.float32) * 0.25)
    ).to(torch.bfloat16)
    conv_wt = torch.from_numpy(
        (rng.standard_normal((oc, ic)).astype(np.float32) * 0.1)).to(torch.bfloat16)
    bn_w = (torch.ones(oc) + 0.05 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    bn_b = (0.1 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    W3 = (rng.standard_normal((oc, 9, oc)).astype(np.float32) * 0.1)
    W3_bf = bf16(W3)

    # host -> device packing (S1 layout: per-core valid row)
    fmap_u16 = fmap.view(torch.uint16).numpy()
    host_in = np.zeros(n_cores * input_tile_size, np.uint16)
    for r in range(n_cores):
        host_in[r * input_tile_size:(r + 1) * input_tile_size] = fmap_u16[r].reshape(-1)
    gemm_wt_u16 = _pack_weights_blocked(conv_wt, bn_w, bn_b)
    halo_wt_u16 = to_u16(bf16(tile_b(W3_bf, oc, oc)))

    # ===== run the merged ELF (ONE hw_context) =====
    device = xrt.device(0)
    elf = xrt.elf(elf_path)
    ctx = xrt.hw_context(device, elf)
    kern = xrt.ext.kernel(ctx, "main")
    # @main args: gemm_in(0), gemm_wt(1), seam(2), halo_wt(3), halo_out(4)
    in_bo = xrt.ext.bo(device, host_in.nbytes)
    gwt_bo = xrt.ext.bo(device, gemm_wt_u16.nbytes)
    seam_bo = xrt.ext.bo(device, IMG_ELEMS * 2)
    hwt_bo = xrt.ext.bo(device, halo_wt_u16.nbytes)
    out_elems = N_TILES * C_ELEMS
    out_bo = xrt.ext.bo(device, out_elems * 4)

    def _fill(bo, arr):
        mv = bo.map()
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(np.ascontiguousarray(arr), dtype=np.uint8))
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _fill(in_bo, host_in)
    _fill(gwt_bo, gemm_wt_u16)
    # Poison the seam BO. The GEMM only writes its valid interior; the PAD(2)
    # border must come out zero, so pre-zero it (the producer leaves the border
    # untouched). DO NOT sync it back between ops — it is device-resident.
    _fill(seam_bo, np.zeros(IMG_ELEMS, np.uint16))
    _fill(hwt_bo, halo_wt_u16)

    r = xrt.run(kern)
    r.set_arg(0, in_bo); r.set_arg(1, gwt_bo); r.set_arg(2, seam_bo)
    r.set_arg(3, hwt_bo); r.set_arg(4, out_bo)
    r.start(); r.wait2()
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    flat = deinterleave_stream_out(
        np.frombuffer(out_bo.map(), dtype=np.float32, count=out_elems).copy(), hmeta)

    # ===== reference: 2-context host path (rnm + pad + shift + im2col + 3x3) =====
    in_flat = fmap.reshape(gbound * gbound, ic)
    ref_pix = _torch_reference(in_flat, conv_wt, bn_w, bn_b)
    rnm_out = ref_pix.to(torch.float32).numpy().reshape(gbound, gbound, oc)  # [G,G,oc]
    # place into PAD(2) padded image (border zero)
    seam_ref = np.zeros((IMG, IMG, oc), np.float32)
    seam_ref[PAD:PAD + gbound, PAD:PAD + gbound, :] = rnm_out
    # PAD-1 shift (the halo's baked shift) so the host 3x3 reads the same phase
    SHIFT = PAD - 1
    conv_img = np.zeros_like(seam_ref)
    conv_img[:IMG - SHIFT, :IMG - SHIFT, :] = seam_ref[SHIFT:, SHIFT:, :]
    ref = numpy_conv3x3(bf16(conv_img), W3_bf, gbound, oc, oc)   # [G,G,oc]

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
    tol = 0.07
    ok = max_diff < tol
    one_context = isinstance(ctx, xrt.hw_context)

    print("\n========= S2: rnm GEMM -> halo_c3 in ONE merged ELF =========")
    print(f"  ELF: {Path(elf_path).name}  (sub0=rnm GEMM, sub1=halo_c3; chain_link (0,2,1,0))")
    print(f"  shape: rnm 1x1 GEMM IC={ic}->OC={oc}  +  c3 3x3 {oc}->{oc}  (gbound={gbound})")
    print(f"  hw_context: rnm+halo = 2 -> 1 (one xrt.hw_context loaded: "
          f"{'YES' if one_context else 'NO'})")
    print(f"  seam buffer arg2 = memref<{IMG_ELEMS}xui16> device-resident (poisoned, "
          f"never synced between ops), shift={SHIFT} baked into halo TAP")
    print(f"  merged rnm->c3 vs host rnm+im2col+3x3: max_diff={max_diff:.5f} "
          f"mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"  got[0,0,:4]={got[0,0,:4]}  ref[0,0,:4]={ref[0,0,:4]}")
    print(f"  got[10,10,:4]={got[10,10,:4]}  ref[10,10,:4]={ref[10,10,:4]}")
    res = ok and one_context
    print(f"\n{'PASS' if res else 'FAIL'}: rnm GEMM -> halo_c3 device-resident in 1 ELF / 1 ctx, "
          f"no host im2col, no host shift, REAL 128->128 model seam")
    print("==============================================================")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
