#!/usr/bin/env python3
"""B2c3-1 HW proof: rn3 chain -> halo_c3 in ONE merged ELF / ONE hw_context.

Runs the build_re8_chain_halo_merged ELF: chain (padded-HWC producer) and the
halo-gather 3x3 conv (consumer) as two sub-devices, wired so the 50176-u16
PAD(2) HWC buffer threads through both device-side (chain.arg2 == halo.arg0)
with the pad-1 origin offset baked into the halo TAP (shift=PAD-1).

Bit-exact reference = the standalone seam (resident chain merged-ELF output ->
host SHIFT -> halo-conv standalone xclbin), i.e. exactly what
test_halo_conv_seam_hw.py validates -- but here the chain->halo handoff is
ON-DEVICE in 1 context instead of a 2-context host bounce.

CONTEXT: this test loads ONE xrt.hw_context (the merged ELF). The 2-context
baseline (separate chain ELF + halo xclbin) is what the seam test exercises.

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_re8_chain_halo_merged_hw.py
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

from build_re8_chain_halo_merged import build as build_merged_seam
from aie2_rn3_chain_geo import geo_params
from rn3_chain_runner import run_rn3_chain_geo, _pack_geo_iter, PAD as _PAD
from test_rn3_pair_vector_oneblock_hw import f32_to_bf16_u16, bf16_u16_to_f32
from test_re8_chain_merged_hw import _make_weight_pairs

from aie2_halo_conv import TILE, PAD
from test_halo_conv_hw import bf16, to_u16, numpy_conv3x3, tile_b, untile_c

N_ITERS = 3
GEO = "re8"
OC = 32


def main():
    p = geo_params(GEO)
    ic, G = p["IC"], p["GBOUND"]              # 64, 20
    IMG_W = p["IMG"]                          # 28
    assert p["IMG_ELEMS"] == IMG_W * IMG_W * ic == 50176

    elf_path, meta = build_merged_seam(GEO, n_iters=N_ITERS, oc=OC)
    if elf_path is None:
        print("FAIL: merged chain->halo ELF build failed"); return 1
    GRID, N_TILES, C_ELEMS = meta["GRID"], meta["N_TILES"], meta["C_ELEMS"]

    rng = np.random.default_rng(7)
    inp = torch.from_numpy(
        (rng.standard_normal((G, G, ic)).astype(np.float32) * 0.5)).to(torch.bfloat16)
    pairs = _make_weight_pairs(rng, N_ITERS, ic)
    nt = p["WORKER_TILES"][0]
    chain_w = np.concatenate([
        np.tile(_pack_geo_iter(w1, w2, ic, p["WSLOT"], p["N_BLK"]), nt)
        for w1, w2 in pairs])

    img = np.zeros(p["IMG_ELEMS"], np.float32)
    img.reshape(p["IMG_H"], IMG_W, ic)[_PAD:_PAD + G, _PAD:_PAD + G, :] = inp.float().numpy()
    img_u16 = f32_to_bf16_u16(img.reshape(-1))

    # halo weights (host-prepacked, same layout as standalone seam)
    W = (rng.standard_normal((OC, 9, ic)).astype(np.float32) * 0.1)
    W_bf = bf16(W)
    halo_wt_u16 = to_u16(bf16(tile_b(W_bf, ic, OC)))

    # ===== run the merged ELF (ONE hw_context) =====
    device = xrt.device(0)
    elf = xrt.elf(elf_path)
    ctx = xrt.hw_context(device, elf)
    kern = xrt.ext.kernel(ctx, "main")
    # @main args: chain_in, chain_wt, [seam], halo_wt, halo_out
    a_bo = xrt.ext.bo(device, img_u16.nbytes)            # chain_in (arg0)
    cwt_bo = xrt.ext.bo(device, chain_w.astype(np.uint16).nbytes)  # chain_wt (arg1)
    seam_bo = xrt.ext.bo(device, img_u16.nbytes)         # chain_out/halo_in (arg2)
    hwt_bo = xrt.ext.bo(device, halo_wt_u16.nbytes)      # halo_wt (arg3)
    out_elems = N_TILES * C_ELEMS
    out_bo = xrt.ext.bo(device, out_elems * 4)           # halo_out f32 (arg4)

    def _fill(bo, arr):
        mv = bo.map()
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(np.ascontiguousarray(arr), dtype=np.uint8))
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _fill(a_bo, img_u16)
    _fill(cwt_bo, chain_w.astype(np.uint16))
    _fill(seam_bo, img_u16)                # init the seam BO (chain overwrites it)
    _fill(hwt_bo, halo_wt_u16)

    r = xrt.run(kern)
    r.set_arg(0, a_bo); r.set_arg(1, cwt_bo); r.set_arg(2, seam_bo)
    r.set_arg(3, hwt_bo); r.set_arg(4, out_bo)
    r.start(); r.wait2()
    out_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    flat = np.frombuffer(out_bo.map(), dtype=np.float32, count=out_elems).copy()

    # ===== reference: the 2-context baseline (resident chain ELF output ->
    # host shift -> host-im2col numpy 3x3). This is EXACTLY what
    # test_halo_conv_seam_hw.py validates; B2c3-1 asserts the merged 1-context
    # seam reproduces it. We run the standalone chain merged-ELF to get the SAME
    # padded buffer the on-device seam threads (not the resident runner, which
    # differs from the ELF by ~0.023 BFP-quant and would inflate the delta).
    from build_re8_chain_merged import build as build_chain
    chain_elf, _ = build_chain(GEO, n_iters=N_ITERS), None
    cdev = xrt.device(0)
    celf = xrt.elf(chain_elf); cctx = xrt.hw_context(cdev, celf)
    ckern = xrt.ext.kernel(cctx, "main")
    ca = xrt.ext.bo(cdev, img_u16.nbytes); cw = xrt.ext.bo(cdev, chain_w.astype(np.uint16).nbytes)
    cb = xrt.ext.bo(cdev, img_u16.nbytes)
    _fill(ca, img_u16); _fill(cw, chain_w.astype(np.uint16)); _fill(cb, img_u16)
    cr = xrt.run(ckern); cr.set_arg(0, ca); cr.set_arg(1, cw); cr.set_arg(2, cb)
    cr.start(); cr.wait2()
    final = cb if (N_ITERS % 2 == 1) else ca
    final.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    chain_pad = bf16_u16_to_f32(
        np.frombuffer(final.map(), dtype=np.uint16, count=p["IMG_ELEMS"]).copy()
    ).reshape(IMG_W, IMG_W, ic)
    SHIFT = PAD - 1
    seam = np.zeros_like(chain_pad)
    seam[:IMG_W - SHIFT, :IMG_W - SHIFT, :] = chain_pad[SHIFT:, SHIFT:, :]
    ref = numpy_conv3x3(bf16(seam), W_bf, G, ic, OC)               # [G,G,oc]

    got = np.zeros((G, G, OC), np.float32)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS], OC)
        for pl in range(64):
            oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
            if oh < G and ow < G:
                got[oh, ow, :] = tile[pl, :]

    d = np.abs(got - ref)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    max_diff, mean_diff = float(d.max()), float(d.mean())
    tol = 0.06
    ok = max_diff < tol
    print("\n========= B2c3-1: chain -> halo_c3 in ONE merged ELF =========")
    print(f"  ELF: {Path(elf_path).name}  (sub0=chain, sub1=halo; chain_link (0,2,1,0))")
    print(f"  hw_context: chain+halo = 2 -> 1 (one xrt.hw_context loaded)")
    print(f"  seam buffer arg2 = memref<{p['IMG_ELEMS']}xui16> device-resident, "
          f"shift={SHIFT} baked into halo TAP (no host shift)")
    print(f"  merged halo_c3 vs numpy 3x3 on chain fmap: max_diff={max_diff:.5f} "
          f"mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"  got[0,0,:6]={got[0,0,:6]}")
    print(f"  ref[0,0,:6]={ref[0,0,:6]}")
    print(f"\n{'PASS' if ok else 'FAIL'}: rn3 chain -> halo_c3 device-resident in 1 ELF / 1 ctx, "
          f"no host im2col, no host shift")
    print("==============================================================")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
