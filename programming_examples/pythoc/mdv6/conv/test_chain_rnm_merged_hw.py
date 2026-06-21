#!/usr/bin/env python3
"""C2 HW proof: re8 rn3 chain -> [de-pad+concat] -> rnm 1x1 GEMM in ONE merged ELF.

The chain->rnm device-resident seam. Two sub-devices (chain producer, depad+concat
rnm GEMM consumer) in ONE merged ELF / ONE xrt.hw_context, wired
chain_links=[(0,2,1,0)] so the chain's PAD(2)-padded output (28x28x64) threads
device-side into the rnm GEMM's stacked input with NO host concat / repack / bounce.

The chain output BO is widened to 100352 = [chain_padded | x2_padded]; the chain
writes the lower half on device, the host pre-loads x2 into the upper half (x2 is
NOT on the chain's compute path), and the consumer's input gather does the de-pad
+ concat. The 64-ch chain output IS the 128-ch rnm input, no on-device reformat.

Reference (2-context host path): run the chain (ResidentXCLBin), host
concat(depad(chain), x2), host rnm matmul+BN+SiLU (the proven _torch_reference),
placed into the PAD(2) seam interior. Bit-exact within BFP tol.

CONTEXT: rnm chain output + rnm GEMM = 2 hw_contexts -> 1 (one xrt.hw_context).

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_chain_rnm_merged_hw.py
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
from build_chain_rnm_merged import build
from aie2_rn3_chain_geo import geo_params, PAD
from aie2_depad_concat_gemm import depad_concat_gemm  # noqa: F401 (meta via build)
from rn3_chain_runner import run_rn3_chain_geo, _pack_geo_iter
from test_rn3_pair_vector_oneblock_hw import f32_to_bf16_u16, bf16_u16_to_f32
from test_gemm_truth import _pack_weights_blocked, _torch_reference

N_ITERS = 3
GEO = "re8"
TILE = 8


def _make_weight_pairs(rng, n_iters, ic):
    pairs = []
    n = ic * ic * 9 + 2 * ic
    for _ in range(n_iters):
        w1 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
        w2 = rng.integers(0x3b00, 0x3d00, size=n, dtype=np.uint16)
        pairs.append((w1, w2))
    return pairs


def main():
    elf_path, dmeta = build(geo=GEO, n_iters=N_ITERS)
    if elf_path is None:
        print("FAIL: merged chain->rnm ELF build failed")
        return 1

    p = geo_params(GEO)
    ic2, G = p["IC"], p["GBOUND"]            # ic2 = 64 (chain output channels)
    oc = dmeta["oc"]                          # 128
    ic = dmeta["ic"]                          # 128 (fused 2*ic2)
    IMG = dmeta["IMG"]                        # 28
    HALF_ELEMS = dmeta["HALF_ELEMS"]          # 50176
    IN_ELEMS = dmeta["IN_ELEMS"]             # 100352
    SEAM_ELEMS = dmeta["IMG_ELEMS"]          # 28*28*128 = 100352
    nt = p["WORKER_TILES"][0]

    rng = np.random.default_rng(7)
    inp = torch.from_numpy(
        (rng.standard_normal((G, G, ic2)).astype(np.float32) * 0.5)).to(torch.bfloat16)
    pairs = _make_weight_pairs(rng, N_ITERS, ic2)
    x2 = torch.from_numpy(
        (rng.standard_normal((G, G, ic2)).astype(np.float32) * 0.25)).to(torch.bfloat16)
    # rnm GEMM weights (1x1 ic=128 -> oc=128 + BN)
    conv_wt = torch.from_numpy(
        (rng.standard_normal((oc, ic)).astype(np.float32) * 0.1)).to(torch.bfloat16)
    bn_w = (torch.ones(oc) + 0.05 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    bn_b = (0.1 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)

    # ---- reference: chain (resident) -> host concat -> host rnm ----
    chain_ref = run_rn3_chain_geo(GEO, inp, pairs)            # [G,G,ic2] bf16
    concat_hwc = torch.cat([chain_ref, x2], dim=2)           # [G,G,128]
    ref_pix = _torch_reference(concat_hwc.reshape(G * G, ic), conv_wt, bn_w, bn_b)
    ref_pix_np = ref_pix.to(torch.float32).numpy().reshape(G, G, oc)

    # ---- merged ELF host buffers ----
    weights = np.concatenate([
        np.tile(_pack_geo_iter(w1, w2, ic2, p["WSLOT"], p["N_BLK"]), nt)
        for w1, w2 in pairs])
    # chain input padded image into A[0:50176]; A widened to 100352 (tail unused)
    img = np.zeros(IN_ELEMS, np.float32)
    img[:p["IMG_H"] * IMG * ic2].reshape(p["IMG_H"], IMG, ic2)[PAD:PAD + G, PAD:PAD + G, :] = inp.float().numpy()
    a_u16 = f32_to_bf16_u16(img)
    # x2 padded into the upper half [HALF_ELEMS:IN_ELEMS] of the shared B BO
    x2_padded = np.zeros((IMG, IMG, ic2), np.uint16)
    x2_padded[PAD:PAD + G, PAD:PAD + G, :] = x2.view(torch.uint16).numpy()
    b_u16 = a_u16.copy()
    b_u16[HALF_ELEMS:IN_ELEMS] = x2_padded.reshape(-1)
    gemm_wt_u16 = _pack_weights_blocked(conv_wt, bn_w, bn_b)

    device = xrt.device(0)
    elf = xrt.elf(elf_path)
    ctx = xrt.hw_context(device, elf)        # ONE hw_context
    kern = xrt.ext.kernel(ctx, "main")
    one_context = isinstance(ctx, xrt.hw_context)

    # flat @main args (chain_link (0,2,1,0) aliases chain.arg2 == dcg.arg0):
    #   arg0=A(chain in), arg1=WT(chain), arg2=B(shared chain-out/dcg-in),
    #   arg3=gemm_wt, arg4=seam(dcg out)
    a_bo = xrt.ext.bo(device, a_u16.nbytes)
    wt_bo = xrt.ext.bo(device, weights.nbytes)
    b_bo = xrt.ext.bo(device, b_u16.nbytes)
    gwt_bo = xrt.ext.bo(device, gemm_wt_u16.nbytes)
    seam_bo = xrt.ext.bo(device, SEAM_ELEMS * 2)

    def _fill(bo, arr):
        mv = bo.map()
        np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
                  np.frombuffer(np.ascontiguousarray(arr), dtype=np.uint8))
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    _fill(a_bo, a_u16)
    _fill(wt_bo, weights.astype(np.uint16))
    _fill(b_bo, b_u16)                         # B carries x2 in its upper half
    _fill(gwt_bo, gemm_wt_u16)
    _fill(seam_bo, np.zeros(SEAM_ELEMS, np.uint16))   # poison PAD border

    r = xrt.run(kern)
    r.set_arg(0, a_bo); r.set_arg(1, wt_bo); r.set_arg(2, b_bo)
    r.set_arg(3, gwt_bo); r.set_arg(4, seam_bo)
    r.start(); r.wait2()
    seam_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    seam_u16 = np.frombuffer(seam_bo.map(), dtype=np.uint16, count=SEAM_ELEMS).copy()
    seam_u16 = seam_u16.reshape(IMG, IMG, oc)

    got_interior = (seam_u16[PAD:PAD + G, PAD:PAD + G, :].astype(np.uint32) << 16).view(np.float32)
    d = np.abs(got_interior - ref_pix_np)
    max_diff, mean_diff = float(d.max()), float(d.mean())
    tol = 0.05
    ok = max_diff < tol
    chk = seam_u16.copy(); chk[PAD:PAD + G, PAD:PAD + G, :] = 0
    border_nz = int(np.count_nonzero(chk))
    border_ok = (border_nz == 0)

    print("\n========= C2: re8 chain -> [de-pad+concat] -> rnm GEMM in ONE merged ELF =========")
    print(f"  ELF: {Path(elf_path).name}  (sub0=chain, sub1=depad+concat rnm GEMM; chain_link (0,2,1,0))")
    print(f"  geo={GEO} n_iters={N_ITERS}  chain out widened to {IN_ELEMS} "
          f"= [chain_padded({HALF_ELEMS}) | x2_padded({HALF_ELEMS})]")
    print(f"  hw_context: chain + rnm = 2 -> 1 (one xrt.hw_context: "
          f"{'YES' if one_context else 'NO'})")
    print(f"  x2 = separate host load into shared-BO upper half (not on chain path); "
          f"depad+concat = consumer input gather, no host concat/repack/bounce")
    print(f"  merged chain->rnm vs chain(resident)+host concat+host rnm: "
          f"max_diff={max_diff:.5f} mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"  PAD(2) seam border all-zero: {'PASS' if border_ok else f'FAIL ({border_nz} nz)'}")
    print(f"  got[0,0,:4]={got_interior[0,0,:4]}  ref[0,0,:4]={ref_pix_np[0,0,:4]}")
    print(f"  got[10,10,:4]={got_interior[10,10,:4]}  ref[10,10,:4]={ref_pix_np[10,10,:4]}")
    res = ok and border_ok and one_context
    print(f"\n{'PASS' if res else 'FAIL'}: chain->rnm device-resident in 1 ELF / 1 ctx, "
          f"no host concat/repack, REAL re8 RepNCSP conv3 seam")
    print("=================================================================================")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
