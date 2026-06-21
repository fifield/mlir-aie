#!/usr/bin/env python3
"""C1 HW proof: de-pad(chain) + channel-concat(x2) -> rnm 1x1 GEMM, device-resident.

Standalone proof of the chain->rnm seam's two on-device ops:
  1. DE-PAD: strided read of a PAD(2)-padded chain image (28x28x64) -> the valid
     20x20x64 interior, into channels [0:64] of the rnm input.
  2. CHANNEL-CONCAT: x2 (PAD(2)-padded 28x28x64) -> channels [64:128].
Both done in ONE rt.fill gather TAP per core (de-pad = PAD offset + IMG*ic2 row
stride; concat = HALF_ELEMS half stride). The fused [20,20,128] HWC feeds the
proven rnm GEMM (1x1 128->128 + BN + SiLU), which drains a PAD(2)-padded HWC seam.

Reference (host): concat(depad(chain), x2) on [20,20,128] -> per-pixel fused
matmul+BN+SiLU (the proven _torch_reference for gemm_conv1x1_fused_packed_bf16),
placed into the PAD(2) seam interior with zero border. Bit-exact within BFP tol.

The de-pad + concat compose: the device output's de-padded interior == the host
reference, AND the seam's PAD(2) border is exactly zero (== halo_c3 input).

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_depad_concat_gemm_hw.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MDV6 = HERE.parent
for _p in (str(HERE), str(MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime
from aie.utils.compile import compile_mlir_module

from aie2_depad_concat_gemm import depad_concat_gemm, PAD, TILE
from test_gemm_truth import _pack_weights_blocked, _torch_reference


def _pad_image(valid_hwc_u16: np.ndarray, IMG: int, gbound: int, ch: int) -> np.ndarray:
    """Place a valid [gbound,gbound,ch] u16 image into a PAD(2)-padded
    [IMG,IMG,ch] u16 image (interior at [PAD:PAD+gbound], border zero), flat."""
    padded = np.zeros((IMG, IMG, ch), np.uint16)
    padded[PAD:PAD + gbound, PAD:PAD + gbound, :] = valid_hwc_u16
    return padded.reshape(-1)


def main():
    ic2 = int(os.environ.get("DCG_IC2", "64"))     # per-source channels (chain/x2)
    oc = int(os.environ.get("DCG_OC", "128"))
    gbound = int(os.environ.get("DCG_GBOUND", "20"))
    ic = 2 * ic2                                    # fused input channels (128)

    module, meta = depad_concat_gemm(ic2=ic2, oc=oc, gbound=gbound)
    assert module.operation.verify()
    IMG, IMG_ELEMS, HALF_ELEMS = meta["IMG"], meta["IMG_ELEMS"], meta["HALF_ELEMS"]
    IN_ELEMS = meta["IN_ELEMS"]

    wd = HERE / "build_depad_concat_gemm"; wd.mkdir(parents=True, exist_ok=True)
    print(f"  compiling depad_concat_gemm (ic2={ic2} ic={ic} oc={oc} gbound={gbound} "
          f"IMG={IMG}x{IMG}) ...", flush=True)
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    print("  COMPILED", flush=True)

    # ---- inputs: chain bottleneck (valid 20x20x64) and x2 (valid 20x20x64) ----
    rng = np.random.default_rng(0)
    chain = torch.from_numpy(
        (rng.standard_normal((gbound, gbound, ic2)).astype(np.float32) * 0.25)
    ).to(torch.bfloat16)                                    # [G,G,ic2]
    x2 = torch.from_numpy(
        (rng.standard_normal((gbound, gbound, ic2)).astype(np.float32) * 0.25)
    ).to(torch.bfloat16)                                    # [G,G,ic2]
    # rnm GEMM weights (1x1 ic->oc + BN)
    conv_wt = torch.from_numpy(
        (rng.standard_normal((oc, ic)).astype(np.float32) * 0.1)).to(torch.bfloat16)
    bn_w = (torch.ones(oc) + 0.05 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    bn_b = (0.1 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)

    # ---- host reference: concat(depad(chain), x2) -> fused matmul+BN+SiLU ----
    concat_hwc = torch.cat([chain, x2], dim=2)              # [G,G,128]
    in_flat = concat_hwc.reshape(gbound * gbound, ic)
    ref_pix = _torch_reference(in_flat, conv_wt, bn_w, bn_b)  # torch bf16 [G*G, oc]
    ref_pix_np = ref_pix.to(torch.float32).numpy().reshape(gbound, gbound, oc)

    # ---- host -> device: STACKED [chain_padded(28x28x64) | x2_padded(28x28x64)] ----
    chain_u16 = chain.view(torch.uint16).numpy()           # [G,G,ic2]
    x2_u16 = x2.view(torch.uint16).numpy()
    chain_padded = _pad_image(chain_u16, IMG, gbound, ic2)  # [IMG*IMG*ic2]
    x2_padded = _pad_image(x2_u16, IMG, gbound, ic2)
    host_in = np.concatenate([chain_padded, x2_padded])    # [IN_ELEMS]
    assert host_in.size == IN_ELEMS
    wt_u16 = _pack_weights_blocked(conv_wt, bn_w, bn_b)     # [ic*oc + 2oc] u16

    # ---- run C1: depad+concat -> padded-HWC GEMM ----
    npu = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    seam = iron.zeros(IMG_ELEMS, dtype=np.uint16)          # host pre-zeros border
    DefaultNPURuntime.run(h, [iron.tensor(host_in, dtype=np.uint16),
                             iron.tensor(wt_u16, dtype=np.uint16), seam])
    seam_u16 = np.array(seam.numpy()).reshape(IMG, IMG, oc)

    # ---- check: de-pad device output vs host concat+matmul+BN+SiLU ----
    got_interior = (seam_u16[PAD:PAD + gbound, PAD:PAD + gbound, :].astype(np.uint32) << 16)\
        .view(np.float32)
    d = np.abs(got_interior - ref_pix_np)
    max_diff, mean_diff = float(d.max()), float(d.mean())
    tol = 0.03   # SiLU rational-approx + bf16 tail (same floor as gemm_pad_out)
    ok = max_diff < tol

    # border must be exactly zero (== halo_c3 input contract)
    chk = seam_u16.copy(); chk[PAD:PAD + gbound, PAD:PAD + gbound, :] = 0
    border_nz = int(np.count_nonzero(chk))
    border_ok = (border_nz == 0)

    print("\n========= C1: de-pad(chain) + concat(x2) -> rnm GEMM (device-resident) =========")
    print(f"  shape: chain[{gbound}x{gbound}x{ic2}] (PAD{PAD} 28x28) + x2[{gbound}x{gbound}x{ic2}] "
          f"(PAD{PAD} 28x28) -> concat[{gbound}x{gbound}x{ic}] -> rnm 1x1 {ic}->{oc} +BN+SiLU")
    print(f"  input gather TAP (de-pad+concat in ONE BD):")
    print(f"      offset=((PAD+r0)*IMG+PAD)*ic2  sizes=[cc,{gbound},2,{ic2}]  "
          f"strides=[IMG*ic2={IMG*ic2}, ic2={ic2}, HALF_ELEMS={HALF_ELEMS}, 1]")
    print(f"      (de-pad = PAD offset + IMG*ic2 row stride; concat = HALF_ELEMS half stride)")
    print(f"  drain TAP: offset=((PAD+r0)*IMG+PAD)*oc  strides=[0,IMG*oc={IMG*oc},*,1] "
          f"(one valid row per core, PAD(2) seam = halo_c3 input)")
    print(f"  de-pad device output vs host concat+matmul+BN+SiLU: "
          f"max_diff={max_diff:.5f} mean={mean_diff:.6f} tol={tol} -> {'PASS' if ok else 'FAIL'}")
    print(f"  PAD(2) seam border all-zero: {'PASS' if border_ok else f'FAIL ({border_nz} nz)'}")
    print(f"  got[0,0,:4]={got_interior[0,0,:4]}  ref[0,0,:4]={ref_pix_np[0,0,:4]}")
    print(f"  got[10,10,:4]={got_interior[10,10,:4]}  ref[10,10,:4]={ref_pix_np[10,10,:4]}")
    res = ok and border_ok
    print(f"\n{'PASS' if res else 'FAIL'}: C1 de-pad + concat compose device-resident, "
          f"bit-exact (BFP tol), seam = halo_c3 input")
    print("================================================================================")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
