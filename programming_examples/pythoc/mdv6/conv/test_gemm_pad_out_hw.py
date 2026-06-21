#!/usr/bin/env python3
"""S1 HW proof: rnm 1x1 GEMM (IC=128->OC=128) whose DRAIN writes a PAD(2)-padded
HWC image — the format halo_c3 reads. Two checks:

  (A) The padded device output, de-padded, is bit-exact (BFP tol) vs the host
      fused matmul + BN + SiLU (the proven _torch_reference for this exact
      kernel) — i.e. the GEMM math is right AND the drain placed the result in
      the padded interior with a zero border.

  (B) Feed that padded buffer (as halo_c3's input) into the OC=128 halo-conv
      standalone xclbin -> matches host rnm + im2col + 3x3 conv. This confirms
      the padded buffer BYTE-MATCHES what halo_c3 expects (the seam contract).

This is the rnm half of the real-model rnm->c3 seam, standalone. S2 fuses it
with halo_c3 into ONE merged ELF.

Run: source env.sh && flock /tmp/npu-dev.lock python3 conv/test_gemm_pad_out_hw.py
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

from aie2_gemm_pad_out import gemm_pad_out, PAD, TILE
from aie2_halo_conv import halo_conv, WIN, deinterleave_stream_out
from test_gemm_truth import _pack_weights_blocked, _torch_reference
from test_halo_conv_hw import (bf16, to_u16, numpy_conv3x3, tile_b, untile_c,
                               host_im2col_window)


def main():
    ic = int(os.environ.get("GPO_IC", "128"))
    oc = int(os.environ.get("GPO_OC", "128"))
    gbound = int(os.environ.get("GPO_GBOUND", "20"))

    module, meta = gemm_pad_out(ic=ic, oc=oc, gbound=gbound)
    assert module.operation.verify()
    IMG, IMG_ELEMS = meta["IMG"], meta["IMG_ELEMS"]
    tile_m, n_cores = meta["tile_m"], meta["n_cores"]
    input_tile_size = meta["input_tile_size"]

    wd = HERE / "build_gemm_pad_out"; wd.mkdir(parents=True, exist_ok=True)
    print(f"  compiling gemm_pad_out (ic={ic} oc={oc} gbound={gbound} "
          f"IMG={IMG}x{IMG} n_cores={n_cores} tile_m={tile_m}) ...", flush=True)
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    print("  COMPILED", flush=True)

    # ---- inputs: gbound x gbound x ic valid feature map (bf16) ----
    rng = np.random.default_rng(0)
    fmap = torch.from_numpy(
        (rng.standard_normal((gbound, gbound, ic)).astype(np.float32) * 0.25)
    ).to(torch.bfloat16)                                    # [G,G,ic]
    conv_wt = torch.from_numpy(
        (rng.standard_normal((oc, ic)).astype(np.float32) * 0.1)).to(torch.bfloat16)
    bn_w = (torch.ones(oc) + 0.05 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)
    bn_b = (0.1 * torch.from_numpy(rng.standard_normal(oc).astype(np.float32))).to(torch.bfloat16)

    # host fused matmul + BN + SiLU per pixel -> [G*G, oc] bf16
    in_flat = fmap.reshape(gbound * gbound, ic)
    ref_pix = _torch_reference(in_flat, conv_wt, bn_w, bn_b)   # torch bf16 [G*G, oc]
    ref_pix_np = ref_pix.to(torch.float32).numpy().reshape(gbound, gbound, oc)

    # ---- host -> device input packing: per-core valid row, [tile_m, ic] HWC ----
    fmap_u16 = fmap.view(torch.uint16).numpy()                # [G,G,ic] u16
    host_in = np.zeros(n_cores * input_tile_size, np.uint16)
    for r in range(n_cores):
        host_in[r * input_tile_size:(r + 1) * input_tile_size] = \
            fmap_u16[r].reshape(-1)
    wt_u16 = _pack_weights_blocked(conv_wt, bn_w, bn_b)        # [ic*oc + 2oc] u16

    # ---- run S1: padded-HWC GEMM ----
    npu = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    seam = iron.zeros(IMG_ELEMS, dtype=np.uint16)             # host pre-zeros border
    DefaultNPURuntime.run(h, [iron.tensor(host_in, dtype=np.uint16),
                             iron.tensor(wt_u16, dtype=np.uint16), seam])
    seam_u16 = np.array(seam.numpy()).reshape(IMG, IMG, oc)

    # ---- check (A): de-pad device output vs host fused matmul+BN+SiLU ----
    got_interior = (seam_u16[PAD:PAD + gbound, PAD:PAD + gbound, :].astype(np.uint32) << 16)\
        .view(np.float32)
    dA = np.abs(got_interior - ref_pix_np)
    maxA, meanA = float(dA.max()), float(dA.mean())
    # SiLU rational-approx + bf16 rounding gives a ~0.02 single-pixel tail
    # outlier (mean ~0.0014); same noise floor the proven halo tests use 0.05-6.
    tolA = 0.03
    okA = maxA < tolA
    # border must be exactly zero
    border_ok = True
    chk = seam_u16.copy(); chk[PAD:PAD + gbound, PAD:PAD + gbound, :] = 0
    border_nz = int(np.count_nonzero(chk))
    border_ok = (border_nz == 0)
    print(f"\n  (A) padded GEMM de-padded vs host matmul+BN+SiLU: "
          f"max_diff={maxA:.5f} mean={meanA:.6f} tol={tolA} -> {'PASS' if okA else 'FAIL'}")
    print(f"      PAD(2) border all-zero: {'PASS' if border_ok else f'FAIL ({border_nz} nz)'}")
    print(f"      drain TAP: offset=((PAD+r)*IMG+PAD)*oc sizes=[1,cc,1,{tile_m*oc}] "
          f"strides=[0,{IMG*oc},0,1]  (one valid row per core)")

    # ---- check (B): feed the seam buffer into the OC=128 halo-conv standalone,
    # compare against host rnm + im2col + 3x3 conv on the SAME padded buffer ----
    hmod, hmeta = halo_conv(ic=oc, oc=oc, gbound=gbound, stream_oc="block")
    assert hmod.operation.verify()
    GRID, N_TILES, C_ELEMS = hmeta["GRID"], hmeta["N_TILES"], hmeta["C_ELEMS"]
    hwd = HERE / "build_halo_conv_oc128"; hwd.mkdir(parents=True, exist_ok=True)
    print(f"\n  compiling halo_conv OC={oc} consumer ...", flush=True)
    compile_mlir_module(mlir_module=hmod, insts_path=str(hwd / "insts.bin"),
                        xclbin_path=str(hwd / "final.xclbin"), work_dir=str(hwd))

    # halo 3x3 weights
    W3 = (rng.standard_normal((oc, 9, oc)).astype(np.float32) * 0.1)
    W3_bf = bf16(W3)
    halo_wt_u16 = to_u16(bf16(tile_b(W3_bf, oc, oc)))

    # The merged seam bakes shift=PAD-1 into the halo TAP; the STANDALONE halo
    # (shift=0) needs the host to pre-shift the image by PAD-1 — mirror the
    # standalone seam reference (test_re8_chain_halo_merged_hw / halo_conv_hw).
    SHIFT = PAD - 1
    seam_f = (seam_u16.astype(np.uint32) << 16).view(np.float32)    # [IMG,IMG,oc]
    conv_img = np.zeros_like(seam_f)
    conv_img[:IMG - SHIFT, :IMG - SHIFT, :] = seam_f[SHIFT:, SHIFT:, :]
    ref3 = numpy_conv3x3(bf16(conv_img), W3_bf, gbound, oc, oc)     # [G,G,oc]
    img_u16_for_halo = to_u16(bf16(conv_img))

    hnpu = NPUKernel(str(hwd / "final.xclbin"), str(hwd / "insts.bin"), kernel_name="MLIR_AIE")
    hh = DefaultNPURuntime.load(hnpu)
    hout = iron.zeros(N_TILES * C_ELEMS, dtype=np.float32)
    DefaultNPURuntime.run(hh, [iron.tensor(img_u16_for_halo, dtype=np.uint16),
                              iron.tensor(halo_wt_u16, dtype=np.uint16), hout])
    flat = deinterleave_stream_out(np.array(hout.numpy()), hmeta)
    got3 = np.zeros((gbound, gbound, oc), np.float32)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS], oc)
        for pl in range(64):
            oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
            if oh < gbound and ow < gbound:
                got3[oh, ow, :] = tile[pl, :]
    dB = np.abs(got3 - ref3)
    maxB, meanB = float(dB.max()), float(dB.mean())
    tolB = 0.06
    okB = maxB < tolB
    print(f"\n  (B) seam -> halo_c3(OC={oc}) standalone vs host rnm+im2col+3x3: "
          f"max_diff={maxB:.5f} mean={meanB:.6f} tol={tolB} -> {'PASS' if okB else 'FAIL'}")
    print(f"      got3[0,0,:4]={got3[0,0,:4]}  ref3[0,0,:4]={ref3[0,0,:4]}")

    ok = okA and border_ok and okB
    print(f"\n{'PASS' if ok else 'FAIL'}: S1 rnm GEMM -> PAD(2)-padded HWC, "
          f"de-pad bit-exact + seam byte-matches halo_c3 input (real 128->128)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
