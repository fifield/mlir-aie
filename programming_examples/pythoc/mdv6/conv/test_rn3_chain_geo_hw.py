#!/usr/bin/env python3
"""HW gate: geometry-generic rn3 chain (re4/re8) vs torch conv reference."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aie.utils.compile import compile_mlir_module
from conv.aie2_rn3_chain_geo import rn3_chain_geo, geo_params, PAD
from conv.rn3_pair_vector_memtile_runner import fused_weight_u16_to_parts
from conv.test_rn3_pair_vector_oneblock_hw import (
    pack_3x3_weights_u16, f32_to_bf16_u16, bf16_u16_to_f32,
)
from conv.resident_xclbin_runner import ResidentXCLBinRunner

GEO = os.environ.get("GEO", "re8")
N_ITERS = int(os.environ.get("N_ITERS", "3"))


def pack_iter(w1_u16, w2_u16, ic, wslot, n_blk):
    w1, b1w, b1b = fused_weight_u16_to_parts(w1_u16, ic, ic)
    w2, b2w, b2b = fused_weight_u16_to_parts(w2_u16, ic, ic)
    slots = np.zeros((2 * n_blk, wslot), np.uint16)
    for b in range(n_blk):
        s = pack_3x3_weights_u16(w1[b*16:(b+1)*16], b1w[b*16:(b+1)*16], b1b[b*16:(b+1)*16])
        slots[b, :s.size] = s
    for b in range(n_blk):
        s = pack_3x3_weights_u16(w2[b*16:(b+1)*16], b2w[b*16:(b+1)*16], b2b[b*16:(b+1)*16])
        slots[n_blk + b, :s.size] = s
    return slots.reshape(-1)


def torch_pair(x, w_u16, ic):
    w, bw, bb = fused_weight_u16_to_parts(w_u16, ic, ic)
    y = F.conv2d(x.permute(2, 0, 1)[None].float(), torch.from_numpy(w), padding=1)[0]
    y = y * torch.from_numpy(bw)[:, None, None] + torch.from_numpy(bb)[:, None, None]
    y = torch.nn.functional.silu(y)
    return y.permute(1, 2, 0).to(torch.bfloat16)


def main():
    p = geo_params(GEO)
    ic, wslot, n_blk = p["IC"], p["WSLOT"], p["N_BLK"]
    G, IMG, IMG_H, IMG_ELEMS = p["GBOUND"], p["IMG"], p["IMG_H"], p["IMG_ELEMS"]
    bd = Path(__file__).parent / f"build_rn3_chain_{GEO}_i{N_ITERS}"
    xclbin, insts = bd / "final.xclbin", bd / "insts.bin"
    if not (xclbin.exists() and insts.exists()):
        (bd / "work").mkdir(parents=True, exist_ok=True)
        compile_mlir_module(mlir_module=rn3_chain_geo(GEO, n_iters=N_ITERS),
                            insts_path=str(insts), xclbin_path=str(xclbin),
                            work_dir=str(bd / "work"), verbose=False)

    rng = np.random.default_rng(0)
    x0 = torch.from_numpy(rng.standard_normal((G, G, ic)).astype(np.float32) * 0.5).to(torch.bfloat16)
    mk = lambda: (rng.integers(0, 60, size=ic*ic*9 + 2*ic).astype(np.uint16) + 15000).astype(np.uint16)
    pairs = [(mk(), mk()) for _ in range(N_ITERS)]
    nt = p["WORKER_TILES"][0]
    # NT duplicate slot blocks per iter (one per tile pass; broadcast stream)
    weights = np.concatenate([np.tile(pack_iter(w1, w2, ic, wslot, n_blk), nt) for w1, w2 in pairs])

    img = np.zeros((IMG_H, IMG, ic), np.float32)
    img[PAD:PAD+G, PAD:PAD+G, :] = x0.float().numpy()
    img_u16 = f32_to_bf16_u16(img.reshape(-1))

    r = ResidentXCLBinRunner(xclbin, insts)
    res = r.run(img_u16, weights, np.zeros(IMG_ELEMS, np.uint16),
                bo_key=f"geo_{GEO}", output_indices={0, 2}, inout_indices={0, 2})
    final_bo = res[0] if N_ITERS % 2 == 0 else res[2]
    out = bf16_u16_to_f32(final_bo).reshape(IMG_H, IMG, ic)[PAD:PAD+G, PAD:PAD+G, :]

    ref = x0
    for w1, w2 in pairs:
        ref = (torch_pair(torch_pair(ref, w1, ic), w2, ic).float() + ref.float()).to(torch.bfloat16)
    d = np.abs(out - ref.float().numpy())
    print(f"out mean|x|={np.abs(out).mean():.4f} ref mean|x|={ref.float().abs().mean():.4f}")
    print(f"chain_{GEO}({N_ITERS}): max={d.max():.6f} mean={d.mean():.6f}")
    print("PASS" if d.max() < 0.05 else "FAIL")


if __name__ == "__main__":
    main()
