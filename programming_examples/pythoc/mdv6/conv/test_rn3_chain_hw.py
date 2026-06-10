#!/usr/bin/env python3
"""Standalone HW test: 2 chained rn3-pair iterations in one launch vs runner 2x."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aie.utils.compile import compile_mlir_module
from conv.aie2_rn3_pair_vector_chain import (
    rn3_pair_vector_chain, IMG, IMG_H, IC, PAD, GRID, TILE, MASK, WSLOT, N_WSLOTS,
    IMG_ELEMS, TILES_PER_COL,
)
from conv.rn3_pair_vector_memtile_runner import (
    fused_weight_u16_to_parts, run_re6_rn3_pair,
)
from conv.test_rn3_pair_vector_oneblock_hw import pack_3x3_weights_u16, f32_to_bf16_u16, bf16_u16_to_f32
from conv.resident_xclbin_runner import ResidentXCLBinRunner

import os
N_ITERS = int(os.environ.get('N_ITERS', '2'))
N_COLS = int(os.environ.get('N_COLS', '5'))
STAGES = int(os.environ.get('STAGES', '4'))
LINEAR = int(os.environ.get('LINEAR', '0'))
BUILD = Path(__file__).parent / f"build_rn3_chain_i{N_ITERS}_c{N_COLS}_s{STAGES}_l{int(LINEAR)}"


def pack_slots(w1_u16, w2_u16):
    w1, b1w, b1b = fused_weight_u16_to_parts(w1_u16, 48, 48)
    w2, b2w, b2b = fused_weight_u16_to_parts(w2_u16, 48, 48)
    slots = np.zeros((N_WSLOTS, WSLOT), np.uint16)
    for mb in range(3):
        s = pack_3x3_weights_u16(w1[mb*16:(mb+1)*16], b1w[mb*16:(mb+1)*16], b1b[mb*16:(mb+1)*16])
        slots[mb, :s.size] = s
    for ob in range(3):
        s = pack_3x3_weights_u16(w2[ob*16:(ob+1)*16], b2w[ob*16:(ob+1)*16], b2b[ob*16:(ob+1)*16])
        slots[3+ob, :s.size] = s
    return slots.reshape(-1)


def main():
    (BUILD / "work").mkdir(parents=True, exist_ok=True)
    xclbin, insts = BUILD / "final.xclbin", BUILD / "insts.bin"
    if not (xclbin.exists() and insts.exists()):
        compile_mlir_module(mlir_module=rn3_pair_vector_chain(n_iters=N_ITERS, n_cols=N_COLS, stages=STAGES, linear=LINEAR),
                            insts_path=str(insts), xclbin_path=str(xclbin),
                            work_dir=str(BUILD / "work"), verbose=False)

    rng = np.random.default_rng(0)
    x0 = torch.from_numpy(rng.standard_normal((40, 40, IC)).astype(np.float32) * 0.5).to(torch.bfloat16)
    import os as _os
    if _os.environ.get('BIGW', '0') == '1':
        mk = lambda: np.full(48*48*9 + 96, 0x3F80, np.uint16)  # all 1.0 bf16
    else:
        mk = lambda: (rng.integers(0, 60, size=48*48*9 + 96).astype(np.uint16) + 15000).astype(np.uint16)
    wA, wB, wC, wD = mk(), mk(), mk(), mk()

    img = np.zeros((IMG_H, IMG, IC), np.float32)
    img[PAD:PAD+40, PAD:PAD+40, :] = x0.float().numpy()
    img_u16 = f32_to_bf16_u16(img.reshape(-1))
    s1, s2 = pack_slots(wA, wB), pack_slots(wC, wD)
    weights = np.concatenate(([s1, s1, s2, s2])[: 2 * N_ITERS])

    runner = ResidentXCLBinRunner(xclbin, insts)
    res = runner.run(img_u16.copy(), weights, np.zeros(IMG_ELEMS, np.uint16),
                     bo_key="chain", output_indices={0}, inout_indices={0})
    out = bf16_u16_to_f32(res[0]).reshape(IMG_H, IMG, IC)[PAD:PAD+40, PAD:PAD+40, :]

    ref = x0
    for wp in ((wA, wB), (wC, wD))[:N_ITERS]:
        ref = (run_re6_rn3_pair(ref, wp[0], wp[1]).float() + ref.float()).to(torch.bfloat16)
    d = np.abs(out - ref.float().numpy())
    print(f"out: mean|x|={np.abs(out).mean():.4f} nonzero={np.count_nonzero(out)}/{out.size}")
    print(f"ref: mean|x|={ref.float().abs().mean():.4f}")
    print(f"chain({N_ITERS}) vs runner: max={d.max():.6f} mean={d.mean():.6f}")
    print("PASS" if d.max() < 0.05 else "FAIL")


if __name__ == "__main__":
    main()
