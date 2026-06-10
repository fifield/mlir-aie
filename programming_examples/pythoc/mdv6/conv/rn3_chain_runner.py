#!/usr/bin/env python3
"""Model-facing runner for the chained rn3-pair launch.

Runs N bottleneck iterations out_{i+1} = pair(x_i, w1_i, w2_i) + x_i of an
re6 RepNCSP stack (40x40x48) in ONE NPU launch via the DDR-bounced halo
chain (aie2_rn3_pair_vector_chain). Replaces N x run_re6_rn3_pair + host
residual adds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conv.aie2_rn3_pair_vector_chain import (
    rn3_pair_vector_chain, IMG, IMG_H, IC, PAD, WSLOT, N_WSLOTS, IMG_ELEMS,
)
from conv.rn3_pair_vector_memtile_runner import fused_weight_u16_to_parts
from conv.test_rn3_pair_vector_oneblock_hw import (
    pack_3x3_weights_u16, f32_to_bf16_u16, bf16_u16_to_f32,
)
from conv.resident_xclbin_runner import ResidentXCLBinRunner

_RUNNERS: dict = {}
_WEIGHT_CACHE: dict = {}
_OUT_ARG = None


def _build_dir(n_iters: int) -> Path:
    return Path(__file__).parent / f"build_rn3_chain_i{n_iters}_c5_s4_l0"


def _get_runner(n_iters: int) -> ResidentXCLBinRunner:
    r = _RUNNERS.get(n_iters)
    if r is None:
        bd = _build_dir(n_iters)
        xclbin, insts = bd / "final.xclbin", bd / "insts.bin"
        if not (xclbin.exists() and insts.exists()):
            from aie.utils.compile import compile_mlir_module
            (bd / "work").mkdir(parents=True, exist_ok=True)
            compile_mlir_module(
                mlir_module=rn3_pair_vector_chain(n_iters=n_iters, n_cols=5, stages=4, linear=0),
                insts_path=str(insts), xclbin_path=str(xclbin),
                work_dir=str(bd / "work"), verbose=False)
        r = ResidentXCLBinRunner(xclbin, insts)
        _RUNNERS[n_iters] = r
    return r


def _pack_iter_slots(w1_u16: np.ndarray, w2_u16: np.ndarray) -> np.ndarray:
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


def pack_chain_weights(weight_pairs) -> np.ndarray:
    """weight_pairs: [(w1_u16, w2_u16)] x n_iters; 6 slots per iter, streamed twice."""
    key = tuple(id(a) for p in weight_pairs for a in p)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        parts = []
        for w1, w2 in weight_pairs:
            s = _pack_iter_slots(w1, w2)
            parts += [s, s]
        cached = (np.concatenate(parts), list(weight_pairs))
        _WEIGHT_CACHE[key] = cached
    return cached[0]


def run_re6_rn3_chain(inp_hwc: torch.Tensor, weight_pairs, *, bo_key: str | None = None) -> torch.Tensor:
    """Run len(weight_pairs) chained pair+residual iterations on 40x40x48 HWC."""
    global _OUT_ARG
    n_iters = len(weight_pairs)
    weights = pack_chain_weights(weight_pairs)
    img = np.zeros((IMG_H, IMG, IC), np.float32)
    img[PAD:PAD+40, PAD:PAD+40, :] = inp_hwc.float().numpy()
    img_u16 = f32_to_bf16_u16(img.reshape(-1))
    if _OUT_ARG is None:
        _OUT_ARG = np.zeros(IMG_ELEMS, np.uint16)
    r = _get_runner(n_iters)
    # per-stack bo_key keyed by weight identity so the 1MB slot stream
    # uploads once per stack (static), only the image syncs per call
    key = bo_key or f"rn3chain_{n_iters}_{id(weights)}"
    res = r.run(img_u16, weights, _OUT_ARG, bo_key=key,
                output_indices={0}, inout_indices={0}, static_indices={1})
    out = bf16_u16_to_f32(res[0]).reshape(IMG_H, IMG, IC)[PAD:PAD+40, PAD:PAD+40, :]
    return torch.from_numpy(out).to(torch.bfloat16)


def last_stats(n_iters: int):
    r = _RUNNERS.get(n_iters)
    return None if r is None else r.last_stats


# ── geometry-generic (re4 80x80x32, re8 20x20x64) ────────────────────────


def _get_geo_runner(geo: str, n_iters: int) -> ResidentXCLBinRunner:
    key = (geo, n_iters)
    r = _RUNNERS.get(key)
    if r is None:
        bd = Path(__file__).parent / f"build_rn3_chain_{geo}_i{n_iters}"
        xclbin, insts = bd / "final.xclbin", bd / "insts.bin"
        if not (xclbin.exists() and insts.exists()):
            from aie.utils.compile import compile_mlir_module
            from conv.aie2_rn3_chain_geo import rn3_chain_geo
            (bd / "work").mkdir(parents=True, exist_ok=True)
            compile_mlir_module(mlir_module=rn3_chain_geo(geo, n_iters=n_iters),
                                insts_path=str(insts), xclbin_path=str(xclbin),
                                work_dir=str(bd / "work"), verbose=False)
        r = ResidentXCLBinRunner(xclbin, insts)
        _RUNNERS[key] = r
    return r


def _pack_geo_iter(w1_u16, w2_u16, ic, wslot, n_blk):
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


def run_rn3_chain_geo(geo: str, inp_hwc: torch.Tensor, weight_pairs) -> torch.Tensor:
    from conv.aie2_rn3_chain_geo import geo_params
    p = geo_params(geo)
    ic, G = p["IC"], p["GBOUND"]
    nt = p["WORKER_TILES"][0]
    n_iters = len(weight_pairs)
    key = (geo,) + tuple(id(a) for pr in weight_pairs for a in pr)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        blocks = [np.tile(_pack_geo_iter(w1, w2, ic, p["WSLOT"], p["N_BLK"]), nt)
                  for w1, w2 in weight_pairs]
        cached = (np.concatenate(blocks), list(weight_pairs))
        _WEIGHT_CACHE[key] = cached
    weights = cached[0]
    img = np.zeros((p["IMG_H"], p["IMG"], ic), np.float32)
    img[PAD:PAD+G, PAD:PAD+G, :] = inp_hwc.float().numpy()
    r = _get_geo_runner(geo, n_iters)
    res = r.run(f32_to_bf16_u16(img.reshape(-1)), weights, np.zeros(p["IMG_ELEMS"], np.uint16),
                bo_key=f"rn3geo_{geo}_{id(weights)}", output_indices={0, 2},
                inout_indices={0, 2}, static_indices={1})
    final = res[0] if n_iters % 2 == 0 else res[2]
    out = bf16_u16_to_f32(final).reshape(p["IMG_H"], p["IMG"], ic)[PAD:PAD+G, PAD:PAD+G, :]
    return torch.from_numpy(out).to(torch.bfloat16)
