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


def _get_geo_runner(geo: str, n_iters: int, rnm: int = 0) -> ResidentXCLBinRunner:
    key = (geo, n_iters, rnm)
    r = _RUNNERS.get(key)
    if r is None:
        # "chain2" = 5-arg ABI (img, wt, img2, x2, rnm_out)
        bd = Path(__file__).parent / f"build_rn3_chain2_{geo}_i{n_iters}_r{rnm}"
        xclbin, insts = bd / "final.xclbin", bd / "insts.bin"
        if not (xclbin.exists() and insts.exists()):
            from aie.utils.compile import compile_mlir_module
            from conv.aie2_rn3_chain_geo import rn3_chain_geo
            (bd / "work").mkdir(parents=True, exist_ok=True)
            compile_mlir_module(mlir_module=rn3_chain_geo(geo, n_iters=n_iters, rnm=rnm),
                                insts_path=str(insts), xclbin_path=str(xclbin),
                                work_dir=str(bd / "work"), verbose=False)
        r = ResidentXCLBinRunner(xclbin, insts)
        _RUNNERS[key] = r
    return r


def _pack_rnm_slots(rnm_w_u16: np.ndarray, ic: int, wslot: int, nt: int) -> np.ndarray:
    """Pack rnm 1x1 weights (oc=2*ic over concat(cur, x2)) into epilogue slots.

    Layout per 16-oc slot: [2*(ic/8)][8ic][8oc] mmul blocks for oc 0-7, then
    8-15, + bn_w(16) + bn_b(16), zero-padded to wslot. Slot order: oc-pass pe
    then slot, each duplicated nt times (one per worker tile acquire).
    """
    oc = 2 * ic
    w = bf16_u16_to_f32(rnm_w_u16[:oc * oc]).reshape(oc, oc)
    bnw = rnm_w_u16[oc * oc:oc * oc + oc]
    bnb = rnm_w_u16[oc * oc + oc:oc * oc + 2 * oc]
    kkmax = oc // 8
    parts = []
    for pe in range(2):
        group = []
        for s in range(ic // 16):
            oc0 = pe * ic + s * 16
            slot = np.zeros(wslot, np.uint16)
            for half in range(2):
                base = half * kkmax * 64
                blk = w[oc0 + half * 8:oc0 + half * 8 + 8]  # [8oc, oc_in]
                for kk in range(kkmax):
                    sub = blk[:, kk * 8:(kk + 1) * 8].T  # [8ic][8oc]
                    slot[base + kk * 64:base + (kk + 1) * 64] = f32_to_bf16_u16(
                        np.ascontiguousarray(sub).reshape(-1))
            slot[2 * oc * 8:2 * oc * 8 + 16] = bnw[oc0:oc0 + 16]
            slot[2 * oc * 8 + 16:2 * oc * 8 + 32] = bnb[oc0:oc0 + 16]
            group.append(slot)
        # core acquires t -> slot, so tile the whole slot group NT times
        parts.append(np.tile(np.concatenate(group), nt))
    return np.concatenate(parts)


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


def run_rn3_chain_raster(geo: str, inp_hwc: torch.Tensor, weight_pairs) -> torch.Tensor:
    """Raster chain: 1 tile/core over COLS*NWORK cores (re6w = 25 of 28)."""
    from conv.aie2_rn3_chain_geo import raster_params
    p = raster_params(geo)
    ic, G = p["IC"], p["GBOUND"]
    n_iters = len(weight_pairs)
    import os as _os
    wr = _os.environ.get("MDV6_WTREPLAY", "0") not in ("", "0", "false", "False")
    wr2 = _os.environ.get("MDV6_WTREPLAY2", "1") not in ("", "0", "false", "False")
    key = (geo, "raster", wr, wr2) + tuple(id(a) for pr in weight_pairs for a in pr)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        # wt replay: memtile re-emits per round — no host TPR duplication
        rep = 1 if (wr or wr2) else p["TPR"]
        blocks = [np.tile(_pack_geo_iter(w1, w2, ic, p["WSLOT"], p["N_BLK"]), rep)
                  for w1, w2 in weight_pairs]
        cached = (np.concatenate(blocks), list(weight_pairs))
        _WEIGHT_CACHE[key] = cached
    weights = cached[0]
    c1b = _os.environ.get("MDV6_CONV1_BFP", "1") not in ("", "0", "false", "False")
    c2b = _os.environ.get("MDV6_CONV2_BFP", "1") not in ("", "0", "false", "False")
    btag = ("_c1b" if c1b else "") + ("_c2b" if c2b else "")
    rkey = (geo, n_iters, "raster", wr, wr2, c1b, c2b)
    r = _RUNNERS.get(rkey)
    if r is None:
        tag = ("_wr2" if wr2 else ("_wr" if wr else "")) + btag
        bd = Path(__file__).parent / f"build_rn3_raster_{geo}_i{n_iters}{tag}"
        xclbin, insts = bd / "final.xclbin", bd / "insts.bin"
        if not (xclbin.exists() and insts.exists()):
            from aie.utils.compile import compile_mlir_module
            from conv.aie2_rn3_chain_geo import rn3_chain_raster, rn3_chain_raster_wr
            if wr2:
                def gen(geo, n_iters): return rn3_chain_raster(geo, n_iters=n_iters, wr2=1)
            else:
                gen = rn3_chain_raster_wr if wr else rn3_chain_raster
            (bd / "work").mkdir(parents=True, exist_ok=True)
            compile_mlir_module(mlir_module=gen(geo, n_iters=n_iters),
                                insts_path=str(insts), xclbin_path=str(xclbin),
                                work_dir=str(bd / "work"), verbose=False)
        r = ResidentXCLBinRunner(xclbin, insts)
        _RUNNERS[rkey] = r
    img = np.zeros((p["IMG_H"], p["IMG"], ic), np.float32)
    img[PAD:PAD+G, PAD:PAD+G, :] = inp_hwc.float().numpy()
    img_u16 = f32_to_bf16_u16(img.reshape(-1))
    res = r.run(img_u16, weights, img_u16.copy(),
                bo_key=f"rn3raster_{geo}_{id(weights)}", output_indices={0, 2},
                inout_indices={0, 2}, static_indices={1})
    final = res[0] if n_iters % 2 == 0 else res[2]
    out = bf16_u16_to_f32(final).reshape(p["IMG_H"], p["IMG"], ic)[PAD:PAD+G, PAD:PAD+G, :]
    return torch.from_numpy(out).to(torch.bfloat16)


def run_rn3_chain_geo(geo: str, inp_hwc: torch.Tensor, weight_pairs,
                      x2_hwc: torch.Tensor | None = None,
                      rnm_w_u16: np.ndarray | None = None):
    """Chained bottleneck iterations; optional fused rnm epilogue.

    Without rnm: returns the chain output (G x G x ic).
    With x2_hwc + rnm_w_u16: the launch additionally computes
    rnm = SiLU(BN(W3 . concat(chain_out, x2))) and returns that (G x G x 2ic),
    replacing the model's separate rnm gemm launch.
    """
    from conv.aie2_rn3_chain_geo import geo_params
    p = geo_params(geo)
    ic, G = p["IC"], p["GBOUND"]
    nt = p["WORKER_TILES"][0]
    rnm = 1 if rnm_w_u16 is not None else 0
    n_iters = len(weight_pairs)
    key = (geo, rnm) + tuple(id(a) for pr in weight_pairs for a in pr)
    if rnm:
        key += (id(rnm_w_u16),)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        blocks = [np.tile(_pack_geo_iter(w1, w2, ic, p["WSLOT"], p["N_BLK"]), nt)
                  for w1, w2 in weight_pairs]
        if rnm:
            blocks.append(_pack_rnm_slots(np.asarray(rnm_w_u16), ic, p["WSLOT"], nt))
        cached = (np.concatenate(blocks), list(weight_pairs), rnm_w_u16)
        _WEIGHT_CACHE[key] = cached
    weights = cached[0]
    n_elems = p["IMG_ELEMS_RNM"] if rnm else p["IMG_ELEMS"]
    img = np.zeros(n_elems, np.float32)
    imgv = img[:p["IMG_H"] * p["IMG"] * ic].reshape(p["IMG_H"], p["IMG"], ic)
    imgv[PAD:PAD+G, PAD:PAD+G, :] = inp_hwc.float().numpy()
    r = _get_geo_runner(geo, n_iters, rnm)
    tpc8 = p["TILES_PER_COL"] * 8
    if rnm:
        # x2 lives in BOTH image BOs below the chain region; rnm out drains
        # into the OUT region of the final image BO (shims only have 2 S2MM
        # channels, so a separate out BO can't be drained)
        x0 = (p["X2_ROW0"] + PAD) * p["IMG"] * ic
        x2v = img[x0:x0 + G * p["IMG"] * ic].reshape(G, p["IMG"], ic)
        x2v[:, PAD:PAD+G, :] = x2_hwc.float().numpy()
    img_u16 = f32_to_bf16_u16(img)
    res = r.run(img_u16, weights, img_u16.copy(),
                bo_key=f"rn3geo_{geo}_{rnm}_{id(weights)}", output_indices={0, 2},
                inout_indices={0, 2}, static_indices={1})
    final = res[0] if n_iters % 2 == 0 else res[2]
    if rnm:
        o0 = p["OUT_OFF"]
        out = bf16_u16_to_f32(final[o0:o0 + p["OUT_ELEMS"]]).reshape(tpc8, G, 2 * ic)[:G]
        return torch.from_numpy(out).to(torch.bfloat16)
    out = bf16_u16_to_f32(final[:p["IMG_H"] * p["IMG"] * ic]).reshape(
        p["IMG_H"], p["IMG"], ic)[PAD:PAD+G, PAD:PAD+G, :]
    return torch.from_numpy(out).to(torch.bfloat16)
