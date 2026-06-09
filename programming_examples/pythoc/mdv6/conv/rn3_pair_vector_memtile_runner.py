#!/usr/bin/env python3
"""Reusable 32-core memtile vector rn3-pair runner for re6.

Opt-in integration target for MDV6 `mc_re6_rn3` bottleneck pairs. The runner:

* compiles/loads `aie2_rn3_pair_vector_memtile` once;
* packs HWC bf16/float input into 25 row-major halo patches padded to 32;
* converts existing fused Conv+BN bf16 weight arrays into the 12 vector slots;
* runs one resident xclbin dispatch with static weight BO reuse;
* scatters the first 25 tile outputs back to HWC torch.bfloat16.

This is intentionally re6-only for now: H=W=40, C=48, tile=8, output C=48.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aie.utils.compile import compile_mlir_module

from conv.aie2_rn3_pair_vector_memtile import rn3_pair_vector_memtile  # noqa: E402
from conv.aie2_rn3_pair_vector_ocb import (  # noqa: E402
    ARENA_SIZE,
    FINAL_OFFSET,
    IC,
    INPUT_SIZE,
    MASK_OFFSET,
    MASK_SIZE,
    MID_BLOCK,
    N_MID_BLOCKS,
    N_OC_BLOCKS,
    N_WEIGHT_SLOTS,
    OC_BLOCK,
    TILE_H,
    TILE_W,
    W1_SIZE,
    W2_SIZE,
    WEIGHT_SLOT_SIZE,
)
from conv.resident_xclbin_runner import ResidentXCLBinRunner  # noqa: E402
from conv.rn3_pair_layout import pack_rn3_pair_input_patches, scatter_tile_hwc_to_image  # noqa: E402
from conv.test_rn3_pair_vector_oneblock_hw import (  # noqa: E402
    bf16_u16_to_f32,
    f32_to_bf16_u16,
    pack_3x3_weights_u16,
)

_N_CORES = 32
_BUILD_DIR = Path(os.environ.get("MDV6_RN3PAIR_MEMTILE_BUILD_DIR", "conv/build_rn3_pair_vector_memtile_re6_runtime_fullacc_plane"))
_RUNNER: ResidentXCLBinRunner | None = None
_RUNNER_KEY: Tuple[str, str] | None = None


def _compile_once() -> tuple[Path, Path]:
    wd = _BUILD_DIR / "cores32"
    xclbin = wd / "final.xclbin"
    insts = wd / "insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts
    wd.mkdir(parents=True, exist_ok=True)
    module = rn3_pair_vector_memtile(n_cores=_N_CORES)
    with open(wd / "kernel.mlir", "w", encoding="utf-8") as f:
        print(module, file=f)
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(insts),
        xclbin_path=str(xclbin),
        work_dir=str(wd),
        verbose=False,
    )
    return xclbin, insts


def _get_runner() -> ResidentXCLBinRunner:
    global _RUNNER, _RUNNER_KEY
    xclbin, insts = _compile_once()
    key = (str(xclbin), str(insts))
    if _RUNNER is None or _RUNNER_KEY != key:
        if _RUNNER is not None:
            _RUNNER.close()
        _RUNNER = ResidentXCLBinRunner(xclbin, insts)
        _RUNNER.__enter__()
        _RUNNER_KEY = key
    return _RUNNER


def close_runner():
    global _RUNNER, _RUNNER_KEY
    if _RUNNER is not None:
        _RUNNER.close()
    _RUNNER = None
    _RUNNER_KEY = None


def fused_weight_u16_to_parts(w_u16: np.ndarray, oc: int = 48, ic: int = 48):
    arr = np.asarray(w_u16, dtype=np.uint16).reshape(-1)
    wt_size = oc * ic * 3 * 3
    need = wt_size + 2 * oc
    if arr.size < need:
        raise ValueError(f"fused weight has {arr.size} u16, need at least {need}")
    w = bf16_u16_to_f32(arr[:wt_size]).reshape(oc, ic, 3, 3)
    bn_w = bf16_u16_to_f32(arr[wt_size:wt_size + oc])
    bn_b = bf16_u16_to_f32(arr[wt_size + oc:wt_size + 2 * oc])
    return w, bn_w, bn_b


def pack_vector_weight_slots_from_fused(w1_u16: np.ndarray, w2_u16: np.ndarray) -> np.ndarray:
    w1, bn1w, bn1b = fused_weight_u16_to_parts(w1_u16, MID_BLOCK * N_MID_BLOCKS, IC)
    w2, bn2w, bn2b = fused_weight_u16_to_parts(w2_u16, OC_BLOCK * N_OC_BLOCKS, MID_BLOCK * N_MID_BLOCKS)
    slots = np.zeros((N_WEIGHT_SLOTS, WEIGHT_SLOT_SIZE), dtype=np.uint16)
    wi = 0
    for mb in range(N_MID_BLOCKS):
        lo = mb * MID_BLOCK
        hi = lo + MID_BLOCK
        packed = pack_3x3_weights_u16(w1[lo:hi], bn1w[lo:hi], bn1b[lo:hi])
        slots[wi, :W1_SIZE] = packed
        wi += 1
    for ob in range(N_OC_BLOCKS):
        olo = ob * OC_BLOCK
        ohi = olo + OC_BLOCK
        packed = pack_3x3_weights_u16(w2[olo:ohi, :], bn2w[olo:ohi], bn2b[olo:ohi])
        slots[wi, :W2_SIZE] = packed
        wi += 1
    return slots.reshape(-1)


def _conv1_valid_masks(image_h: int = 40, image_w: int = 40) -> np.ndarray:
    """Return row-major [25,100] mask for the 10x10 conv1 scratch per re6 tile.

    Scratch position (sr, sc) maps to full-image conv1 output coordinate
    (tile_r*8 + sr - 1, tile_c*8 + sc - 1). Baseline conv2 padding supplies
    zero for coordinates outside [0,H)×[0,W); fused tiles must zero those
    scratch entries instead of computing biased conv1 values there.
    """
    masks = []
    for tr in range(image_h // TILE_H):
        for tc in range(image_w // TILE_W):
            m = np.zeros((TILE_H + 2, TILE_W + 2), dtype=np.float32)
            for sr in range(TILE_H + 2):
                gh = tr * TILE_H + sr - 1
                for sc in range(TILE_W + 2):
                    gw = tc * TILE_W + sc - 1
                    if 0 <= gh < image_h and 0 <= gw < image_w:
                        m[sr, sc] = 1.0
            masks.append(m.reshape(-1))
    return np.stack(masks, axis=0)


_ARENA_TEMPLATE_CACHE: dict = {}


def pack_input_arenas_from_hwc(inp_hwc: torch.Tensor | np.ndarray) -> tuple[np.ndarray, int]:
    if isinstance(inp_hwc, torch.Tensor):
        arr = inp_hwc.detach().cpu().float().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(inp_hwc, dtype=np.float32)
    if arr.shape != (40, 40, IC):
        raise ValueError(f"re6 runner expects HWC shape (40,40,{IC}), got {arr.shape}")
    patches = pack_rn3_pair_input_patches(arr, TILE_H, TILE_W, halo=2)
    n_valid = patches.shape[0]
    if n_valid != 25:
        raise ValueError(f"expected 25 re6 patches, got {n_valid}")
    patch_u16 = f32_to_bf16_u16(patches.reshape(-1))
    key = (arr.shape[0], arr.shape[1], n_valid)
    tmpl = _ARENA_TEMPLATE_CACHE.get(key)
    if tmpl is None:
        tmpl = np.zeros((_N_CORES, ARENA_SIZE), dtype=np.uint16)
        mask_u16 = f32_to_bf16_u16(_conv1_valid_masks(arr.shape[0], arr.shape[1]).reshape(-1)).reshape(n_valid, MASK_SIZE)
        tmpl[:n_valid, MASK_OFFSET:MASK_OFFSET + MASK_SIZE] = mask_u16
        # Padded workers are unused, but keep their mask all-ones to preserve
        # sane standalone behavior if a padded output is ever inspected.
        if n_valid < _N_CORES:
            tmpl[n_valid:, MASK_OFFSET:MASK_OFFSET + MASK_SIZE] = f32_to_bf16_u16(np.ones(MASK_SIZE, dtype=np.float32))
        _ARENA_TEMPLATE_CACHE[key] = tmpl
    arenas = tmpl.copy()
    arenas[:n_valid, :INPUT_SIZE] = patch_u16.reshape(n_valid, INPUT_SIZE)
    return arenas.reshape(-1), n_valid


def extract_image_from_output(raw_flat: np.ndarray, n_valid: int = 25) -> np.ndarray:
    raw = np.asarray(raw_flat, dtype=np.uint16).reshape(_N_CORES, ARENA_SIZE)
    blk = TILE_H * TILE_W * OC_BLOCK
    final = raw[:n_valid, FINAL_OFFSET:FINAL_OFFSET + N_OC_BLOCKS * blk]
    f32 = bf16_u16_to_f32(final.reshape(-1)).reshape(n_valid, N_OC_BLOCKS, TILE_H, TILE_W, OC_BLOCK)
    tile_arr = np.ascontiguousarray(f32.transpose(0, 2, 3, 1, 4)).reshape(n_valid, TILE_H, TILE_W, N_OC_BLOCKS * OC_BLOCK)
    return scatter_tile_hwc_to_image(tile_arr, 40, 40, TILE_H, TILE_W)


def extract_scratch_image_from_output(raw_flat: np.ndarray, n_valid: int = 25) -> np.ndarray:
    """Extract the valid 8x8 center of fused conv1 scratch as 40x40x48 HWC."""
    raw = np.asarray(raw_flat, dtype=np.uint16).reshape(_N_CORES, ARENA_SIZE)
    tiles = []
    for p in range(n_valid):
        blocks = []
        for mb in range(N_MID_BLOCKS):
            start = mb * (TILE_H + 2) * (TILE_W + 2) * MID_BLOCK
            stop = start + (TILE_H + 2) * (TILE_W + 2) * MID_BLOCK
            scratch = bf16_u16_to_f32(raw[p, start:stop]).reshape(TILE_H + 2, TILE_W + 2, MID_BLOCK)
            blocks.append(scratch[1:1 + TILE_H, 1:1 + TILE_W, :])
        tiles.append(np.concatenate(blocks, axis=2))
    tile_arr = np.stack(tiles, axis=0)
    return scatter_tile_hwc_to_image(tile_arr, 40, 40, TILE_H, TILE_W)


def run_re6_rn3_pair_debug(inp_hwc: torch.Tensor, w1_u16: np.ndarray, w2_u16: np.ndarray, *, bo_key: str | None = None):
    """Return (conv2_out, conv1_scratch_center, raw_arena) for site debugging."""
    input_arenas, n_valid = pack_input_arenas_from_hwc(inp_hwc)
    weights = pack_vector_weight_slots_from_fused(w1_u16, w2_u16)
    out_arg = np.zeros(_N_CORES * ARENA_SIZE, dtype=np.uint16)
    runner = _get_runner()
    key = bo_key or f"re6_rn3pair_dbg_{id(w1_u16)}_{id(w2_u16)}"
    res = runner.run(input_arenas, weights, out_arg, bo_key=key, output_indices={2}, static_indices={1})
    raw = res[2]
    out_f32 = extract_image_from_output(raw, n_valid)
    scratch_f32 = extract_scratch_image_from_output(raw, n_valid)
    return torch.from_numpy(out_f32).to(torch.bfloat16), torch.from_numpy(scratch_f32).to(torch.bfloat16), raw


_WEIGHT_SLOT_CACHE: dict = {}
_OUT_ARG = None


def run_re6_rn3_pair(inp_hwc: torch.Tensor, w1_u16: np.ndarray, w2_u16: np.ndarray, *, bo_key: str | None = None) -> torch.Tensor:
    global _OUT_ARG
    input_arenas, n_valid = pack_input_arenas_from_hwc(inp_hwc)
    wkey = (id(w1_u16), id(w2_u16))
    cached = _WEIGHT_SLOT_CACHE.get(wkey)
    if cached is None:
        # keep refs to the source arrays so the ids stay valid
        cached = (pack_vector_weight_slots_from_fused(w1_u16, w2_u16), w1_u16, w2_u16)
        _WEIGHT_SLOT_CACHE[wkey] = cached
    weights = cached[0]
    if _OUT_ARG is None:
        _OUT_ARG = np.zeros(_N_CORES * ARENA_SIZE, dtype=np.uint16)
    out_arg = _OUT_ARG
    runner = _get_runner()
    key = bo_key or f"re6_rn3pair_{id(w1_u16)}_{id(w2_u16)}"
    res = runner.run(input_arenas, weights, out_arg, bo_key=key, output_indices={2}, static_indices={1})
    out_f32 = extract_image_from_output(res[2], n_valid)
    return torch.from_numpy(out_f32).to(torch.bfloat16)


def last_stats():
    r = _RUNNER
    return None if r is None else r.last_stats
