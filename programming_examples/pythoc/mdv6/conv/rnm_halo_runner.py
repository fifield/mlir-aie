#!/usr/bin/env python3
"""A1 — rnm GEMM -> halo_c3 fused device-resident runner for the re8 RepNCSP hop.

Fuses the RepNCSP rnm (conv3, 1x1 128->128 + BN + SiLU) and the run_re_mc c3
(conv2[1]/conv3[1], 3x3 128->128 + BN + SiLU) into ONE merged ELF / ONE
hw_context (build_rnm_halo_merged), replacing TWO model launches (the rnm GEMM
dispatch + the mc_c3 3x3 dispatch) with ONE fused dispatch whose intermediate
PAD(2)-padded HWC seam stays device-resident.

LAYOUT BRIDGES (per the milestone's inter-hop layout question):
  - rnm input  : the model hands us `concat(chain_out, x2)` as [G,G,128] HWC. The
    rnm GEMM wants per-core valid-row patches (S1 layout): core c = valid row c,
    [gbound, ic] contiguous. Cheap host repack (reshape/copy), no compute.
  - rnm -> c3  : DEVICE-RESIDENT. The rnm GEMM drains a PAD(2)-padded HWC seam
    (gemm_pad_out) that IS the halo_c3 input (same MLIR type); shift=PAD-1 baked.
  - c3 output  : halo_c3 emits tiled-C stream layout; we deinterleave+untile to
    [G,G,oc] HWC on host (the model's c3 also returns HWC). No host conv.

BN+SiLU GAP (documented host touch): the halo_conv3x3_bfp kernel does RAW conv
(no BN/bias/SiLU). The model's mc_c3 applies BN+SiLU. We bridge by:
  (a) folding the c3 BN *scale* into the conv weights (BN is linear), so the
      device conv emits scale*conv(x), then
  (b) applying the c3 BN *bias* + SiLU on host over the [G,G,128] readback
      (a cheap elementwise op, NOT a conv).
This is NOT bit-identical to the model's mc_c3 (which keeps conv weights and BN
separate, so its per-block BFP quant of the conv weights differs from the
scale-folded weights here). The diff is a BFP-quant-level perturbation; the
milestone tolerance is detection max_class_diff < 5.0. Measured by the standalone
test below and by the frame-level detection check.

Gated behind MDV6_FUSE_RE8 in run_re_mc/run_rn_mc; default OFF.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

# See chain_rnm_c3_runner: the merged-ELF path holds a large frame-invariant
# live object graph; the harness's per-frame gc.collect() re-scans it (the
# dominant pre_post regression). Freeze the live set once the fused ELF is
# registered + weights cached so collect() skips it. Default ON.
_GC_FREEZE = os.environ.get("MDV6_FUSE_NO_GC_FREEZE", "0") in ("", "0", "false", "False")
_GC_FROZEN = False

_HERE = Path(__file__).resolve().parent
_MDV6 = _HERE.parent
for _p in (str(_HERE), str(_MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_rnm_halo_merged import build as _build_rnm_halo
from aie2_gemm_pad_out import PAD, TILE
from test_gemm_truth import _pack_weights_blocked
from test_halo_conv_hw import bf16, to_u16, tile_b, untile_c
from aie2_halo_conv import deinterleave_stream_out

_MCR = None
_REGISTERED = False
_ELF_NAME = None
_META = None  # (gmeta, hmeta)
_WT_CACHE = {}     # id(weights_u16) -> packed buffers (frame-stable weight ids)
_UNTILE_IDX = {}  # (gbound, oc, N_TILES) -> cached gather index per output shape

# --- REUSED PER-HOP HOST SCRATCH (filled in place; avoids per-call np.zeros) ---
# Mirrors chain_rnm_c3_runner: the rnm-input S1 host buffer and the output
# gather/bf16 buffers were re-allocated every dispatch and then scanned by the
# harness's per-frame gc.collect(). Pool them by output shape. The returned
# bf16 result rotates through a small ring so x3/x4 (both consumed by the c4
# concat) never alias.
_IN_SCRATCH = {}   # n_cores*input_tile_size -> u16 host_in buffer
_OUT_SCRATCH = {}  # (gbound, oc, n_out) -> dict(f32, bf16=[ring], i)
_OUT_RING = 4


def _in_scratch(n):
    s = _IN_SCRATCH.get(n)
    if s is None:
        s = np.zeros(n, np.uint16)
        _IN_SCRATCH[n] = s
    return s


def _out_scratch(n_out, gbound, oc):
    key = (gbound, oc, n_out)
    s = _OUT_SCRATCH.get(key)
    if s is None:
        s = dict(f32=np.empty(n_out, np.float32),
                 bf16=[torch.empty(gbound, gbound, oc, dtype=torch.bfloat16)
                       for _ in range(_OUT_RING)],
                 i=0)
        _OUT_SCRATCH[key] = s
    s["i"] = (s["i"] + 1) % _OUT_RING
    return s

# The model's kernel-side SiLU approximation (mirrors conv3x3_fused_packed_bf16).
# Matches test_gemm_truth._silu_kernel so the host epilogue is the SAME SiLU the
# model's mc_c3 kernel applies.
def _silu_kernel(x: torch.Tensor) -> torch.Tensor:
    ax = torch.abs(x)
    return x * (0.5 + x / (2.0 + 2.0 * ax))


def _u16_to_bf16_t(u: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(u, np.uint16)).view(torch.bfloat16)


def _build_untile_index(hmeta, gbound):
    """Precompute a gather index `idx` of length G*G*oc such that
    out_f32[idx].reshape(G,G,oc) == the deinterleave+untile+scatter HWC result.

    Derived once by tracing the reference path on an arange (values = source
    positions), so it stays bit-identical to deinterleave_stream_out + untile_c.
    Cached per output shape (re8 + re6 each have a distinct (gbound, oc))."""
    N_TILES, C_ELEMS = hmeta["N_TILES"], hmeta["C_ELEMS"]
    GRID = hmeta["GRID"]
    oc = (C_ELEMS // 64)  # C_ELEMS = N_BLK_OC*64; oc = N_BLK_OC*8
    ckey = (gbound, oc, N_TILES)
    if ckey in _UNTILE_IDX:
        return _UNTILE_IDX[ckey]
    src = np.arange(N_TILES * C_ELEMS, dtype=np.int64)
    flat = deinterleave_stream_out(src.astype(np.float32), hmeta).astype(np.int64)
    idx = np.zeros(gbound * gbound * oc, np.int64)
    for t in range(N_TILES):
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[t * C_ELEMS:(t + 1) * C_ELEMS].astype(np.float32), oc).astype(np.int64)
        for pl in range(64):
            oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
            if oh < gbound and ow < gbound:
                base = (oh * gbound + ow) * oc
                idx[base:base + oc] = tile[pl, :]
    _UNTILE_IDX[ckey] = idx
    return idx


def _build_untile_index_mt(hmeta, gbound):
    """Gather index for the tiles-per-core (tpc>1) merged halo OUT.

    The mt halo drains a column-round-packed, BLOCK-major raster-slot stream
    (n_slots slots, slot == raster tile idx). `idx` (length G*G*oc) such that
    raw_out[idx].reshape(G,G,oc) == the deinterleave_stream_mt + de-raster +
    untile_c HWC result. Traced once on an arange so it stays byte-identical."""
    from aie2_halo_conv_mt import deinterleave_stream_mt, slot_to_tile
    GRID = hmeta["GRID"]; C_ELEMS = hmeta["C_ELEMS"]; n_slots = hmeta["n_slots"]
    oc = (C_ELEMS // 64)
    ckey = ("mt", gbound, oc, n_slots)
    if ckey in _UNTILE_IDX:
        return _UNTILE_IDX[ckey]
    src = np.arange(n_slots * C_ELEMS, dtype=np.int64)
    flat = deinterleave_stream_mt(src.astype(np.float32), hmeta).astype(np.int64)
    idx = np.zeros(gbound * gbound * oc, np.int64)
    for slot in range(n_slots):
        t = slot_to_tile(slot, hmeta)
        if t is None:
            continue
        tr, tc = t // GRID, t % GRID
        tile = untile_c(flat[slot * C_ELEMS:(slot + 1) * C_ELEMS].astype(np.float32), oc).astype(np.int64)
        for pl in range(64):
            oh, ow = tr * TILE + pl // 8, tc * TILE + pl % 8
            if oh < gbound and ow < gbound:
                base = (oh * gbound + ow) * oc
                idx[base:base + oc] = tile[pl, :]
    _UNTILE_IDX[ckey] = idx
    return idx


def _ensure_registered():
    """Build + register the fused rnm->halo ELF into the model's shared pool."""
    global _REGISTERED, _ELF_NAME, _META
    mcr = _MCR
    bd = getattr(mcr, "_MERGED_BD", None)
    elf_path, gmeta, hmeta = _build_rnm_halo(ic=128, oc=128, gbound=20,
                                             build_dir=bd, stream_oc="block")
    if elf_path is None:
        raise RuntimeError("rnm_halo merged ELF build failed")
    _ELF_NAME = Path(elf_path).stem
    _META = (gmeta, hmeta)
    entry = mcr._get_merged_kernel(_ELF_NAME)
    if entry is None:
        raise RuntimeError(f"fused ELF {_ELF_NAME}.elf not loadable")
    _REGISTERED = True
    return entry


def _c3_weights_for_halo(c3_w_u16: np.ndarray, oc: int, ic: int, stream_block=True):
    """Pack the model's fuse_bn(c3) buffer into the halo weight slot layout for
    in-kernel BN+SiLU: RAW (un-scaled) BFP-tiled conv weights with bn_w/bn_b
    appended (chain layout). The BN scale is NO LONGER folded into the weights —
    the kernel applies silu(conv(x)*bn_w + bn_b) on the f32 accumulator, so the
    weights quantize UN-scaled (eliminates the scale-into-BFP rounding error).

    c3_w_u16 = [conv_OIHW(oc*ic*9), bn_w(oc), bn_b(oc)] (uint16 bf16).

    stream_block=True (the model's stream_oc="block" path): emit per-oc-block
    units of [conv slot (KKMAX*64) + bn_w(8) + bn_b(8)], oc-block-major, so the
    host wt TAP (offset p*WSLOT_PAIR) hands each unit its own conv+bn tail.
    stream_block=False (non-stream OC<=64 path): [all conv blocks][bn_w(oc)][bn_b(oc)].
    Returns ONLY the uint16 halo weight buffer (BN bias is now in-kernel)."""
    conv_n = oc * ic * 9
    conv = _u16_to_bf16_t(c3_w_u16[:conv_n]).float().reshape(oc, ic, 3, 3)
    bn_w = _u16_to_bf16_t(c3_w_u16[conv_n:conv_n + oc]).float()          # [oc]
    bn_b = _u16_to_bf16_t(c3_w_u16[conv_n + oc:conv_n + 2 * oc]).float()  # [oc]
    # tile_b expects W[oc, 9, ic]; OIHW [oc, ic, 3, 3] -> [oc, 9, ic] (UN-scaled).
    W = conv.reshape(oc, ic, 9).permute(0, 2, 1).contiguous().numpy()
    conv_tiled = bf16(tile_b(bf16(W), ic, oc))            # [N_BLK_OC*KKMAX*64]
    bn_w_u16 = to_u16(bf16(bn_w.numpy()))
    bn_b_u16 = to_u16(bf16(bn_b.numpy()))
    n_blk_oc = oc // 8
    kkmax = (ic // 8) * 9
    conv_u16 = to_u16(conv_tiled).reshape(n_blk_oc, kkmax * 64)
    if stream_block:
        raw = kkmax * 64 + 16                    # 1 conv slot + bn_w(8) + bn_b(8)
        wslot_pair = ((raw + 63) // 64) * 64      # padded to 64-elem multiple
        units = []
        for p in range(n_blk_oc):
            unit = np.concatenate([conv_u16[p], bn_w_u16[p * 8:(p + 1) * 8],
                                   bn_b_u16[p * 8:(p + 1) * 8]])
            if unit.size < wslot_pair:
                unit = np.concatenate([unit, np.zeros(wslot_pair - unit.size, np.uint16)])
            units.append(unit)
        return np.concatenate(units).astype(np.uint16)
    return np.concatenate([conv_u16.reshape(-1), bn_w_u16, bn_b_u16]).astype(np.uint16)


def run_rnm_c3(concat_hwc: torch.Tensor, rnm_w_u16, c3_w_u16, H, W, oc,
               mcr_mod=None) -> torch.Tensor:
    """Fused rnm (1x1 + BN + SiLU) -> c3 (3x3 + BN + SiLU), device-resident seam.

    concat_hwc: [H, W, ic] bf16 HWC = concat(chain_out, x2), the rnm input.
    rnm_w_u16 : fuse_bn(repncsp.conv3)  flat uint16 [oc*ic + 2*oc].
    c3_w_u16  : fuse_bn(layer.conv2[1]) flat uint16 [oc*ic*9 + 2*oc].
    Returns [H, W, oc] bf16 = c3 output (BN+SiLU applied), matching the model's
    `x3 = rt(mc_c3, ...)` to within BFP-quant tolerance.
    """
    global _MCR
    if mcr_mod is not None:
        _MCR = mcr_mod
    mcr = _MCR
    assert H == W == 20 and oc == 128, "rnm_halo fused path is the re8 20x20x128 shape"

    if not _REGISTERED:
        _ensure_registered()
    gmeta, hmeta = _META
    IMG, IMG_ELEMS = gmeta["IMG"], gmeta["IMG_ELEMS"]
    n_cores, input_tile_size = gmeta["n_cores"], gmeta["input_tile_size"]
    N_TILES, C_ELEMS, GRID = hmeta["N_TILES"], hmeta["C_ELEMS"], hmeta["GRID"]
    gbound = gmeta["gbound"]
    ic = concat_hwc.shape[2]

    entry = mcr._get_merged_kernel(_ELF_NAME)
    device, _elf, _ctx, kernel = entry

    # ---- rnm input: per-core valid-row S1 packing (reused host buffer) ----
    fmap_u16 = concat_hwc.contiguous().view(torch.uint16).numpy()  # [G,G,ic]
    host_in = _in_scratch(n_cores * input_tile_size)
    for r in range(n_cores):
        host_in[r * input_tile_size:(r + 1) * input_tile_size] = fmap_u16[r].reshape(-1)

    # ---- weights: cache the (slow) blocked/tiled repacks by weight id ----
    wkey = (id(rnm_w_u16), id(c3_w_u16))
    cached = _WT_CACHE.get(wkey)
    if cached is None:
        rnm_conv = _u16_to_bf16_t(rnm_w_u16[:oc * ic]).reshape(oc, ic)
        rnm_bnw = _u16_to_bf16_t(rnm_w_u16[oc * ic:oc * ic + oc])
        rnm_bnb = _u16_to_bf16_t(rnm_w_u16[oc * ic + oc:oc * ic + 2 * oc])
        gemm_wt_u16 = _pack_weights_blocked(rnm_conv, rnm_bnw, rnm_bnb)
        halo_wt_u16 = _c3_weights_for_halo(np.asarray(c3_w_u16, np.uint16), oc, ic)
        cached = (gemm_wt_u16, halo_wt_u16)
        _WT_CACHE[wkey] = cached
    gemm_wt_u16, halo_wt_u16 = cached

    out_elems = N_TILES * C_ELEMS
    in_bo = mcr._get_merged_bo(device, _ELF_NAME, "in", host_in.nbytes)
    gwt_bo = mcr._get_merged_bo(device, _ELF_NAME, "gwt", gemm_wt_u16.nbytes)
    seam_bo = mcr._get_merged_bo(device, _ELF_NAME, "seam", IMG_ELEMS * 2)
    hwt_bo = mcr._get_merged_bo(device, _ELF_NAME, "hwt", halo_wt_u16.nbytes)
    out_bo = mcr._get_merged_bo(device, _ELF_NAME, "out", out_elems * 4)

    mcr._xrt_fill_bo(in_bo, host_in)
    mcr._xrt_fill_bo(gwt_bo, gemm_wt_u16)
    mcr._xrt_fill_bo(seam_bo, np.zeros(IMG_ELEMS, np.uint16))  # poison PAD border
    mcr._xrt_fill_bo(hwt_bo, halo_wt_u16)
    mcr._xrt_run_kernel(kernel, [in_bo, gwt_bo, seam_bo, hwt_bo, out_bo])
    global _GC_FROZEN
    if _GC_FREEZE and not _GC_FROZEN:
        # ELF registered + weights cached + BOs allocated: freeze the now-live
        # object graph so the harness's per-frame gc.collect() stops re-scanning
        # it (the dominant pre_post cost of the fused path).
        gc.collect(); gc.freeze()
        _GC_FROZEN = True
    import pyxrt as _xrt
    out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, out_elems * 4, 0)
    out_view = np.frombuffer(out_bo.map(), dtype=np.float32, count=out_elems)

    # ---- deinterleave + untile -> [G,G,oc] HWC (vectorized via cached index) ----
    # The kernel already applied BN + SiLU on the f32 accumulator, so out_view IS
    # the final activation (no host epilogue, no BN-scale weight fold). Gather into
    # reused scratch + a rotating bf16 ring slot (no per-hop allocation).
    idx = _build_untile_index(hmeta, gbound)   # maps raw out -> [G*G*oc]
    osc = _out_scratch(idx.size, gbound, oc)
    got_f32 = np.take(out_view, idx, out=osc["f32"])
    result = osc["bf16"][osc["i"]]
    result.copy_(torch.from_numpy(got_f32).reshape(gbound, gbound, oc))
    return result
