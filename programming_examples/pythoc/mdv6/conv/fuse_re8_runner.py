#!/usr/bin/env python3
"""Milestone B1 — on-device concat -> conv4 for the re8 RepNCSPELAN block.

Productionizes the M0 proof (`aie2_concat_proof.concat_proof`) into a model
callable. The four branch tensors x1,x2,x3,x4 (each [H,W,Q_IC=128] bf16 HWC) are
delivered to one device input BO STACKED (each quarter flat-contiguous, NO host
np.concatenate). The device does the channel-dim concat as a single strided
gather-fill (M0's mechanism), then runs the model's K-blocked conv1x1 GEMM
(gemm_conv1x1_kblocked_bf16) — bit-identical to the deployed re8 c4 because the
concat gather is bit-exact (proven M0 concat_only) and the GEMM kernel object,
weight repack, and tiling are the SAME as _run_gemm_kblocked_merged resolves.

CONTEXT MODEL: the fused op is built as a MERGED ELF (build_fuse_re8_merged.py,
@main entry, xrt.elf-loadable) so it lives in run_tiled_mc._MERGED_KERNELS and
shares the model's hw_context budget/LRU — instead of a standalone xclbin
context that is purely additive and wedges the at-ceiling (29) frame. The merged
LRU evicts one least-recently-used context when the fused ELF loads, keeping the
live count at the ceiling (one extra reload/frame of the evicted ELF). The fused
ELF's dispatch (in, wt, out) is bit-for-bit the model's GEMM dispatch machinery.

Gated behind MDV6_FUSE_RE8 in run_re_mc; default OFF.
"""
from __future__ import annotations

import math
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

import importlib.util as _ilu  # noqa: E402
_ett_spec = _ilu.spec_from_file_location(
    "_ett_for_fuse", str(_MDV6 / "_full_model_helpers" / "elan_test_tiled.py"))
_ett = _ilu.module_from_spec(_ett_spec)
_ett_spec.loader.exec_module(_ett)
bf16_to_uint16 = _ett.bf16_to_uint16
uint16_to_bf16 = _ett.uint16_to_bf16

from build_fuse_re8_merged import build as _build_merged_elf, ELF_NAME  # noqa: E402

N_CORES = 32

# The full mdv6 frame peaks at 29 live merged contexts (firmware ceiling). The
# (24,512,256,kb32,p1) GEMM shape ELF is used ONLY by re8 c4, re21 c4, and spp9
# conv5 — ALL of which are routed through this fused op when MDV6_FUSE_RE8=1. So
# the GEMM-only ELF has no remaining user: we DISPLACE it one-for-one (drop +
# mark unavailable), and the fused ELF takes its context slot. Net-neutral on
# the budget, no LRU thrash.
_DISPLACED_ELF = "merged_gemm_t24_ic512_oc256_kb32_p1_x1"

_MCR = None              # live run_tiled_mc module (shared context pool)
_REGISTERED = False


def _ensure_built():
    """Build the merged fused ELF if missing (into the model's merged ELF dir)."""
    mcr = _MCR
    bd = getattr(mcr, "_MERGED_BD", None)
    return _build_merged_elf(build_dir=bd)


def _ensure_registered():
    """Build + register the fused merged ELF into the model's pool, displacing
    the now-unused GEMM-only c4 ELF. Returns the kernel entry."""
    global _REGISTERED
    mcr = _MCR
    _ensure_built()
    if not _REGISTERED:
        # Free the GEMM-only c4 context (if loaded) and block reloads — all its
        # users are now fused, so the fused ELF inherits its slot.
        if hasattr(mcr, "_drop_merged_kernel"):
            mcr._drop_merged_kernel(_DISPLACED_ELF)
        mcr._MERGED_KERNELS[_DISPLACED_ELF] = None
        _REGISTERED = True
    entry = mcr._get_merged_kernel(ELF_NAME)
    if entry is None:
        raise RuntimeError(f"fused ELF {ELF_NAME}.elf not loadable")
    return entry


def run_concat_c4(quarters_hwc, conv4_weights_u16, H, W, oc,
                  tile_m=24, k_block=32, mcr_mod=None):
    """On-device concat[x1,x2,x3,x4] -> conv4 (1x1, n_q*q_ic -> oc) + BN + SiLU.

    quarters_hwc: list of n_q bf16 tensors, each [H, W, q_ic] HWC.
    conv4_weights_u16: fuse_bn(layer.conv4) flat uint16 [oc*ic + 2*oc].
    mcr_mod: the model's live run_tiled_mc module (shared context pool).
    Returns [H, W, oc] bf16, matching _run_gemm_kblocked_merged's c4 output.
    """
    global _MCR
    if mcr_mod is not None:
        _MCR = mcr_mod
    mcr = _MCR

    n_q = len(quarters_hwc)
    q_ic = quarters_hwc[0].shape[2]
    ic = n_q * q_ic
    M = H * W
    assert ic % k_block == 0

    ppc = max(1, math.ceil(M / (N_CORES * tile_m)))
    covered = N_CORES * tile_m * ppc          # >= M, zero-padded tail
    total_slots = N_CORES * ppc

    entry = _ensure_registered()
    device, _elf, _ctx, kernel = entry

    # ---- input: 4 quarters STACKED, each [covered, q_ic] flat, tail zero. ----
    n_pix = covered
    in_u16 = np.zeros(n_q * n_pix * q_ic, dtype=np.uint16)
    for k, q in enumerate(quarters_hwc):
        flat = bf16_to_uint16(q.reshape(M, q_ic).contiguous())
        in_u16[k * n_pix * q_ic: k * n_pix * q_ic + M * q_ic] = flat.reshape(-1)

    # ---- weights: model's exact K-blocked repack ----
    wt_u16 = mcr._repack_weights_kblocked(
        np.asarray(conv4_weights_u16, np.uint16), ic, oc, k_block)

    out_elems = total_slots * tile_m * oc
    # concat_proof dispatcher @main arg order is (I_stacked, wt, out).
    in_bo = mcr._get_merged_bo(device, ELF_NAME, "in", in_u16.nbytes)
    wt_bo = mcr._get_merged_bo(device, ELF_NAME, "wt", wt_u16.nbytes)
    out_bo = mcr._get_merged_bo(device, ELF_NAME, "out", out_elems * 2)
    mcr._xrt_fill_bo(in_bo, in_u16)
    mcr._xrt_fill_bo(wt_bo, wt_u16)
    mcr._xrt_run_kernel(kernel, [in_bo, wt_bo, out_bo])
    out_data = mcr._xrt_read_bo(out_bo, out_elems)

    got = uint16_to_bf16(np.asarray(out_data, np.uint16))
    got = got.reshape(covered, oc)[:M].reshape(H, W, oc)
    return got.to(torch.bfloat16)
