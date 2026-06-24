#!/usr/bin/env python3
"""Full re8 hop — chain -> rnm GEMM -> halo_c3 in ONE merged ELF / ONE hw_context.

Subsumes THREE model ops (the rn3 chain over x1b, the RepNCSP rnm 1x1 GEMM over
concat(chain_out, x2b), and the run_re_mc c3 3x3) into a single device dispatch
whose two intermediate seams (chain->rnm and rnm->c3) stay device-resident. This
folds the chain's previously-separate ResidentXCLBin dispatch INTO the merged
ELF, so vs the prior rnm->c3-only fusion (run_rnm_c3) it removes BOTH the chain
launch AND the host concat/repack between chain and rnm.

Built by build_chain_rnm_halo_merged.build (sub0=chain, sub1=depad+concat rnm
GEMM, sub2=halo_c3; chain_links=[(0,2,1,0),(1,2,2,0)]). Proven bit-exact by
conv/test_chain_rnm_halo_merged_hw.py (max_diff 0.045 within BFP tol).

MARSHALLING (the construct):
  - chain input A (arg0): x1b host-PAD(2)-padded into the LOWER half of a widened
    IMG*IMG*ic2 stacked BO (size IN_ELEMS = 2*HALF_ELEMS), upper half zeros.
  - chain weights (arg1): per-iter _pack_geo_iter tiled nt times (== chain runner).
  - chain output B (arg2): a copy of A with x2b host-PAD(2)-padded into the UPPER
    half (HALF_ELEMS:IN_ELEMS). The chain overwrites the lower half in place; the
    x2b upper half survives the ping-pong and is the dcg concat's other half.
  - rnm GEMM weights (arg3): _pack_weights_blocked(rnm_conv, rnm_bnw, rnm_bnb).
  - seam (arg4): the device-resident rnm->c3 PAD(2) seam (poisoned border on host).
  - halo (c3) weights (arg5): c3 conv with BN-scale folded in, tiled (tile_b).
  - out (arg6): tiled-C stream; deinterleave+untile -> [G,G,oc] HWC on host, then
    the c3 BN-bias + SiLU epilogue (reused verbatim from rnm_halo_runner).

BN+SiLU GAP (same documented host touch as run_rnm_c3): the halo kernel does RAW
scale*conv; the c3 BN-bias + SiLU is applied on the [G,G,oc] readback. This is a
BFP-quant-level perturbation, not bit-identical to the model's separate-BN mc_c3.

Gated behind MDV6_FUSE_RE8 in run_re_mc; default OFF.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Static-weight residency is ON by default for the fused re8 path. Set
# MDV6_FUSE_RE8_RESIDENT=0 to A/B against the old re-upload-every-call behavior
# (weights then route through the shared per-(elf,role) merged BO pool, colliding
# across the 4 hops and re-filled each dispatch).
_RESIDENT_WT = os.environ.get("MDV6_FUSE_RE8_RESIDENT", "1") not in ("", "0", "false", "False")

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MDV6 = _HERE.parent
for _p in (str(_HERE), str(_MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_chain_rnm_halo_merged import build as _build_chain_rnm_halo
from aie2_rn3_chain_geo import geo_params, PAD
from rn3_chain_runner import _pack_geo_iter
from test_rn3_pair_vector_oneblock_hw import f32_to_bf16_u16
from test_gemm_truth import _pack_weights_blocked
from rnm_halo_runner import (
    _silu_kernel, _u16_to_bf16_t, _build_untile_index, _c3_weights_for_halo,
)

_MCR = None
# PER-GEO REGISTRY: (H==W, proc) -> geo. The fused full-hop seam supports both
# the re8 (20x20, proc=128, oc=128) and re6 (40x40, proc=96, oc=96) RepNCSP
# shapes. re6 appears in rep_elan6/12/18 x2 hops each = 6 hops/frame; re8 in
# rep_elan8/21 x2 hops = 4 hops/frame. Each geo gets its OWN merged ELF +
# registration entry (keyed below); the chain image is tall for re6 (52 rows).
_GEO_BY_SHAPE = {
    (20, 128): "re8",
    (40, 96): "re6",
}
_GEO_CFG = {  # geo -> (oc, gbound) for the merged build
    "re8": dict(oc=128, gbound=20),
    "re6": dict(oc=96, gbound=40),
}
# per-geo registration state (was module-global; now keyed so re8 + re6 coexist)
_REG = {}   # geo -> dict(elf_name, meta=(geo_p, dmeta, hmeta))
_N_ITERS = 3
_WT_CACHE = {}    # (id(chain weights tuple), id(rnm_w), id(c3_w)) -> packed buffers

# --- STATIC-WEIGHT RESIDENCY (ports ResidentXCLBinRunner.static_indices) ---
# The 4 re8 hops/frame share one merged ELF but carry DIFFERENT weights. The
# shared _get_merged_bo pool keys by (elf, role, size), so the three weight roles
# (wt/gwt/hwt) and the poison seam COLLIDE across hops — every dispatch re-fills
# (host->device DMA) the frame-invariant weights. We instead pool the weight +
# seam BOs by a STABLE per-hop key (the weight-identity tuple `wkey`, identical
# frame-to-frame because fuse_bn/fuse_repconv cache by module identity), fill+sync
# each ONCE, and skip the copy+sync on later dispatches. Only the mutable image
# BOs (a/b) and the output BO stay on the per-call shared pool.
_RESIDENT_WT_BO = {}      # (wkey, role) -> resident xrt BO (filled once)
_RESIDENT_FILLED = set()  # wkeys whose weight+seam BOs are already filled+synced


def _resident_wt_bo(device, wkey, role, nbytes):
    """Per-hop resident weight/seam BO, keyed by stable weight identity."""
    import pyxrt as _xrt
    k = (wkey, role)
    bo = _RESIDENT_WT_BO.get(k)
    if bo is None:
        bo = _xrt.ext.bo(device, nbytes)
        _RESIDENT_WT_BO[k] = bo
    return bo


def _ensure_registered(geo, n_iters):
    """Build + register the full chain->rnm->halo ELF for `geo` into the pool."""
    reg = _REG.get(geo)
    if reg is not None:
        return reg
    mcr = _MCR
    bd = getattr(mcr, "_MERGED_BD", None)
    cfg = _GEO_CFG[geo]
    elf_path, dmeta, hmeta = _build_chain_rnm_halo(
        geo=geo, n_iters=n_iters, ic2=geo_params(geo)["IC"],
        oc=cfg["oc"], gbound=cfg["gbound"], build_dir=bd, stream_oc="block")
    if elf_path is None:
        raise RuntimeError(f"chain_rnm_halo merged ELF build failed (geo={geo})")
    elf_name = Path(elf_path).stem
    if mcr._get_merged_kernel(elf_name) is None:
        raise RuntimeError(f"fused ELF {elf_name}.elf not loadable")
    reg = dict(elf_name=elf_name, meta=(geo_params(geo), dmeta, hmeta))
    _REG[geo] = reg
    return reg


def _pad_half(hwc_u16: np.ndarray, IMG: int, ic2: int, G: int,
              img_h: int = None) -> np.ndarray:
    """PAD(2)-pad a [G,G,ic2] uint16 HWC tensor into a flat img_h*IMG*ic2 half.
    img_h defaults to IMG (square, re8); pass the chain's tall height for re6."""
    if img_h is None:
        img_h = IMG
    half = np.zeros((img_h, IMG, ic2), np.uint16)
    half[PAD:PAD + G, PAD:PAD + G, :] = hwc_u16.reshape(G, G, ic2)
    return half.reshape(-1)


def run_chain_rnm_c3(x1b: torch.Tensor, x2b: torch.Tensor, chain_pairs,
                     rnm_w_u16, c3_w_u16, H, W, oc, mcr_mod=None) -> torch.Tensor:
    """Full re8 hop: rn3 chain(x1b) -> rnm 1x1(concat(chain,x2b)) -> c3 3x3, fused.

    x1b      : [G,G,ic2] bf16 HWC, the chain input (the gemm_pair x1 half).
    x2b      : [G,G,ic2] bf16 HWC, the RepNCSP concat's other half (gemm_pair x2).
    chain_pairs: [(w1_u16, w2_u16)] x n_iters bottleneck weight pairs (the chain).
    rnm_w_u16: fuse_bn(repncsp.conv3)  flat uint16 [oc*ic + 2*oc] (rnm 1x1 + BN).
    c3_w_u16 : fuse_bn(layer.conv2[1]/conv3[1]) flat uint16 [oc*ic*9 + 2*oc] (c3 3x3 + BN).
    Returns [G,G,oc] bf16 = c3 output (BN+SiLU applied), matching the model's
    `x3 = rt(mc_c3, ...)` to within BFP-quant tolerance.
    """
    global _MCR
    if mcr_mod is not None:
        _MCR = mcr_mod
    mcr = _MCR
    geo = _GEO_BY_SHAPE.get((H, oc))
    assert H == W and geo is not None, (
        f"chain_rnm_c3 fused path supports {_GEO_BY_SHAPE} shapes; got H={H} W={W} oc={oc}")
    n_iters = len(chain_pairs)
    assert n_iters == _N_ITERS, f"chain_rnm_c3 built for n_iters={_N_ITERS}, got {n_iters}"

    reg = _ensure_registered(geo, n_iters)
    elf_name = reg["elf_name"]
    geo_p, dmeta, hmeta = reg["meta"]

    ic2, G = geo_p["IC"], geo_p["GBOUND"]
    nt = geo_p["WORKER_TILES"][0]
    IMG = dmeta["IMG"]
    IMG_H = dmeta["IMG_H"]          # chain image height (tall for re6, == IMG for re8)
    HALF_ELEMS = dmeta["HALF_ELEMS"]
    IN_ELEMS = dmeta["IN_ELEMS"]
    SEAM_ELEMS = dmeta["IMG_ELEMS"]
    ic = dmeta["ic"]
    N_TILES, C_ELEMS = hmeta["N_TILES"], hmeta["C_ELEMS"]
    gbound = G

    entry = mcr._get_merged_kernel(elf_name)
    device, _elf, _ctx, kernel = entry

    # ---- chain input A: x1b PAD(2)-padded into lower half of widened stacked BO ----
    # The chain half is the tall (IMG_H, IMG) padded image; x2 stacks above it.
    x1b_u16 = x1b.contiguous().view(torch.uint16).numpy()  # [G,G,ic2]
    a_u16 = np.zeros(IN_ELEMS, np.uint16)
    a_u16[:HALF_ELEMS] = _pad_half(x1b_u16, IMG, ic2, G, img_h=IMG_H)
    # ---- chain output B: copy of A + x2b PAD(2)-padded into upper half ----
    x2b_u16 = x2b.contiguous().view(torch.uint16).numpy()
    b_u16 = a_u16.copy()
    b_u16[HALF_ELEMS:IN_ELEMS] = _pad_half(x2b_u16, IMG, ic2, G, img_h=IMG_H)

    # ---- weights: cache the (slow) packs by weight identity ----
    wkey = (tuple(id(a) for pr in chain_pairs for a in pr),
            id(rnm_w_u16), id(c3_w_u16))
    cached = _WT_CACHE.get(wkey)
    if cached is None:
        chain_wt = np.concatenate([
            np.tile(_pack_geo_iter(w1, w2, ic2, geo_p["WSLOT"], geo_p["N_BLK"]), nt)
            for w1, w2 in chain_pairs]).astype(np.uint16)
        rnm_conv = _u16_to_bf16_t(np.asarray(rnm_w_u16, np.uint16)[:oc * ic]).reshape(oc, ic)
        rnm_bnw = _u16_to_bf16_t(np.asarray(rnm_w_u16, np.uint16)[oc * ic:oc * ic + oc])
        rnm_bnb = _u16_to_bf16_t(np.asarray(rnm_w_u16, np.uint16)[oc * ic + oc:oc * ic + 2 * oc])
        gemm_wt_u16 = _pack_weights_blocked(rnm_conv, rnm_bnw, rnm_bnb)
        halo_wt_u16 = _c3_weights_for_halo(np.asarray(c3_w_u16, np.uint16), oc, ic)
        cached = (chain_wt, gemm_wt_u16, halo_wt_u16)
        _WT_CACHE[wkey] = cached
    chain_wt, gemm_wt_u16, halo_wt_u16 = cached

    out_elems = N_TILES * C_ELEMS
    # @main args (links (0,2,1,0),(1,2,2,0)):
    #   0=A(chain in) 1=WT(chain) 2=B(chain out=dcg in) 3=gemm_wt
    #   4=seam(dcg out=halo in) 5=halo_wt 6=halo_out
    #
    # MUTABLE per-call (shared pool, re-filled every dispatch):
    #   a/b = the stacked x1b|x2 image; out = the result stream.
    a_bo = mcr._get_merged_bo(device, elf_name, "a", a_u16.nbytes)
    b_bo = mcr._get_merged_bo(device, elf_name, "b", b_u16.nbytes)
    out_bo = mcr._get_merged_bo(device, elf_name, "out", out_elems * 4)
    mcr._xrt_fill_bo(a_bo, a_u16)
    mcr._xrt_fill_bo(b_bo, b_u16)
    if _RESIDENT_WT:
        # STATIC per-hop (resident pool keyed by wkey, filled once):
        #   wt/gwt/hwt = frame-invariant packed weights; seam = poison PAD border.
        wt_bo = _resident_wt_bo(device, wkey, "wt", chain_wt.nbytes)
        gwt_bo = _resident_wt_bo(device, wkey, "gwt", gemm_wt_u16.nbytes)
        seam_bo = _resident_wt_bo(device, wkey, "seam", SEAM_ELEMS * 2)
        hwt_bo = _resident_wt_bo(device, wkey, "hwt", halo_wt_u16.nbytes)
        if wkey not in _RESIDENT_FILLED:
            # First dispatch for this hop: fill+sync its static weight + seam BOs.
            # Later frames reuse the device-resident copies (no host->device DMA).
            mcr._xrt_fill_bo(wt_bo, chain_wt)
            mcr._xrt_fill_bo(gwt_bo, gemm_wt_u16)
            mcr._xrt_fill_bo(seam_bo, np.zeros(SEAM_ELEMS, np.uint16))  # poison border
            mcr._xrt_fill_bo(hwt_bo, halo_wt_u16)
            _RESIDENT_FILLED.add(wkey)
    else:
        # A/B baseline: old behavior — shared per-(elf,role) pool, re-filled every
        # dispatch (the +36 ms weight-reupload tax this experiment removes).
        wt_bo = mcr._get_merged_bo(device, elf_name, "wt", chain_wt.nbytes)
        gwt_bo = mcr._get_merged_bo(device, elf_name, "gwt", gemm_wt_u16.nbytes)
        seam_bo = mcr._get_merged_bo(device, elf_name, "seam", SEAM_ELEMS * 2)
        hwt_bo = mcr._get_merged_bo(device, elf_name, "hwt", halo_wt_u16.nbytes)
        mcr._xrt_fill_bo(wt_bo, chain_wt)
        mcr._xrt_fill_bo(gwt_bo, gemm_wt_u16)
        mcr._xrt_fill_bo(seam_bo, np.zeros(SEAM_ELEMS, np.uint16))
        mcr._xrt_fill_bo(hwt_bo, halo_wt_u16)
    mcr._xrt_run_kernel(kernel, [a_bo, wt_bo, b_bo, gwt_bo, seam_bo, hwt_bo, out_bo])

    import pyxrt as _xrt
    out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, out_elems * 4, 0)
    out_f32 = np.frombuffer(out_bo.map(), dtype=np.float32, count=out_elems).copy()

    # ---- deinterleave + untile -> [G,G,oc] HWC (vectorized via cached index) ----
    # The kernel already applied BN + SiLU on the f32 accumulator (un-scaled
    # weights), so out_f32 IS the final activation — no host epilogue / weight fold.
    idx = _build_untile_index(hmeta, gbound)
    got = out_f32[idx].reshape(gbound, gbound, oc)
    return torch.from_numpy(got).to(torch.bfloat16)
