"""Multicore run_tiled_fused_conv — drops in for the single-core version.

Uses 32-core xclbins to process spatial tiles in parallel.
Falls back to single-core if the multicore xclbin is not available.
"""
import math
import os, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))
import torch
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime
from regime_config import conv_regime_for_layer

# Import single-core helpers
import importlib.util
_base = os.path.dirname(__file__)
spec1 = importlib.util.spec_from_file_location('ett', os.path.join(_base, '_full_model_helpers', 'elan_test_tiled.py'))
ett = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ett)
extract_patch = ett.extract_patch
bf16_to_uint16 = ett.bf16_to_uint16
uint16_to_bf16 = ett.uint16_to_bf16

N_CORES = 32
# MDV6_BUILD_DIR (set by lit) routes merged ELFs out of the source tree
# so test runs never see "already built, skipping" from stale artifacts.
_MDV6_BUILD_ROOT = os.environ.get("MDV6_BUILD_DIR")

# Per-layer ppc bumps from mlir-aie-0pf-B1 (2026-04-18).
#
# The rule that makes ppc>1 worthwhile: bump only when
#     n_tiles > N_CORES × prior_ppc
# so that `calls_per_ocb = ceil(n_tiles / (N_CORES * ppc))` actually drops.
# Otherwise the core runs `ppc` padded iterations per call with the same
# number of calls — a strict regression.
#
# Table (n_tiles = tiles_h × tiles_w; calls_per_ocb at ppc=1/2/4):
#   mc_ftconv0    256 tiles   8/4/2     (ppc=4 build overflows L2; keeping p2 option)
#   mc_ftconv1    196 tiles   7/4/2  →  bumped to p2 (p4 overflows L2)
#   mc_elan_c3    400 tiles  13/7/4  →  bumped to p4
#   mc_aconv3     100 tiles   4/2/1     (p4 overflows L2; p2 option exists)
#   mc_aconv5     100 tiles   4/2/1  →  bumped to p4
#   mc_aconv7      25 tiles   1/1/1     (no benefit — tiles fit in one call at any ppc)
#   mc_aconv16     25 tiles   1/1/1     (no benefit)
#   mc_aconv19      9 tiles   1/1/1     (no benefit)
#   mc_re4_c3      49 tiles   2/1/1  →  bumped to p2
#   mc_re4_rn3    100 tiles   4/2/1  →  bumped to p4
#   mc_re6_c3      25 tiles   1/1/1     (no benefit)
#   mc_re6_rn3     25 tiles   1/1/1     (no benefit; earlier p2 was neutral-at-best)
#   mc_re8_c3      25 tiles   1/1/1     (no benefit)
#   mc_re8_rn3      9 tiles   1/1/1     (no benefit)
#
# Verified via launch-count reduction in --profile sweep (deterministic, unlike
# wall time which is noisy on this machine). Each bump drops launches as the
# table predicts; no-benefit layers were previously whitelisted at ppc=2 and
# have been removed from _MC_PPC (they made every call do 2× padded work).
_MC_PPC = {
    "mc_ftconv1":  2,   # 196 tiles: 7 → 4 calls per ocb
    "mc_elan_c3":  4,   # 400 tiles: 13 → 4
    "mc_aconv5":   4,   # 100 tiles: 4 → 1
    "mc_re4_c3":   2,   # 49 tiles: 2 → 1
    "mc_re4_rn3":  4,   # 100 tiles: 4 → 1
}

# Input sub-FIFO depth (L1 ping-pong). depth=2 lets the memtile pre-fetch
# patch N+1 into the free L1 slot while the core computes patch N. Costs
# +patch_size bytes per core in L1.
#
# Tested on mc_elan_c3 (2026-04-18): no measurable wall-time change
# (wall -12 ms / -0.6% within noise). Reason: every conv path layer in
# aie2_multicore.py is conv3x3 with AI 140-340 — compute-bound, so DMA
# was already hidden by compute. The mechanism is retained here in case
# a future DMA-bound conv layer lands (unlikely: all conv1x1 routes
# through the GEMM path).
#
# Variant filename convention: {base}_p{ppc}_d{depth}.
_MC_INPUT_DEPTH = {}

# Pack caches — bead mlir-aie-d6f. Keyed by (id(weights_uint16), params).
# fuse_bn in elan/test_tiled.py uses a WeakKeyDictionary on Module so its uint16
# arrays live and die with their Module — keeping ids stable while modules are
# alive, but allowing cross-frame model recreation. We additionally include
# expected_len in the key and verify on hit (mlir-aie-woi guard) so a recycled
# id cannot silently return blocks for a different layer's weights.
_WTBLOCK_CACHE_3x3 = {}   # (id(wts_u16), ocb, oc_block, out_ch, C, ks, expected_len) -> np.ndarray
_GEMM_OCB_CACHE = {}      # (id(wts_u16), ic, oc, oc_block, expected_len) -> list[np.ndarray]
_GEMM_KB_CACHE = {}       # (id(wts_u16), ic, oc, k_block, expected_len) -> np.ndarray


# Cap chosen so the full MDV6 working set (~300 unique (wts_id, ocb) tuples)
# fits without eviction. Cache entries are id-keyed and the source modules
# are pinned by the test driver / video-stream loop, so the dict size is
# bounded by the model graph itself — not by the number of frames. The cap
# is a defence against pathological growth (e.g. someone instantiating a
# fresh model per frame), not a steady-state throttle.
_GEMM_CACHE_MAX = 1024


def _gemm_cache_evict_dead_ids(cache):
    """Bound cache size by dropping the oldest entry on overflow.

    The expected_len field in the key + on-hit length verification catches
    stale-id collisions from recycled Python ids; this just keeps the dict
    from growing without bound across many frames.
    """
    while len(cache) > _GEMM_CACHE_MAX:
        cache.pop(next(iter(cache)))


# XRT buffer pool — bead mlir-aie-0pf sub-task A. The full model fires ~1500
# iron.tensor/iron.zeros allocations per warm frame (130–150 µs each = ~180 ms
# total). Buffer dimensions repeat across calls, so we keep one pinned buffer
# per (role, size, dtype) and overwrite it in place. DefaultNPURuntime.run is
# synchronous — buffers can be safely reused once it returns.
#
# Inputs/outputs change every call (activations flow through the model), so
# they stay size-keyed and refill per call. Weights are different: the same
# packed array object is fed back every frame because the upstream packers
# (_pack_3x3_weights, _repack_weights_for_gemm, _repack_weights_kblocked,
# fuse_bn) cache by id(weights_uint16). _WEIGHT_BUF_CACHE keys on id(arr) so
# repeat calls hit a pre-filled+pre-synced buffer with no host copy/sync.
#
# Separate pools per role: a single run() call may pass different buffers of
# the same size as input + weight + output (e.g., 1×1 conv with ic == oc).
# Aliasing input and output to the same XRT buffer hangs the kernel.
_INPUT_POOL = {}    # (size, dtype.kind, itemsize) -> iron.Tensor
_OUTPUT_POOL = {}
_WEIGHT_BUF_CACHE = {}  # (id(arr), size, kind) -> (iron.Tensor, arr_strongref)


def _pool_key(size, dtype):
    dt = np.dtype(dtype)
    return (size, dt.kind, dt.itemsize)


def _pooled_buf(pool, size, dtype):
    key = _pool_key(size, dtype)
    buf = pool.get(key)
    if buf is None:
        buf = iron.zeros(size, dtype=dtype)
        pool[key] = buf
    return buf


def _fill_and_sync(buf, arr):
    """Write arr into buf's host-mapped memory and sync to device.

    Bypasses XRTTensor.numpy() because it would unconditionally
    sync_from_device first (overwriting our pending write with stale device
    data) and never sync back. We write to .data directly and call
    _sync_to_device() ourselves.
    """
    buf.data.reshape(-1)[:] = arr.ravel()
    buf._sync_to_device()


def get_in_buf(arr):
    """Return the pooled XRT input buffer initialised with arr's contents."""
    buf = _pooled_buf(_INPUT_POOL, arr.size, arr.dtype)
    _fill_and_sync(buf, arr)
    return buf


def get_wt_buf(arr):
    """Return an XRT weight buffer pre-filled with arr.

    First call for a given array identity uploads + syncs the bytes; later
    calls (e.g. the same packed weights re-fed on the next frame) return the
    same handle with no host work. The strong ref to arr in the cache entry
    pins its id, so we can safely key on id(arr) without recycling collisions.
    """
    key = (id(arr), arr.size, np.dtype(arr.dtype).kind)
    entry = _WEIGHT_BUF_CACHE.get(key)
    if entry is not None:
        return entry[0]
    buf = iron.zeros(arr.size, dtype=arr.dtype)
    _fill_and_sync(buf, arr)
    _WEIGHT_BUF_CACHE[key] = (buf, arr)
    return buf


def get_out_buf(size, dtype=np.uint16):
    """Return the pooled XRT output buffer (contents undefined; kernel overwrites)."""
    return _pooled_buf(_OUTPUT_POOL, size, dtype)


# ----------------------------------------------------------------------
# Merged-ELF dispatch (Phase 3-lite: batch loop collapse).
#
# `merged_ftconv0_x8.elf` holds 8 clones of mc_ftconv0 in one ELF with a
# dispatcher whose runtime_sequence is (wt, in_0, out_0, ..., in_7, out_7)
# = 17 args. Replacing the 8 per-batch launches that mc_ftconv0 fires today
# with one merged launch saves ~7 × launch_gap floor per frame.
#
# Notes on coexistence with the xclbin path:
#   - xrt.elf-based kernels use xrt.hw_context + xrt.ext.kernel; they live in
#     a different runtime path than NPUKernel/DefaultNPURuntime (which loads
#     .xclbin via xrt.xclbin). The two contexts can coexist in one process
#     but each xclbin/elf occupies one of the 32 XRT context slots.
#   - When mc_ftconv0 is dispatched through the merged ELF, the standalone
#     mc_ftconv0.xclbin is never loaded, so net cache pressure is unchanged.
# USE_MERGED_KERNELS gate removed (no alternative path remains).

if _MDV6_BUILD_ROOT:
    _MERGED_BD = os.path.abspath(os.path.join(_MDV6_BUILD_ROOT, "build_merged"))
else:
    _MERGED_BD = os.path.join(_base, "conv", "build_merged")
_MERGED_KERNELS = OrderedDict()  # elf_name -> (device, elf, hw_context, kernel) or None
_MERGED_BO_POOL = {}             # (elf_name, role, size) -> xrt.ext.bo
_USE_PACKED_GEMM = os.environ.get("MDV6_USE_PACKED_GEMM", "0") not in ("", "0", "false", "False")
_DEFAULT_MAX_LIVE_CONTEXTS = "30" if _USE_PACKED_GEMM else "1000000"
_MERGED_MAX_LIVE_CONTEXTS = int(os.environ.get("MDV6_MAX_LIVE_MERGED_CONTEXTS", _DEFAULT_MAX_LIVE_CONTEXTS))


def _drop_merged_kernel(elf_name):
    """Release one cached xrt.elf hw_context and any BOs tied to that ELF."""
    _MERGED_KERNELS.pop(elf_name, None)
    for key in list(_MERGED_BO_POOL.keys()):
        if key[0] == elf_name:
            _MERGED_BO_POOL.pop(key, None)
    # Cached run objects may reference this ELF's kernel/BOs; rare path, so
    # just flush the whole pool.
    _RUN_CACHE.clear()
    _RUN_CACHE_REFS.clear()


def _trim_merged_kernel_cache():
    """Keep live xrt.elf hw_contexts below the driver/context limit.

    The full mdv6 frame can touch more unique merged ELFs than the XRT driver
    allows to remain open simultaneously. Profile runs are sequential, so an
    LRU cache preserves common reuse while releasing stale hw_contexts before
    CREATE_HWCTX starts failing late in the model.
    """
    live = sum(1 for entry in _MERGED_KERNELS.values() if entry is not None)
    while live > _MERGED_MAX_LIVE_CONTEXTS:
        for old_name, old_entry in list(_MERGED_KERNELS.items()):
            if old_entry is None:
                _MERGED_KERNELS.pop(old_name, None)
                continue
            _drop_merged_kernel(old_name)
            live -= 1
            break
        else:
            break


def _get_merged_kernel(elf_name):
    """Load a merged ELF kernel (cached). Returns (device, kernel) or None."""
    if elf_name in _MERGED_KERNELS:
        entry = _MERGED_KERNELS.pop(elf_name)
        _MERGED_KERNELS[elf_name] = entry
        return entry
    elf_path = os.path.join(_MERGED_BD, f"{elf_name}.elf")
    if not os.path.exists(elf_path):
        _MERGED_KERNELS[elf_name] = None
        return None
    import pyxrt as _xrt
    _trim_merged_kernel_cache()
    device = _xrt.device(0)
    elf = _xrt.elf(elf_path)
    hw_context = _xrt.hw_context(device, elf)
    kernel = _xrt.ext.kernel(hw_context, "main")
    entry = (device, elf, hw_context, kernel)
    _MERGED_KERNELS[elf_name] = entry
    _trim_merged_kernel_cache()
    return entry


# Per-layer attribution: set by run_tiled_fused_conv_mc / run_gemm_conv1x1_mc
# at entry and cleared at exit. _xrt_run_kernel and the harness's wrap can read
# this to attribute NPU time + launch_gap to a specific model layer.
_CURRENT_LAYER = None

# xrt.run object pool: (id(kernel), id(arg0), ...) -> prepared run. BOs are
# pooled per (elf, role, size), so for a given dispatch site the arg tuple is
# stable across frames and the run object (including set_arg state) can be
# reused. _RUN_CACHE_REFS pins kernel + arg BOs so dict ids stay valid.
_RUN_CACHE = {}
_RUN_CACHE_REFS = {}


def _xrt_run_kernel(kernel, args):
    """One xrt.run launch: set_arg per positional, start(), wait2().

    Hookable seam for profile_harness — wrapping this counts merged-ELF
    dispatches into the same npu_run / launch_gap buckets that the standalone
    DefaultNPURuntime.run path uses. Module-level so the harness can rebind
    it without touching each merged dispatch site.
    """
    import pyxrt as _xrt
    key = (id(kernel),) + tuple(id(a) for a in args)
    run = _RUN_CACHE.get(key)
    if run is None:
        run = _xrt.run(kernel)
        for i, a in enumerate(args):
            run.set_arg(i, a)
        _RUN_CACHE[key] = run
        # keep arg BOs alive as long as the cached run object
        _RUN_CACHE_REFS[key] = (kernel, args)
    run.start()
    run.wait2()
    return run


def _get_merged_bo(device, elf_name, role, nbytes):
    """Per-(elf, role) BO pool. Reused across calls so warm-frame allocations
    collapse to a dict lookup, same idea as _INPUT_POOL/_OUTPUT_POOL."""
    key = (elf_name, role, nbytes)
    bo = _MERGED_BO_POOL.get(key)
    if bo is not None:
        return bo
    import pyxrt as _xrt
    bo = _xrt.ext.bo(device, nbytes)
    _MERGED_BO_POOL[key] = bo
    return bo


def _xrt_fill_bo(bo, arr):
    import pyxrt as _xrt
    mv = bo.map()
    np.copyto(
        np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes),
        np.frombuffer(arr, dtype=np.uint8),
        casting="no",
    )
    # Sync only the bytes written, not the whole (pooled, possibly larger) BO.
    bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, arr.nbytes, 0)


# Per-layer merged-ELF registry. Maps mc_name → (elf_name, n_batches).
# An entry implies: this layer's batch loop can be collapsed into a single
# merged ELF launch with 1 shared wt + n_batches × (in, out) args.
_MERGED_LAYERS_ALL = {
    # Layer-active-name → (ELF base, n_batches per OCB). Keys are post-_MC_PPC
    # (e.g. mc_ftconv1 is bumped to mc_ftconv1_p2, so the merged ELF is named
    # for the active variant). One ELF call replaces n_batches DefaultNPURuntime
    # launches per OCB; the host loops over n_ocb to cover the layer's output.
    #
    # Multi-clone (batch-fanout) ELFs — collapse n_batches launches per OCB
    # into one xrt.run. Built by conv/build_merged.py + invocations in tests.
    "mc_ftconv0":      ("merged_ftconv0_x8",       8),
    "mc_ftconv1_p2":   ("merged_ftconv1_p2_x4",    4),
    "mc_elan_c3_p4":   ("merged_elan_c3_p4_x4",    4),
    "mc_aconv3":       ("merged_aconv3_x4",        4),
    "mc_aconv16":      ("merged_aconv16_x4",       4),
    # Single-clone ELFs (Phase A.1, mlir-aie-mi7 Phase A). Same launch count
    # as standalone, but dispatched through xrt.elf+xrt.run so no xclbin
    # context is allocated. Built by conv/build_x1_mc.py for every other 3x3
    # variant the model dispatches.
    "mc_aconv5_p4":    ("merged_aconv5_p4_x1",     1),
    "mc_aconv7":       ("merged_aconv7_x1",        1),
    "mc_aconv19":      ("merged_aconv19_x1",       1),
    "mc_re4_c3_p2":    ("merged_re4_c3_p2_x1",     1),
    "mc_re4_rn3_p4":   ("merged_re4_rn3_p4_x1",    1),
    "mc_re6_c3":       ("merged_re6_c3_x1",        1),
    "mc_re6_rn3":      ("merged_re6_rn3_x1",       1),
    "mc_re8_c3":       ("merged_re8_c3_x1",        1),
    "mc_re8_rn3":      ("merged_re8_rn3_x1",       1),
}

# Comma-sep allowlist of mc_names. Empty/"all" = enable everything in the
# registry. Useful for bisecting which merged layers regress when enabled.
_merged_enabled = os.environ.get("MERGED_LAYERS", "all").strip()
if _merged_enabled in ("", "all"):
    _MERGED_LAYERS = dict(_MERGED_LAYERS_ALL)
else:
    _allowed = {s.strip() for s in _merged_enabled.split(",") if s.strip()}
    _MERGED_LAYERS = {k: v for k, v in _MERGED_LAYERS_ALL.items() if k in _allowed}


def _regime_conv_artifact(mc_name, actual_name, ppc):
    artifact = conv_regime_for_layer(mc_name)
    if artifact is None:
        return None
    active = artifact.members[mc_name]
    active_tile_h, active_tile_w, active_ic, active_oc, active_stride, active_padding, active_ppc = active
    if active_ppc != ppc:
        raise RuntimeError(
            f"Regime artifact ppc mismatch for {mc_name}: runtime ppc={ppc}, "
            f"artifact ppc={active_ppc}"
        )
    if artifact.patches_per_core < ppc:
        raise RuntimeError(
            f"Regime artifact ppc envelope too small for {mc_name}: "
            f"runtime ppc={ppc}, envelope ppc={artifact.patches_per_core}"
        )
    if active_ic != artifact.ic and active_ic > artifact.ic:
        raise RuntimeError(f"Regime artifact IC envelope too small for {mc_name}")
    if active_oc > artifact.oc_block:
        raise RuntimeError(f"Regime artifact OC envelope too small for {mc_name}")
    patch_h = (artifact.tile_h - 1) * artifact.stride + artifact.kernel_size
    patch_w = (artifact.tile_w - 1) * artifact.stride + artifact.kernel_size
    patch_size_raw = patch_h * patch_w * artifact.ic
    patch_size = patch_size_raw + (patch_size_raw % 2)
    return {
        "xclbin_name": artifact.xclbin_name,
        "insts_name": f"{artifact.xclbin_name}_{mc_name}",
        "active_tile_h": active_tile_h,
        "active_tile_w": active_tile_w,
        "active_ic": active_ic,
        "active_oc": active_oc,
        "active_stride": active_stride,
        "active_padding": active_padding,
        "ppc": artifact.patches_per_core,
        "tile_h": artifact.tile_h,
        "tile_w": artifact.tile_w,
        "ic": artifact.ic,
        "oc_block": artifact.oc_block,
        "patch_size": patch_size,
        "weight_size": artifact.oc_block * artifact.ic * artifact.kernel_size * artifact.kernel_size
                       + 2 * artifact.oc_block,
        "output_tile_size": artifact.tile_h * artifact.tile_w * artifact.oc_block,
    }


def _handle_exists(name, insts_name=None):
    """Check whether a merged ELF (in any of OCB / single-clone / fanout
    forms) exists for this variant. _get_mc_variant uses this to pick
    between a base name and its _p{ppc} suffix variant.
    """
    if name in _MERGED_LAYERS_OCB:
        return True
    if name in _MERGED_LAYERS:
        # Confirm the ELF file is actually on disk; the registry entry
        # alone isn't enough (build might be missing).
        elf_name = _MERGED_LAYERS[name][0]
        return os.path.exists(os.path.join(_MERGED_BD, f"{elf_name}.elf"))
    return False


def _get_mc_variant(name):
    """Prefer a batched multicore variant when available.

    Variant name: {base}[_p{ppc}][_d{depth}] where ppc>1 and/or depth>1.
    Falls back to lower ppc / depth=1 if the chosen variant isn't built.
    """
    ppc = _MC_PPC.get(name, 1)
    depth = _MC_INPUT_DEPTH.get(name, 1)
    variant = name
    if ppc > 1:
        variant = f"{variant}_p{ppc}"
    if depth > 1:
        variant = f"{variant}_d{depth}"
    if variant != name and not _handle_exists(variant):
        # Back off the depth suffix first, then the ppc suffix.
        if depth > 1:
            alt = f"{name}_p{ppc}" if ppc > 1 else name
            if _handle_exists(alt):
                return alt, ppc
        return name, 1
    return variant, ppc


def run_tiled_fused_conv_mc(mc_name, sc_name, input_hwc, weights_uint16,
                             out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                             stride=1, kernel_size=3, padding=1):
    """Multicore tiled fused conv.

    Fails hard if the MC xclbin is missing from disk — silent SC fallback
    would hide build-graph bugs by running on a different weight layout.
    Retries once on transient XRT execution errors (stale/evicted handle).

    Args:
        mc_name: multicore xclbin name (e.g., 'mc_re4_c1')
        sc_name: unused in current flow; retained for signature compatibility.
    """
    global _CURRENT_LAYER
    _prev_layer = _CURRENT_LAYER
    _CURRENT_LAYER = mc_name
    try:
        return _run_tiled_fused_conv_mc_impl(
            mc_name, sc_name, input_hwc, weights_uint16,
            out_h, out_w, out_ch, tile_h, tile_w, oc_block,
            stride, kernel_size, padding,
        )
    finally:
        _CURRENT_LAYER = _prev_layer


def _run_tiled_fused_conv_mc_impl(mc_name, sc_name, input_hwc, weights_uint16,
                                   out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                   stride=1, kernel_size=3, padding=1):
    actual_name, ppc = _get_mc_variant(mc_name)

    # OCB-unroll merged-ELF fast path (Phase E): one xrt.run covers all OCBs
    # AND all spatial batches. Takes precedence over the per-OCB merged path
    # for layers registered in _MERGED_LAYERS_OCB. Forces the regime's active
    # config (tile/oc_block) since the OCB-unroll ELF is built against those
    # values. effective_ppc from the registry overrides regime_ppc inside the
    # OCB dispatch (the ELF was built with effective_ppc baked in).
    if actual_name in _MERGED_LAYERS_OCB:
        elf_name, n_ocb, effective_ppc, ocb_oc_block = _MERGED_LAYERS_OCB[actual_name]
        merged = _get_merged_kernel(elf_name)
        if merged is not None:
            regime = _regime_conv_artifact(mc_name, actual_name, ppc)
            if regime is not None:
                ocb_tile_h = regime["active_tile_h"]
                ocb_tile_w = regime["active_tile_w"]
                ocb_stride = regime["active_stride"]
                ocb_padding = regime["active_padding"]
            else:
                ocb_tile_h, ocb_tile_w = tile_h, tile_w
                ocb_stride, ocb_padding = stride, padding
            # oc_block comes from the registry (must match the ELF's built
            # oc_block, NOT necessarily regime active_oc — see re4_rn3).
            return _run_tiled_mc_inner_ocb_merged(
                merged, elf_name, n_ocb, effective_ppc,
                input_hwc, weights_uint16,
                out_h, out_w, out_ch, ocb_tile_h, ocb_tile_w, ocb_oc_block,
                ocb_stride, kernel_size, ocb_padding,
            )

    # Single-clone (or batch-fanout) merged-ELF path. One xrt.run dispatches
    # n_batches sub-runs internally; the host still loops over OCBs.
    if actual_name in _MERGED_LAYERS:
        elf_name, n_batches = _MERGED_LAYERS[actual_name]
        merged = _get_merged_kernel(elf_name)
        if merged is not None:
            return _run_tiled_mc_inner_merged(
                merged, elf_name, n_batches, ppc,
                input_hwc, weights_uint16,
                out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                stride, kernel_size, padding,
            )
    # No merged ELF for this layer. The legacy xclbin fall-through was
    # removed in the Phase G+ cleanup — every conv layer must have a merged
    # ELF in _MERGED_LAYERS_OCB or _MERGED_LAYERS.
    raise RuntimeError(
        f"No merged ELF found for {mc_name} (actual_name={actual_name}). "
        f"Build the merged ELFs first: conv/build_x1_mc.py + "
        f"conv/build_ocb.py --layer all + conv/build_pair_rn1.py."
    )


def _pack_3x3_weights(conv_block_u16, oc_block, ic):
    """Repack OIHW [oc_block, ic, 3, 3] bf16 (as uint16) to vectorized layout
    [oc_block/8, ic/8, 9, 8ic, 8oc] for aie::mmul<4,8,8>.

    Both oc_block and ic must be multiples of 8.
    """
    w_f = uint16_to_bf16(conv_block_u16).reshape(oc_block, ic, 9)
    oc_blks = oc_block // 8
    ic_blks = ic // 8
    w_f = w_f.reshape(oc_blks, 8, ic_blks, 8, 9)
    # Permute (oc_blk=0, 8_oc=1, ic_blk=2, 8_ic=3, 9=4)
    #  → (oc_blk=0, ic_blk=2, 9=4, 8_ic=3, 8_oc=1)
    w_blocked = w_f.permute(0, 2, 4, 3, 1).contiguous()
    return bf16_to_uint16(w_blocked.flatten())




def _run_tiled_mc_inner_merged(merged_entry, elf_name, n_batches, ppc,
                                input_hwc, weights_uint16,
                                out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                stride, kernel_size, padding):
    """Single-XRT-call-per-OCB merged dispatch (Phase 3-lite).

    Each ELF holds n_batches clones of one kernel config with a shared wt
    arg. For a layer with M output channel blocks, the host calls the ELF M
    times (once per OCB) with that OCB's weights and the same set of input
    patches. Compared to standalone:
      standalone: n_ocb × n_batches DefaultNPURuntime.run calls
      merged:     n_ocb            xrt.run calls
    so each OCB saves (n_batches - 1) launches' worth of pyxrt plumbing.

    Patch extraction is done once across all OCBs (input is constant per
    layer call). Weight packing reuses _WTBLOCK_CACHE_3x3 so warm frames
    are pure cache hits.
    """
    import pyxrt as _xrt
    device, _elf, _ctx, kernel = merged_entry

    H, W, C = input_hwc.shape
    tiles_h = (out_h + tile_h - 1) // tile_h
    tiles_w = (out_w + tile_w - 1) // tile_w
    n_oc_blocks = (out_ch + oc_block - 1) // oc_block
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)

    patch_h = (tile_h - 1) * stride + kernel_size
    patch_w = (tile_w - 1) * stride + kernel_size
    patch_size_raw = patch_h * patch_w * C
    patch_size = patch_size_raw + (patch_size_raw % 2)
    output_tile_size = tile_h * tile_w * oc_block
    active_conv_wt_size = oc_block * C * kernel_size * kernel_size
    weight_slot_size = active_conv_wt_size + 2 * oc_block

    total_conv_wts = out_ch * C * kernel_size * kernel_size
    all_conv_wts = weights_uint16[:total_conv_wts]
    all_bn_w = weights_uint16[total_conv_wts:total_conv_wts + out_ch]
    all_bn_b = weights_uint16[total_conv_wts + out_ch:total_conv_wts + 2 * out_ch]
    wts_id = id(weights_uint16)
    expected_wts_len = total_conv_wts + 2 * out_ch

    # Patch extraction (same as standalone path) — done once, reused across OCBs.
    all_patches = []
    all_coords = []
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            patch = extract_patch(input_hwc, tr, tc, tile_h, tile_w,
                                   stride, kernel_size, padding)
            patch_u16 = bf16_to_uint16(patch.flatten())
            if len(patch_u16) < patch_size:
                patch_u16 = np.pad(patch_u16, (0, patch_size - len(patch_u16)))
            all_patches.append(patch_u16)
            all_coords.append((tr, tc))

    patches_per_call = N_CORES * ppc
    expected_batches = (len(all_patches) + patches_per_call - 1) // patches_per_call
    if expected_batches != n_batches:
        raise RuntimeError(
            f"merged dispatch for {elf_name}: expected {n_batches} batches, "
            f"computed {expected_batches} from {len(all_patches)} patches"
        )

    # Pack input: group patches by core × ppc slot, padding incomplete trailing
    # calls with slot-0 data (same convention as _run_tiled_mc_inner).
    inputs_per_batch = []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * patches_per_call
        batch_end = min(batch_start + patches_per_call, len(all_patches))
        batch_patches = list(all_patches[batch_start:batch_end])
        while len(batch_patches) < patches_per_call:
            batch_patches.append(batch_patches[0])
        per_core_batches = []
        for core in range(N_CORES):
            core_start = core * ppc
            core_end = core_start + ppc
            per_core_batches.append(np.concatenate(batch_patches[core_start:core_end]))
        inputs_per_batch.append(np.concatenate(per_core_batches))

    output_per_batch = N_CORES * ppc * output_tile_size

    wt_bo = _get_merged_bo(device, elf_name, "wt", weight_slot_size * 2)
    in_bos = [
        _get_merged_bo(device, elf_name, f"in_{i}", inputs_per_batch[i].nbytes)
        for i in range(n_batches)
    ]
    out_bos = [
        _get_merged_bo(device, elf_name, f"out_{i}", output_per_batch * 2)
        for i in range(n_batches)
    ]

    # Inputs are the same across every OCB; fill them once.
    for i in range(n_batches):
        _xrt_fill_bo(in_bos[i], inputs_per_batch[i])

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start

        # Reuse the standalone-path cache: pack/keep wt per (id, ocb, shape).
        wt_key = (wts_id, ocb, oc_block, out_ch, C, kernel_size,
                  expected_wts_len, weight_slot_size)
        wt_block = (_WTBLOCK_CACHE_3x3.get(wt_key)
                    if len(weights_uint16) == expected_wts_len else None)
        if wt_block is None:
            cw_per_oc = C * kernel_size * kernel_size
            conv_block = all_conv_wts[oc_start * cw_per_oc:oc_end * cw_per_oc]
            if actual_oc < oc_block:
                conv_block = np.pad(conv_block, (0, (oc_block - actual_oc) * cw_per_oc))
            if kernel_size == 3:
                conv_block = _pack_3x3_weights(conv_block, oc_block, C)
            bn_w_block = all_bn_w[oc_start:oc_end]
            bn_b_block = all_bn_b[oc_start:oc_end]
            wt_block = np.concatenate([conv_block, bn_w_block, bn_b_block])
            if len(wt_block) < weight_slot_size:
                wt_block = np.pad(wt_block, (0, weight_slot_size - len(wt_block)))
            _WTBLOCK_CACHE_3x3[wt_key] = wt_block
            _gemm_cache_evict_dead_ids(_WTBLOCK_CACHE_3x3)

        _xrt_fill_bo(wt_bo, wt_block)

        args = [wt_bo]
        for i in range(n_batches):
            args.append(in_bos[i])
            args.append(out_bos[i])
        _xrt_run_kernel(kernel, args)

        for batch_idx in range(n_batches):
            out_bos[batch_idx].sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            out_data = np.frombuffer(
                out_bos[batch_idx].map(), dtype=np.uint16, count=output_per_batch
            )
            # one bf16 conversion per batch instead of one per tile
            out_f = uint16_to_bf16(out_data)
            batch_start = batch_idx * patches_per_call
            batch_end = min(batch_start + patches_per_call, len(all_patches))
            for j in range(batch_end - batch_start):
                tr, tc = all_coords[batch_start + j]
                oh_s = tr * tile_h; ow_s = tc * tile_w
                oh_e = min(oh_s + tile_h, out_h)
                ow_e = min(ow_s + tile_w, out_w)
                core = j // ppc
                slot = j % ppc
                start = (core * ppc + slot) * output_tile_size
                tile_out = out_f[start:start + output_tile_size]
                tile_out = tile_out.reshape(tile_h, tile_w, oc_block)
                output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                    tile_out[:oh_e - oh_s, :ow_e - ow_s, :actual_oc]

    return output


# Per-layer OCB-unroll merged ELF registry (Phase E/F/G). Maps mc_name →
# (elf_name, n_ocb, effective_ppc, oc_block). The OCB-unrolled ELF packs
# all n_ocb OCBs AND all spatial batches into a single xrt.run via
# compile-time-unrolled runtime sequence with strided weight/output TAPs.
# effective_ppc = regime_ppc × n_spatial_batches: the kernel processes
# that many patches per core per worker invocation (compile-time-unrolled).
# oc_block must match the value baked into the ELF (= total_oc / n_ocb);
# the dispatch uses this instead of regime active_oc when they differ
# (e.g. re4_rn3 builds at oc_block=32 to keep n_ocb=1, regime says 16).
#
# Examples:
#   re8_rn3: regime_ppc=1, 25 spatial patches < 32 cores → n_spatial_batches=1,
#            effective_ppc=1. OCB collapse 4→1, total work 4×.
#   re6_rn3: regime_ppc=1, 100 spatial patches → n_spatial_batches=4,
#            effective_ppc=4. OCB collapse 3→1 + spatial 4→1, total 12×.
_MERGED_LAYERS_OCB_ALL = {
    # ---- rn3 layers (Phase E) ----
    "mc_re8_rn3":      ("ocb_re8_rn3_x1",     4, 1,  16),
    "mc_re6_rn3":      ("ocb_re6_rn3_x1",     3, 4,  16),
    # mc_re4_rn3 NOT registered: caller passes oc_block=32 (full OC) so
    # legacy n_ocb=1 already → no OCB dispatches to collapse. The Phase E
    # "step 3" reported saving was an artifact of a build/regime mismatch
    # where the kernel computed only half the output channels (correct
    # version is identical to legacy in wall time, see PHASE_G_NOTES).
    # ---- c3 layers (Phase F) ----
    "mc_re8_c3":       ("ocb_re8_c3_x1",      8, 1,  16),
    "mc_re6_c3":       ("ocb_re6_c3_x1",      6, 4,  16),
    # mc_re4_c3 is variant-selected to mc_re4_c3_p2 (ppc=2 from _MC_PPC).
    "mc_re4_c3_p2":    ("ocb_re4_c3_x1",      4, 16, 16),
    # ---- stride-2 aconv layers (Phase G) ----
    # aconv3/aconv16 today use merged_aconv*_x4 (4-clone spatial fanout);
    # OCB-unroll absorbs spatial via ppc and collapses OCB iterations into
    # one xrt.run. aconv7/aconv19 use single-sub merged today (x1).
    # aconv5 skipped — too many OCB iterations to fit in unroll budget.
    "mc_aconv3":       ("ocb_aconv3_x1",      8, 4,  16),
    "mc_aconv7":       ("ocb_aconv7_x1",      16, 1, 16),
    "mc_aconv16":      ("ocb_aconv16_x1",     6, 4,  16),
    "mc_aconv19":      ("ocb_aconv19_x1",     16, 1, 8),
}

_merged_ocb_enabled = os.environ.get("MERGED_OCB", "1").strip()
if _merged_ocb_enabled in ("", "all", "1"):
    _MERGED_LAYERS_OCB = dict(_MERGED_LAYERS_OCB_ALL)
elif _merged_ocb_enabled in ("0", "off", "none"):
    _MERGED_LAYERS_OCB = {}
else:
    _allowed_ocb = {s.strip() for s in _merged_ocb_enabled.split(",") if s.strip()}
    _MERGED_LAYERS_OCB = {k: v for k, v in _MERGED_LAYERS_OCB_ALL.items() if k in _allowed_ocb}


def _run_tiled_mc_inner_ocb_merged(merged_entry, elf_name, n_ocb, ppc,
                                    input_hwc, weights_uint16,
                                    out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                    stride, kernel_size, padding):
    """OCB-unrolled merged dispatch (Phase E).

    One xrt.run processes all n_ocb output channel blocks AND all spatial
    batches. Host pre-concatenates per-OCB weight slots into one big BO;
    the kernel's runtime sequence has the OCB loop unrolled at compile
    time with strided memtile TAPs serving each OCB its weight/output
    slice. The `ppc` argument is the EFFECTIVE patches_per_core baked
    into the ELF (= regime_ppc × n_spatial_batches), which absorbs
    spatial batching into the same xrt.run via the existing patches_per_core
    compile-time unroll mechanism.
    """
    import pyxrt as _xrt
    device, _elf, _ctx, kernel = merged_entry

    H, W, C = input_hwc.shape
    tiles_h = (out_h + tile_h - 1) // tile_h
    tiles_w = (out_w + tile_w - 1) // tile_w
    n_oc_blocks = (out_ch + oc_block - 1) // oc_block
    if n_oc_blocks != n_ocb:
        raise RuntimeError(
            f"OCB-unroll {elf_name}: registered n_ocb={n_ocb} but layer needs "
            f"{n_oc_blocks} OCBs (out_ch={out_ch}, oc_block={oc_block})")

    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)

    patch_h = (tile_h - 1) * stride + kernel_size
    patch_w = (tile_w - 1) * stride + kernel_size
    patch_size_raw = patch_h * patch_w * C
    patch_size = patch_size_raw + (patch_size_raw % 2)
    output_tile_size = tile_h * tile_w * oc_block
    active_conv_wt_size = oc_block * C * kernel_size * kernel_size
    weight_slot_size = active_conv_wt_size + 2 * oc_block

    total_conv_wts = out_ch * C * kernel_size * kernel_size
    all_conv_wts = weights_uint16[:total_conv_wts]
    all_bn_w = weights_uint16[total_conv_wts:total_conv_wts + out_ch]
    all_bn_b = weights_uint16[total_conv_wts + out_ch:total_conv_wts + 2 * out_ch]
    wts_id = id(weights_uint16)
    expected_wts_len = total_conv_wts + 2 * out_ch

    # Patch extraction — single spatial batch per OCB.
    all_patches = []
    all_coords = []
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            patch = extract_patch(input_hwc, tr, tc, tile_h, tile_w,
                                   stride, kernel_size, padding)
            patch_u16 = bf16_to_uint16(patch.flatten())
            if len(patch_u16) < patch_size:
                patch_u16 = np.pad(patch_u16, (0, patch_size - len(patch_u16)))
            all_patches.append(patch_u16)
            all_coords.append((tr, tc))

    patches_per_call = N_CORES * ppc
    if len(all_patches) > patches_per_call:
        raise RuntimeError(
            f"OCB-unroll {elf_name}: layer needs {len(all_patches)} patches > "
            f"{patches_per_call} (cores*effective_ppc={ppc}); ELF built with "
            f"insufficient effective_ppc to absorb all spatial batches")

    # Pack single input batch, padding incomplete trailing slots with slot-0.
    batch_patches = list(all_patches)
    while len(batch_patches) < patches_per_call:
        batch_patches.append(batch_patches[0])
    per_core_batches = []
    for core in range(N_CORES):
        core_start = core * ppc
        core_end = core_start + ppc
        per_core_batches.append(np.concatenate(batch_patches[core_start:core_end]))
    input_concat = np.concatenate(per_core_batches)

    output_per_batch = N_CORES * ppc * output_tile_size

    # Pre-pack per-OCB weights, concatenate into big_wt.
    wt_blocks = []
    for ocb in range(n_ocb):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start

        wt_key = (wts_id, ocb, oc_block, out_ch, C, kernel_size,
                  expected_wts_len, weight_slot_size)
        wt_block = (_WTBLOCK_CACHE_3x3.get(wt_key)
                    if len(weights_uint16) == expected_wts_len else None)
        if wt_block is None:
            cw_per_oc = C * kernel_size * kernel_size
            conv_block = all_conv_wts[oc_start * cw_per_oc:oc_end * cw_per_oc]
            if actual_oc < oc_block:
                conv_block = np.pad(conv_block, (0, (oc_block - actual_oc) * cw_per_oc))
            if kernel_size == 3:
                conv_block = _pack_3x3_weights(conv_block, oc_block, C)
            bn_w_block = all_bn_w[oc_start:oc_end]
            bn_b_block = all_bn_b[oc_start:oc_end]
            wt_block = np.concatenate([conv_block, bn_w_block, bn_b_block])
            if len(wt_block) < weight_slot_size:
                wt_block = np.pad(wt_block, (0, weight_slot_size - len(wt_block)))
            _WTBLOCK_CACHE_3x3[wt_key] = wt_block
            _gemm_cache_evict_dead_ids(_WTBLOCK_CACHE_3x3)
        wt_blocks.append(wt_block)

    big_wt = np.concatenate(wt_blocks)
    big_out_nelem = n_ocb * output_per_batch

    big_wt_bo = _get_merged_bo(device, elf_name, "big_wt", big_wt.nbytes)
    in_bo = _get_merged_bo(device, elf_name, "in_0", input_concat.nbytes)
    big_out_bo = _get_merged_bo(device, elf_name, "big_out", big_out_nelem * 2)

    _xrt_fill_bo(in_bo, input_concat)
    _xrt_fill_bo(big_wt_bo, big_wt)

    # Single dispatch — args follow merged-x1 share_arg_idxs={1} convention:
    # arg0=wt(big), arg1=in, arg2=out(big).
    _xrt_run_kernel(kernel, [big_wt_bo, in_bo, big_out_bo])

    big_out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    big_out_data = np.frombuffer(
        big_out_bo.map(), dtype=np.uint16, count=big_out_nelem
    ).copy()

    # Unpack tile-by-tile per OCB.
    for ocb in range(n_ocb):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start
        ocb_base = ocb * output_per_batch
        for j in range(len(all_patches)):
            tr, tc = all_coords[j]
            oh_s = tr * tile_h; ow_s = tc * tile_w
            oh_e = min(oh_s + tile_h, out_h)
            ow_e = min(ow_s + tile_w, out_w)
            core = j // ppc
            slot = j % ppc
            start = ocb_base + (core * ppc + slot) * output_tile_size
            tile_out = uint16_to_bf16(big_out_data[start:start + output_tile_size])
            tile_out = tile_out.reshape(tile_h, tile_w, oc_block)
            output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                tile_out[:oh_e - oh_s, :ow_e - ow_s, :actual_oc]

    return output


# ---------------------------------------------------------------------------
# GEMM Conv1x1 — vectorized 1×1 conv using mmul<4,8,8>
# ---------------------------------------------------------------------------

# Import fuse_bn_transposed for weight repacking
fuse_bn_transposed = ett.fuse_bn_transposed

# L1 budget for GEMM conv1x1 (depth=1, stack=8KB, RTP=32B reserved).
# Must stay in sync with the AVAIL constant in aie2_gemm_conv1x1.py.
_GEMM_L1 = 65536 - 8192 - 32


def _gemm_tile_m(ic, oc_block):
    """Max tile_m (mult of 4) that fits in L1."""
    wt_bytes = (ic * oc_block + 2 * oc_block) * 2
    remaining = _GEMM_L1 - wt_bytes
    if remaining <= 0:
        return 0
    return (remaining // ((ic + oc_block) * 2) // 4) * 4


def _gemm_tile_m_kblocked(ic, oc, k_block):
    """Max tile_m (mult of 4) for K-blocked config."""
    wt_chunk_bytes = (k_block * oc + 2 * oc) * 2
    remaining = _GEMM_L1 - wt_chunk_bytes
    if remaining <= 0:
        return 0
    return (remaining // ((ic + oc) * 2) // 4) * 4


def _gemm_choose_oc_block(ic, oc):
    """Choose largest oc_block that fits with >=16 tile_m."""
    for ob in [oc, 128, 64, 48, 32, 16]:
        if ob > oc or oc % ob != 0:
            continue
        if _gemm_tile_m(ic, ob) >= 16:
            return ob
    return None


_MAX_K_BLOCKS = 16
_XRT_BUF_MAX = 16 * 1024 * 1024
_L2_BUDGET = 400 * 1024


def _gemm_choose_k_block(ic, oc, M):
    """Choose k_block for K-blocking. Returns (k_block, tile_m) or (0, tile_m)."""
    # First try non-K-blocked
    tm_full = _gemm_tile_m(ic, oc)
    if tm_full >= 16:
        return 0, min(tm_full, 256)
    # K-blocked: minimize spatial calls with cap on K-blocks
    best_kb, best_tm, best_calls = 0, 0, float('inf')
    for n_kb in range(2, _MAX_K_BLOCKS + 1):
        kb = ic // n_kb
        if kb < 8 or kb % 8 != 0 or ic % kb != 0:
            continue
        tm = _gemm_tile_m_kblocked(ic, oc, kb)
        tm = min(tm, 256)
        if tm < 4:
            continue
        calls = math.ceil(M / (tm * N_CORES))
        if calls < best_calls or (calls == best_calls and n_kb < ic // best_kb):
            best_kb, best_tm, best_calls = kb, tm, calls
    return best_kb, best_tm


def _gemm_compute_ppc(M, tile_m, ic, oc_block):
    """Compute optimal patches_per_core to minimize NPU calls."""
    ideal = math.ceil(M / (32 * tile_m))
    in_bytes = 32 * tile_m * ic * 2
    out_bytes = 32 * tile_m * oc_block * 2
    max_xrt_in = _XRT_BUF_MAX // in_bytes if in_bytes > 0 else 999
    max_xrt_out = _XRT_BUF_MAX // out_bytes if out_bytes > 0 else 999
    col_in = 4 * tile_m * ic * 2
    col_out = 4 * tile_m * oc_block * 2
    wt = (ic * oc_block + 2 * oc_block) * 2
    per_ppc = col_in + col_out
    max_l2 = (_L2_BUDGET - wt) // per_ppc if per_ppc > 0 else 999
    return max(1, min(ideal, max_xrt_in, max_xrt_out, max_l2, 32))


def _gemm_compute_ppc_kblocked(M, tile_m, ic, oc, k_block):
    """Compute ppc for K-blocked config."""
    ideal = math.ceil(M / (N_CORES * tile_m))
    in_bytes = N_CORES * tile_m * ic * 2
    out_bytes = N_CORES * tile_m * oc * 2
    max_xrt_in = _XRT_BUF_MAX // in_bytes if in_bytes > 0 else 999
    max_xrt_out = _XRT_BUF_MAX // out_bytes if out_bytes > 0 else 999
    col_in = 4 * tile_m * ic * 2
    col_out = 4 * tile_m * oc * 2
    wt = (k_block * oc + 2 * oc) * 2
    per_ppc = col_in + col_out
    max_l2 = (_L2_BUDGET - wt) // per_ppc if per_ppc > 0 else 999
    return max(1, min(ideal, max_xrt_in, max_xrt_out, max_l2, 32))


# Legacy xclbin loaders (_load_gemm_handle, _get_gemm_handle) and the
# regime-artifact lookups (_regime_gemm_artifact, _regime_gemm_kblocked_artifact)
# were removed in the Phase G+ cleanup. Every GEMM dispatch now resolves a
# merged ELF via _merged_gemm_elf_name + _get_merged_kernel.


def _repack_weights_for_gemm(weights_uint16, ic, oc, oc_block):
    """Repack flat [OC,IC] + BN weights to GEMM blocked layout [ic/8,oc_block/8,8,8].

    Input:  weights_uint16 = [conv_wts(OC*IC), bn_w(OC), bn_b(OC)] (flat OIHW)
    Output: per oc_block: [blocked_wts(ic*oc_block), bn_w(oc_block), bn_b(oc_block)]
    """
    # mlir-aie-d6f cache + mlir-aie-woi guard: id() may be reused across frames
    # after the source array is freed. Verify size on hit so a stale id never
    # silently returns blocks for a different layer.
    expected_len = oc * ic + 2 * oc
    cache_key = (id(weights_uint16), ic, oc, oc_block, expected_len)
    cached = _GEMM_OCB_CACHE.get(cache_key)
    if cached is not None and len(weights_uint16) == expected_len:
        return cached

    total_conv = oc * ic
    if len(weights_uint16) < total_conv + 2 * oc:
        raise ValueError(
            f"_repack_weights_for_gemm: weights_uint16 len={len(weights_uint16)} "
            f"too small for ic={ic} oc={oc} (need {total_conv + 2 * oc})"
        )
    all_conv = weights_uint16[:total_conv]
    all_bn_w = weights_uint16[total_conv:total_conv + oc]
    all_bn_b = weights_uint16[total_conv + oc:total_conv + 2 * oc]

    blocks = []
    for ocb_start in range(0, oc, oc_block):
        ocb_end = min(ocb_start + oc_block, oc)
        actual_ob = ocb_end - ocb_start

        # Extract per-oc-block conv weights: rows [ocb_start:ocb_end] of [OC, IC]
        conv_block = np.zeros(oc_block * ic, dtype=np.uint16)
        for o in range(actual_ob):
            src_start = (ocb_start + o) * ic
            conv_block[o * ic:(o + 1) * ic] = all_conv[src_start:src_start + ic]

        # Reshape to blocked layout [ic/8, oc_block/8, 8ic, 8oc]
        w = conv_block.reshape(oc_block, ic)  # [oc_block, ic] as uint16
        # View as bf16 for transpose, then back to uint16
        w_f = uint16_to_bf16(w.flatten()).reshape(oc_block, ic)
        ic_blks = ic // 8
        ob_blks = oc_block // 8
        w_blocked = w_f.reshape(ob_blks, 8, ic_blks, 8)
        w_blocked = w_blocked.permute(2, 0, 3, 1).contiguous()  # [ic/8, oc/8, 8ic, 8oc]
        blocked_u16 = bf16_to_uint16(w_blocked.flatten())

        # BN params
        bn_w_block = np.zeros(oc_block, dtype=np.uint16)
        bn_b_block = np.zeros(oc_block, dtype=np.uint16)
        bn_w_block[:actual_ob] = all_bn_w[ocb_start:ocb_end]
        bn_b_block[:actual_ob] = all_bn_b[ocb_start:ocb_end]

        blocks.append(np.concatenate([blocked_u16, bn_w_block, bn_b_block]))

    _GEMM_OCB_CACHE[cache_key] = blocks
    _gemm_cache_evict_dead_ids(_GEMM_OCB_CACHE)
    return blocks


def _repack_weights_kblocked(weights_uint16, ic, oc, k_block):
    """Repack flat [OC,IC] + BN weights to K-blocked layout.

    Input:  weights_uint16 = [conv_wts(OC*IC), bn_w(OC), bn_b(OC)] (flat OIHW)
    Output: single buffer [chunk_0, chunk_1, ..., chunk_{n_k_blocks-1}]
    Each chunk: [k_block/8, oc/8, 8ic, 8oc, bn_w(oc), bn_b(oc)]
    """
    # mlir-aie-d6f cache + mlir-aie-woi guard.
    expected_len = oc * ic + 2 * oc
    cache_key = (id(weights_uint16), ic, oc, k_block, expected_len)
    cached = _GEMM_KB_CACHE.get(cache_key)
    if cached is not None and len(weights_uint16) == expected_len:
        return cached
    if len(weights_uint16) < expected_len:
        raise ValueError(
            f"_repack_weights_kblocked: weights_uint16 len={len(weights_uint16)} "
            f"too small for ic={ic} oc={oc} (need {expected_len})"
        )

    total_conv = oc * ic
    all_conv = weights_uint16[:total_conv]
    all_bn_w = weights_uint16[total_conv:total_conv + oc]
    all_bn_b = weights_uint16[total_conv + oc:total_conv + 2 * oc]

    n_k_blocks = ic // k_block
    oc_blks = oc // 8
    chunks = []

    for kb_idx in range(n_k_blocks):
        k_start = kb_idx * k_block
        kb_blks = k_block // 8

        # Extract conv weights for this K-block: [oc, k_block] from [oc, ic]
        # Original layout is [OC, IC] row-major
        w_slice = np.zeros(oc * k_block, dtype=np.uint16)
        for o in range(oc):
            src = all_conv[o * ic + k_start:o * ic + k_start + k_block]
            w_slice[o * k_block:o * k_block + k_block] = src

        # Reshape to blocked layout [k_block/8, oc/8, 8ic, 8oc]
        w_f = uint16_to_bf16(w_slice).reshape(oc, k_block)
        w_blocked = w_f.reshape(oc_blks, 8, kb_blks, 8)
        w_blocked = w_blocked.permute(2, 0, 3, 1).contiguous()  # [kb/8, oc/8, 8ic, 8oc]
        blocked_u16 = bf16_to_uint16(w_blocked.flatten())

        # Append BN params to every chunk (kernel only reads on last K-block)
        chunks.append(np.concatenate([blocked_u16, all_bn_w.copy(), all_bn_b.copy()]))

    out = np.concatenate(chunks)
    _GEMM_KB_CACHE[cache_key] = out
    _gemm_cache_evict_dead_ids(_GEMM_KB_CACHE)
    return out


def _repack_weights_kblocked_regime(weights_uint16, ic, oc, env_ic, env_oc, env_k_block):
    """Repack K-blocked weights into a padded regime envelope.

    The regime kernel sees full_ic=env_ic, oc=env_oc, k_block=env_k_block.
    Logical weights outside ic/oc are zero, and BN params are appended to every
    chunk so the envelope's final chunk can apply the active layer's BN.
    """
    expected_len = oc * ic + 2 * oc
    cache_key = (id(weights_uint16), ic, oc, env_ic, env_oc, env_k_block, expected_len)
    cached = _GEMM_KB_CACHE.get(cache_key)
    if cached is not None and len(weights_uint16) == expected_len:
        return cached
    if len(weights_uint16) < expected_len:
        raise ValueError(
            f"_repack_weights_kblocked_regime: weights_uint16 len={len(weights_uint16)} "
            f"too small for ic={ic} oc={oc} (need {expected_len})"
        )

    total_conv = oc * ic
    all_conv = weights_uint16[:total_conv]
    all_bn_w = weights_uint16[total_conv:total_conv + oc]
    all_bn_b = weights_uint16[total_conv + oc:total_conv + 2 * oc]

    n_k_blocks = env_ic // env_k_block
    oc_blks = env_oc // 8
    chunks = []

    for kb_idx in range(n_k_blocks):
        k_start = kb_idx * env_k_block
        kb_blks = env_k_block // 8
        w_slice = np.zeros(env_oc * env_k_block, dtype=np.uint16)

        active_k = max(0, min(env_k_block, ic - k_start))
        if active_k > 0:
            for o in range(oc):
                src = all_conv[o * ic + k_start:o * ic + k_start + active_k]
                dst = o * env_k_block
                w_slice[dst:dst + active_k] = src

        w_f = uint16_to_bf16(w_slice).reshape(env_oc, env_k_block)
        w_blocked = w_f.reshape(oc_blks, 8, kb_blks, 8)
        w_blocked = w_blocked.permute(2, 0, 3, 1).contiguous()
        blocked_u16 = bf16_to_uint16(w_blocked.flatten())

        bn_w_block = np.zeros(env_oc, dtype=np.uint16)
        bn_b_block = np.zeros(env_oc, dtype=np.uint16)
        bn_w_block[:oc] = all_bn_w
        bn_b_block[:oc] = all_bn_b
        chunks.append(np.concatenate([blocked_u16, bn_w_block, bn_b_block]))

    out = np.concatenate(chunks)
    _GEMM_KB_CACHE[cache_key] = out
    _gemm_cache_evict_dead_ids(_GEMM_KB_CACHE)
    return out


def _merged_gemm_elf_name(tile_m, ic, oc, k_block, ppc):
    """Filename convention shared with conv/build_x1_gemm.py."""
    kb_str = f"kb{k_block}_" if k_block > 0 else ""
    return f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_{kb_str}p{ppc}_x1"


def _merged_gemm_packed_elf_name(tile_m, ic, oc, k_block, ppc, n_batches):
    """Filename convention shared with conv/build_packed_gemm.py."""
    kb_str = f"kb{k_block}_" if k_block > 0 else ""
    return f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_{kb_str}p{ppc}_x{n_batches}_packed"


def _pack_gemm_spatial_inputs(input_flat, total_slots, input_size, tile_m, pixels_per_call):
    """Pack all GEMM spatial batches into concat(old_x1_host_in_batch_i).

    input_flat is [M, IC] uint16/bf16 bits. Each x1 batch layout is the same
    as _run_gemm_*_merged used: total_slots slots, each slot contains tile_m
    pixels flattened. Incomplete trailing slots are padded with slot0.
    Returns (packed_host_in_u16, n_batches).
    """
    M = int(input_flat.shape[0])
    n_batches = int(math.ceil(M / pixels_per_call)) if M else 0
    host_in_size = total_slots * input_size
    packed = np.zeros(n_batches * host_in_size, dtype=np.uint16)
    for batch_idx, batch_start in enumerate(range(0, M, pixels_per_call)):
        batch_end = min(batch_start + pixels_per_call, M)
        batch_pixels = batch_end - batch_start
        n_active_slots = (batch_pixels + tile_m - 1) // tile_m
        base = batch_idx * host_in_size
        for s in range(n_active_slots):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            active = input_flat[pix_start:pix_end].reshape(-1)
            if isinstance(active, np.ndarray) and active.dtype == np.uint16:
                active_u16 = active
            else:
                active_u16 = bf16_to_uint16(active)
            dst = base + s * input_size
            packed[dst:dst + len(active_u16)] = active_u16
        slot0 = packed[base:base + input_size].copy()
        for s in range(n_active_slots, total_slots):
            dst = base + s * input_size
            packed[dst:dst + input_size] = slot0
    return packed, n_batches


def run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, weights_uint16,
                         out_h, out_w, out_ch, oc_block=None):
    """GEMM-based 1×1 conv with 32-core multicore.

    Tries K-blocked path first (no OC blocking), falls back to OC-blocked,
    then to scalar MC.
    """
    global _CURRENT_LAYER
    _prev_layer = _CURRENT_LAYER
    _CURRENT_LAYER = gemm_name
    try:
        return _run_gemm_conv1x1_mc_impl(
            gemm_name, sc_name, input_hwc, weights_uint16,
            out_h, out_w, out_ch, oc_block,
        )
    finally:
        _CURRENT_LAYER = _prev_layer


def _run_gemm_conv1x1_mc_impl(gemm_name, sc_name, input_hwc, weights_uint16,
                               out_h, out_w, out_ch, oc_block=None):
    H, W, IC = input_hwc.shape
    M = H * W

    # --- Try K-blocked path first ---
    k_block, tile_m_kb = _gemm_choose_k_block(IC, out_ch, M)
    if _USE_PACKED_GEMM and k_block > 0 and tile_m_kb >= 4:
        ppc = _gemm_compute_ppc_kblocked(M, tile_m_kb, IC, out_ch, k_block)
        n_batches = int(math.ceil(M / (N_CORES * tile_m_kb * ppc)))
        if n_batches > 1:
            packed_elf_name = _merged_gemm_packed_elf_name(tile_m_kb, IC, out_ch, k_block, ppc, n_batches)
            packed_merged = _get_merged_kernel(packed_elf_name)
            if packed_merged is not None:
                return _run_gemm_kblocked_packed_merged(
                    packed_merged, packed_elf_name, input_hwc, weights_uint16,
                    out_h, out_w, out_ch, tile_m_kb, k_block, ppc,
                )
    if k_block > 0 and tile_m_kb >= 4:
        ppc = _gemm_compute_ppc_kblocked(M, tile_m_kb, IC, out_ch, k_block)
        elf_name = _merged_gemm_elf_name(tile_m_kb, IC, out_ch, k_block, ppc)
        merged = _get_merged_kernel(elf_name)
        if merged is not None:
            return _run_gemm_kblocked_merged(
                merged, elf_name, input_hwc, weights_uint16,
                out_h, out_w, out_ch, tile_m_kb, k_block, ppc,
            )
        raise RuntimeError(
            f"GEMM K-blocked merged ELF missing: {elf_name}.elf "
            f"(layer={gemm_name}, IC={IC}, OC={out_ch}, M={M}, k_block={k_block}, ppc={ppc}). "
            f"Build with conv/build_x1_gemm.py."
        )

    # --- OC-blocked path ---
    if oc_block is None:
        oc_block = _gemm_choose_oc_block(IC, out_ch)
    if oc_block is None:
        raise RuntimeError(
            f"GEMM oc_block selection failed (layer={gemm_name}, IC={IC}, OC={out_ch}, M={M})"
        )

    tile_m = min(_gemm_tile_m(IC, oc_block), 256)
    ppc = _gemm_compute_ppc(M, tile_m, IC, oc_block)

    # The ELFs were built with k_block=0 for layers where the full IC fits
    # in L1; they match the OC-blocked kernel exactly when oc_block == OC
    # (the only case build_x1_gemm.py emits).
    if _USE_PACKED_GEMM and oc_block == out_ch:
        n_batches = int(math.ceil(M / (N_CORES * tile_m * ppc)))
        if n_batches > 1:
            packed_elf_name = _merged_gemm_packed_elf_name(tile_m, IC, out_ch, 0, ppc, n_batches)
            packed_merged = _get_merged_kernel(packed_elf_name)
            if packed_merged is not None:
                return _run_gemm_oc_blocked_packed_merged(
                    packed_merged, packed_elf_name, input_hwc, weights_uint16,
                    out_h, out_w, out_ch, tile_m, ppc,
                )
    if oc_block == out_ch:
        elf_name = _merged_gemm_elf_name(tile_m, IC, out_ch, 0, ppc)
        merged = _get_merged_kernel(elf_name)
        if merged is not None:
            return _run_gemm_oc_blocked_merged(
                merged, elf_name, input_hwc, weights_uint16,
                out_h, out_w, out_ch, tile_m, ppc,
            )
    raise RuntimeError(
        f"GEMM OC-blocked merged ELF missing for layer={gemm_name} "
        f"(IC={IC}, OC={out_ch}, M={M}, oc_block={oc_block}, tile_m={tile_m}, ppc={ppc}). "
        f"Build with conv/build_x1_gemm.py."
    )



def _run_gemm_packed_merged_common(merged_entry, elf_name, input_hwc, wt_u16,
                                   out_h, out_w, out_ch, tile_m, ppc):
    """Run packed single-dispatch GEMM spatial fanout ABI: [W, packed_I, packed_O]."""
    import pyxrt as _xrt
    device, _elf, _ctx, kernel = merged_entry

    H, W, IC = input_hwc.shape
    M = H * W
    input_size = tile_m * IC
    output_size = tile_m * out_ch
    total_slots = N_CORES * ppc
    pixels_per_call = N_CORES * tile_m * ppc
    host_in_size = total_slots * input_size
    host_out_size = total_slots * output_size

    input_flat = input_hwc.reshape(M, IC)
    packed_in, n_batches = _pack_gemm_spatial_inputs(
        input_flat, total_slots, input_size, tile_m, pixels_per_call,
    )
    if n_batches <= 0:
        return torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)

    packed_out_nelem = n_batches * host_out_size
    wt_bo = _get_merged_bo(device, elf_name, "wt", wt_u16.nbytes)
    in_bo = _get_merged_bo(device, elf_name, "packed_in", packed_in.nbytes)
    out_bo = _get_merged_bo(device, elf_name, "packed_out", packed_out_nelem * 2)

    _xrt_fill_bo(wt_bo, wt_u16)
    _xrt_fill_bo(in_bo, packed_in)
    _xrt_run_kernel(kernel, [wt_bo, in_bo, out_bo])
    out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    packed_out = np.frombuffer(out_bo.map(), dtype=np.uint16,
                               count=packed_out_nelem).copy()

    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    output_flat = output.reshape(M, out_ch)
    for batch_idx, batch_start in enumerate(range(0, M, pixels_per_call)):
        batch_end = min(batch_start + pixels_per_call, M)
        batch_pixels = batch_end - batch_start
        n_active_slots = (batch_pixels + tile_m - 1) // tile_m
        base = batch_idx * host_out_size
        for s in range(min(n_active_slots, total_slots)):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            if pix_start >= batch_end:
                break
            n_pix = pix_end - pix_start
            start = base + s * output_size
            tile_out = uint16_to_bf16(packed_out[start:start + n_pix * out_ch])
            tile_out = tile_out.reshape(n_pix, out_ch)
            output_flat[pix_start:pix_end, :] = tile_out.to(torch.bfloat16)
    return output


def _run_gemm_kblocked_packed_merged(merged_entry, elf_name, input_hwc, weights_uint16,
                                      out_h, out_w, out_ch, tile_m, k_block, ppc):
    H, W, IC = input_hwc.shape
    wt_kblocked = _repack_weights_kblocked(weights_uint16, IC, out_ch, k_block)
    return _run_gemm_packed_merged_common(
        merged_entry, elf_name, input_hwc, wt_kblocked,
        out_h, out_w, out_ch, tile_m, ppc,
    )


def _run_gemm_oc_blocked_packed_merged(merged_entry, elf_name, input_hwc, weights_uint16,
                                        out_h, out_w, out_ch, tile_m, ppc):
    H, W, IC = input_hwc.shape
    wt_blocks = _repack_weights_for_gemm(weights_uint16, IC, out_ch, out_ch)
    return _run_gemm_packed_merged_common(
        merged_entry, elf_name, input_hwc, wt_blocks[0],
        out_h, out_w, out_ch, tile_m, ppc,
    )


def _run_gemm_kblocked_merged(merged_entry, elf_name, input_hwc, weights_uint16,
                               out_h, out_w, out_ch, tile_m, k_block, ppc):
    """Merged-ELF variant of _run_gemm_kblocked.

    Per-batch xrt.run instead of DefaultNPURuntime.run. No regime envelope:
    the ELF encodes (tile_m, IC, OC, k_block, ppc) exactly, so env_*==active_*.
    Weight, input, output use the same packing as the standalone path (the BO
    pool keyed by (elf_name, role) means warm frames reuse pinned buffers).
    """
    import pyxrt as _xrt
    device, _elf, _ctx, kernel = merged_entry

    H, W, IC = input_hwc.shape
    M = H * W

    input_size = tile_m * IC
    output_size = tile_m * out_ch
    wt_kblocked = _repack_weights_kblocked(weights_uint16, IC, out_ch, k_block)

    pixels_per_call = N_CORES * tile_m * ppc
    total_slots = N_CORES * ppc

    wt_bo = _get_merged_bo(device, elf_name, "wt", wt_kblocked.nbytes)
    in_bo = _get_merged_bo(device, elf_name, "in", total_slots * input_size * 2)
    out_bo = _get_merged_bo(device, elf_name, "out", total_slots * output_size * 2)
    _xrt_fill_bo(wt_bo, wt_kblocked)

    input_flat = input_hwc.reshape(M, IC)
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    output_flat = output.reshape(M, out_ch)

    for batch_start in range(0, M, pixels_per_call):
        batch_end = min(batch_start + pixels_per_call, M)
        batch_pixels = batch_end - batch_start

        host_in = np.zeros(total_slots * input_size, dtype=np.uint16)
        n_active_slots = (batch_pixels + tile_m - 1) // tile_m
        for s in range(n_active_slots):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            active_u16 = bf16_to_uint16(input_flat[pix_start:pix_end].flatten())
            dst = s * input_size
            host_in[dst:dst + len(active_u16)] = active_u16
        # Pad trailing slots with slot 0 (multicore_padding feedback).
        slot0 = host_in[:input_size]
        for s in range(n_active_slots, total_slots):
            host_in[s * input_size:(s + 1) * input_size] = slot0

        _xrt_fill_bo(in_bo, host_in)
        _xrt_run_kernel(kernel, [wt_bo, in_bo, out_bo])
        out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out_data = np.frombuffer(
            out_bo.map(), dtype=np.uint16, count=total_slots * output_size
        ).copy()

        for s in range(min(n_active_slots, total_slots)):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            if pix_start >= batch_end:
                break
            n_pix = pix_end - pix_start
            start = s * output_size
            tile_out = uint16_to_bf16(out_data[start:start + n_pix * out_ch])
            tile_out = tile_out.reshape(n_pix, out_ch)
            output_flat[pix_start:pix_end, :] = tile_out.to(torch.bfloat16)

    return output


def _run_gemm_oc_blocked_merged(merged_entry, elf_name, input_hwc, weights_uint16,
                                 out_h, out_w, out_ch, tile_m, ppc):
    """Merged-ELF variant for layers where the full IC*OC weight fits in L1
    (k_block=0). Single OCB (oc_block == out_ch) is the only shape
    build_x1_gemm.py emits; multi-OCB layers would need a per-OCB loop here.
    """
    import pyxrt as _xrt
    device, _elf, _ctx, kernel = merged_entry

    H, W, IC = input_hwc.shape
    M = H * W
    input_size = tile_m * IC
    output_size = tile_m * out_ch

    # Reuse the existing OC-blocked weight repacker; oc_block=out_ch gives a
    # single block matching the ELF's wt buffer.
    wt_blocks = _repack_weights_for_gemm(weights_uint16, IC, out_ch, out_ch)
    wt_block = wt_blocks[0]

    pixels_per_call = N_CORES * tile_m * ppc
    total_slots = N_CORES * ppc

    wt_bo = _get_merged_bo(device, elf_name, "wt", wt_block.nbytes)
    in_bo = _get_merged_bo(device, elf_name, "in", total_slots * input_size * 2)
    out_bo = _get_merged_bo(device, elf_name, "out", total_slots * output_size * 2)
    _xrt_fill_bo(wt_bo, wt_block)

    input_flat = input_hwc.reshape(M, IC)
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    output_flat = output.reshape(M, out_ch)

    for batch_start in range(0, M, pixels_per_call):
        batch_end = min(batch_start + pixels_per_call, M)
        batch_pixels = batch_end - batch_start

        host_in = np.zeros(total_slots * input_size, dtype=np.uint16)
        n_active_slots = (batch_pixels + tile_m - 1) // tile_m
        for s in range(n_active_slots):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            active_u16 = bf16_to_uint16(input_flat[pix_start:pix_end].flatten())
            dst = s * input_size
            host_in[dst:dst + len(active_u16)] = active_u16
        slot0 = host_in[:input_size]
        for s in range(n_active_slots, total_slots):
            host_in[s * input_size:(s + 1) * input_size] = slot0

        _xrt_fill_bo(in_bo, host_in)
        _xrt_run_kernel(kernel, [wt_bo, in_bo, out_bo])
        out_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out_data = np.frombuffer(
            out_bo.map(), dtype=np.uint16, count=total_slots * output_size
        ).copy()

        for s in range(min(n_active_slots, total_slots)):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            if pix_start >= batch_end:
                break
            n_pix = pix_end - pix_start
            start = s * output_size
            tile_out = uint16_to_bf16(out_data[start:start + n_pix * out_ch])
            tile_out = tile_out.reshape(n_pix, out_ch)
            output_flat[pix_start:pix_end, :] = tile_out.to(torch.bfloat16)

    return output


def _run_gemm_oc_blocked_pair_merged(merged_entry, elf_name, input_hwc,
                                     wt_a_u16, wt_b_u16,
                                     out_h, out_w, out_ch, tile_m, ppc):
    """Phase C step A: dispatch two 1x1 convs sharing one input BO in a
    single xrt.run. Mirror of _run_gemm_oc_blocked_merged but with paired
    weights/outputs — saves one launch per pair vs sequential dispatch.

    ELF arg layout (from build_pair_rn1.py via chain_links): in, wt_a, out_a,
    wt_b, out_b. No share_arg_idxs, so weights & outputs are NOT shared.
    """
    import pyxrt as _xrt
    device, _elf, _ctx, kernel = merged_entry

    H, W, IC = input_hwc.shape
    M = H * W
    input_size = tile_m * IC
    output_size = tile_m * out_ch

    wt_blocks_a = _repack_weights_for_gemm(wt_a_u16, IC, out_ch, out_ch)
    wt_blocks_b = _repack_weights_for_gemm(wt_b_u16, IC, out_ch, out_ch)
    wt_a = wt_blocks_a[0]
    wt_b = wt_blocks_b[0]

    pixels_per_call = N_CORES * tile_m * ppc
    total_slots = N_CORES * ppc

    in_bo = _get_merged_bo(device, elf_name, "in", total_slots * input_size * 2)
    wt_a_bo = _get_merged_bo(device, elf_name, "wt_a", wt_a.nbytes)
    wt_b_bo = _get_merged_bo(device, elf_name, "wt_b", wt_b.nbytes)
    out_a_bo = _get_merged_bo(device, elf_name, "out_a",
                              total_slots * output_size * 2)
    out_b_bo = _get_merged_bo(device, elf_name, "out_b",
                              total_slots * output_size * 2)
    _xrt_fill_bo(wt_a_bo, wt_a)
    _xrt_fill_bo(wt_b_bo, wt_b)

    input_flat = input_hwc.reshape(M, IC)
    out_a = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    out_b = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    out_a_flat = out_a.reshape(M, out_ch)
    out_b_flat = out_b.reshape(M, out_ch)

    for batch_start in range(0, M, pixels_per_call):
        batch_end = min(batch_start + pixels_per_call, M)
        batch_pixels = batch_end - batch_start

        host_in = np.zeros(total_slots * input_size, dtype=np.uint16)
        n_active_slots = (batch_pixels + tile_m - 1) // tile_m
        for s in range(n_active_slots):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            active_u16 = bf16_to_uint16(input_flat[pix_start:pix_end].flatten())
            dst = s * input_size
            host_in[dst:dst + len(active_u16)] = active_u16
        slot0 = host_in[:input_size]
        for s in range(n_active_slots, total_slots):
            host_in[s * input_size:(s + 1) * input_size] = slot0

        _xrt_fill_bo(in_bo, host_in)
        _xrt_run_kernel(kernel, [in_bo, wt_a_bo, out_a_bo, wt_b_bo, out_b_bo])
        out_a_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out_b_bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out_a_data = np.frombuffer(out_a_bo.map(), dtype=np.uint16,
                                    count=total_slots * output_size).copy()
        out_b_data = np.frombuffer(out_b_bo.map(), dtype=np.uint16,
                                    count=total_slots * output_size).copy()

        for s in range(min(n_active_slots, total_slots)):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            if pix_start >= batch_end:
                break
            n_pix = pix_end - pix_start
            start = s * output_size
            ta = uint16_to_bf16(out_a_data[start:start + n_pix * out_ch])
            tb = uint16_to_bf16(out_b_data[start:start + n_pix * out_ch])
            out_a_flat[pix_start:pix_end, :] = ta.reshape(n_pix, out_ch).to(torch.bfloat16)
            out_b_flat[pix_start:pix_end, :] = tb.reshape(n_pix, out_ch).to(torch.bfloat16)

    return out_a, out_b


def _gemm_pair_elf_name(tile_m, ic, oc, ppc):
    """Filename convention for a rn1-style 2-sub pair ELF (no k_block)."""
    return f"merged_gemm_t{tile_m}_ic{ic}_oc{oc}_p{ppc}_pair_x1"


def run_gemm_pair_mc(gemm_name, sc_name, input_hwc, wt_a_u16, wt_b_u16,
                     out_h, out_w, out_ch):
    """Phase C step A entry point. Dispatches two 1x1 convs sharing one
    input BO in one xrt.run if a pair ELF exists for the layer's shape;
    otherwise falls back to two sequential run_gemm_conv1x1_mc calls.

    Returns (out_a, out_b). Caller uses this for back-to-back rn1 calls
    inside run_rn_mc (RepNCSP.conv1 + RepNCSP.conv2 on the same input).
    """
    H, W, IC = input_hwc.shape
    M = H * W
    # Pair ELFs are only built for the non-K-blocked, single-OCB shape today.
    k_block, _ = _gemm_choose_k_block(IC, out_ch, M)
    if k_block > 0:
        out_a = run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, wt_a_u16,
                                     out_h, out_w, out_ch)
        out_b = run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, wt_b_u16,
                                     out_h, out_w, out_ch)
        return out_a, out_b

    oc_block = _gemm_choose_oc_block(IC, out_ch)
    if oc_block is None or oc_block != out_ch:
        out_a = run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, wt_a_u16,
                                     out_h, out_w, out_ch)
        out_b = run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, wt_b_u16,
                                     out_h, out_w, out_ch)
        return out_a, out_b

    tile_m = min(_gemm_tile_m(IC, oc_block), 256)
    ppc = _gemm_compute_ppc(M, tile_m, IC, oc_block)
    elf_name = _gemm_pair_elf_name(tile_m, IC, out_ch, ppc)
    merged = _get_merged_kernel(elf_name)
    if merged is None:
        out_a = run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, wt_a_u16,
                                     out_h, out_w, out_ch)
        out_b = run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, wt_b_u16,
                                     out_h, out_w, out_ch)
        return out_a, out_b

    # Pair-path uses the same _CURRENT_LAYER attribution as the singletons.
    global _CURRENT_LAYER
    _prev_layer = _CURRENT_LAYER
    _CURRENT_LAYER = gemm_name
    try:
        return _run_gemm_oc_blocked_pair_merged(
            merged, elf_name, input_hwc, wt_a_u16, wt_b_u16,
            out_h, out_w, out_ch, tile_m, ppc,
        )
    finally:
        _CURRENT_LAYER = _prev_layer


