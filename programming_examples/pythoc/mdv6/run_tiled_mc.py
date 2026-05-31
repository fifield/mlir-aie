"""Multicore run_tiled_fused_conv — drops in for the single-core version.

Uses 32-core xclbins to process spatial tiles in parallel.
Falls back to single-core if the multicore xclbin is not available.
"""
import math
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))
import torch
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime
from regime_config import conv_regime_for_layer, gemm_regime_for_layer

# Import single-core helpers
import importlib.util
_base = os.path.dirname(__file__)
spec1 = importlib.util.spec_from_file_location('ett', os.path.join(_base, '_full_model_helpers', 'elan_test_tiled.py'))
ett = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ett)
_run_tiled_sc = ett.run_tiled_fused_conv
extract_patch = ett.extract_patch
bf16_to_uint16 = ett.bf16_to_uint16
uint16_to_bf16 = ett.uint16_to_bf16

N_CORES = 32
# MDV6_BUILD_DIR (set by lit) routes mc/gemm xclbins out of the source tree
# so test runs never see "already built, skipping" from stale artifacts.
_MDV6_BUILD_ROOT = os.environ.get("MDV6_BUILD_DIR")
if _MDV6_BUILD_ROOT:
    _bd = os.path.abspath(os.path.join(_MDV6_BUILD_ROOT, "mc"))
else:
    _bd = os.path.join(_base, "conv", "build")
_mc_cache = {}
USE_REGIME_XCLBINS = os.environ.get("USE_REGIME_XCLBINS", "0") == "1"
USE_REGIME_KBLOCKED = os.environ.get("USE_REGIME_KBLOCKED", "0") == "1"

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
USE_MERGED_KERNELS = os.environ.get("USE_MERGED_KERNELS", "0") == "1"

_MERGED_BD = os.path.join(_bd, "..", "build_merged") if _bd.endswith("/mc") else os.path.normpath(
    os.path.join(_bd, "..", "build_merged"))
_MERGED_KERNELS = {}     # elf_name -> (device, elf, hw_context, kernel)
_MERGED_BO_POOL = {}     # (elf_name, role, size) -> xrt.ext.bo


def _get_merged_kernel(elf_name):
    """Load a merged ELF kernel (cached). Returns (device, kernel) or None."""
    if elf_name in _MERGED_KERNELS:
        return _MERGED_KERNELS[elf_name]
    elf_path = os.path.join(_MERGED_BD, f"{elf_name}.elf")
    if not os.path.exists(elf_path):
        _MERGED_KERNELS[elf_name] = None
        return None
    import pyxrt as _xrt
    device = _xrt.device(0)
    elf = _xrt.elf(elf_path)
    hw_context = _xrt.hw_context(device, elf)
    kernel = _xrt.ext.kernel(hw_context, "main")
    entry = (device, elf, hw_context, kernel)
    _MERGED_KERNELS[elf_name] = entry
    return entry


def _xrt_run_kernel(kernel, args):
    """One xrt.run launch: set_arg per positional, start(), wait2().

    Hookable seam for profile_harness — wrapping this counts merged-ELF
    dispatches into the same npu_run / launch_gap buckets that the standalone
    DefaultNPURuntime.run path uses. Module-level so the harness can rebind
    it without touching each merged dispatch site.
    """
    import pyxrt as _xrt
    run = _xrt.run(kernel)
    for i, a in enumerate(args):
        run.set_arg(i, a)
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
    bo.sync(_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


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


def _load_handle(name, insts_name=None):
    """Load an xclbin, returning handle or None."""
    insts_name = name if insts_name is None else insts_name
    xclbin = os.path.join(_bd, f"{name}.xclbin")
    insts = os.path.join(_bd, f"{insts_name}.bin")
    if os.path.exists(xclbin) and os.path.exists(insts):
        return DefaultNPURuntime.load(NPUKernel(xclbin, insts))
    return None


def _get_mc_handle(name, insts_name=None):
    """Load a multicore xclbin (cached, with eviction recovery)."""
    key = name if insts_name is None else (name, insts_name)
    if key not in _mc_cache:
        _mc_cache[key] = _load_handle(name, insts_name)
    return _mc_cache[key]


def _regime_conv_artifact(mc_name, actual_name, ppc):
    if not USE_REGIME_XCLBINS:
        return None
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
    """Check that {name}.xclbin and {insts_name}.bin both exist on disk.

    Distinct from _load_handle: this is a file-system probe, NOT a hw_context
    load. _get_mc_variant uses this for variant selection so the existence
    check doesn't burn an XRT context slot — important when the merged-ELF
    path replaces the standalone xclbin (avoid loading both).
    """
    insts_name = name if insts_name is None else insts_name
    return (os.path.exists(os.path.join(_bd, f"{name}.xclbin"))
            and os.path.exists(os.path.join(_bd, f"{insts_name}.bin")))


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


def _get_sc_handle(name):
    """Load a single-core xclbin lazily (cached)."""
    key = f"sc_{name}"
    if key not in _mc_cache:
        xclbin = os.path.join(_bd, f"{name}.xclbin")
        insts = os.path.join(_bd, f"{name}.bin")
        if os.path.exists(xclbin):
            _mc_cache[key] = DefaultNPURuntime.load(NPUKernel(xclbin, insts))
        else:
            _mc_cache[key] = None
    return _mc_cache[key]


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
    actual_name, ppc = _get_mc_variant(mc_name)

    # Merged-ELF fast path: collapse the per-batch launch loop into one XRT
    # call per OCB when this layer's *active* variant has a registered merged
    # ELF and USE_MERGED_KERNELS is set. Falls through to the standard xclbin
    # path on any mismatch (no merged ELF, ELF missing on disk, etc.).
    if USE_MERGED_KERNELS and actual_name in _MERGED_LAYERS:
        elf_name, n_batches = _MERGED_LAYERS[actual_name]
        merged = _get_merged_kernel(elf_name)
        if merged is not None:
            return _run_tiled_mc_inner_merged(
                merged, elf_name, n_batches, ppc,
                input_hwc, weights_uint16,
                out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                stride, kernel_size, padding,
            )
    regime = _regime_conv_artifact(mc_name, actual_name, ppc)
    if regime is not None:
        handle_name = regime["xclbin_name"]
        insts_name = regime["insts_name"]
        mc_kh = _get_mc_handle(handle_name, insts_name)
        tile_h = regime["active_tile_h"]
        tile_w = regime["active_tile_w"]
        oc_block = regime["active_oc"]
        stride = regime["active_stride"]
        padding = regime["active_padding"]
        ppc = regime["ppc"]
    else:
        handle_name = actual_name
        insts_name = actual_name
        mc_kh = _get_mc_handle(actual_name)
    if mc_kh is None:
        raise RuntimeError(
            f"MC xclbin missing: {handle_name}/{insts_name}.bin "
            f"(requested: {mc_name})"
        )
    try:
        return _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                                    out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                    stride, kernel_size, padding, ppc, regime)
    except (RuntimeError, AttributeError) as e:
        # Transient XRT error (e.g., context-cache eviction). Reload and retry once.
        cache_key = handle_name if regime is None else (handle_name, insts_name)
        _mc_cache[cache_key] = _load_handle(handle_name, insts_name)
        mc_kh = _mc_cache[cache_key]
        if mc_kh is None:
            raise RuntimeError(
                f"MC xclbin reload failed after transient error: "
                f"{handle_name}/{insts_name}.bin ({e})"
            )
        return _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                                    out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                    stride, kernel_size, padding, ppc, regime)


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


def _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                         out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                         stride=1, kernel_size=3, padding=1, patches_per_core=1,
                         regime=None):

    H, W, C = input_hwc.shape
    tiles_h = (out_h + tile_h - 1) // tile_h
    tiles_w = (out_w + tile_w - 1) // tile_w
    n_oc_blocks = (out_ch + oc_block - 1) // oc_block
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)

    patch_h = (tile_h - 1) * stride + kernel_size
    patch_w = (tile_w - 1) * stride + kernel_size
    patch_size_raw = patch_h * patch_w * C
    active_patch_size = patch_size_raw + (patch_size_raw % 2)
    active_output_tile_size = tile_h * tile_w * oc_block
    active_conv_wt_size = oc_block * C * kernel_size * kernel_size
    if regime is None:
        patch_size = active_patch_size
        output_tile_size = active_output_tile_size
        weight_slot_size = active_conv_wt_size + 2 * oc_block
    else:
        patch_size = regime["patch_size"]
        output_tile_size = regime["output_tile_size"]
        weight_slot_size = regime["weight_size"]

    # Unpack full weight array
    total_conv_wts = out_ch * C * kernel_size * kernel_size
    all_conv_wts = weights_uint16[:total_conv_wts]
    all_bn_w = weights_uint16[total_conv_wts:total_conv_wts + out_ch]
    all_bn_b = weights_uint16[total_conv_wts + out_ch:total_conv_wts + 2 * out_ch]
    wts_id = id(weights_uint16)
    expected_wts_len = total_conv_wts + 2 * out_ch
    if len(weights_uint16) < expected_wts_len:
        raise ValueError(
            f"_run_tiled_mc_inner: weights_uint16 len={len(weights_uint16)} "
            f"too small for out_ch={out_ch} C={C} ks={kernel_size} "
            f"(need {expected_wts_len})"
        )

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start

        # Pack-cache the assembled [packed_conv | bn_w | bn_b] per ocb.
        # mlir-aie-d6f cache + mlir-aie-woi guard (expected_wts_len in key,
        # length verified on hit so a recycled id with different shape misses).
        wt_key = (wts_id, ocb, oc_block, out_ch, C, kernel_size,
                  expected_wts_len, weight_slot_size)
        wt_block = (_WTBLOCK_CACHE_3x3.get(wt_key)
                    if len(weights_uint16) == expected_wts_len else None)
        if wt_block is None:
            # Extract per-block weights (flat OIHW)
            cw_per_oc = C * kernel_size * kernel_size
            conv_block = all_conv_wts[oc_start * cw_per_oc:oc_end * cw_per_oc]
            # Pad conv_block to full oc_block (zero-fill if actual_oc < oc_block)
            if actual_oc < oc_block:
                conv_block = np.pad(conv_block, (0, (oc_block - actual_oc) * cw_per_oc))

            # For 3x3, repack OIHW → [oc_block/8, ic/8, 9, 8ic, 8oc] vectorized layout
            if kernel_size == 3:
                conv_block = _pack_3x3_weights(conv_block, oc_block, C)

            bn_w_block = all_bn_w[oc_start:oc_end]
            bn_b_block = all_bn_b[oc_start:oc_end]
            wt_block = np.concatenate([conv_block, bn_w_block, bn_b_block])
            expected = active_conv_wt_size + 2 * oc_block
            if len(wt_block) < expected:
                wt_block = np.pad(wt_block, (0, expected - len(wt_block)))
            if len(wt_block) < weight_slot_size:
                wt_block = np.pad(wt_block, (0, weight_slot_size - len(wt_block)))
            _WTBLOCK_CACHE_3x3[wt_key] = wt_block
            _gemm_cache_evict_dead_ids(_WTBLOCK_CACHE_3x3)

        # Collect all spatial patches for this oc_block
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

        # Process in batches of N_CORES * patches_per_core. Buffer pool
        # (bead mlir-aie-0pf-A): get_in_buf copies into a pinned buffer;
        # avoids per-call iron.tensor allocation.
        wt_buf = get_wt_buf(wt_block)
        patches_per_call = N_CORES * patches_per_core
        for batch_start in range(0, len(all_patches), patches_per_call):
            batch_end = min(batch_start + patches_per_call, len(all_patches))
            batch_size = batch_end - batch_start

            # Pack input: group patches by core, with each core receiving
            # `patches_per_core` consecutive tiles in one invocation. Pad
            # incomplete calls with slot-0 data because fully zero slots can
            # perturb real slots on current hardware/runtime.
            batch_patches = list(all_patches[batch_start:batch_end])
            while len(batch_patches) < patches_per_call:
                batch_patches.append(batch_patches[0])
            per_core_batches = []
            for core in range(N_CORES):
                core_start = core * patches_per_core
                core_end = core_start + patches_per_core
                per_core_batches.append(np.concatenate(batch_patches[core_start:core_end]))
            input_concat = np.concatenate(per_core_batches)

            in_buf = get_in_buf(input_concat)
            out_buf = get_out_buf(N_CORES * patches_per_core * output_tile_size)

            DefaultNPURuntime.run(mc_kh, [in_buf, wt_buf, out_buf])

            # Unpack results
            out_data = out_buf.numpy().copy()
            for i in range(batch_size):
                tr, tc = all_coords[batch_start + i]
                oh_s = tr * tile_h; ow_s = tc * tile_w
                oh_e = min(oh_s + tile_h, out_h)
                ow_e = min(ow_s + tile_w, out_w)
                core = i // patches_per_core
                slot = i % patches_per_core
                start = (core * patches_per_core + slot) * output_tile_size
                tile_out = uint16_to_bf16(out_data[start:start + active_output_tile_size])
                tile_out = tile_out.reshape(tile_h, tile_w, oc_block)
                output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                    tile_out[:oh_e - oh_s, :ow_e - ow_s, :actual_oc]

    return output


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
            ).copy()
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
                tile_out = uint16_to_bf16(out_data[start:start + output_tile_size])
                tile_out = tile_out.reshape(tile_h, tile_w, oc_block)
                output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                    tile_out[:oh_e - oh_s, :ow_e - ow_s, :actual_oc]

    return output


# ---------------------------------------------------------------------------
# GEMM Conv1x1 — vectorized 1×1 conv using mmul<4,8,8>
# ---------------------------------------------------------------------------

if _MDV6_BUILD_ROOT:
    _gemm_bd = os.path.abspath(os.path.join(_MDV6_BUILD_ROOT, "gemm"))
else:
    _gemm_bd = os.path.join(_base, "gemm_conv1x1", "build")

# Import fuse_bn_transposed for weight repacking
fuse_bn_transposed = ett.fuse_bn_transposed

# L1 budget for GEMM conv1x1 (depth=1, stack=8KB, RTP=32B reserved).
# Must stay in sync with AVAIL in gemm_conv1x1/build_gemm_conv1x1.py.
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


def _load_gemm_handle(name, insts_name=None):
    """Load a GEMM conv1x1 xclbin."""
    insts_name = name if insts_name is None else insts_name
    xclbin = os.path.join(_gemm_bd, f"{name}.xclbin")
    insts = os.path.join(_gemm_bd, f"{insts_name}.bin")
    if os.path.exists(xclbin) and os.path.exists(insts):
        return DefaultNPURuntime.load(NPUKernel(xclbin, insts))
    return None


def _get_gemm_handle(name, insts_name=None):
    """Cached GEMM xclbin handle."""
    key = f"gemm_{name}" if insts_name is None else ("gemm", name, insts_name)
    if key not in _mc_cache:
        _mc_cache[key] = _load_gemm_handle(name, insts_name)
    return _mc_cache[key]


def _regime_gemm_artifact(gemm_name, tile_m, ic, oc_block, ppc):
    if not USE_REGIME_XCLBINS:
        return None
    artifact, member = gemm_regime_for_layer(gemm_name, ic, oc_block, 0)
    if artifact is None or artifact.k_block > 0:
        return None
    active_tile_m = member.tile_m
    active_ic = member.ic
    active_oc = member.oc
    active_ppc = member.ppc
    if (active_tile_m, active_ic, active_oc, active_ppc) != (tile_m, ic, oc_block, ppc):
        # The regime may intentionally reduce tile_m and/or use a larger ppc
        # envelope. IC/OC must still identify the intended logical member.
        if (active_ic, active_oc) != (ic, oc_block):
            raise RuntimeError(
                f"GEMM regime contract mismatch for {gemm_name}: "
                f"runtime={(tile_m, ic, oc_block, ppc)} "
                f"contract={(active_tile_m, active_ic, active_oc, active_ppc)}"
            )
    if artifact.patches_per_core < ppc:
        raise RuntimeError(
            f"GEMM regime ppc envelope too small for {gemm_name}: "
            f"runtime={ppc}, envelope={artifact.patches_per_core}"
        )
    if active_ic > artifact.ic or active_oc > artifact.oc:
        raise RuntimeError(f"GEMM regime envelope too small for {gemm_name}")
    return {
        "xclbin_name": artifact.xclbin_name,
        "insts_name": f"{artifact.xclbin_name}_{member.runtime_name}_ic{member.ic}_oc{member.oc}",
        "tile_m": artifact.tile_m,
        "ic": artifact.ic,
        "oc": artifact.oc,
        "ppc": artifact.patches_per_core,
        "active_tile_m": active_tile_m,
        "active_ic": active_ic,
        "active_oc": active_oc,
        "input_size": artifact.tile_m * artifact.ic,
        "output_size": artifact.tile_m * artifact.oc,
        "weight_size": artifact.ic * artifact.oc + 2 * artifact.oc,
    }


def _regime_gemm_kblocked_artifact(gemm_name, tile_m, ic, oc, k_block, ppc):
    if not USE_REGIME_XCLBINS or not USE_REGIME_KBLOCKED:
        return None
    artifact, member = gemm_regime_for_layer(gemm_name, ic, oc, k_block)
    if artifact is None or artifact.k_block <= 0:
        return None
    if artifact.patches_per_core < ppc:
        raise RuntimeError(
            f"K-blocked GEMM regime ppc envelope too small for {gemm_name}: "
            f"runtime={ppc}, envelope={artifact.patches_per_core}"
        )
    if ic > artifact.ic or oc > artifact.oc:
        raise RuntimeError(f"K-blocked GEMM regime envelope too small for {gemm_name}")
    return {
        "xclbin_name": artifact.xclbin_name,
        "insts_name": f"{artifact.xclbin_name}_{member.runtime_name}_ic{member.ic}_oc{member.oc}",
        "tile_m": artifact.tile_m,
        "ic": artifact.ic,
        "oc": artifact.oc,
        "k_block": artifact.k_block,
        "ppc": artifact.patches_per_core,
        "logical_ic": member.ic,
        "logical_oc": member.oc,
        "logical_k_block": member.k_block,
        "input_size": artifact.tile_m * artifact.ic,
        "output_size": artifact.tile_m * artifact.oc,
        "weight_chunk_size": artifact.k_block * artifact.oc + 2 * artifact.oc,
    }


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


def run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, weights_uint16,
                         out_h, out_w, out_ch, oc_block=None):
    """GEMM-based 1×1 conv with 32-core multicore.

    Tries K-blocked path first (no OC blocking), falls back to OC-blocked,
    then to scalar MC.
    """
    H, W, IC = input_hwc.shape
    M = H * W

    # --- Try K-blocked path first ---
    k_block, tile_m_kb = _gemm_choose_k_block(IC, out_ch, M)
    if k_block > 0 and tile_m_kb >= 4:
        ppc = _gemm_compute_ppc_kblocked(M, tile_m_kb, IC, out_ch, k_block)
        kb_name = f"gemm_t{tile_m_kb}_ic{IC}_oc{out_ch}_kb{k_block}_p{ppc}"
        # Phase A.2: merged-ELF fast path. When USE_MERGED_KERNELS is set and a
        # per-layer ELF exists, bypass the regime path entirely and dispatch
        # through xrt.run. The ELF bakes in (tile_m, IC, OC, k_block, ppc), so
        # no envelope/active-shape divergence is possible.
        if USE_MERGED_KERNELS:
            elf_name = _merged_gemm_elf_name(tile_m_kb, IC, out_ch, k_block, ppc)
            merged = _get_merged_kernel(elf_name)
            if merged is not None:
                return _run_gemm_kblocked_merged(
                    merged, elf_name, input_hwc, weights_uint16,
                    out_h, out_w, out_ch, tile_m_kb, k_block, ppc,
                )
        regime = _regime_gemm_kblocked_artifact(
            gemm_name, tile_m_kb, IC, out_ch, k_block, ppc
        )
        if regime is not None:
            handle_name = regime["xclbin_name"]
            insts_name = regime["insts_name"]
            gemm_kh = _get_gemm_handle(handle_name, insts_name)
            run_tile_m = regime["tile_m"]
            run_k_block = regime["k_block"]
            run_ppc = regime["ppc"]
        else:
            handle_name = kb_name
            insts_name = kb_name
            gemm_kh = _get_gemm_handle(kb_name)
            run_tile_m = tile_m_kb
            run_k_block = k_block
            run_ppc = ppc
        if gemm_kh is None:
            raise RuntimeError(
                f"GEMM xclbin missing: {handle_name}/{insts_name}.bin "
                f"(layer={gemm_name}, IC={IC}, OC={out_ch}, M={M}, k_block={k_block}, ppc={ppc})"
            )
        return _run_gemm_kblocked(
            gemm_kh, handle_name, insts_name, input_hwc, weights_uint16,
            out_h, out_w, out_ch, run_tile_m, run_k_block, run_ppc, regime
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

    # Merged-ELF fast path (Phase A.2). The ELFs were built with k_block=0 for
    # layers where the full IC fits in L1; they match the OC-blocked kernel
    # exactly when oc_block == OC (the only case build_x1_gemm.py emits).
    if USE_MERGED_KERNELS and oc_block == out_ch:
        elf_name = _merged_gemm_elf_name(tile_m, IC, out_ch, 0, ppc)
        merged = _get_merged_kernel(elf_name)
        if merged is not None:
            return _run_gemm_oc_blocked_merged(
                merged, elf_name, input_hwc, weights_uint16,
                out_h, out_w, out_ch, tile_m, ppc,
            )
    actual_name = f"gemm_t{tile_m}_ic{IC}_oc{oc_block}_p{ppc}"
    regime = _regime_gemm_artifact(gemm_name, tile_m, IC, oc_block, ppc)
    if regime is not None:
        handle_name = regime["xclbin_name"]
        insts_name = regime["insts_name"]
        gemm_kh = _get_gemm_handle(handle_name, insts_name)
        tile_m = regime["active_tile_m"]
        ppc = regime["ppc"]
    else:
        handle_name = actual_name
        insts_name = actual_name
        gemm_kh = _get_gemm_handle(actual_name)
    if gemm_kh is None:
        raise RuntimeError(
            f"GEMM xclbin missing: {handle_name}/{insts_name}.bin "
            f"(layer={gemm_name}, IC={IC}, OC={out_ch}, M={M}, oc_block={oc_block}, tile_m={tile_m}, ppc={ppc})"
        )

    return _run_gemm_oc_blocked(gemm_kh, handle_name, insts_name, input_hwc,
                                 weights_uint16, out_h, out_w, out_ch, tile_m,
                                 oc_block, ppc, regime)


def _run_gemm_kblocked(gemm_kh, handle_name, insts_name, input_hwc, weights_uint16,
                        out_h, out_w, out_ch, tile_m, k_block, ppc, regime=None):
    """K-blocked GEMM: full OC in one pass, K-blocked weight streaming."""
    H, W, IC = input_hwc.shape
    M = H * W
    if regime is None:
        env_ic = IC
        env_oc = out_ch
        input_size = tile_m * IC
        output_size = tile_m * out_ch
        wt_kblocked = _repack_weights_kblocked(weights_uint16, IC, out_ch, k_block)
    else:
        env_ic = regime["ic"]
        env_oc = regime["oc"]
        input_size = regime["input_size"]
        output_size = regime["output_size"]
        wt_kblocked = _repack_weights_kblocked_regime(
            weights_uint16, IC, out_ch, env_ic, env_oc, k_block
        )

    wt_buf = get_wt_buf(wt_kblocked)

    pixels_per_call = N_CORES * tile_m * ppc

    input_flat = input_hwc.reshape(M, IC)
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    output_flat = output.reshape(M, out_ch)

    for batch_start in range(0, M, pixels_per_call):
        batch_end = min(batch_start + pixels_per_call, M)
        batch_pixels = batch_end - batch_start

        total_slots = N_CORES * ppc
        host_in_size = total_slots * input_size
        host_in = np.zeros(host_in_size, dtype=np.uint16)

        n_active_slots = (batch_pixels + tile_m - 1) // tile_m
        for s in range(n_active_slots):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            active_u16 = bf16_to_uint16(input_flat[pix_start:pix_end].flatten())
            if env_ic == IC:
                dst = s * input_size
                host_in[dst:dst + len(active_u16)] = active_u16
            else:
                rows = pix_end - pix_start
                active_rows = active_u16.reshape(rows, IC)
                dst = s * input_size
                slot = host_in[dst:dst + input_size].reshape(tile_m, env_ic)
                slot[:rows, :IC] = active_rows

        # Fill unused slots with slot 0's data to avoid hangs
        slot0 = host_in[:input_size]
        for s in range(n_active_slots, total_slots):
            host_in[s * input_size:(s + 1) * input_size] = slot0

        in_buf = get_in_buf(host_in)
        out_buf = get_out_buf(total_slots * output_size)

        try:
            DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])
        except Exception:
            cache_key = f"gemm_{handle_name}" if insts_name == handle_name else ("gemm", handle_name, insts_name)
            _mc_cache[cache_key] = _load_gemm_handle(handle_name, insts_name)
            gemm_kh = _mc_cache[cache_key]
            if gemm_kh is None:
                raise
            DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])

        # Unpack — each slot is tile_m pixels × full OC
        out_data = out_buf.numpy().copy()
        for s in range(min(n_active_slots, total_slots)):
            pix_start = batch_start + s * tile_m
            pix_end = min(pix_start + tile_m, batch_end)
            if pix_start >= batch_end:
                break
            n_pix = pix_end - pix_start
            start = s * output_size
            tile_out = uint16_to_bf16(out_data[start:start + n_pix * env_oc])
            tile_out = tile_out.reshape(n_pix, env_oc)
            output_flat[pix_start:pix_end, :] = tile_out[:, :out_ch].to(torch.bfloat16)

    return output


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


def _run_gemm_oc_blocked(gemm_kh, handle_name, insts_name, input_hwc, weights_uint16,
                          out_h, out_w, out_ch, tile_m, oc_block, ppc,
                          regime=None):
    """OC-blocked GEMM: legacy path with OC blocking loop."""
    H, W, IC = input_hwc.shape
    M = H * W

    n_oc_blocks = out_ch // oc_block
    wt_blocks = _repack_weights_for_gemm(weights_uint16, IC, out_ch, oc_block)

    active_input_size = tile_m * IC
    active_output_size = tile_m * oc_block
    if regime is None:
        input_size = active_input_size
        output_size = active_output_size
        weight_size = None
    else:
        input_size = regime["input_size"]
        output_size = regime["output_size"]
        weight_size = regime["weight_size"]
    pixels_per_call = N_CORES * tile_m * ppc

    input_flat = input_hwc.reshape(M, IC)
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    output_flat = output.reshape(M, out_ch)

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start
        wt_block = wt_blocks[ocb]
        if weight_size is not None and len(wt_block) < weight_size:
            wt_block = np.pad(wt_block, (0, weight_size - len(wt_block)))
        wt_buf = get_wt_buf(wt_block)

        for batch_start in range(0, M, pixels_per_call):
            batch_end = min(batch_start + pixels_per_call, M)
            batch_pixels = batch_end - batch_start

            total_slots = N_CORES * ppc
            host_in_size = total_slots * input_size
            host_in = np.zeros(host_in_size, dtype=np.uint16)

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

            in_buf = get_in_buf(host_in)
            out_buf = get_out_buf(total_slots * output_size)

            try:
                DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])
            except Exception:
                cache_key = f"gemm_{handle_name}" if insts_name == handle_name else ("gemm", handle_name, insts_name)
                _mc_cache[cache_key] = _load_gemm_handle(handle_name, insts_name)
                gemm_kh = _mc_cache[cache_key]
                if gemm_kh is None:
                    raise
                DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])

            out_data = out_buf.numpy().copy()
            for s in range(min(n_active_slots, total_slots)):
                pix_start = batch_start + s * tile_m
                pix_end = min(pix_start + tile_m, batch_end)
                if pix_start >= batch_end:
                    break
                n_pix = pix_end - pix_start
                start = s * output_size
                tile_out = uint16_to_bf16(out_data[start:start + n_pix * oc_block])
                tile_out = tile_out.reshape(n_pix, oc_block)
                output_flat[pix_start:pix_end, oc_start:oc_end] = \
                    tile_out[:, :actual_oc].to(torch.bfloat16)

    return output
