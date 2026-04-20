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

# Import single-core helpers
import importlib.util
_base = os.path.dirname(__file__)
spec1 = importlib.util.spec_from_file_location('ett', os.path.join(_base, 'elan', 'test_tiled.py'))
ett = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ett)
_run_tiled_sc = ett.run_tiled_fused_conv
extract_patch = ett.extract_patch
bf16_to_uint16 = ett.bf16_to_uint16
uint16_to_bf16 = ett.uint16_to_bf16

N_CORES = 32
_bd = os.path.join(_base, "conv", "build")
_mc_cache = {}

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


_GEMM_CACHE_MAX = 256  # Bound to avoid unbounded growth across long video streams.


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
# Separate pools per role: a single run() call may pass different buffers of
# the same size as input + weight + output (e.g., 1×1 conv with ic == oc).
# Aliasing input and output to the same XRT buffer hangs the kernel.
_INPUT_POOL = {}    # (size, dtype.kind, itemsize) -> iron.Tensor
_WEIGHT_POOL = {}
_OUTPUT_POOL = {}


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
    """Return the pooled XRT weight buffer initialised with arr's contents."""
    buf = _pooled_buf(_WEIGHT_POOL, arr.size, arr.dtype)
    _fill_and_sync(buf, arr)
    return buf


def get_out_buf(size, dtype=np.uint16):
    """Return the pooled XRT output buffer (contents undefined; kernel overwrites)."""
    return _pooled_buf(_OUTPUT_POOL, size, dtype)


def _load_handle(name):
    """Load an xclbin, returning handle or None."""
    xclbin = os.path.join(_bd, f"{name}.xclbin")
    insts = os.path.join(_bd, f"{name}.bin")
    if os.path.exists(xclbin):
        return DefaultNPURuntime.load(NPUKernel(xclbin, insts))
    return None


def _get_mc_handle(name):
    """Load a multicore xclbin (cached, with eviction recovery)."""
    if name not in _mc_cache:
        _mc_cache[name] = _load_handle(name)
    return _mc_cache[name]


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
    if variant != name and _load_handle(variant) is None:
        # Back off the depth suffix first, then the ppc suffix.
        if depth > 1:
            alt = f"{name}_p{ppc}" if ppc > 1 else name
            if _load_handle(alt) is not None:
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
    mc_kh = _get_mc_handle(actual_name)
    if mc_kh is None:
        raise RuntimeError(
            f"MC xclbin missing: {actual_name} (requested: {mc_name})"
        )
    try:
        return _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                                    out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                    stride, kernel_size, padding, ppc)
    except (RuntimeError, AttributeError) as e:
        # Transient XRT error (e.g., context-cache eviction). Reload and retry once.
        _mc_cache[actual_name] = _load_handle(actual_name)
        mc_kh = _mc_cache[actual_name]
        if mc_kh is None:
            raise RuntimeError(
                f"MC xclbin reload failed after transient error: {actual_name} ({e})"
            )
        return _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                                    out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                    stride, kernel_size, padding, ppc)


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
                         stride=1, kernel_size=3, padding=1, patches_per_core=1):

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
    conv_wt_size = oc_block * C * kernel_size * kernel_size

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
        wt_key = (wts_id, ocb, oc_block, out_ch, C, kernel_size, expected_wts_len)
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
            expected = conv_wt_size + 2 * oc_block
            if len(wt_block) < expected:
                wt_block = np.pad(wt_block, (0, expected - len(wt_block)))
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
                tile_out = uint16_to_bf16(out_data[start:start + output_tile_size])
                tile_out = tile_out.reshape(tile_h, tile_w, oc_block)
                output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                    tile_out[:oh_e - oh_s, :ow_e - ow_s, :actual_oc]

    return output


# ---------------------------------------------------------------------------
# GEMM Conv1x1 — vectorized 1×1 conv using mmul<4,8,8>
# ---------------------------------------------------------------------------

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


def _load_gemm_handle(name):
    """Load a GEMM conv1x1 xclbin."""
    xclbin = os.path.join(_gemm_bd, f"{name}.xclbin")
    insts = os.path.join(_gemm_bd, f"{name}.bin")
    if os.path.exists(xclbin):
        return DefaultNPURuntime.load(NPUKernel(xclbin, insts))
    return None


def _get_gemm_handle(name):
    """Cached GEMM xclbin handle."""
    key = f"gemm_{name}"
    if key not in _mc_cache:
        _mc_cache[key] = _load_gemm_handle(name)
    return _mc_cache[key]


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
        gemm_kh = _get_gemm_handle(kb_name)
        if gemm_kh is None:
            raise RuntimeError(
                f"GEMM xclbin missing: {kb_name} "
                f"(layer={gemm_name}, IC={IC}, OC={out_ch}, M={M}, k_block={k_block}, ppc={ppc})"
            )
        return _run_gemm_kblocked(gemm_kh, kb_name, input_hwc, weights_uint16,
                                   out_h, out_w, out_ch, tile_m_kb, k_block, ppc)

    # --- OC-blocked path ---
    if oc_block is None:
        oc_block = _gemm_choose_oc_block(IC, out_ch)
    if oc_block is None:
        raise RuntimeError(
            f"GEMM oc_block selection failed (layer={gemm_name}, IC={IC}, OC={out_ch}, M={M})"
        )

    tile_m = min(_gemm_tile_m(IC, oc_block), 256)
    ppc = _gemm_compute_ppc(M, tile_m, IC, oc_block)

    actual_name = f"gemm_t{tile_m}_ic{IC}_oc{oc_block}_p{ppc}"
    gemm_kh = _get_gemm_handle(actual_name)
    if gemm_kh is None:
        raise RuntimeError(
            f"GEMM xclbin missing: {actual_name} "
            f"(layer={gemm_name}, IC={IC}, OC={out_ch}, M={M}, oc_block={oc_block}, tile_m={tile_m}, ppc={ppc})"
        )

    return _run_gemm_oc_blocked(gemm_kh, actual_name, input_hwc, weights_uint16,
                                 out_h, out_w, out_ch, tile_m, oc_block, ppc)


def _run_gemm_kblocked(gemm_kh, actual_name, input_hwc, weights_uint16,
                        out_h, out_w, out_ch, tile_m, k_block, ppc):
    """K-blocked GEMM: full OC in one pass, K-blocked weight streaming."""
    H, W, IC = input_hwc.shape
    M = H * W
    n_k_blocks = IC // k_block

    # Repack weights into K-blocked layout
    wt_kblocked = _repack_weights_kblocked(weights_uint16, IC, out_ch, k_block)
    wt_buf = get_wt_buf(wt_kblocked)

    input_size = tile_m * IC       # per core per patch
    output_size = tile_m * out_ch  # per core per patch
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
        active_end = min(batch_start + n_active_slots * tile_m, batch_end)
        active_u16 = bf16_to_uint16(input_flat[batch_start:active_end].flatten())
        host_in[:len(active_u16)] = active_u16

        # Fill unused slots with slot 0's data to avoid hangs
        slot0 = host_in[:input_size]
        for s in range(n_active_slots, total_slots):
            host_in[s * input_size:(s + 1) * input_size] = slot0

        in_buf = get_in_buf(host_in)
        out_buf = get_out_buf(total_slots * output_size)

        try:
            DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])
        except Exception:
            _mc_cache[f"gemm_{actual_name}"] = _load_gemm_handle(actual_name)
            gemm_kh = _mc_cache[f"gemm_{actual_name}"]
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
            tile_out = uint16_to_bf16(out_data[start:start + n_pix * out_ch])
            tile_out = tile_out.reshape(n_pix, out_ch)
            output_flat[pix_start:pix_end, :] = tile_out.to(torch.bfloat16)

    return output


def _run_gemm_oc_blocked(gemm_kh, actual_name, input_hwc, weights_uint16,
                          out_h, out_w, out_ch, tile_m, oc_block, ppc):
    """OC-blocked GEMM: legacy path with OC blocking loop."""
    H, W, IC = input_hwc.shape
    M = H * W

    n_oc_blocks = out_ch // oc_block
    wt_blocks = _repack_weights_for_gemm(weights_uint16, IC, out_ch, oc_block)

    input_size = tile_m * IC
    output_size = tile_m * oc_block
    pixels_per_call = N_CORES * tile_m * ppc

    input_flat = input_hwc.reshape(M, IC)
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    output_flat = output.reshape(M, out_ch)

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start
        wt_buf = get_wt_buf(wt_blocks[ocb])

        for batch_start in range(0, M, pixels_per_call):
            batch_end = min(batch_start + pixels_per_call, M)
            batch_pixels = batch_end - batch_start

            total_slots = N_CORES * ppc
            host_in_size = total_slots * input_size
            host_in = np.zeros(host_in_size, dtype=np.uint16)

            n_active_slots = (batch_pixels + tile_m - 1) // tile_m
            active_end = min(batch_start + n_active_slots * tile_m, batch_end)
            active_u16 = bf16_to_uint16(input_flat[batch_start:active_end].flatten())
            host_in[:len(active_u16)] = active_u16

            slot0 = host_in[:input_size]
            for s in range(n_active_slots, total_slots):
                host_in[s * input_size:(s + 1) * input_size] = slot0

            in_buf = get_in_buf(host_in)
            out_buf = get_out_buf(total_slots * output_size)

            try:
                DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])
            except Exception:
                _mc_cache[f"gemm_{actual_name}"] = _load_gemm_handle(actual_name)
                gemm_kh = _mc_cache[f"gemm_{actual_name}"]
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
