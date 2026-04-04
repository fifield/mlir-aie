"""Multicore run_tiled_fused_conv — drops in for the single-core version.

Uses 32-core xclbins to process spatial tiles in parallel.
Falls back to single-core if the multicore xclbin is not available.
"""
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
    """Multicore tiled fused conv. SC handle loaded lazily only on fallback.

    Args:
        mc_name: multicore xclbin name (e.g., 'mc_re4_c1')
        sc_name: single-core xclbin name for fallback (e.g., 're4_conv1')
    """
    mc_kh = _get_mc_handle(mc_name)
    if mc_kh is not None:
        try:
            return _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                                        out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                        stride, kernel_size, padding)
        except (RuntimeError, AttributeError) as e:
            # Handle stale/evicted XRT handles — reload and retry once
            _mc_cache[mc_name] = _load_handle(mc_name)
            mc_kh = _mc_cache[mc_name]
            if mc_kh is not None:
                try:
                    return _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                                                out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                                                stride, kernel_size, padding)
                except Exception:
                    pass
            print(f"\n    [MC fail: {e}]", end="", flush=True)
            _mc_cache[mc_name] = None

    # Fallback: lazy-load single-core
    if isinstance(sc_name, str):
        sc_kh = _get_sc_handle(sc_name)
    else:
        sc_kh = sc_name
    return _run_tiled_sc(sc_kh, input_hwc, weights_uint16,
                          out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                          stride, kernel_size, padding)


def _run_tiled_mc_inner(mc_kh, input_hwc, weights_uint16,
                         out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                         stride=1, kernel_size=3, padding=1):

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

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start

        # Extract per-block weights
        cw_per_oc = C * kernel_size * kernel_size
        conv_block = all_conv_wts[oc_start * cw_per_oc:oc_end * cw_per_oc]
        bn_w_block = all_bn_w[oc_start:oc_end]
        bn_b_block = all_bn_b[oc_start:oc_end]
        wt_block = np.concatenate([conv_block, bn_w_block, bn_b_block])
        expected = conv_wt_size + 2 * oc_block
        if len(wt_block) < expected:
            wt_block = np.pad(wt_block, (0, expected - len(wt_block)))

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

        # Process in batches of N_CORES
        wt_buf = iron.tensor(wt_block, dtype=np.uint16)
        for batch_start in range(0, len(all_patches), N_CORES):
            batch_end = min(batch_start + N_CORES, len(all_patches))
            batch_size = batch_end - batch_start

            # Pack input: concatenate patches, pad to N_CORES
            batch_patches = all_patches[batch_start:batch_end]
            while len(batch_patches) < N_CORES:
                batch_patches.append(np.zeros(patch_size, dtype=np.uint16))
            input_concat = np.concatenate(batch_patches)

            in_buf = iron.tensor(input_concat, dtype=np.uint16)
            out_buf = iron.zeros(N_CORES * output_tile_size, dtype=np.uint16)

            DefaultNPURuntime.run(mc_kh, [in_buf, wt_buf, out_buf])

            # Unpack results
            out_data = out_buf.numpy().copy()
            for i in range(batch_size):
                tr, tc = all_coords[batch_start + i]
                oh_s = tr * tile_h; ow_s = tc * tile_w
                oh_e = min(oh_s + tile_h, out_h)
                ow_e = min(ow_s + tile_w, out_w)
                start = i * output_tile_size
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

# L1 budget for GEMM conv1x1 (depth=1, stack=8KB)
_GEMM_L1 = 65536 - 8192


def _gemm_tile_m(ic, oc_block):
    """Max tile_m (mult of 4) that fits in L1."""
    wt_bytes = (ic * oc_block + 2 * oc_block) * 2
    remaining = _GEMM_L1 - wt_bytes
    if remaining <= 0:
        return 0
    return (remaining // ((ic + oc_block) * 2) // 4) * 4


def _gemm_choose_oc_block(ic, oc):
    """Choose largest oc_block that fits with >=16 tile_m."""
    for ob in [oc, 128, 64, 48, 32, 16]:
        if ob > oc or oc % ob != 0:
            continue
        if _gemm_tile_m(ic, ob) >= 16:
            return ob
    return None


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
    total_conv = oc * ic
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

    return blocks


def run_gemm_conv1x1_mc(gemm_name, sc_name, input_hwc, weights_uint16,
                         out_h, out_w, out_ch, oc_block=None):
    """GEMM-based 1×1 conv with 32-core multicore.

    Args:
        gemm_name: hint name (used for fallback to scalar MC)
        sc_name: scalar fallback xclbin name
        input_hwc: [H, W, IC] torch bf16 tensor
        weights_uint16: packed [conv_wts, bn_w, bn_b] uint16 array (flat OIHW layout)
        out_h, out_w, out_ch: output dimensions
        oc_block: output channel block (auto-computed if None)
    """
    H, W, IC = input_hwc.shape
    M = H * W

    if oc_block is None:
        oc_block = _gemm_choose_oc_block(IC, out_ch)
    if oc_block is None:
        # Fall back to scalar MC
        return run_tiled_fused_conv_mc(
            gemm_name.replace('gemm_', 'mc_'), sc_name,
            input_hwc, weights_uint16,
            out_h, out_w, out_ch, 8, 8, 16, 1, 1, 0)

    tile_m = min(_gemm_tile_m(IC, oc_block), 256)

    # Derive xclbin name from actual (tile_m, ic, oc_block) — avoids
    # mc_name aliasing issues where different layers share the same mc_name
    actual_name = f"gemm_t{tile_m}_ic{IC}_oc{oc_block}"
    n_oc_blocks = out_ch // oc_block

    # Try to load GEMM handle (keyed by actual tile/channel config)
    gemm_kh = _get_gemm_handle(actual_name)
    if gemm_kh is None:
        # Fall back to scalar MC
        mc_fallback = gemm_name.replace('gemm_', 'mc_')
        return run_tiled_fused_conv_mc(
            mc_fallback, sc_name,
            input_hwc, weights_uint16,
            out_h, out_w, out_ch, 8, 8, min(oc_block, 64), 1, 1, 0)

    # Repack weights to GEMM blocked layout
    wt_blocks = _repack_weights_for_gemm(weights_uint16, IC, out_ch, oc_block)

    input_size = tile_m * IC
    output_size = tile_m * oc_block

    # Flatten input to [M, IC]
    input_flat = input_hwc.reshape(M, IC)
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start

        wt_buf = iron.tensor(wt_blocks[ocb], dtype=np.uint16)

        # Process M pixels in batches of N_CORES * tile_m
        pixels_per_call = N_CORES * tile_m
        for batch_start in range(0, M, pixels_per_call):
            batch_end = min(batch_start + pixels_per_call, M)
            batch_pixels = batch_end - batch_start

            # Pack input: tile_m pixels per core
            # All cores must receive valid data (zeros cause hangs).
            # Unused cores get a copy of core 0's data — output is discarded.
            in_chunks = []
            n_active_cores = (batch_pixels + tile_m - 1) // tile_m
            core0_chunk = None
            for c in range(N_CORES):
                pix_start = batch_start + c * tile_m
                pix_end = min(pix_start + tile_m, batch_end)
                if pix_start < batch_end:
                    chunk = bf16_to_uint16(input_flat[pix_start:pix_end].flatten())
                    if len(chunk) < input_size:
                        chunk = np.pad(chunk, (0, input_size - len(chunk)))
                    if core0_chunk is None:
                        core0_chunk = chunk
                else:
                    chunk = core0_chunk.copy()
                in_chunks.append(chunk)

            in_buf = iron.tensor(np.concatenate(in_chunks), dtype=np.uint16)
            out_buf = iron.zeros(N_CORES * output_size, dtype=np.uint16)

            try:
                DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])
            except Exception as e:
                # Reload handle and retry once
                _mc_cache[f"gemm_{actual_name}"] = _load_gemm_handle(actual_name)
                gemm_kh = _mc_cache[f"gemm_{actual_name}"]
                if gemm_kh is None:
                    raise
                DefaultNPURuntime.run(gemm_kh, [in_buf, wt_buf, out_buf])

            # Unpack results back to [H, W, OC]
            out_data = out_buf.numpy().copy()
            for c in range(min(n_active_cores, N_CORES)):
                pix_start = batch_start + c * tile_m
                pix_end = min(pix_start + tile_m, batch_end)
                if pix_start >= batch_end:
                    break
                n_pix = pix_end - pix_start
                start = c * output_size
                tile_out = uint16_to_bf16(out_data[start:start + n_pix * oc_block])
                tile_out = tile_out.reshape(n_pix, oc_block)
                # Map flat pixel indices back to (h, w)
                for p in range(n_pix):
                    flat_idx = pix_start + p
                    h = flat_idx // W
                    w = flat_idx % W
                    if h < out_h and w < out_w:
                        output[h, w, oc_start:oc_end] = \
                            tile_out[p, :actual_oc].to(torch.bfloat16)

    return output
