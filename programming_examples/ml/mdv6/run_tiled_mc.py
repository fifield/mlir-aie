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
