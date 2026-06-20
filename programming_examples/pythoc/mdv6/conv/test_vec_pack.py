#!/usr/bin/env python3
"""Offline bit-exact check of the vectorized pack/unpack helpers against the
old per-tile reference path. No NPU required.

Covers:
  - stride1 even-divide   (re6_c3:  40x40 tile8  -> 5x5 tiles)
  - stride1 edge          (re4_c3:  80x80 tile12 -> 7x7 tiles, 84>80)
  - stride2               (aconv:  stride-2 shapes, edge)
  - elan_c3               (160x160 tile8 -> 20x20 tiles)
"""
import os, sys
import numpy as np
import torch

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)
import importlib.util
spec = importlib.util.spec_from_file_location(
    'ett', os.path.join(_base, '_full_model_helpers', 'elan_test_tiled.py'))
ett = importlib.util.module_from_spec(spec); spec.loader.exec_module(ett)

extract_patch = ett.extract_patch
bf16_to_uint16 = ett.bf16_to_uint16
uint16_to_bf16 = ett.uint16_to_bf16
extract_all_patches_u16 = ett.extract_all_patches_u16
pack_input_batch_u16 = ett.pack_input_batch_u16
reassemble_output_hwc = ett.reassemble_output_hwc

N_CORES = 32


def ref_extract_all(image_hwc, tiles_h, tiles_w, tile_h, tile_w, stride, ks, pad):
    """Old per-tile path -> list of uint16 patch rows."""
    patch_h = (tile_h - 1) * stride + ks
    patch_w = (tile_w - 1) * stride + ks
    patch_size_raw = patch_h * patch_w * image_hwc.shape[2]
    patch_size = patch_size_raw + (patch_size_raw % 2)
    rows = []
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            patch = extract_patch(image_hwc, tr, tc, tile_h, tile_w, stride, ks, pad)
            u = bf16_to_uint16(patch.flatten())
            if len(u) < patch_size:
                u = np.pad(u, (0, patch_size - len(u)))
            rows.append(u)
    return np.stack(rows), patch_size


def ref_pack_batch(all_patches, batch_start, patches_per_call, ppc):
    """Old per-core concatenation packing for one batch."""
    n = len(all_patches)
    batch_end = min(batch_start + patches_per_call, n)
    batch_patches = list(all_patches[batch_start:batch_end])
    while len(batch_patches) < patches_per_call:
        batch_patches.append(batch_patches[0])
    per_core = []
    for core in range(N_CORES):
        cs = core * ppc
        per_core.append(np.concatenate(batch_patches[cs:cs + ppc]))
    return np.concatenate(per_core)


def ref_reassemble_single(big_out_data, n_ocb, tiles_h, tiles_w, tile_h, tile_w,
                          oc_block, ppc, out_h, out_w, out_ch, output_tile_size,
                          output_per_batch, all_coords):
    """Old per-tile reassembly (ocb_merged single-buffer variant)."""
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    n_tiles = tiles_h * tiles_w
    for ocb in range(n_ocb):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start
        ocb_base = ocb * output_per_batch
        for j in range(n_tiles):
            tr, tc = all_coords[j]
            oh_s = tr * tile_h; ow_s = tc * tile_w
            oh_e = min(oh_s + tile_h, out_h); ow_e = min(ow_s + tile_w, out_w)
            core = j // ppc; slot = j % ppc
            start = ocb_base + (core * ppc + slot) * output_tile_size
            tile_out = uint16_to_bf16(big_out_data[start:start + output_tile_size])
            tile_out = tile_out.reshape(tile_h, tile_w, oc_block)
            output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                tile_out[:oh_e - oh_s, :ow_e - ow_s, :actual_oc]
    return output


def run_case(name, H, W, C, out_h, out_w, out_ch, tile_h, tile_w, oc_block,
             stride, ks, pad, ppc):
    torch.manual_seed(hash(name) & 0xffff)
    img = torch.randn(H, W, C, dtype=torch.bfloat16)
    tiles_h = (out_h + tile_h - 1) // tile_h
    tiles_w = (out_w + tile_w - 1) // tile_w
    n_oc_blocks = (out_ch + oc_block - 1) // oc_block
    output_tile_size = tile_h * tile_w * oc_block

    # --- patch extraction ---
    ref_patches, patch_size = ref_extract_all(img, tiles_h, tiles_w, tile_h,
                                              tile_w, stride, ks, pad)
    vec_patches = extract_all_patches_u16(img, tiles_h, tiles_w, tile_h, tile_w,
                                          stride, ks, pad)
    assert vec_patches.shape == ref_patches.shape, \
        f"{name}: shape {vec_patches.shape} != {ref_patches.shape}"
    assert np.array_equal(vec_patches, ref_patches), f"{name}: patch mismatch"

    # --- input packing (both single-batch and multi-batch) ---
    patches_per_call = N_CORES * ppc
    n_tiles = tiles_h * tiles_w
    n_batches = (n_tiles + patches_per_call - 1) // patches_per_call
    for b in range(n_batches):
        bs = b * patches_per_call
        ref_buf = ref_pack_batch(list(ref_patches), bs, patches_per_call, ppc)
        vec_buf = pack_input_batch_u16(vec_patches, bs, patches_per_call)
        assert np.array_equal(vec_buf, ref_buf), \
            f"{name}: input pack mismatch batch {b}"

    # --- output reassembly (single-buffer / ocb_merged style) ---
    output_per_batch = N_CORES * ppc * output_tile_size
    all_coords = [(tr, tc) for tr in range(tiles_h) for tc in range(tiles_w)]
    # single spatial batch case only valid if n_tiles <= patches_per_call
    if n_tiles <= patches_per_call:
        big_len = n_oc_blocks * output_per_batch
        big = np.random.randint(0, 65536, size=big_len, dtype=np.uint16)
        ref_out = ref_reassemble_single(
            big, n_oc_blocks, tiles_h, tiles_w, tile_h, tile_w, oc_block, ppc,
            out_h, out_w, out_ch, output_tile_size, output_per_batch, all_coords)
        vec_out = reassemble_output_hwc(
            big, n_oc_blocks, tiles_h, tiles_w, tile_h, tile_w, oc_block, ppc,
            out_h, out_w, out_ch, output_tile_size, output_per_batch)
        ref_u = bf16_to_uint16(ref_out.contiguous())
        vec_u = bf16_to_uint16(vec_out.contiguous())
        assert np.array_equal(ref_u, vec_u), f"{name}: output reassembly mismatch"
        oc_tag = "single-batch reassembly OK"
    else:
        # multi-batch: reassemble per-OCB from concatenated real-tile slices.
        big_per_ocb_len = n_oc_blocks * n_batches * output_per_batch
        # simulate per-batch out buffers, then build the contiguous real-tile
        # buffer the way the wired merged path will.
        per_batch_bufs = [
            np.random.randint(0, 65536, size=n_oc_blocks * output_per_batch,
                              dtype=np.uint16) for _ in range(n_batches)
        ]
        # Reference: old merged scatter loop.
        ref_out = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
        for ocb in range(n_oc_blocks):
            oc_start = ocb * oc_block
            oc_end = min(oc_start + oc_block, out_ch)
            actual_oc = oc_end - oc_start
            for b in range(n_batches):
                out_f = uint16_to_bf16(
                    per_batch_bufs[b][ocb * output_per_batch:
                                      (ocb + 1) * output_per_batch])
                bstart = b * patches_per_call
                bend = min(bstart + patches_per_call, n_tiles)
                for j in range(bend - bstart):
                    tr, tc = all_coords[bstart + j]
                    oh_s = tr * tile_h; ow_s = tc * tile_w
                    oh_e = min(oh_s + tile_h, out_h); ow_e = min(ow_s + tile_w, out_w)
                    core = j // ppc; slot = j % ppc
                    start = (core * ppc + slot) * output_tile_size
                    t = out_f[start:start + output_tile_size].reshape(
                        tile_h, tile_w, oc_block)
                    ref_out[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                        t[:oh_e - oh_s, :ow_e - ow_s, :actual_oc]
        # Vectorized: build big_out_u16 [n_ocb, n_tiles*output_tile_size] by
        # gathering each batch's real-tile prefix, then reassemble.
        big = np.empty((n_oc_blocks, n_tiles * output_tile_size), dtype=np.uint16)
        for ocb in range(n_oc_blocks):
            ofs = 0
            for b in range(n_batches):
                bstart = b * patches_per_call
                bend = min(bstart + patches_per_call, n_tiles)
                real = bend - bstart
                src = per_batch_bufs[b][ocb * output_per_batch:
                                        ocb * output_per_batch + real * output_tile_size]
                big[ocb, ofs:ofs + real * output_tile_size] = src
                ofs += real * output_tile_size
        vec_out = reassemble_output_hwc(
            big.reshape(-1), n_oc_blocks, tiles_h, tiles_w, tile_h, tile_w,
            oc_block, ppc, out_h, out_w, out_ch, output_tile_size,
            n_tiles * output_tile_size)
        ref_u = bf16_to_uint16(ref_out.contiguous())
        vec_u = bf16_to_uint16(vec_out.contiguous())
        assert np.array_equal(ref_u, vec_u), f"{name}: multi-batch reassembly mismatch"
        oc_tag = f"multi-batch ({n_batches}) reassembly OK"

    print(f"  [PASS] {name}: patches {vec_patches.shape}, "
          f"{n_oc_blocks} OCB, {oc_tag}")


def main():
    print("Vectorized pack/unpack bit-exact self-check:")
    # name, H,W,C, out_h,out_w,out_ch, tile_h,tile_w,oc_block, stride,ks,pad, ppc
    cases = [
        # stride1 even-divide: re6_c3-like 40x40 tile8 (5x5=25 tiles)
        ("re6_c3_s1_even", 40, 40, 64, 40, 40, 48, 8, 8, 16, 1, 3, 1, 1),
        # stride1 edge: re4_c3-like 80x80 tile12 -> 7x7=49, 84>80
        ("re4_c3_s1_edge", 80, 80, 96, 80, 80, 64, 12, 12, 16, 1, 3, 1, 16),
        # stride2 even: 80x80 -> 40x40 out, tile8 (5x5=25)
        ("aconv_s2_even", 80, 80, 64, 40, 40, 64, 8, 8, 16, 2, 3, 1, 1),
        # stride2 edge: 84->42 out is even; make 82 in -> out 41, tile8 -> 6x6=36, 48>41
        ("aconv_s2_edge", 82, 82, 64, 41, 41, 96, 8, 8, 16, 2, 3, 1, 4),
        # elan_c3: 160x160 tile8 -> 20x20=400 tiles, multi-batch ppc4
        ("elan_c3_s1", 160, 160, 32, 160, 160, 64, 8, 8, 16, 1, 3, 1, 4),
        # actual_oc < oc_block edge: out_ch=8 with oc_block=16
        ("oc_edge", 40, 40, 64, 40, 40, 8, 8, 8, 16, 1, 3, 1, 1),
    ]
    for c in cases:
        run_case(*c)
    print("ALL CASES BIT-EXACT.")


if __name__ == "__main__":
    main()
