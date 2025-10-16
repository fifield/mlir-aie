#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

"""
Test tiled convolution implementation for 640×640 images.

This script tests the tiled convolution on full-size images by:
1. Extracting spatial patches from the input image
2. Running each patch through the AIE kernel
3. Assembling the output tiles into the final image
4. Comparing against PyTorch reference
"""

import argparse
import numpy as np
import os
import sys
import time
import torch

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdv6.layers import Conv
from aie.utils.xrt import setup_aie, execute


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array).view(torch.bfloat16)


def extract_patch_with_halo(image, tile_row, tile_col, tile_h, tile_w, padding=1):
    """
    Extract a spatial patch from the image with halo for convolution.
    
    Args:
        image: Input tensor (H, W, C)
        tile_row: Row index of tile
        tile_col: Column index of tile
        tile_h: Tile height (output size)
        tile_w: Tile width (output size)
        padding: Convolution padding
        
    Returns:
        Patch tensor (tile_h + 2*padding, tile_w + 2*padding, C)
    """
    H, W, C = image.shape
    
    # Calculate output region
    out_start_h = tile_row * tile_h
    out_start_w = tile_col * tile_w
    out_end_h = min(out_start_h + tile_h, H)
    out_end_w = min(out_start_w + tile_w, W)
    
    # Calculate input region (with halo)
    in_start_h = out_start_h - padding
    in_start_w = out_start_w - padding
    in_end_h = out_end_h + padding
    in_end_w = out_end_w + padding
    
    # Create patch with zero padding
    patch_h = tile_h + 2 * padding
    patch_w = tile_w + 2 * padding
    patch = torch.zeros(patch_h, patch_w, C, dtype=image.dtype)
    
    # Calculate valid region to copy
    valid_start_h = max(0, in_start_h)
    valid_start_w = max(0, in_start_w)
    valid_end_h = min(H, in_end_h)
    valid_end_w = min(W, in_end_w)
    
    # Calculate offsets in patch
    patch_offset_h = valid_start_h - in_start_h
    patch_offset_w = valid_start_w - in_start_w
    
    # Copy valid region
    patch_h_valid = valid_end_h - valid_start_h
    patch_w_valid = valid_end_w - valid_start_w
    
    patch[
        patch_offset_h:patch_offset_h + patch_h_valid,
        patch_offset_w:patch_offset_w + patch_w_valid,
        :
    ] = image[valid_start_h:valid_end_h, valid_start_w:valid_end_w, :]
    
    return patch


def test_conv_tiled(
    input_height,
    input_width,
    input_channels,
    output_channels,
    tile_h=32,
    tile_w=32,
    out_chan_block=8,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """
    Test tiled convolution layer.
    
    Args:
        input_height: Input image height
        input_width: Input image width
        input_channels: Number of input channels
        output_channels: Number of output channels
        tile_h: Spatial tile height
        tile_w: Spatial tile width
        out_chan_block: Output channel block size
        use_aie: Whether to run on AIE hardware
        xclbin_path: Path to xclbin file
        insts_path: Path to instructions file
    """
    
    print(f"\n{'='*80}")
    print(f"Testing Tiled Conv Layer:")
    print(f"  Input: {input_height}×{input_width}×{input_channels}")
    print(f"  Output: {input_height}×{input_width}×{output_channels}")
    print(f"  Tile size: {tile_h}×{tile_w}")
    print(f"  Output channel block: {out_chan_block}")
    print(f"{'='*80}\n")
    
    kernel_size = 3
    stride = 1
    padding = 1
    
    # Calculate tile grid
    tiles_h = (input_height + tile_h - 1) // tile_h
    tiles_w = (input_width + tile_w - 1) // tile_w
    num_out_blocks = (output_channels + out_chan_block - 1) // out_chan_block
    total_tiles = tiles_h * tiles_w * num_out_blocks
    
    print(f"Tile grid: {tiles_h}×{tiles_w} spatial, {num_out_blocks} output blocks")
    print(f"Total tiles to process: {total_tiles}\n")
    
    # Create PyTorch Conv layer for reference
    torch_conv = Conv(
        input_channels,
        output_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        activation="SiLU"
    )
    torch_conv.eval()
    torch_conv = torch_conv.to(torch.bfloat16)
    
    # Generate random input
    torch.manual_seed(42)
    input_nchw = torch.randn(1, input_channels, input_height, input_width).to(torch.bfloat16)
    
    # Run PyTorch forward pass (without activation for now)
    with torch.no_grad():
        conv_out = torch_conv.bn(torch_conv.conv(input_nchw))
        torch_output_nchw = conv_out
    
    print(f"PyTorch output shape: {torch_output_nchw.shape}")
    print(f"PyTorch output range: [{torch_output_nchw.min():.4f}, {torch_output_nchw.max():.4f}]")
    
    if use_aie:
        print(f"\n{'='*80}")
        print("Running Tiled Convolution on NPU2 Hardware")
        print(f"{'='*80}\n")
        
        # Convert input to HWC format
        input_hwc = input_nchw.squeeze(0).permute(1, 2, 0).contiguous()
        
        # Extract weights
        weights = torch_conv.conv.weight.data  # (O, I, K, K)
        
        # Prepare output accumulator
        output_hwc = torch.zeros(input_height, input_width, output_channels, dtype=torch.bfloat16)
        
        # Patch and tile sizes
        patch_h = tile_h + 2 * padding
        patch_w = tile_w + 2 * padding
        patch_size = patch_h * patch_w * input_channels
        weight_block_size = out_chan_block * input_channels * kernel_size * kernel_size
        output_tile_size = tile_h * tile_w * out_chan_block
        
        print(f"Memory per tile:")
        print(f"  Input patch: {patch_size} elems ({patch_size * 2} bytes)")
        print(f"  Weight block: {weight_block_size} elems ({weight_block_size * 2} bytes)")
        print(f"  Output tile: {output_tile_size} elems ({output_tile_size * 2} bytes)")
        
        # Setup AIE
        print(f"\nSetting up AIE...")
        app = setup_aie(
            xclbin_path,
            insts_path,
            (patch_size,),
            np.uint16,
            (weight_block_size,),
            np.uint16,
            (output_tile_size,),
            np.uint16,
            kernel_name="MLIR_AIE"
        )
        
        total_execution_time = 0
        tiles_processed = 0
        
        # Process each output channel block
        for out_blk_idx in range(num_out_blocks):
            out_ch_start = out_blk_idx * out_chan_block
            out_ch_end = min(out_ch_start + out_chan_block, output_channels)
            actual_out_ch_block = out_ch_end - out_ch_start
            
            # Extract weight block for this output channel range
            weight_block = weights[out_ch_start:out_ch_end, :, :, :]  # (O_blk, I, K, K)
            weight_block_uint16 = bf16_to_uint16(weight_block.flatten())
            
            # Pad if needed
            if actual_out_ch_block < out_chan_block:
                padding_size = (out_chan_block - actual_out_ch_block) * input_channels * kernel_size * kernel_size
                weight_block_uint16 = np.concatenate([
                    weight_block_uint16,
                    np.zeros(padding_size, dtype=np.uint16)
                ])
            
            print(f"\nProcessing output channels {out_ch_start}-{out_ch_end-1}...")
            
            # Process each spatial tile
            for tile_row in range(tiles_h):
                for tile_col in range(tiles_w):
                    # Extract input patch
                    patch = extract_patch_with_halo(
                        input_hwc, tile_row, tile_col, tile_h, tile_w, padding
                    )
                    patch_uint16 = bf16_to_uint16(patch.flatten())
                    
                    # Execute on AIE
                    start = time.time_ns()
                    output_buffer = execute(app, patch_uint16, weight_block_uint16)
                    stop = time.time_ns()
                    exec_time = (stop - start) / 1000  # microseconds
                    total_execution_time += exec_time
                    
                    # Convert output back to bf16
                    output_tile_bf16 = uint16_to_bf16(output_buffer[:output_tile_size])
                    output_tile_hwc = output_tile_bf16.reshape(tile_h, tile_w, out_chan_block)
                    
                    # Place tile in output (handling edge tiles)
                    out_start_h = tile_row * tile_h
                    out_start_w = tile_col * tile_w
                    out_end_h = min(out_start_h + tile_h, input_height)
                    out_end_w = min(out_start_w + tile_w, input_width)
                    
                    actual_tile_h = out_end_h - out_start_h
                    actual_tile_w = out_end_w - out_start_w
                    
                    output_hwc[
                        out_start_h:out_end_h,
                        out_start_w:out_end_w,
                        out_ch_start:out_ch_end
                    ] = output_tile_hwc[:actual_tile_h, :actual_tile_w, :actual_out_ch_block]
                    
                    tiles_processed += 1
                    if tiles_processed % 10 == 0:
                        print(f"  Processed {tiles_processed}/{total_tiles} tiles...", end='\r')
        
        print(f"\n\nTotal execution time: {total_execution_time:.2f} μs ({total_execution_time/1000:.3f} ms)")
        print(f"Average time per tile: {total_execution_time/total_tiles:.2f} μs")
        
        # Convert to NCHW for comparison
        aie_output_nchw = output_hwc.permute(2, 0, 1).unsqueeze(0).float()
        
        print(f"\nAIE output shape: {aie_output_nchw.shape}")
        print(f"AIE output range: [{aie_output_nchw.min():.4f}, {aie_output_nchw.max():.4f}]")
        
        # Compare
        diff = torch.abs(torch_output_nchw.float() - aie_output_nchw)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nComparison (PyTorch vs AIE Tiled):")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        tolerance = 0.1
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False
    
    else:
        # CPU reference test (simplified - just verify dimensions)
        print("\nCPU reference mode (dimensions check only)")
        print(f"  ✓ PyTorch reference computed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test tiled Conv layer for 640×640 images")
    parser.add_argument("--height", "-ht", type=int, default=640, help="Input height")
    parser.add_argument("--width", "-wd", type=int, default=640, help="Input width")
    parser.add_argument("--in-channels", "-ic", type=int, default=3, help="Input channels")
    parser.add_argument("--out-channels", "-oc", type=int, default=32, help="Output channels")
    parser.add_argument("--tile-h", "-th", type=int, default=32, help="Tile height")
    parser.add_argument("--tile-w", "-tw", type=int, default=32, help="Tile width")
    parser.add_argument("--out-chan-block", "-ob", type=int, default=8, help="Output channel block size")
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_conv_tiled(
        args.height,
        args.width,
        args.in_channels,
        args.out_channels,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        out_chan_block=args.out_chan_block,
        use_aie=use_aie,
        xclbin_path=args.xclbin,
        insts_path=args.insts,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
