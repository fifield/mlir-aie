#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

"""
Tiled convolution implementation for large images (640x640).

This implementation uses spatial tiling to process large images that don't fit
in a single AIE tile's L1 memory (64 KB). 

Conv0 Strategy (3→32 channels):
- Spatial tiles: 32×32 pixels
- Output channel blocks: 8 channels per pass (4 passes total)
- Memory per tile: ~23 KB (fits comfortably in L1)

Conv1 Strategy (32→64 channels):
- Spatial tiles: 20×20 pixels  
- Input channel blocks: 16 channels per pass (2 passes)
- Output channel blocks: 16 channels per pass (4 passes)
- Memory per tile: ~45 KB with float32 accumulation
"""

import numpy as np
import sys

from aie.iron import (
    GlobalBuffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_


def conv_layer_bf16_tiled(
    dev,
    input_height: int,
    input_width: int,
    input_channels: int,
    output_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    tile_h: int = 32,
    tile_w: int = 32,
    out_chan_block: int = 8,
):
    """
    Tiled Conv layer for large images.
    
    Args:
        dev: AIE device (NPU2Col1)
        input_height: Input image height
        input_width: Input image width
        input_channels: Number of input channels
        output_channels: Number of output channels
        kernel_size: Convolution kernel size (only 3 supported)
        stride: Convolution stride
        padding: Convolution padding
        tile_h: Spatial tile height
        tile_w: Spatial tile width
        out_chan_block: Output channel block size
    """
    
    if kernel_size != 3:
        raise ValueError("Only 3x3 kernels supported in tiled mode")
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Calculate tile grid
    tiles_h = (output_height + tile_h - 1) // tile_h
    tiles_w = (output_width + tile_w - 1) // tile_w
    num_out_blocks = (output_channels + out_chan_block - 1) // out_chan_block
    
    # Patch dimensions (includes halo for convolution)
    patch_h = tile_h + 2 * padding
    patch_w = tile_w + 2 * padding
    
    # Calculate buffer sizes
    patch_size = patch_h * patch_w * input_channels
    weight_block_size = out_chan_block * input_channels * kernel_size * kernel_size
    output_tile_size = tile_h * tile_w * out_chan_block
    
    # Type definitions (bfloat16 as uint16)
    # Input buffer holds 2 patches - ObjectFifo will chunk them
    patch_ty = np.ndarray[(patch_size,), np.dtype[np.uint16]]
    input_buffer_ty = np.ndarray[(2 * patch_size,), np.dtype[np.uint16]]
    weight_block_ty = np.ndarray[(weight_block_size,), np.dtype[np.uint16]]
    output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]
    output_buffer_ty = np.ndarray[(2 * output_tile_size,), np.dtype[np.uint16]]
    
    # Full image types for host interface
    input_ty = np.ndarray[(input_height * input_width * input_channels,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(output_channels * input_channels * kernel_size * kernel_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_height * output_width * output_channels,), np.dtype[np.uint16]]
    
    # Kernel declaration for tiled conv
    conv_tiled_kernel = Kernel(
        "conv3x3_tiled_bf16",
        "conv_bf16.o",
        [
            patch_ty,         # input patch pointer
            weight_block_ty,  # weight block pointer
            output_tile_ty,   # output tile pointer
            np.int32,         # tile_height
            np.int32,         # tile_width
            np.int32,         # input_channels
            np.int32,         # output_channels_block
            np.int32,         # stride
            np.int32,         # padding
        ],
    )
    
    # ObjectFifos for data movement
    # Use single buffering (depth=1) for Conv1 to fit in 64 KB L1 memory
    # Double buffering (depth=2) can be used for Conv0 with smaller memory footprint
    fifo_depth = 1  # Single buffering for now to support both Conv0 and Conv1
    
    # Input patch from L3 to L2 to core
    of_input_patch = ObjectFifo(patch_ty, depth=fifo_depth, name="input_patch")
    
    # Weight block from L3 to L2 to core
    of_weights = ObjectFifo(weight_block_ty, depth=1, name="weights")
    
    # Output tile from core to L2 to L3
    of_output_tile = ObjectFifo(output_tile_ty, depth=fifo_depth, name="output_tile")
    
    # Core task: process 2 spatial tiles with one output channel block
    def core_fn(of_in_patch, of_wts, of_out_tile, kernel):
        # Acquire weights once for both patches
        elem_wts = of_wts.acquire(1)
        
        # Process 2 patches in sequence with same weights
        for _ in range_(2):
            # Acquire input and output buffers
            elem_patch = of_in_patch.acquire(1)
            elem_out = of_out_tile.acquire(1)
            
            # Call tiled convolution kernel
            kernel(
                elem_patch,
                elem_wts,
                elem_out,
                tile_h,
                tile_w,
                input_channels,
                out_chan_block,
                stride,
                padding,
            )
            
            # Release input and output buffers
            of_in_patch.release(1)
            of_out_tile.release(1)
        
        # Release weights after processing both patches
        of_wts.release(1)
    
    # Create worker
    worker = Worker(
        core_fn,
        [
            of_input_patch.cons(),
            of_weights.cons(),
            of_output_tile.prod(),
            conv_tiled_kernel,
        ],
    )
    
    # Runtime sequence
    # Host sends 2 patches at a time in a single buffer
    # ObjectFifo chunks it into individual patches for the core
    # Host receives 2 output tiles in a single buffer
    rt = Runtime()
    with rt.sequence(input_buffer_ty, weight_block_ty, output_buffer_ty) as (I_patches, W_block, O_tiles):
        rt.start(worker)
        
        # Process 2 tiles at a time (host orchestrates the tiling loop)
        # Host sends buffer with 2 patches concatenated
        # ObjectFifo will chunk into 2 separate patches for the core loop
        # Host receives buffer with 2 output tiles concatenated
        rt.fill(of_input_patch.prod(), I_patches)
        rt.fill(of_weights.prod(), W_block)
        rt.drain(of_output_tile.cons(), O_tiles, wait=True)
    
    # Generate program
    return Program(dev, rt).resolve_program(SequentialPlacer())


# Parse command line arguments
if __name__ == "__main__":
    try:
        device_name = str(sys.argv[1])
        if device_name != "npu2":
            raise ValueError(f"Device {device_name} not supported. Use 'npu2'")
        
        dev = NPU2Col1()
        
        # Default to Conv0 tiling parameters (can be overridden)
        input_height = int(sys.argv[2]) if len(sys.argv) > 2 else 640
        input_width = int(sys.argv[3]) if len(sys.argv) > 3 else 640
        input_channels = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        output_channels = int(sys.argv[5]) if len(sys.argv) > 5 else 32
        
        # Tiling parameters
        tile_h = int(sys.argv[6]) if len(sys.argv) > 6 else 32
        tile_w = int(sys.argv[7]) if len(sys.argv) > 7 else 32
        out_chan_block = int(sys.argv[8]) if len(sys.argv) > 8 else 8
        
    except (IndexError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Usage: python aie2_tiled.py npu2 [H] [W] [C_in] [C_out] [tile_h] [tile_w] [out_chan_block]", file=sys.stderr)
        print("Example: python aie2_tiled.py npu2 640 640 3 32 32 32 8", file=sys.stderr)
        sys.exit(1)
    
    print(f"Generating tiled convolution:", file=sys.stderr)
    print(f"  Input: {input_height}×{input_width}×{input_channels}", file=sys.stderr)
    print(f"  Output: {input_height}×{input_width}×{output_channels}", file=sys.stderr)
    print(f"  Tile size: {tile_h}×{tile_w}", file=sys.stderr)
    print(f"  Output channel block: {out_chan_block}", file=sys.stderr)
    
    tiles_h = (input_height + tile_h - 1) // tile_h
    tiles_w = (input_width + tile_w - 1) // tile_w
    num_out_blocks = (output_channels + out_chan_block - 1) // out_chan_block
    total_tiles = tiles_h * tiles_w * num_out_blocks
    
    print(f"  Total tiles: {tiles_h}×{tiles_w}×{num_out_blocks} = {total_tiles}", file=sys.stderr)
    
    module = conv_layer_bf16_tiled(
        dev,
        input_height,
        input_width,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        tile_h=tile_h,
        tile_w=tile_w,
        out_chan_block=out_chan_block,
    )
    
    print(module)
