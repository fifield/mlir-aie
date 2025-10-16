# Conv Layer for MDV6 on AIE2P

This directory contains convolution layer implementations for the MDV6 (MegaDetectorV6) model on AMD AIE2P (NPU2), including **tiled convolution for large 640×640 images**.

## Overview

This implementation provides:
- **Bfloat16 convolution kernels** (3×3 and 1×1)
- **IRON-based AIE designs** for both small and large images
- **Spatial and channel tiling** to overcome L1 memory constraints
- **PyTorch validation** against the MDV6 reference model

## Quick Start

### Small Images (8×8) - Single Tile
```bash
# Test on CPU
make test

# Build for hardware
make build/final.xclbin

# Run on NPU2
make run
```

### Large Images (640×640) - Tiled
```bash
# Test Conv0 (3→32 channels) CPU reference
make test-conv0-640

# Build tiled xclbin
make build-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=3 OUT_CHANNELS=32

# Run on NPU2 (when hardware available)
make run-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=3 OUT_CHANNELS=32
```

## Files

### Core Implementation
- `conv_bf16.cc` - C++ kernels (single-tile + tiled variants)
- `aie2.py` - IRON design for small images (single-tile)
- `aie2_tiled.py` - IRON design for large images (tiled) ⭐ NEW
- `test.py` - Test script for single-tile
- `test_tiled.py` - Test script for tiled convolution ⭐ NEW
- `Makefile` - Build system with tiling support

### Documentation
- `README.md` - This file
- `TILING_STRATEGY.md` - Complete tiling strategy guide ⭐ NEW
- `CONV_TILING_SUMMARY.md` - Implementation summary ⭐ NEW
- `CONV_MODEL_DIMENSIONS_VALIDATION.md` - Model dimension tests

## Kernels

### Single-Tile Kernels (for 8×8 images)

#### conv3x3_bf16
3x3 convolution with configurable stride and padding.

**Signature:**
```c
void conv3x3_bf16(bfloat16 *input, bfloat16 *weights, bfloat16 *output,
                  int32_t input_height, int32_t input_width,
                  int32_t input_channels, int32_t output_channels,
                  int32_t stride, int32_t padding);
```

**Layout:**
- Input: (H, W, C_in) - Height-Width-Channel order
- Weights: (C_out, C_in, 3, 3) - flattened
- Output: (H_out, W_out, C_out)

### conv1x1_bf16
1x1 pointwise convolution (essentially a matrix multiplication per spatial location).

**Signature:**
```c
void conv1x1_bf16(bfloat16 *input, bfloat16 *weights, bfloat16 *output,
                  int32_t input_height, int32_t input_width,
                  int32_t input_channels, int32_t output_channels);
```

### conv3x3_fused_bf16
Fused 3x3 convolution + BatchNorm + SiLU activation (not yet used).

### Tiled Kernels (for 640×640 images) ⭐ NEW

#### conv3x3_tiled_bf16
Process one spatial tile with output channel blocking (Conv0).

**Signature:**
```c
void conv3x3_tiled_bf16(bfloat16 *input_patch, bfloat16 *weights,
                        bfloat16 *output_tile,
                        int32_t tile_height, int32_t tile_width,
                        int32_t input_channels, int32_t output_channels_block,
                        int32_t stride, int32_t padding);
```

**Memory**: ~28 KB for 32×32 tile with 8 output channels

#### conv3x3_partial_bf16
Process one input channel block with float32 accumulation (Conv1).

**Signature:**
```c
void conv3x3_partial_bf16(bfloat16 *input_patch, bfloat16 *weights,
                          float *accum_out,
                          int32_t tile_height, int32_t tile_width,
                          int32_t input_channels_block,
                          int32_t output_channels_block,
                          int32_t stride, int32_t padding);
```

**Memory**: ~50 KB for 20×20 tile with 16×16 channel blocks

#### convert_accum_bf16
Convert float32 accumulation buffer to bfloat16 output.

## Tiling Strategy

### Why Tiling?

Large images (640×640) cannot fit in L1 memory (64 KB):
- Conv0 input: 640×640×3 = 2.34 MB ❌
- Conv1 input: 640×640×32 = 25 MB ❌

**Solution**: Process image in smaller tiles that fit in L1.

### Conv0 Tiling (3→32 channels)
- **Spatial tile**: 32×32 pixels
- **Output channel block**: 8 channels
- **Total tiles**: 20×20×4 = 1,600
- **Memory per tile**: ~28 KB ✓

### Conv1 Tiling (32→64 channels)
- **Spatial tile**: 20×20 pixels
- **Input channel block**: 16 (2 passes)
- **Output channel block**: 16 (4 passes)
- **Total kernel calls**: 1,024×8 = 8,192
- **Memory per tile**: ~50 KB ✓

See [TILING_STRATEGY.md](TILING_STRATEGY.md) for complete details.

## Memory Requirements

### Small Images (8×8)
For a typical small test case (8×8 spatial, 8 channels):
- Input: 8 × 8 × 8 × 2 bytes = 1 KB
- Weights (3×3): 8 × 8 × 3 × 3 × 2 bytes = 1.125 KB
- Output: 8 × 8 × 8 × 2 bytes = 1 KB
- **Total: ~3.1 KB** - easily fits in L1 (64 KB)

### Large Images (640×640) - Tiled
**Conv0 per tile** (32×32, 8 output channels):
- Input patch: 34×34×3 = 6.8 KB
- Weights: 8×3×3×3 = 0.4 KB
- Output: 32×32×8 = 16 KB
- **Total: ~28 KB** ✓ Fits in L1

**Conv1 per tile** (20×20, 16×16 channel blocks):
- Input patch: 22×22×16 = 15.1 KB
- Weights: 16×16×3×3 = 4.5 KB
- Accum: 20×20×16×4 = 25 KB (float32)
- **Total: ~50 KB** ✓ Fits in L1

## Building

### Prerequisites
- mlir-aie toolchain installed
- NPU2 (AIE2P) hardware or emulator
- Python 3 with PyTorch and mdv6 module

### Build Commands

```bash
# Generate MLIR and compile kernel
make

# Run CPU reference test (no hardware needed)
make test

# Run on NPU2 hardware (requires xclbin)
make run
```

### Custom Parameters

You can override the default parameters:

```bash
make HEIGHT=16 WIDTH=16 IN_CHANNELS=16 OUT_CHANNELS=16 KERNEL_SIZE=3
```

## Testing

### CPU Reference Test

```bash
python3 test.py -ht 8 -wd 8 -ic 8 -oc 8 -k 3 -s 1 -p 1
```

This runs a NumPy reference implementation and compares against PyTorch.

### MDV6 Model Dimension Tests

Test the specific Conv0 and Conv1 configurations used in the MDV6 model:

```bash
# Test both Conv0 (3→32) and Conv1 (32→64)
make test-model-dims

# Test Conv0 only (8×8, 3→32)
make test-conv0

# Test Conv1 only (8×8, 32→64)
make test-conv1
```

Or run the test script directly:

```bash
# Test all model configurations
python3 test_model_dimensions.py --config all

# Test specific configuration
python3 test_model_dimensions.py --config conv0
python3 test_model_dimensions.py --config conv1

# Verbose output
python3 test_model_dimensions.py --config all -v
```

**Validation Status:**
- ✅ Conv0 (8×8, 3→32): PASS - Memory: 6.06 KB, Max diff: 0.003854
- ✅ Conv1 (8×8, 32→64): PASS - Memory: 48.00 KB, Max diff: 0.003887

See `CONV_MODEL_DIMENSIONS_VALIDATION.md` for detailed validation results.

### Hardware Test (TODO)

```bash
python3 test.py -ht 8 -wd 8 -ic 8 -oc 8 -k 3 -s 1 -p 1 \
    -x build/final.xclbin -i build/insts.bin
```

This will run on NPU2 hardware once XRT integration is complete.

## Current Status

✅ **Completed:**
- Basic scalar convolution kernels (3x3, 1x1)
- IRON design for single-tile execution
- PyTorch validation framework
- Build system

⏳ **In Progress:**
- XRT integration for hardware execution
- Vectorized kernels using AIE intrinsics
- BatchNorm + SiLU fusion

🔜 **Planned:**
- Spatial tiling for larger inputs
- Multi-tile parallelization
- Channel tiling
- Performance optimization

## Design Notes

### Data Layout

The AIE kernel uses **HWC (Height-Width-Channel)** layout, while PyTorch uses **NCHW (Batch-Channel-Height-Width)**. The test script handles the conversion.

### Bfloat16 Representation

Bfloat16 values are passed as `uint16` to/from the AIE, then cast to `bfloat16` inside the kernel.

### ObjectFIFO Depth

Currently using depth=1 (single buffering) for simplicity. This can be increased to 2 for double buffering to hide memory latency.

### Single Tile Implementation

This initial implementation uses a single compute tile. The entire input, weights, and output must fit in the tile's L1 memory (64 KB) or use L2 (512 KB) as a staging area.

## Next Steps

1. **Add XRT integration** to run on hardware
2. **Vectorize kernels** using AIE-API intrinsics for better performance
3. **Implement tiling** for larger layers that don't fit in L1
4. **Add BatchNorm + SiLU fusion** to match MDV6 Conv layer
5. **Multi-tile parallelization** for spatial or channel dimensions

## Related Files

- `python/mdv6/layers.py` - PyTorch reference implementation
- `third_party/ironclad/aie_kernels/aie2p/` - Example optimized kernels
- `programming_examples/ml/conv2d/` - Similar conv2d example for int8

## References

- [IRON Programming Guide](../../../programming_guide/)
- [AIE2P Architecture Manual](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview)
- [MDV6 Model Summary](../../../../python/mdv6/PROJECT_SUMMARY.md)
