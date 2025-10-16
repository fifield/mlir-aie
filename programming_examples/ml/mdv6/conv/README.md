# Conv Layer for MDV6 on AIE2P

This directory contains a basic implementation of a Convolution layer for the MDV6 (MegaDetectorV6) model on AMD AIE2P (NPU2).

## Overview

This is the first layer implementation in the MDV6 port to mlir-aie. It provides:
- **Bfloat16 convolution kernels** (3x3 and 1x1)
- **IRON-based AIE design** for single-tile execution
- **PyTorch validation** against the MDV6 reference model

## Files

- `conv_bf16.cc` - C++ kernels for convolution operations
- `aie2.py` - IRON design describing the AIE configuration
- `test.py` - Python test script for validation
- `Makefile` - Build system
- `README.md` - This file

## Kernels

### conv3x3_bf16
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

## Memory Requirements

For a typical small test case (8x8 spatial, 8 channels):
- Input: 8 × 8 × 8 × 2 bytes = 1 KB
- Weights (3x3): 8 × 8 × 3 × 3 × 2 bytes = 1.125 KB
- Output: 8 × 8 × 8 × 2 bytes = 1 KB
- **Total: ~3.1 KB** - easily fits in L1 (64 KB)

For larger layers, we'll need tiling strategies.

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
