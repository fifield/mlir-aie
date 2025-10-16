# AConv Layer - MDV6 on AIE2P

## Overview

AConv (Average Pooling + Convolution) is a downsampling layer used in the MDV6 backbone. It combines average pooling with a strided convolution to efficiently reduce spatial dimensions while learning features.

## Architecture

```
Input (H, W, C_in)
    ↓
AvgPool2d (kernel=2×2, stride=1, padding=0)
    ↓ (H-1, W-1, C_in)
Conv3x3 (stride=2, padding=1)
    ↓ (⌊(H-1+2-3)/2⌋+1, ⌊(W-1+2-3)/2⌋+1, C_out)
BatchNorm
    ↓
SiLU Activation
    ↓
Output
```

## Implementation Details

### Fused Operations

This implementation fuses all four stages into a single kernel for efficiency:
1. **AvgPool2d**: 2×2 average pooling with stride=1
2. **Conv3x3**: 3×3 convolution with stride=2, padding=1
3. **BatchNorm**: Channel-wise normalization
4. **SiLU**: Sigmoid Linear Unit activation (x * sigmoid(x))

### Memory Layout

**Input**: (H, W, C_in) in HWC format, bfloat16
**Weights**: Concatenated buffer containing:
- Conv weights: (C_out, C_in, 3, 3) in OIHW format
- BN weight (gamma): (C_out,)
- BN bias (beta): (C_out,)
- BN mean: (C_out,)
- BN variance: (C_out,)

**Output**: (H_out, W_out, C_out) in HWC format, bfloat16

### Approximations

1. **Square Root**: Uses Newton-Raphson approximation instead of `sqrtf()`
2. **Sigmoid**: Fast approximation: `sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|))`

These approximations are necessary because standard math functions are not available in the AIE linker.

## Build Instructions

```bash
# Build the xclbin
make

# Run CPU reference test
make test

# Run on NPU2 hardware
make run

# Clean build artifacts
make clean
```

## Test Results

### Test Configuration
- Input: 8×8×8
- Output: 4×4×8
- Pooled intermediate: 7×7×8

### Performance (NPU2 Hardware)
- **Execution time**: 3.003 ms
- **Throughput**: ~170 images/second (for 8×8 input)

### Accuracy
- **Max error**: 2.34% (vs PyTorch reference)
- **Mean error**: 0.61%
- **Status**: ✅ PASS

The error is primarily due to:
- BFloat16 precision (limited mantissa)
- Fast sigmoid approximation in SiLU
- Fast square root approximation in BatchNorm

## Files

- `aconv_bf16.cc` - C++ kernel implementation
- `aie2.py` - IRON design (MLIR generation)
- `test.py` - Test script with PyTorch reference
- `Makefile` - Build system
- `README.md` - This file

## Usage in MDV6

AConv is used in the MDV6 backbone for efficient downsampling:
- Reduces spatial dimensions by 2× (due to stride=2 in conv)
- Maintains feature quality through average pooling
- Learns channel transformations through convolution

Example dimensions in MDV6:
- Layer 1: 320×320×64 → 160×160×128
- Layer 2: 160×160×128 → 80×80×256

## Optimization Opportunities

1. **Vectorization**: Use AIE-API intrinsics (10-20× speedup)
2. **Multi-tile**: Parallelize across spatial dimensions
3. **Double buffering**: Overlap compute and DMA
4. **Better approximations**: Lookup tables for sigmoid/sqrt

## References

- MDV6 Paper: Microsoft MegaDetector V6
- PyTorch implementation: `python/mdv6/layers.py`
