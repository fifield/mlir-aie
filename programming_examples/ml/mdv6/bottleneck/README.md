# Bottleneck Layer - MDV6 on AIE2P

## Overview

Bottleneck is a composite block that combines RepConv with an additional convolution and optional residual connection. It's a fundamental building block used in the MDV6 backbone for feature extraction.

## Architecture

```
Input (H, W, C_in)
    ↓
RepConv (Conv3x3+BN + Conv1x1+BN → Add → SiLU)
    ↓ (H, W, C_out with expand=1.0)
Conv3x3 + BatchNorm + SiLU
    ↓ (H, W, C_out)
Residual Add (if enabled and dimensions match)
    ↓
Output (H, W, C_out)
```

## Implementation Details

### Fused Operations

This implementation fuses all stages into a single kernel:

**Stage 1: RepConv**
1. Conv3x3 + BN (no activation) → temp1
2. Conv1x1 + BN (no activation) → temp2
3. Add temp1 + temp2 → temp3
4. SiLU activation → temp3 (in-place)

**Stage 2: Conv + BN + SiLU**
5. Conv3x3 on temp3 → temp4
6. BatchNorm on temp4 → temp4 (in-place)
7. SiLU activation → temp4 (in-place)

**Stage 3: Residual (conditional)**
8. If residual enabled and dimensions match:
   - Add input + temp4 → output
9. Else:
   - Copy temp4 → output

### Memory Layout

**Input**: (H, W, C_in) in HWC format, bfloat16

**Weights**: Concatenated buffer containing:
- RepConv Conv3x3 weights: (C_out, C_in, 3, 3)
- RepConv BN1 params: [weight, bias, mean, var] × C_out
- RepConv Conv1x1 weights: (C_out, C_in, 1, 1)
- RepConv BN2 params: [weight, bias, mean, var] × C_out
- Conv2 weights: (C_out, C_out, 3, 3)
- Conv2 BN params: [weight, bias, mean, var] × C_out

**Output**: (H, W, C_out) in HWC format, bfloat16

**LocalBuffers** (5 total):
1. `input_copy`: Copy of input for residual (C_in elements)
2. `temp1`: Conv3x3+BN output (C_out elements)
3. `temp2`: Conv1x1+BN output (C_out elements)
4. `temp3`: RepConv output (C_out elements)
5. `temp4`: Final conv output (C_out elements)

### Hardware Constraints

**L1 Memory Usage** (8×8×8 test):
- 5 LocalBuffers × 512 elements × 2 bytes = **5 KB**
- Available: 64 KB
- **Utilization: 7.8%** ✅

**DMA Channels**:
- Input: 2 channels (input tensor + weights)
- Output: 1 channel (output tensor)
- **Total: 2 in + 1 out** ✅ (within 2 in + 2 out limit)

### Approximations

1. **Square Root** (3×): Newton-Raphson for BatchNorm in all 3 convolutions
2. **Sigmoid** (2×): Fast approximation for SiLU in RepConv and final Conv

These approximations are necessary because standard math functions are not available in the AIE linker.

## Build Instructions

```bash
# Build the xclbin
make

# Run CPU reference test
make test

# Run on NPU2 hardware
make run

# Test without residual
make run RESIDUAL=0

# Clean build artifacts
make clean
```

## Test Results

### Test Configuration
- Input: 8×8×8
- Output: 8×8×8 (same dimensions)
- Residual: Enabled (in_channels == out_channels)

### Performance (NPU2 Hardware)
- **Execution time**: 15.977 ms
- **Throughput**: ~63 images/second (for 8×8 input)

### Accuracy
- **Max error**: 3.1% (vs PyTorch reference)
- **Mean error**: 0.70%
- **Status**: ✅ PASS

**Surprisingly good accuracy!** Despite 3× sqrt and 2× sigmoid approximations, the error is only 3.1% - much better than the expected 15-20%.

## Files

- `bottleneck_bf16.cc` - C++ fused kernel implementation
- `aie2.py` - IRON design (MLIR generation)
- `test.py` - Test script with PyTorch reference
- `Makefile` - Build system
- `README.md` - This file

## Usage in MDV6

Bottleneck blocks are used extensively in MDV6:
- **RepNCSP blocks**: Multiple bottlenecks in sequence
- **Feature extraction**: Deep feature hierarchies
- **Residual learning**: Skip connections for gradient flow

Typical usage:
- Bottleneck(64, 64, residual=True) - Identity mapping
- Bottleneck(64, 128, residual=False) - Channel expansion

## Design Decisions

### Why Fully Fused?
- **Minimal data movement**: All intermediate results stay in L1
- **Single kernel call**: Reduces overhead
- **Better performance**: No L1↔L3 transfers between stages

### Why 5 LocalBuffers?
- **input_copy**: Needed for residual add (can't re-read from L3)
- **temp1, temp2**: RepConv parallel branches
- **temp3**: RepConv output (input to Conv2)
- **temp4**: Conv2 output (before residual)

### Residual Logic
- Only active when `in_channels == out_channels`
- Checked at runtime in kernel
- Matches PyTorch reference behavior

## Optimization Opportunities

1. **Vectorization**: Use AIE-API intrinsics (10-20× speedup)
2. **Multi-tile**: Parallelize across spatial dimensions
3. **Better approximations**: Lookup tables for sigmoid/sqrt
4. **Memory tiling**: Support larger inputs (>8×8)

## Performance Analysis

**Breakdown** (estimated):
- RepConv: ~9 ms (from RepConv measurements)
- Conv+BN+SiLU: ~6 ms (similar to RepConv single branch)
- Residual add: ~0.5 ms
- **Total**: ~15.5 ms (matches measured 15.977 ms)

## References

- MDV6 Paper: Microsoft MegaDetector V6
- PyTorch implementation: `python/mdv6/layers.py`
- RepConv layer: `../repconv/`
- Conv layer: `../conv/`
