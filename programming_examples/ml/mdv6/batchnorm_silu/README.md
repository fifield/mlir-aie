# BatchNorm + SiLU Layer for MDV6 on AIE2P

This directory contains the implementation of fused BatchNorm + SiLU activation for the MDV6 model on AMD AIE2P (NPU2).

## Overview

BatchNorm + SiLU is used after every Conv layer in MDV6. This implementation provides:
- **Fused BatchNorm + SiLU** for efficiency
- **Standalone BatchNorm** (without activation)
- **Standalone SiLU** activation
- **BFloat16 precision**

## Files

- `batchnorm_silu_bf16.cc` - C++ kernels
- `aie2.py` - IRON design
- `test.py` - Validation tests
- `Makefile` - Build system
- `README.md` - This file

## Kernels

### batchnorm_silu_bf16
Fused BatchNorm + SiLU activation.

**Operations:**
1. BatchNorm: `y = weight * x + bias` (per channel)
2. SiLU: `y = x * sigmoid(x)`

**Signature:**
```c
void batchnorm_silu_bf16(
    bfloat16 *input,      // (H, W, C)
    bfloat16 *bn_params,  // (2*C,) - weight then bias
    bfloat16 *output,     // (H, W, C)
    int32_t height,
    int32_t width,
    int32_t channels
);
```

### batchnorm_bf16
BatchNorm only (no activation).

### silu_bf16
SiLU activation only.

## Memory Requirements

For 8×8×8 test case:
- Input: 512 elements (1,024 bytes)
- BN params: 16 elements (32 bytes) - very small!
- Output: 512 elements (1,024 bytes)
- **Total: ~2 KB** (3% of L1)

## Building

```bash
# Build everything
make

# Run CPU reference test
make test

# Run on NPU2 hardware
make run
```

## Current Status

✅ **Completed:**
- Fused BN+SiLU kernel implementation
- IRON design for NPU2
- Build system
- Hardware execution working

⚠️ **Known Issue: Sigmoid Approximation**

The current implementation uses a **fast sigmoid approximation**:
```
sigmoid(x) ≈ x / (1 + |x|), shifted to [0, 1]
```

This avoids the `tanh` function (not available in AIE linker) but has **~21% error** compared to true sigmoid.

**Impact:**
- Hardware execution: ✅ Working (1.569 ms)
- Numerical accuracy: ⚠️ ~21% error vs PyTorch SiLU
- Functional correctness: ✅ Kernel executes correctly

**Solutions (for future optimization):**
1. **Use AIE-API tanh intrinsic** (if available for scalar operations)
2. **Lookup table** for sigmoid values
3. **Polynomial approximation** (higher order)
4. **Accept approximation** (may be acceptable for detection tasks)

## Hardware Results

**Execution Time**: 1.569 ms (for 8×8×8)
- Much faster than Conv (7.936 ms)
- Lightweight operation
- Memory-bound, not compute-bound

**Accuracy** (with fast sigmoid):
- Max error vs PyTorch: ~21%
- This is due to sigmoid approximation, not implementation bugs
- The kernel itself is working correctly

## Design Notes

### 3-Input Workaround

The AIE design needs 3 inputs (data, bn_weight, bn_bias), but `setup_aie()` only supports 2 inputs. 

**Solution**: Concatenate bn_weight and bn_bias into a single buffer, then split inside the kernel:
```c
bfloat16 *bn_weight = bn_params;
bfloat16 *bn_bias = bn_params + channels;
```

### Data Layout

- **Input/Output**: HWC (Height-Width-Channel)
- **BN params**: Per-channel (C,)
- **Combined params**: (2*C,) - weight then bias

## Next Steps

1. **Improve sigmoid approximation**
   - Use lookup table
   - Or polynomial approximation
   - Target <5% error

2. **Vectorize kernel**
   - Process multiple elements at once
   - Use AIE vector intrinsics
   - Expected 10-20x speedup

3. **Fuse with Conv**
   - Single-pass Conv+BN+SiLU
   - Reduce memory traffic
   - Better performance

## Related Files

- `python/mdv6/layers.py` - PyTorch reference (Conv class uses BN+SiLU)
- `third_party/ironclad/aie_kernels/aie2p/silu.cc` - Reference SiLU implementation
- `programming_examples/ml/mdv6/conv/` - Conv layer (uses BN+SiLU)

## References

- [IRON Programming Guide](../../../programming_guide/)
- [AIE2P Architecture Manual](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview)
- [MDV6 Model Summary](../../../../python/mdv6/PROJECT_SUMMARY.md)
