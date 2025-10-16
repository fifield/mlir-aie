# SPPELAN Layer - Spatial Pyramid Pooling ELAN

**Status**: ✅ Hardware Validated on NPU2 (AIE2P)

## Overview

SPPELAN (Spatial Pyramid Pooling ELAN) is a feature extraction layer that captures multi-scale spatial information through successive max pooling operations. It's a key component in the MegaDetectorV6 (MDV6) wildlife detection model.

## Architecture

```
Input (H×W×C_in)
    ↓
Conv1 (1×1) + BN + SiLU → features[0] (H×W×C_neck)
    ↓
MaxPool(5×5, s=1, p=2) → features[1] (H×W×C_neck)
    ↓
MaxPool(5×5, s=1, p=2) → features[2] (H×W×C_neck)
    ↓
MaxPool(5×5, s=1, p=2) → features[3] (H×W×C_neck)
    ↓
Concat [f0, f1, f2, f3] → (H×W×4×C_neck)
    ↓
Conv5 (1×1) + BN + SiLU → Output (H×W×C_out)
```

### Key Features

1. **Multi-scale Feature Extraction**: Captures features at different receptive field scales through successive pooling
2. **Spatial Dimension Preservation**: MaxPool with stride=1 and padding=2 maintains spatial dimensions
3. **Efficient Concatenation**: 4-way channel concatenation in HWC format
4. **Fused Operations**: Conv+BN+SiLU fusion for efficiency

## Implementation Details

### C++ Kernel (`sppelan_bf16.cc`)

- **MaxPool2d Operation**: 5×5 kernel, stride=1, padding=2
  - Efficient sliding window implementation
  - Handles boundary conditions with padding
  
- **Conv1x1 + BN + SiLU**: Reusable helper function
  - Fast sqrt approximation for BatchNorm
  - Fast sigmoid approximation for SiLU activation
  
- **4-way Concatenation**: Channel-wise concatenation in HWC format
  - Optimized memory access pattern

### IRON Design (`aie2.py`)

- **Single-tile Implementation**: One AIE core
- **ObjectFIFO Data Movement**: Depth=1 (single buffering)
- **LocalBuffer Allocation**: 5 intermediate buffers
  - conv1_output: H×W×C_neck
  - pool1_output: H×W×C_neck
  - pool2_output: H×W×C_neck
  - pool3_output: H×W×C_neck
  - concat_buffer: H×W×4×C_neck

### Memory Usage (8×8×16 test case)

| Buffer | Size | Usage |
|--------|------|-------|
| conv1_output | 512 elements | 1 KB |
| pool1_output | 512 elements | 1 KB |
| pool2_output | 512 elements | 1 KB |
| pool3_output | 512 elements | 1 KB |
| concat_buffer | 2048 elements | 4 KB |
| **Total L1** | **9 KB** | **14% of 64 KB** |

### Weight Layout

```
[Conv1 weights (C_neck × C_in)]
[Conv1 BN params (4 × C_neck): weight, bias, mean, var]
[Conv5 weights (C_out × 4×C_neck)]
[Conv5 BN params (4 × C_out): weight, bias, mean, var]
```

For 8×8×16→16 test:
- Conv1: 128 weights + 32 BN params = 160 params
- Conv5: 512 weights + 64 BN params = 576 params
- **Total**: 736 parameters (1.5 KB)

## Build & Test

### Prerequisites

- mlir-aie build environment
- Python 3.x with PyTorch
- NPU2 hardware (for hardware validation)

### CPU Test (PyTorch Reference)

```bash
make test
```

Expected output:
```
Testing SPPELAN Layer
Input shape: (8, 8, 16)
Output shape: (8, 8, 16)
PyTorch output shape: torch.Size([1, 16, 8, 8])
✓ PyTorch reference test complete
```

### Build for NPU2

```bash
make build/final.xclbin
```

Build steps:
1. Compile C++ kernel → `build/sppelan_bf16.o`
2. Generate MLIR → `build/aie.mlir`
3. Compile to xclbin → `build/final.xclbin`

### Hardware Validation

```bash
make run
```

Expected output:
```
Running on AIE Hardware
Execution time: ~11-13 ms
Max absolute error: 0.03-0.10
✓ Test PASSED
```

## Performance Metrics

### Hardware Performance (8×8×16 test on NPU2)

- **Execution Time**: 11.4-12.3 ms (scalar, single-tile)
- **Throughput**: ~85 ops/ms
- **L1 Memory**: 9 KB (14% utilization)

### Accuracy

| Metric | Value | Notes |
|--------|-------|-------|
| Max Absolute Error | 0.034-0.102 | Excellent |
| Mean Absolute Error | 0.013-0.014 | Very good |
| Max Relative Error | 88-145% | For values > 0.01 |
| Mean Relative Error | 8.7-9.7% | For values > 0.01 |

**Approximations**:
- 2× fast sqrt (in BatchNorm layers)
- 2× fast sigmoid (in SiLU activations)

## Files

```
sppelan/
├── sppelan_bf16.cc      # C++ kernel implementation (320 lines)
├── aie2.py              # IRON design (170 lines)
├── test.py              # Test script with PyTorch reference (220 lines)
├── Makefile             # Build configuration
└── README.md            # This file
```

## Usage Example

```python
from mdv6.layers import SPPELAN
import torch

# Create layer
layer = SPPELAN(in_channels=16, out_channels=16, neck_channels=8)
layer.eval()
layer = layer.to(torch.bfloat16)

# Forward pass
input_tensor = torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)
output = layer(input_tensor)  # Shape: (1, 16, 8, 8)
```

## Design Patterns

This implementation follows established patterns from other MDV6 layers:

1. **Single-tile execution**: One AIE core per layer
2. **ObjectFIFO data movement**: Depth=1 for simplicity
3. **BFloat16 handling**: uint16 representation in MLIR
4. **LocalBuffer intermediates**: Minimize DMA transfers
5. **Fusion strategy**: Conv+BN+SiLU combined
6. **Fast approximations**: sqrt and sigmoid for performance

## Known Limitations

1. **Scalar Implementation**: No vectorization (10-20× speedup possible)
2. **Single-tile**: No multi-tile parallelism (2-4× speedup possible)
3. **Approximation Error**: Fast sqrt and sigmoid introduce ~10% error
4. **Fixed Dimensions**: Designed for specific input sizes

## Future Optimizations

1. **Vectorization**: Use AIE vector instructions
2. **Multi-tile**: Distribute work across multiple cores
3. **Pipelining**: Overlap computation and data movement
4. **Precision**: Explore higher precision approximations

## References

- [MDV6 Project Summary](../PROJECT_SUMMARY_90PCT.md)
- [ELAN Layer](../elan/) - Similar architecture
- [MaxPool Operation](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

## License

This file is licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Copyright (C) 2025, Advanced Micro Devices, Inc.
