# CLAUDE.md — MDV6 IRON for AIE2P

## Goal
Implement the full MDV6-mit-yolov9-c (MegaDetector V6) wildlife detection model using IRON for AMD AIE2P (Strix Halo NPU). BFloat16 throughout.

## Hardware
- AMD Strix Halo with NPU (AIE2P, 6×8 tile array)
- Device target: `npu2` in aie2.py scripts
- XRT at `/opt/xilinx/xrt`, Peano toolchain for AIE kernel compilation

## References
- **PyTorch reference**: `~/npu-dev-mdv6/mdv6/mdv6/` (model.py, layers.py)
- **PyTorch layers for tests**: `mlir-aie/python/mdv6/layers.py` (same content, importable from test.py)
- **Working IRON examples**: `mlir-aie/programming_examples/basic/` (passthrough_kernel, vector_scalar_mul)
- **Working ML example**: `mlir-aie/programming_examples/ml/bottleneck/` (int8, uses DefaultNPURuntime)

## Build & Test Commands
```bash
# Source environment (required before all builds)
source ~/npu-dev-mdv6/env.sh

# Build a single layer
cd programming_examples/ml/mdv6/conv
make clean && make all    # Compile kernel + generate xclbin

# Run CPU-only reference test
make test

# Run on NPU hardware
make run

# Run all layers
for dir in */; do (cd "$dir" && make clean && make run 2>&1); done
```

## Per-Layer Structure
Each subdirectory contains:
- `aie2.py` — IRON program generating MLIR for AIE2P (imports from `aie.iron`)
- `*.cc` — BFloat16 AIE kernel C++ source (compiled with Peano/clang)
- `test.py` — PyTorch reference + NPU hardware comparison
- `Makefile` — Build targets: `all`, `test` (CPU), `run` (NPU), `clean`

## IRON API (installed mlir-aie)
```python
from aie.iron import (
    Kernel,
    Buffer,          # NOTE: was LocalBuffer in older API
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1
```

## XRT Runtime API (for test.py hardware execution)
```python
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime

# Load kernel
npu_kernel = NPUKernel(xclbin_path, insts_path, kernel_name="MLIR_AIE")
kernel_handle = DefaultNPURuntime.load(npu_kernel)

# Create buffers (bf16 data passed as uint16)
in1 = iron.tensor(input_uint16, dtype=np.uint16)
in2 = iron.tensor(weights_uint16, dtype=np.uint16)
out = iron.zeros(output_size, dtype=np.uint16)

# Execute
ret = DefaultNPURuntime.run(kernel_handle, [in1, in2, out])

# Read results
output_data = out.numpy()
```

## Kernel Compilation
```bash
# Peano clang for AIE2P kernels
${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -c kernel.cc -o kernel.o

# aiecc.py for xclbin generation
aiecc.py --aie-generate-xclbin --aie-generate-npu-insts \
    --no-compile-host --no-xchesscc --no-xbridge \
    --xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir
```

## MDV6 Model Architecture
Input: (1, 3, 640, 640) RGB bf16 → 3 detection scales

### Layer inventory
| Layer | Type | Dimensions | Count in model |
|-------|------|-----------|----------------|
| Conv | Conv2d+BN+SiLU | 1x1 and 3x3, various strides | ~80+ |
| AConv | AvgPool+Conv | downsample 2× | 5 |
| RepConv | dual-path 3x3+1x1→add→SiLU | same spatial | inside Bottleneck |
| Bottleneck | RepConv→Conv+residual | same spatial | inside RepNCSP |
| RepNCSP | CSP with Bottleneck | same spatial | inside RepNCSPELAN |
| ELAN | split→conv→conv→4-cat→conv | same spatial | 1 (elan2) |
| RepNCSPELAN | split→RepNCSP+Conv→4-cat→conv | same spatial | 7 |
| SPPELAN | Conv→3×MaxPool→cat→Conv | same spatial | 1 (spp9) |
| Upsample | nearest 2× | spatial 2× | 2 |
| Concat | channel dim | various | 4 |
| Detection | Conv chains + DFL | per-scale | 3 heads |

### Channel progression
```
Input(3) → Conv0(32) → Conv1(64) → ELAN(64) →
AConv(128) → RepNCSPELAN(128) [B3] →
AConv(192) → RepNCSPELAN(192) [B4] →
AConv(256) → RepNCSPELAN(256) [B5] →
SPPELAN(256) [N3] →
Upsample+Cat(448) → RepNCSPELAN(192) [N4] →
Upsample+Cat(320) → RepNCSPELAN(128) [P3] →
AConv+Cat(288) → RepNCSPELAN(192) [P4] →
AConv+Cat(384) → RepNCSPELAN(256) [P5] →
Detect([P3,P4,P5])
```

### Skip connections (must buffer)
- B3 (128ch, 80×80) → concat with upsampled N4
- B4 (192ch, 40×40) → concat with upsampled N3
- N3 (256ch, 20×20) → concat with downsampled P4
- N4 (192ch, 40×40) → concat with downsampled P3

## Current Status (2026-03-16)

### NPU hardware test results (8×8 test dimensions)
| Layer | Build | NPU Run | Notes |
|-------|-------|---------|-------|
| elementwise | PASS | **PASS** | max diff 0.031, 1.1ms |
| conv | PASS | **PASS** | max diff 0.008, 2.1ms |
| aconv | PASS | **PASS** | max diff 0.026, 1.7ms |
| batchnorm_silu | PASS | **PASS** | max diff 0.219, tolerance 0.25 |
| repconv | PASS | **PASS** | max diff < 0.3, stack_size=4096 |
| bottleneck | PASS | **PASS** | max diff < 0.35, stack_size=4096 |
| elan | PASS | **PASS** | max diff < 0.45, stack_size=4096 |
| repncsp | PASS | **PASS** | max diff < 0.4, stack_size=4096 |
| repncsp_elan | PASS | **PASS** | composed: 6 sub-layers, max diff 0.026, 31ms |
| sppelan | PASS | **PASS** | max abs error 0.035, stack_size=4096 |

### Key findings
- Default Worker stack_size is too small for kernels with deep nested loops. Use `stack_size=4096` (or 8192 for complex layers).
- Buffer objects must be created at top level and passed as Worker fn_args (not inside core_fn).
- repncsp_elan needs multi-tile or buffer-sharing design (too many Buffers for single tile).
- XRT buffer-backed numpy arrays become invalid after `iron.tensor`/`iron.zeros` are overwritten. Always `.copy()` slices before storing.
- Tiled conv at 640×640×3→32 works: 1600 tiles, 3.1s total, max diff 0.016.

### Resolved bugs
- `mlir-aie-595` (closed): LocalBuffer→Buffer rename + Worker arg pattern
- `mlir-aie-220` (closed): batchnorm_silu tolerance fixed (0.1→0.25)
- `mlir-aie-985` (closed): test.py setup_aie/execute stubs replaced with NPUKernel+DefaultNPURuntime
- `mlir-aie-lie` (closed): NaN output from Buffer layers — root cause was stack overflow

### Open bugs
None — all 10/10 layers pass on NPU.

### Tiled fused conv (key building block)
`aie2_tiled_fused.py` — generates tiled Conv+BN+SiLU xclbins for any dimension/stride:
```bash
# Args: npu2 H W C_in C_out kernel_size tile_h tile_w oc_block [n_patches] [stride]
python3 aie2_tiled_fused.py npu2 160 160 64 64 1 8 8 64 1 1  # 1x1 stride=1
python3 aie2_tiled_fused.py npu2 320 320 32 64 3 10 10 16 1 2 # 3x3 stride=2
```
Uses packed weights: [conv_weights, fused_bn_weight, fused_bn_bias]

### Host-orchestrated composition
Complex layers (RepNCSPELAN) that exceed 64KB tile memory are decomposed into sub-layer
NPU invocations orchestrated by the host. This uses fused Conv+BN+SiLU kernels
(`conv1x1_fused_packed_bf16`, `conv3x3_fused_packed_bf16`) that take packed weight buffers
with [conv_weights, fused_bn_weight, fused_bn_bias]. The fused BN params are pre-computed:
  bn_w_fused = gamma / sqrt(var + eps)
  bn_b_fused = beta - gamma * mean / sqrt(var + eps)

## Full Model Status
**End-to-end MDV6 forward pass PASS on NPU** (2026-03-17)
- max_class_diff=0.020, max_vector_diff=0.031 (vs PyTorch bf16 reference)
- Total: 96.3s (single-core scalar kernels, ~30 xclbin configs)
- On NPU: conv0, conv1, ELAN, all AConv, all RepNCSPELAN (×7), SPPELAN, Conv sub-layers
- On CPU: detection head (grouped conv + softmax), RepConv sub-layers, AvgPool, Upsample

## Performance Optimization Plan (`mlir-aie-mi7`)
| Phase | Optimization | Expected speedup | Target |
|-------|-------------|-----------------|--------|
| A (`mlir-aie-326`) | Vectorized bf16 kernels (aie::mmul) | 10-30× | 3-10s |
| B (`mlir-aie-1wy`) | Multi-core spatial parallelism (8+ cores) | 8× | 0.4-1.2s |
| C | On-chip pipelining (layer chaining) | 2-3× | <0.5s |
| D (`mlir-aie-9xq`) | Move RepConv/AvgPool/Upsample/detect to NPU | — | True E2E |

Priority: A > B > D > C

### Verified 30-core spatial layout
- 6 columns × 5 compute tiles = 30 cores
- DMA per column: shim 3/4, memtile 11/12, compute 2/2 — all fit
- Weight broadcast: 1 ObjectFifo → 5 consumers per column (saves 4 DMA channels)
- L1 memory: all operators verified ≤ 64KB
- Dataflow: 20×20 chains via L2 (200KB), larger go external
- Skip connections (B3/B4/N3/N4) must go external (1.6MB each)

### Test plan (bottom-up)
Level 0-1: Single tile scalar/vec ✓
Level 2: 2-tile weight broadcast (next step)
Level 3: 5-tile full column
Level 4: Operator chain L1→L2→L1
Level 5: 30-tile spatial parallel
Level 6: Full model integration

## Completed Phases
1. Phase 1: All 10 layer types pass on NPU at 8×8 test dims ✓
2. Phase 2: Tiled conv at model dims (stride-1 and stride-2) ✓
3. Phase 3-6: All backbone/neck/head layers at model dims ✓
4. Phase 7: Full model integration and end-to-end validation ✓

## Data Layout Conventions
- PyTorch: NCHW (batch, channels, height, width)
- AIE kernels: HWC (height, width, channels) — no batch dimension
- Weights stored as bf16 packed into uint16 arrays
- BN parameters: [weight, bias, running_mean, running_var] concatenated after conv weights

## Numerical Precision Notes
- BFloat16 has ~7 bits mantissa (vs fp32's 23)
- SiLU uses fast sigmoid approximation: σ(x) ≈ 0.5 × (x/(1+|x|) + 1)
- Expected max error per SiLU: ~0.21
- Expected max error per BN: ~0.01 (sqrt approximation)
- Cumulative error grows with layer depth; tolerances scale accordingly

## Beads Issue Tracking
Track work with `bd` (beads). Run `bd ready` to find available work, `bd list` to see all issues.
Epic: `mlir-aie-w7q` (MDV6 IRON implementation for AIE2P)
