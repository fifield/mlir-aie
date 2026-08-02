<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>AIE4 Compute Examples</ins>

These examples exercise the AIE4 (npu3) compute pipeline — the vector unit (VMAC/VADD) and matrix unit (MMAC) — across the data types AIE4 adds on top of AIE2: standard integer and floating point, plus the MicroXcaling (MX) block-scaled formats (MXFP8/6/4, MXINT8).

Every example builds and runs two ways:
* **Simulator**: `aie4.mlir` + `test_sim.cpp`, driven through `aiecc --get-aiesim`.
* **Hardware**: `<name>_placed.py` + `test.cpp`, driven through a `Makefile` that builds `<name>.elf` via `aiecc --get-full-elf` and runs it on npu3 (medusa) hardware over XRT.

A shared CPU reference library lives in [`cpu_ref/`](./cpu_ref) (`int_types.h`, `fp_types.h`, `mx_types.h`, `sparse_types.h`, `reference_ops.h`) and is used by every example's host test to check results.

## Elementwise Multiply (VMAC)

* [vmul_int8x8](./vmul_int8x8) - int8 × int8 → int32 acc, 32 lanes.
* [vmul_int16x16](./vmul_int16x16) - int16 × int16 → int32 acc, 32 lanes.
* [vmul_int32x16](./vmul_int32x16) - int32 × int16 → int64 acc, 16 lanes.
* [vmul_int32x32](./vmul_int32x32) - int32 × int32 → int64 acc, 16 lanes.
* [vmul_fp8](./vmul_fp8) - fp8 × fp8 → fp32 acc, 32 lanes.
* [vmul_fp16](./vmul_fp16) - fp16 × fp16 → fp32 acc, 32 lanes.
* [vmul_bf16](./vmul_bf16) - bf16 × bf16 → fp32 acc, 32 lanes.
* [vmul_bf16_64lane](./vmul_bf16_64lane) - bf16 × bf16 → bf16, 64 lanes, multiply-only.
* [vmul_fp32](./vmul_fp32) - fp32 × fp32 → fp32 acc, 16 lanes.
* [vmul_int32_fp32](./vmul_int32_fp32) - int32 × fp32 → fp32 acc, 16 lanes.

## Vector Add (VADD)

* [vadd_int8](./vadd_int8) - 128 lanes.
* [vadd_int16](./vadd_int16) - 64 lanes.
* [vadd_int32](./vadd_int32) - 32 lanes.
* [vadd_fp32](./vadd_fp32) - 32 lanes.

## Matrix Multiply (MMAC)

* [matmul_int4](./matmul_int4) - int4 × int4 → int32, 1×128 · 128×64, 8192 mults/cycle.
* [matmul_int8](./matmul_int8) - int8 × int8 → int32, 1×64 · 64×64, 4096 mults/cycle.
* [matmul_int8_sparse](./matmul_int8_sparse) - sparse int8, 4:2 structured sparsity, 1×128 · 128×64(sp), 8192 mults/cycle.
* [matmul_fp8](./matmul_fp8) - fp8 × fp8 → fp32, 1×64 · 64×64, 4096 mults/cycle.
* [matmul_bf16](./matmul_bf16) - bf16 × bf16 → fp32, 1×32 · 32×64, 2048 mults/cycle.
* [matmul_fp8_sparse](./matmul_fp8_sparse) - sparse fp8, 4:2 structured sparsity, 1×128 · 128×64(sp), 8192 mults/cycle.
* [matmul_mxfp8](./matmul_mxfp8) - MXFP8 × MXFP8 → fp32, 1×64 · 64×64, 4096 mults/cycle.
* [matmul_mxfp6](./matmul_mxfp6) - MXFP6 × MXFP6 → fp32, 1×64 · 64×64, 4096 mults/cycle (processed as MXFP8).
* [matmul_mxfp4](./matmul_mxfp4) - MXFP4 × MXFP4 → fp32, 1×64 · 64×64, 4096 mults/cycle.
* [matmul_mxint8](./matmul_mxint8) - MXINT8 × MXINT8 → fp32, 1×64 · 64×64, 4096 mults/cycle.

---

## Architecture Quick Reference

### Data Type Formats
| Format | Bits | Layout | Bias | Range | Notes |
|--------|------|--------|------|-------|-------|
| fp8 E4M3 | 8 | s1+e4+m3 | 7 | [-448, 448] | OCP standard. No inf, NaN=0x7F/0xFF. Subnorms → 0 on MAC input |
| fp16 IEEE | 16 | s1+e5+m10 | 15 | [-65504, 65504] | Standard IEEE half |
| bf16 | 16 | s1+e8+m7 | 127 | same as fp32 | Truncated fp32 mantissa |
| fp6 E2M3 | 6 | s1+e2+m3 | 1 | [-7.5, 7.5] | AIE4b only, no inf/NaN |
| fp4 E2M1 | 4 | s1+e2+m1 | 1 | [-6, 6] | AIE4b only, no inf/NaN |
| MXFP8 | 8+scale | E4M3 per element, E8M0 per block(32) | 127 (scale) | — | `val = fp8(elem) * 2^(scale-127)` |
| MXFP6 | 6+scale | E2M3 per element, E8M0 per block(32) | 1/127 | — | Processed internally as MXFP8 |
| MXFP4 | 4+scale | E2M1 per element, E8M0 per block(32) | 1/127 | — | 2 elements per byte, 17B per block |
| MXINT8 | 8+scale | S8 (6 frac bits) per element, E8M0 per block(8) | 127 | [-2, 1.984] | Max neg = -2 (asymmetric) |

### Hardware Dimensions
| Operation | Data Types | Matrix A | Matrix B | Output | Mults/cycle |
|-----------|-----------|----------|----------|--------|-------------|
| MMAC int4×4 | int4 | 1×128 | 128×64 | 1×64 (int32) | 8192 |
| MMAC int8×8 | int8 | 1×64 | 64×64 | 1×64 (int32) | 4096 |
| MMAC fp8×fp8 | fp8 | 1×64 | 64×64 | 1×64 (fp32) | 4096 |
| MMAC bf16×bf16 | bf16 | 1×32 | 32×64 | 1×64 (fp32) | 2048 |
| MMAC MXFP8/6/4 | MX | 1×64 | 64×64 | 1×64 (fp32) | 4096 |
| MMAC MXINT8 | MX | 1×64 | 64×64 | 1×64 (fp32) | 4096 |
| MMAC sparse int8 | int8 | 1×128 | 128×64(sp) | 1×64 (int32) | 8192 |
| MMAC sparse fp8 | fp8 | 1×128 | 128×64(sp) | 1×64 (fp32) | 8192 |
| VMAC int8 | int8 | — | — | 32 lanes (int32) | 32 |
| VMAC int16 | int16 | — | — | 32 lanes (int32) | 32 |
| VMAC int32×16 | int32,int16 | — | — | 16 lanes (int64) | 16 |
| VMAC int32 | int32 | — | — | 16 lanes (int64) | 16 |
| VMAC fp8 | fp8 | — | — | 32 lanes (fp32) | 32 |
| VMAC fp16/bf16 | fp16/bf16 | — | — | 32 lanes (fp32) | 32 |
| VMAC bf16 64-lane | bf16 | — | — | 64 lanes (bf16) | 64 (MUL only) |
| VMAC fp32 | fp32 | — | — | 16 lanes (fp32) | 16 |
| VMAC int32×fp32 | int32,fp32 | — | — | 16 lanes (fp32) | 16 |
| VADD int8 | int8 | — | — | 128 lanes | — |
| VADD int16 | int16 | — | — | 64 lanes | — |
| VADD int32 | int32 | — | — | 32 lanes | — |

### Key Intrinsics
- **MMAC setup**: `insert_staging()` → `staging_to_matrix_m64x64int8()` / `staging_to_matrix_m64x64mxfp8()`
- **MMAC execute**: `mac_conf(vector, sign_x, sign_z, acc, zero_acc)` (int8) or `mac_conf(vector, acc, zero_acc)` (MX/fp)
- **VMAC**: `aie::mul(a, b)` / `aie::mac(acc, a, b)` via aie_api
- **VADD**: `aie::add(a, b)` / `aie::sub(a, b)` via aie_api
- **MX FIFO load**: `fifo_ld_pop_staging()` → `staging_to_matrix_m64x64mxfp8()`
- **Accumulators**: `v64acc32` (int), `v64accfloat` (fp), `undef_v64acc32()`/`undef_v64accfloat()` to init
- **SRS/UPS**: `aie::vector_cast<>()` or SRS/UPS intrinsics for acc→vector conversion

### Verification Tolerances
- Integer results: **exact match**
- fp32 from fp8/bf16 inputs: **relative tolerance 1e-3**
- fp32 from MX formats: **tolerance depends on element precision** (1e-2 for MXFP4, 1e-3 for MXFP8)
- Native fp32 operations: **1 ULP**

---

## Hardware Port (npu3 / medusa, XRT)

Every example has a hardware flow alongside the simulator flow:

| file | role |
|---|---|
| `kernel.cc` | compute kernel, shared by both flows |
| `<name>_placed.py` | placed object-FIFO design (shim -> compute tile -> shim) |
| `Makefile` | builds `<name>.elf` via `aiecc --get-full-elf`, plus `test.exe` |
| `test.cpp` | XRT host: fills BOs, launches `main:sequence`, checks against `cpu_ref` |
| `test_sim.cpp` | the original simulator host (was `test.cpp`) |
| `aie4.mlir` | the original simulator design; its RUN line now points at `test_sim.cpp` |

Build and run one example:

```
make            # <name>.elf + test.exe
make run        # ./test.exe <name>.elf
```

Each example's hardware flow is exercised in CI on npu3 hardware via its
`run_npu3_makefile.lit` (`REQUIRES: ryzen_ai_npu3, chess`), independent of
the `aie4.mlir` simulator RUN lines above.

Design notes:

- A single compute tile at (0, 2) in column 0. Inputs arrive on two shim MM2S
  channels, the result drains on one S2MM channel. The `controller_id`
  attribute on the shim tile and the `Control32 -> Controller32` packetflow
  are required npu3 CERT boilerplate.
- A compute tile has only **two** input DMA channels. The two sparse examples
  need three inputs, so they stream the activations and the sparsity mask as
  one contiguous buffer and split it again on the core with an additive
  `matmul_hw` wrapper at the bottom of their `kernel.cc`.
- The two sparse examples follow AIE4 ISA Spec for Sparsity Registers and
  Sparse matrix multiplication modesS:
  a 3-bit code per column per quartet selects which 2 of every 4 logical K
  positions are stored, so the sparsity register is 6 Kbit and must be
  written with the two-argument `load_staging(B, mask)` -- the one-argument
  `insert_staging_sparsity()` fills only one of its six 1-Kbit row slices.
  Both tests drive an independent random 4:2 pattern per (column, quartet).
  See `matmul_int8_sparse/README.md`.
