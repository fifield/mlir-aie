# Vector Add bf16

Elementwise addition of two bf16 (bfloat16) vectors producing bf16 output.

## Status: PASS

## ISA Reference

The ISA defines VADD.f for floating-point vector addition and subtraction.

Uses the VMAC datapath in floating-point add mode. bf16 is selected via the
X/Y buffer f16 type configuration bits (r[5], r[6] = 1 for bf16).

## Kernel

- **Lanes**: 32 (512-bit register / 16-bit bf16)
- **Elements**: 256 bf16 values per invocation
- **API**: `aie::vector<bfloat16, 32>` with `aie::add()`
- **Output**: bf16 (same precision as input)

## Data Layout

Simple flat arrays:
- A[256]: 256 bf16 values = 512 bytes
- B[256]: 256 bf16 values = 512 bytes
- C[256]: 256 bf16 values = 512 bytes

## Test Pattern

Uses direct tile memory access (simple lock pattern, no DMA/shim).
bf16 ↔ float conversion in test harness via bit manipulation (truncate upper
16 bits of float32). Tolerance: 1 ULP in bf16.
