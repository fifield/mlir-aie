# Vector Add fp16

Elementwise addition of two fp16 (IEEE half-precision) vectors producing fp16 output.

## Status: PASS

## ISA Reference

ISA Spec Section 13.9: "VADD.f — Floating-point vector addition / subtraction"

Uses the VMAC datapath in floating-point add mode. fp16 is selected via the
X/Y buffer f16 type configuration bits (r[5], r[6] = 0 for fp16).

## Kernel

- **Lanes**: 32 (512-bit register / 16-bit fp16)
- **Elements**: 256 fp16 values per invocation
- **API**: `aie::vector<float16, 32>` with `aie::add()`
- **Output**: fp16 (same precision as input)

Note: AIE4 uses the `float16` type name for IEEE 754 half-precision, not `half`.

## Data Layout

Simple flat arrays:
- A[256]: 256 fp16 values = 512 bytes
- B[256]: 256 fp16 values = 512 bytes
- C[256]: 256 fp16 values = 512 bytes

## Test Pattern

Uses direct tile memory access (simple lock pattern, no DMA/shim).
fp16 ↔ float conversion in test harness via manual sign/exponent/mantissa
bit manipulation. Tolerance: 1 ULP in fp16.
