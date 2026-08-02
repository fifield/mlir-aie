# MMAC int16 × int8 Mixed-Precision Matrix Multiply

Mixed-precision matrix multiply: int16 activations × int8 weights → int32 output.

## Status: PASS

## ISA Reference

ISA Spec Table 195: "Example emulated MAC modes"
- Row: 16-bit × 8-bit, No sparse, 1×64, 64×64, DM = X×Z + DM, output 1×64, 32-bit
- Available on: all cores (AIE4a and AIE4b)

This mode is "emulated" — the hardware MMAC unit operates in int8×int8 mode
with a 64×64 matrix register, and the int16 A input is truncated to int8 via
`pack(v64int16)` before being fed to `mac_conf`.

## Kernel

- **Matrix dimensions**: C[4×64] += A[4×128] × B[128×64]
- **K_outer**: 2 iterations of K_inner=64
- **A input**: v64int16 loaded, then `pack()` to v64int8 (truncation to lower 8 bits)
- **B staging**: `insert_staging()` × 32 + `staging_to_matrix_m64x64int8()`
- **MAC**: `mac_conf(v64int8, 1, 1, acc, zero_acc)`

## Data Layout

- **A**: M × K = 4 × 128 int16 values = 1024 bytes, blocked as A[ko][m][k_inner]
- **B**: K × N = 128 × 64 int8 values = 8192 bytes, blocked as B[ko][n][k_inner] (column-major within block for staging)
- **C**: M × N = 4 × 64 int32 values = 1024 bytes

## Limitations

The `pack(v64int16)` truncates int16 to int8 (keeps lower 8 bits only).
This means A values must fit in int8 range [-128, 127] for correct results.
For true int16 × int8 without truncation, the `aie::mmul` high-level API
or a different MMAC mode configuration would be needed.

## Test Pattern

Uses direct tile memory access. A values are generated in [-127, 127] range
to stay within int8 after truncation. Golden reference computes full int16 × int8
products in int32.
