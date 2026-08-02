# MMAC fp8 Sparse Matrix Multiply (4:2 Structured Sparsity)

## Status: runs and passes on npu3 hardware

`make && ./test.exe matmul_fp8_sparse.elf`

- `aie4.mlir` + `test_sim.cpp` — original simulator design (preserved)
- `matmul_fp8_sparse_placed.py` + `test.cpp` — npu3 (medusa) hardware port

## ISA Reference

AIE4 ISA Specification v1.1:

- **14.6** "MMAC — fp8 sparse matrix multiplication"
  - Instruction: `MMAC.f <adst>, <asrc>, <xsrc>, <qf_zsrc>, <mode>`
  - xsrc = Y register (1x128 fp8 vector), qf_zsrc = QF_Zb register
  - Output: 1x64 fp32 accumulator
- **14.1.5** "Sparse matrix multiplication modes" — the 3-bit quartet encoding
- **2.7.2.5** "Sparsity Registers" — the 6-Kbit register layout

## Problem shape and data layout

```
C[4x64] += A[4x128] * B_sparse[128x64],  fp8 (E4M3) x fp8 -> fp32
```

| buffer | logical shape | bytes |
|--------|---------------|-------|
| A      | 4 x 128 dense fp8 activations | 512 |
| mask   | 192 x uint32 sparsity register image | 768 |
| B      | 64 x 64 stored fp8 matrix     | 4096 |
| C      | 4 x 64 fp32                   | 1024 |

The shape originally assumed by `test_sim.cpp` was wrong (it treated A as
128 pre-packed non-zeros and B as a 128x64 / 8192-byte matrix). Correctly:

- The activation operand is a `v128`, i.e. **128 dense logical values** per
  row. Logical K is 128.
- `load_staging()` consumes exactly 32 x `v128` = **4096 bytes** of B, i.e. a
  64 x 64 stored matrix. Anything beyond that is ignored.
- B is stored **column-major**: stored slot `k` of output column `n` lives at
  byte offset `n * 64 + k`.
- Sparsity is 4:2 per quartet: each group of 4 logical K positions keeps
  exactly 2, giving 32 quartets x 2 = 64 stored slots per column. Column
  `n`'s quartet `q` puts its two kept positions in stored slots `2q` and
  `2q+1`.

## Sparsity mask

Per ISA 14.1.5, which two of the four positions in a quartet are kept is
given by a **3-bit code, per column and per quartet**:

| code | selection |
|------|-----------|
| 001  | x[0], x[2] |
| 010  | x[0], x[3] |
| 011  | x[1], x[2] |
| 100  | x[1], x[3] |
| 101  | x[2], x[3] |
| else | x[0], x[1] |

Since the selection is per column, the register holds
64 columns x 32 quartets x 3 bits = **6 Kbit**. Per ISA 2.7.2.5 it is
addressed as six 1-Kbit horizontal row slices, each holding 16 bits per
column (two columns per 32-bit word); a quartet's 3 bits may straddle two
slices. `test.cpp::set_sparsity_code()` implements exactly this packing.

The kernel must write the register with the **two-argument**
`load_staging(B, mask)`, which stages the data and all six sparsity slices
together. The one-argument `insert_staging_sparsity()` writes only a single
1-Kbit slice and leaves the other five stale — which shows up as a selection
that is mostly, but not exactly, `x[0], x[1]`.

With the encoding implemented correctly the test uses an **independent random
4:2 pattern for every (column, quartet) pair** and checks a fully general
golden reference against it.

## Hardware entry point

An AIE4 compute tile has only two input DMA channels but the kernel needs
three inputs, so A and the mask are streamed as one contiguous 1280-byte AM
buffer and re-split by the additive `matmul_hw` wrapper at the end of
`kernel.cc`:

```
DDR --(shim MM2S ch0)--> inAM (A 512 B | mask 768 B) --> core(0,2) --> outC --(S2MM)--> DDR
DDR --(shim MM2S ch1)--> inB  (4096 B)               -->
```
