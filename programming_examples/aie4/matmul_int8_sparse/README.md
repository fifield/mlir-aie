# MMAC int8 Sparse Matrix Multiply (4:2 Structured Sparsity)

## Status: working on npu3 hardware

`make && make run` builds `matmul_int8_sparse.elf` and runs it under XRT.

Earlier revisions of this example were blocked because the kernel was written
against the AIE2 sparse API (`insert(v128int8_sparse, sparsity_t)`,
`concat` into `v256int8_sparse`, `mul_4x16_16x16`). Those built-ins do not
exist for AIE4. AIE4 drives its sparse MMAC through the **staging path**
instead, exactly like `matmul_fp8_sparse`:

```c
load_staging(pB, (v32uint32 *)mask);       // 4096 B of data + 6 Kbit of Q
staging_to_matrix_m128x64int8_sparse();    // expand into the sparse matrix reg
acc = mac_conf(Xbuff, 1, 1, acc, /*zero_acc=*/1);
```

## ISA Reference

According to the AIE4 ISA:

- `MMAC` performs 8-bit integer sparse matrix multiplication:
  - Instruction: `MMAC.i <adst>, <asrc>, <xsrc>, <qzsrc>, <mode>`
  - xsrc = Y register (1x128 int8 vector), qzsrc = QZb register
- Sparse matrix multiplication uses a 3-bit quartet encoding.
- Sparsity registers use a 6-Kbit register layout.

## Problem shape

```
C[4x64] += A[4x128] * B_sparse[128x64],  int8 x int8 -> int32
```

| operand | form | size |
|---|---|---|
| `A` | dense, 128 logical K per row | 512 B int8 |
| `B` | stored (compressed), 64 slots x 64 columns | 4096 B int8 |
| `mask` | sparsity register image | 768 B |
| `C` | | 1024 B int32 |

## Operand layout

- **The staging image is column-major**, 64 contiguous bytes per output
  column: stored slot `k` of output column `n` is at byte offset `n * 64 + k`.
  `load_staging()` consumes exactly 32 x `v128` = 4096 bytes.
- **A is dense** — the activation operand is a `v128`, so 128 logical K values
  per row, not 64 pre-packed non-zeros.
- **Sparsity is 4:2 per quartet**: column `n`'s quartet `q` (logical K
  positions `4q..4q+3`) keeps exactly two, which land in stored slots `2q`
  and `2q+1`.

## Sparsity mask

Per ISA, which two positions are kept is given by a **3-bit code, per
column and per quartet**:

| code | selection |
|------|-----------|
| 001  | x[0], x[2] |
| 010  | x[0], x[3] |
| 011  | x[1], x[2] |
| 100  | x[1], x[3] |
| 101  | x[2], x[3] |
| else | x[0], x[1] |

Because the selection is per column, the register is
64 columns x 32 quartets x 3 bits = **6 Kbit**. Per ISA it is
addressed as six 1-Kbit horizontal row slices, each holding 16 bits per
column (two columns packed per 32-bit word); a quartet's 3 bits may straddle
two slices. `test.cpp::set_sparsity_code()` implements exactly this packing.

The register must be written with the **two-argument**
`load_staging(B, mask)`, which stages the matrix data and all six sparsity
slices together. The one-argument `insert_staging_sparsity()` writes only a
single 1-Kbit slice, leaving the other five stale — the symptom is a
selection that is mostly, but not exactly, `x[0], x[1]`, with deterministic
per-column exceptions.

`test.cpp` uses an **independent random 4:2 pattern for every (column,
quartet) pair** and checks a fully general golden reference against it.

## Hardware data movement

An AIE4 compute tile has only two input DMA channels but the kernel needs
three inputs, so `A` and the mask are streamed as one contiguous buffer and
split again on the core by the `matmul_hw` wrapper at the bottom of
`kernel.cc`:

```
DDR --(shim MM2S ch0)--> inAM (A 512 B | mask 768 B) --> core(0,2) --> outC --(S2MM)--> DDR
DDR --(shim MM2S ch1)--> inB  (4096 B)               -->
```
