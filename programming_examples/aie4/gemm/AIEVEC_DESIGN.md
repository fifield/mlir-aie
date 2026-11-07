<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# GEMM core in flat AIEVec

`KERNEL=aievec` emits the core computation in `gemm_placed.py` using
`stage_stream`, `commit`, `mmac_active` and `srs`. Peano compiles the core without
an external `kernel.o`. Placement, DMA, routing, locks and the host reference are
shared with the C kernel. See [README.md](README.md) for usage and performance
comparisons.

## Build

```bash
make check_aievec KERNEL=aievec M=512 N=768 K=1024 OUT=i8 CT=4
make run KERNEL=aievec M=512 N=768 K=1024 OUT=i8 CT=4
```

Set `PEANO_INSTALL_DIR` or `PEANO` to the Peano installation. The generator takes
`--kernel aievec` and `--shift` (0–59, default 12). `SHIFT` controls the i8 output
conversion; i32 output stores the accumulators directly.

The AIEVec path requires `C_REASM=1` and `K >= 128`, in addition to the example's
shape and memory constraints. `KERNEL_CC` on the make command line and `TIMING=1`
are rejected. Use `TRACE_SIZE` for timing.

The default `AIEVEC_LLC_ARGS` are:

```text
-aie-force-postpipeliner -aie-postpipeliner-maxii=80
```

The Makefile passes each through `aiecc --peano-llc-arg`. Default builds use an
`_aievec` suffix; non-default flag sets get separate build directories. AIE4
expansion, lint and LLVM lowering run in the normal aiecc pipeline; this path
needs no `--enable-aie4-vectorization` flag. aiecc also supplies the staging-load
combine workaround when required by the IR.

## Data types

The core is generated for every `DTYPE` (`i8`, `u8`, `i8u8`, `u8i8`, `i4`, `fp8`, `bf16`),
with outputs `i8`/`u8` (SRS), `i32` and, for the float types, `f32`. `K` below is the A row
length in bytes, so `KT = K/64` for every type. `fmt`, `lhs_signed` and `rhs_signed` select the
MMAC format and X/Z signs: `INT8` with (T,T), (F,F), (T,F), (F,T) for `i8`, `u8`, `i8u8`,
`u8i8`; `INT4` for `i4` with a `vector<128xi4>` row; `FP8` and `BF16` for the float types with
`vector<64xf8E4M3FN>` and `vector<32xbf16>` rows.

`zero_if` is integer-only, so the float cores peel K-tile 0 (an MMAC without accumulator) and
loop over K-tiles `1..KT-2`: three K-tile bodies per tile body, and no loop at `KT=2`. There is
no f32 narrowing op in the dialect, so fp8 and bf16 produce `f32` only (`conv_f32_bf16`
narrows with whatever rounding mode is ambient and nothing here sets it). Peano spills about a
quarter of the float accumulators around the last K-tile; `check_aievec_core.py --allow-spills`
accepts that for the float types and counts the rest exactly.

## Core structure

Each core computes 64×64 output tiles. Let `KT = K/64`, `NT = N/(COLS*64)` and
`NG = NT/CT`. A strip is one core's 64-row slice of an M-block, with `NT` output
tiles grouped into `NG` C buffers of `CT` tiles each.

For each strip, the core:

1. Acquires its A buffers and primes the 32 W staging slots.
2. Runs a runtime loop over the first `NT-1` output tiles. Each tile prefetches
   the first W K-tile of the next output tile.
3. Computes the final output tile without that prefetch, then releases A.

The emitted code has two tile bodies, or one when `NT=1`. Runtime loops and
buffer selection keep code size independent of the number of output tiles.

Each output tile carries 64 `vector<64xi32>` accumulators through a K-loop over
K-tiles `0..KT-2`, followed by a peeled final K-tile. Each K-tile commits W,
performs 64 row MMACs, and, when prefetching, reads one W chunk after every odd
row into slots 0–31. Trace events 0 and 1 bracket the tile computation and stores.

The accumulators start as `ub.poison`. `zero_if(kt == 0)` ignores the incoming
accumulator on the first iteration, avoiding explicit zero-vector setup. This
requires `KT >= 2`: otherwise the peeled final K-tile would consume poison.
With `KT=2`, canonicalization inlines the single loop iteration and folds
`zero_if` to true. Peano spills accumulators in this case, so the core stack is
4 KiB instead of the usual 2 KiB.

The original Python MLIR-builder sketches below show the imports, tile body
and output conversion. They use indexed A loads; the current implementation
carries A pointers as described under [Buffer layout](#buffer-layout-and-synchronization).
Surrounding setup and small helper definitions are omitted.

```python
from aie.dialects import arith as arith_d, scf as scf_d, ub as ub_d, vector as vector_d
from aie.ir import IndexType, InsertionPoint, IntegerType, VectorType
from aie.helpers.dialects.scf import _for as range_, if_, yield_
from aie.helpers.dialects.aievec_aie4 import (
    INT8, POS_INF, SAT, commit, mmac_active, srs, stage_stream)

i32 = IntegerType.get_signless(32)
V64I8, V16I32, V64I32 = (VectorType.get([64], IntegerType.get_signless(8)),
                         VectorType.get([16], i32), VectorType.get([64], i32))
KT = K // TILE
KT_WORDS = (TILE // a_split) * TILE // 4     # words per K-tile in one A part: 512 (split 2), 1024 (split 1)
C_ROW_WORDS = ct * TILE * out_bytes // 4     # C row stride: CT*16 (i8), CT*64 (i32)
C_TILE_WORDS = TILE * out_bytes // 4         # C tile step j: 16 (i8), 64 (i32)
```

```python
def idx(v): return arith_d.constant(IndexType.get(), v)

def a_row(a_lo, a_hi, kt_base, r):                 # kt_base: index Value, kt*KT_WORDS
    buf, rr = (a_hi, r - 32) if (a_split == 2 and r >= 32) else (a_lo, r)
    w = vector_d.load(V16I32, buf, [arith_d.addi(kt_base, idx(rr * 16))], alignment=64)
    return vector_d.bitcast(V64I8, w)

def ktile(a_lo, a_hi, kt_base, accs, *, zero_if, prefetch):
    commit(INT8, rhs_signed=True)
    out = []
    for r in range(64):                                         # Python-unrolled
        out.append(mmac_active(INT8, a_row(a_lo, a_hi, kt_base, r), accs[r],
                               zero_if=zero_if, lhs_signed=True, rhs_signed=True))
        if prefetch and r & 1:
            stage_stream(INT8, W_STREAM_PORT, r // 2, rhs_signed=True)
    return out

def tile(a_lo, a_hi, c_buf, j_words, *, prefetch_last):
    if trace_size:
        event(0)
    init = [ub_d.poison(V64I32)] * 64                           # acc is ignored at kt == 0
    for kt, accs, fin in range_(0, KT - 1, iter_args=init, insert_yield=False):
        yield_(ktile(a_lo, a_hi, arith_d.muli(kt, idx(KT_WORDS)), accs,
                     zero_if=arith_d.cmpi(arith_d.CmpIPredicate.eq, kt, idx(0)),
                     prefetch=True))
    last = ktile(a_lo, a_hi, idx((KT - 1) * KT_WORDS), fin,
                 zero_if=None, prefetch=prefetch_last)
    for r in range(64):
        off = arith_d.addi(j_words, idx(r * C_ROW_WORDS))
        if out_i8:
            y = srs(last[r], shift=shift, signed=True, rounding=POS_INF, saturation=SAT)
            vector_d.store(vector_d.bitcast(V16I32, y), c_buf, [off], alignment=64)
        else:
            vector_d.store(last[r], c_buf, [off], alignment=64)
    if trace_size:
        event(1)
```

For i8 output, SRS uses explicit `POS_INF` rounding and `SAT` saturation. Lowering
writes `crRnd` and `crSat` once at core entry, and SRS fuses with the stores. The
core has no external calls that could clobber these modes. i32 output needs no
SRS.

## Buffer layout and synchronization

L1 buffers remain i32 memrefs to preserve DMA lengths. A rows use 64-byte-aligned
vector loads and bitcasts to int8; C stores are also 64-byte-aligned. Omitting
load alignment can cause scalar byte loads.

A is laid out as `[kt][rows per part][64 int8]`. With `ASPLIT=2`, each part holds
32 rows; with `ASPLIT=1`, one part holds all 64. The K-loop carries one pointer
per A part and advances it by 2048 or 4096 bytes per iteration, respectively.
Within a part, row `r` starts at byte offset `64*r`. Carrying pointers lets Peano
use post-increment loads without rebuilding the K-tile address from the loop
counter. The `ptr` dialect survives core conversion and lowers to LLVM later.

For C row `r` and tile `j` within a buffer, the offset in i32 words is:

```text
C_TILE_WORDS = 64 * output_bytes / 4
C_ROW_WORDS  = CT * C_TILE_WORDS
offset      = j * C_TILE_WORDS + r * C_ROW_WORDS
```

### C parity and locks

For global M-block index `m` and group `g`, the C ping/pong buffer is:

```text
(m * NG + g) & 1
```

The outer core loop keeps counting across GEMMs and host launches, matching the
persistent MM2S BD chain. For odd `NG`, group 0 changes buffer each strip; for
even `NG`, its buffer is fixed. An `scf.if` selects the memref when parity is
dynamic.

For output-tile index `t`, `g = t/CT` and `j = t%CT`. The core acquires the
producer lock at `j=0` and releases the consumer lock after `j=CT-1`. The peeled
strip-end tile acquires its own lock only when `CT=1`; otherwise the first tile
of its group already acquired it.

The original core-loop sketch makes the parity and lock guards explicit. It
shows the normal 2 KiB integer stack; the implementation selects 4 KiB for
integer `KT=2`. Floating-point stacks are sized by `aiecc` from the compiled
core's spill requirements and runtime frame.

```python
bufs, prod, cons = c_l1[c][r]

def c_select(sel_i32):                                # sel: 0 -> bufs[0], 1 -> bufs[1]
    if isinstance(sel_i32, int):
        return bufs[sel_i32].result
    op = scf_d.IfOp(arith_d.cmpi(arith_d.CmpIPredicate.ne, sel_i32, arith_d.constant(i32, 0)),
                    [bufs[0].result.type], has_else=True)
    with InsertionPoint(op.then_block):
        scf_d.yield_([bufs[1].result])
    with InsertionPoint(op.else_block):
        scf_d.yield_([bufs[0].result])
    return op.results[0]

def xor_parity(a, b):                                 # each: int, or i32 Value, or None (= 0)
    ...                                               # folds ints in Python, emits arith.xori otherwise

@core(cores[c][r], stack_size=0x800)
def _():
    for m in range_(sys.maxsize):                     # one M-block (strip) per iteration
        a = [f.acquire(ObjectFifoPort.Consume, 1) for f in a_core[c][r]]
        a_lo, a_hi = a[0], a[-1]
        for i in range(32):                           # prime the strip
            stage_stream(INT8, W_STREAM_PORT, i, rhs_signed=True)
        base = (arith_d.andi(arith_d.index_cast(i32, m), arith_d.constant(i32, 1))
                if NG % 2 else None)                  # parity of this strip's group 0
        n_loop = NG * ct - 1
        if n_loop:                                    # all tiles but the strip's last; with prefetch
            for t in range_(0, n_loop):
                g = arith_d.divui(t, idx(ct)); j = arith_d.remui(t, idx(ct))
                if ct > 1:
                    with if_(cmp_eq(j, 0), hasElse=False):
                        use_lock(prod, LockAction.AcquireGreaterEqual, 1)
                else:
                    use_lock(prod, LockAction.AcquireGreaterEqual, 1)
                g_par = None if NG == 1 else arith_d.andi(index_cast_i32(g), const_i32(1))
                tile(a_lo, a_hi, c_select(xor_parity(base, g_par)),
                     arith_d.muli(j, idx(C_TILE_WORDS)), prefetch_last=True)
                if ct > 1:
                    with if_(cmp_eq(j, ct - 1), hasElse=False):
                        use_lock(cons, LockAction.Release, 1)
                else:
                    use_lock(cons, LockAction.Release, 1)
        if ct == 1:                                   # else group NG-1 was acquired at t = (NG-1)*ct
            use_lock(prod, LockAction.AcquireGreaterEqual, 1)
        tile(a_lo, a_hi, c_select(xor_parity(base, (NG - 1) & 1)),
             idx((ct - 1) * C_TILE_WORDS), prefetch_last=False)   # peeled strip end
        use_lock(cons, LockAction.Release, 1)
        for f in a_core[c][r]:
            f.release(ObjectFifoPort.Consume, 1)
```

Lock values remain plain constants. A FIFOs have depth 1, and the output-tile
and K-loops contain no objectfifo acquire/release operations. These choices
prevent objectfifo lowering from unrolling or peeling the large tile body.
`C_REASM=0` is excluded because its C FIFO acquisitions would change that loop
structure.

### W stream accounting

The number of 1024-bit W reads per strip is:

```text
32 + NT*(KT-1)*32 + (NT-1)*32 = NT*KT*32
```

The terms are the prime, K-loop prefetches and final-K-tile prefetches. This
matches DMA replay exactly, so each strip drains its W input. The final tile
must omit prefetch to avoid consuming the next strip's data.

The prime defines all staging slots before the first commit. Each prefetching
K-tile refills them for the next commit. Lock guards and buffer-selection
branches preserve matrix state; the next strip's prime overwrites stale slots.
These invariants also let the AIE4 lint validate the runtime loops.

## Compiler scheduling

The default flags software-pipeline the K-loop at an initiation interval of
64–65 bundles for 64 MMACs. Raising the post-pipeliner's maximum interval above
its default of 64 permits the 65-bundle schedule.

The generated schedule can issue a commit before the previous K-tile's final
MMACs, and before some of its own staging inserts. Correctness depends on when
those instructions read and write the active and staging matrices. The
strip-end loop reaches the checker's write-after-read limit with no spare
margin. Source order alone does not establish safety.

`check_aievec_core.py` checks these dependencies with
`test/aiecc/aie4-mmac-schedule-checker.py`. Its timing model shares assumptions
with Peano, so hardware bit-exactness remains an independent check. Moving a
body commit three bundles earlier was used as a negative control and failed
the timing check.

## Validation

### Static checks

`make check_aievec KERNEL=aievec ...` examines the linked core ELFs under
`project/elfs_main_core_*`. For every core it checks:

- `.text` at most 16 KiB, within the 32 KiB program memory.
- Exact commit, MMAC, stream-read and staging-insert counts, adjusted for tile
  copies and software-pipelined loops.
- No scalar `lda.s8`, vector stack traffic or post-incremented staging indices.
- For i8, 64 fused `vst.srs` stores per tile body and one write each to `crRnd`
  and `crSat`.
- The expected number of K-loops, identified from zero-overhead-loop bounds.

With two tile bodies and two pipelined K-loops, the expected totals are 6
commits, 384 MMACs, 192 stream reads and 192 staging inserts. i8 adds 128 fused
stores. Each pipelined loop contributes one extra K-tile to the static counts.

Timing checks run on core (0,2), covering the straight-line block around each
K-loop: entry/prologue, two iterations, exit/epilogue and the peeled last
K-tile. A synthetic closing commit models the next boundary. This does not
simulate the entire nested core loop; stream reads are counted but not grouped
per tile. At `K=128`, there is no K-loop to extract and vector spills are
allowed, so timing is checked by hardware only.

### Test configurations

All configurations use `KERNEL=aievec COLS=3`.

| Config | M × N × K | OUT | CT | ASPLIT | ITERS | Coverage |
|---|---|---|---|---|---|---|
| A | 512 × 768 × 1024 | i8 | 4 | 2 | 1 | One group per strip; parity flips; guarded locks; dynamic `zero_if` |
| B | 512 × 576 × 128 | i8 | 1 | 2 | 2 | Inlined K-loop; unguarded locks; odd group count; repeated strips |
| C | 512 × 576 × 256 | i32 | 1 | 2 | 2 | i32 stores; odd group count; repeated strips |
| D | 256 × 1536 × 192 | i8 | 4 | 1 | 3 | Even group count; one A part; odd K-tile count; guarded locks |
| E | 256 × 576 × 128 | i8 | 1 | 2 | 1 | Odd group count and one strip per launch; parity persists across launches |

`aievec_codegen.lit` runs the static checks on A, B and C, with `TRACE_SIZE=8192`
for C. `run_npu3_aievec.lit` runs A–E on npu3. The host verifies the first and
last of 41 launches. llvm-lit handles the per-test timeout, including hangs
in later launches caused by parity or stream-count errors.

Two compiler regressions cover the supporting behavior:

- `test/Target/LLVMIR/aie4_flat_kloop_poison.py`: poison accumulator initialization
  and dynamic `zero_if` through aiecc and Peano.
- `test/aiecc/aie4-peano-llc-arg.test`: forwarding additional llc arguments.
