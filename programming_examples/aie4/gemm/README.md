<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Integer and floating-point GEMM on npu3 (AIE4)

This example computes `C = A × Wᵀ` for several input types, selected with
`DTYPE`:

| `DTYPE` | A × W | Accumulator | `OUT` choices (default first) | MMAC tile |
|---|---|---|---|---|
| `i8` | int8 × int8 | int32 | `i8`, `i32` | 1×64 · 64×64 |
| `u8` | uint8 × uint8 | int32 | `u8`, `i32` | 1×64 · 64×64 |
| `i8u8` | int8 × uint8 | int32 | `i8`, `i32` | 1×64 · 64×64 |
| `u8i8` | uint8 × int8 | int32 | `i8`, `i32` | 1×64 · 64×64 |
| `i4` | int4 × int4 | int32 | `i8`, `i32` | 1×128 · 128×64 |
| `fp8` | fp8 E4M3 | fp32 | `fp8`, `bf16`, `f32` | 1×64 · 64×64 |
| `bf16` | bfloat16 | fp32 | `bf16`, `f32` | 1×32 · 32×64 |

`A[M][K]`, `W[N][K]` and `C[M][N]` are row-major host buffers; DMA programs
handle tiling. `K` counts elements; int4 packs two per byte, low nibble first.
Narrow integer output (`i8`, `u8`) is rounded with ties toward positive infinity
and saturated after a right shift. fp8 and bf16 output round to nearest even.
`i32` and `f32` are raw accumulators.

## Build and run

Use an environment with MLIR-AIE tools and Python bindings, the AIE4 Chess
compiler (`AIETOOLS_DIR`), and XRT. Running requires an npu3 device.

```bash
make run                       # M=2048 N=3072 K=1024, int8 output
make run M=512 N=768 ITERS=10
make run OUT=i32                # raw int32 output
make run DTYPE=fp8              # fp8 inputs, fp8 output (K=1024)
make run DTYPE=bf16             # bf16 inputs, bf16 output (K=512)
make run DTYPE=bf16 OUT=f32     # raw fp32 output
make run DTYPE=i4               # int4 inputs, int8 output (K=2048)
make run DTYPE=u8i8             # uint8 A, int8 W
make run WBUF=1                 # one W strip buffer
make run ABCAST=0 ASPLIT=1       # fetch A separately for each column
make run C_REASM=0              # drain C through two joins per column
make run KERNEL_CC=peano        # compile the kernel and cores with Peano (kernel_peano.cc)
```

Each configuration uses its own `build_*` directory. Kernel objects are cached
in `kernel_K*` directories by their compilation parameters.

| Parameter | Default | Meaning |
|---|---|---|
| `DTYPE` | `i8` | Input type: `i8`, `fp8` or `bf16` |
| `M`, `N`, `K` | 2048, 3072, 1024 (`K`=512 for `bf16`, 2048 for `i4`) | Matrix dimensions; `K` in elements |
| `COLS` | 3 | Active columns (1–3), with four cores each |
| `OUT` | per `DTYPE` | Output type; see the table above |
| `SHIFT` | 12 | Right shift for narrow integer output, in [0, 59]; unused for float types |
| `ITERS` | 1 | GEMMs per launch |
| `ABCAST` | 1 | Broadcast A across columns; 0 fetches A per column |
| `ASPLIT` | 2 | Core DMA channels for A: 1 or 2 |
| `CT` | 4 for 1-byte C, 2 for `bf16` output, 1 for 4-byte C | C tiles per core buffer |
| `WBUF` | 2 | Memtile W strip buffers: 1 or 2 |
| `C_REASM` | 1 | Reassemble C in the memtile; 0 uses two joins per column |
| `C_RUN` | 1024 | Target maximum C write-run length in bytes with reassembly |
| `TIMING` | 0 | Replace each tile's first four output bytes with a cycle timestamp |
| `TRACE_SIZE` | 0 | Trace buffer size in bytes; 0 disables tracing |
| `KERNEL_CC` | `chess` | Kernel compiler: `chess` or `peano` |

The host verifies C against a CPU reference after the first and final timed
launches. Integer results are bit-exact. Float results are checked against a double
reference with a tolerance of 2⁻¹⁶ of Σ|a·w| for accumulation order, plus half
an output ulp for `bf16` and `fp8` output. It poisons C before the final launch, outside the measured interval,
to detect missing writes. `TIMING=1` also reports tile periods; its timestamps overwrite C values
and cause verification mismatches. The host reports median and best time per GEMM, throughput, and the fraction
of an estimated compute peak (4096 MAC/cycle per core, 2048 for `bf16`, 8192 for `i4`). Set `AIE_CLK_GHZ` to override the clock used for
that estimate. Increasing `ITERS` amortizes launch overhead.

## Design

Each core computes 64×64 output tiles. Core `(c, r)` owns 64 rows of each
256-row M-block and column `c`'s N strip, containing `NT = N / (COLS × 64)`
tiles.

- **A:** Each core retains its 64×K slice in L1 for the M-block. With broadcast,
  contiguous row groups are distributed across shims and sent to the
  corresponding core row in every column. `ASPLIT=2` loads two 32-row halves
  through separate core DMA channels.
- **W:** Each column loads its W strip into the memtile once per GEMM,
  scattering it into staging order. The strip is replayed once per M-block
  and broadcast to all four cores. W streams directly into MMAC staging,
  bypassing core L1. The kernel reuses each W tile across 64 A rows and
  overlaps compute with staging the next tile. Prefetched W persists across
  kernel calls; the final call in a strip skips prefetch.
- **C:** Each core fills two L1 buffers of `CT` tiles. With reassembly,
  memtile DMAs scatter the four producers into separate row slices of a
  shared ping/pong block. Each producer has its own empty/full credits;
  the drain returns each credit after reading that slice. One shim S2MM
  channel per column writes the reassembled rows. Without reassembly, two
  memtile joins feed two shim S2MM channels per column.

Two W buffers allow the next GEMM's W load to overlap the current replay.
Reassembly widens C write runs up to the column-strip width, subject to
`C_RUN`. A remains single-buffered in L1 and reloads at M-block boundaries.

### Configuration constraints

- `M` must be a multiple of 256, `N` a multiple of `COLS × 64 × CT`, and
  `K` a multiple of 64 (32 for `bf16`, 128 for `i4`) with at least two K-tiles.
  A K-tile is always 64 bytes per A row: 64 int8 or fp8 elements, 32 bf16
  elements or 128 int4 elements.
- Core L1 must fit `64 × K × sizeof(A element)` bytes of A, two C buffers of
  `64 × CT × 64 × sizeof(C element)` bytes each, and the stack. Keep A at or
  below 1024 bytes per row (`K <= 1024`, `K <= 512` for `bf16`, `K <= 2048`
  for `i4`); larger C buffers can require a smaller K or CT.
- The DMA programs depend only on byte counts, so the generator, the memtile
  budget and the layouts below hold for all types with `K` replaced by bytes
  per row.
- `SHIFT=0` saturates the unshifted accumulator to int8.
- The generator checks that A, W and C buffers fit in the memtile. If they
  do not, reduce `WBUF`, `N` or `C_RUN`. Two W buffers require
  `2 × M / 256 <= 16` replay BDs; otherwise the generator uses one buffer.
- Object-FIFO memtile MM2S channels must fit below reserved W replay
  channel 6. A needs `ASPLIT` channels per sourced row group, and
  `C_REASM=0` needs two more for C. For example, `ABCAST=0` requires
  `ASPLIT=1`.
- A reassembled row contains a whole number of core-buffer rows and divides
  the column strip evenly. Its width is at most `C_RUN`, unless one
  core-buffer row is wider. With an odd number of C buffers per M-block,
  the generated core loop spans two M-blocks to preserve ping/pong parity.

All shim DMA completion tokens return to column 0's controller through both
Control32 channels; channel 1 serves the upper shim BD pool. W retiling uses
memtile BDs because shim tasks cannot express a strided outer iteration
dimension. C uses one task per M-block and column with reassembly, or two
without it.

## Peano kernel (`KERNEL_CC=peano`)

The Peano and AIEVec paths support `DTYPE=i8` only. Use
`KERNEL=extern KERNEL_CC=chess` (the defaults) for fp8 and bf16.

`kernel_peano.cc` implements the int8 path of `kernel.cc` with Peano builtins (same entry points,
macros and semantics; verified bit-exact for ASPLIT=2, SHIFT=12, K=1024, CT=4 (i8) and CT=1
(i32)). `install/peano` is a symlink to the venv llvm-aie, i.e. the same Peano build. `KERNEL_CC=peano` builds it with Peano `clang++` and lets Peano compile
and link the cores; the build directories carry a `_peano` suffix so chess and Peano artifacts
never mix. `TIMING=1` is a compile error there (Peano has no usable cycle counter); use
`TRACE_SIZE`. `kernel_bench/` takes the same `KERNEL_CC` switch, but its Peano build reports 0
for the cycle columns.

### AIEVec kernel (`KERNEL=aievec`)

`KERNEL=aievec` has no `kernel.o`. The generator (`--kernel aievec --shift $(SHIFT)`) emits each
core body in flat AIEVec (`aievec.aie4.stage_stream`, `commit`, `mmac_active`, `srs`), and
`aiecc --peano` compiles it. Data movement, locks, the runtime sequence and `test.cpp` are those of
the extern build. Build directories carry an `_aievec` suffix.

- **Structure.** Per strip: 32 W pops (prime), a runtime output-tile loop, and a peeled
  strip-end tile without prefetch. Each tile is a runtime K-loop over K-tiles 0..KT−2, then the
  peeled last K-tile and 64 stores. A K-loop iteration is a commit, 64 row MMACs with a dynamic
  `zero_if(kt == 0)`, and a W pop after every odd row; the accumulators start as `ub.poison`.
  The K-loop also carries one `ptr`-dialect pointer per A part, advanced one K-tile per
  iteration, so Peano walks the A rows with post-increment loads. OUT=i8 rounds to +inf and
  saturates through explicit-mode SRS, so the core writes `crRnd`/`crSat` once at entry. The
  tile body exists twice per core whatever NG is (once when NG·CT = 1). At K = 128 the core stack
  is raised to 4 KiB, because the accumulator spills there use the whole default 2 KiB.
- **Peano flags.** `AIEVEC_LLC_ARGS` (default `-aie-force-postpipeliner
  -aie-postpipeliner-maxii=80`) reaches llc through `aiecc --peano-llc-arg`. With it, Peano's
  post-pipeliner software-pipelines each K-loop at II = 64-65 bundles for 64 MMACs, issuing each
  commit 4-5 bundles before the previous K-tile's last MMACs (early commit). The strip-end K-loop
  sits exactly at the zb WAR limit. Each commit also issues before 1-2 of its own K-tile's
  inserts. Both rely on exact pipeline stage numbers, which the checker shares with Peano, so
  hardware bit-exactness is the independent check. The post-pipeliner's
  default maximum II of 64 is one bundle short of the 65 it needs, hence the flag. Pipelining adds a
  prologue/epilogue K-tile per K-loop, so `.text` is 2.6–6.4 KB of the 32 KB program memory.
  Non-default `AIEVEC_LLC_ARGS` build into their own directories.
- **Restrictions.** `C_REASM=1` and K ≥ 128 are required; `TIMING=1` and `KERNEL_CC` are errors.
- **Static gate.** `make check_aievec KERNEL=aievec ...` runs `check_aievec_core.py`:
  - `.text` ≤ 16 KB;
  - exact counts per core: 4 commits, 256 MMACs, 128 pops and 128 staging inserts (half of each
    when NG·CT = 1), plus one K-tile (1 commit, 64 MMACs, 32 pops and inserts) per
    software-pipelined K-loop; for OUT=i8 also 128 `vst.srs` and one `crRnd` and one `crSat`
    write;
  - no `lda.s8`, no vector stack traffic, no post-incremented staging index, and the expected
    number of K-loops;
  - `test/aiecc/aie4-mmac-schedule-checker.py` on the straight-line block around each K-loop
    of core (0,2). The block starts after the tile loop's `jnz`, so the prime and the previous
    tile's prefetch inserts are outside it. It covers the entry or pipeline prologue, two loop
    iterations, the exit and the peeled last K-tile, so the early commit is checked against the
    zb/za timing rules. A conservative synthetic closing commit models the tile back edge and
    strip end. Pops are counted but not grouped per tile.

  At K = 128 the one-iteration K-loop is inlined and Peano spills accumulators (the P0 C kernel
  spills there too). The Makefile then expects no K-loop and allows vector stack traffic, so the
  K = 128 timing is checked by hardware only.
- **Tests.** `aievec_codegen.lit` checks code size, instruction counts and scheduling.
  `run_npu3_aievec.lit` describes and runs five npu3 configurations covering int8 and int32
  output, odd and even C-buffer group counts, and parity across repeated launches, each under
  `timeout 300`. Additional regressions are
  `test/Target/LLVMIR/aie4_flat_kloop_poison.py` (poison init and dynamic `zero_if` at 64 rows)
  and `test/aiecc/aie4-peano-llc-arg.test`.

## Event trace

```bash
make run_trace TRACE_SIZE=4194304 ITERS=2
```

Tracing uses a separate build and a shared start broadcast for column 0's
cores, memtile and shim. It records tile boundaries, compute and stall events,
and DMA activity. The memtile trace monitors W replay and, with reassembly,
C drain; without reassembly it monitors W fill and A output instead.

The build lowers object FIFOs before inserting trace flows so C FIFOs do not
count as trace channels. Memtile monitoring selects DMA channels directly;
stream-switch `trace_port` monitoring is unavailable on npu3.

`run_trace` writes these files in the configuration's build directory:

- `trace.txt`: raw trace words.
- `trace.json`: parsed events for Perfetto.
- `trace_timeline.png`: an event timeline.

It also runs the trace summary and `analyze_trace.py`, which report event
occupancy, per-core tile periods and compute gaps with their active stall
events.

## Benchmarks and tests

`sweep.sh` builds and runs GEMM configurations sequentially and appends timing
and verification results to `sweep_results.md`. Override the output path with
`OUT_FILE` or pass configurations as quoted arguments:

```bash
./sweep.sh "512 768 1024" "2048 3072 1024 OUT=i32"
```

`kernel_bench/` measures one core's MMAC and accumulator-store cycles with A in
L1 and W streamed from the memtile. `KT` sets K/64 and `CALLS` sets the number
of kernel calls; both select the build directory.

```bash
make -C kernel_bench run KT=8 CALLS=4
```

`ddr_write_bench/` measures DDR writes with optional concurrent reads.
`L` sets write-run bytes, `COLS` the column count, `CH` the shim write channels
per column, and `RD` the ratio of read bytes to written bytes. Its sweep uses
two `REPS` values to estimate bandwidth from the change in launch time,
reducing the contribution of fixed launch overhead.

```bash
make -C ddr_write_bench run L=1024 COLS=3 CH=1 RD=1
(cd ddr_write_bench && ./sweep.sh "256 3 1" "1024 3 1")
```

`run_npu3_makefile.lit` builds and runs a GEMM with CPU verification and
requires the `ryzen_ai_npu3` and `chess` lit features.
