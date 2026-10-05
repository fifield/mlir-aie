<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# int8 GEMM on npu3 (AIE4)

This example computes `C = A × Wᵀ` with int8 inputs and int32 accumulation.
`A[M][K]`, `W[N][K]` and `C[M][N]` are row-major host buffers; DMA programs
handle tiling. Output is either int8, rounded with ties toward positive infinity
and saturated after a right shift, or raw int32.

## Build and run

Use an environment with MLIR-AIE tools and Python bindings, the AIE4 Chess
compiler (`AIETOOLS_DIR`), and XRT. Running requires an npu3 device.

```bash
make run                       # M=2048 N=3072 K=1024, int8 output
make run M=512 N=768 ITERS=10
make run OUT=i32                # raw int32 output
make run WBUF=1                 # one W strip buffer
make run ABCAST=0 ASPLIT=1       # fetch A separately for each column
make run C_REASM=0              # drain C through two joins per column
```

Each configuration uses its own `build_*` directory. Kernel objects are cached
in `kernel_K*` directories by their compilation parameters.

| Parameter | Default | Meaning |
|---|---|---|
| `M`, `N`, `K` | 2048, 3072, 1024 | Matrix dimensions |
| `COLS` | 3 | Active columns (1–3), with four cores each |
| `OUT` | `i8` | Output type: `i8` or `i32` |
| `SHIFT` | 12 | Right shift for int8 output, in [0, 59] |
| `ITERS` | 1 | GEMMs per launch |
| `ABCAST` | 1 | Broadcast A across columns; 0 fetches A per column |
| `ASPLIT` | 2 | Core DMA channels for A: 1 or 2 |
| `CT` | 4 for `i8`, 1 for `i32` | C tiles per core buffer |
| `WBUF` | 2 | Memtile W strip buffers: 1 or 2 |
| `C_REASM` | 1 | Reassemble C in the memtile; 0 uses two joins per column |
| `C_RUN` | 1024 | Target maximum C write-run length in bytes with reassembly |
| `TIMING` | 0 | Replace each tile's first four output bytes with a cycle timestamp |
| `TRACE_SIZE` | 0 | Trace buffer size in bytes; 0 disables tracing |

The host verifies C against a CPU reference after the first and final timed
launches. It poisons C before the final launch, outside the measured interval,
to detect missing writes. `TIMING=1` also reports tile periods; its timestamps overwrite C values
and cause verification mismatches. The host reports median and best time per GEMM, throughput, and the fraction
of an estimated compute peak. Set `AIE_CLK_GHZ` to override the clock used for
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
  `K` a multiple of 64 with `K >= 128`.
- Core L1 must fit `64 × K` bytes of A, two C buffers of
  `64 × CT × 64 × sizeof(C element)` bytes each, and the stack. Keep
  `K <= 1024`; larger C buffers can require a smaller K or CT.
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

`check_c_reassembly.lit` checks generated DMA scheduling with delayed producers
and verifies FIFO lowering, lock/BD allocation, buffer placement and routing
without hardware. `run_npu3_makefile.lit` builds and runs a GEMM with CPU
verification and requires the `ryzen_ai_npu3` and `chess` lit features.
