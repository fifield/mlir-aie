# gemv_4096_192_cascade

INT8 × INT8 → INT32 GEMV (M=1) for npu3, with configurable output width,
cascade rows, weight streams, and K-chunk size. The default geometry is
K=4096, N=192, across three columns and four compute rows.

## Design

Each column computes 64 output lanes. K is split evenly across the active
rows, whose partial sums are reduced through a north-to-south cascade.
The bottom row writes the column's output; a single-row configuration
writes its partial directly without using the cascade.

Each core receives two weight FIFOs, `AW_lo` and `W_hi`, through its
memtile. Activation data is embedded in `AW_lo` to stay within the core's
two input-DMA-channel limit. The shim uses two or four weight streams per
column, with the memtile distributing each stream to its row group.

The kernel stages a 64×64 INT8 matrix using 16 `v128int8` loads from each
weight pointer. It overlaps staging the next K-tile with the current
tile's `mac_conf` operation, accumulating into `v64acc32`.

## Build and run

```bash
make           # builds the default and batched ELFs, plus test.exe
make run       # correctness and per-call latency
make run_bench # correctness and batched throughput
make run_trace # event trace and timeline
make clean
```

The host checks the requested output lanes against a CPU reference and
reports latency, operation throughput, and weight-load bandwidth. Batched
execution amortizes XRT launch overhead; per-call measurements include
that overhead.

The `iters_per_call` argument to `test.exe` must match the iteration count
compiled into the ELF. It is a reporting divisor, not a runtime iteration
control. The Makefile supplies the matching value for each target. The
host also rejects implausible bandwidth, but this check does not establish
that the divisor is correct.

## Vector variant

`gemv_placed.py --vector` keeps the design (tiling, object FIFOs, cascade,
DMA, host layout) but emits each compute entry point as vector/arith IR in
the core body instead of calling `kernel.o`. aiecc compiles it with
`--enable-aie4-vectorization` through the AIE4 Vector -> AIEVec -> LLVM
pipeline and Peano; neither Chess nor `kernel.cc` is involved.

```bash
make vector           # vector ELFs + test.exe (no kernel.o)
make run_vector       # correctness and per-call latency
make run_vector_bench # correctness and batched throughput
make run_vector_full_range # correctness on full-range int8 inputs
```

`test.exe --full-range` (targets `run_vector_full_range` and
`run_full_range`) draws A and W from all of [-128, 127], pins the
extremes, and adds two columns whose products all share one sign; the
largest possible |C| is 128 * 128 * K = 2^26 at K = 4096, well inside
int32. The default inputs are small ([-4, 3] and [-2, 2]).

Each output lane is a dot product: in the weight layout below, one N
column's 64 K values in a K-tile are contiguous. The core multiplies
32-lane `vector<32xi8>` slices of A and W (sign-extended to i32),
accumulates across the chunk's K-tiles, and reduces each column once per
chunk. The pipeline lowers this to `vldb.unpack.2x` (exact i8 -> i16),
`vmac` (i16 x i16 -> i32, 32 lanes) and a `vadd.32` reduction tree; the
cascade finish uses `vmov mcd`/`vmov scd`. It uses the element-wise VMAC
rather than the 64x64 MMAC that `kernel.cc` drives, so it is compute-bound
and slower than the `kernel.o` build.

All trip counts are static multiples of the vector width: K per row is a
multiple of 64 by construction (the host packs whole 64-K tiles), each
tile's column splits into two 32-lane steps, and N per column is 64.
There is no remainder loop to mask.

## Configuration

Run `make clean` before rebuilding with different settings: make does not
track changes to these variables.

```bash
make all N=192 ROWS=4 STREAMS=2  # default geometry
make all N=128 STREAMS=4        # two columns, four streams per column
make all N=100                  # padded to two columns; checks 100 outputs
make all ROWS=1 K=256           # no cascade
make all K=2048                 # shorter contraction
make all K=6144 N=64            # longer contraction, one column
make all KCHUNK=4 AW_DEPTH=2 WHI_DEPTH=2 # double-buffered K chunks
```

| Variable | Meaning and constraints |
|---|---|
| `N` | Requested output width, 1–192; padded to a multiple of 64 on device. `COLS = ceil(N/64)` is derived, not independently configurable. The host checks only the requested lanes. |
| `ROWS` | Active rows per column, 1–4, starting at hardware row 2. `ROWS=1` uses the no-cascade kernel. |
| `STREAMS` | Shim weight streams per column: 2 or 4. Four streams divide each payload kind across two row groups and require `ROWS=2` or `ROWS=4`. |
| `K` | Contraction length; must be a multiple of `ROWS × 64`. Defaults to 4096. Buffer residency must fit core L1. |
| `KCHUNK` | K-tiles per FIFO item; must divide `K_TILES_PER_ROW = K/ROWS/64`. Zero selects the whole row. See K-chunking below for limits. |
| `BUF_DEPTH` | Shim-to-memtile FIFO depth, default 2; uses memtile SRAM. |
| `AW_DEPTH`, `WHI_DEPTH` | Memtile-to-core FIFO depths, default 1 each; use core L1. Depths of at least 2 allow adjacent chunks to overlap DMA and computation. |
| `ITERS` | Iterations compiled into the default ELF, default 1. The `run` target passes a reporting divisor of 1, so use the batched target for multi-iteration measurements. |
| `BENCH_ITERS` | Iterations compiled into the batched ELF, default 100. |
| `TRACE_SIZE` | Trace-buffer capacity in INT32 elements, default 32768. |

The Makefile passes matching geometry to the generator, kernel, and host.
Changing the number of shim streams does not change the host tensor
layout. Stream count and FIFO depths are tuning parameters, not
performance guarantees.

## Weight layout

The AIE4 64×64 INT8 staging matrix is loaded column-major. Each
`v128int8` packs two adjacent N columns: bytes 0–63 hold one column's
64 K elements, and bytes 64–127 hold the next column.

The host tensor concatenates one blob per compute column:

```text
column blob = AW_lo blob || W_hi blob
AW_lo blob  = row 0 chunks || row 1 chunks || ...
W_hi blob   = row 0 chunks || row 1 chunks || ...

AW_lo chunk = A_chunk || W_lo_chunk
W_hi chunk  = W_hi_chunk

A_chunk bytes    = K_CHUNK_TILES * 64
W_lo_chunk bytes = K_CHUNK_TILES * 64 * 32
W_hi_chunk bytes = K_CHUNK_TILES * 64 * 32
```

`W_lo` carries N lanes 0–31 and `W_hi` carries lanes 32–63.
Each column contains a copy of its rows' activation slices.
`pack_weight_byte()` and `pack_a_byte()` in `test.cpp` define the host
packing; `gemv_chunk_accumulate()` consumes that layout in `kernel.cc`.

## DMA batching and K-chunking

Each core has one FIFO per payload kind, with one chunk per FIFO item.
Using multiple chunks allows DMA for chunk i+1 to overlap computation on
chunk i when FIFO depth permits. The core accumulates all chunks into a
partial-sum buffer before performing its cascade operation.

The shim descriptor traverses
`[reps, NUM_CHUNKS, rows_per_stream, CHUNK_BYTES]`. A zero outer stride
reuses the same host data for repeated GEMVs. Hardware repetition avoids a
start/await barrier between individual iterations. `emit_iters()` splits
the requested count into batches of at most `MAX_BD_REPEAT = 64`, awaiting
completion at batch boundaries.

Core weight-buffer residency is
`AW_DEPTH * AW_LO_CHUNK_BYTES + WHI_DEPTH * W_HI_CHUNK_BYTES`, plus space
for partial sums, output, and stack. The generator applies an approximate
L1 budget check; the kernel build determines whether the complete layout
fits.

Constraints and limitations:

- One-tile chunks are rejected when a row contains multiple K-tiles:
  this configuration produces incorrect output on npu3.
- With `AW_DEPTH > 1`, `AW_LO_CHUNK_BYTES` must be a multiple of 128
  so vector loads remain aligned in every FIFO slot. Use an even chunk
  size with multiple AW_lo slots.
- Large chunks can exceed the BD encoding capacity when chunk and row
  dimensions are both active, producing a build-time
  `aie.dma_bd` "buffer descriptor length does not match" error.
  At the default geometry, `KCHUNK=8` encounters this limit;
  `KCHUNK=2` and `KCHUNK=4` are supported configurations.

## npu3 routing constraints

- Memtile S2MM is not a full crossbar. South ports reach these DMA subsets:
  South0→{0,2,3}, South1→{1,2,3}, South2→{4,6,7}, South3→{5,6,7}.
  Four inbound streams use {0,1,4,5}; {0,1,2,4} requires a Switch32
  loopback that deadlocks on hardware.
- Memtile MM2S channel 4 reaches only North 0. The eight northbound FIFOs
  use {0,1,2,3,5,6,7,8}, skipping channel 4.
- Completion tokens for shim MM2S channels 2 and 3 return on
  `Control32:1`. Four-stream designs route both control return ports to
  the shim controller, using packet ID c for column c.

Inspect `project*/cert.mlir` for unintended Switch32 data paths and
`cert_input.mlir` for `packet_rules(Control32 : 1)` when using four
streams.

## Event trace

`make run_trace` traces all active column-0 cores and produces
`trace_timeline.png`. Events include `INSTR_EVENT_0`, `INSTR_MATRIX`,
`MEMORY_STALL`, `STREAM_STALL`, and `LOCK_STALL`.
`INSTR_MATRIX` observes the MMAC unit; `INSTR_VECTOR` observes the
separate vector-ALU pipeline and is not selected here.

Trace packet IDs and broadcast channels must not collide with control
traffic. `trace_host_config` takes bytes and appends the trace-buffer
argument to the runtime sequence. The Makefile retains physical MLIR with
`--dump-intermediates`, and the host writes `trace.bin` as one hexadecimal
word per line for `parse.py`.

## Files

| File | Purpose |
|---|---|
| `kernel.cc` | Chunk accumulation and send/pass/sink/solo finish kernels. |
| `run_npu3_vector.lit` | Builds and runs the `--vector` variant on npu3, checking native instructions. |
| `gemv_placed.py` | Device geometry, weight distribution, cascade, runtime DMA, optional trace, and the `--vector` compute IR. |
| `test.cpp` | Host packing, CPU reference, correctness checks, benchmark, and trace capture. |
| `Makefile` | Default, batched, and trace builds and run targets. |
