# gemv_4096_192_cascade

High-performance GEMV (M=1) example with multi-row cascade reduction,
ported from `pacifica-apps/kernels/llama2/llama_4096_512` and scaled
to fit npu3 (3 columns × 4 compute rows = 12 cores).

## What this demonstrates

* **Spatial tiling**: M=1, K=4096, N=192 INT8×INT8 → INT32 GEMV
  partitioned across all 12 cores. Each row owns 1024 K elements
  (16 K-tiles of 64); each column owns 64 N elements (one tile_N).
* **Two parallel weight streams per column**: each compute column's
  shim drives **two** MM2S channels — one for the low half of every
  K-tile (N cols 0..31) and one for the high half (N cols 32..63),
  mirroring the source design's dual weight streams. Each blob is
  fanned out across the column's 4 rows by the memtile via
  `object_fifo_link` with row offsets.
* **Activation slice inlined into weights**: the compute tile has only
  2 DMA input channels, so the per-row 1 KB activation slice is
  prepended to each per-core `AW_lo` buffer instead of riding a third
  FIFO. The kernel reads `A` from offset 0 and `W_lo` from offset 1024.
* **North → south cascade reduction**: hardware row 5 produces a partial
  and forwards it via the cascade bus; rows 4 and 3 add their local
  partial and forward; row 2 adds its partial and writes the column's
  final 64 INT32 lanes back to the shim.
* **Batched runtime sequence** (optional): the `bench` ELF rolls
  `BENCH_ITERS=100` GEMV iterations into a single kernel call so the
  per-iteration wall time excludes the per-launch XRT/host overhead
  and exposes steady-state device throughput.

The kernel inner loop preserves the source design's staging-matrix
pipeline: per K-tile it issues 32 `staging_m64x64int8.upd` calls (16
v128int8 each from the low and high pointers) interleaved with one
`mac_conf` MMAC, accumulating into a single `v64acc32`.

## What's preserved vs. what's stripped from the source

| Preserved | Stripped |
|---|---|
| 64×64 INT8 staging-matrix MMAC inner loop (`mac_conf`, `staging_to_matrix_m64x64int8`) | SiLU activation and lookup tables |
| `gemv_reduce_K()` pipelined K-tile loop with overlapped weight staging | Element-wise multiply (SiLU(U) × G) |
| `cascade_north_south()` reduction across 4 rows | Quantize INT32 → INT8 |
| Multi-row spatial layout, dual weight streams | DP projection + west→east cascade |
| Two weight pointers per core (mirrors dual streams) | Multi-projection (U/G/DP) chaining |

The source design ran on a 9-column NPU (8 compute cols + 1 dedicated
activation source) with up to 8 MM2S per shim. On npu3 the shim DMA
budget is **4 MM2S / 2 S2MM per shim** and the compute tile has only
**2 input DMA channels**, so this port:

* collapses the activation source onto col 0 (no dedicated col 8 shim);
* inlines the per-row A slice into each core's `AW_lo` buffer so the
  compute tile only consumes 2 FIFOs (AW_lo + W_hi);
* fans weights out via the memtile (the source went shim → core direct).

The source pulled weights from streams via `get_ss_v128int8(0/1)`. This
example feeds weights through L1 buffers (object FIFOs) so the design
fits the standard aie4 `object_fifo` build flow; the high-perf
staging-register pipeline pattern itself is unchanged.

## Weight layout

Per the AIE4 ISA spec the 64×64 INT8 staging matrix
`staging_m64x64int8` is loaded **column-major**: each `v128int8` load
pair-packs two adjacent N columns of 64 K-rows each. The host therefore
stores W in this device layout (per column blob):

```
col_blob = AW_lo blob (135168 B) || W_hi blob (131072 B)

AW_lo blob = N_ROWS * [A_slice 1024 B | W_lo_half 32768 B]   per row
W_hi  blob = N_ROWS * [             W_hi_half 32768 B]       per row

W_lo_half / W_hi_half per row = K_TILES_PER_ROW * 16 v128 slots
  v128 slot = 128 bytes pair-packing two N cols: bytes 0..63 = col_a,
                                                  bytes 64..127 = col_b
```

`pack_weight_byte()` and `pack_a_byte()` in `test.cpp` are the
canonical reference packers.

## Files

| File | Purpose |
|---|---|
| `kernel.cc` | Three core entry points (`gemv_core_send/pass/sink`) sharing `gemv_reduce_K` + cascade helpers. |
| `gemv_placed.py` | Placed IRON design: 3 cols × 4 rows, two W FIFOs per (col, row) via memtile fan-out, A inlined in AW_lo, cascade flows N→S. Optional `--trace` flag emits a trace.config + packetflow. |
| `test.cpp` | XRT host harness: arbitrary INT8 inputs, host CPU reference compare, latency + throughput benchmark. Detects trace ELF by name and allocates an extra trace BO. |
| `Makefile` | Three build targets: `gemv_4096_192_cascade.elf` (1 iter), `..._bench.elf` (BENCH_ITERS iters), `..._trace.elf` (1 iter + trace). |

## Build & run

```bash
make           # builds the default + bench ELFs and test.exe
make run       # correctness + per-call latency (single iter per launch)
make run_bench # correctness + steady-state throughput (BENCH_ITERS=100 per launch)
make clean
```

`make run_bench` prints something like:

```
PASS (all 192 outputs match host reference)

Benchmark over 200 launches x 100 iters/launch (5 warmup):
  min:        17.65 us/iter    89.11 GOPS    45.25 GB/s
  median:     17.75 us/iter    88.60 GOPS    44.99 GB/s
  mean:       17.80 us/iter    88.39 GOPS    44.88 GB/s
  max:        18.70 us/iter    84.11 GOPS    42.71 GB/s
```

Per-call mode (`make run`) is dominated by XRT launch overhead — wall
time is ~210 µs even though the device-side work is only ~17 µs. Use it
for correctness checks; use `run_bench` for throughput numbers.

## Optional: event trace (experimental, currently hangs on this hw)

```bash
make run_trace   # builds gemv_4096_192_cascade_trace.elf and runs it
```

The trace pipeline (configure `aie.trace.config` for the cascade-sink
core + `aie.trace.host_config` to drain into a host trace buffer,
followed by `python/utils/trace/parse.py` + `visualize_trace.py` to
produce `trace_timeline.png`) is wired up in `gemv_placed.py` and
`Makefile`, but as of writing the trace ELF deadlocks on `wait2()` on
npu3. Bisecting shows the deadlock is not caused by the shim-DMA
completion token (verified by patching the BD task-queue value to drop
bit 31 — the hang is unchanged), but the actual root cause hasn't been
pinpointed yet (the `event_trace` sibling example traces successfully
on the same system with very similar configuration). Treat this target
as a TODO; the correctness + bench paths above are unaffected.
