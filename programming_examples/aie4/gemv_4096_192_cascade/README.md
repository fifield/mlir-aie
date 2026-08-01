# gemv_4096_192_cascade

High-performance GEMV (M=1) example with multi-row cascade reduction,
ported from `pacifica-apps/kernels/llama2/llama_4096_512` and scaled
to fit npu3 (3 columns × 4 compute rows = 12 cores).

## What this demonstrates

* **Spatial tiling**: M=1, K=4096, N=192 INT8×INT8 → INT32 GEMV
  partitioned across all 12 cores. Each row owns 1024 K elements
  (16 K-tiles of 64); each column owns 64 N elements (one tile_N).
* **Configurable weight streams per column (2 or 4)**: each compute
  column's shim drives `--streams` MM2S channels. With 2 (default) one
  carries the low half of every K-tile (N cols 0..31) and one the high
  half (N cols 32..63). With 4 each half is additionally split across two
  row groups. Each blob is fanned out across the column's rows by the
  memtile via `object_fifo_link` with row offsets. See **Configuration**
  and **npu3 routing notes** below — 4 streams works but is *not* faster
  for this problem size.
* **Configurable column count (1, 2 or 3)**: `--cols` sets N = cols × 64.
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

Benchmark over 200 launches x 100 iters/launch (5 warmup), MACs/call = 786432, bytes loaded/call = 798720:
  min:         9.46 us/iter   166.22 GOPS    84.41 GB/s
  median:      9.71 us/iter   162.03 GOPS    82.28 GB/s
  mean:        9.79 us/iter   160.71 GOPS    81.61 GB/s
  max:        11.86 us/iter   132.64 GOPS    67.36 GB/s
```

(captured 2026-07-30, after the repeat-BD runtime-sequence fix described in
**DMA streaming fix** below, default `COLS=3 STREAMS=2 K=4096` config.
Absolute numbers depend on system power mode.)

Per-call mode (`make run`) is dominated by XRT launch overhead — wall
time is ~210 µs even though the device-side work is only ~17 µs. Use it
for correctness checks; use `run_bench` for throughput numbers.

> **Benchmark gotcha:** the ELF runs a *fixed* number of GEMVs per launch
> (its build-time `-i` value). The second arg to `test.exe` is only the
> divisor used to report per-iteration time and **must equal that count**
> (1 for the default ELF, `BENCH_ITERS` for the bench ELF). Passing a
> larger number just divides the constant per-launch wall time by a wrong
> value and reports nonsense (e.g. "thousands of GB/s"). `make run_bench`
> always passes the matching value; `test.exe` now warns and exits non-zero
> if the implied bandwidth is physically impossible.

## Configuration

Geometry is selectable at build time (rebuild after changing — `make clean`
first, since make does not track these variables):

```bash
make all COLS=3 STREAMS=2          # default: K=4096, N=192, 2 streams/column
make all COLS=2 STREAMS=4          # N=128, 4 weight streams/column
make all COLS=1 STREAMS=2          # N=64
make all K=2048                    # shorter contraction (K=2048, N=192)
make all K=6144 COLS=1             # longer K, single column
```

* `COLS` ∈ {1, 2, 3} → N = COLS × 64. Threaded to `gemv_placed.py --cols`
  and to `test.exe` as `-DN_COLS` (stream count does not affect host layout).
* `STREAMS` ∈ {2, 4} → shim MM2S channels per column (`gemv_placed.py
  --streams`). All six combinations build, route, and pass on hardware.
* `K` (contraction length) → `gemv_placed.py --ktotal`, plus `-DK_TOTAL` to
  both `kernel.cc` (via `xchesscc`) and `test.exe`. Must be a multiple of
  `N_ROWS × TILE_K = 256`. Upper bound is L1: each core holds
  `K_per_row × (1 + TILE_N)` bytes of weights (`K_per_row = K/4`), so K is
  capped at **~7680** for the default 4-row layout; `gemv_placed.py` errors
  out above that. Default 4096. All three sources derive their geometry from
  the same `K`, so it stays consistent.
* `BUF_DEPTH`, `AW_DEPTH`, `WHI_DEPTH` (default 2/1/1) → object-FIFO depth
  for, respectively, the shim→memtile weight stream (memtile SRAM), the
  memtile→core `AW_lo` FIFO, and the memtile→core `W_hi` FIFO (both core L1,
  tightly budgeted — see the L1 cap above, which scales with these depths).
  Threaded to `gemv_placed.py --bufdepth`/`--awdepth`/`--whidepth`. Investigated
  as a fix for the bandwidth gap described below; empirically it made no
  measurable difference (see **DMA streaming fix**) — kept as build knobs for
  future experimentation, not because raising them currently helps.

**Performance note:** more streams is *not* faster for this problem size.
The 799 KB weight load is already served by 2 MM2S/column; 4 streams adds
DMA/routing overhead and runs ~16 % slower (3-col: 52 GB/s at 2 streams vs
45 GB/s at 4). 4-stream support exists to exercise the upper shim/memtile
DMA paths, not to speed up this GEMV.

## npu3 routing notes (why 4 streams is non-trivial)

Getting 4 streams/column working on npu3 required three hardware-specific
fixes (the first two in the MLIR-AIE compiler, the third in this design):

1. **Memtile S2MM is not a full crossbar.** A stream on South port *p* can
   only reach a private DMA-channel subset (South0→{0,2,3}, South1→{1,2,3},
   South2→{4,6,7}, South3→{5,6,7}). For 4 inbound streams use the private
   set **{0,1,4,5}**; using {0,1,2,4} forces a `South:3 → Switch32:1 → DMA`
   loopback that routes in the model but **deadlocks on hardware**.
2. **Memtile MM2S channel 4** reaches only North 0, so the 8 northbound
   FIFOs use **{0,1,2,3,5,6,7,8}** (skip 4; needs MM2S dest count = 9).
3. **TCT completion tokens for upper shim MM2S channels (2, 3) return on
   `Control32:1`**, not `Control32:0`. Designs using >2 streams must route
   *both* control return ports to the shim controller — column `c` uses
   packet id `c` on both `Control32:0` and `Control32:1` — or the host
   `WAIT_TCTS`/`dma_await_task` hangs forever even though the math is
   correct. This was the final root cause of the original 4-stream hang.

Sanity-check a routed build by grepping `project*/cert.mlir` for `Switch32`
(any `Switch32` on a data path = bad) and `cert_input.mlir` for
`packet_rules(Control32 : 1)` (must be present for 4-stream).

## DMA streaming fix: hardware BD repeat_count (2026-07-30)

GEMV is memory-bound, so its achievable bandwidth should approach the
shim-DMA ceiling measured by the sibling `shim_dma_memtile_bw` benchmark
(~110-115 GB/s steady-state on this hardware, ~73-75% of the ~153.6 GB/s
theoretical peak of this system's 4×LPDDR5 @ 9600 MT/s — see that example's
`reports/` for the full sweep). Before this fix, `run_bench` measured only
~56-58 GB/s in the same power mode — roughly half of that ceiling.

**Root cause**: the batched (`bench`) runtime sequence unrolled
`iters_per_call` (e.g. `BENCH_ITERS=100`) copies of the per-iteration DMA
setup, each ending in a `dma_start_task` / `dma_await_task` pair. That pair
is a hard barrier — it drains the whole pipeline and waits for completion
before the *next* iteration's transfers can even be issued. Raising
`BUF_DEPTH`/`AW_DEPTH`/`WHI_DEPTH` (the object-FIFO depths) does nothing
useful here: there's no barrier-crossing work to prefetch when every
iteration is fenced off from its neighbors. This was verified empirically —
`BUF_DEPTH` 2/3/4 and `AW_DEPTH`/`WHI_DEPTH` up to 2 (including at a reduced
`K=2048` to leave L1 headroom) showed no meaningful bandwidth change.

**Fix**: `gemv_placed.py`'s `emit_one_batch()` now takes a `reps` count and
folds it into the shim BD's outer dimension (`sizes=[reps,1,1,X]`,
`strides=[0,0,0,1]`) instead of always issuing exactly one transfer per
call. This uses the shim BD's hardware `repeat_count` field to stream `reps`
back-to-back transfers from a single descriptor, with no host/runtime-sequence
barrier between them — the same continuous-streaming pattern
`shim_dma_memtile_bw` uses. The field is 6 bits wide (max 64), so
`emit_iters()` chunks `iters_per_call` into `MAX_BD_REPEAT = 64`-sized
batches. This is purely a runtime-sequence change; the ITERS=1
(correctness-mode) MLIR is byte-identical to the pre-fix output, and
`run_bench`'s correctness check still passes.

**Result**: ~56-58 GB/s → ~80-85 GB/s (min-sample) in the same power mode —
per-channel bandwidth now meets or exceeds the pure-DMA reference from
`shim_dma_memtile_bw`.

**Remaining gap** (~80-85 GB/s vs. the ~110-115 GB/s ceiling, open, not yet
implemented): the per-core `AW_lo`/`W_hi` object FIFOs are each sized as one
whole per-iteration slab, so the consumer lock only releases after the
*entire* BD lands — buffer depth can't mask that whole-slab handoff latency,
only smaller per-chunk objects can. The proposed next step is restructuring
`gemv_reduce_K`'s acquire/release pattern to consume K in several smaller
(~8 KB) chunks instead of one large slab, so store-and-forward latency
overlaps with compute instead of serializing before it. There's also a
secondary, weaker effect where per-column bandwidth scales sub-linearly with
`COLS` (likely a shared host-DRAM/NoC path upstream of the per-column shims),
which the K-chunking change would not address. Neither has been attempted
yet.

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
