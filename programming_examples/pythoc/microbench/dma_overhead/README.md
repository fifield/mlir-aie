# Core-Programmed DMA Overhead Experiment

## Question

When a compute core programs its **own** tile DMA buffer-descriptor (BD) chains at
runtime — instead of relying on the BDs that `mlir-aie`/IRON bakes into the static
portion of the program — how much does that cost, and how does that cost compare to
the block/compute size of the kernel?

`dynamic_dma_add_one.py` already proved the core *can* program its DMA via `write_tm`
(processor-bus register writes) and `read_tm` (poll a lock value). This experiment turns
that capability into a controlled measurement.

## Run it

```bash
source env.sh        # repo root: venv + XRT + PYTHONPATH for aiecc
cd programming_examples/pythoc/microbench/dma_overhead

python dma_overhead_experiment.py --sweep          # full sweep -> results table
python dma_overhead_experiment.py --mode per-iter --block-words 256 --num-blocks 4  # single point

python slides/make_charts.py     # regenerate the deck figures (writes slides/img/*.png)
```

Modes: `static` (host arms the BD), `once` (core arms a repeat-BD once), `per-iter`
(core reprograms every block). Knobs: `--block-words B`, `--num-blocks M`,
`--chain-depth D`, `--compute-passes C`. Requires XRT + a built mlir-aie; runs on
NPU2 (Strix Halo) hardware. Raw sweep output is checked in as
`dma_overhead_sweep_results.txt`.

## What "overhead" means here

Per the brief, the kernel **pulls** data off the stream itself: at the start of each
iteration it programs a BD with a **finite** iteration/repeat count, starts the channel,
and drains a fixed amount of data — never an infinitely-looping BD chain. The tile DMA
must be **idle at the end** of kernel execution (verified, not assumed).

The cost of "doing it from the core" is the cycles the core spends executing the
`write_tm` register-programming sequence (BD words + start queue), which steals issue
slots from compute. Everything else — the actual stream transfer time and the compute —
is held identical across modes, so the difference *is* the overhead.

## Controlled design: three modes, one wait/compute loop

The only independent variable is **who programs the receive BD, and when**. The data
flow, the lock-wait protocol, the compute, and the instrumentation are byte-for-byte
identical across all three modes.

| Mode          | Who arms the S2MM BD | When                          | Per-block core cost |
|---------------|----------------------|-------------------------------|---------------------|
| `static`      | Host (runtime seq)   | once, before the kernel runs  | 0 (the "usual way") |
| `once`        | Core (`write_tm`)    | once, before its block loop   | amortized over M    |
| `per-iter`    | Core (`write_tm`)    | every block iteration         | full, every block   |

* **`static`** is the baseline "set it up in the static portion of the program." The
  host's `runtime_sequence` writes the exact same BD registers (via `npu_write32`) that
  the core would write — so the register *values* are identical; only the agent and
  timing differ. The core does zero BD programming and just consumes.
* **`once`** programs a single BD with `Iteration_Wrap = M-1`, `Iteration_Stepsize = B-1`
  and start-queue `Repeat_Count = M-1`. The channel walks the BD M times, advancing the
  base address by B words each repeat, releasing the lock once per block, then goes idle.
* **`per-iter`** reprograms a fresh single-shot BD (or a D-deep chain) every block. This
  is the configuration the brief describes and the one whose overhead we most want.

### Why receive-side only (single bulk send at the end)

The "pull data off the stream" cost is entirely on the **S2MM (receive)** side, so that
is what the three modes vary. The core writes its results into a local `out_buf` and a
**single** MM2S transfer ships `telemetry + results` back at the very end. That final
send is one core-issued `write_tm` event, identical in all three modes, and is timed
separately as an "epilogue" — it never enters the per-block comparison.

This also keeps the design clean: because the send is deferred and the receive buffer is
sized for all M blocks, **no core-side lock *release* is ever required** (the receive BD
never has to wait on the consumer). That sidesteps the one primitive PythoC does not
currently expose.

## Measurement: core `Timer_Low` instrumentation

The core reads its own free-running cycle counter, `Timer_Low` (core module `0x340f8`),
via `read_tm`, and brackets each phase:

```
t0 = read_tm(TIMER_LOW)
   ... program BD(s) ...            # per-iter only does work here
t1 = read_tm(TIMER_LOW)
   while lock_value < expected: ... # wait for the stream to deliver the block
t2 = read_tm(TIMER_LOW)
   ... compute block ...
t3 = read_tm(TIMER_LOW)
program_cycles += t1 - t0
wait_cycles    += t2 - t1
compute_cycles += t3 - t2
```

Accumulated per-phase totals (plus min/max per-block program cost, total cycles, and the
cycle at which the channel first read **idle**) are written to a small telemetry buffer
and shipped back with the results. `Timer_Low` is 32-bit; at ~1.8 GHz it wraps after
~2.4 s, far longer than any kernel here, so a plain subtraction is safe.

The `read_tm(TIMER_LOW)` calls themselves cost a few cycles; that fixed bias is measured
once (two back-to-back reads) and reported so it can be subtracted.

## Idle-at-end verification

After the block loop the core polls `DMA_S2MM_Status_0` bit 19 (`Channel_Running`:
`0` = data path idle **and** queue empty) until it reads idle, and records the timestamp.
The host additionally reads the status post-run. A mode "passes" only if the receive
channel is observed idle with an empty task queue — proving the finite/repeat BDs drained
and nothing is left looping.

## Sweep axes

All four are swept; each isolates a different facet of the overhead/size relationship.

1. **Block size `B`** (words/block: 64, 128, 256, 512, 1024, 2048) — the headline curve.
   Fixed per-block programming cost vs. growing compute/transfer. Shows the amortization
   knee: the block size above which self-programming is "free."
2. **Number of blocks `M`** — confirms per-iter total overhead is linear in block count
   and that `once`/`static` overhead is flat in M.
3. **BD-chain depth `D`** (BDs programmed per pull: 1, 2, 4) — programming cost vs. chain
   length; each extra BD is ~6 `write_tm` + linkage.
4. **Compute intensity `C`** (extra passes/element) — moves the kernel between
   DMA-bound and compute-bound regimes, showing overhead matters only when compute is
   cheap relative to the BD-programming sequence.

### Constraints discovered while implementing

* **Buffer size.** Every mode pulls all M blocks into one `in_buf` and produces one
  `out_buf` (both `M*B` words) so compute is identical; the compute-tile data memory
  bounds `M*B ≲ 4096` words (in+out+telem+stack within 64 KB). The sweep stays inside
  this; per-block metrics are mode-intrinsic and well-resolved.
* **BD count.** `once`/`static` lay out one BD per block, so `M ≤ 15` (the tile has 16
  BDs; BD15 is reserved for the epilogue MM2S send, which reuses it only *after* the
  S2MM chain has completed). `per-iter` reuses BDs `0..D-1` every block, so it is not
  BD-limited.
* **No vector integer multiply.** The compute-intensity load must avoid `<16 x i32>`
  `G_MUL`, which the aie2p GISel backend cannot legalize. The kernel keeps the
  verifiable output as a vector add (`out = in + 1`) and adds load via a *sequential
  scalar* recurrence (`work = work*work + v`) whose loop-carried dependence prevents
  vectorization and whose non-affine form prevents the optimizer from collapsing the
  `C` iterations. `C` is injected as a compile-time constant (a runtime-scalar splat
  into a vector also fails to legalize).
* **`Timer_Low` read bias** is measured (two back-to-back reads) and reported; it is
  ~1 cycle here, negligible.

## Results (NPU Strix Halo, aie2p, all modes PASS, channel idle in every run)

Raw data: `dma_overhead_sweep_results.txt`. `program_cycles` and `compute_cycles` are
cycle-stable across repeats (compute = 11348–11353 for the same config; program = exactly
4 / 72 / 164); `wait`/`total` are noisier because the spin-wait overlaps DMA/stream timing.

**Headline — pure per-block programming cost (core cycles stolen from compute):**

| mode      | program cost / block            |
|-----------|---------------------------------|
| `static`  | ~0 (≈1, empty-bracket noise)    |
| `once`    | ~15–22 (one-time, ÷ M)          |
| `per-iter`| **41, constant** (B, M, C indep)|

* **`per-iter` is a flat 41 cyc/block** = one BD (6 register writes) + start-queue write,
  ~5 cyc per `write_tm`. It does **not** grow with block size, block count, or compute.
* **Chain depth** adds ~16 cyc per extra BD: 41 (D=1) → 57 (D=2) → 89 (D=4), i.e.
  `≈ 41 + 16·(D−1)`.
* **`once`** pays the whole chain up front; per-block it is `≈ (28 + 14·M)/M` →
  22 (M=2), 18 (M=4), 16 (M=8), 15 (M=16).

**Overhead as a fraction of runtime (per-iter `program / total`):**

| sweep            | overhead |
|------------------|----------|
| B=64  (M=4,C=1)  | 2.4 %    |
| B=256 (M=4,C=1)  | 1.1 %    |
| B=1024 (M=4,C=1) | 0.3 %    |
| C=16  (M=4,B=256)| 0.35 %   |
| C=64  (M=4,B=256)| 0.10 %   |

**Conclusion.** Letting a core program its own receive DMA costs a *fixed* ~41 cycles
per block (per BD: ~16). Relative to a kernel's block/compute size that is **under ~1 %
once a block is ≳256 words or the compute is more than trivial**, and it vanishes as
either grows. The `once` (program-the-chain-once) pattern roughly halves even that, and a
host-armed `static` chain removes it entirely — but the absolute numbers show the
core-driven "pull your own data" model is essentially free for realistically-sized tiles,
while buying full runtime control over the DMA. The one regime to watch is **deep
per-iteration BD chains on tiny blocks** (e.g. D=4, B=256 showed elevated wait/total from
chained-BD reload latency), where reprogramming and chain-reload start to matter.

## Outputs / what we expect to learn

* A per-block **programming cost in cycles** for `per-iter` (roughly constant: ~`6*D + ~3`
  `write_tm` + the poll setup), and the amortized per-block cost for `once` (≈ that /M).
* The **break-even block size** `B*` where `program_cycles ≪ wait+compute`, i.e. where
  letting the core drive its own DMA is effectively free.
* Confirmation that `static` and `once` have identical wait/compute profiles to
  `per-iter`, isolating the programming delta.
* Confirmation the tile DMA is **idle at end** in every mode.

## Files

* `dma_overhead_experiment.py` — kernel (all three modes, parameterized), MLIR design,
  single-point runner, and `--sweep` driver with a results table.
* `dma_overhead_sweep_results.txt` — raw measured sweep output.
* `slides/dma_overhead.md` + `slides/make_charts.py` — Marp deck and figure generator.
* This README — methodology and register model.

Related: `../memtile_program_cost/` measures the cost of a core programming a
*different* tile's (memtile) DMA via control packets.
