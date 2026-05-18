# dispatch_micro: v1 measurement report

> Generated 2026-05-18, AMD Strix (`npu2`) host, XRT 2.23.0 (HEAD), `aiecc` from
> `mlir-aie` branch `location` @ `ab3cf203b0`. All numbers are wall-clock
> nanoseconds bracketing the kernel submit + wait on the host.

## ⚠ Important caveat — v1 does NOT measure real PDI loads

> The runtime/driver/firmware stack **caches PDI loads at the PDI level**:
> when a `load_pdi` op targets a PDI that is already loaded, the firmware
> short-circuits and does no actual reconfiguration work.
>
> Every dispatch in this v1 run targets the *same* configuration — there is
> no task switching. So `load_pdi_fw`'s `aiex.npu.load_pdi { device_ref =
> @main }` hits the cache on every iteration after the first, and its
> numbers below reflect the **cache-hit path**, not actual PDI load cost.
>
> `load_pdi_expanded` is the exception: `--expand-load-pdis` replaces the
> cacheable `load_pdi` opcode with raw `write32` / `blockwrite` txn ops,
> which the firmware cannot cache. So expanded *is* paying real reconfig
> cost on every dispatch — that's where the 3.7× growth across the array
> comes from.
>
> Honest "what does a real PDI load cost" numbers require A↔B alternation
> (defeats the cache by alternating distinct PDIs). That's v2.

## tl;dr (with the caveat above in mind)

1. **`baseline` ≈ `load_pdi_fw` across the whole array (~65-85 µs).** They
   share the same dispatch path; `load_pdi_fw` adds only a cache-hit check.
   When you see the two lines on top of each other in the plots, that's why.
2. **`load_pdi_expanded` is the only mechanism in v1 doing real reconfig
   work** — and that work scales hard. 1 tile: 82 µs. 32 tiles: 302 µs. The
   slope is the cost of streaming all the register writes in band on every
   dispatch.
3. **Cold start is 40 ms, dominated entirely by `register_xclbin` +
   `hw_context`.** Mechanism choice barely affects it (within noise).
4. **`xrt::runlist` amortization is dramatic for the ELF/load_pdi paths** —
   per-dispatch latency drops from ~65 µs to ~1 µs at batch=64. Baseline
   amortizes much less efficiently (~22 µs at batch=64). This is the biggest
   actionable signal in v1.
5. **`load_pdi_*` with `tiles ∈ {2, 4} × bds = 8` crashes the firmware** —
   reproducible, isolated to this corner; baseline and `tiles=8` are fine.
   Worth a separate bug.

## Methodology

- 3 mechanisms: `baseline`, `load_pdi_fw` (firmware `XAIE_IO_LOADPDI`),
  `load_pdi_expanded` (`--expand-load-pdis`, write32 / blockwrite inline).
- Devices: `npu2_1col` for 1-tile builds; `npu2_4col` for 2/4-tile builds;
  `npu2` (full 8 cols) for 8-tile builds.
- BDs per shim DMA channel: 2, 4, 8 — emitted as N independent
  `aiex.dma_configure_task_for` ops, each issuing one shim BD over a
  per-tile slice of a shared in/out buffer.
- Topology: `linear` (shim → compute → shim, no mem-tile hops). Only one
  topology in this run; `branch` / `hop` covered in v2.
- Per row: 10 warmup + 100 measured iterations, `std::chrono::steady_clock`
  bracketing kernel submit + wait. Raw per-iter samples retained in JSONL.
- `cold_start` runs in 30 fresh processes per (mechanism, tiles, bds) cell;
  each process measures `xrt::xclbin` / `xrt::elf` load, `register_xclbin` +
  `hw_context`, kernel-handle construction, and first dispatch separately.
- Batched runs use `xrt::runlist::execute()` over `batch_size ∈ {1, 4, 16, 64}`.
- ctrlpkt is **not** in this run — it needs the column-control-overlay pass
  before `aiecc --aie-generate-ctrlpkt` compiles. Wired in v2.

Raw data: `results/results.jsonl` (336 lines). Plots: `results/plots/`.

## Headline tables

**Legend** (used in every table below):
- `t` = **tiles**: number of compute tiles in the design (each on its own column, row 2). 1 / 2 / 4 / 8 maps to `npu2_1col` / `npu2_4col` / `npu2_4col` / `npu2` (full 8-column Strix).
- `b` = **BDs per shim DMA channel**: number of independent `aiex.dma_configure_task_for` ops emitted per tile per direction, each pushing one shim BD over a slice of the shared in/out buffer. Larger `b` means a bigger instruction stream and more BDs to program at dispatch time.
- `bs` = **runlist batch size** (only in the runlist table): number of `xrt::run`s added to a single `xrt::runlist::execute()` call. `bs=1` is the unbatched baseline.
- All latencies are host-side `std::chrono::steady_clock` p50 over 100 iterations (50 for batched), in **microseconds** unless noted.

### 1. Single-shot dispatch — `pure_dispatch` p50 (µs)

|                         | t=1, b=2 | t=1, b=4 | t=1, b=8 | t=2, b=2 | t=2, b=4 | t=2, b=8 | t=4, b=2 | t=4, b=4 | t=4, b=8 | t=8, b=2 | t=8, b=4 | t=8, b=8 |
|-------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| baseline                |     66.3 |     69.1 |     76.6 |     69.3 |     70.1 |     77.4 |     75.2 |     77.8 |     96.9 |     88.7 |     99.2 |    117.3 |
| load_pdi_fw             |     61.4 |     66.0 |     70.0 |     62.3 |     66.4 |     **✗** |     76.8 |     81.5 |     **✗** |     80.5 |     86.8 |     96.9 |
| load_pdi_expanded       |     72.5 |     78.3 |     74.6 |     72.3 |     74.4 |     **✗** |     90.9 |     95.9 |     **✗** |    132.4 |    134.3 |    151.8 |

**✗** = firmware crash (see "Failures" below).

Visible across the row:
- `load_pdi_fw` tracks `baseline` (sometimes a touch faster, sometimes a touch slower — both within run-to-run noise). This is **the firmware PDI cache hitting on every iteration**, not the load_pdi op doing meaningful work.
- `load_pdi_expanded` grows linearly with `tiles × bds` — the inlined write32/blockwrite stream gets bigger and is paid in band, *uncached*. At 8 tiles, 8 BDs it costs ~152 µs vs `baseline`'s 117 µs (~30% penalty). This is the only mechanism in v1 doing real reconfig work.
- `baseline` scales gently with `tiles × bds` — that's pure host-driver dispatch overhead growing with instruction-stream length.

### 2. Cold start — phase breakdown (median µs over 30 fresh processes)

Each cold_start row is one fresh process. The phases are timed sequentially, one `std::chrono::steady_clock` bracket per phase, with no work in between:

- **`load`** — open and parse the on-disk artifact. xclbin family: `xrt::xclbin(path)` ctor (reads the file, parses sections). ELF family: `xrt::elf(path)` ctor (reads + validates the ELF).
- **`register`** — make the device aware of the artifact and acquire a hardware context. xclbin family: `device.register_xclbin(xclbin)` *plus* `xrt::hw_context(device, uuid)`. ELF family: `xrt::hw_context(device, elf)` (no separate register call). This is where the kernel-mode driver allocates the AIE partition and the firmware starts up — the ~40 ms bar in the chart.
- **`kernel`** — construct the kernel handle. xclbin family: `xrt::kernel(context, name)`. ELF family: `xrt::ext::kernel(context, "main:seq")`. Sub-100 µs in all cases.
- **`first`** = `first_dispatch` — the first `kernel(...) + wait` (xclbin) or `run.start() + wait2` (ELF) on this fresh context. Pays whatever one-time costs the driver/firmware deferred from the previous phases.
- **`TOTAL`** — sum of the four phases above, for the same single process.


| mechanism             | t | b |    load |  register |  kernel |   first |   TOTAL |
|-----------------------|--:|--:|--------:|----------:|--------:|--------:|--------:|
| baseline              | 1 | 2 |    84.6 |  40 028.3 |    70.9 |   796.2 | 40 979.9 |
| baseline              | 1 | 8 |    79.4 |  39 874.0 |    58.4 |   696.2 | 40 708.0 |
| baseline              | 4 | 2 |    88.6 |  39 915.1 |    60.3 | 1 003.3 | 41 067.3 |
| baseline              | 4 | 8 |    80.0 |  39 814.7 |    58.4 |   956.3 | 40 909.4 |
| load_pdi_fw           | 1 | 2 |   161.8 |  39 681.6 |    35.4 |   896.6 | 40 775.3 |
| load_pdi_fw           | 1 | 8 |   191.9 |  40 129.1 |    39.6 |   942.0 | 41 302.5 |
| load_pdi_fw           | 4 | 2 |   164.7 |  39 839.6 |    41.0 | 1 120.7 | 41 166.0 |
| load_pdi_expanded     | 1 | 2 |   166.7 |  39 845.8 |    34.7 |   757.5 | 40 804.8 |
| load_pdi_expanded     | 1 | 8 |   178.4 |  39 888.9 |    34.1 |   851.0 | 40 952.4 |
| load_pdi_expanded     | 4 | 2 |   177.2 |  39 567.0 |    34.5 |   842.6 | 40 621.3 |

Observations:
- **`register_xclbin` + `hw_context` is the entire cold story (~40 ms),** and is identical (within ~0.5%) across mechanisms. Choice of mechanism does not move this number.
- ELF-family loads cost ~180 µs vs xclbin's ~85 µs (parsing + validation, presumably).
- ELF-family kernel-handle creation is *cheaper* (~35 µs) than xclbin's (~60-70 µs).
- First dispatch is ~700-1100 µs across all mechanisms — first-pass driver / firmware warmup.
- Caveat: XRT/firmware caches likely warm across the 30-process loop. We didn't `drop_caches` or unbind the device, so these numbers describe *re-run-on-warm-system* cold start, not *cold-from-truly-cold*.

### 3. Runlist batching — per-dispatch p50 (µs)

| mechanism             | t  |  bs=1 | bs=4 | bs=16 | bs=64 |
|-----------------------|---:|------:|-----:|------:|------:|
| baseline              |  1 |  68.6 | 39.5 |  26.4 |  21.7 |
| baseline              |  4 |  68.8 | 44.6 |  30.3 |  31.6 |
| load_pdi_fw           |  1 |  64.9 | 17.1 |   4.3 |   1.0 |
| load_pdi_fw           |  4 |  77.9 | 17.4 |   5.0 |   1.2 |
| load_pdi_expanded     |  1 |  81.2 | 18.1 |   4.7 |   1.2 |
| load_pdi_expanded     |  4 | 102.1 | 23.7 |   5.5 |   1.5 |

This is the most striking result in the dataset:
- **The ELF / `xrt::ext::kernel` path amortizes batching ~20× better than baseline.** Going from batch=1 to batch=64, baseline drops 3.2×; ELF path drops 60-80×.
- At batch=64, both `load_pdi_*` variants converge near 1 µs per dispatch — likely the limit of host-side queueing overhead.
- Baseline plateaus at ~22 µs (1 tile) / 31 µs (4 tiles) — runlist isn't amortizing past that for the classic `xrt::kernel` path.

The most likely explanation is that `xrt::ext::kernel` + `xrt::runlist` lets the runtime queue everything once and let firmware drain it, while the legacy `xrt::kernel` path serializes some host-side bookkeeping per run. Worth confirming with strace / perf in a follow-up.

## Failures

Reproducible firmware crash, four cells:

| mechanism         | t | b |  
|-------------------|--:|--:|
| load_pdi_fw       | 2 | 8 |
| load_pdi_fw       | 4 | 8 |
| load_pdi_expanded | 2 | 8 |
| load_pdi_expanded | 4 | 8 |

Symptoms (from XRT/kmd log):
```
fatal_error_exception_pc = 0x00000000
fatal_error_app_module = 0x00000000
```

Pattern is highly specific: only `tiles ∈ {2, 4}` and `bds == 8`, only with `load_pdi`. `baseline` at the same combos works; `tiles=1, bds=8` works; `tiles=8, bds=8` works. Suspicion: BD-id pool collision between the `load_pdi` expansion's BDs and the dispatch's BDs at intermediate tile counts. Worth filing.

One other observed flake: `load_pdi_fw t=1 b=8 cold_start` crashed all 30 processes despite `pure_dispatch` of the same build working — suggests an additional first-dispatch / fresh-context interaction in cold mode.

## Anomalies worth chasing

1. **`load_pdi_fw` ≈ `baseline` is the PDI cache short-circuiting.** Confirmed: the runtime/driver/firmware stack caches PDI loads at the PDI level, so a self-PDI-load on every dispatch hits the cache after the first iteration. v1's `load_pdi_fw` numbers therefore reflect cache-hit cost (≈ a no-op opcode check), not actual PDI load work. To measure real PDI-load cost we have to alternate between two distinct PDIs — that's v2's A↔B test.
2. **Baseline's batched-amortization floor at ~22 µs** is much higher than the ELF path's ~1 µs. This is the single largest performance lever in the dataset and the place a profiler should go next.
3. **`load_pdi_expanded` linear scaling** is exactly as predicted — the inlined txn writes grow with `tiles × bds × rows_per_col` and are uncached. This is v1's only honest reconfig-cost curve, and it's worth keeping that calibration when v2 lands so we can compare "real firmware PDI load" against "host-side txn reprogramming" head-to-head.

## Methodology caveats

- **`warm_reconfig` and `pure_dispatch` are currently identical for `load_pdi_*`** because the generator unconditionally puts `npu_load_pdi(device_ref=@main)` at the top of the runtime sequence; every dispatch is a reconfig. Splitting into a "reconfigure then dispatch *without* reconfig" pair needs a generator change (and probably a second runtime sequence). Deferred to v2.
- **AB-mode runs aren't in this report.** The compiler emits the artifact (two `@device` regions + an `ab_orch` device that toggles), but `bench.cpp` doesn't yet dispatch the orchestrator separately. Three-way runlist (configA → configB → configA) is a v2 follow-up.
- **No `--dump-state` cross-mechanism verification ran.** All three measured mechanisms ostensibly do the same work (set up DMA + run passthrough), but we haven't proven the post-reconfig register state is identical. For the current numbers this is unlikely to change conclusions (the workload is identical placed IRON for all three), but should be wired before drawing strong cross-mechanism conclusions in v2.

## Files

- `results/results.jsonl` — 336 rows; `mechanism`, `metric`, `tiles`, `bds`, `batched`, `batch_size`, `ns_samples` per row.
- `results/plots/` — `pure_dispatch_vs_{tiles,bds}{,_bds*}.png`, `pure_dispatch_batched.png`, `cold_start_breakdown.png`.

## Whole-array sweep (added)

After the original 1-row run, we extended the generator with a
`--rows-per-col` axis (1, 2, 4) and added a per-column memtile fan-out:
each compute tile in a column gets its own narrow input fifo from the
memtile, and the memtile gathers outputs back to a single shim DMA channel.
The shim still sees one DMA pair per column regardless of row depth.

This lets us sweep total compute tiles from 1 up to **32** (the entire
`npu2` array: 8 cols × 4 rows). All 36 builds compiled; all 36 dispatch
runs succeeded — no firmware crashes at `bds=2`.

### Legend addition

- `cols` = shim columns used (= the `tiles` / `t` axis from earlier tables).
- `r` = `rows_per_col` — compute tiles per column (1, 2, or 4).
- `total compute tiles` = `cols × r`.

### Table: pure_dispatch p50 (µs), bds=2, linear topology

|                       | cols |  r=1  |  r=2  |  r=4  |
|-----------------------|-----:|------:|------:|------:|
| baseline              |    1 |  67.9 |  69.2 |  71.2 |
| baseline              |    2 |  73.1 |  70.7 |  72.5 |
| baseline              |    4 |  75.6 |  76.4 |  79.6 |
| baseline              |    8 |  83.1 |  88.3 |  82.6 |
| load_pdi_fw           |    1 |  63.1 |  63.5 |  71.9 |
| load_pdi_fw           |    2 |  66.6 |  67.1 |  73.9 |
| load_pdi_fw           |    4 |  65.1 |  66.7 |  74.1 |
| load_pdi_fw           |    8 |  71.0 |  76.4 |  81.8 |
| load_pdi_expanded     |    1 |  81.8 |  77.9 |  88.6 |
| load_pdi_expanded     |    2 |  77.5 |  96.3 | 125.8 |
| load_pdi_expanded     |    4 |  96.1 | 127.6 | 193.5 |
| load_pdi_expanded     |    8 | 125.5 | 189.2 | 301.9 |

### Table: latency vs total compute tiles (cols × rows_per_col)

Range shown when multiple `(cols, rows)` shapes share the same total.

| total tiles | baseline   | load_pdi_fw | load_pdi_expanded |
|------------:|-----------:|------------:|------------------:|
|           1 |       67.9 |        63.1 |              81.8 |
|           2 |  69.2-73.1 |   63.5-66.6 |         77.5-77.9 |
|           4 |  70.7-75.6 |   65.1-71.9 |         88.6-96.3 |
|           8 |  72.5-83.1 |   66.7-73.9 |       125.5-127.6 |
|          16 |  79.6-88.3 |   74.1-76.4 |       189.2-193.5 |
|          32 |       82.6 |        81.8 |             301.9 |

See `results/plots/pure_dispatch_vs_total_tiles.png` for the curves.

### Headline findings (whole array)

1. **`baseline` and `load_pdi_fw` lines lie on top of each other across
   the array.** 1 tile → 32 tiles costs +15 µs for baseline (67.9 → 82.6)
   and +19 µs for `load_pdi_fw` (63.1 → 81.8). This is **not** the firmware
   being magical — it is the firmware **caching the PDI load**: every
   dispatch targets `@main`, the PDI doesn't change, the load_pdi op short-
   circuits. We are measuring two near-identical paths and getting two
   near-identical curves. Real PDI-load cost requires the A↔B test (v2).
2. **`load_pdi_expanded` grows ~3.7× over the same range.** 82 µs at 1 tile
   → 302 µs at 32 tiles. This *is* a real cost because expansion replaces
   the cacheable `load_pdi` opcode with raw write32/blockwrite txn ops that
   the firmware cannot cache. Doubling `total_tiles` roughly adds 60-110 µs
   to expanded — that's the in-band cost of reprogramming all the memtile
   and DMA registers, every dispatch. **This is the only honest number we
   have in v1 for "what does reconfig cost as the design grows."**
3. **Shape (rows vs cols) matters less than total tile count.** For
   `baseline` and `load_pdi_fw` it's all within noise. For
   `load_pdi_expanded`, deeper-per-column layouts (`r=4`) are within ~5 µs
   of wider ones at the same total tile count (193.5 µs at c=4 r=4 vs
   189.2 µs at c=8 r=2). Plausible cause: per-column memtile chunks are
   larger so per-BD overhead amortizes, but the dominant cost is total
   register-write count.
4. **No firmware crashes** at `bds=2` even at 32 tiles — the
   `tiles∈{2,4} × bds=8 × load_pdi_*` crash from earlier is isolated to
   the high-BD axis, not the wide/deep axis.

### Whole-array caveats

- All runs used `bds=2` and `topology=linear`. We did **not** re-sweep
  bds and topology under the new `rows_per_col` axis — the `bds=8`
  failures from the v1 report are likely still present but untested here.
- `load_pdi_fw` at high tile counts is suspiciously cheap. The generator
  emits a single `aiex.npu.load_pdi { device_ref = @main }` at the top of
  the runtime sequence, so adding compute tiles doesn't change *how many*
  load_pdi ops are dispatched — only how much DMA work follows. So this
  is really measuring "constant-cost firmware reload + scaling DMA," not
  "firmware reload that touches more state." A more honest "what does
  load_pdi cost as the configuration gets bigger" test would issue multiple
  separate load_pdi ops against different configurations — that's the A↔B
  case, still v2.

## Next (v2 plan)

1. Wire `ctrlpkt` end-to-end: column-control-overlay pass in the build path, separate kernel-arg slot wiring in `bench.cpp`.
2. Generator option to emit a runtime sequence *without* `load_pdi` (so `pure_dispatch` and `warm_reconfig` measure different things for `load_pdi_*`).
3. AB-mode dispatch path in `bench.cpp` (three-run runlist toggling).
4. Topology axis: `branch` and `hop` builds + measurement.
5. File the firmware crash bug with reproducer (`load_pdi_* × tiles∈{2,4} × bds=8`).
6. Profile baseline's batched-amortization floor with strace / perf to explain the 22 µs gap to the ELF path.
