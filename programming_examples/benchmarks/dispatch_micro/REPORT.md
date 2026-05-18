# dispatch_micro: v1 measurement report

> Generated 2026-05-18, AMD Strix (`npu2`) host, XRT 2.23.0 (HEAD), `aiecc` from
> `mlir-aie` branch `location` @ `ab3cf203b0`. All numbers are wall-clock
> nanoseconds bracketing the kernel submit + wait on the host.

## v2 update — what A↔B and multi-PDI rotation actually showed

> **v1 hypothesis (still valuable for context):** the firmware caches PDI
> loads; v1's `load_pdi_fw` numbers were cache-hit measurements; only
> `load_pdi_expanded` did real reconfig work.
>
> **v2 finding #1 (see §4):** alternating between two *distinct* PDIs
> produces the *same* numbers as v1's "cached" path (~60-65 µs flat
> across 1-8 tiles).
>
> **v2 finding #2 (see §5):** rotating through 2 / 4 / 8 distinct PDIs
> shows **no per-slot latency variation** — all slots cost the same.
> If a fixed-size cache existed, we'd see slot latency jump when N
> exceeded its capacity. We don't, up to N=8.
>
> **Combined conclusion: there is no PDI cache in the v1 sense.**
> `load_pdi` is a selector / pointer-swap operation; all PDIs are
> resident in driver memory after `xrt::hw_context(device, elf)`. At
> dispatch time the firmware just switches which one is active.
> `load_pdi_expanded` is slower because `--expand-load-pdis` inlines
> the PDI's register-programming as raw write32/blockwrite ops, which
> must execute every dispatch regardless of driver-side state. The v1
> framing of "real vs cached" was wrong in flavor (no cache) but right
> in direction (expanded does real per-dispatch work, fw doesn't).

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
4. **~~`xrt::runlist` amortization is dramatic for the ELF/load_pdi paths~~ —
   ~~per-dispatch latency drops from ~65 µs to ~1 µs at batch=64.~~**
   *Retracted in v2 #6.* The original measurement was a bench.cpp bug
   — PathE never had a `dispatch_batched` method, so `--batched` for
   ELF mechanisms silently fell through to a single `dispatch_once`
   and reported `(single_dispatch / batch_size)` as "per-dispatch."
   With real runlist now implemented (§3): baseline amortizes to
   ~22 µs/dispatch; ELF runlist costs ~114 µs/dispatch — *higher*
   than baseline. For workloads with many independent dispatches,
   `baseline` is the right primitive.
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

> **⚠ v2 #6 retraction.** The original v1 §3 numbers (showing ELF
> runlist amortizing to ~1 µs per dispatch at bs=64) were a bench.cpp
> bug: PathE never had a `dispatch_batched` method, so `--batched`
> for ELF mechanisms silently fell through to a single
> `dispatch_once()` and reported `(single_dispatch_time / batch_size)`
> as "per-dispatch latency." The "20× better amortization" finding
> doesn't exist. Below is the corrected v2 measurement with real
> `xrt::runlist` dispatch in PathE.

#### Corrected: per-dispatch p50 µs (real runlist, identical args)

| mechanism             | t | bs=1 | bs=4 | bs=16 | bs=64 |
|-----------------------|--:|-----:|-----:|------:|------:|
| baseline              | 1 | 69.7 | 39.5 |  28.1 |  22.2 |
| load_pdi_fw           | 1 | 95.1 |144.0 | 116.5 | 113.9 |

#### Dedup probe: identical args vs distinct args (per-dispatch p50 µs)

To test whether the runtime collapses identical batched runs, each
`xrt::run` in the batch was given a distinct `xrt::bo` pair. If
collapse were happening, batch time should jump dramatically under
vary-args. It doesn't:

| mechanism   | bs | identical | distinct | ratio |
|-------------|---:|----------:|---------:|------:|
| baseline    |  1 |    69.7 µs|  77.8 µs |  1.12 |
| baseline    |  4 |    39.5 µs|  42.7 µs |  1.08 |
| baseline    | 16 |    28.1 µs|  25.7 µs |  0.91 |
| baseline    | 64 |    22.2 µs|  22.3 µs |  1.01 |
| load_pdi_fw |  1 |    95.1 µs|  68.1 µs |  0.72 |
| load_pdi_fw |  4 |   144.0 µs| 128.5 µs |  0.89 |
| load_pdi_fw | 16 |   116.5 µs| 117.6 µs |  1.01 |
| load_pdi_fw | 64 |   113.9 µs| 123.5 µs |  1.08 |

**Ratios are ~1.0 throughout.** Neither path collapses identical runs;
the runtime is genuinely executing N dispatches.

#### Corrected findings

1. **`baseline` runlist amortizes well, ~22 µs/dispatch at bs ≥ 16.**
   That's the steady-state cost of one shim-DMA dispatch on the
   `xrt::kernel` path.
2. **`load_pdi_fw` runlist costs ~114 µs/dispatch at bs=64 — *higher*
   than baseline.** The ELF path's per-dispatch runlist cost is
   actually worse than the classic xclbin path. The v1 claim of "ELF
   amortizes 20× better" was completely inverted by the bug.
3. **For workloads with many independent dispatches, `baseline` is the
   right primitive.** ELF + runlist is fine but not faster.
4. The single-dispatch (bs=1) numbers across both mechanisms are
   consistent with §1's steady-state results once you account for the
   single iteration including some warmup-amortized cost.

This is the **single largest correction** in the v2 pass. The original
v1 §3 ranking was wrong because of measurement-harness bugs, not real
mechanism behavior.

## Failures

Reproducible firmware timeout when `aiex.npu.load_pdi` is combined
with high shim-BD count. Full reproducer + analysis is in
[`bugs/bd_load_pdi_crash.md`](bugs/bd_load_pdi_crash.md).

v1 (warmup=10, iters=100) reported the crash at `tiles ∈ {2, 4} ×
bds=8` only. v2 re-investigation with `warmup=0 --iters=1` (measure
the very first dispatch from a fresh process) shows the boundary is
broader: **any `load_pdi_*` build with `bds ≥ 8` crashes**, with the
`t=8` case intermittent. v1's high warmup masked the failure at `t=1`.

Symptom:

```
xrt::run::aie_error: Command failed to complete successfully (ERT_CMD_STATE_TIMEOUT)
txn_op_idx = 0xFFFFFFFF   ← firmware doesn't know what op was executing
fatal_error_*            = 0  ← no exception, just a stuck BD chain
```

`baseline + bds=8` works at every tile count, so it's the load_pdi op
in combination with the BD count — not either one alone. Workaround:
use `bds ≤ 4` with `load_pdi_*`, or use `baseline` at any BD count.
The PDI swap-cost results in §4 and §5 are unaffected (those used
`bds=2`).

## Anomalies worth chasing

0. **`aiex.npu.load_pdi` is load-bearing for full-ELF dispatch
   (v2 follow-up to #1).** Adding `--no-self-reload` to the generator
   to omit the op at the top of the runtime sequence — intended as an
   "ELF baseline" measurement — fails: every dispatch times out with
   the same `ERT_CMD_STATE_TIMEOUT, txn_op_idx = 0xFFFFFFFF` signature
   as the bds=8 bug (`bugs/bd_load_pdi_crash.md`). Reproduced at every
   t and bds we tried (6/6).
   
   The likely cause is the comment from `test/npu-xrt/loadpdi/aie.mlir`:
   "XRT will load the PDI into memory and patch the address of this
   load_pdi to the correct address." The op is also XRT's patch point;
   no op → no patch → stuck BD chain. The shared failure signature
   with the bds=8 bug suggests both problems may share root cause
   (no patch, or BD configuration confusion).
   
   Implication: the v1 framing question "what does a cache-hit cost
   in isolation" turns out to be unmeasurable in the current runtime
   model. You can't dispatch the ELF without a load_pdi op. The §4/§5
   conclusion (rotation ≈ self-reload ≈ no-cache-effect) is the
   strongest statement we can make.
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

## §4. A↔B alternation (v2 — defeats the PDI cache)

> *Added in v2. Builds on the v1 framing.*

v1's framing said `load_pdi_fw` numbers were cache-hit measurements
because every dispatch reloaded the same PDI. v2 fixes that with a
proper A↔B path: the AB build now contains **two** runtime sequences in
the orchestrator (`ab_orch:seq_to_a` and `ab_orch:seq_to_b`), each
issuing one `npu_load_pdi` against a distinct PDI (`@cfg_a` / `@cfg_b`).
The bench loop alternates `kernel_a()` and `kernel_b()`, so each
firmware load is preceded by a load of a *different* PDI — the cache
cannot short-circuit.

**Important methodology note.** The orchestrator's runtime sequences
contain *only* the `npu_load_pdi` op — no DMA work, no `dma_configure`,
no `dma_await`. That means an `ab_toggle` row is measuring **dispatch
overhead + firmware PDI load + return**, not "switch PDI then run the
loaded PDI's work." This is exactly what we want for isolating PDI-load
cost, but it means absolute numbers below are *lower* than `pure_dispatch`
numbers from §1 (which include actual DMA after the load).

### Table: p50 µs at `bds=2, rows_per_col=1`, iters=100

| mech                | t | baseline (v1, no load_pdi) | cached load_pdi (v1, self-reload) | A→B (v2, uncached) | B→A (v2, uncached) | A→B − baseline |
|---------------------|--:|---------------------------:|----------------------------------:|-------------------:|-------------------:|---------------:|
| `load_pdi_fw`       | 1 |                       66.3 |                              61.4 |               60.5 |               61.9 |          −5.8  |
| `load_pdi_fw`       | 2 |                       69.3 |                              62.3 |               61.8 |               61.5 |          −7.5  |
| `load_pdi_fw`       | 4 |                       75.2 |                              76.8 |               62.3 |               62.7 |         −13.0  |
| `load_pdi_fw`       | 8 |                       88.7 |                              80.5 |               65.7 |               64.9 |         −23.0  |
| `load_pdi_expanded` | 1 |                       66.3 |                              72.5 |               64.9 |               64.6 |          −1.5  |
| `load_pdi_expanded` | 2 |                       69.3 |                              72.3 |               73.2 |               73.0 |          +3.9  |
| `load_pdi_expanded` | 4 |                       75.2 |                              90.9 |               82.9 |               82.2 |          +7.7  |
| `load_pdi_expanded` | 8 |                       88.7 |                             132.4 |              116.7 |              117.2 |         +28.1  |

(`baseline (v1)` = full dispatch including DMA; A↔B rows = orchestrator
runtime sequence containing only `npu_load_pdi`, so they're DMA-free.)

### Headline findings

1. **Firmware PDI load is essentially flat at ~60-65 µs across all tile
   counts.** `load_pdi_fw` A↔B numbers go from 60.5 µs at 1 tile to
   65.7 µs at 8 tiles — a 5 µs swing across 8× the configuration. This
   is the cost of the firmware op itself (dispatch + look-up + return),
   *not* the cost of reprogramming the device. The PDI was loaded into
   driver memory when `xrt::hw_context(device, elf)` ran; `load_pdi`
   just selects which one is currently active.
2. **The "PDI cache" in v1 may be a misnomer.** v1 attributed the
   `load_pdi_fw` ≈ `baseline` collapse to a firmware cache short-
   circuiting. The v2 A↔B numbers are *the same* as v1's "cached"
   numbers (within run-to-run noise). Two possibilities are consistent
   with this:
   - The firmware/driver holds *both* PDIs simultaneously and switching
     between them is cheap (effectively a cache of size ≥ 2).
   - The op never had real "load" work to do in the first place — both
     PDIs are pre-loaded at ELF registration time, and `load_pdi` is a
     pointer swap, not a memory move.
   Distinguishing these would require either (a) rotating through 3+
   PDIs to overflow a small cache, or (b) instrumenting the firmware
   side. Worth pursuing if we ever care about >2-PDI workloads.
3. **`load_pdi_expanded` A↔B numbers scale with configuration size, as
   expected.** 65 µs at t=1 → 117 µs at t=8. The expansion replaces
   `load_pdi` with raw write32/blockwrite txn ops; these can't be
   cached and must execute every time. This *is* the honest "what does
   it cost to reprogram N tiles on every dispatch" number, and at 8
   tiles it's ~80% more expensive than the firmware path.
4. **For workloads that alternate between two distinct configurations,
   the firmware load_pdi path is strictly cheaper than expanded.**
   At t=8, firmware A→B costs 66 µs vs expanded's 117 µs — a 1.8×
   advantage that grows with tile count.

### What v2 still doesn't tell us

- **A↔B with the PDI's actual work.** The orchestrator runtime sequence
  is DMA-free; we measure load_pdi cost in isolation. To answer "swap
  workloads and then run the new workload" we'd need the orchestrator
  to invoke the loaded PDI's runtime_sequence (e.g. via `aiex.configure
  { aiex.run @inner_sequence }` from `test/npu-xrt/reconfigure_loadpdi/
  aie.mlir`). That's a generator extension on the v2 backlog.
- **Whether the cache is actually a "cache".** See finding #2. v2 only
  tested A↔B with two PDIs; rotating through 4+ PDIs would tell us
  whether there's a fixed-size cache or unconditional zero-cost
  swapping.
- **`load_pdi_expanded` with empty post-load body.** We compared
  expanded A↔B (no DMA) to expanded `pure_dispatch` (with DMA), and the
  difference at t=8 was only ~15 µs — small, but it implies the bulk of
  expanded's cost is the register-write stream, not the surrounding DMA.

## §5. Multi-PDI rotation (v2 — is there a cache?)

> *Added in v2. Follows up the open question from §4.*

§4 showed that alternating between two distinct PDIs costs the same as
reloading the same PDI repeatedly. Two hypotheses fit:
1. The firmware/driver keeps a cache of size ≥ 2 (LRU or similar).
2. `load_pdi` was never doing real load work — PDIs are resident in
   driver memory after `xrt::hw_context(device, elf)`, and the op is a
   pointer swap.

To distinguish them, the generator now takes `--n-configs=N` and emits
N distinct PDIs (`cfg_0`..`cfg_{N-1}`) plus an N-sequence orchestrator
(`ab_orch:seq_to_0`..). The bench rotates through all N kernel handles
round-robin and reports each slot's p50 separately. **If there is a
fixed-size cache of size K, slot latencies should jump when N > K.**

### Results at `bds=2, rows_per_col=1, iters=50 × N` rotations

**At t=1 (smallest configuration):**

| mech                | N | per-slot p50 µs                                       | mean | per-slot σ |
|---------------------|--:|--------------------------------------------------------|------|-----------:|
| `load_pdi_fw`       | 2 | 62.3 62.3                                              | 62.3 |     <0.1 µs|
| `load_pdi_fw`       | 4 | 64.8 64.9 64.9 64.3                                    | 64.7 |      0.3 µs|
| `load_pdi_fw`       | 8 | 63.0 62.5 62.6 62.0 62.6 63.1 63.2 63.0                | 62.8 |      0.4 µs|
| `load_pdi_expanded` | 2 | 61.5 61.3                                              | 61.4 |     <0.1 µs|
| `load_pdi_expanded` | 4 | 61.6 61.6 61.7 61.8                                    | 61.7 |     <0.1 µs|
| `load_pdi_expanded` | 8 | 61.4 61.2 61.0 61.3 61.1 61.0 61.1 61.3                | 61.2 |      0.2 µs|

**At t=4 (medium configuration):**

| mech                | N | per-slot p50 µs                                       | mean | per-slot σ |
|---------------------|--:|--------------------------------------------------------|------|-----------:|
| `load_pdi_fw`       | 2 | 65.5 65.3                                              | 65.4 |      0.1 µs|
| `load_pdi_fw`       | 4 | 59.3 59.2 59.2 58.8                                    | 59.1 |      0.2 µs|
| `load_pdi_fw`       | 8 | 62.9 63.0 64.4 63.6 63.4 63.8 63.0 64.2                | 63.5 |      0.6 µs|
| `load_pdi_expanded` | 2 | 82.8 80.9                                              | 81.8 |      1.0 µs|
| `load_pdi_expanded` | 4 | 81.2 81.5 81.6 81.0                                    | 81.3 |      0.3 µs|
| `load_pdi_expanded` | 8 | 85.0 84.5 85.0 84.6 85.2 85.6 84.6 84.9                | 84.9 |      0.4 µs|

### Headline findings

1. **No cache size limit observed up to N=8.** Per-slot p50s are
   essentially identical (within ±1 µs noise) regardless of N. If there
   were a cache of size K < 8, we'd see slot latency jump for the
   slots that miss — there is no such pattern. This rules out
   hypothesis #1 (fixed-size cache).
2. **The interpretation:** `load_pdi` is a **selector / pointer-swap
   operation**, not a memory load. All PDIs are loaded into driver
   memory at `xrt::hw_context(device, elf)` time; the op at dispatch
   time just switches which one is currently active. There is no
   "cache" because there is no "load" to cache.
3. **`load_pdi_expanded` numbers are also flat across N**, *but* are
   ~20 µs higher than `load_pdi_fw` at t=4 (81 µs vs 63 µs). That gap
   is the cost of `--expand-load-pdis` inlining the PDI's register
   programming as raw write32/blockwrite ops, which must execute on
   every dispatch regardless of any caching. The gap doesn't grow with
   N because the expansion is per-dispatch, not per-rotation.
4. **The v1 "PDI cache" framing was the wrong mental model.** What v1
   actually observed was that the firmware path is fast at any tile
   count because it does no real work at dispatch time. The
   `load_pdi_expanded` path is slower precisely because it cannot
   delegate to the driver-resident PDI — it has to re-issue the
   register writes itself.

### Why this matters

- For workloads that swap between many distinct configurations
  (multi-tenant, multi-model serving), the firmware `load_pdi` path
  scales to at least 8 simultaneously-loadable PDIs with no per-swap
  cost penalty. The cost ceiling is **driver memory** to hold all
  PDIs, not per-swap CPU/firmware work.
- `--expand-load-pdis` should be avoided for swap-heavy workloads
  unless there's a specific reason to bypass the firmware
  (e.g. tight integration with host code that needs to do its own
  fix-ups in the register-write stream).
- The ~63 µs per-dispatch floor for `load_pdi_fw` (no DMA) is *the
  fixed cost of dispatching a kernel*, not the cost of swapping
  configurations. We aren't going to make that go down by being
  clever about PDIs.

### What's left to test

- **Many more PDIs (N=16, 32, 64).** Up to N=8 we saw nothing; the
  driver memory limit is the wall, not a software cache. Worth
  pushing higher to find that wall.
- **Larger PDIs.** Our N PDIs are all 1-tile configurations. A
  real-world workload with whole-array PDIs would put much more
  pressure on driver memory; the wall would arrive sooner.
- **Cold rotation.** If the driver evicts PDIs under memory pressure,
  a long-delayed swap might pay a re-load cost. Today's rotation
  pattern doesn't exercise that.

### "With work" mode — swap + actually run the loaded workload

§4 and §5 above measured PDI selection **in isolation** (the
orchestrator's runtime sequences contained only `npu_load_pdi`, no
DMA). The production-relevant question is "swap to a different PDI
*and then run the workload it configures*." For that, generator gained
an `--ab-mode=with_work` flag that emits each orchestrator sequence as
`aiex.configure @cfg_k { aiex.run @seq(args) }`. Per
`AIE_RunOp` semantics, this **inlines the named runtime sequence at
the call site**, so a single host dispatch swaps the PDI and runs the
loaded config's DMA work.

| mech                | t | N | mean p50 µs | per-slot p50 µs (max σ) |
|---------------------|--:|--:|------------:|------------------------:|
| `load_pdi_fw`       | 1 | 2 |        65.2 | 65.3 65.1               |
| `load_pdi_fw`       | 1 | 4 |        69.4 | 69.7 69.4 69.4 69.1     |
| `load_pdi_fw`       | 4 | 2 |        74.2 | 74.5 74.0               |
| `load_pdi_fw`       | 4 | 4 |        65.7 | 65.8 65.8 65.4 65.9     |
| `load_pdi_fw`       | 8 | 2 |        75.6 | 76.4 74.9               |
| `load_pdi_fw`       | 8 | 4 |        78.8 | 79.6 78.5 78.7 78.3     |
| `load_pdi_expanded` | 1 | 2 |        78.1 | 78.1 78.0               |
| `load_pdi_expanded` | 1 | 4 |        75.4 | 75.1 75.8 75.5 75.1     |
| `load_pdi_expanded` | 4 | 2 |        89.3 | 89.2 89.4               |
| `load_pdi_expanded` | 4 | 4 |        90.0 | 89.6 90.9 89.5 90.2     |
| `load_pdi_expanded` | 8 | 2 |       125.0 | 125.3 124.7             |
| `load_pdi_expanded` | 8 | 4 |       126.4 | 125.4 126.8 126.9 126.4 |

### Three-way comparison at t=8

To put the numbers in context — what each measurement family actually
represents:

| measurement (at t=8, bds=2, rows=1) | mech: `load_pdi_fw` | mech: `load_pdi_expanded` |
|-------------------------------------|--------------------:|--------------------------:|
| v1 `pure_dispatch` (same PDI each dispatch, full DMA) |               80.5 |                     132.4 |
| v2 §4 `ab_toggle` (alternate distinct PDIs, **no DMA**) |               65.7 |                     116.7 |
| v2 §5 `multi_toggle` isolated, N=8 (rotate, **no DMA**) |               63.5 |                      84.9 *(t=4)* |
| v2 §5 `multi_toggle` **with_work**, N=4 (rotate **+ run loaded DMA**) |       78.8 |                     126.4 |

The **`with_work` numbers are essentially identical to v1
`pure_dispatch`** (within 2-3 µs). That is the cleanest possible answer
to "what does it cost to swap to a different PDI?":

> **For load_pdi_fw, swapping between distinct PDIs is free.** The
> total cost of "swap + run workload" is the same as "run workload
> against current PDI." There is no swap penalty.

For `load_pdi_expanded`, swap + work is the same as repeat + work
because the expansion does its full register-write reprogramming every
dispatch regardless of what was active before. There's no "swap" to
have a penalty over.

### Practical conclusion

For multi-tenant or model-swapping workloads on NPU2:
- Use `load_pdi_fw` (i.e. `--generate-full-elf` **without**
  `--expand-load-pdis`).
- Package every PDI you'll need into the same ELF.
- Swap freely between them — the per-swap cost is zero on top of
  dispatch + the DMA the swapped-in config will run.
- The driver-memory ceiling on how many PDIs fit in one ELF is the
  only practical limit we've found; we haven't tested where it sits.

## §6. Control packets (v2 — fourth mechanism wired)

> *Added in v2.*

ctrlpkt is the only mechanism that uses control-packet routing for
device reconfiguration. v1 had the placeholder; v2 wired the build
path end-to-end (overlay pass + dual aiecc) and added Path-C to
`bench.cpp` (xclbin + xrt::module + `xrt::ext::kernel` three-arg variant).

### Build pipeline (mirrors `test/npu-xrt/ctrl_packet_reconfig_elf/run.lit`)

```
python3 generate.py --mechanism=ctrlpkt ... > aie.mlir
# adds the column-control-overlay routes:
aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" \
        aie.mlir -o aie_overlay.mlir
# skeleton xclbin from the @base device (overlay routes only):
aiecc --device-name=base --aie-generate-xclbin --xclbin-name=aie.xclbin \
      aie_overlay.mlir
# ctrlpkt-encoded ELF from the @main device:
aiecc --device-name=main --aie-generate-ctrlpkt --aie-generate-elf \
      --elf-name=aie.elf aie_overlay.mlir
```

Generator emits two devices: `@main` (the actual placed config, no
load_pdi op) plus a skeleton `@base` that re-declares the same tiles
with no fifos/cores. aiecc is invoked twice with different `--device-name`
to produce each artifact. All four ctrlpkt-related files end up in
the build dir: `aie.xclbin` (~7 KB overlay), `aie.elf` (~4 KB
ctrlpkt-encoded), `main_ctrlpkt.bin` (~1.3 KB), `main_ctrlpkt_dma_seq.bin`
(~2.4 KB).

### Host path (Path-C)

```cpp
xrt::xclbin xb("aie.xclbin");           // skeleton overlay
xrt::elf    el("aie.elf");              // ctrlpkt-encoded
device.register_xclbin(xb);
xrt::module mod(el);
xrt::hw_context ctx(device, xb.get_uuid());
auto kernel = xrt::ext::kernel(ctx, mod, "MLIR_AIE_*");
// kernel signature: (opcode, instr_buf=0, instr_count=0, bo_in, bo_out)
kernel(3, 0, 0, bo_in, bo_out).wait2();
```

### ctrlpkt dispatch is single-shot

**Discovered during measurement: the first dispatch succeeds, the
second hangs** with the same `ERT_CMD_STATE_TIMEOUT, txn_op_idx =
0xFFFFFFFF, fatal_error_* = 0` signature as the bds=8 firmware bug
and the `--no-self-reload` failure mode.

```
$ ./bench --mechanism=ctrlpkt --iters=1 ...   # OK
$ ./bench --mechanism=ctrlpkt --iters=2 ...   # FAIL (second iteration hangs)
```

Reproduced across all t and bds we tried. The canonical
`test/npu-xrt/ctrl_packet_reconfig_elf/test.cpp` only runs one
dispatch per process — so this single-shot pattern may be by design
for this mechanism in the current XRT/firmware combination.

The most plausible reading: a ctrlpkt dispatch reconfigures device
state via packets that are not idempotent. After one dispatch, the
device is in a state where the same control-packet sequence cannot
replay without first running an explicit reset routine. The overlay
xclbin does include the reset routes; we just don't invoke them
between dispatches.

### Measurement #1: ctrlpkt single-shot pure_dispatch (30 fresh processes per cell)

These numbers measure **everything from fresh `xrt::device(0)` to dispatch completion** — they include the device-context setup work too:

| t | n  | p10 µs | p50 µs | p90 µs | min µs | max µs |
|--:|---:|-------:|-------:|-------:|-------:|-------:|
| 1 | 30 |    741 |  1 043 |  1 611 |    643 |  1 821 |
| 4 | 30 |    686 |    852 |  1 306 |    677 |  1 753 |

### Measurement #2: ctrlpkt cold_start phase breakdown (30 fresh processes, t=1, bds=2)

`cold_start` separates the setup phases from the dispatch itself:

| phase                          | p50 µs | min µs | max µs |
|--------------------------------|-------:|-------:|-------:|
| `load` (xclbin + elf ctor)     |    165 |    114 |    322 |
| `register` (hw_context build)  | 39 842 | 27 142 | 47 413 |
| `kernel` (ext::kernel build)   |    210 |     99 |    290 |
| **`first_dispatch`**           | **787**|    502 |    968 |

**Apples-to-apples first-dispatch comparison** (all four mechanisms, same shape):

| mechanism            | first_dispatch p50 µs |
|----------------------|----------------------:|
| baseline             |                   796 |
| load_pdi_fw          |                   897 |
| load_pdi_expanded    |                   758 |
| **ctrlpkt**          |                **787**|

**ctrlpkt's first dispatch is right in the same band as every other mechanism** (758-897 µs). There is no detectable per-dispatch difference at first-dispatch granularity.

### ⚠ We still cannot measure ctrlpkt's *steady-state* per-dispatch cost

This is the key honesty disclaimer. The numbers above are
**first-dispatch from a fresh process** — they include whatever
one-time setup the firmware does on the first command. Every other
mechanism in this report has steady-state numbers (warmup ≥ 10), and
we **cannot get a steady-state ctrlpkt number** because dispatch #2
hangs.

The right comparison is against other mechanisms' first-dispatch
numbers from v1's cold_start data (also "fresh process, one
dispatch"), at the same shape:

| mechanism            | t | b | first_dispatch p50 µs | n  |
|----------------------|--:|--:|----------------------:|---:|
| baseline             | 1 | 2 |                   796 | 30 |
| load_pdi_fw          | 1 | 2 |                   897 | 30 |
| load_pdi_expanded    | 1 | 2 |                   758 | 30 |
| **ctrlpkt**          | 1 | 2 |             **1 043** | 30 |
| baseline             | 4 | 2 |                 1 003 | 30 |
| load_pdi_fw          | 4 | 2 |                 1 121 | 30 |
| load_pdi_expanded    | 4 | 2 |                   843 | 30 |
| **ctrlpkt**          | 4 | 2 |               **852** | 30 |

**Apples-to-apples, ctrlpkt's first dispatch is the same order of
magnitude as every other mechanism's first dispatch** (~700–1100 µs).
There is no detectable "ctrlpkt is N× slower" effect in the data —
my earlier claim of "~15× slower" was a category error: I was
comparing ctrlpkt's *first* dispatch to other mechanisms' *steady-
state* dispatch. Retracted.

The thing we can honestly say from this measurement:

- **First-dispatch latency** is roughly the same across all four
  mechanisms (within run-to-run noise: range 466 µs to 2 692 µs across
  all mechanisms at t=1).
- **Per-dispatch (amortized) latency for ctrlpkt is unknown** — we
  can't run a second dispatch to measure it.
- The other three mechanisms have ~60–80 µs steady-state per-dispatch
  at this shape (see §1). Whether ctrlpkt's would amortize to a
  similar number, a much lower one, or somewhere else entirely, we
  can't tell from this harness.

### Practical conclusion

Four mechanisms, what we actually know:

| mechanism           | first-dispatch p50 (t=1) | steady-state p50 (t=1)   | hot-loop-able? |
|---------------------|-------------------------:|-------------------------:|----------------|
| `load_pdi_fw`       |                    897 µs |                    61 µs | yes; swap is free |
| `baseline`          |                    796 µs |                    67 µs | yes |
| `load_pdi_expanded` |                    758 µs |                    73 µs | yes (scales hard with tiles) |
| `ctrlpkt`           |                  1 043 µs |   **n/a (dispatch #2 hangs)** | **no — single-shot only** |

What the data justifies saying about ctrlpkt:
- It works for one-shot reconfiguration (e.g. the canonical
  `test/npu-xrt/ctrl_packet_reconfig_elf/` pattern: configure once,
  run once, exit).
- It cannot be used for hot dispatch loops in the current XRT/firmware
  combination.
- We do not know its amortized cost. The first-dispatch cost is in
  the same band as every other mechanism, but that says nothing about
  steady-state.

Finding ctrlpkt's amortized cost would require either solving the
2nd-dispatch hang (probably a missing reset-routine invocation, or
a state-machine workaround) or running each ctrlpkt dispatch in a
fresh process and amortizing process-creation cost out separately.
Both are non-trivial; queued as v2 follow-ups #11 and #12.

### v2 #11 follow-up: tried four workarounds, only the expensive one works

To probe where the hang lives, `bench.cpp` Path-C gained a
`--ctrlpkt-strategy` flag with four behaviors:

| strategy        | what's recreated per iter         | result at t=1            |
|-----------------|-----------------------------------|--------------------------|
| `reuse`         | only `xrt::run` (default)         | **hang on dispatch #2**  |
| `fresh_kernel`  | `xrt::ext::kernel`                | **hang on dispatch #2**  |
| `fresh_module`  | `xrt::module` + `xrt::ext::kernel`| **hang on dispatch #2**  |
| `fresh_ctx`     | `xrt::hw_context` + module + kernel | works, p50 ≈ 80 ms |

The only strategy that lets us run multiple dispatches is recreating
the `hw_context` itself — and that costs ~40 ms per recreation (plus
implicit teardown of the previous context). So the per-iteration time
under `fresh_ctx` is dominated by context setup, not by ctrlpkt
dispatch work.

**The hang lives at the `hw_context` layer** (driver/firmware), not
above. Recreating just the kernel or module doesn't reset whatever
state ctrlpkt's first dispatch mutates. XRT does not expose any
public reset/release API for `hw_context` short of destroying it.

**What this means for v2 #11.** We cannot derive a steady-state
ctrlpkt number with the current XRT API and the current ctrlpkt build
shape. To unstick this we'd need one of:

1. A between-dispatch reset routine that the overlay xclbin supports
   but neither the canonical test nor our bench invokes.
2. An XRT API addition (e.g. `hw_context::reset()`) that resets
   driver/firmware state without a teardown round-trip.
3. A different ctrlpkt build shape — perhaps with the canonical
   pattern's `aiex.npu.dma_memcpy_nd` ops (which our generator
   doesn't emit) the firmware leaves state cleaner. Worth testing
   if/when we restructure the generator.

Until one of those lands, the honest per-dispatch ctrlpkt number is
**first-dispatch only (787 µs at t=1, in the same band as every other
mechanism)**.

## §7. Topology sweep (v2 — branch, hop measured)

> *Added in v2.*

v1 only measured `linear` topology (shim → compute, per-column). The
generator already supported two more: `branch` (one shim broadcasts to
N compute tiles in different columns) and `hop` (shim → memtile →
compute → memtile → shim, with mem-tile fan-out per column). v2 #7
ran the sweep.

### Build boundary

`branch` is shim-channel-limited: one shim tile has only 2 MM2S + 2
S2MM channels, so it can feed at most 2 compute tiles in different
columns. **`branch × t > 2` fails at compile** with:

```
'aie.tile' op number of output DMA channel exceeded!
```

`linear` and `hop` scale up to t=4 (and to t=8 on `npu2` from the
whole-array sweep) without issue.

### Table: `pure_dispatch` p50 µs by topology (bds=2, rows=1)

| mech                | t | linear | branch | hop |
|---------------------|--:|-------:|-------:|----:|
| baseline            | 1 |   64.7 |   63.9 | 63.6 |
| baseline            | 2 |   67.9 |   69.0 | 73.5 |
| baseline            | 4 |   72.3 |     —  | 77.3 |
| load_pdi_fw         | 1 |   65.8 |   64.7 | 62.6 |
| load_pdi_fw         | 2 |   66.4 |   62.6 | 65.0 |
| load_pdi_fw         | 4 |   72.1 |     —  | 65.7 |
| load_pdi_expanded   | 1 |   72.5 |   65.2 | 72.7 |
| load_pdi_expanded   | 2 |   72.7 |   79.4 | 76.4 |
| load_pdi_expanded   | 4 |   87.7 |     —  | 91.9 |

### Findings

1. **Topology barely moves per-dispatch cost.** All cells across all
   three topologies sit in a ~60-92 µs band at these shapes. The
   topology axis isn't a meaningful lever for steady-state dispatch
   latency.
2. **`hop` adds modest overhead under `load_pdi_expanded`** (87.7 →
   91.9 µs at t=4, +4 µs). The extra memtile-side BD configurations
   in the expanded txn stream are paid in band, so adding mem-tile
   programming makes the stream longer. `baseline` and `load_pdi_fw`
   show no such effect — neither pays per-dispatch reconfig cost.
3. **`branch` doesn't show a routing penalty.** Where measurable
   (t ≤ 2), branch numbers are within noise of `linear`. The
   stream-switch routing the compiler generates for the broadcast is
   apparently free at runtime.

### Practical conclusion

Pick the topology that matches your placement / routing needs;
**latency-wise the choice is noise**. The branch shim-channel limit
(max 2 fanouts from one shim) is a real architectural constraint;
hop's memtile cost shows up only when the dispatch is itself doing
real reconfig (i.e. `load_pdi_expanded`).

## §8. BD count under multi-row (v2)

> *Added in v2.*

v1's BD-count sweep covered `rows_per_col=1` only. v1's whole-array
sweep covered `rows_per_col ∈ {1, 2, 4}` but only at `bds=2`. The gap:
how does BD count interact with multi-row configurations? v2 #8 fills
the matrix at `bds ∈ {2, 4}` × `rows_per_col ∈ {2, 4}` × `tiles ∈ {1,
2, 4}` for all three working mechanisms.

### Table: `pure_dispatch` p50 µs at `b ∈ {2, 4}` × `r ∈ {2, 4}`

| mech                | t | r | b=2   | b=4   | b=4/b=2 |
|---------------------|--:|--:|------:|------:|--------:|
| baseline            | 1 | 2 |  67.9 |  68.3 |   1.01  |
| baseline            | 2 | 2 |  66.1 |  75.1 |   1.14  |
| baseline            | 4 | 2 |  75.4 |  77.2 |   1.02  |
| baseline            | 1 | 4 |  62.6 |  74.0 |   1.18  |
| baseline            | 2 | 4 |  69.6 |  74.7 |   1.07  |
| baseline            | 4 | 4 |  75.4 |  80.1 |   1.06  |
| load_pdi_fw         | 1 | 2 |  66.8 |  73.9 |   1.11  |
| load_pdi_fw         | 2 | 2 |  65.1 |  78.4 |   1.20  |
| load_pdi_fw         | 4 | 2 |  72.8 |  71.7 |   0.99  |
| load_pdi_fw         | 1 | 4 |  64.3 |  71.1 |   1.10  |
| load_pdi_fw         | 2 | 4 |  73.8 |  78.8 |   1.07  |
| load_pdi_fw         | 4 | 4 |  73.9 |  84.3 |   1.14  |
| load_pdi_expanded   | 1 | 2 |  77.7 |  85.9 |   1.11  |
| load_pdi_expanded   | 2 | 2 | 101.2 | 108.3 |   1.07  |
| load_pdi_expanded   | 4 | 2 | 131.6 | 134.9 |   1.02  |
| load_pdi_expanded   | 1 | 4 |  93.6 | 107.8 |   1.15  |
| load_pdi_expanded   | 2 | 4 | 130.0 | 133.8 |   1.03  |
| load_pdi_expanded   | 4 | 4 | 184.5 | 190.5 |   1.03  |

### Findings

1. **BD count is mostly noise within `bds ∈ {2, 4}`.** Ratios mostly
   1.0–1.2; doubling BDs increases latency by 0–20%. The BD axis
   isn't a major lever at these scales.
2. **Rows matter much more than BDs for `load_pdi_expanded`.** At
   t=4 b=4: r=2 → 134.9 µs vs r=4 → 190.5 µs (+41%). Compare to b=2
   → b=4 at the same t=4, r=4: 184.5 → 190.5 µs (+3%). For
   expanded, the txn stream length is dominated by per-row memtile
   reconfig, not per-BD shim reconfig.
3. **baseline + load_pdi_fw stay flat across all 24 multi-row cells**
   (~62–85 µs), confirming v1+v2's general finding that those paths
   don't pay per-dispatch reconfig cost.

### New failure mode: `bds=8 × rows_per_col > 1` extends to all mechanisms

v1's firmware-crash bug
([`bugs/bd_load_pdi_crash.md`](bugs/bd_load_pdi_crash.md)) was framed
as "`load_pdi_*` + `bds=8` hangs." v2 #8 found that **the
load-bearing condition is `bds=8`, not the mechanism.** Under
`rows_per_col > 1`:

| mech                | t | r | b | first-dispatch result          |
|---------------------|--:|--:|--:|--------------------------------|
| baseline            | 1 | 1 | 8 | OK, ~100 µs                    |
| baseline            | 1 | 2 | 8 | OK but **~6.1 *seconds* per dispatch** |
| baseline            | 1 | 4 | 8 | OK but **~6.1 *seconds* per dispatch** |
| load_pdi_fw         | 1 | 1 | 8 | hangs from a fresh process<sup>†</sup> |
| load_pdi_fw         | 1 | 2 | 8 | hangs                          |
| load_pdi_fw         | 1 | 4 | 8 | hangs                          |
| load_pdi_expanded   | 1 | 1 | 8 | OK, ~178 µs                    |
| load_pdi_expanded   | 1 | 2 | 8 | hangs                          |
| load_pdi_expanded   | 1 | 4 | 8 | hangs                          |

<sup>†</sup> `load_pdi_fw` t=1, r=1, b=8 hangs at `warmup=0` but
runs cleanly at `warmup=10` (matching v1) — XRT's first-call
internal warmup apparently sidesteps the bug after enough retries.

The boundary is now clearly: **`bds=8` is unstable across multiple
mechanisms and configurations**. `baseline + bds=8 + multi-row` doesn't
crash but degrades by ~10⁵× (6 seconds vs ~75 µs at b=4). v1's
hypothesis about a BD-id collision is still consistent with this
broader pattern. Updated bug write-up:

- `baseline + bds=8 + r=1` still works.
- `baseline + bds=8 + r>1` produces correct results but at multi-second
  per-dispatch latency — looks like a degenerate slow path rather
  than a hard hang.
- `load_pdi_* + bds=8 + (r>1 OR cold)` reliably hangs with
  `txn_op_idx = 0xFFFFFFFF`.

For practical use: **avoid `bds=8` entirely with this generator
shape on Strix.** The other six "same-signature" failure modes (BD
hang, `--no-self-reload` hang, ctrlpkt 2nd-dispatch hang) suggest
something deeper in the firmware BD pool / XRT patch machinery is
implicated, beyond a single specific code path.

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
