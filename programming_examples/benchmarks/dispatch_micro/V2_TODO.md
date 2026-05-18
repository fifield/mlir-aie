# dispatch_micro v2 — TODO

Durable task list for the v2 pass. Order is priority: do #1 first.
Each item lists its **goal**, **acceptance criteria**, and any **dependencies**.
When done, move the item under "## Completed" and update REPORT.md.

Status keys: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked

---

## 1. A↔B dispatch in `bench.cpp` (defeats the PDI cache)  **[x] DONE**

**Result.** Generator now emits two runtime sequences in the AB
orchestrator (`ab_orch:seq_to_a`, `ab_orch:seq_to_b`), each calling one
`npu_load_pdi`. `bench.cpp` got a `--metric=ab_toggle` path that
alternates `xrt::ext::kernel` handles and reports each direction's p50
in JSONL. Ran on Strix at `tiles ∈ {1,2,4,8} × bds=2 × rows=1` for
both `load_pdi_fw` and `load_pdi_expanded` (16 rows).

**Headline.** PDI cache may not exist as v1 described it. Alternating
distinct PDIs produces the *same* numbers as v1's "cached" path
(~60-65 µs flat at any tile count). Firmware path is essentially free
regardless of cache state; expanded path pays ~117 µs at t=8.
Written up as §4 in REPORT.md.

**Limitation surfaced.** The AB orchestrator's runtime sequences
contain only `npu_load_pdi` — no DMA after. So we measured load_pdi in
isolation, not "swap workload then run it." That requires the
`aiex.configure { aiex.run @inner_sequence }` pattern from
`test/npu-xrt/reconfigure_loadpdi/aie.mlir`. Filed as new task #9.



**Goal.** Get an honest `load_pdi_fw` number by alternating between two
distinct PDIs in the same loaded ELF, so the firmware's PDI cache can't
short-circuit the load.

**Why it's first.** Without this, every v1 `load_pdi_fw` number is a
cache-hit measurement. This is the single biggest unanswered question
in the v1 report.

**Implementation sketch.**
- `generate.py --ab` already emits `@cfg_a`, `@cfg_b`, and an `@ab_orch`
  device. The orchestrator currently has one `seq` that calls
  `npu_load_pdi(@cfg_a)` then `npu_load_pdi(@cfg_b)` back-to-back.
  Replace with **two** runtime sequences (`seq_to_a`, `seq_to_b`) so the
  host can dispatch each direction independently.
- `bench.cpp`: add a `--metric=ab_toggle` path. Two `xrt::ext::kernel`
  handles (`ab_orch:seq_to_a`, `ab_orch:seq_to_b`). Loop alternates
  A→B and B→A dispatches and reports each direction's p50 separately
  in JSONL (`direction: "a_to_b" | "b_to_a"`).
- Build path stays `--generate-full-elf` — aiebu packages both PDIs.
- Sanity check: confirm `load_pdi_fw` numbers under `ab_toggle` are
  *higher* than under `pure_dispatch` (cache miss vs cache hit). If
  they're the same, the cache isn't getting busted — debug before
  publishing numbers.

**Acceptance.**
- One new metric in `results/results.jsonl`: `ab_toggle` with
  `direction` field. 100-iter run completes for `mechanism ∈
  {load_pdi_fw, load_pdi_expanded}` × `tiles ∈ {1, 4}` × `bds=2`.
- Headline number reproduces: "real load_pdi cost = X µs at N tiles,
  vs Y µs cache-hit cost from v1." Numbers added to a new `REPORT.md`
  section "§4. A↔B alternation (real load_pdi cost)".
- `pure_dispatch` still works for `baseline` / `load_pdi_*` (no
  regressions in the existing v1 path).

**Files touched.** `generate.py`, `bench.cpp`, `REPORT.md`,
`results/results.jsonl`, maybe `scripts/plot.py` for a direction-aware
chart.

---

## 2. `--no-self-reload` generator flag

**Goal.** Subtract cache-hit cost from baseline-with-load_pdi to isolate
what the PDI-cache lookup itself costs (~ a few µs we believe).

**Why.** Once #1 lands, we'll have three numbers per mechanism:
"baseline (no load_pdi)", "load_pdi cache-hit", "load_pdi cache-miss".
The cache-hit minus baseline gap is currently invisible.

**Implementation sketch.**
- Add `--no-self-reload` to `generate.py`. When set, omit the
  `npu_load_pdi(device_ref=name)` op at the top of the single-device
  runtime sequence.
- Build matrix grows by one cell per `load_pdi_*` mechanism. Mark the
  resulting builds with a suffix in the key (`_nsr`).
- Compare to the existing v1 numbers in the same row.

**Acceptance.**
- A new column in §1 of REPORT.md showing per-mechanism with/without
  self-reload deltas. The cache-hit cost should be ≤ 5 µs based on the
  v1 observation that `load_pdi_fw` ≈ `baseline`.

**Files touched.** `generate.py`, `Makefile`, `REPORT.md`.

---

## 3. ctrlpkt end-to-end

**Goal.** Get the fourth mechanism into the head-to-head matrix.

**Two pieces.**
1. **Build path:** `aiecc --aie-generate-ctrlpkt` currently fails on our
   placed-IRON output with:
   ```
   failed to legalize operation 'aiex.npu.dma_memcpy_nd' ...
   metadata = @ctrlpkt_col0_mm2s_chan0 ...
   Error: Control packet DMA pipeline failed
   ```
   Reference recipe (`test/npu-xrt/ctrl_packet_reconfig_elf/run.lit:11`)
   shows the column-control-overlay pass must run first:
   ```
   aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true"
   ```
   Add this to the Makefile's `MECH=ctrlpkt` recipe (insert between
   `generate.py` and `aiecc`).
2. **Host path:** ctrlpkt builds emit `ctrlpkt.bin` + `ctrlpkt_dma_seq.bin`
   (instead of `insts.bin`). The kernel signature is different — ctrlpkt
   DMA seq buffer goes at a specific arg slot, ctrlpkt payload at
   another. Reference: `test/npu-xrt/add_one_ctrl_packet/test.cpp:80-148`.
   Add a Path-C in `bench.cpp` that loads both bins, allocates the
   right BO at `kernel.group_id(...)`, and dispatches.

**Acceptance.**
- `make MECH=ctrlpkt DEVICE=npu2_1col TILES=1 BDS=2` compiles cleanly.
- `./bench --mechanism=ctrlpkt --metric=pure_dispatch ...` returns a
  JSON row with reasonable timings.
- §1 of REPORT.md adds a `ctrlpkt` row across the same `tiles × bds`
  grid as the other mechanisms.

**Dependencies.** Independent of #1, but cleaner to land #1 first so
the comparison context is set.

---

## 4. `--dump-state` cross-mechanism verification

**Goal.** Prove that `baseline`, `load_pdi_fw`, `load_pdi_expanded`, and
`ctrlpkt` all leave the device in the *same* register state after their
respective dispatches. Otherwise we're benchmarking different amounts
of work and the head-to-head numbers are misleading.

**Implementation sketch.**
- `bench.cpp` flag `--dump-state` that, after one dispatch, reads back
  a small fixed list of AIE registers (lock states, BD descriptor
  contents, core-program-counter snapshot) via control packets and
  emits them as JSON.
- One driver-side dry pass per mechanism, then `diff` the resulting
  JSONs.
- If they diverge, either trim the comparison or annotate REPORT.md
  with what each mechanism doesn't program.

**Acceptance.**
- A new `results/state_snapshots/` directory with one JSON per
  mechanism for a fixed cell.
- A new §6 in REPORT.md showing the diff and stating whether the
  comparison is apples-to-apples.

**Dependencies.** Plays best after #3 (so all four mechanisms can be
compared); could start before.

---

## 5. File the firmware-crash bug

**Goal.** Minimal MLIR reproducer for the `load_pdi_* × tiles ∈ {2,4} ×
bds=8` firmware crash, with a written-up bug report.

**Implementation sketch.**
- Pull the failing build's `aie.mlir` from `build/load_pdi_fw_npu2_4col_t2_r1_b8_linear/aie.mlir`.
- Strip it to the smallest case that still crashes (probably 2 cols ×
  many BDs is enough).
- Write a one-page report including: reproduction recipe, dmesg output
  (`fatal_error_exception_pc = 0x00000000`), suspected cause (BD-id
  collision between load_pdi expansion and dispatch DMAs).
- Save under `programming_examples/benchmarks/dispatch_micro/bugs/bd_load_pdi_crash.md`.

**Acceptance.**
- Bug write-up exists; reproducer build dir + run command listed.
- Linked from REPORT.md's "Failures" section.

**Dependencies.** None. Can be tackled any time as a standalone.

---

## 6. Profile baseline's batched-amortization floor (~22 µs)

**Goal.** Explain why `xrt::kernel` plateaus at 22 µs per dispatch in a
runlist while `xrt::ext::kernel` hits ~1 µs. v1 spotted this as the
biggest unexplained signal in the dataset.

**Implementation sketch.**
- Build a stripped-down single-mechanism test (baseline, t=1, b=2,
  batch=64).
- Run under `strace -c` and `perf stat -e ...`.
- Compare against the same shape with the full-ELF path
  (`load_pdi_fw` with no self-reload after #2).
- Goal is a paragraph-length writeup, not necessarily a fix.

**Acceptance.**
- A `notes/baseline_batched_amortization.md` with hypothesis,
  measurements, and conclusion.
- §3 of REPORT.md gets a new paragraph pointing at the root cause.

**Dependencies.** Easiest after #2 (need the `load_pdi_fw_nsr` baseline
to compare to). Otherwise standalone.

---

## 7. Topology axis broadening

**Goal.** Actually measure `branch` and `hop` end-to-end (generator
supports both but v1 only swept `linear`).

**Implementation sketch.**
- Add `branch` and `hop` to the `run_matrix.sh` defaults.
- Expect: `hop` adds memtile latency over `linear` but doesn't change
  scaling characteristics. `branch` likely hits shim-channel-allocation
  limits past a few cols. Document what happens at the edges.

**Acceptance.**
- §7 of REPORT.md compares latency across topologies for a fixed
  `(mech, tiles, bds)`.
- Plot `pure_dispatch_vs_tiles_topology.png` overlays three lines per
  mechanism.

**Dependencies.** None. Trivially parallelizable with everything else.

---

## 8. Re-sweep `bds` under `rows_per_col > 1`

**Goal.** Confirm/refute that the firmware-crash bug from v1 (BD=8 ×
tiles∈{2,4}) is independent of `rows_per_col`. v1 whole-array sweep
only used `bds=2`.

**Implementation sketch.**
- Sweep `mechanism × rows_per_col ∈ {2, 4} × tiles ∈ {2, 4, 8} × bds
  ∈ {2, 4, 8}`. Skip the v1-known-bad cells.
- Watch for *new* failure modes at multi-row configs with high BD
  counts.

**Acceptance.**
- A new column in §"Whole-array sweep" of REPORT.md showing the BD
  axis under multi-row layouts.
- Any new firmware crashes get added to #5's bug write-up.

**Dependencies.** Cheap to do after #5 is filed; doesn't block anything.

---

## 9. A↔B with actual workload after the PDI load (new, surfaced by #1)

**Goal.** Measure "swap to a new PDI and then run the workload it
configures." Today's `ab_toggle` measures load_pdi in isolation; the
real production cost includes the DMA work that follows.

**Implementation sketch.**
- Add a `--ab-mode={isolated,with_work}` flag to `generate.py`. The
  `with_work` variant emits the orchestrator as the `aiex.configure {
  aiex.run @inner }` pattern from `test/npu-xrt/reconfigure_loadpdi/
  aie.mlir`, so each toggle invokes the loaded PDI's runtime sequence.
- `bench.cpp` shouldn't need changes — same `ab_toggle` dispatch loop.
- Compare against §4's isolated numbers to see what fraction of cost
  is PDI selection vs subsequent DMA work.

**Acceptance.**
- §4 of REPORT.md gets a second sub-table for "with_work" mode and a
  paragraph explaining the delta vs isolated.

---

## 10. Probe whether PDI "cache" has a size limit (new)

**Goal.** Distinguish "firmware caches ≥ 2 PDIs" from "load_pdi is
unconditionally a no-op pointer swap" — both are consistent with the
v2 data. Rotate through N distinct PDIs and watch for a step change in
per-dispatch cost.

**Implementation sketch.**
- Add `--n-configs=N` to `generate.py` (default 2). Emit N device
  regions (`cfg_0`..`cfg_{N-1}`) and N corresponding `seq_to_k` ops in
  the orchestrator.
- `bench.cpp`: `--metric=multi_toggle --n-configs=N` rotates through
  all N handles in a round-robin.
- Sweep N ∈ {2, 4, 8, 16}. If there's a fixed-size cache, latency
  jumps when N exceeds it.

**Acceptance.**
- A new §5 in REPORT.md with the multi-config rotation curve.
- Either a confirmed cache size or a confident "no cache, just cheap
  pointer swap" conclusion.

---

## Stretch (not for this round, but on the radar)

- **`xrt::runlist` for the whole array.** v1 batched only tested t=1
  and t=4. See if `batch=64 × tiles=32` still hits ~1 µs/dispatch for
  the ELF path, or if memtile fan-out introduces a new floor.
- **`warm_reconfig` properly isolated.** Define a runtime sequence
  that takes a flag to skip the load_pdi op, so we can bracket only
  the reconfig submit. Currently `warm_reconfig` collapses to
  `pure_dispatch` for `load_pdi_*`.
- **Per-mechanism artifact-size table.** Compare bytes shipped to
  device for each mechanism × scale. We have the data in
  `sizes.json` already; just need to surface it in REPORT.md.

---

## Completed

- **#1 A↔B dispatch (2026-05-18).** `bench.cpp` `--metric=ab_toggle` +
  generator emits two runtime sequences in the orchestrator. Headline:
  PDI cache may not exist as framed in v1 — alternating distinct PDIs
  costs the same as the v1 "cached" path; firmware load_pdi is
  essentially free at any tile count. Documented as §4 in REPORT.md.
  Surfaced two new tasks: #9 (with-work AB) and #10 (multi-PDI cache
  probe).
