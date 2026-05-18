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

## 2. `--no-self-reload` generator flag  **[x] DONE (unexpected outcome)**

**Result.** Added the flag to generate.py + wired through Makefile.
Generator works correctly: with `--no-self-reload`, no `aiex.npu.load_pdi`
op is emitted at the top of the runtime sequence for `load_pdi_fw` /
`load_pdi_expanded` builds.

**Unexpected finding.** **Every NSR dispatch times out** with the same
signature as the bds=8 firmware crash:

```
xrt::run::aie_error: ERT_CMD_STATE_TIMEOUT
txn_op_idx = 0xFFFFFFFF
fatal_error_* = 0
```

This held at every tile count and BD count we tried. Reproduced 6/6
times across `mech ∈ {fw,expanded} × t ∈ {1,4,8}`.

**Interpretation.** **The `aiex.npu.load_pdi` op is load-bearing for
the full-ELF dispatch path.** Even though `xrt::hw_context(device,
elf)` loads the PDI into driver memory, and even with only one PDI in
the ELF, dispatch fails without a `load_pdi` op at the top of the
runtime sequence. The most likely reason (per the comment in
`test/npu-xrt/loadpdi/aie.mlir`): `load_pdi` is also a patch point
where XRT fixes up addresses at dispatch time. No patch point ⇒ no
address fix-up ⇒ dispatch never finds the right state ⇒ stuck BD
chain.

**Implication for the v1 framing.** The original goal of this task —
isolating cache-hit cost — turns out to be unmeasurable in the current
ELF runtime model. You cannot omit the load_pdi op and still dispatch.
The "no load_pdi" baseline simply doesn't exist for the ELF path.

The v2 §4 + §5 findings stand: rotating between distinct PDIs is the
same cost as reloading the same one, regardless of whether anything
is "cached." The op is essentially a patch + selector with no
load-time work. Trying to remove it from the sequence breaks the
dispatch entirely.

Documented in REPORT.md (new Anomalies entry) and cross-linked from
`bugs/bd_load_pdi_crash.md` since the failure mode is identical
(suggests both bugs share an underlying cause).

---

## 3. ctrlpkt end-to-end  **[x] DONE (with caveat)**

**Result.** Built end-to-end + measured. Three changes:
- `generate.py`: ctrlpkt mechanism now emits both `@main` (config) and
  `@base` (skeleton) devices alongside.
- `Makefile`: MECH=ctrlpkt recipe runs `aie-opt
  -aie-generate-column-control-overlay`, then aiecc twice
  (`--device-name=base` → xclbin, `--device-name=main` → ctrlpkt ELF).
- `bench.cpp`: new Path-C using `xrt::module` + the three-arg
  `xrt::ext::kernel(context, module, name)` plus `(opcode, 0, 0,
  bo_in, bo_out)` dispatch signature.

All four artifacts produced: `aie.xclbin` (overlay) + `aie.elf`
(ctrlpkt-encoded) + `main_ctrlpkt.bin` + `main_ctrlpkt_dma_seq.bin`.

**Caveat: ctrlpkt dispatch is single-shot.** First dispatch succeeds
(~1043 µs p50 at t=1, ~852 µs at t=4 across 30 fresh processes); the
second dispatch hangs with the same `ERT_CMD_STATE_TIMEOUT, txn_op_idx
= 0xFFFFFFFF` signature as the bds=8 bug and the `--no-self-reload`
failure mode. Probable cause: a ctrlpkt reconfig isn't idempotent —
after one dispatch the device state can't replay the same packet
sequence without an explicit reset. The canonical test
(`test/npu-xrt/ctrl_packet_reconfig_elf/test.cpp`) only runs one
dispatch per process, consistent with this being the intended mode.

**Practical takeaway** (now in REPORT.md §6): ctrlpkt is ~15× slower
than baseline per dispatch with huge variance, and is single-shot.
It's a one-shot reconfiguration mechanism, not a dispatch mechanism.
For hot loops, use baseline or load_pdi_fw.

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

## 5. File the firmware-crash bug  **[x] DONE**

**Result.** Bug write-up at `bugs/bd_load_pdi_crash.md`, with
reproducer command, minimal failing MLIR copy at
`bugs/repro_load_pdi_bds8.mlir`, symptom dump, and a boundary table.
REPORT.md "Failures" section rewritten to point at it.

**Surfaced.** v2 re-investigation widened the boundary — the v1
report said "tiles ∈ {2, 4}" but at `warmup=0 --iters=1` the crash
reproduces at **any tile count with bds ≥ 8 + a load_pdi op**.
v1's warmup masked the t=1 case. Wrote up history note in the bug
file.

**Workaround documented.** Use `bds ≤ 4` with `load_pdi_*`, or use
baseline at any BD count.

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

## 9. A↔B with actual workload after the PDI load  **[x] DONE**

**Result.** Added `--ab-mode={isolated,with_work}` to generate.py.
`with_work` mode emits each orchestrator runtime sequence as
`aiex.configure @cfg_k { aiex.run @seq(args) }`, which selects the PDI
AND inlines its full runtime sequence (including DMA) at the dispatch
site. Reuses the existing `multi_toggle` bench path. Swept mech ∈
{fw, expanded} × t ∈ {1,4,8} × N ∈ {2,4}.

**Headline.** **Swapping between distinct PDIs is free.** The
`with_work` numbers at t=8 (76 µs for fw, 126 µs for expanded) are
within 2-3 µs of v1's `pure_dispatch` numbers at the same shape (where
every call self-reloaded the same PDI). There is no per-swap penalty
on top of dispatch + DMA. For multi-tenant / model-swapping workloads,
package all needed PDIs into one ELF and swap freely.

Documented as the new "With work mode" + "Three-way comparison" +
"Practical conclusion" subsections of §5 in REPORT.md.

---

## 10. Probe whether PDI "cache" has a size limit  **[x] DONE**

**Result.** Added `--n-configs=N` to `generate.py` (1 → N distinct
PDIs in orchestrator) and `--metric=multi_toggle` to `bench.cpp`
(rotates N kernel handles, reports per-slot p50). Swept N ∈ {2,4,8}
at t=1 and t=4 for both `load_pdi_fw` and `load_pdi_expanded`.

**Headline.** **No fixed-size cache exists up to N=8.** Per-slot
latencies are within ±1 µs of each other at every N. Combined with §4,
this confirms `load_pdi` is a selector / pointer-swap operation — PDIs
are loaded into driver memory at `hw_context(device, elf)` time, and
the dispatch-time op just switches which one is active. There's no
"load" to cache.

Documented as §5 in REPORT.md. Updated the top-of-report v2 summary to
combine §4 + §5 into a single "no PDI cache in the v1 sense" finding.

**Still open.** Pushing N higher (16, 32, 64) and using larger PDIs
(whole-array configurations) would tell us where the driver-memory
wall sits. Today's tiny 1-tile PDIs don't pressure it.

---

## 11. Solve ctrlpkt's 2nd-dispatch hang (new)

**Goal.** Find ctrlpkt's *steady-state* per-dispatch cost, not just
the first-dispatch cost. v2 #3 left this open because dispatch #2
reliably hangs.

**Hypothesis to chase.** The column-control-overlay xclbin includes
reset routes (per the test/npu-xrt/ctrl_packet_reconfig_elf comment
about routes). Maybe a reset ctrlpkt sequence has to be dispatched
between functional dispatches. If so:
- Identify which control packets reset state.
- Add a "between-dispatch reset" call in bench.cpp's Path-C.
- Compare steady-state numbers against §1's other mechanisms.

**Acceptance.**
- bench can run N consecutive ctrlpkt dispatches without timing out.
- §6 gets a real steady-state number (warmup + iters > 1).

---

## 12. Per-process amortization for ctrlpkt (new, fallback if #11 hard)

**Goal.** If we can't solve the hang, at least amortize ctrlpkt's
fresh-process cost across enough samples to get a tight first-dispatch
distribution and subtract estimated process-startup overhead.

**Implementation sketch.**
- Wrap bench in a helper that times `time.time_ns()` deltas across the
  Python process boundary (in run_matrix.sh).
- Subtract baseline's first-dispatch p50 as an estimate of
  "everything except ctrlpkt-specific work."
- Report what's left.

**Acceptance.**
- A defensible "ctrlpkt costs ~X µs more than baseline's first
  dispatch" number, with confidence bounds.
- Clear caveat that this is *not* steady-state.

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
- **#10 Multi-PDI rotation (2026-05-18).** `--n-configs=N` in generator;
  `--metric=multi_toggle` in bench. Swept N ∈ {2,4,8} × t ∈ {1,4} ×
  {fw,expanded}. **No cache size limit observed.** All slots within
  ±1 µs at every N. Combined with #1 this confirms `load_pdi` is a
  selector op, not a memory load — PDIs are driver-resident after
  ELF registration. Documented as §5; v2 top-of-report summary
  rewritten to combine §4 + §5.
- **#9 AB with actual workload (2026-05-18).** `--ab-mode=with_work`
  in generator emits `aiex.configure { aiex.run @seq }` so the
  orchestrator both swaps PDIs and inlines the loaded config's full
  runtime sequence. Swept fw/expanded × t ∈ {1,4,8} × N ∈ {2,4}.
  **Headline: PDI swap is free.** with_work numbers match v1
  pure_dispatch within 2-3 µs at all shapes. The practical conclusion
  in §5 is: package all PDIs in one ELF, swap freely, pay only the
  cost of dispatch + the loaded config's DMA work.
- **#5 File firmware-crash bug (2026-05-18).** Reproducer + write-up
  at `bugs/bd_load_pdi_crash.md`. v2 widened the boundary: at
  warmup=0/iters=1 the crash reproduces at any tile count with bds≥8
  and a load_pdi op (v1 had only seen t∈{2,4} because of warmup).
  Workaround: bds≤4 with load_pdi_*, or use baseline at any BD.
- **#2 `--no-self-reload` flag (2026-05-18).** Added the flag —
  generator works. But every NSR dispatch times out with the same
  signature as the bds=8 bug, regardless of tile count. The `load_pdi`
  op turns out to be load-bearing for the full-ELF dispatch path
  (likely an XRT patch point). v1's question "what does the cache
  hit cost in isolation" is unmeasurable in the current runtime
  model: you cannot dispatch the ELF without the op present.
  Documented as Anomaly #0 in REPORT.md and as a "Related failure"
  section in the bug write-up.
- **#3 ctrlpkt end-to-end (2026-05-18).** Build pipeline (overlay
  pass + dual aiecc) + Path-C in bench.cpp end-to-end. **Caveat:
  dispatch is single-shot** — first call succeeds, second hangs with
  the now-familiar `txn_op_idx = 0xFFFFFFFF` timeout. Captured 30
  fresh-process single-shot samples at t ∈ {1,4}; first-dispatch p50
  is ~1043 µs and ~852 µs respectively (range 640-1820 µs).
  
  **Retraction (2026-05-18, same day).** My initial v2 §6 claim of
  "ctrlpkt is ~15× slower than baseline" was a category error: I
  compared ctrlpkt's *first* dispatch to baseline's *steady-state*
  dispatch. Apples-to-apples (v1 cold_start first_dispatch p50 at
  same shape: baseline 796 µs, load_pdi_fw 897 µs, load_pdi_expanded
  758 µs, ctrlpkt 1043 µs), ctrlpkt's first dispatch is in the same
  band as every other mechanism. **We have no measurement of
  ctrlpkt's steady-state cost** because dispatch #2 hangs. §6 in
  REPORT.md was rewritten to retract the misleading number and
  state the limitation explicitly. Surfaced tasks #11 (solve the
  hang) and #12 (per-process amortization fallback).
