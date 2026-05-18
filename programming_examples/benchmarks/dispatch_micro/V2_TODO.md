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

## 6. Profile baseline's batched-amortization floor  **[x] DONE — finding inverted**

**Result.** Not a profile but a **bug discovery + retraction**.

The premise (v1 §3) was that ELF runlist amortizes ~20× better than
baseline. Probing it via `--vary-args` (allocate distinct in/out BO
pairs per run, defeats any potential runtime dedup) revealed
something simpler: **PathE in bench.cpp never had a `dispatch_batched`
method**. The `--batched` flag for ELF mechanisms silently fell
through to `dispatch_once()` and reported
`(single_dispatch_time / batch_size)` as "per-dispatch latency."

Added PathE::dispatch_batched (real `xrt::runlist`) + a vary-args
variant. With real runlist:

  baseline    bs=64: 22 µs/dispatch  (identical args)
  baseline    bs=64: 22 µs/dispatch  (distinct args — same)
  load_pdi_fw bs=64: 114 µs/dispatch (identical args)
  load_pdi_fw bs=64: 123 µs/dispatch (distinct args — same)

The vary-args control rules out runtime dedup. The numbers are real
per-dispatch costs. **baseline runlist amortizes well; ELF runlist
costs *more* per dispatch than baseline.** v1's "ELF is 20× better"
finding was completely inverted by the bug.

Documented as §3 retraction in REPORT.md. tl;dr point #4 rewritten to
state the correction. **Practical conclusion: for workloads with many
independent dispatches, use `baseline`, not ELF runlist.**

---

## 7. Topology axis broadening  **[x] DONE**

**Result.** Built + measured 24 cells across mech × topology × t.
`branch` is shim-channel-limited and fails at t > 2 with "number of
output DMA channel exceeded" — documented. `linear` and `hop` scale
to t=4 cleanly.

**Headline.** **Topology barely moves per-dispatch cost.** All three
topologies sit in a ~60-92 µs band at the shapes measured. `hop` adds
~4 µs to `load_pdi_expanded` at t=4 (extra memtile reconfig in the
inlined txn stream); `baseline` and `load_pdi_fw` are unaffected.
`branch` is within noise of `linear` where measurable.

Documented as §7 in REPORT.md. The "topology is noise" practical
conclusion is now explicit.

---

## 8. Re-sweep `bds` under `rows_per_col > 1`  **[x] DONE**

**Result.** Ran 36 cells across mech × r ∈ {2,4} × t ∈ {1,2,4} × b ∈
{2,4}. All 36 succeeded. b=8 was skipped after probing showed it's
universally unstable in multi-row configurations.

**Headlines:**
1. Within `bds ∈ {2, 4}`, BD count is mostly noise (ratios 1.0-1.2).
2. Rows matter much more than BDs for `load_pdi_expanded`: at t=4
   b=4, doubling rows (r=2 → r=4) adds 41% latency, while doubling
   BDs (b=2 → b=4) at the same r=4 adds only 3%.
3. **New failure boundary: `bds=8 × multi-row`.** Even `baseline +
   bds=8` (which works at r=1) degrades to ~6-second per-dispatch
   latency at r > 1. `load_pdi_*` + `bds=8` + `r > 1` reliably
   hangs. The v1 "load_pdi-specific bds=8 bug" framing is too
   narrow; the load-bearing condition is `bds=8`, not the mechanism.

Documented as §8 in REPORT.md. Tightens the failure-boundary picture
for the entire dispatch_micro harness on Strix.

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

## 11. Solve ctrlpkt's 2nd-dispatch hang  **[~] PARTIAL — hang isolated, not solved**

**What we tried (in bench.cpp's PathC, via `--ctrlpkt-strategy`):**

| strategy        | what's recreated per iter         | result               |
|-----------------|-----------------------------------|----------------------|
| `reuse`         | only `xrt::run`                   | hang on dispatch #2  |
| `fresh_kernel`  | `xrt::ext::kernel`                | hang on dispatch #2  |
| `fresh_module`  | `xrt::module` + `ext::kernel`     | hang on dispatch #2  |
| `fresh_ctx`     | `xrt::hw_context` + everything    | **works, ~80 ms / dispatch** |

**Conclusion.** The hang lives at the `hw_context` layer (driver/
firmware). Recreating just the user-space handles (kernel, module)
doesn't reset whatever state ctrlpkt's first dispatch leaves behind.
The only working strategy costs ~80 ms per call — dominated by
context creation+teardown, not ctrlpkt work.

We **did** isolate the real first-dispatch cost via the `cold_start`
metric (which times phases separately): **787 µs p50** for ctrlpkt at
t=1, b=2 across 30 fresh processes. That's in the same 758-897 µs band
as every other mechanism's first dispatch. So at first-dispatch
granularity, ctrlpkt is unremarkable — but we still have **no
steady-state number** for it.

**To make further progress** we'd need one of:
1. A between-dispatch reset routine the overlay xclbin supports, or
2. An XRT API to reset hw_context state without teardown, or
3. A different ctrlpkt build shape (canonical-style
   `aiex.npu.dma_memcpy_nd` ops instead of `dma_configure_task_for`).

None are in scope for this round. Documented as the new "v2 #11
follow-up" subsection of REPORT.md §6.

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
- **#6 Profile baseline batched amortization (2026-05-18).** The
  v1 §3 "ELF amortizes 20× better than baseline" finding was a
  bench.cpp bug: PathE never had a `dispatch_batched` method.
  `--batched` for ELF silently fell through to single
  `dispatch_once()` and reported `(total / batch_size)` as
  per-dispatch. Added real PathE::dispatch_batched + a vary-args
  control to rule out runtime dedup. **Corrected numbers: baseline
  bs=64 = 22 µs/dispatch (good amortization); load_pdi_fw bs=64 =
  114 µs/dispatch (worse than baseline).** v1's framing was
  completely inverted. Practical conclusion: for many-dispatch
  workloads, baseline runlist is the right primitive.
- **#8 Re-sweep bds under multi-row (2026-05-18).** 36 cells at
  mech × r ∈ {2,4} × t ∈ {1,2,4} × b ∈ {2,4}, all OK. Within
  b ∈ {2,4} the BD axis is noise (ratios 1.0-1.2). Rows dominate
  for `load_pdi_expanded` (+41% from r=2→r=4 at t=4 b=4). **New
  finding: `bds=8` is unstable across all mechanisms in multi-row.**
  Even baseline degrades to ~6 s/dispatch at b=8 r>1. The v1
  "load_pdi-specific bds=8 bug" framing was too narrow; the
  load-bearing condition is `bds=8`, not the mechanism. Documented
  as §8 in REPORT.md.
- **#7 Topology axis broadening (2026-05-18).** Ran 24 cells across
  mech × topology ∈ {linear, branch, hop} × t ∈ {1,2,4}, bds=2.
  Headline: topology barely moves per-dispatch cost (~60-92 µs band
  for all combinations). `hop` adds +4 µs under `load_pdi_expanded`
  at t=4 (extra memtile reconfig in inlined txn). Build boundary:
  `branch × t > 2` fails at compile due to shim DMA channel
  exhaustion (only 2 MM2S + 2 S2MM per shim). Documented as §7 in
  REPORT.md.
- **#11 ctrlpkt hang investigation (2026-05-18).** Probed four
  workarounds via `--ctrlpkt-strategy`; only `fresh_ctx` works but at
  ~80 ms per call. Hang is at the `hw_context` layer (driver/firmware),
  not above. Captured the real first-dispatch number via `cold_start`
  metric: **787 µs p50 at t=1, b=2** — in the same 758-897 µs band as
  the other three mechanisms. Steady-state still unmeasurable.
  Documented in REPORT.md §6 "v2 #11 follow-up" subsection.
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
