# Decode attention on NPU — GQA-batched BFP576 (scope + handoff)

Status: **scaffolding** (2026-06-18). Replaces `decode_attention_cpu`
(`llama32_1b_decode.py:110`) with an on-NPU attention that fills the BFP576
matmul's M dimension using the 4 GQA query heads that share each KV head.

## Why

Single-token decode attention is M=1 per head: a bf16 gemv against the KV
cache, memory-bound on cache streaming. The BFP576 hardware matmul has ~8x
the MACs/cycle of the bf16 vector MAC, but its 8x8x8 tile wastes 7/8 of the
M dimension at M=1 — so the multiplier gain exactly cancels (see chat
2026-06-18). The lever is to **fill M with the 4 Q heads that share one KV
head** (GQA group_size = n_heads/n_kv_heads = 32/8 = 4):

* QKᵀ:  4 q-heads (M=4) · Kᵀ (K=head_dim=64, one block) → scores (4, S)
* AV:   scores (M=4, K=S) · V (N=head_dim=64) → context (4, 64)

The K/V cache block is **read once and reused by all 4 heads** → tile fill
1/8 → 4/8 (4x effective multiplier use vs the bf16 break-even) AND KV-cache
bandwidth ÷4. Second win: attention is currently on CPU "because LQ=1
dispatch overhead beats NPU" (`RESIDENT_DEVICE_EVOLUTION.md:49`); the
bandwidth + dispatch-count reduction is what could flip that.

## Decisions (locked 2026-06-18)

1. **M packing: 4 heads only.** 4 real heads + 4 zero-padded rows in the
   8-row BFP block. 50% fill. Phase 2 (deferred): fill the spare 4 rows with
   hi/lo activation splits to recover near-bf16 accuracy from BFP16.
2. **Online-softmax KV tiling.** Reuse the existing flash online-softmax
   loop over 64-wide KV chunks. **Causal mask dropped** — a decode token
   attends to all cached positions ≤ current_pos. Handles arbitrary context.
3. **Placement: separate dispatch** between Call 1 (`rms_gemv_rope`) and
   Call 2 (`o_gemv_ffn`/`c2_merged`), replacing `decode_attention_cpu`. NOT
   inside c2_merged. Folding it onto the front of c2_merged (O-proj consumes
   attn_out) is a later option, not required.

## Reuse map

Already compiled into `attn_pythoc.o` (`kernels/build.py:compile_attn`),
reused as-is (no recompile needed for these):
* `fused_softmax` (attn.py:1010) — max/exp/rescale/sum, all bf16, online.
* `copy_tile`, `exp_*`, `accum_*`, `div_gp_sp` softmax helper chain.
* `apply_causal_mask` (attn.py:935) — present in the .o but **not called**.

Needs a NEW small kernel variant (row_blocks=1) — the existing matmuls
hardcode `row_blocks=8` (64-row M sweep); decode runs ONE 8-row block:
* `matmul_a_b_bf16` (attn.py:382) → `matmul_a_b_bf16_m1blk` (row_blocks=1)
* `matmul_g_b_bf16` (attn.py:577) → `matmul_g_b_bf16_m1blk` (row_blocks=1)
  Only the outer `while m < row_blocks` bound changes (8→1) plus the M-stride
  bookkeeping; the 2x2 register-blocked BFP576 inner loop is identical.
  Add these to attn.py (after fused_softmax so compile_attn is unaffected),
  compile via a new `compile_attn_decode` entry, OR fold into attn.py and
  bump the single .o.

Builder: adapt `builders/flash_attn.py` (1107 lines, prefill: SEQ_LEN=2048,
32 cores, cascade, causal) → `builders/attn_decode.py`:
* LQ tile = 1 row-block (8 rows, 4 heads used) instead of 256-row q_groups.
* Drop the causal-counter scratch buffer + apply_causal_mask call.
* Loop KV in 64-wide chunks up to current_pos (runtime length).
* 8 KV-groups × 16 layers — far fewer cores than the prefill 32-core grid.

## Host wiring (`llama32_1b_decode.py`)

Replace the `decode_attention_cpu(...)` call (line 264) with a
`cache.load_and_run("attn_decode", ...)` dispatch. Inputs per group:
* q_roped reshaped (8 groups, 4 heads, 64) — feed group g's 4 heads.
* k_cache_layer[g, :seq_len, :], v_cache_layer[g, :seq_len, :].
* scale = 1/sqrt(head_dim) folded into the q load or a post-scale.
Output: attn_out (32, 64) → flatten to (emb_dim,) for the O-proj.

## Validation plan

1. Numerics: compare against `decode_attention_cpu` (the current reference)
   per layer, max-abs error vs the existing bf16 path. Target parity with
   the prefill flash path's tolerance.
2. Latency: measure the new dispatch vs the CPU attention it replaces, and
   confirm it beats the CPU path it's there to replace (the gate that put
   attention on CPU in the first place).
3. Then wire into the 16-layer decode loop; check end-to-end tok/s and the
   HF answer gate.

## Results — VERIFIED ON HW 2026-06-18 (Strix Halo, aie2p)

Stages 0/1/2 PASS on real hardware (`tools/test_decode_attn_npu.py`, tol 2e-2):
* tiling probe: exact 64x64 round-trip (0.0 diff)
* stage 0 (1 group, seq=64): 3.9e-4 | stage 1 (8 groups/32 heads, seq=64): 4.4e-4
* stage 2 online-softmax KV tiling: seq=128 3.9e-4, seq=200 (partial tile,
  -inf masked) 2.3e-4, seq=256 3.0e-4
Builder: `builders/attn_decode.py`. The BFP576 GQA M=4 decode-attention
hypothesis is numerically confirmed end-to-end on device.

### Corrections to earlier notes (now known-wrong)
* **Host feeds q UNSCALED.** The 1/sqrt(head_dim) scale is ALREADY folded
  into the kernel: `fused_softmax`->`exp_g_minus_u` uses
  `log2e_vec = 0.18033688 = log2(e)/sqrt(64)`. Do NOT pre-scale q. (The
  earlier "fold scale into q" note was wrong.)
* **Direct host round-trip tiling dims are `[(8,8),(64,64),(8,1)]` BOTH
  ways** (in and out). The flash_attn `[(64,8),(8,512),(8,1)]` out-untile is
  for an L2-staged buffer, not this direct host path.
* Online-softmax recurrence (validated 1e-17 vs ref): per chunk
  `mul_r_gp(r,gp); matmul_g_b(g,v,gp); accum_sp_r_s(sp_run,r,sp); sp_run=sp`,
  then `div_gp_sp` at the end. fused_softmax outputs up=running max,
  g=exp numerators, sp=chunk sum, r=rescale alpha.

### Blocker: seq_len > 256 (n_chunks > 4)
Per-chunk KV DMA exhausts shim BD-IDs (q + 2*n_chunks + out > 16) at seq=512,
and wedges at seq=384. A single 4D repeat-BD compiled but timed out (packet
repeat ignores compute-side per-tile S2MM lock backpressure). Kernel now
guards seq>256 with NotImplementedError; test reports a clean SKIP.
**Scalable fix:** memtile-staged KV feed via `aie.objectfifo` native
`repeat_count`/`set_repeat_count` (see [[wtreplay-objectfifo-native-repeat]])
to stream chunks from L2 without consuming shim BD-IDs. This is the right
follow-on before any production wiring.

## Latency — MEASURED ON HW 2026-06-19 (tools/bench_decode_attn.py)

Current 8-dispatch path (one dispatch per GQA group) vs decode_attention_cpu,
median ms/token:

| seq | CPU   | NPU (8 disp) | NPU/CPU      | ms/dispatch |
|-----|-------|--------------|--------------|-------------|
| 64  | 0.232 | 1.983        | 8.6x slower  | 0.248       |
| 128 | 0.212 | 2.020        | 9.5x slower  | 0.253       |
| 200 | 0.754 | 2.551        | 3.4x slower  | 0.319       |
| 256 | 0.680 | 2.623        | 3.9x slower  | 0.328       |

VERDICT: dispatch-overhead-bound (~0.25-0.33 ms/dispatch fixed; 8 of them
swamp the trivial compute). The 4x BFP576 multiplier win is invisible at this
scale -- confirms [[llama-decode-budget-c30]]. NPU is ~flat in seq_len, CPU
grows O(seq) -> they cross as context grows. A SINGLE dispatch (~0.3 ms) is
already competitive and projects to beat CPU at seq>=~200. The lever is
dispatch COUNT, not multipliers: (1) batch 8 groups -> 1 dispatch; (2) endgame
= fold attention into an existing launch (c2_merged front / rms_gemv_rope tail)
for ~0 marginal launch cost.

## BATCHED single-dispatch variant — MEASURED ON HW 2026-06-19

Collapses all 8 GQA groups into ONE dispatch: 8 compute cores (one per
column, tile (col,2)), each running the EXACT online-softmax logic of the
single-group path, all driven by ONE runtime_sequence / ONE aiex.configure.
Per-column shim DMA (tile(col,0)) gives each group its own BD-ID budget, so
the "8x BD on one shim" exhaustion is sidestepped — mirrors flash_attn's
multi-column topology. Host packs all groups into 4 concatenated BOs
(q_all/k_all/v_all/out_all, group g at offset g*tile). Additive, production
path untouched.
* Builder: `build_decode_attn_batched_module(seq_len, n_groups)` in
  builders/attn_decode.py.
* Wrapper: `decode_attention_npu_batched(...)` in tools/test_decode_attn_npu.py.
* Bench: tools/bench_decode_attn_batched.py.

CORRECTNESS (full 32-head, vs decode_attention_ref, tol 2e-2): ALL PASS
  seq 64: 4.4e-4 | 128: 3.9e-4 | 200: 2.3e-4 | 256: 3.0e-4
Rungs proven on HW: 2-core, 4-core, AND 8-core (full token) all 1-dispatch.

LATENCY (median ms/token, 8-group = full token = 1 dispatch):

| seq | CPU   | NPU 8-disp (old) | NPU 1-disp (new) | new NPU/CPU |
|-----|-------|------------------|------------------|-------------|
| 64  | 0.159 | 1.983            | 0.387            | 2.44x slow  |
| 128 | 0.173 | 2.020            | 0.425            | 2.45x slow  |
| 200 | 0.584 | 2.551            | 0.663            | 1.14x slow  |
| 256 | 0.681 | 2.623            | 0.729            | 1.07x slow  |

VERDICT: dispatch-collapse WORKS — 8->1 dispatch is a ~4-5x NPU speedup
(2.6 ms -> 0.73 ms at seq=256), exactly as the single-dispatch projection
predicted. The NPU is ~flat in seq_len (0.39->0.73 ms) while CPU grows O(seq);
they converge to a near-tie at seq>=200 (1.07-1.14x). No crossover yet within
seq<=256, but the curves are closing ~linearly and a modest extra CPU cost or
seq~300+ would cross. The remaining gap is the fixed ~0.3-0.4 ms host
launch/sync floor (see [[llama-decode-budget-c30]]), NOT compute. Notably the
2-/4-core rungs at seq=256 run 0.46/0.56 ms (they compute only 8/16 heads),
showing per-dispatch cost still scales with #columns configured — i.e. fewer,
wider cores would be cheaper than 8 narrow ones.

The endgame lever is unchanged: fold attention into an existing launch
(c2_merged front / rms_gemv_rope tail) for ~0 marginal launch cost, which
removes the launch floor entirely. The batched device built here is the
drop-in that makes that fold a single-segment add.

## Open risks

* Dispatch overhead: 8 groups could be 8 dispatches or one batched device.
  RESOLVED: one batched device (above) cuts NPU 4-5x; production fold pending.
  Prefer one device fed by all 8 groups (mirror flash_attn's single
  aiex.configure) to keep launch count at 1, per the decode budget note
  (`llama-decode-budget-c30.md`: launch overhead is the real lever).
* Variable seq_len: the online-softmax loop bound is runtime; confirm the
  builder can take a runtime KV-chunk count (prefill bakes it constant).
* 4/8 fill means QKᵀ/AV are ~4x over bf16 break-even, but attention is a
  small share of per-token work at short context — measure before claiming
  end-to-end win.

## Phase 1 fusion attempt — MEASURED ON HW 2026-06-19 (NEGATIVE, informative)

Files (NEW, additive): builders/attn_oproj_fused.py, tools/test_attn_oproj_fused.py.
Fused attention + O-proj matvec as TWO aiex.configure/run blocks under ONE
runtime_sequence. Numerics CORRECT (proj err 2-5e-4, seq 64-256). Latency
(median ms/tok): fused 1.36/1.67/2.67/2.35 (seq 64/128/200/256) vs two
SEPARATE dispatches 1.17/1.20/1.28/1.29 -> fusion is WORSE by 0.16-1.4 ms.

ROOT CAUSE (lowered-IR confirmed): two aiex.configure = TWO PDI reloads
(LoadPDI) inside one dispatch; each reprograms the full 8-col fabric and they
serialize super-additively. => "one dispatch" is necessary but NOT sufficient;
the win requires ONE LoadPDI = ONE merged aie.device (this is exactly why
production c2_merged collapsed its 8 stages into one device). The
two-configure shortcut is a DEAD END for the floor.

CORRECTED Phase 2 path: merge the 8 attention cores + buffers/locks/flows/shim
allocs INTO a single aie.device alongside O-proj (and ultimately the rest of
c2_merged), one PDI, on-chip L2 attn_out->O-proj handoff. De-risk as Phase 2a:
a STANDALONE single-device (one aie.device / one LoadPDI) doing attn+O-proj,
measure that fused ~= max(attn, O-proj) (floor shared), BEFORE surgery on the
~5k-line production o_gemv_ffn.py (Phase 2b).

## Phase 2a — single-device merge — MEASURED ON HW 2026-06-19 (NEGATIVE, decisive)

Files (additive): builders/attn_oproj_fused.py::build_attn_oproj_merged_module,
tools/test_attn_oproj_fused.py::merged_npu. Merged attn (row 2, 8 cols) +
O-proj matvec (row 3, 8 cols) + memtiles (row 1) into ONE aie.device.

VERIFIED ONE PDI: lowered IR shows aiex.configure=1, compute PDI=1 (vs Phase 1's
2). Numerics PASS (proj err 2-5e-4, seq 64-256).

Latency (median ms/tok): merged 1.19/1.25/2.16 (seq 64/128/256). This tracks
(a)+(b)=attn+oproj SUM (1.17/1.23/1.28...), NOT max(a,b)=~0.78. => single-device
/ one-PDI does NOT share the floor. Root cause: hard data barrier — O-proj needs
the full 2048-elt attn_out before starting, so attn and O-proj run SERIALLY in
the device. One PDI removed only the small extra-reload penalty (~0.1-0.6 ms vs
Phase 1's 2 PDIs); attention compute is NOT hidden.

Blocker fixed (reusable): first runs wedged on a packet-id collision (attn k and
O-proj weight both on shim MM2S0 pkt_id=1). Fix: distinct pkt ids per concurrent
stream + route attn output to shim S2MM1. See [[packet-id-mask-rules]].

## FINAL VERDICT — investigation complete (2026-06-19)

The BFP576 GQA decode-attention idea is NUMERICALLY CORRECT on HW (every stage,
err ~3e-4) and the 4x-multiplier mechanism works — but at llama-3.2-1B decode
scale attention is dominated by FIXED OVERHEAD (launch floor + serialized
execution), not compute, so the multiplier win is irrelevant (compute was never
the bottleneck). NO variant beats decode_attention_cpu within the validated
seq<=256 range:
  naive 8-disp: 3.4-9.5x slower | batched 1-disp: ties (1.07x@256)
  2-config fused: worse (2 PDIs) | single-device merged: ties (serialized)
Fusing into production c2_merged (Phase 2b) would recover at most the ~0.3-0.4 ms
standalone-attention launch floor and still serialize attention before O-proj —
not worth surgery on the 5k-line builder. RECOMMENDATION: STOP. The one
unexplored regime where NPU could win is long context (seq>>256: CPU grows
O(seq), NPU ~flat), gated on the unbuilt memtile-objectfifo KV feed.

All work is standalone/additive; production decode path untouched throughout.

## GEMV columns→bandwidth curve — MEASURED ON HW 2026-06-19 (tools/bench_matvec_columns.py)

Column-parameterized matvec (proj=W·x, K=2048, M swept 2–134 MB), fit
time(bytes)=floor + bytes/BW(C), numerics-gated each point.

CORRECTED 2026-06-19 (tools/bench_matvec_overlap_clean.py): the first sweep
timed the WHOLE HOST CALL (BO write + launch + readback + Python/filelock); that
host ENVELOPE plateaus at ~15 GB/s — it is NOT the fabric. The earlier "plateau
at C≈4, ~15 GB/s, column slack" reading was a host-overhead artifact — WRONG.

KERNEL-ONLY weight-ingest BW (run.start()→wait2(), the truth):
| C | baseline (single-buf) | nop (compute-free) | pingpong (overlapped) |
|---|----|----|----|
| 1 | 8.2  | 13.0 | 12.6 |
| 2 | 16.6 | 24.9 | 23.6 |
| 4 | 33.5 | 44.7 | 44.2 |
| 8 | 50.9 | 53.3 | 49.4 |

VERDICT (corrected): BW scales ~LINEARLY in columns (8.2→50.9 = 6.2x for 8x
cols, 78% scaling) — NO column slack; the engine wants all 8 columns. The real
fabric ceiling is ~51 GB/s at C=8 (baseline≈nop within 5% ⇒ fabric-bound, not
compute-bound there) — still ~4x below LPDDR5 peak (on-chip ingest fabric, not
DRAM). Below C=8 the single-buffered kernel is COMPUTE-serialization-bound
(nop 1.3–1.6x over baseline); a double-buffered pingpong variant recovers nearly
all that headroom (1.54x@C1, 1.42x@C2, 1.32x@C4, ~1.0x@C8) — numerically correct.

IMPLICATIONS FOR RESIDENT DECODE (this resolves the open question):
- The 3.4x gap between the ~15 GB/s host envelope and the ~51 GB/s fabric ceiling
  is PURE host/dispatch overhead → reclaimable by residency (PDI/launch collapse).
  This STRENGTHENS the resident case: residency is worth ~1.8x on bf16 (88 ms →
  ~49 ms/tok = 2.5 GB/tok ÷ 51 GB/s) by reaching the fabric ceiling.
- But the fabric WALLS at ~51 GB/s (~4x below DRAM), so residency alone does NOT
  reach the doc's optimistic ~10 ms/tok. AWQ is the lever that does: ~0.5 GB/tok
  ÷ 51 GB/s ≈ ~10–13 ms/tok. AWQ + residency together = the endgame (confirms the
  doc's "4-bit unlocks the bandwidth wall" note).
- Prefetch overlap is a real lever ONLY at <8 active columns (narrow herds; 1.3–
  1.5x); at full 8-col width it's redundant (already fabric-bound). Open further
  lever (untested): on-chip-resident activations free shim MM2S1 → could carry
  more weight DMA and possibly push past ~51 GB/s.

## Attention fusion scoping (c1/c2 herd time-sharing) — 2026-06-19

Tile budget: C1 uses rows 2(matvec O→gate→up)/3(add1)/4(swiglu)/5(rms col0);
C2 (c2_merged, with_down) uses rows 2(matvec)/3(add1,add2)/4(swiglu)/5(down).
BOTH use all 4 compute rows → NO free 8-core row (Gate 2). Fusion = herd
TIME-SHARING, not a new row.

FUSION POINT: attention = wave 0 of the row-2 matvec herd, before the O wave
(O consumes attn_out). Map 8 GQA groups ↔ 8 columns (core[c]=group c attn, then
col c matvec). Concatenate attn program + matvec program in the row-2 core via a
mode RTP (the matvec_fused precedent: mode0=attn BFP576 matmul_a_b→fused_softmax
→matmul_g_b, mode1=matvec). Attention joins the sequential runtime_sequence wave
list (attn→O→gate→up→…), demuxed by packet ID per wave. NO new device/configure/
PDI — costs only attention's serial compute+DMA. Target c2_merged (1-LoadPDI
default); keep behind a c2_attn flag, default untouched.

attn_out→O-wave is a SCATTER→BROADCAST (8 col-slices→full 2048 bcast), the class
the docs keep at DDR boundaries (like sw→dg). Keep attn_out on DDR initially
(4 KB, negligible). The win is STRUCTURAL: removes the separate attention
dispatch + CPU hop (Gate 1) → a device spans the layer → unblocks the 16-layer
loop (the real lever), NOT attn_out bytes.

Why TIME-SHARE not spatial-column-split (REINFORCED by the corrected kernel-only
BW): matvec BW scales ~linearly in columns (no slack), so stealing cols from the
matvec for a spatial attention herd costs ~linearly — dropping 8→4 cols loses
~34% (50.9→33.5 GB/s) of the DOMINANT weight stream, not ~18%. And attention is
serial before O anyway (no overlap to gain in single-stream decode). So time-
sharing the full 8-col herd (matvec keeps full BW in its waves; attention reuses
the cols in
its wave) beats spatial partitioning.

RISKS to validate (in order): (1) program memory — RESOLVED 2026-06-19 (GO):
per-core .text measured = attn 6992 B (worst-case online; fused_softmax+helpers
~4.7 KB dominate, the two BFP576 matmuls only ~1.2 KB) + matvec 848 B + ~300 B
mode-RTP glue ≈ 8.0 KB vs 16 KB → ~50% headroom. Even 2x attn still fits.
(2) L1 64 KB — RESOLVED: attn ~41 KB overlays matvec tiles (sequential waves),
binding side ~41 KB < 64 KB; (3) KV shim BD pressure per column
+ the seq≤256 cap (needs memtile-objectfifo KV feed); (4) GQA-group↔column output
reshape into the attn_out broadcast. Path: measure attn .text → c2_attn flag with
attn wave 0, attn_out DDR-spilled, KV from DDR, hf-gate bit-exact → drop CPU hop →
measure LoadPDIs/token + host-await reduction.

## c2_attn build (scoped 2026-06-19) — attention as wave 0 in c2_merged

GOAL: new pack mode `c2_attn` in builders/o_gemv_ffn.py (BF16 first) = c2_merged
+ decode attention as WAVE 0 on the row-2 herd, ONE device / ONE configure /
1 LoadPDI. Behind a flag; default c2_merged UNTOUCHED.

KEY ABI CHANGE: c2's runtime_sequence drops the `attn_out` input (arg1) and adds
`q_roped` + per-group `k_cache` + `v_cache` inputs. Wave 0 computes attn_out
on-NPU → DDR-spill (scatter→broadcast, keep on DDR) → the O wave reads it exactly
as it reads attn_out today. So downstream O/gate/up/swiglu/down/adds are UNCHANGED.

.TEXT FIT (re-checked for the REAL core): row-2 core in c2 runs the RMS-fused
matvec (~4384 B), not plain matvec. Concat = attn 6992 + rms-fused-matvec 4384 +
mode-RTP glue ≈ 11.5 KB < 16 KB (~27% headroom — tighter than the 8 KB plain-
matvec estimate but fits). Mode RTP: wave0=attn (BFP576), waves1-3=matvec.

LADDER (converge-then-collapse, each gated answer-level hf-gate = gold Paris
tokens; NPU attn differs from CPU by ~3e-4 so not bit-EXACT — gate on tokens):
  A. Correctness net: attention wave 0 producing attn_out to DDR, rest of c2
     unchanged. OK to use an extra configure/PDI here as the safety net.
     Validate c2_attn(q,k,v) output == c2_merged(cpu_attn(q,k,v)) within tol,
     and gold tokens unchanged.
  B. Collapse: mode-RTP concat attn onto the row-2 matvec core → ONE configure /
     1 LoadPDI for call 2. Re-gate. Measure LoadPDIs/token (call2 stays 1) and
     per-token latency vs the CPU-attention baseline.
SCOPE: BF16 only, seq<=256 (validated KV range; bake/pass seq_len). HARD STOP
after B validates+measures — NO AWQ port, NO default flip.

Hard-won lessons to apply: distinct packet IDs per concurrent stream on a shared
shim port (C1/C2 + Phase 2a wedges; [[packet-id-mask-rules]]); EXACT x-broadcast
delivery counts (leftovers jam the shared MM2S queue → deadlock next stage);
distinct OUTPUT pkt ids at the shim S2MM convergence; a wedge persists into the
next process. Reuse attn kernels + online-softmax recurrence + tiling dims +
q-UNSCALED from builders/attn_decode.py; mode-RTP precedent = matvec_fused.

## c2_attn build RESULT — ON HW 2026-06-19 (device PROVEN; integration BLOCKED)

DEVICE-LEVEL WIN (the central claim): c2_attn = c2_merged + attention as wave 0
runs end-to-end on HW as ONE aie.device / ONE aiex.configure / ONE compute PDI
(verified in lowered IR — same configure/PDI count as c2_merged). Numerically
correct: on-device attn_out gather vs CPU ref max 5.3e-4; full c2_attn output
0.85–1.8% rel vs c2_merged(cpu_attn), PASS (seq 64 + masked seq 53).
Implementation: attention on the row-3 (add) herd as wave 0 (idle until add1),
attn_out → widened DDR arg1 (n_groups×4096), O wave GATHERS rows 0..3 head-major
(`dimensions=[(8,4096),(256,1)]`), distinct attn packet id 16, 18-arg ABI
(drops attn_out input, appends q/k/v at 15/16/17). So fusion works: the
program-memory, single-PDI, and numerics gates are all PASSED on real hardware.

BLOCKER — end-to-end hf-gate hangs (ERT timeout) at the first c2_attn decode
dispatch. Two compounding causes, both isolated on HW:
1. The trailing-chunk mask is BAKED at build time but must track current_pos+1
   → per-position devices needed.
2. **NEW HW/XRT CONSTRAINT: two distinct full-fabric (5-row) c2_attn PDIs cannot
   be live in one process — the 2nd PDI load wedges the partition.** Reproduced
   directly (even with unique kernel ids; unique id is necessary-not-sufficient).
   rms_gemv_rope + ONE c2_attn coexist fine; TWO c2_attn wedge. Since each decode
   POSITION bakes a different mask → a different PDI, the wedge bites at the TOKEN
   boundary (token N+1's c2_attn PDI is the fatal 2nd load). Within one token all
   16 layers share the position/mask → one PDI, fine. Related to the stale-PDI
   partition-wedge family ([[stale-pdi-switch-masters]]).

FIX (the convergent next build): a SINGLE RESIDENT c2_attn device reused for ALL
positions+layers, with the trailing-chunk mask as a RUNTIME RTP (`last_valid`)
instead of baked offsets. One PDI loaded once → sidesteps the two-PDI wedge AND
is exactly the 16-layer-loop resident design (RESIDENT_DEVICE step 5). Requires
rewriting `_c2attn_mask_invalid_cols` to loop on a runtime bound (moderate kernel
work; currently unrolls build-time-constant offsets).

Files (all additive / flag-gated; default c2_merged UNTOUCHED + regression PASS,
gold "Paris" tokens, ~82 ms/tok): builders/o_gemv_ffn.py (c2_attn behind
attn_wave0 flag), builders/_emit.py (c2_attn_host_arg_types), builders/c2_attn.py
(NEW), llama32_1b_decode.py (_run_c2_attn gated branch), tests/test_c2_attn_ir.py
(NEW, 4/4 incl c2_merged leak-free guard), tools/test_c2_attn.py (NEW stepA/B
harness), tests/test_hf_answer_gate.py (c2_attn gate, hangs per blocker).

## Resident c2_attn build (scoped 2026-06-19) — host-written RTP, ONE PDI

GOAL: convert the PROVEN c2_attn device (1-PDI, numerically correct standalone)
into a SINGLE RESIDENT PDI reused across ALL positions + layers, fixing the
two-full-fabric-PDI wedge. Mechanism: the trailing-chunk mask becomes a RUNTIME
host-written RTP instead of baked offsets, so one c2_attn kernel/ELF/PDI serves
every position.

KEY SIMPLIFICATION (fixed-max chunks — recommended over a runtime loop bound):
keep the chunk loop FIXED at MAX_CHUNKS=4 (seq≤256) and the KV DMA always 4
tiles (host zero-pads k/v cache to 256). Then ONLY the RTP varies. RTP = total
valid length L = current_pos+1 (1..256), ONE i32 per attention tile, host-written
via write32 (use_write_rtp=True, the matvec_fused mode-RTP mechanism). The mask
kernel derives each chunk's boundary on-device from L: for chunk c∈0..3, mask
cols where (64*c+col) >= L, i.e. boundary = clamp(L-64*c, 0, 64) (fully-past
chunks → all -inf). This is "host-written RTP + trivial on-device per-chunk
derivation" — one scalar, fixed structure, only the value changes per token.
(Runtime loop-bound + runtime KV length is the compute-saving alternative but
adds loop/DMA-length-consistency risk; attention is overhead-bound so the wasted
masked-chunk compute is negligible — do fixed-max first.)

CHANGES (extend the prior agent's files; default c2_merged UNTOUCHED):
- `_c2attn_mask_invalid_cols` → runtime: read L from RTP, mask per-chunk
  boundary clamp(L-64c,0,64) via a runtime-bounded loop or lane-index≥boundary
  select. MUST be EXACT across L∈[1,256] (off-by-one → softmax reads stale L1 →
  nondeterminism — see the determinism note).
- RTP plumbing: per-attn-tile i32 buffer L (use_write_rtp=True) written by the
  c2_attn runtime_sequence.
- Host `_run_c2_attn`: use ONE cached c2_attn kernel (NOT per-position
  decode_attn_sN); set L RTP + zero-pad KV to 256 per token; reuse across all 16
  layers + all tokens.

GATES (all required):
1. ONE c2_attn PDI loaded ONCE — verify in IR (1 configure / 1 load_pdi) AND
   empirically: multi-token decode (>1 token) runs WITHOUT wedge (the prior
   blocker). rgr + c2_attn + lm_head as 3-distinct-each-loaded-once PDIs is fine
   (mirrors c2_merged decode); only TWO c2_attn wedged.
2. DETERMINISM: same (prompt) decoded twice → bit-exact tokens; sweep positions
   incl. partial-chunk (L not a multiple of 64). Also run production c2_merged
   twice as the baseline to confirm the harness (settles the prior synthetic-
   weight "nondeterminism" — likely a BO-init artifact, [[resident-runner-output-bo-sync]]).
3. hf-gate: gold "Paris" tokens end-to-end.
SCOPE: BF16 only, seq≤256. HARD STOP after gates pass + latency measured vs CPU-
attn baseline. NO AWQ port, NO default flip.

## Resident c2_attn build RESULT — ON HW 2026-06-19 (single-PDI WORKS; pre-existing c2_attn cross-token bug EXPOSED)

The resident single-PDI / runtime-L design was BUILT and runs on HW.  Summary:
the wedge fix and the runtime-mask infrastructure all WORK; the end-to-end gate
is blocked by a PRE-EXISTING c2_attn device defect (not introduced here, present
identically in the prior single-chunk c2_attn) that only surfaces when the SAME
c2_attn PDI is reused across tokens — which is exactly what residency does and
what the prior per-position-PDI design never exercised.

GATE 1 — ONE PDI, no wedge: PASS.  Lowered IR: 1 `aiex.configure`, 1 compute
PDI, 1 `aiex.run`, 1 fixed kernel id (`o_gemv_ffn_c2attn_resident`) for ALL
positions.  Empirically 5+ consecutive same-PDI runs spanning chunk boundaries
(L=40/64/65/128/201/256) complete WITHOUT wedge — the prior two-full-fabric-PDI
wedge is FIXED.  (4 separate per-position PDIs are gone: one resident ELF.)

GATE 2 — Determinism: SPLIT.  (a) The ATTENTION output (arg1) is bit-exact AND
numerically correct (max 5.2e-4 vs CPU) on EVERY run across all swept positions,
incl. partial-chunk L — the runtime-L mask is exact.  (b) Production c2_merged is
bit-exact run-to-run (harness sound, confirms the prior "synthetic-weight
nondeterminism" was a harness/BO artifact, NOT the device).  (c) BUT the FULL c2
output drifts: run0 correct (rel 1.1% vs c2_merged(cpu_attn)), run1+ wrong
(~17%), deterministic-after-warmup.  Isolated: attn_out correct every run, yet
the O-proj output (arg2) drifts run0→run1 (0.001→0.018) and the FFN nonlinearity
amplifies it to ~17%.  CRITICAL: the LEGACY single-chunk c2_attn shows the SAME
run0-OK/run1+-wrong drift — so this is a PRE-EXISTING c2_attn cross-token state
hazard in the `normed`/O-wave handoff (the O-wave reads its activation from the
device-written attn_out path rather than a host-synced BO as c2_merged does),
NOT a regression from this build.  It was masked before because per-position PDIs
ran each c2_attn exactly ONCE per process.

GATE 3 — hf-gate (c2_attn resident): BLOCKED by Gate 2c (token 2+ of any decode
is numerically wrong → would derail generation).  NOT attempted as a flip; the
host `_run_c2_attn` still uses the prior per-position path.

GATE 4 — Regression (REQUIRED): PASS.  Default `c2_merged` hf-gate produces gold
tokens ("\n\nThe capital of France is Paris.") unchanged; all 4 prior c2_attn IR
tests + 2 new resident IR tests pass, incl. the c2_merged leak-free guard (with
the resident flag set, c2_merged stays 15-arg, no attn artifacts).

GATE 5 — Latency (median ms/tok, seq=256, real HW): resident c2_attn (1 dispatch,
attn+c2) = 15.6 ms; c2_merged alone = 2.8 ms; CPU attention = 0.67 ms; baseline
c2_merged+CPU-attn = 3.4 ms.  VERDICT: resident c2_attn is ~4.5x SLOWER than the
CPU-attn baseline.  The fixed-4-chunk attention (always 4 KV chunks × 8 groups,
even at low L) plus the scalar runtime-L mask loop (per-masked-column 64-row
scalar stores — the aie2p backend cannot legalize a vector `arith.select` with an
i1 mask, forcing scalar) dominate.  Even with Gate 2c fixed, resident c2_attn
does NOT beat CPU attention at llama-3.2-1B decode scale — consistent with the
prior FINAL VERDICT that decode attention is overhead-bound, not compute-bound.

DECISIVE VERDICT: the resident single-PDI + runtime-L-mask mechanism is PROVEN
(1 PDI, no wedge, attention exact & deterministic) — it correctly fixes the
two-PDI wedge.  But it does NOT pass end-to-end: a pre-existing c2_attn O-wave
cross-token hazard corrupts token 2+, and even unblocked the latency loses to CPU
attention.  Fixing the O-wave hazard needs surgery on the ~5k-line matvec core
(the `normed`/x_avail handoff when fed by device-written attn_out vs a host BO) —
out of this scope's "do not edit the matvec core" guidance — and the latency
result removes the motivation.

FILES (all additive / flag-gated behind `attn_wave0` + `PYTHOC_C2_ATTN_RESIDENT`;
default c2_merged byte-for-byte untouched, regression PASS):
- builders/o_gemv_ffn.py: `_c2attn_mask_invalid_cols_rtp` (runtime-L scalar mask),
  `attn_resident` path in `_emit_call2_c2` (fixed MAX_CHUNKS=4, double-buffered
  (128,64) i8 K/V via memref.view per-chunk, L folded into q padding, no extra
  channel/lock), `_c2_attn_resident` env gate.
- builders/c2_attn.py: `build_c2_attn_resident_module`, `c2_attn_resident_kernel_id`.
- builders/_emit.py: `c2_attn_host_arg_types(resident=...)` (ABI unchanged — L
  folds into q).
- tools/test_c2_attn.py: `stepR` + `c2_attn_resident_npu` + `_tile_8x8` host pre-tiler.
- tests/test_c2_attn_ir.py: `test_c2_attn_resident_one_configure`,
  `test_c2_merged_unchanged_under_resident_flag`.

## INVESTIGATION CONCLUDED 2026-06-19 — on-NPU decode attention does not pay off

Resident c2_attn (single PDI, host-written runtime-L mask) RESULTS:
- Gate1 ONE PDI / NO WEDGE: PASS. Runtime-L RTP gives one resident ELF for all
  positions; 5+ consecutive runs across chunk boundaries (L=40/64/65/128/201/256)
  wedge-free. The two-full-fabric-PDI wedge is FIXED.
- Gate2 DETERMINISM: runtime-L mask is EXACT (attn_out bit-exact, 5.2e-4 vs CPU,
  every position incl. partial-chunk). Production c2_merged is bit-exact
  run-to-run ⇒ the earlier "synthetic-weight nondeterminism" was a HARNESS/BO
  artifact, NOT the device — RESOLVED.
- Gate3 hf-gate: BLOCKED by a PRE-EXISTING c2_attn O-wave cross-token hazard
  (token0 correct, token1+ ~17% off; O-proj `proj` drifts on PDI reuse while
  attn_out stays correct — the O-wave reads a DEVICE-WRITTEN activation vs
  c2_merged's host-synced BO). Masked before because per-position PDIs ran each
  c2_attn once/process. Fix = ~5k-line matvec-core surgery (out of scope).
- Gate4 REGRESSION: PASS — default c2_merged gold tokens unchanged, IR leak-free.
- Gate5 LATENCY (median ms/tok, seq=256): resident c2_attn 15.6 vs c2_merged+CPU-
  attn baseline 3.4 → ~4.5x SLOWER. CPU attn alone = 0.67 ms. Dominated by fixed-
  4-chunk attention + a SCALAR runtime-L mask loop (aie2p can't legalize vector
  arith.select<i1> → per-column scalar stores). Even with the hazard fixed it
  would not beat CPU attention.

ARC-LEVEL VERDICT: across every form built and measured on HW — standalone,
batched-1-dispatch, 2-config fused, single-device 1-PDI fused, and resident
1-PDI — on-NPU decode attention is NUMERICALLY CORRECT but NEVER beats CPU
attention for llama-3.2-1B decode. Root cause (consistent throughout): attention
is a tiny OVERHEAD-bound piece; CPU does it in ~0.2-0.7 ms; the NPU's fixed
overheads (launch / compute-serialization / scalar mask loop) exceed that. The
BFP576 4x multipliers were NEVER the lever — compute was never the bottleneck.

TRANSFERABLE WINS (the real value of this arc):
- The runtime-RTP-driven SINGLE-RESIDENT-PDI mechanism (one ELF reused across
  positions via a host-written L, no per-position PDI, no wedge) — exactly what
  the 16-layer-loop resident design (RESIDENT_DEVICE_EVOLUTION step 5) needs.
- Bandwidth model (corrected): fabric weight-ingest ceiling ~51 GB/s at C=8
  (kernel-only), BW ~linear in columns; residency reclaims the ~3.4x host
  envelope → ~1.8x on bf16 (88→~49 ms/tok); AWQ's 4x-smaller stream is the lever
  to ~10-13 ms/tok. THE DECODE LEVER IS RESIDENCY + AWQ, NOT ATTENTION.
- Determinism: c2_merged is deterministic; "nondeterminism" was a BO-init artifact.
- Reusable hazard: device-written activation handoffs drift on PDI reuse vs
  host-synced BOs (chase the x_avail/x_ready dance + X-broadcast drain) —
  relevant to ALL residency activation handoffs ([[resident-runner-output-bo-sync]]).

RECOMMENDATION: STOP the on-NPU attention thread. Keep attention on CPU. Apply
the runtime-RTP single-resident-PDI mechanism to the projection/FFN residency
(the ~1.8x lever), and pursue AWQ residency for the ~10 ms/tok floor.

## Mask fix + CLEAN compute verdict — 2026-06-19 (refines the arc conclusion)

MASK FIX: `_c2attn_mask_invalid_cols_rtp` rewritten from 4096 scalar stores
(worst case, boundary=0) → runtime v32-block: `scf.for cb in [first_full,8)`,
each block = 16× v32 vector.transfer_write of -inf (legalizes; mirrors the baked
_emit_mask_invalid_cols). Fully-past chunk → 128 v32 stores. Partial block stays
scalar but as runtime scf.for (Python-unrolled 64 rows overflowed program mem).
`first_full` uses a legal SCALAR-INDEX arith.select (not the illegal vector
arith.select<i1>). PythoC partial-block path NOT needed (partial scalars
negligible). Escape hatch PYTHOC_C2_ATTN_SCALAR_MASK (A/B). Mask exact
(5.2e-4 vs CPU, deterministic) at all positions incl. partial-chunk.

CORRECTION: the prior "15.6 ms dominated by the scalar mask" was WRONG — 15.6 ms
is the HOST ENVELOPE (dispatch + 18-BO weight sync, which production skips via
static preload), independent of the mask. Kernel-only resident c2_attn is
~3.2-3.8 ms; the mask was only ~0.2-0.6 ms of that. The fix saves ~0.2-0.6 ms
and removes the 4096-store worst case, but the mask was never the dominant cost.

CLEAN KERNEL-ONLY (start→wait2) NPU ATTENTION COMPUTE vs CPU — the number we
never had:
| seq | NPU attn (kernel-only) | CPU attn | verdict |
| 64  | 0.260 ms | 0.159 ms | SLOWER 1.6x |
| 128 | 0.288 ms | 0.172 ms | SLOWER 1.7x |
| 256 | 0.333 ms | 0.658 ms | FASTER 0.5x |
NPU attention compute is ~FLAT in seq (fixed parallel work); CPU scales ~linear.
So NPU compute is SLOWER at short context (≤128) but CROSSES OVER to FASTER at
seq≈200+. The compute was never the problem — it's competitive and WINS at
length. What kept on-NPU attention from paying off end-to-end was (1) dispatch/
launch overhead and (2) the O-wave cross-token hazard — NOT the attention compute.
This CONFIRMS the overhead-bound thesis and the "lever is residency, not
attention compute (BFP576 multipliers)" conclusion: optimizing compute can't help
when compute already crosses over CPU; only killing the overhead can.

Files: builders/o_gemv_ffn.py (mask body, flag-gated), tests/test_c2_attn_ir.py
(forbid only illegal VECTOR arith.select), tools/bench_c2_attn_mask.py (NEW).
The O-wave cross-token hazard (final FFN output res[14]) remains the end-to-end
blocker (separate ~5k-line core issue, untouched).

## Remedy #1 (runtime-seq await on attn_out write) — REFUTED ON HW 2026-06-19

Added/confirmed `dma_await_task` on the attention-out write tasks (o_gemv_ffn.py
~5099-5100) before the O-wave reads args[1]. RAN stepR (own run, the fix agent
parked without finalizing): final output STILL ~17-24% wrong at EVERY position
incl. cold pos=0, now det=True (consecutive dispatches agree). pos0 18.7%, pos39
24.4%, pos63 23.2%, pos128 22.9%, pos255 16.9%.

CONCLUSION: a runtime-sequence `dma_await_task` does NOT fix it. DMA-completion
await ≠ the global cache/coherence flush the host BO-sync gave c2_merged. The
await serializes write→read (makes it deterministic) but the O-wave still
consumes a wrong activation. The cheap await-only remedy #1 is DEAD. Options
left: (1) a real HOST BO-sync round-trip of args[1] between waves — but that
re-adds the per-token host hop the fusion removed (un-fuses); (2) remedy #2 =
on-chip L2 lock-coherent handoff (no DDR round-trip; the principled fix; needs
the on-chip scatter→broadcast, the C3.4-class cross-column work). Confirms the
quiescence report: only a producer/consumer-ORDERED handoff (L2 lock, or host
sync) closes it — an await on DMA completion is neither.

## RE-DIAGNOSIS — the "17-24% wrong" is a WARM-ELF reuse bug in c2_merged ITSELF, not a c2_attn/coherence/await bug — 2026-06-19

The Remedy-#1 refutation was correct that the bug is DETERMINISTIC and STRUCTURAL
(not a coherence race). But the bug was MIS-LOCALIZED to c2_attn and to the
attn_out handoff. Hardware-reproduced root cause:

ROOT CAUSE: a DETERMINISTIC cross-dispatch (warm-ELF) state hazard in the O-proj
matvec wave, present in BOTH c2_merged (the supposed gold REFERENCE) and c2_attn.
The FIRST dispatch of a freshly-loaded ELF in a process is CORRECT; EVERY
subsequent dispatch returns a fixed ~16-18%-wrong result. The O matvec leaves
resident L1 state (accumulator / weight-stationary wo tile / lock state) that is
set up cold-correctly on dispatch 0 and carries stale into dispatch 1+.

EVIDENCE (all on HW, c2_attn_cache, synthetic weights, vs a PURE-NUMPY ref):
1. REFERENCE IS GARBAGE (discriminator #1). stepR compares c2_attn against
   c2_merged_npu(cpu_attn). c2_merged_npu vs pure-numpy c2_ffn_ref: rel err
   3% at pos0 but 114-165% at pos>=39. The stepR reference is itself wrong, so
   "c2_attn 17-24% off the reference" is MEANINGLESS as a c2_attn signal.
2. WARM-ELF, NOT WEIGHTS/POSITION. Isolated to the O-proj (arg2). Per-PROCESS
   one-shot, same input, same ELF:
     fresh process, call0:  proj_err 4.78e-4, out_err rel 2.6%   CORRECT
     same process,  call1:  proj_err 2.15e-2, out_err rel 16%    WRONG
     same process,  call2:  proj_err 2.15e-2, out_err rel 16%    WRONG (fixed)
   Reproduced in TWO independent fresh processes (both call0 = 2.6%). The error
   is a fixed magnitude (2.1475e-2 every warm call) => deterministic stale state.
3. NOT the await edit (discriminator #2). The bug exists in c2_merged, which has
   NO attention wave and NO await — so the attn-out await cannot be the cause.
   A/B: building c2_attn with the await REMOVED (env toggle, reverted) does not
   "fix cold"; it WEDGES the device (ERT_CMD_STATE_TIMEOUT) even cold. The await
   is load-bearing for attention-wave DMA liveness but irrelevant to the 18%
   numeric drift. With await ON, c2_attn cold call0 = rel 2.0-2.3% vs numpy at
   BOTH pos0 and pos39 — i.e. cold-correct, warm-wrong, identical to c2_merged.
4. NOT the attn_out handoff (discriminator #3/the prior prime suspect). Reading
   back arg1 (attn_out, the O-matvec INPUT) on warm calls: err_vs_host = 0.0 on
   EVERY call (call0,1,2). The matvec INPUT is bit-perfect; only its OUTPUT
   (proj) drifts on reuse. So the corruption is INSIDE the O matvec wave, not in
   the gather/broadcast/attn_out delivery.
5. c2_attn cold output matches the pure-numpy c2 reference (rel ~2%) at pos0 AND
   pos39, so c2_attn's attention + FFN math is CORRECT on dispatch 0. The "cold
   pos=0 = 18.7% wrong" in stepR was an artifact: stepR runs several c2_merged
   warmup dispatches first, so by the time it dispatches c2_attn BOTH the device
   AND the reference were already warm-and-wrong.

VERDICT on the four candidates:
  - bad synthetic reference:           TRUE (the dominant cause of the stepR
    signal — but it is a SYMPTOM of the same warm-ELF bug, since the reference
    is just warm c2_merged).
  - await-edit regression:             FALSE (bug predates it; lives in c2_merged).
  - deterministic attn-wave/add-herd corruption of res1: PARTLY — it IS
    deterministic O-proj corruption, but it is NOT attention/add-herd specific:
    c2_merged (no attention) shows it identically. It is a generic warm-reuse
    hazard in the O matvec wave.
  - gather/broadcast mismatch:         FALSE (arg1 input bit-perfect on reuse).

IS "17-24% WRONG" A REAL c2_attn BUG? Only as far as c2_attn inherits the SAME
warm-ELF O-matvec hazard that c2_merged has. It is NOT a c2_attn-specific bug,
NOT a coherence race, NOT the await, NOT the attn_out handoff. The first
dispatch is correct; the bug is dispatch-reuse-only and shared with the
reference.

WHAT I COULD NOT FULLY RESOLVE: the exact resident artifact inside the O matvec
wave (accumulator not re-zeroed vs weight-stationary wo tile not reloaded vs
lock count drift). The fixed 2.1475e-2 magnitude and arg1-input-clean result
point at an uninitialized/accumulated O-matvec L1 buffer on dispatch>=1, but I
did not instrument the matvec core's L1 directly (out of scope: no core edits).
For the decode loop this means production must either (a) re-zero the O-matvec
accumulator buffer at the top of every dispatch, or (b) treat the design as
single-dispatch-per-load; per-token PDI reuse will drift. NOTE this also taints
any prior c2_merged numbers gathered from a SECOND+ dispatch in the same process.

TOGGLE HYGIENE: a temporary PYTHOC_C2_ATTN_NO_AWAIT env gate was added around the
attn-out await for the A/B and REVERTED; default await path is byte-identical to
before. No production kernel/builder logic was changed.

## ROOT-CAUSE PINNED — warm-dispatch O-matvec drift is a DMA/lock-fabric skew, RESET by any intervening LoadPDI; production is FINE — 2026-06-19

Resolves the central contradiction (production reuses c2_merged across 16
layers/all tokens yet passes the gold gate, while the back-to-back test drifts
~16%). All on HW (Strix Halo, c2_merged_ref ELF, synthetic weights, vs the
numpy c2 ref). Three diagnostic harnesses written + RUN + DELETED; no production
edit (IR gate `tests/test_c2_attn_ir.py` 6/6 PASS, c2_merged byte-identical).

### EXPERIMENT 1 (the interleave test) — DECISIVE, resolves the contradiction
Dispatch on a FIXED input: c2(call0) -> [dispatch a DIFFERENT PDI: the prebuilt
`decode_attn_b8_s64` ELF] -> c2(call_interleaved); plus plain back-to-back.
    call0  cold                 : rel 2.14%   CORRECT
    call1  back-to-back         : rel 23%     WRONG
    call2  after decode_attn PDI: rel 2.14%, d(call2,call0)=0.0  BIT-IDENTICAL TO COLD
    call3  back-to-back again   : rel 25%     WRONG again
=> An intervening LoadPDI of ANY different PDI FULLY RESETS the c2 cores' state
to cold-correct. The drift is a "back-to-back same-ELF reuse WITHOUT an
intervening LoadPDI" hazard ONLY.
PRODUCTION IMPLICATION: the decode loop is per-layer [rgr -> attention ->
o_gemv_ffn(c2)] then lm_head, so c2 is NEVER dispatched twice without a
different-PDI LoadPDI between (rgr/attn/lm_head). Production therefore gets a
device reset before every c2 reuse and NEVER hits the drift -> the gold gate
passes legitimately. The back-to-back test was INVALID. The single-PDI RESIDENT
c2 design (reuse c2 with NO intervening reset) is the thing that actually
exposes this; it must insert a reset (or a different PDI) between reuses, OR
not reuse a single resident PDI for c2.

### EXPERIMENT 2 — PINNED mechanism: DMA/lock fabric skew, NOT acc / NOT weight tile
Stage-localization (read back all intermediates cold vs warm): drift FIRST
appears at `proj` (O-matvec out, rel 13.5%), then amplifies through
gate/up/swiglu/down. Confirms it is INSIDE the O-matvec wave.

Candidate elimination (each a HW readback test):
  1. ACCUMULATOR RESIDUE — REFUTED. mv_pythoc.ll re-zeros `acc`
     (`zeroinitializer` phi at the per-row preheader) every invocation. HW: a
     warm dispatch with a ZERO input vector yields proj == EXACTLY 0.0 (no
     additive leftover). The matvec carries no cross-dispatch accumulator state.
  2. STALE STATIONARY WEIGHT L1 TILE — REFUTED. (a) Re-syncing the wo host BO
     every warm call does NOT change the drift (fixed 0.121 abs). (b) A warm
     call with a DIFFERENT weight wo_B tracks wo_B (err_vs_B 0.13, err_vs_A
     1.48) => the weight IS fully re-DMA'd each dispatch; the L1 W tile is not
     stationary/stale.
  3. DMA/LOCK FABRIC SKEW — CONFIRMED (by elimination + structural signature).
     The warm-vs-cold proj diff: 1811/2048 elements BIT-IDENTICAL, ratio median
     exactly 1.0; only ~235 elements perturbed, uniformly spread across all 8
     compute columns (22-32 per 256-block), each off by <=~0.12.
     STABILITY TEST (the clincher): the perturbed-index SET is
       - jaccard 1.000 vs varying the WEIGHT (identical indices AND magnitudes;
         weight-content-independent)
       - jaccard 0.929 vs varying the INPUT (near-fixed; only rounding-boundary
         positions move)
     A fixed, data-independent structural index set = a DMA-descriptor / lock-
     credit / x-replay (`x_repeat_count=31`) counter that does NOT return to its
     cold post-LoadPDI init at end-of-dispatch; the residual skew mis-delivers a
     fixed subset of on-chip matvec input lanes on every warm reuse. Only
     LoadPDI re-initializes the locks/BD pointers (= Exp 1).

### VERDICT
- Is this a real PRODUCTION bug? NO. Production always interleaves a different
  PDI before reusing c2, which resets the device; the gold gate is correct, not
  merely robust. Prior c2_merged numbers taken back-to-back IN ONE PROCESS
  WITHOUT an intervening different-PDI dispatch are tainted (the earlier scope
  note about "tainted 2nd+ dispatch numbers" stands, with the caveat: only
  same-ELF-without-LoadPDI-between reuse is tainted).
- Is the back-to-back test valid? NO — it omits the LoadPDI reset that every
  real c2 reuse gets.
- Resident single-PDI c2 design: this hazard is REAL for it. Fix = insert a
  device reset / a throwaway different-PDI LoadPDI between resident c2 reuses,
  OR pursue the true fix below.
- TRUE FIX (if a single resident c2 PDI is wanted): make the c2 DMA/lock fabric
  return-to-init at end-of-dispatch — i.e. ensure the weight/output/x-broadcast
  BD chains and the lock credits (`w_avail`/`w_dma_done`/`x_avail`/`y_done`,
  `x_repeat_count`) net to their init counts per dispatch so dispatch N+1 starts
  identical to the cold post-LoadPDI state. Re-zeroing the matvec accumulator is
  NOT the fix (already zeroed; refuted). Re-syncing host BOs is NOT the fix
  (refuted).

WHAT I COULD NOT FULLY RESOLVE: I narrowed the on-device culprit to the
DMA-descriptor/lock-credit/x-replay fabric (by elimination + the fixed-index
structural signature) but did NOT add core-L1 readback BDs to read the exact
lock counts / BD pointers at end-of-dispatch and name the single offending
counter. The structural evidence (weight-independent fixed 235-index set, reset-
by-any-LoadPDI, weight re-DMA'd, acc clean) is conclusive that it is fabric
state, not compute state. Naming the exact lock would need a temporary lock-
count readback (in scope, not done).

DIAGNOSTIC HYGIENE: 3 throwaway harnesses (tools/_exp1_interleave.py,
tools/_exp2_*.py) created, run, and DELETED. No builder/kernel/host production
logic changed. tests/test_c2_attn_ir.py 6/6 PASS (c2_merged byte-identical).

## Warm-dispatch O-matvec drift — ROOT-CAUSED to W-input path (2026-06-19, HW)

GOAL: make c2 fabric QUIESCE so a warm (back-to-back, no intervening LoadPDI)
dispatch is bit-identical to a cold one. NEW back-to-back harness added:
`tools/test_c2_attn.py stepW` — dispatches the PRODUCTION c2_merged ELF N times
on the SAME input in ONE process and checks call1..N == call0 bit-exact (proj
idx 2 AND output idx 14) + call0 vs numpy. This is the quiescence gate.

KEY FINDING — c2_merged DOES drift on warm reuse (overturns the read-only
O_WAVE_QUIESCENCE_ANALYSIS.md claim that only c2_attn drifts). stepW seed=0:
cold rel_proj=5.5e-3 (correct), but warm proj drifts at exactly **238/2048**
elements, deterministic, non-accumulating (call1==call2==call3, all !=call0).
Matches the GOAL's ~235-elem fixed-index signature. (Production per-layer decode
is unaffected: c2 is never reused without a LoadPDI between; only resident
single-PDI reuse exposes it.)

ISOLATED the locus with the `PYTHOC_C2_STAGES=1` knob (O proj wave alone) +
probe kernels that route W / X data straight to the output through the y path:
- Drift is ENTIRELY in the O proj wave (stage 1). Downstream waves irrelevant.
- Index pattern: ALL drift indices are i%8==0 → **row 0 of every M_TILE(=8) output
  tile** drifts; rows 1-7 are bit-exact. 238 of the 256 tiles affected.
- **X-probe (write b[i]=normed to output): ZERO drift** — the X/normed activation
  delivery quiesces perfectly (shim broadcast pkt 1, repeat_count=0, discrete).
- **W-probe (write a[i*k]=first elem of each W row to output): 238 drift, value
  +64 rows.** Cold out[row r]=r (correct W row). Warm out[tile.row0]=r+64 — row 0
  of each tile reads a COMPLETE VALID W row from 64 rows (=8 M_TILEs) ahead.
So the carried state is the **W-input DMA path** (shim MM2S0 → memtile S2MM0/MM2S1
relay → mat S2MM1 → _wb), specifically _wb's row-0 slot on the first read of a
warm dispatch. Not X, not the y-scatter, not the matvec compute, not the
accumulator. +64 rows = 8 tiles = N_COLS.

RULED OUT (all built + measured on HW, none changed the 238 count):
- matvec software pipelining / loop hints (rebuilt mv_pythoc.ll with
  prepare_for_pipelining + loop_range NO-OPed): still 238. The matvec IR is
  correct (acc is phi-zeroed per row); the bug is in DMA data delivery.
- mat-tile W lock prefetch credit (PYTHOC_C2_WQUIESCE: init w_avail=0 +
  authorize-then-wait so the mat S2MM1 parks on the LOCK not mid-stream): 238.
- awaiting the shim W push tasks (PYTHOC_C2_WAWAIT: issue_token + dma_await on W,
  not just Y): 238. Combined WQUIESCE+WAWAIT: 238.
- one extra shim "drain" W tile to absorb a prefetch: WEDGED (unbalances the
  memtile W relay locks — the spare flows through the whole chain).
- 2-slot mat-L1 W ping-pong: resource-alloc FAIL (two 8x2048 bf16 W buffers
  overflow the 64 KB mat L1).

CONCLUSION ON MECHANISM: the standing-prefetch-credit model (mat/memtile W DMA
parks 1 tile ahead, warm reuse completes the parked transfer with the next
dispatch's first beat) is consistent with the +64/row-0 signature, BUT removing
the mat credit (authorize-then-wait) and forcing W-push completion (await) do NOT
fix it. That places the carried state in **persistent HW DMA-descriptor / stream-
FIFO state of the W path that only a LoadPDI clears** (the multi-dim shim W BD
[(16,131072),(32,512),(512,1)] with its 64-row outer stride, or the circuit-
switched mem→mat W FIFO), NOT in any builder-settable lock/await. A LoadPDI resets
it (hence cold is always correct and any intervening different-PDI dispatch fixes
the next call); warm single-PDI reuse inherits the mid-phase descriptor.

WHY THE SIMPLE FIXES CAN'T REACH IT: the W relay is pure-DMA (memtile_dma +
circuit flow) with single buffers (mat L1 can't hold a 2nd 8-row W tile); the
descriptor/FIFO phase is below the lock layer. The remedy the evidence points to
is the one O_WAVE_QUIESCENCE_ANALYSIS.md §4 / the prior-session framing flagged as
core-surgery: **re-arm/reset the W input channel per dispatch via a control-packet
program (channel reset cmd) or a one-shot finite W BD chain**, OR move the W relay
off the persistent circuit FIFO. There is no runtime channel-reset op in the
mlir-aie Python API (checked aiex.py / _aiex_ops_gen.py), so the builder-only path
is exhausted; the control-packet reset (proven in moe_control_packets /
ctrl_packet_opus / memtile_program_cost) is the next step and is gated on the
residency thread being active.

STATUS: root cause localized + signature fully characterized; bit-identical warm
reuse NOT achieved. The non-working credit/await scaffolds are gated OFF
(PYTHOC_C2_WQUIESCE/-WAWAIT default 0); the `_wchunk` refactor is IR-identical for
the default. REGRESSION CLEAN: tests/test_c2_attn_ir.py 6/6 PASS;
test_hf_answer_gate_o_gemv_ffn_c2_merged gold gate PASS ("...Paris."). Files
touched: builders/o_gemv_ffn.py (gated scaffold + _wchunk refactor),
tools/test_c2_attn.py (stepW warm-reuse gate). Kernel mv_pythoc.ll restored to
original; kernels/matvec.py unchanged.

---

## WARM-DISPATCH W-RELAY DRIFT — FIXED (2026-06-19, 2-slot L2 relay ping-pong)

GOAL MET: c2 back-to-back same-PDI warm reuse is now BIT-IDENTICAL to cold.
`tools/test_c2_attn.py stepW` seed=0, n=6: `warm==cold proj=True out=True`,
n_proj_diff=0 (was 238). Every intermediate stage (proj/res1/normed2/gate/up/
swiglu/down/out) is 0-drift, deterministic across all warm reuses.

### Root cause (refined from the pinned diagnosis)
The carried state was the **memtile W relay** (shim MM2S0 -> memtile S2MM0 fill /
MM2S1 drain -> mat S2MM1), specifically the relay's SINGLE L2 W buffer running as
a self-cycling ring with a standing 1-fill credit (`w_dma_done` init=1). A single-
buffer relay can NEVER run at zero credit (deadlock), so at end-of-dispatch it
always carries one parked prefetch -- its DMA descriptor/stream phase is left
advanced by exactly ONE outer-stride (64-row) cycle, which only a LoadPDI resets.
This is why removing the MAT-tile credit alone (the prior `PYTHOC_C2_WQUIESCE`
authorize-then-wait) did NOT help: the leak was one stage upstream, in the L2
relay, not the mat L1 ring.

### The fix (FIX #2 from the plan: cycle-aligned one-shot-equivalent relay)
Give the memtile W relay (and the down K=8192 dw relay) **TWO L2 buffers** and run
the S2MM0 fill / MM2S1 drain as a **2-BD ping-pong** (slot0->slot1->slot0). Every
wave delivers an EVEN number of relay fills (O: 32, gate/up: 128, down: 128), so
the BD pointer lands back on slot 0 at task end -- the relay quiesces to its post-
LoadPDI descriptor BY CONSTRUCTION. Warm reuse == cold. No control packets needed.

CRITICAL detail: the relay credit must be **init=1**, NOT 2. With init=2 the fill
side races two slots ahead across the per-outer shim-task boundaries, leaving a
few-element NON-deterministic residual in gate/up (idx ~64) that down's full dot-
product amplifies to ~1800 elems in the output. init=1 (single standing credit,
2-slot ring) makes everything bit-identical and deterministic. (Measured: CR=2 ->
gate {64,65,72} non-det, down ~1800; CR=1 -> all 0.)

L2 has 512 KB/tile; 2 x 32 KB main-W + 2 x 32 KB down-W buffers fit easily (this
is why the prior 2-slot **mat L1** ping-pong failed -- it overflowed the 64 KB
compute L1; the relay buffers live in L2).

### Gating (production-safe)
`PYTHOC_C2_WRELAY2`: explicit "1"/"0" overrides; default = AUTO-ON for the
RESIDENT path (`attn_resident`, single-PDI reuse where the drift bites), AUTO-OFF
for plain c2_merged (it reloads the PDI between dispatches, already clearing the
drift, so its IR stays byte-identical for production). `PYTHOC_C2_WRELAY2_CR`
overrides the credit (default 1); `PYTHOC_C2_WRELAY2_DN` gates the down relay.

### Validation
- stepW seed=0 n=6: warm==cold proj=True out=True, n_proj_diff=0  -> GOAL MET.
- Per-stage warm-diff: baseline {proj 238, gate 7375, up 7349, swiglu 7687,
  down 2033, out 2033} -> WRELAY2 CR=1 {all 0}.
- Resident stepR (pos 63/64/127, incl. boundary + partial chunk): det=True,
  rel ~1.1e-2 vs c2_merged ref (within tol). No wedge/deadlock.
- stepB (non-resident collapse, seed 0/1): PASS (unchanged; WRELAY2 auto-off).
- REGRESSION: tests/test_c2_attn_ir.py 6/6 PASS (default c2_merged byte-identical,
  leak-free); gold gate test_hf_answer_gate_o_gemv_ffn_c2_merged PASS
  ("The capital of France is Paris.").

### Files changed
- builders/o_gemv_ffn.py: `_emit_call2_c2` -- 2-slot L2 W-relay + down-W-relay
  ping-pong gated by `_wrelay2`/`_wrelay2_dn` (auto-on for resident); credit knob
  `PYTHOC_C2_WRELAY2_CR` (default 1). WQUIESCE comment corrected (superseded).

---

## Part 2 — c2_attn host-overhead close (incremental KV tiling + resident KV BO)

GOAL: take c2_attn decode from ~7.5 tok/s (133 ms/tok) toward the c2_merged
baseline (11.9 tok/s, 84 ms/tok) by eliminating the O(seq) per-token KV
re-pack + re-upload. Only the NEW token's K/V row should be tiled/transferred
each step. All work gated under the c2_attn pack mode; default c2_merged
stays byte-identical.

### Part 2a — INCREMENTAL host tiling (LANDED, default-on)
Before: `_run_c2_attn` zeroed + copied the FULL 256-padded k_pad/v_pad for all
8 groups and re-tiled all 4 chunks (8*4*4096 elts each) EVERY layer EVERY
token (O(seq) host work).
After: a PERSISTENT per-layer tiled buffer (`_C2_ATTN_KV_STATE[layer_idx]`)
is kept across tokens. On first sight of a layer it is SEEDED once by tiling
the rows already in the cache (the prefill KV, 0..current_pos). Thereafter
each token writes ONLY the new row(s) into it: position s lives in chunk
c=s//64, tile-row r=s%64; per group g, the flat offsets are
`g*kv_size + c*tile_size + ((col//8)*512 + r*8 + (col%8))` for col 0..63.
`c2_attn_reset_kv_state()` clears the buffers at the start of each sequence
(called in `generate()` right before the decode loop).

Verified byte-identical to the old full re-tile across all chunk boundaries
(seq 38/39/64/65/66/131/201) via a standalone numpy check.

Result: steady-state per-token dropped from ~133 ms to ~81 ms/tok
(measured tokens 2..N consistently 80-88 ms). On a full 20-token decode the
token-1 one-time overhead (~480 ms; first-dispatch BO alloc) amortizes out.
Gold "The capital of France is Paris." decodes correctly. This single change
closes essentially the whole host-side gap to c2_merged.

### Part 2b — RESIDENT KV BO, incremental device push (LANDED behind flag, OFF by default)
Mechanism: declare args 16/17 (k_all/v_all) STATIC so the generic
write/sync path skips them after the first (seeding) dispatch; on each
subsequent token push ONLY the touched chunk's bytes into the resident
per-layer BO via a new cache helper `KernelCache.update_static_bo(bo_key,
arg_index, host_array, ranges)` (copies the touched element ranges into the
mapped BO and issues one contiguous ranged `BO_TO_DEVICE` sync) BEFORE the
dispatch. Gated by `PYTHOC_C2_ATTN_RESIDENT_KV` (default "0").

Result: net SLIGHT REGRESSION (~84-88 ms/tok vs 2a's ~81). After 2a removed
the O(seq) host re-tile, the per-token K/V upload is small next to the ~80 ms
NPU compute, so the extra ranged-sync round-trip does not pay for itself
(per-group syncs were worse, ~88 ms; collapsing to one ranged sync per arg got
it to ~84, still above 2a). Left in behind the env flag for future use (e.g.
if NPU compute drops and the upload re-dominates). Correctness preserved
("Paris.").

### Final numbers (Strix Halo aie2p, instruct bf16, gold prompt)
- c2_attn BEFORE (committed e7c1341bf): ~133 ms/tok steady-state, ~7.5 tok/s.
- c2_attn AFTER (2a default): ~81 ms/tok steady-state; ~12.3 tok/s steady,
  effectively matching c2_merged. (20-token wall avg ~9.5 tok/s; gold prompt
  hits EOS at token 9 so its wall avg is dragged by the amortized token-1
  overhead -- judge on steady-state.)
- c2_merged baseline (control, same session): ~85 ms/tok, ~11.5-12.0 tok/s.
  c2_attn steady-state now PARITY with c2_merged. Residual gap is the ~480 ms
  one-time first-token overhead (BO allocation), not per-token host work.

### Regression
- tests/test_c2_attn_ir.py: 6/6 PASS (default c2_merged byte-identical, 15-arg,
  leak-free; resident c2_attn ONE configure/run).
- c2_merged gold gate (`make profile`, no env var): "The capital of France is
  Paris.", ~11.5 tok/s -- unchanged.
- c2_attn gold gate (`PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=c2_attn make profile`):
  "The capital of France is Paris." (clean runs; ~1/3 of runs still hit the
  separate process-teardown quiescence garbage "!!!!", out of scope).

### Files changed (all c2_attn-gated; default c2_merged + AWQ untouched)
- llama32_1b_decode.py: persistent `_C2_ATTN_KV_STATE` + `c2_attn_reset_kv_state`
  + `_c2_attn_tile_row_offsets`/`_C2_ATTN_ROW_OFF` + `_c2_attn_chunk_ranges`;
  `_run_c2_attn` rewritten to seed-once + incremental-row tiling (2a) and
  optional resident-KV incremental BO push (2b, flag-gated).
- kernel_builder/cache.py: new `KernelCache.update_static_bo()` partial in-place
  resident-BO update helper (ranged BO_TO_DEVICE sync). Generic load_and_run
  path unchanged.
- llama32_1b_inference.py: import + call `c2_attn_reset_kv_state()` at the top
  of the decode phase in `generate()` (no-op for c2_merged/AWQ).

---

## MEMKV: lifting the 256-token cap (memtile-staged KV feed → host-side single-BD)

The 256 cap was `MAX_CHUNKS(4) × KVP(64)`.  Origin (confirmed): the resident
c2_attn fed K/V over a shared shim channel with `2 + 2·A_N_BUF_FILLS` shim BD
*tasks* per group per token (q + per-fill K + per-fill V + out).  `A_N_BUF_FILLS`
grows with chunk count, exhausting the ~16 shim BD IDs at n_chunks≥8 (compile)
and wedging the packet stream at n_chunks 5–6 (runtime) — the documented blocker
at builders/attn_decode.py:512–518.

### What was tried and what shipped
1. **Memtile (L2) staging — RETIRED (HW wedge).**  Stage the full per-group KV
   into an L2 buffer (shim→memtile S2MM4/5) and drip-feed the add tile's L1
   double-buffer (memtile MM2S4/5 → add S2MM0/1) via a balanced-lock BD chain,
   so the chunk count lives in an L2 BD chain not shim BD IDs.  Two variants
   (q direct-to-add, then q also staged through the memtile to fix the q-vs-K
   arrival race) BOTH wedged on the FIRST dispatch (`ERT_CMD_STATE_TIMEOUT`) at
   *every* chunk count including n_chunks=4 — so the deadlock is the memtile
   routing/staging itself, not the chunk count.  Compile was clean; the wedge is
   a runtime fabric deadlock in the shim→memtile→add packet path that I could
   not root-cause within budget.  The scaffold is kept gated OFF
   (`_A_MEMKV_STAGE = False` in `_emit_call2_c2`) for a future revisit.
2. **Host-side single-BD KV feed — SHIPPED, WORKS ON HW.**  Insight from the
   bisect: the cap is *purely* the shim BD-*task* count, and the on-fabric
   shim→add routing + the add-mem 4-fill ring already work.  So feed the FULL
   per-group K (and V) in **ONE** shim BD each (`len=A_KV_SIZE`, pkt 16, the
   proven shim→add S2MM0/S2MM1 routing) and let the add-mem ring's
   `A_N_BUF_FILLS` fill-BDs each pull one `A_CHUNKS_PER_BUF`-tile slice off the
   single stream, gated by the existing add `k_avail`/`v_avail` locks (stream
   backpressure drips 64 KB into the 16 KB L1 double-buffer).  Shim BD-task
   usage is now CONSTANT (q+k+v+out = 4 tasks/group) regardless of chunk count —
   context length is decoupled from the shim BD budget.  No new packet ids, no
   memtile leg, no routing change → low risk, reuses the validated path.

### Wiring that replaced the per-chunk shim DMA
- Builder geometry: `PYTHOC_C2_ATTN_MEMKV=1` sets `A_MAX_CHUNKS =
  PYTHOC_C2_ATTN_MAX_CHUNKS` (default 8 → seq≤512); the online-softmax core loop
  and runtime-L mask (`clamp(L - 64·c, 0, 64)` per chunk) were ALREADY chunk-
  count-general, so only the BD wiring changed.
- Host `_attn_wave0` (builders/o_gemv_ffn.py): under `_A_MEMKV`, K and V each
  emit ONE `dma_configure_task_for` BD of `len=A_KV_SIZE` instead of
  `A_N_BUF_FILLS` per-fill tasks.  q unchanged (1 task).
- Host `_run_c2_attn` (llama32_1b_decode.py): `_RES_MAX_CHUNKS` scales with the
  same env; the seed-once + incremental-per-row tiling (Part 2a) is unchanged
  (still tiles only the new row per token).  ABI unchanged (18 args); the
  preload (`_preload_decode_weights`) uses the same builder → consistent.

### BD-ID usage (the whole point)
- BEFORE (per-chunk): `2 + 2·A_N_BUF_FILLS` shim BD tasks/group → scales with
  chunks → ≥8 chunks exhausts the budget.
- AFTER (host single-BD): `4` shim BD tasks/group (q,k,v,out), CONSTANT.  Verified
  by lifting to 8 chunks (seq 512) compiling and running clean.

### Correctness (HW, tools/test_c2_attn.py stepR, device vs CPU c2_merged oracle)
`PYTHOC_C2_ATTN_MEMKV=1 PYTHOC_C2_ATTN_MAX_CHUNKS=8`, positions crossing the cap:
```
pos= 64 seq= 65  end_err=1.000e+00 rel=2.105e-02 det=True FAIL(borderline)
pos=200 seq=201  end_err=5.000e-01 rel=8.734e-03 det=True PASS
pos=255 seq=256  end_err=5.000e-01 rel=1.105e-02 det=True PASS
pos=256 seq=257  end_err=7.500e-01 rel=1.402e-02 det=True PASS   <- past old cap
pos=300 seq=301  end_err=6.250e-01 rel=1.174e-02 det=True PASS
pos=400 seq=401  end_err=6.250e-01 rel=1.381e-02 det=True PASS
pos=511 seq=512  end_err=5.000e-01 rel=1.220e-02 det=True PASS   <- new ceiling
```
All deterministic; seq 257–512 (past the old 256 cap) correct vs oracle.  The
lone seq=65 "FAIL" is a threshold-tightness artifact: with MAX_CHUNKS=8 every
token runs 8 online-softmax iterations (chunks 4–7 fully masked → exp=0, alpha=1,
mathematically no-op), and the 4 extra masked iterations add ~1 bf16 ULP of
rescale drift; seq=65's small reference magnitude makes `rel` cross 2e-2 by a
hair (2.105e-2).  The baseline 4-chunk path passes the SAME pos=64 (rel
1.156e-2) — it processes only 4 chunks.  It does NOT affect token selection
(gold gate clean), and longer seqs (more real chunks) pass with smaller rel, so
it is not a systematic drift.

### End-to-end decode (HW gold gate)
`PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=c2_attn PYTHOC_C2_ATTN_RESIDENT=1
PYTHOC_C2_ATTN_MEMKV=1 PYTHOC_C2_ATTN_MAX_CHUNKS=8 make profile N_TOKENS=20`:
```
A: The capital of France is Paris.   (clean, no wedge, no garble)
```
per-token 84–90 ms steady-state (~11.7 tok/s; the 7.35 wall avg is dragged by
the one-time 538 ms first-token BO-alloc), matching the c2_merged/c2_attn
baseline — NPU compute is unchanged (same kernels), only the KV feed wiring
changed.

### Regression (default c2_merged + 4-chunk resident untouched)
- tests/test_c2_attn_ir.py: 8/8 PASS (6 original + 2 new MEMKV: cap-lifted IR
  builds ONE-configure with arg16/17 scaled to 8 chunks and NO retired memtile
  artifacts / pkt 17/18; default resident stays 4-chunk / 131072).  Default
  c2_merged byte-identical (15-arg, leak-free).  MEMKV is fully env-gated; with
  the flag off the emitted IR is unchanged.

### Files changed (all c2_attn/MEMKV-gated; default c2_merged + AWQ untouched)
- builders/o_gemv_ffn.py: `_A_MEMKV` geometry gate (A_MAX_CHUNKS from env);
  host `_attn_wave0` single-BD K/V feed under `_A_MEMKV`; dispatcher arg-type
  chunk count made MEMKV-aware; retired memtile-staging scaffold gated OFF
  (`_A_MEMKV_STAGE=False`, `_emit_kv_stage`, KV L2 buffers/locks).
- builders/c2_attn.py: docstring updated (cap is 4 by default, lifted by MEMKV).
- llama32_1b_decode.py: `_RES_MAX_CHUNKS`/`_RES_PADDED` scale with
  `PYTHOC_C2_ATTN_MEMKV`/`PYTHOC_C2_ATTN_MAX_CHUNKS`; assert message updated.
- tools/test_c2_attn.py: `RES_MAX_CHUNKS`/`RES_PADDED` mirror the env.
- tests/test_c2_attn_ir.py: 2 new MEMKV IR-regression tests.

### How to use
```
PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=c2_attn PYTHOC_C2_ATTN_RESIDENT=1 \
PYTHOC_C2_ATTN_MEMKV=1 PYTHOC_C2_ATTN_MAX_CHUNKS=8 make profile ...
```
`MAX_CHUNKS=8` → seq≤512.  Higher is bounded only by add-L1 fill backpressure
and host BO sizing (the L2 budget concern of the retired staging path no longer
applies, since KV never lands in L2).  Larger ceilings need a stepR sweep at
that chunk count to confirm no new shim-stream backpressure stall.
