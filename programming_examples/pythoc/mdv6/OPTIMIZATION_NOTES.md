# MDV6 launch_gap / dispatch optimization — working notes

Goal: meaningful, production-ready latency improvement via launch_gap or dispatch reduction.
Orchestrator = me; opus subagents do implementation/debug.

## Baseline (2026-06-20, `--profile 6`, default config)
- wall **518 ms**, 84 launches, ~1.93 fps
- launch_gap **221.6 ms (42.8%)** = 2638 µs/launch  ← PRIZE
- pre_post 116.7 ms (22.5%)
- npu_run 159.4 ms (30.8%)
- numpy 8.5, cpu_layers 11.4
- saved to /tmp/mdv6_base.json
- accuracy gate: max_class_diff/vector_diff < 5.0 (PASS required)

## Bucket accounting (from profile_harness.py)
- npu_run = full duration of `_xrt_run_kernel` (incl. blocking wait2) + DefaultNPURuntime.run
- launch_gap = inter_launch_wall − (numpy+pack+fuse+cpu+iron_alloc)
  → = host patch-packing + BO fill/sync_in + BO read/sync_out + output reassembly + python dispatch glue
- pre_post = wall − npu − inter_launch_gaps (first-layer setup + last-layer/detection)

## launch_gap decomposition (MDV6_SYNC_PROF) — buckets: copy_in/sync_in/sync_out/copy_out/start/wait
- (pending re-run; wired _sync_prof_report to atexit under MDV6_SYNC_PROF)

## Ideas backlog (ranked, will update)
1. Vectorize host packing + output reassembly (per-tile python loops in _run_*_merged / ocb / gemm) — IF launch_gap is python-bound
2. Reduce BO copy/sync cost (avoid double-copy: np.copyto into bo.map then sync) — IF copy/sync-bound
3. Async/batched submission: submit independent sub-runs without per-launch wait2
4. Reduce launch COUNT: layers firing N calls/frame (mc_re6_c3 ×6, gemm_re6_rn1 ×6, mc_re4_c3 ×4)
5. Persistent run-object / arg reuse already done; check run_cache hit rate

## launch_gap decomposition (MDV6_SYNC_PROF, 4 frames) — DECISIVE
- wait (NPU blocking) ~147 ms/frame  → counted in npu_run
- start ~12 ms/frame → npu_run
- copy_in+sync_in+copy_out+sync_out = **~19 ms/frame** → the ONLY BO part of launch_gap
- => of 222 ms/frame launch_gap, **~200 ms is pure-Python host packing + output reassembly**
- TARGET CONFIRMED: vectorize host pack/unpack. Bit-exact, zero HW risk.

## KEY INSIGHT (simplifies vectorization)
In _run_tiled_mc_inner_ocb_merged / _merged: patch index j → core=j//ppc, slot=j%ppc,
so buffer offset = (core*ppc+slot)*ots = **j*ots**. The per-core concat is identity in j.
=> input_concat = all patches in row-major (tr,tc) order, padded to N_CORES*ppc with patch0, flattened.
=> output buffer for ocb = tiles in j-order; reassembly = reshape/transpose, not a python loop.
extract_patch (torch, per-tile) is the other hot spot: replace with one padded-image + sliding-window gather.

## DECISION: Optimization #1 = vectorize pack/unpack in the 3x3 dispatch paths
(_run_tiled_mc_inner_ocb_merged, _run_tiled_mc_inner_merged) + extract_patch.
Heaviest layers (mc_re4_c3/ftconv1/re6_c3/elan_c3/aconv*) all flow through these.
Then extend to GEMM paths if win confirmed. Guard: A/B bit-exact vs current per-tile path.

## FINDING: rn3 chain is profiler-blind (attribution leak into launch_gap)
- Chain (run_re6_rn3_chain / run_rn3_chain_raster / _geo) dispatches via ResidentXCLBinRunner.run
  (conv/resident_xclbin_runner.py), NOT DefaultNPURuntime.run nor mcr._xrt_run_kernel.
- Profiler hooks only the latter two → chain launches are NOT in n_launches=84, and the chain's
  whole wall (NPU kernel + host packing) falls into an inter-launch gap → inflates launch_gap.
- Per frame ~14 chain stacks (rep_elan4/6/8/12/15/18/21 × 2 RepNCSP each, residual ones).
- Chain host-pack per call (rn3_chain_runner.py): np.zeros padded image + slice + f32_to_bf16_u16
  of the FULL padded image every call + reshape/slice on readback. Vectorizable / reducible.
- ResidentXCLBinRunner exposes last_stats (write_ms/kernel_ms/read_ms) — usable to measure.

## Orchestration plan (avoid file conflicts with running agents)
- Agent #1 (RUNNING): vectorize 3x3 pack/unpack — owns run_tiled_mc.py 3x3 fns + extract_patch + new conv/test_vec_pack.py.
  Leave run_tiled_mc.py + profile_harness.py UNTOUCHED until it lands (keep its before/after honest).
- Idea #2 (queued): GEMM pack/unpack vectorization — _run_gemm_oc_blocked_merged/_kblocked/_pair (same file → dispatch AFTER #1).
- Idea #3 (queued, non-overlapping): chain host-pack vectorization in rn3_chain_runner.py + hook ResidentXCLBinRunner into profiler.

## RESULT #1 (VERIFIED, agent #1) — vectorize 3x3 pack/unpack  ✅ GOAL MET
- wall 518 → **418 ms (-19%)**, launch_gap 221.6 → **135.9 ms (-38.7%)**, numpy -95%, npu_run flat.
- 1.93 → 2.39 fps. Accuracy PASS (max_class 0.1423). Offline bit-exact ALL cases.
- Files: _full_model_helpers/elan_test_tiled.py (+3 numpy helpers), run_tiled_mc.py (3x3 fns), conv/test_vec_pack.py (new).
- Legacy path preserved via MDV6_VEC_PACK=0. Independently re-verified by orchestrator.

## RESULT #2 (VERIFIED, agent #2) — vectorize GEMM 1x1 pack/unpack
- Applied same pattern to _run_gemm_oc_blocked_merged / _kblocked / _pair. Helpers
  pack_gemm_input_batch_u16 + reassemble_gemm_output in elan_test_tiled.py. New test conv/test_vec_pack_gemm.py.
- Offline bit-exact ALL gemm shapes (divisible/partial/multibatch/ppc>1/kblocked/pair).

## COMBINED RESULT (VERIFIED vs original /tmp/mdv6_base.json, profile 8) ✅✅
- wall 518 → **399 ms (-23%)**, 1.93 → **2.50 fps**
- launch_gap 221.6 → **117.3 ms (-47%)**
- numpy -95%, pack ~0, npu_run flat (158), accuracy PASS max_class 0.1423 (bit-identical)
- steady-state warm frame ~291-304 ms (one host-scheduling outlier ~483).
- All host-Python, no kernel/MLIR/ELF. Gated MDV6_VEC_PACK (default on, =0 legacy). Production-ready.

## ITERATION #3 — measurement: make chain visible + split pre/post (profile_harness.py)
Added (measurement-only, additive): pre/post split, and a hook on ResidentXCLBinRunner.run
that buckets the chain as npu_run + sub-buckets chain.kernel/host_write/host_read.

### DECOMPOSITION (chain visible) — REFRAMES the win
- launches 84 → **98** (84 wrapped + **14 chain**, previously uncounted).
- npu_run 158 → **222 ms** (chain.total 63.5 ms moved in from the gaps).
- launch_gap 117 → **50.6 ms** (517 µs/call) = the TRUE host plumbing.
- chain.total 63.5 ms = **chain.kernel 61.7 (97%)** + host 1.8 ms → chain is NPU compute, NOT a host target.
- pre_post 104.8 = pre 85 (= harness gc.collect(), artifact) + post 20. Inner forward timer ~305 ms.

### HONEST WIN (vs original, decomposed)
- Original launch_gap 221.6 = host_plumbing(~158) + chain_NPU(~63, constant).
- Now host_plumbing **~50 ms → host dispatch plumbing cut ~68% (158→50)**; the headline -47% was diluted by constant chain-NPU.
- wall 518 → ~389-399 (-23..25%), 1.93 → ~2.5 fps, bit-exact, accuracy PASS.

### TRUE remaining bottleneck = NPU dispatch (222 ms incl 63 ms chain)
- 98 launches each ~1.9 ms (wrapped) / 4.4 ms (chain) of mostly on-device dispatch floor.
- Real next lever = DISPATCH COUNT reduction (ELF rebuild / chain redesign, L1-bounded) or the gc artifact.
- launch_gap is now near-minimal (~50 ms, mostly irreducible BO sync ~19 ms).

## STATUS: GOAL MET (meaningful, verified, production-ready launch_gap/dispatch-plumbing reduction).

## DISPATCH-REDUCTION BRAINSTORM (3 opus agents, read-only) — candidate ideas
1. **Enable packed GEMM** (MDV6_USE_PACKED_GEMM): FULLY IMPLEMENTED, OFF by default. Collapses each
   high-M GEMM's spatial-batch loop (n_batches sequential xrt.run) into ONE run via batch-offset TAPs.
   Saves ~15-25 launches/frame (elan_c1/c4, re4_c4, re15_c1/c4, re12_c1, re6_c4...). Within BO/L1/L2 caps.
   Risk: forces max-live-contexts→30 + ~8 new packed ELFs → context-eviction thrash to validate.
   Already has a lit gate (run_full_model_packed.lit). Effort: LOW (flip default + build + validate).
2. **rnm-epilogue chain fold** (MDV6_USE_RNMCHAIN): EVALUATED 2026-06-20 — correct (-6 launches, accuracy PASS)
   but NET WALL REGRESSION because it forces re6 raster→geo and the geo chain kernel is +52%/call (see RESULT
   section below). KEPT ENV-GATED, default OFF. Re-open only after geo chain reaches raster parity. re4/re8
   extension (re4: 24-drain-BD>16 fix; re8: width-trim TAP) deferred — same geo-slowness blocker would apply.
3. **Port ftconv1 to OCB ELF** (effective_ppc=4): 4→2 launches (-2). L1/prog-mem/L2 all PASS at ppc=4
   (ppc=7→1 launch fails L2 ~700KB>512). Effort: ~½ day, low risk (mirrors aconv3 OCB entry).
4. **c3↔c3 intra-block memtile fusion**: fuse the two sequential 3×3 c3 convs in each rep_elan through
   L2 (analogous to rn3 chain). Attacks Source-B re*_c3 counts (-up to 7). Multi-day kernel work. Big but hard.
5. **In-block 2-chain stitch** via dispatcher (2 RepNCSP stacks → 1 xrt.run): -7 chain dispatches but
   only amortizes ~1.9ms floor (compute stays serial) → modest ~10ms. High effort.
6. **Generalize pair mechanism** to other shared-input 1×1s: -2..6, low confidence (most 1×1 are sequential).

DEAD-ENDS: cross-block chain stitch (data-dependent serialization); packed-pair for rn1 (nbatch=1);
raise tile_m on re6/re8 (already nbatch=1); OCB for GEMM (single-OCB by construction); merge dup c3 calls (data-dep).

TOP PICKS to discuss: #1 packed GEMM (biggest, ready), #2 rnm fold (clean, gated), then #3 ftconv1 (safe small) or #4 c3-fusion (big structural).

## EXECUTION (user chose: packed GEMM first, then rnm fold)
- T4: Dispatched opus agent #4 (packed GEMM enable+validate, background). Baseline to beat: /tmp/mdv6_vec2.json
  (~399ms, 84 wrapped launches; note chain hook now also counts 14 chain → 98 total). Agent saves fresh
  /tmp/mdv6_pre_packed.json. Watching for: n_launches DROP, wall improve, accuracy PASS, context-thrash stability.
- NEXT: rnm-fold (MDV6_USE_RNMCHAIN) for re6 after packed lands — touches test_full_model_mc.py gate L122-125 + rn3_chain_runner.

## RESULT (packed GEMM, agent #4) — correct + launch win, WALL FLAT, env-gated
- -9 launches/frame (98→89), npu_run 222→217 (-4.6), **wall FLAT**, accuracy byte-identical PASS.
- HARD CONSTRAINT FOUND: NPU ~29 hw-context ceiling; model already uses exactly 29. Packing all 5 shapes
  loads a 30th → DRM CREATE_HWCTX err=-2 wedge. Fix: _PACKED_SKIP_SHAPES skips elan_c1 (the only shared/additive
  one) → working set stays 29. Forcing _MERGED_MAX_LIVE_CONTEXTS≤28 "works" only via eviction THRASH (~5x slower).
- Kept ENV-GATED (MDV6_USE_PACKED_GEMM=1, default OFF): wall flat (not launch-bound anymore) + context fragility.
- Fixed prereq: ran kernels/build_kernels.py (.o files were missing). Files: run_tiled_mc.py (~37 lines + 2 gates).
- Also: fixed my chain-hook ZeroDivision in profile_harness.py (added sub_bucket_n for chain.host_write/read).

## RESULT (rnm-epilogue fold, MDV6_USE_RNMCHAIN) — correct + launch win, but WALL REGRESSES, KEEP ENV-GATED
- Measured 2026-06-20, `--profile 8`. Baseline /tmp/mdv6_pre_rnm.json (all flags default/off).
- ACCURACY: PASS, 3/3 runs identical. max_class_diff 0.1423→0.1414, max_vector_diff 0.0312→0.0312 (slightly better,
  well under the 5.0 gate). The fold is numerically correct in the full model.
- LAUNCHES: 98→92 (-6 exactly = the 6 re6 rnm gemms folded). No context wedge (EXIT=0, no CREATE_HWCTX err=-2,
  3/3 stable). switching raster→geo for re6 does NOT push past the ~29-context ceiling.
- THE CATCH — geo chain is materially SLOWER than raster: enabling rnm forces re6 off the default RASTER chain
  onto the GEO chain path. chain.kernel **4404 µs/call (raster) → 6717 µs/call (geo), +52%/call**.
  Over 14 chain calls/frame that is +32 ms/frame, which DWARFS the ~4.5 ms/frame saved by folding gemm_re6_rnm.
- NET: npu_run 221.8→250.9 ms (+13.1% REGRESSION), wall 383→409-423 ms (+7-10% WORSE). Launch count fell but
  the slower kernel made wall worse. The -6 launches are not worth a 52%-slower chain kernel.
- DECISION: kept ENV-GATED, default OFF (MDV6_USE_RNMCHAIN=0 unchanged). No code change to the default path.
  To make this a win, the GEO chain kernel must reach raster-chain parity (~4400 µs/call) first; only then does
  folding the rnm gemm + removing the chain→host→rnm DDR round-trip become net-positive.
- NB: the standalone HW gate conv/test_rn3_chain_rnm_hw.py FAILS on random weights (max 0.148 > tol 0.06), but the
  REAL-model accuracy PASSES comfortably — the microtest tol is tight vs random-bf16 magnitudes; not a model blocker.

## KEY REFRAME: dispatch reduction has hit DIMINISHING WALL RETURNS
- Wall now = pre_post ~116 ( of which ~85 is harness gc.collect artifact; real fwd ~305ms inner timer)
  + npu_run 222 (158 wrapped big-convs + 63 chain compute) + launch_gap ~50.
- launch_gap is near-minimal. Removing launches saves ~0.5ms npu + ~0.5ms gap each → small.
- Real wall levers now: (a) gc/per-frame allocation churn (~85ms, methodology/production), (b) big-conv + chain NPU compute (kernel, not dispatch).
- rnm fold (next, per user) ALSO removes a real chain→host→rnm DDR round-trip (not just launch count) — worth measuring, but expect modest wall.

## Log
- T0: baseline captured (518ms); wired atexit sync report.
- T1: SYNC_PROF decomposition → ~200ms/frame launch_gap is python pack/unpack. Dispatched opus agent #1 to vectorize 3x3.
- T2: Found rn3 chain is profiler-blind, leaking into launch_gap. Queued ideas #2 (GEMM) and #3 (chain pack). Doing read-only scoping while #1 runs.
- T3: Scoped GEMM oc_blocked path — input = input_hwc.reshape(M,IC) sliced by tile_m into total_slots; per-slot python loop + per-slot output reshape. Same vectorization pattern as 3x3. Ready for agent #2.
  Gave user status snapshot. Waiting on agent #1 completion notification (staying off NPU to not slow its profiling).

## ITERATION #5 — measured the last launch_gap idea (resident weights): NOT meaningful
SYNC_PROF on current default (4 frames): all BO host-I/O ≈ 21 ms/frame
(copy_in 6.9 + sync_in 6.2 + copy_out 6.0 + sync_out 1.8). Weight fills are ~half
the fills → static/resident-weight caching ceiling ≈ **6-7 ms/frame (~1.6%)**, below noise.
`wait` (NPU on-device dispatch floor) = 1767 µs/call × 84 ≈ 148 ms — dominates, not host-side.

CONCLUSION: launch_gap/dispatch reduction is EXHAUSTED at meaningful scale.
- launch_gap host plumbing already cut 158→50 ms (vectorization, committed 663caa40d).
- Remaining launch_gap ≈ 50 ms is ~21 ms irreducible BO I/O + ~30 ms residual python; weight-cache caps ~7 ms.
- Dispatch COUNT reduction proven wall-flat (packed GEMM -9 = flat; each marginal small launch ~0.5 ms device).
- Big convs + 14-launch chain are data-dependent (can't merge); chain 62 ms is on-device compute/floor.
Real remaining wall levers are OUT OF SCOPE for "launch_gap/dispatch": (a) per-frame gc/alloc churn
(~96 ms pre_post artifact), (b) conv/chain kernel compute (BFP576/mmul). Would need goal re-scope.

## FUSION/STITCHING TRACK (2026-06-21)
### M0 — on-device GELAN concat: PROVEN ✓ (verified by orchestrator)
- concat_only BIT_EXACT (0/32768, max_diff 0.0); e2e concat→conv4 byte-identical to host path.
- Construct: single strided gather rt.fill (sources stacked, interleaved via source strides). NOT 4 dest-scatter DMAs (fill dest is linear).
- Files: conv/aie2_concat_proof.py, conv/test_concat_proof_hw.py.
- Trap: iron.tensor(u16) silently promotes to u32 → pin dtype=np.uint16 (memory'd).

### Option A — producer→consumer stitch: PROVEN GO ✓ (verified)
- PoC-1: 1 ELF / 1 hw-context, device-resident intermediate, BIT-EXACT (0/65536) vs 2-dispatch baseline.
- Construct: build_merged(chain_links=[(0,2,1,0)]) = alias producer OUTPUT arg → consumer INPUT arg (new; prior chain_links were shared-INPUT). Needs producer.out == consumer.in MLIR type.
- Measured: PDI-swap floor ~39 µs/swap (slope) vs ~505 µs host dispatch → ~13x cheaper + frees 1 of 29 contexts. THE dispatch-consolidation win.
- Files: conv/build_stitch_poc.py, conv/test_stitch_poc_hw.py.
- PoC-2 (conv0→conv1): NO-GO short-term. Blocked by on-device OVERLAPPING-WINDOW halo gather (split/join require non-overlapping; can't share halo rows). ~3-5 days for a reformat kernel. This is THE keystone blocker for all tile-blocked-producer → 3×3-consumer stitching.

### Option B — fuse re8 block: IN PROGRESS (agent a4067365089fe1599)
- Milestones: B1 fuse concat→c4 (reformat-free, should land) → B2 chain→c3 resident seam (chain already emits PAD-padded HWC = halo'd layout, may sidestep the reformat blocker) → B3 full block ≤2 launches/1 context.
- Gated behind MDV6_FUSE_RE8 (default OFF); default path must stay bit-exact.
- Expected payoff: re8 ~5-7 ms wall (modest; launch_gap at floor). Strategic value: proves the fusion+stitch template; re6/re4 are the bigger but BD/memtile-limited targets.

### Option B — B1 DONE (verified), B2 BLOCKED — the binding wall is the context ceiling
- B1: on-device concat→c4 for re8/re21/spp9, gated MDV6_FUSE_RE8 (default OFF). Bit-exact (0/102400, max_class 0.1423 PASS), context-neutral 29→29, NO wedge. Wall FLAT (388→387), launches 98→98.
  - Context-neutral trick: the (24,512,256,kb32,p1) c4 GEMM shape is shared by exactly 3 fusible sites → fused ELF DISPLACES the c4-only ELF one-for-one. Naive additive fusion wedges (DRM CREATE_HWCTX err=-2); LRU-capping → ~1.5s thrash.
  - Files: conv/fuse_re8_runner.py, conv/build_fuse_re8_merged.py, test_full_model_mc.py (re8+spp9 fused paths, gated).
- B2 (chain→c3 resident): NO-GO as scoped. Blocked by the SAME wall: model is at 29/29 contexts, ZERO headroom. A stitched chain→c3 ELF is additive (no clean same-shape displacement like c4 had) → wedge/thrash. Also launch-neutral unless it removes a host dispatch.

### STRATEGIC WALL: 29-context ceiling, zero headroom = the binding constraint for the WHOLE fusion/stitch program
- Dispatch-consolidation win (PoC-1 stitch, ~39µs swap vs 505µs dispatch) is proven, but requires merging ops into FEWER ELFs.
- Additive fusion (new ELF alongside existing) WEDGES at 29/29. Only context-neutral (B1 same-shape displacement) or context-NEGATIVE (PoC-1 producer→consumer stitch merges 2 ELFs→1 AND removes a dispatch) levers work.
- OPEN QUESTION (highest leverage): is the ~29 ceiling a HARD firmware limit or a configurable driver/firmware param? If raisable → additive fusion path opens. If fixed → only context-negative consolidation works.

### B2 re-tasked as DISPATCHER MERGE — context-negative PROVEN (corrects B1's separate-ELF artifact)
- **B2a (verified):** 2 back-to-back re8-shape GEMMs (1×1 IC128→OC128, tile_m=44) merged into ONE ELF via
  build_merged + chain_links=[(0,2,1,0)] (producer.out→consumer.in alias). **hw_context 2→1, host dispatch 2→1,
  BIT-EXACT (0/180224), wall -263µs.** 1 ELF = 1 context with 2 aiex.configure sub-devices. Intermediate device-resident.
  → Confirms: merged-ELF dispatcher = 1 context/ELF regardless of sub-device count; merging REDUCES contexts.
- **B1 re-confirmed context-NEUTRAL** (same-shape displacement, peak stays 29) — the artifact the user flagged.
- **Caveat:** NO native re8 seam is drop-in mergeable — every GEMM↔next is GEMM↔3×3 (halo mismatch) OR split/concat-separated,
  and the 4 re8 GEMMs have different tile_m (20/104/44/24). B2a used the real rnm shape with GEMMs co-pinned to tile_m=44
  (numerically identical; tile_m only changes tiling) → proves the mechanism on a production shape, constructed seam.
- **B2b (de-risked, key unknown ANSWERED YES):** the rn3 chain's IRON aie.device **compiles cleanly through the merged
  full-elf path** (rc=0, 435KB ELF). xclbin-vs-elf is NOT a compile blocker. Chain output is PAD-padded HWC = the halo'd
  layout c3 reads → chain→c3 chain_link is layout-compatible. Remaining = plumbing: (1) stage the BFP kernel .o's in
  build_merged, (2) thread the multi-iter ping-pong A/B inout image so final-iter output is chain_link'd to c3,
  (3) migrate chain dispatch ResidentXCLBinRunner → xrt.elf+_MERGED_KERNELS.
- **B2c (full re8 block in 1 ELF/1 context): GO, ~1-2 days, de-risked.** Strategic prize is NOT the ~5ms re8 wall win —
  it's the END-TO-END template (chain-under-dispatcher + co-pinned GEMMs + M0 on-device concat) that, applied model-wide,
  drops contexts 29→~5-8 and collapses ~80 host dispatches → a handful with ~39µs swaps = the projected ~40ms launch_gap win.
- Files: conv/build_re8_gemm_stitch.py, conv/test_re8_gemm_stitch_hw.py, run_tiled_mc.py (+26 gated MDV6_CTX_TRACE: live_merged_context_count()).

### B2c-1 — chain-as-merged-1-context DONE; chain→c3 BLOCKED on the recurring keystone (halo gather)
- ✅ **Chain → merged xrt.elf, 1 hw_context, bit-exact** (max_diff 0.0234 vs ResidentXCLBin, BFP tol). Migrated off
  ResidentXCLBinRunner → now a budget-shareable merged context. Plumbing all solved: BFP .o's compile inline (non-issue),
  ping-pong inout composes, dispatch migrated to xrt.elf/_MERGED_KERNELS. Files: conv/build_re8_chain_merged.py, conv/test_re8_chain_merged_hw.py.
- ❌ **chain→c3 device-resident chain_link: BLOCKED.** chain out = memref<50176> (28×28×64 PAD-HWC) vs c3 in =
  memref<204800> (im2col patch-packed, **4.08× larger** — pixels replicated across overlapping 3×3 halo windows).
  Raw BO alias can't bridge; needs on-device im2col reformat. **This is the SAME overlapping-window halo gather that blocked
  conv0→conv1 (PoC-2).** It is now confirmed THE single recurring keystone gap for all 3×3-consumer fusion.

### THE KEYSTONE (singular, de-risked): on-device halo gather — padded-HWC → 3×3 conv input
- Every on-device fusion across a 3×3 consumer (chain→c3, conv0→conv1, full re8 block, model-wide template) is blocked
  by ONE missing primitive: a 3×3 conv whose input fill TAP gathers halo'd windows from a contiguous padded-HWC buffer,
  instead of consuming host-im2col'd patches.
- **De-risked:** the chain's own conv2res kernel ALREADY reads padded-HWC with halo gather → the read pattern is proven
  expressible in IRON. Path (b): a 3×3 multicore-conv generator whose rt.fill gathers 10×10 halo patches from the shared
  padded image. Build it M0-style (standalone bit-exact), then chain→c3 + full re8 block + conv0→conv1 all become mechanical.
- Proven so far: M0 concat ✓, PoC-1 producer→consumer stitch ✓, B2a context-negative GEMM merge ✓, chain-merged-1-context ✓.
  Remaining single blocker = the halo-gather primitive.

### KEYSTONE LANDED ✓ — on-device halo-gather 3×3 conv from padded-HWC (the unblock)
- Standalone: window gather BYTE-IDENTICAL to host im2col (all tiles); conv PASS max_diff 0.035 (BFP tol).
- chain→c3 seam: merged chain ELF padded-HWC output fed VERBATIM → halo-conv, NO im2col bridge, PASS max_diff 0.043.
- THE INSIGHT: overlapping windows ARE expressible as a plain rt.fill TAP
  `sizes=[1,WIN,WIN,ic], strides=[0,IMG_W*ic,ic,1]`, tile origins 8 apart, windows 10 wide → overlapping source rows.
  The non-overlap restriction only ever applied to `split`, NOT to fill TAPs. That's what unlocked it.
- Files: kernels/halo_conv3x3_bfp.py, conv/aie2_halo_conv.py, conv/test_halo_conv_hw.py, conv/test_halo_conv_seam_hw.py.
- The 4.08× im2col mismatch (50176 padded-HWC vs 204800 patch-packed) that blocked conv0→conv1, chain→c3, full re8 — GONE.
- Remaining to assemble chain→halo_c3→…→concat→c4 in ONE ELF/context: low-med plumbing — per-oc-block wt streaming
  (lift from chain), bake 1px origin offset into chain drain, place halo_c3 as a chain_link consumer sub-device.

## FUSION PROGRAM — all primitives now PROVEN:
M0 concat ✓ · PoC-1 producer→consumer stitch ✓ · B2a context-NEGATIVE GEMM merge ✓ · chain-as-1-context xrt.elf ✓ ·
B1 concat→c4 ✓ · KEYSTONE halo-gather 3×3 from padded-HWC ✓. → full re8 block in 1 ELF/1 context is now mechanical.

### B2c3-1 — chain→halo_c3 device-resident seam in ONE ELF/ctx ✓ (verified)
- hw_context 2→1, chain_link (0,2,1,0) aliases halo.in←chain.out (memref<50176> device-resident),
  shift=PAD-1 baked into halo TAP (no host im2col, no host shift). Bit-exact 0.043 (BFP tol).
- Files: conv/build_re8_chain_halo_merged.py, conv/test_re8_chain_halo_merged_hw.py, conv/test_halo_conv_stream_hw.py.
- Per-oc-block weight streaming mechanism proven at OC≤64 (halo_conv3x3_bfp_ocb + stream_oc).

### TWO remaining blockers to the FULL re8 model block (both substantial, HW-iteration risk):
1. **OC=128 overflows L1** — the f32 C accumulator (16 ocb × 8 × 64 × 4B = 32KB) + win(25KB) + wt(36KB) > 64KB.
   Fix = drain C per oc-block-PAIR (PAIR_C 4KB) → per-pair output-FIFO restructure (split/join topology, deadlock risk).
   mc_re8_c3 is OC=128, so this gates the REAL seam.
2. **Model seam is rnm(1×1,128ch)→c3, not chain(64ch)→c3** — B2c3-1's seam used the chain's raw 64ch output (constructed).
   The real model fusion needs rnm-GEMM + concat fused INTO the chain to emit 128ch padded-HWC.

### INFLECTION POINT: all feasibility answered YES; full model block = substantial remaining engineering, modest per-block payoff
- Proven library: M0 concat, PoC-1 stitch, B2a ctx-neg merge, B1 concat→c4, chain-1-ctx, KEYSTONE halo-gather, B2c3-1 seam.
- re8 full-block wall payoff ~5ms (modest); strategic payoff = context headroom → model-wide ~40ms (multi-week, per-block-class).
- re6/re4 fit L1 more easily (OC overflow is re8/re21-specific) — but same rnm-into-chain subtlety.

### OC=128 C-drain DONE ✓ (verified) — keystone halo-conv at the real mc_re8_c3 shape
- IC=128→OC=128 fits L1 (~49KB: stack 4 + wt-slot 18 + window 25 + C 2KB) via per-SINGLE-oc-block C-drain
  (new kernel halo_conv3x3_bfp_ocb1, BLK_UNIT=1). Per-pair (BLK_UNIT=2) overflowed at IC=128. Bit-exact 0.052 (BFP tol).
- Output: single buffer, collapsed weight fill + contiguous linear drain (unit-major), host deinterleave_stream_out (free perm).
  NO split/join restructure → no deadlock. Window gather still byte-identical.
- Regression: OC=32/64 PASS. (IC=128/OC=64 pair mode retired — use stream_oc="block" for IC=128.)
- Files: kernels/halo_conv3x3_bfp.py (+ocb1), conv/aie2_halo_conv.py (stream_oc block/pair), conv/test_halo_conv_oc128_hw.py (new).
- B2c3-1 at OC=128: GO ~0.5d (pass stream_oc="block" + apply deinterleave at merged runner output).

### REMAINING to full re8 model block: rnm→halo_c3 seam (the real model seam)
- Model seam = rnm(1×1 over concat(bottleneck64, x2)) → c3(3×3, 128→128). Chain emits 64ch (bottleneck), NOT c3's 128ch input.
- Need: concat (M0✓) → rnm 1×1 GEMM (✓) → emit PAD-padded HWC output → halo_c3 OC=128 (✓ now) reads it device-resident.
- Composes proven pieces + M0-style padded-output placement (place 20×20×128 into 24×24×128 padded buffer).

### rnm→c3 SEAM DONE ✓ (verified) — both re8 blockers now CLEARED
- S1: rnm 1×1 GEMM (128→128) drains PAD(2)-padded HWC (24/28-img), de-pad bit-exact 0.023, border all-zero,
  seam→halo_c3 standalone 0.019. Drain TAP offset=((PAD+r0)*IMG+PAD)*oc, out_d0≤1023 factored. Files: conv/aie2_gemm_pad_out.py, test_gemm_pad_out_hw.py.
- S2: rnm→halo_c3 in ONE merged ELF, chain_links=[(0,2,1,0)] (memref<100352> device-resident), OC=128 stream_oc=block,
  shift=PAD-1 baked. **hw_context 2→1**, bit-exact 0.037 at the REAL 128→128 model seam. Files: conv/build_rnm_halo_merged.py, test_rnm_halo_merged_hw.py.
- S3 assess: full block GO, no blocker. 3 seams proven (chain→rnm=B2a, rnm→halo_c3=this, concat→c4=M0/B1).
  Remaining = layout-bridging between hops (tile↔row↔padded), ~3 milestones, NOT topology/L1/deadlock.

### FULL BLOCK INTEGRATION (A1/A2/A4) — rnm→c3 fused IN-MODEL, all 4 re8 hops ✓ (verified)
- FUSE ON: launches 98→94 (-4), hw_context peak 29→28 (-1), wall ~401ms (NEUTRAL), accuracy 0.1420 PASS.
- Default OFF: bit-exact 0.1423 PASS (untouched). Standalone rnm→c3 vs model: max_diff 0.0137.
- Construct: rnm GEMM(pad-out) + halo_c3(OC=128) in 1 ELF (chain_links (0,2,1,0), device-resident seam), ×4 hops.
- Files: conv/rnm_halo_runner.py, conv/test_rnm_c3_model_hw.py, test_full_model_mc.py (gated re8 path).

### A3 (full single-ELF block) NOT reached — chain→rnm layout-incompatible
- chain drains 8×8 TILES into padded HWC; rnm GEMM wants per-core 20px ROWS + x2 channel-interleaved.
  No single TAP bridges tiles→rows+x2 → chain→rnm stays a host repack (the model's existing torch.cat).
- Context only -1 (not ~5): chain is on a SEPARATE ResidentXCLBinRunner context (not in the merged pool CTX_TRACE counts);
  peak now bound by the model TAIL (re12/15/18 + aconv heads), not re8.
- BN+SiLU gap: halo_conv does RAW conv; c3 BN-scale folded into weights (device), BN-bias+SiLU host-side. Tiny BFP perturbation (0.1423→0.1420), not bit-identical.

### HONEST CONCLUSION of the fusion deep-push
- Every primitive proven + the rnm→c3 seam integrated IN-MODEL bit-exact. Mechanism fully de-risked.
- BUT wall is NEUTRAL at re8 scale (-4 launches below noise) — re-confirms dispatch/context reduction doesn't move the wall here.
- Model-wide ~40ms would need: re6/re4 fusion + a chain→rnm tiles→rows reformat + BN/SiLU-in-kernel + displacing tail ELFs.
  Multi-week; payoff still bounded by dispatch-floor economics (heavy compute stays; launch_gap already at floor).

### chain→rnm→c3 FULL HOP in ONE ELF ✓ (verified, agent-committed C1/C2/C3)
- The LAST seam (chain→rnm) solved: x2 host-padded to the SAME 28×28 PAD(2) layout as the chain, stacked into a
  widened chain A/B BO (stack_x2_ch=HALF_ELEMS=50176 → [chain64 | x2_64]); ONE gather TAP does de-pad+concat
  (sizes=[cc,20,2,64], strides=[IMG*ic2, ic2, HALF_ELEMS, 1]) → rnm GEMM. chain drains a CONTIGUOUS padded image (verified).
- C1 depad+concat→rnm 0.0195 / C2 chain→rnm 2→1 / C3 chain→rnm→halo_c3 FULL HOP **3→1 ctx**, 0.045. All HW PASS.
- Files: conv/aie2_depad_concat_gemm.py, conv/build_chain_rnm_merged.py, conv/build_chain_rnm_halo_merged.py + tests;
  conv/aie2_rn3_chain_geo.py (gated stack_x2_ch, default 0 = unchanged). Commits 828ee2498/6278ef0db/64d27e573.
- A3 (full single-ELF re8 hop) ACHIEVED. Remaining: wire run_chain_rnm_c3 into run_re_mc (replace run_rnm_c3 for re8)
  → folds the chain's separate ResidentXCLBin dispatch into the merged ELF; measure frame launch/ctx delta.

### FULL-HOP WIRED IN-MODEL — STRUCTURAL WIN, WALL REGRESSION (decisive negative result)
- chain→rnm→c3 fused, all 4 re8 hops, behind MDV6_FUSE_RE8 + MDV6_FUSE_RE8_FULL (default ON when FUSE_RE8 set).
- launches 98→**90** (-8), hw_context 29→**28**, BUT wall 403→**439ms (+36ms REGRESSION)**, accuracy 0.142→0.184 (PASS <5.0 but degraded).
- WHY regression: the 3-iter chain now runs SYNCHRONOUSLY inside the merged dispatch on the critical path, LOSING the
  tuned ResidentXCLBin weight-replay path + larger BO fills (widened 100352 stacked A/B BOs). Structural wins don't convert to wall.
- Default OFF path bit-exact (0.1414 PASS) — UNTOUCHED, safe.
- Files: conv/chain_rnm_c3_runner.py, test_full_model_mc.py (gated re8 full-hop path). Commit pending.

### DECISIVE CONCLUSION — STOP the fusion deep push
- rnm→c3 fusion: wall-NEUTRAL (-4 launches). full chain→rnm→c3 fusion: wall-REGRESSION (+36ms, accuracy degraded).
- The fusion program is FULLY de-risked + proven in-model, but it does NOT help the wall and the full hop HURTS it.
- Propagating to re6/re4 would replicate the regression. The wall is bounded by chain/conv COMPUTE + the gc artifact, not dispatch.
- Banked value: complete gated primitive library + proven full-hop template (default OFF). The shipped win remains the
  default-on vectorization (-26% wall, 518→383ms).

### Accuracy fix: BN(scale+bias)+SiLU IN-KERNEL on f32 accumulator (halo_conv3x3_bfp)
- CORRECTION: the earlier "#2 bf16-readback-before-SiLU" was a MISREAD — halo output is f32, SiLU ran on f32 (host).
  Real culprit was #1: BN-scale folded into conv weights BEFORE BFP576 quantization (shared block exponent shift).
- Fix: adopt the chain's _store_bn_silu_4x8 (no residual) in halo_conv3x3_bfp/_ocb/_ocb1; pass bn_w/bn_b in the chain
  weight-slot layout (+2*oc); STOP folding scale into weights; REMOVE host bias+SiLU epilogue. Kept f32 tiled-C transport
  so untile/drain plumbing unchanged. (HW gotcha: streamed unit must be 64-elem-aligned or DMA mis-delivers → all-zeros.)
- Results: default OFF 0.1423 (bit-exact, untouched). rnm→c3-only 0.1420→0.1425 (≈baseline, was already there).
  FULL chain→rnm→c3 0.184→0.1818 (tiny). Standalone in-kernel BN+SiLU vs faithful f32 ref: BFP tol (mean ~5-7e-3).
- HONEST: the kernel is now FAITHFUL (scale-fold error gone, host touch removed) — a correctness win — but the FULL-path
  accuracy gap is NOT the BN handling; it's dominated by the inherent **BFP576-vs-emulated-mmul** difference (fused c3 uses
  the BFP hardware matmul; baseline mc_re8_c3 uses emulated mmul<4,8,8>) + the in-merged chain BFP nondeterminism. Those are
  different IMPLEMENTATIONS of the same algorithm — irreducible without matching the baseline's matmul engine.

### DECISIVE EXPERIMENT — static-weight residency in merged path: PREMISE WAS WRONG
- Implemented: per-hop resident weight BOs keyed by stable wkey (4 hops × 4 roles), skip-refill after first fill.
  Gated MDV6_FUSE_RE8_RESIDENT (default ON). Only conv/chain_rnm_c3_runner.py changed; default path untouched.
- VERIFIED mechanism: SYNC_PROF host→device fills 768→720 = exactly 48 fewer weight syncs (4 BOs × 4 hops × 3/4 frames). Works as designed.
- DECISIVE FINDING: the +36ms "regression" was MISATTRIBUTED to weight re-upload. XRT BO→device sync is fast bulk DMA:
  the ENTIRE per-frame weight re-upload tax is only **~0.23ms** (micro-bench: 4 hops × 0.058ms). Residency removes it but it's negligible.
- Machine SEVERELY CONTENDED (load ~15-17, 9 users): wall swung 552→1030ms, npu_run 213→348ms for IDENTICAL configs.
  → the original +36ms regression was likely CONTENTION NOISE, not a real fusion cost. In matched windows the agent saw
  npu_run trend LOWER for fused (~340 vs ~378 FUSE-OFF) — i.e. fusion may be ~neutral-or-slightly-positive on npu_run.
- VERDICT: INCONCLUSIVE on wall (unmeasurable under contention). Residency is correct/free/keep, but NOT the lever.
  The real fused costs are chain-kernel compute (~5.3ms/call) + pre_post, not host weight DMA. Need a QUIET machine to get the true A/B.

### CLEAN A/B (load ~1.0, rogue process gone) — VERDICT OVERTURNED: full-hop fusion is a REAL ~10% WIN
Measured back-to-back at load 0.98-1.14 (genuinely uncontended), --profile 6 each:
| config | inner-fwd (TRUE) | npu_run | launch_gap | pre_post | wall(profiler) | launches | acc |
|--|--|--|--|--|--|--|--|
| OFF    | ~300 ms | 222.0 | 53.8 | 103.7 | 392 | 98 | 0.1423 |
| rnm→c3 | ~300 ms | 219.1 | 56.8 | 133.3 | 420 | 94 | 0.1425 |
| FULL   | **~270 ms** | **198.9** | 50.0 | 180.8 | 441 | 90 | 0.1818 |

- **Fusion mechanism CONFIRMED working**: full hop npu_run 222→199 (−23ms); folding the CHAIN in (PDI-swap collapse) is the lever
  (rnm-only barely moves npu_run since chain stays a separate dispatch).
- **TRUE per-frame latency (inner Total-forward-pass timer, EXCLUDES harness gc.collect): 300→270 ms = −30ms (−10%), ~3.3→3.7 fps.**
  Consistent across all 5 warm frames at load 1.0. = npu_run(−23) + launch_gap(−4).
- **Profiler "wall +49ms" is the gc.collect ARTIFACT**: pre_post 104→181 because the fused path allocates more per-frame host
  objects → harness per-frame gc.collect inflates. NOT inference latency; a streaming pipeline doesn't gc every frame.
- CORRECTION: my earlier "+36ms regression / dead track" calls were WRONG — (1) contention noise (rogue dnsmasq-python @728%),
  (2) wrong metric (gc-contaminated profiler-wall vs true inner latency). Clean+right-metric = fusion WINS ~10%.

### TRACK IS ALIVE. Next levers (now evidence-backed):
1. Generalize chain-fold to re6/re4 (the big c3 convs) — each block's chain-fold should compound the npu_run reduction.
2. Reduce the fused path's per-frame host allocation churn (shrinks the gc artifact + helps production jitter).

### re6 GENERALIZED — COMPOUNDS THE WIN (verified clean, load 1.75)
Inner Total-forward-pass timer (true latency, excludes gc artifact), --profile 6:
| config | inner warm | npu_run | launches | acc |
|--|--|--|--|--|
| baseline (FUSE off) | ~0.297s | 222 | 98 | 0.1423 |
| re8-only            | ~0.272s | 199 | 90 | 0.1828 |
| **re8 + re6**       | **~0.251s** | **179** | **78** | 0.1848 PASS |
- re8+re6 = **−46ms (−15.5%) vs baseline, ~3.4→4.0 fps.** npu_run 222→179 (−43ms), launches 98→78 (−20).
- The geo-vs-raster re6 chain penalty did NOT cancel the dispatch-collapse saving across 6 re6 hops → net positive, compounds re8.
- R1 standalone re6 seam bit-exact 0.043 (OC=96 fits L1 easily). Solved: re6 chain emits TALL non-square padded image
  (52 rows, WORKER_TILES=(2,2,2)>5 valid) → chain_img_h param; 40 rows >32 cores → rows_per_core=2 (merged column TAP).
- Per-geo registry in run_chain_rnm_c3 (_GEO_BY_SHAPE: (20,128)→re8, (40,96)→re6). Gates MDV6_FUSE_RE8 + MDV6_FUSE_RE6 (default OFF).
- Default path bit-exact 0.1423. Files: aie2_depad_concat_gemm.py, build_chain_rnm_halo_merged.py, chain_rnm_c3_runner.py,
  rnm_halo_runner.py, test_chain_rnm_halo_merged_hw.py, test_full_model_mc.py.

### Remaining for model-wide: re4 (80×80, mc_re4_c3 18ms — biggest, but shim-BD risk) + accuracy budget (0.1848 vs 0.1423) + gc/alloc churn (so profiler-wall also reflects the win for default-on ship).

### re4 DOES NOT FOLD — compute-tile wall (not the shim-BD risk we flagged)
- Blocker: aie2_halo_conv.py places 1 worker per 8×8 output tile → re4 80×80 = 10×10 = **100 workers vs 32 compute cores (3.1× overflow)**.
  Merged re4 ELF fails aiecc placement ("no available compute tiles"). Verified: halo conv ALONE at re4 fails the same way.
  re8=9 workers, re6=25 — both fit; re4=100 doesn't. The chain (20 tiles) + dcg (20 tiles) sub-devices fit fine; only halo overflows.
- The documented "re4 TPC=12 → 24 drain BDs > 16 shim" risk was NOT the blocker (chain runs rnm=0 here, dcg picks rpc=4 → fits shim).
  The real wall is upstream: the halo generator's flat one-tile-per-core model.
- To fold re4: rewrite aie2_halo_conv.py for **tiles-per-core batching** (each worker does ≥4 output tiles, like the chain's
  WORKER_TILES/raster) — substantial generator rewrite with the overlapping-window gather at higher tpc, NOT incremental.
- Secondary fix (committed): dcg rows_per_core now requires gbound%rpc==0 (re4 picked rpc=3 with 80%3≠0 → assert). re8(rpc1)/re6(rpc2) byte-identical, re6 re-verified bit-exact 0.043.

## FUSION TRACK STATE: re8+re6 = verified −15.5% latency (gated). re4 bounded by halo 1-tile/core model.
## To extend: halo tiles-per-core rewrite (unlocks re4 + makes re6/re8 leaner). To ship re8+re6 default-on: accuracy budget (0.1848) + gc/alloc churn.

### SHIPPABILITY (agent A) — gc.freeze() is the lever; fused path now WINS on the wall too
- Diagnosis: the fused-path +77ms pre_post is gc.collect(), NOT per-frame churn. The merged-ELF path holds a ~758,000-object
  PERSISTENT live set (per-geo MLIR build artifacts + xrt/BO/weight caches); the harness gc.collect() re-traverses it every
  frame → ~226ms. Per-frame garbage delta is only ~23 objects → the cost is the live-set SIZE, not churn.
- Fix: gc.collect()+gc.freeze() after fused ELF build + resident weights → moves the set to gc's permanent gen → per-frame
  gc 226ms → **0.1ms**. Default ON (MDV6_FUSE_NO_GC_FREEZE=1 to A/B). Plus host buffer reuse (stacked image, untile via
  np.take(out=), 4-deep bf16 output ring since x3/x4 coexist for the c4 concat) — cut untile/image churn ~2×.
- Result (same load window): RE8+RE6 gc 226→0.1, pre_post 286→13, **wall 557→278 (now BELOW baseline 451)**, inner 0.294→0.276.
  The profiler wall now REFLECTS the win (was a phantom regression). Production jitter: per-frame gc 170-260ms swings → flat 0.1ms.
- Accuracy intact: default 0.1423, fused 0.1848 (P3 0.131 / P4 0.167 / P5 0.185; vector ≤0.0312). Verified.
- NOTE: gc.freeze() is independently valuable — the ~99-226ms/frame gc artifact is the "gc lever" flagged all session;
  could apply model-wide (baseline too). Files: chain_rnm_c3_runner.py, rnm_halo_runner.py, test_full_model_mc.py (opt-in MDV6_FUSE_TIMING).

### re4 HALO TILES-PER-CORE (agent B) — PLACEMENT SOLVED, prototype GO
- New files (nothing existing touched): conv/aie2_halo_conv_mt.py, kernels/halo_conv3x3_bfp_mt.py, conv/test_halo_conv_mt_hw.py.
- Each worker does `tpc` tiles (raster row-major, lifted from rn3_chain_raster); ONE window+C resident at a time (L1 indep of tpc).
- HW PROOF: small re8 (20×20 OC32 tpc2, 8 workers) bit-exact; **re4 (80×80 GRID10 OC32 tpc4 = 28 workers ≤ 32) PLACES + bit-exact
  (max_diff 0.156, 2/204800 BFP tail)**. The 100-workers>32-cores wall is GONE.
- OC=64/128 L1-overflow (full-OC wt slot 72KB/288KB) = the SAME issue ocb1 streaming already solves (orthogonal to tpc) — confirmed
  by the OC=64 test failing on buffer-alloc (got PAST placement). Effort to a real re4 fold ~0.5-1 day: port ocb1 into the mt
  core_fn + drain-layout match to dcg + plumb tpc. Bonus: re6/re8 get fewer workers at tpc>1.

### CLEAN FINAL A/B (load ~2.0, rogue gone) — RE8+RE6 + gc.freeze
| metric | baseline OFF | RE8+RE6+freeze |
|--|--|--|
| wall    | 403 ms | **246 ms (−39%)** |
| fps     | 2.48 | **4.07** |
| inner   | ~0.30s | ~0.245s |
| npu_run | 221.7 | 179.6 |
| pre_post| 111.8 (pre 85) | 9.6 (pre 4) |
| launches| 98 | 78 |
| acc     | 0.1423 | 0.1848 PASS |
- HONEST DECOMPOSITION of the −157ms wall: ~−55ms = FUSION (real latency; npu_run 222→180, inner 0.30→0.245 = −18%);
  ~−102ms = gc.freeze killing the per-frame gc.collect artifact (baseline pre_post 112→ fused 9.6). The gc.freeze portion
  is the harness gc artifact — it could be applied to the baseline too (would drop baseline ~300ms). Fair fusion-only ≈ −18%.
- Shipping RE8+RE6 today yields the full 403→246 because gc.freeze ships with it. Cumulative vs ORIGINAL 518ms: 518→246 = −52% (gated).
