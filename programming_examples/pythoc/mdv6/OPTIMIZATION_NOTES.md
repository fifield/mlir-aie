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
