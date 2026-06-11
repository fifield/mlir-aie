# Weight-Replay Handoff — MDV6 rn3 chain

Status as of commit `be94038aa`. Production is **unaffected** (~501 ms / PASS);
all replay is behind `MDV6_WTREPLAY` (default off). This doc lets a fresh
session resume the weight-replay optimization without re-deriving anything.

## The goal

The rn3 bottleneck chains re-stream the same weight slots from DDR every
round (TPR rounds × n_iters iterations). Weight DMA is the chain's pace
setter: ~83 KB/col/round at shim BW ~1.7 GB/s ≈ half the per-iter wall.
**Replay** = fill weights into the memtile once, re-emit (replay) them from
L2 at ~14 GB/s (measured, 8× shim) instead of re-reading DDR. Expected
model win ~40+ ms/frame across the chain stacks.

## What is PROVEN (committed, on hardware)

- **Memtile replay bandwidth**: `conv/test_wt_replay_micro_hw.py` — static
  memtile MM2S `repeat_count` replays a resident 83 KB stream at 13.98 GB/s.
- **Column topology**: `conv/test_wt_replay_col_hw.py` — per-iter shim refill
  + lock-paced MM2S broadcast to 4 cores arming their own S2MM, PASS 10.5
  GB/s/core.
- **i1 end-to-end in the chain** (`26c7951d4`): re4w 1-iter, numerics PASS
  (max 0.000977 vs torch), 2.5–3.9 ms vs column 3.7–9.1 (≈2.3×). Real convs.
- **i3 numerically correct** (`be94038aa`): per-iteration host refill of
  `wt_src` content at the drain barrier + credit reset. max 0.002930 =
  identical to non-replay raster. **This overturned the earlier "i3
  architecturally infeasible" claim** — the offset advance is per-iteration
  (coarse), not per-slot, and the runtime sequence's per-round
  `drain(wait=True)` is the natural sync barrier.

## What is UNSOLVED

**Performant i3+ in the credit design.** Three attempts, three failures:
1. Per-iter refill, fills not freed → 21 un-freed shim BD tasks exhaust the
   ~16-BD pool → host polls → **2128 ms** (but numerics PASS).
2. Added `issue_token`+await+free → BDs recycle but credit-reset-before-fill
   replays stale `wt_src` → **numerics FAIL** (max 3.16) and still ~750 ms.
3. Reordered to fill-then-reset → **wedges** on a clean device (the
   host `dma_await_task` on an injected mid-sequence fill deadlocks against
   the credit/drain machinery).

Conclusion: the credit / core-self-arm framing is the wrong architecture for
multi-iter. It exists only because the cores self-arm their S2MM (which then
needs autonomous pacing to avoid parking).

## The CLEAN PATH (recommended, untried)

User's reframing, validated by reasoning: **the replay runtime sequence
should be identical to the fresh-from-DDR sequence, except the per-round
"stream weights" task becomes a memtile-DMA task (memtile BD register writes)
instead of a shim-DMA task (shim BD writes), in the same positions.**

Key refinement that collapses the complexity: **leave the cores as plain
iron ObjectFifo consumers** (memtile→core leg stays iron-managed, cores
`acquire` weights — no MMIO self-arm). Only the **producer leg** changes:
instead of shim→memtile fill per round, weights are resident in a memtile
buffer and the host re-triggers the memtile MM2S producer per round.

This eliminates, in one move:
- core S2MM self-arming (`chain_wt_arm` / `program_dma_and_start`)
- the **Core_Processor_Bus enable** requirement (only needed because cores
  did MMIO) — see gotcha below
- the credit / echo / join-splice / absorber lock machinery
- the parking hazard (host explicitly triggers each emission, as the working
  baseline already does for the shim)

### Open question to resolve first
Iron's `ObjectFifo` (python wrapper, `aie/iron/dataflow/objectfifo.py`) has
**no `repeat_count`/`initValues` on the producer** — so you cannot express
"memtile-resident re-emitting producer" at the iron level. You will still
hand-build the memtile MM2S driver in raw dialect ops (post-resolve patch,
as the current `_patch_wt_replay` does), BUT keep the memtile→core FIFO as
the iron consumer path. Verify: can the FIFO's memtile producer leg be
pointed at a resident buffer and restarted per round while the core side
stays a normal FIFO consumer? Check whether the FIFO's producer lock can be
host-released (npu_write32) per round to re-fire the memtile MM2S, or whether
a separate memtile MM2S (not the FIFO's) broadcasting into the FIFO's core
buffers is cleaner.

### Sketch
- One DDR→memtile fill of all n_iters weights up front (resident, ~55 KB,
  fits L2 512 KB), OR per-iter content refill if resident-all is awkward.
- memtile→core: iron broadcast FIFO, cores consume via `acquire` (unchanged
  from the working DDR-fresh chain).
- Runtime sequence: per round, host writes memtile MM2S BD offset
  (= current iter base) + pushes memtile start-queue. Same positions the
  shim fill occupied in the fresh chain. n_iters offset advances; within an
  iter the offset is fixed (zero-offset replay, which `repeat_count` does).

## Code map

- `conv/aie2_rn3_chain_geo.py`
  - `rn3_chain_raster_wr(geo, n_iters, compute, static_wt)` — the replay
    generator (raster, 1 tile/core). `compute`: 1=full, 2=arms+dma only,
    0=no compute. `static_wt`: CDO-bake weights vs shim fill.
  - `_patch_wt_replay(...)` — post-resolve patcher: lowers placement+FIFOs
    in-process (PassManager), then injects memtile wt buffer, credit locks,
    MM2S ring (BDs 44–47), join-splice credit BDs, per-launch lock re-init,
    per-iter refill at drain barriers. **This is the credit edifice to
    replace.**
- `conv/rn3_chain_runner.py` — `run_rn3_chain_raster(geo, inp, pairs)`,
  `MDV6_WTREPLAY` env gate; `rep=1` packs n_iters blocks (each `mem_stream`)
  into the WT BO; refill reads `it*mem_stream`.
- `conv/test_rn3_chain_raster_hw.py` — gate vs torch + bench vs column.
  `MDV6_WTREPLAY=1 GEO=re4w N_ITERS=n`.
- `conv/test_wt_iron_stage4_hw.py` — minimal iron-Worker + raw wt micro
  (the bisect that found the processor-bus bug). MODE env: 0/1/2.
- `conv/test_wt_arm_micro_hw.py`, `test_wt_replay_col_hw.py`,
  `test_wt_replay_micro_hw.py` — bare-metal replay micros (all PASS).

## GOTCHAS (each cost hours — all in memory file too)

1. **Core_Processor_Bus**: iron never enables it (reg `0x32038`). ANY core
   MMIO write (DMA self-arm) wedges the whole design until it's set via
   `npu_maskwrite32(0x32038, 1, mask=1, col, row)` per worker at sequence
   start. Root cause of ~a dozen "hangs". Likely worth an upstream issue.
   **The clean path avoids this entirely (no core MMIO).**
2. **BD pool restarts per `memtile_dma` op**: a second memtile_dma op on a
   tile restarts the odd BD pool at 24 and clobbers iron's BDs. Pin replay
   BDs explicitly (44–47).
3. **Queue `repeat_count` over a multi-BD `EndOp` chain is one-shot** — only
   single-BD chains replay. Zero-offset `repeat_count` on one BD DOES work.
4. **Parked circuit beat head-of-line blocks the column**: a memtile→core
   stream that emits before the consumer is ready stalls shared switch
   routes and kills the (separate) patch fills. Never let the producer run
   ahead of the consumer.
5. **stride-0 in any non-leading dim is illegal** on shim fills, shim
   drains, AND memtile MM2S (silent hang). Killed the "round dim stride-0"
   and "iteration step/wrap" variants for i3 (also: iteration advances
   per-execution → iter-major order, wrong for our consumption order).
6. **AIERT BDs need acquire+release lock pairs**; only one release per BD
   (chain a zero-length credit BD if you need a second). BD `next_bd` must
   target end-only blocks for the assigner.
7. **Two-phase compile for post-resolve patching**: run
   `aie-place-tiles,aie-assign-lock-ids,aie-register-objectFifos,
   aie-objectFifo-stateful-transform` in-process via PassManager, THEN patch
   (join BDs / lowered structure only exist after). Locate tiles by parsing
   `aie.tile(col=,row=)` from str (logical tiles lose attrs).
8. **Device wedge discipline**: a failed run wedges the NPU; always run a
   passing raster gate (`GEO=re4w N_ITERS=3`) before re-probing, and check
   `pgrep -f "python3.*test_c[12]_"` for the other session's wedge tests.

## Decision economics

Replay wins only where it's applied to the right iter count. i1: 2.3×.
Model bottleneck stacks are i3–i12. The credit design can't do performant
i3. The producer-leg redesign is the path to i3–i12. If it lands, wire
`MDV6_WTREPLAY` into the model's re4w/re6w raster chains and profile.
Alternative if replay is deprioritized: untouched levers are the gemm-swarm
fusion (~48 launches) and 3×3 wire-bytes (~64 ms over 16 launches:
ftconv1/re4_c3/re6_c3/elan_c3, the chain-style padded-image fill).
