# Medusa router rebase — 2026-10-07

Work branch: `rebase/medusa-router-20261007`, published as draft PR #376 against
`medusa-dev`. On 2026-10-08, the integration was rebased onto current Medusa
`64a7ab07d11aa0731aac22d3f811f6cc182e1161`. The target branch and original
checkout remain unchanged; the PR remains a draft.

## Compared revisions

- Previous upstream/common base: `b15f3508006988ea0252300a1feac447990ce6ec`
- Original Medusa: `317cb0b96be4f5ae8feb4bb78b438459d8c3c3ea`
- New upstream: `c5e495fb22c96b123f0e000758d6654bcbda43d4`
- Replayed Medusa tip: `a24537c90d542f20de497933ad1585ad4fa18895`

All 370 replay steps completed. There are 369 replayed commits: the obsolete
same-ID routing-golden refresh `423e14a9ac8` became empty. Its four fixtures have
identical non-comment IR before and after that original commit. Assertions are
validated against the combined router rather than the obsolete route choices.
The range-diff is retained in the adjacent validation build directory.

For the 2026-10-08 base update, the exact reviewed tree at
`337e26db6aca27081a14f823c3557268207823da` was consolidated into one integration
commit relative to original Medusa, then rebased onto current Medusa. This avoids
replaying Medusa commits already present on the target branch. Original replayed
history is retained locally at
`backup/medusa-router-before-base-update-20261008`.

Current Medusa adds #362 (AIE4 staging). The rebase applied cleanly. Its only
shared path is `tools/aiecc/IRTransforms.h`: #362 adds staging checks/expansion in
the address-assignment pipeline, while this integration supplies deadlock policy
to the routing pipeline. Both changes are retained. Independent review verified every source blob and
file mode: 314 disjoint integration paths match the reviewed draft, 70 disjoint
#362 paths match current Medusa, and the shared file contains both edits exactly.
Before this report update, the original and rebased deltas had the same stable
patch ID, `4fc22da5c1d23d00cf4616dc28e980a68fcbeebb`. No new local build or test
suite was run for this base update; CI validates the combined tree.

## Upstream behavior retained

Upstream #3808 makes pathfinding and packet arbitration cooperate: the planner
reports illegal output groups, rule exhaustion, pinned-overlay conflicts, and
potential arbiter hold cycles as routing faults. The router retries, separates
or joins packet trees, and preserves the requested destinations. DMA volume,
receiver capacity, lock ordering, and runtime waits inform deadlock analysis.
Eligible packet transit hops may use circuits to save arbiters. Reload overlays
retain their routes and physical packet-rule slots. Upstream #3877 configures
stream switches before enabling DMA.

## Medusa behavior carried into the new architecture

- AIE4 wide, narrow, and control networks coexist. Explicit graph and packet-tree
  edge kinds distinguish crossbars from physical wires, including same-tile
  32/512-bit bridges. Bridges retain channel indices and a width-change cost of
  256; congestion history is scaled by 64 where bridges exist.
- A shared physical-link helper supports routing, overlay recognition, and
  dependency tracing. Wire emission is idempotent and covers direct AIE4 shim
  endpoints and narrow/control links. Lifting enumerates high DMA source
  channels and preserves true same-tile endpoint circuits.
- Circuit outputs remain exclusive across source rows; mixed circuit/packet
  collisions also trigger rerouting. Legal packet fan-in remains supported.
  Upstream's tree ownership and destination-set checks replace the old same-ID
  ownership matrix. Whole-output-set arbitration replaces the old multicast
  guard without dropping broadcast destinations.
- Mandatory AIE4 TileControl/DMAControl injection remains circuit switched even
  when optional circuit-hop optimization is disabled. Injection validates output
  exclusivity and equal first-hop destination sets across IDs before conversion.
- DMA selectors remain distinct: S2MM=0, MM2S=1, ControlMM2S=2, TraceS2MM=3.
  Dependency analysis uses the target's stream-port mapping, including separate
  data/control/trace engines with the same channel number. Packet headers retain
  their volume through circuit hops and into trace receivers.
- Overlay detection recognizes direct control DMA and bridge/narrow paths.
  Stale-device cleanup preserves overlay-owned packet slots on shared slave
  ports and disables application slots without renumbering them.
- DMA channel preferences, BD pools, D3 addressing, register banks, direct shim
  routing, and the rest of the Medusa history are retained.
- New LLVM strict-property assembly requirements are handled for AIE4 vector
  conversions and matrix operations without changing signedness or shift values.

## Independent review

Two reviewers inspected separate implementation areas. Findings addressed:
mandatory-injection collisions and unintended per-ID broadcasts; bridge edge
identity through pinned trees; physical wire and high-channel flow recovery;
same-tile non-DMA circuit lifting; shared-overlay slot cleanup; and packet-header
accounting through circuits. Core routing and dependency/overlay changes each
received review from someone other than their author. Source approval is
separate from execution results below.

## Validation

The selected build and validation work is complete. At the user's request, the
remaining Python tests and GEMV builds were stopped; further testing is left to
CI. No tests were weakened to hide a behavior change.

The target built with upstream's exact LLVM wheel
`24.0.0.2026100106+bdf04ebb` and fresh private submodules at all six final Medusa
gitlink pins. Built tools include aie-opt, aie-translate, aiecc, Python bindings,
runtime libraries, and the C++ tests.

Completed checks (suite counts overlap; do not add them as unique tests):

| Check | Result |
|---|---|
| Focused routing, flow lifting, overlays, and cleanup | 226 passed |
| Complementary compiler suites: conversions, dialects, DMA/BD passes, targets, object FIFOs, and placement | 1,043 passed, 4 expected failures |
| AIEVec and core-entry-mode compatibility | 195 passed, 50 unsupported |
| Additional signedness/shift boundary and round-trip checks | 20 passed |
| C++ tests, including target DMA identity and new dependency analysis | 8 passed |
| Additional control/reconfiguration compiler subset | 15 passed |
| Three updated generic golden fixtures, original vs routed/lifted endpoints | All 94 circuit or packet-ID endpoint relations identical |

The initial clean new-upstream baseline also built: 191 routing tests and 21
overlay/load-PDI tests passed. Three selected aiecc tests passed; two failed while
linking with the custom Medusa Peano on `.tctmem*` orphan sections. These baseline
results are separate from target validation.

Hardware was a RyzenAI-npu9 / AIE4 6x3 device. All 11 completed cases passed:

- Vector-add and trace routing on compute rows 2, 3, 4, and 5. Each trace decoded
  into 1,596 events, including begin/end events.
- Control-packet demo: zero reference errors.
- Three successive device configurations: all 576 output elements checked.
- Concurrent shim DMA with 1, 6, and 12 channels: completion and bandwidth checks.
  This sink benchmark does not compare received payloads.
- Three-column 512x768x1024 int8 GEMM with int8 and int32 outputs. Each checked all
  393,216 results on the first and last timed launches.

The two multistream GEMV cases were not completed. The Python compiler suite's
corrected rerun was interrupted, so no passing result is claimed for that suite.

The initial failures exposed and helped resolve:

- Mandatory-injection legality and packet-header accounting issues identified
  during independent source review.
- Missing physical links/high DMA source counts and same-tile endpoint lifting.
- Legacy overlay marker compatibility and AIE4-only control-header stripping.
- New LLVM strict-property defaults breaking original AIEVec assembly syntax.
- Exact-route assertions whose channel/arbiter choices changed while endpoint,
  width-preference, multicast, and strict-deadlock invariants stayed intact.

Two independent reviewers approved the core router, dependency/overlay port,
compatibility fixes, and changed test expectations. Each implementation area
was reviewed by someone other than its author.

Build commands, logs, JSON results, endpoint proofs, and hardware artifacts are
retained in the adjacent `.worktrees/medusa-router-build-20261007` directory.
Its `BUILD_SETUP.md`, `run-lit.sh`, and `hardware-validation.py` document the
isolated toolchain and reproduction commands. CMake normally patches five
aie-rt files from tracked patches; the generated edits were byte-compared with
those patches before cleanup. Reconfiguration reapplies them for another build.

## CI follow-up

The original draft's first CI run found a missing NPU3 variant in the new upstream
`memtile_dma_starts_at_load` test. The fixture now selects `npu3_1col` and uses the
supported full-ELF compile/launch path. Existing NPU1/NPU2 commands, initialized
lock, static DMA start, and all 256 payload checks are preserved. Independent
review approved this fix and the single targeted NPU3 hardware test passed.

[CI run 3837802](https://gitenterprise.xilinx.com/XRLabs/mlir-aie/actions/runs/3837802)
succeeded on the previous draft tip `337e26db6ac`. That result predates the
2026-10-08 base update and is not a test result for the newly combined tree.

## PR #376 review findings — 2026-10-08

1. **AIE4 shim arbiter pressure: confirmed.** Both optional circuit-hop extraction
   and the upfront arbiter check excluded all shim tiles. They now exclude only
   legacy shims with a mux. The new
   `aie4_shim_circuit_hop_reserved_arbiters.mlir` reserves every shim amsel: a
   packet can still cross the shim through an exclusive circuit hop, and routing
   with that optimization disabled must report no free arbiter. The old compiler
   rejects the valid default case before routing.
2. **Pinned direct-DMA overlay lookup: confirmed.** The pinned-rule lookup still
   translated an AIE4 shim DMA input into a legacy South mux port. It now uses
   DMA directly on AIE4. The new `aie4_pinned_overlay_direct_shim_dma.mlir` compares
   an overlay routed alone with a reload containing application traffic. Its
   data-DMA source reaches both tile controllers, and the old compiler reproduces
   the exact reported “takes no packet rule there alone” failure. This regression
   checks the installed channel, amsel, and rule with circuit-hop extraction both
   enabled and disabled.
3. **Missing ordering regression and duplicate verifier counts: confirmed.**
   Source and destination collection already uses separate passes, so the original
   ordering bug is fixed. The upstream `packet_flow_dest_before_source.mlir` was
   nevertheless missing; it is restored byte-for-byte from `c5e495`. Duplicate
   increments are removed from `PacketFlowOp::verify`, retaining multiple-source
   support, required endpoints, and packet-mask checks.
4. **Duplicate shim-row fan-in fixture: confirmed.**
   `same_id_fanin_5way_shimrow_npu2.mlir` had identical executable MLIR and RUN
   instructions to `same_id_fanin_5way_memtile_dest.mlir`. The duplicate is removed;
   the retained fixture checks all five complete source paths more precisely.
5. **Destination pressure affecting legacy targets: confirmed, retained with
   explicit justification.** Historical Medusa #226 deliberately applied this
   rule globally; it was not newly broadened by this rebase. The rule is retained
   for the following reasons:

   - AIE4 regression `aie4_width_change_congestion.mlir` exercises two circuits
     competing for the only free North:0 output, requiring a bridge fallback,
     and mixed packet/circuit contention. #226 originally fixed multi-stream
     memtile/shim DMA routing, including the four-weight-stream GEMV case.
   - Gating all destination legality to AIE4 would weaken correctness elsewhere:
     upstream's per-wire accounting cannot cover every terminal-output collision
     or a flow ending at a directional port used by another stream.
   - We tried the narrower alternative of preserving global legality and actual
     overuse history while making only proactive destination pressure AIE4-only.
     `packet_shared_ids_split_last.mlir` then failed to find a deadlock-safe route
     on NPU2, rather than merely choosing a different channel. The diagnostic
     identifies id24 from (0,4) DMA:0 holding arbiter 0 at (0,3), needed by id9
     from (0,4) DMA:1, with receiver/DMA waits completing a cycle. This fixture
     passes with the retained rule. The tentative change was reverted.
   - The congested NPU1 fixture also still differed from its original upstream
     golden under that attempted narrowing. We therefore retain the three
     previously reviewed golden updates and their endpoint-preservation evidence
     instead of claiming that the same legality constraint restores all original
     route choices. This is an intentional, documented deviation in routing cost;
     it preserves the design's endpoints and strict deadlock checks.

Independent review approved the shim fixes and fixtures, the ordering/fan-in
cleanup, and retaining global destination pressure after the failed narrowing
experiment. The exact NPU2 diagnostic is retained in the adjacent build directory
as `pr376-r11-aie4-only-regression.log`. No hardware or broad compiler suite was
rerun for these review fixes.

Final review validation: rebuilt `aie-opt` and passed all **19 selected lit
fixtures** on the final source. This includes both new shim regressions, the
restored ordering test, retained fan-in coverage, legacy arbiter/overlay cases,
AIE4 width/bridge/control cases, and all three cited generic routing tests. Each
fixture's existing RUN lines were executed, including strict and disabled-hop
variants where present. Logs/results: `pr376-review-final-build.log`,
`pr376-review-final-focused.log`, and `pr376-review-final-focused.json` in the
adjacent build directory. No further local testing is planned; CI covers the
broader checks.
