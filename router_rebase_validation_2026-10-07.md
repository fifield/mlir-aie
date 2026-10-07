# Medusa router rebase — 2026-10-07

Work branch: `rebase/medusa-router-20261007`, prepared for a draft PR against
`medusa-dev` so CI can validate the integration. Publishing this branch does not
update `medusa-dev`; its eventual history integration is deferred. The original
checkout remains on its original branch.

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
