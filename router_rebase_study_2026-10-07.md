# Router rebase study — 2026-10-07

This records the pre-rebase study. See
[the validation report](router_rebase_validation_2026-10-07.md) for implementation
and execution results from the fresh worktree.

The recommended approach is to adopt upstream's new routing and packet-planning
architecture, then port Medusa's remaining hardware semantics into it. Keeping
Medusa's old router wholesale would discard the deadlock fixes. Replaying its
router patches mechanically would duplicate fixes upstream now implements and
miss new places that assume an AIE1/AIE2 topology.

This is a source study and three-way textual merge probe, not a completed rebase
or a build validation. The current `gemm-fp8-bf16` checkout was not rebased.

## Exact comparison

Refs were refreshed from `origin` on October 7. This repository's public-upstream
mirror is `origin/main-mirror`; there is no configured remote named `upstream`.

| Role | Commit | Meaning |
|---|---|---|
| Previous upstream / common base | `b15f3508006988ea0252300a1feac447990ce6ec` | Current Medusa's upstream base |
| Medusa studied | `317cb0b96be` | Refreshed `origin/medusa-dev` |
| New upstream target | `c5e495fb22c` | Refreshed `origin/main-mirror` |
| Router rewrite | `92bf1670ec5` | Upstream #3808, “Route packet flows around arbiter deadlocks” |
| Related initialization fix | `a8ba20da32b` | Upstream #3877, configure switches before enabling DMA |

There are 15 upstream commits since the common base and 370 Medusa commits to
carry. Medusa's `57d43bb5794` is the previous rebase integration-fix commit and
must be reviewed alongside the original feature commits.

The old base already includes dense graph IDs (#3251), direction-aware graph
states (#3592), arbiter deadlock avoidance (#3702), and idempotent routing / flow
lifting (#3711). Those are not new in this update. #3808 substantially replaces
and extends the routing and hazard machinery on top of them.

The August `aie4_routing_review.md` is historical context, not the current delta:
bridge fixes, AIE4 tests, control overlays, DMA separation, and another upstream
rebase have landed since then.

## What upstream changed

**Routing and arbiter allocation now negotiate together.** Previously a path
could fit the channel graph but fail subsequent packet arbiter allocation.
Upstream introduces `PacketConstraints`, `RoutingCheck`, and `RoutingFaults`.
The planner can reject a candidate and identify connections to move, packet
trees to split, crowded tiles, or sources to separate or bring together. The
pathfinder penalizes or restructures those choices and tries again.

**It analyzes why streams can block each other.** The new
`AIEStreamDependencyAnalysis.{h,cpp}` reads DMA programs, BD lengths, repeat
counts, locks, and runtime issue/wait relationships. It estimates sender volume
and receiver capacity, builds waits between agents, and detects cycles involving
links, arbiters, and receiver draining. Unknown programs or volumes are treated
conservatively within the endpoint kinds the analysis understands. It can
distinguish route-dependent hazards from designs that can deadlock regardless
of routing.

**Packet trees are explicit.** `Routing` contains ordinary switch settings,
packet trees, and per-ID settings. The search can join trees, split IDs from a
source onto different paths, and reroute trees that force an illegal shared
arbiter. Safe same-ID fan-in is supported: identical IDs can merge when their
continuation preserves the intended destination set. This supersedes the old
blanket prohibition on same-ID sharing and overlaps Medusa #258.

**Arbiter and rule allocation are more complete.** The planner groups outputs
that must share an arbiter, respects configured resources, plans msels and
first-match packet rules, and feeds resource exhaustion back into routing.
It can replace eligible packet transit hops with circuit connections when
arbiters are scarce. Those hops retain packet headers for subsequent switches.
This replaces Medusa's smaller fix that diagnosed conflicting broadcast outputs
instead of silently dropping a leg.

**The search relaxes packet constraints in stages.** It starts with sharing
within destination groups, then can permit broader sharing, cap packet fanout
at crowded tiles, and split IDs/trees. These are legality-aware retries, not
permission to ignore an arbiter conflict. `allow-deadlock-prone`, default false,
is a separate explicit escape hatch for supported deadlock diagnoses; aiecc
exposes `--allow-deadlock-prone-routing`.

**Priority routes and reload overlays get stronger semantics.** Prioritized
flows are routed alone first. Ordinary priority routing can fall back to joint
routing; control-packet reloads must preserve the installed overlay. Overlay
markers can occur on individual rules sharing a slave port. `AIERT.cpp` skips
installed overlay rules without renumbering their hardware slots, and the PDI
expansion path checks the installed routing.

**Interfaces changed substantially.** The abstract `Router` interface is gone;
`DynamicTileAnalysis` owns a `Pathfinder`, returning `llvm::Error` and a
`Routing` rather than a map wrapped in `optional`. Parallel per-edge matrices
became a flat `SwitchboxConnect::Cell` array. New destination equivalence classes
track arbiter units. Target models must implement `getNumArbiters()` and
`getNumMselsPerArbiter()`. New endpoint verifiers and shared header-policy helpers
also landed.

#3808 changes 193 files, with 21,763 insertions and 2,649 deletions including its
large test suite. The changes extend beyond the two router sources to flow
lifting, target emission, overlays, Python, aiecc, and documentation.

Separately, #3877 moves switch configuration before DMA initialization/start.
Preserve that ordering with Medusa's stale-switch cleanup and specialized DMA
initialization. The new upstream tip also contains LLVM (#3836) and Peano
(#3857) updates, which need separate toolchain validation during the full rebase.

## Medusa versus the previous upstream

The net delta in `AIEPathFinder.h`, `AIEPathFinder.cpp`, and
`AIECreatePathFindFlows.cpp` is 421 added / 115 removed lines. The much larger
target-model, DMA, trace, and control changes are dependencies of that delta.

| Medusa behavior | Main commits | Treatment on new upstream |
|---|---|---|
| AIE4 narrow network: North32/South32, shim-row East32/West32, and NorthControl→SouthControl | `b9f116c816c` #107, subsequent fixes | Port graph construction and all topology walkers |
| Direct AIE4 shim switch ports, without legacy shim-mux lowering | `85b944f271f` #209 | Preserve in construction, reservations, emission, and analysis |
| Safe same-ID fan-in without unintended replication | `e5a76762e95` #258 | Prefer upstream tree semantics; retain Medusa control/fan-in regressions |
| Circuit destination-port capacity, across different source rows of the crossbar | `9f63719b029` #226 | Port the invariant to upstream's new cell/state structure |
| Real same-index 32↔512 bridge wires; explicit crossbar versus wire identity | `dde7b568d94` #298 | Port throughout graph, tree handling, and route emission |
| Fixed additive width-change cost of 256 and AIE4 history scale of 64 | `dde7b568d94` #298 | Preserve policy initially; validate against new conflict costs and retries |
| Correct priority-queue update after a shorter path is found | `dde7b568d94` #298 | Already upstream; retain upstream implementation |
| Bridge-aware packet reachability with a visited set | `dde7b568d94` #298 | Port into new topology/reachability helpers |
| Reject inconsistent broadcast arbiter assignments rather than dropping outputs | `dde7b568d94` #298 | Use upstream whole-unit planning; preserve the no-lost-leg invariant |
| AIE4 control overlay and circuit injection from TileControl / DMAControl; separate Control32→Controller32 TCT wiring | `e5135587126` #237, `fc2b789c625`, #327 | Integrate with new packet planning and dependency analysis |
| Dedicated trace DMA, trace BD pools and reservations | `c1e51f1c8c5` #326, `7e849c392b5` #227, `1b934bb4346` #360 | Preserve alongside routing and register generation |
| DMA engine selectors separated from routing bundles | `434e7bbfb02` #327 | Extend the new dependency analysis to respect this abstraction |
| Fixes after the previous upstream rebase | `57d43bb5794` | Review each surviving fix; some old call-site adaptations may be obsolete |

For the DMA abstraction, `ControlMM2S` maps to `DMAControl`, `TraceS2MM` maps to
`DMATrace`, and ordinary MM2S/S2MM map to `DMA`. The new dependency analysis
currently constructs ordinary DMA endpoints directly from a channel number.
It must preserve selector/bundle identity, even where the numeric channel index
is unchanged. `getDmaStreamPort()` is the existing forward mapping to reuse.

Medusa's destination-capacity tracking is a different resource constraint from
upstream's `dstUnits`: the latter group master ports for arbiter assignment;
they do not replace `usedDestCapacity` / `overDestCapacity` for circuits.

## Conflicts and semantic traps

A `git merge-file -p` probe compared each file changed on both sides, using
`b15f3508006` as base. It did not modify the index or start a rebase.

Across the entire branch delta, 37 files change on both sides. Fourteen files
have textual conflicts, totaling 44 conflict regions under this probe. This is
a final-tree estimate; replaying 370 individual commits can conflict repeatedly
and produce different counts.

| File | Conflict regions |
|---|---:|
| `include/aie/Dialect/AIE/Transforms/AIEPathFinder.h` | 2 |
| `lib/Dialect/AIE/Transforms/AIEPathFinder.cpp` | 11 |
| `lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp` | 10 |
| `lib/Targets/AIERT.cpp` | 1 |
| `lib/Dialect/AIEX/IR/AIEXDialect.cpp` | 1 |
| Seven routing tests | 16 total |
| Two DMA validation tests | 3 total |

The seven routing tests are `broadcast`, `same_id_shared_link`,
`arbiter_transit_control_flow`, the npu1/core/memtile `same_id_fanin` tests, and
`test_congestion_flow_pktflow_cascade_all_in_one`.

The most consequential problems are not all marked by Git:

1. **A clean target-model merge leaves AIE4 abstract.** Upstream adds two pure
   virtual arbiter-count methods and implements them only in its AIE1/AIE2
   classes. `AIE4TargetModel` derives directly from `AIETargetModel`. It needs
   implementations. The previous allocator used six arbiters and four msels
   globally; preserving that policy is a separate decision from proving the
   complete AIE4 hardware resource model. Audit the 32-bit and 512-bit resource
   domains before expanding capacity or allowing numeric arbiter reuse.

2. **Same tile does not mean crossbar.** Upstream still uses
   `srcCoords == dstCoords` in Dijkstra, weights, tree construction/joining,
   fault handling, and output generation. Medusa's bridge wires stay within a
   tile. Porting only `initialize()` would leave illegal state transitions or
   emit a hard wire as `aie.connect`. Every such use needs classification as
   topology, arbiter resource, diagnostic grouping, or path-length accounting;
   do not replace all comparisons blindly.

3. **Bridge ports are not terminal endpoints.** Upstream's `isDirectional()`
   recognizes only North/South/East/West. `TreeBuilder`'s same-ID mergeability
   check treats a non-directional destination as terminal, while
   `StreamTracer::follow()` stops there. With AIE4 ports added mechanically,
   those assumptions can permit a bad merge or hide the remainder of a stream
   from deadlock analysis. Merely extending `isDirectional()` is insufficient:
   same-tile bridges and inter-tile directions have different geometry.

4. **There are multiple topology consumers.** Adapt `linkedInput()`, packet
   reachability, upstream's new `StreamTracer`, source discovery, and emitted
   `aie.wire`s consistently. Preserve the visited-set guard: upstream's new
   recursive `findPathToDest()` does not carry Medusa's guard. New wire emission
   deduplicates connections; insert AIE4 bridge wires through that helper.
   Check narrow links, control links, and direct shim endpoints during lifting.

5. **New shim handling has more entry points.** Besides circuit lowering,
   upstream has `lowerShimDMAPorts()`, `unmuxShimDMAPacketPorts()`, shim South-port
   reservations in `addFixedConnection()`, and packet-rule matching that maps
   DMA back to shim South ports. All must respect AIE4's direct wiring.

6. **Control injection is mandatory hardware behavior.** Medusa emits
   TileControl/DMAControl injection hops as deduplicated circuits. Upstream's
   optional `findCircuitHops()` is an arbiter-pressure optimization, excludes
   shim tiles, and requires different conditions. It cannot substitute for
   those injection hops. Model them before packet arbiter/resource planning,
   including when `circuit-switch-hops=false`.

7. **An unrecognized AIE4 DMA sink can look infinitely drainable.**
   `StreamWaitGraph::Agent::at()` understands Core, ordinary DMA, and sending
   TileControl; other ports are assumed to drain. `DMATrace`, `DMAControl`,
   Control32, and Controller32 need deliberate classifications and appropriate
   volume/wait modeling. The latter control ports may need different semantics
   from DMA, not a blanket cast. Otherwise the new safety check can miss a real
   dependency; overly broad fallback dependencies can also reject valid designs.

8. **Overlay rule ownership changed.** Port the individual-rule overlay
   markers and stable slot numbering into AIERT, and audit Medusa's
   `disableOtherDeviceSwitches()` against shared overlay/application slave ports.
   Preserve the new switches-before-DMA ordering. Header policy must remain
   consistent among routing, analysis, and emission, including DMATrace's
   existing behavior and explicit `keep_pkt_header`.

## Rebase sequence

1. **Pin the old Medusa and destination SHAs and work in an isolated checkout.**
   Keep a backup ref before history rewriting. This checkout has unrelated
   work and modified submodules. Use the final-tree delta above as the inventory
   even if the actual operation replays individual commits. Do not blindly skip
   a whole historical commit just because one router fix is now upstream.

2. **Use upstream's new router as the implementation base.** Keep `Routing`,
   `PacketPlan`, `PacketConstraints`, flat cells, structured errors, dependency
   analysis, and new arbiter planning. Preserve the other Medusa subsystems
   through the normal rebase. Resolve obsolete router hunks against an explicit
   list of semantic requirements, rather than restoring old functions to make
   a historical patch apply.

3. **Port AIE4 topology first.** Restore explicit crossbar/wire identity and
   graph keys, same-index bridges, bundled narrow/control links, direct shim
   handling, and emission of only configurable crossbar hops. Introduce a shared
   physical-link helper usable by reachability and stream tracing. Keep terminal
   endpoint classification distinct from directional-link classification.

4. **Port capacity and width policy.** Adapt destination occupancy/history to
   cells and the new rip-up/retry paths, preserving packet sharing. Apply the
   additive bridge penalty in upstream's `edgeWeight()` so ordinary, seeded,
   branch, and join searches use one cost policy. Keep upstream's heap handling.
   Exercise both “prefer wide” and “narrow detour is required” cases.

5. **Integrate control, trace, and dependency semantics.** Add target-model
   arbiter methods, mandatory control injection, specialized DMA endpoint
   mapping, and AIE4 controller/TCT dependencies. Carry bridge identity through
   packet-tree joins and hold-cycle analysis. Validate same-ID semantics with
   upstream's tree implementation before deciding any part of #258 remains
   necessary. Do not disable deadlock checks globally to pass AIE4 tests.

6. **Finish overlay/backend integration and reconcile tests.** Keep upstream's
   whole-unit broadcast planning and overlay ownership. Update checks where
   upstream deliberately chooses a different legal route, but retain structural
   assertions against duplication, missing destinations, wrong bridge indices,
   and unnecessary narrow paths. Review `57d43bb5794` for carried fixes.

7. **Validate the complete new base.** Keep the router port distinguishable from
   the LLVM/Peano migration in the integration commits. Review a final
   `range-diff` and the new-upstream-to-rebased-Medusa diff against this inventory.
   Publishing a rebased branch comes after the compiler and hardware gates.

## Validation gates

The following are proposed checks, not results from this study.

| Gate | Required evidence |
|---|---|
| Build | AIE4 target instantiation; AIE/AIE transforms, new stream analysis, AIEX, targets, aie-opt, aie-translate and aiecc compile with the final LLVM pin |
| Upstream routing | `test/create-flows`, `test/create-packet-flows`, `test/find-flows`, relevant dialect/overlay and aiecc routing tests |
| AIE4 topology | `aie4_switch32_512_bridge`, `aie4_bridge_wire_roundtrip`, all three `aie4_width_change_*` tests, and `aie4_control_overlay_fan_in` |
| AIE4 packets | `aie4_bridge_packet_fanout`, `aie4_trace_broadcast_routes`, `aie4_width_penalty_broadcast_conflict`, plus the shim-row same-ID fan-in case |
| Control / DMA | AIE4 control-source lowering, overlay generation, post-placement control wiring, specialized shim selectors, trace channel/BD-pool/reservation tests |
| New analysis coverage | A bridge-crossing dependency/hold cycle, specialized DMA receiver backpressure, and separate data/control/trace channel identity; verify both diagnosis and a valid nonblocking case |
| Round trip | Route twice without duplicate configuration; lift and reroute bridge/narrow/direct-shim paths without losing endpoints or changing packet destination sets |
| Hardware | `vadd_trace_route_test` across rows 2–5; control-packet demo/reload; multistream memtile/shim traffic; mixed data and dedicated trace; representative GEMM/GEMV |
| Performance | Representative congested and multicast routing times plus hardware throughput; legal routes can still regress by crossing the narrow fabric unnecessarily |

Existing CHECK output is evidence of prior behavior, not a requirement to retain
every chosen channel or amsel number. Newly rejected designs need their reported
dependencies examined before either changing a golden or using the explicit
deadlock-prone override.

## Source anchors and probe artifacts

The snapshot files and the per-file probe results are in
`/tmp/medusa-router-study-20261007/{base,medusa,upstream,merged}` and
`/tmp/medusa-router-study-20261007/overlap.json`. The `merged` directory contains
only files with textual conflicts. Git objects at the SHAs above are the durable
source of truth.

Useful upstream anchors at `c5e495fb22c`:

- `AIEPathFinder.h`: `SwitchboxConnect`, `RoutingFaults`, `Routing`, and `PacketConstraints`.
- `AIEPathFinder.cpp`: `initialize`, `edgeWeight`, `TreeBuilder::claim`, `TreeBuilder::join`, and `findPaths`.
- `AIECreatePathFindFlows.cpp`: `linkedInput`, `findCircuitHops`, `collectSlaveFlows`, `PacketFlowRouting::plan`, `route`, and `runOnOperation`.
- `AIEStreamDependencyAnalysis.cpp`: `StreamTracer`, `StreamVolumeAnalysis`, and `StreamWaitGraph::Agent::at`.
- `programming_guide/routing.md`: new user-facing routing semantics and options.

Useful Medusa anchors at `317cb0b96be`:

- `AIEPathFinder.h`: `isCrossbar`, `changesWidth`, destination-capacity arrays, and `GraphKey`.
- `AIEPathFinder.cpp`: `bridgeconnect`, `WidthChangePenalty`, and `historyScale`.
- `AIECreatePathFindFlows.cpp`: `findPathToDestImpl`, mandatory control injection, and bridge wire emission.
- `AIETargetModel.cpp`: `getDmaStreamPort` and `AIE4TargetModel::isLegalTileConnection`.
- `AIERT.cpp`: `disableOtherDeviceSwitches` and `configureSwitches`.
