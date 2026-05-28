# Decode `aie.device` packing — feasibility analysis

Target: collapse the per‑token `aiex.run` reconfiguration cost in the BF16
decode path by packing multiple decode phases into a single `aie.device`
block, ideally one phase per compute row (rows 2/3/4/5).

## 0. Why this matters

Per decode token (BF16 path, 16 transformer layers + LM head):

| kernel                | XRT calls / tok | aie.device dispatches / tok |
| --------------------- | --------------- | --------------------------- |
| `rms_gemv_rope`       | 16              | 96  (6 segs × 16)           |
| `o_gemv_ffn`          | 16              | 128 (8 segs × 16)           |
| `lm_head_gemv`        | 1               | 8                           |
| **total**             | **33**          | **232**                     |

Each `aiex.run` reapplies a fresh per‑segment PDI: switchbox config,
memtile/shim DMA BDs, core ELF program load. A 128‑token generation =
~30 k segment reconfigurations. Even at tens of µs each that is
milliseconds of pure overhead per token.

The phases inside one fused kernel are strictly data‑dependent, so
packing buys **no compute parallelism** — the win is entirely
eliminating reconfig cost.

## 1. Current per‑phase resource footprint (o_gemv_ffn)

Counted directly from `reference_mlir/o_gemv_ffn.npu.air.mlir`:

| phase                 | cores | rows used | shim_alloc | memtile_dma_start | flows | buffers | locks |
| --------------------- | ----- | --------- | ---------- | ----------------- | ----- | ------- | ----- |
| `og_matvec_bf16_0`    | 8     | row 2     | 17         | 8                 | 40    | 40      | 80    |
| `a1_eltwise_add_seg`  | 8     | row 2     | 24         | 0                 | 24    | 24      | 48    |
| `rm_rms_seg`          | 1     | row 2     | 3          | 0                 | 3     | 4       | 6     |
| `gg_matvec_bf16_0`    | 8     | row 2     | 17         | 8                 | 40    | 40      | 80    |
| `ug_matvec_bf16_0`    | 8     | row 2     | 17         | 8                 | 40    | 40      | 80    |
| `sw_silu_mul_seg`     | 8     | row 2     | 24         | 0                 | 24    | 24      | 48    |
| `dg_matvec_bf16_0`    | 8     | row 2     | 17         | 8                 | 40    | 40      | 80    |
| `a2_eltwise_add_seg`  | 8     | row 2     | 24         | 0                 | 24    | 24      | 48    |

Two topologies:

**Matvec phase (40 flows, 17 shim allocs):** per‑col weight stream
`shim col c MM2S 0 → memtile (c,1) DMA 0`; vector broadcast
`shim col 0 MM2S 1 → compute (c,2) DMA 0` for c=0..7; output back via
`memtile (c,1) DMA 0 → shim col c S2MM 0`.

**Eltwise/SiLU phase (24 flows, 24 shim allocs):** per‑col, no memtiles
— `shim col c MM2S {0,1} → compute (c,2) DMA {0,1}` and
`compute (c,2) DMA 0 → shim col c S2MM 0`.

## 2. npu2 hardware budget (per `AIETargetModel.cpp`)

| resource                                   | per tile                                    |
| ------------------------------------------ | ------------------------------------------- |
| Shim DMA channels (DMA bundle, ShimMux)    | **2 MM2S + 2 S2MM**                         |
| Memtile DMA channels                       | **6 MM2S + 6 S2MM**                         |
| Memtile size                               | 512 KB                                      |
| Compute tile DMA channels (DMA bundle)     | **2 MM2S + 2 S2MM**                         |
| Compute tile L1                            | 64 KB                                       |
| Shim → switchbox South bundle              | 6 dest / 8 src                              |

The hard limit is **the shim**: only 2 + 2 DMA channels per column.

## 3. Naive “one phase per row” layout

Map row 2 → phase A, row 3 → B, row 4 → C, row 5 → D, all columns 0–7.

```
                col 0 ... col 7
row 5  [D D D D D D D D]   ← e.g. dg_matvec
row 4  [C C C C C C C C]   ← ug_matvec
row 3  [B B B B B B B B]   ← gg_matvec
row 2  [A A A A A A A A]   ← og_matvec
row 1  [M M M M M M M M]   ← memtiles (shared across rows)
row 0  [S S S S S S S S]   ← shim (shared across rows)
```

Resource demand for the 4‑matvec packing:

* **Compute‑tile DMA (row r, col c):** each row hosts a different phase,
  same 2+2 channel pattern as today. No conflict — different tile means
  different switchbox dest. **OK.**
* **Compute‑tile L1:** each row gets its own kernel + buffers; not
  shared. Largest matvec phase ≈ 32 KB of buffers. **OK.**
* **Memtile DMA (col c, row 1):** every phase wants its own memtile
  staging. Per memtile we’d need:
  - 4 × (S2MM from shim, weight in) = 4 S2MM
  - 4 × (MM2S to its compute row, weight out) = 4 MM2S
  - 4 × (S2MM from its compute row, result in) = 4 S2MM
  - 4 × (MM2S to shim, result out) = 4 MM2S
  → **8 S2MM + 8 MM2S vs 6+6 budget — does not fit.**
* **Shim DMA (col c, row 0):**
  - col 0: 4 × (MM2S 0 weight + MM2S 1 vector + S2MM 0 output) = 8 MM2S
    + 4 S2MM.
  - col 1–7: 4 × (MM2S 0 weight + S2MM 0 output) = 4 MM2S + 4 S2MM.
  → **All columns exceed the 2+2 shim budget.**

So a literal row‑per‑phase mapping of the four matvecs cannot be
statically routed on npu2.

## 4. What actually works

The blocker is shim/memtile channel count when each phase still
independently sources weights from DDR and writes outputs to DDR.
Three workable variants:

### 4a. Memtile‑chained pipeline (eliminates DDR round‑trips)

Don't write intermediates back to DDR. Inside one device, the data flow
is:

```
DDR ──(weights/x)──▶ memtile L2 buffer
                      │
                      ▼
                row 2 cores  (phase A: e.g. og)
                      │  L2/local stream
                      ▼
                row 3 cores  (phase B: a1_add)
                      │
                      ▼
                row 4 cores  (phase C: rm_rms or gg)
                      │
                      ▼
                row 5 cores  (phase D: ...)
                      │
                      ▼
DDR ◀──(final output)── memtile
```

Per column shim now does **1 MM2S (initial input) + 1 S2MM (final
output)** for the whole pipeline = fits in 2+2 with one channel to
spare for weights of one matvec phase. Memtile DMA: at most 4 streams
in transit (one per inter‑row hop) — fits in 6+6.

This is the right shape for an *eltwise+silu+rms* fused block (no
weights). It does **not** fit four matvecs back‑to‑back because each
matvec still wants its own weight stream — but two matvecs + their
post‑ops can.

Natural decomposition of `o_gemv_ffn` (8 → 3 devices):

| device   | phases                                  | rows used | shim channels (col c) |
| -------- | --------------------------------------- | --------- | --------------------- |
| `og+a1`  | og_matvec → a1_eltwise_add              | 2, 3      | MM2S 0 (weight) + MM2S 1 (x) + S2MM 0 (res1) |
| `rm+gg+ug+sw` | rm_rms → gg_matvec ‖ ug_matvec → sw_silu_mul | 2, 3, 4, 5 | MM2S 0 (gg weight) + MM2S 1 (ug weight) + S2MM 0 (swiglu out)  |
| `dg+a2`  | dg_matvec → a2_eltwise_add              | 2, 3      | MM2S 0 (weight) + MM2S 1 (x_res2) + S2MM 0 (output) |

That is **3 device dispatches per layer instead of 8** for the FFN.
Same idea applied to `rms_gemv_rope` (6 → ~2 devices): pack {r_rms +
rq_rope + rk_rope} as one (single‑tile chain on rows 2/2/2 via packet
flows) and {q_matvec ‖ k_matvec ‖ v_matvec} as a 3‑row co‑resident
device (each on its own row, all reading the same normed input from a
shared memtile broadcast).

Initial estimate (revised in §10.7 and §11):
* `rms_gemv_rope`: 96 → ~32 seg dispatches/token
* `o_gemv_ffn`: 128 → 48 seg dispatches/token
* total: 232 → ~88 seg dispatches/token, **~60% reduction**.

**This estimate was optimistic** — it assumed every hand‑off between
phases is an element‑wise L2 chain. §11 shows three hand‑offs in
o_gemv_ffn are scatter‑gather / scatter‑broadcast transitions
(a1→rm, rm→gg/ug, sw→dg) that don't fit a simple L2 chain on a
column‑partitioned herd. The realistic figure with device boundaries
placed at the hard hand‑offs is **8→4 devices for o_gemv_ffn, 6→2
for rms_gemv_rope, ~55% total dispatch reduction** (§10.8).

### 4b. Packet‑switched flows (logical multiplexing of one shim channel)

`aie.packet_flow` lets multiple logical streams share one physical
switchbox channel, distinguished by 5‑bit packet IDs. If we packet‑mux
the col‑0 broadcast and the per‑col weight streams, we could in
principle put 4 matvecs on one device without memtile chaining.

Trade‑offs:
* Adds packet ID overhead (5b header per stream packet) and small
  routing/arbitration cost.
* mlir‑aie supports it but the existing AIR-emitted matvec template
  uses circuit flows; we’d need a custom emitter.
* Still fights the memtile DMA channel budget (6+6) — each matvec
  needs ~4 memtile channels for weight staging, so 4 phases = 16
  channels even with packet IDs.

So 4b alone is insufficient. It's an additive optimization on top of
4a, not a replacement.

### 4c. BD‑level time‑multiplex on shared channels (keep all routing
identical, change only BD chains)

Place every phase on the *same* tiles with the *same* channels and
just chain different BD descriptor groups in the runtime sequence.
Each `aiex.run` becomes “start BD chain N” rather than reload PDI.

This is essentially what an `aiex.runtime_sequence` could already do,
but the cached AIR output emits separate PDIs because each phase was
lowered independently. To get the saving, we need the **lowering**
itself to emit one `aie.device` with `8` BD chains, not 8 devices.

This is the cheapest in routing complexity (single switchbox config
for all 8 phases), and is feasible *only because all 8 phases use the
same set of tiles*. It is most powerful for the eltwise + matvec
combination where the matvec and eltwise both target row‑2 cores with
the same channel pattern — they could share the same lock‑guarded BD
descriptor pool.

Limit: each shim channel has a fixed BD pool (typically 16 BDs).
8 phases × ~4 BDs/phase = 32 BDs needed per channel — exceeds the BD
pool. Would need to either (i) reuse BDs by reconfiguring them between
phase runs (which is mostly what aiex already does — that's where most
of the reconfig time goes), or (ii) split the 8 phases across 2–3
devices each with its own BD pool.

## 5. Synchronization between rows

Inside one packed device the phases run **serially** (data‑dependent).
The runtime sequence in the dispatcher would look like:

```mlir
aiex.runtime_sequence @o_gemv_ffn_packed(%args...) {
  // phase A: og + a1_add (rows 2,3)
  aiex.dma_configure_task ... // shim BDs for x, weights, res1
  aiex.dma_start_task ...
  aiex.dma_await_task ...      // res1 done
  // phase B: rm + gg||ug + sw (rows 2,3,4,5)
  aiex.dma_configure_task ...
  aiex.dma_start_task ...
  aiex.dma_await_task ...      // swiglu done
  // phase C: dg + a2 (rows 2,3)
  ...
}
```

Inter‑row synchronization uses memtile locks, not host‑side awaits.
The wait points only exist for shim‑facing transactions (initial input
load, final output drain, and weight loads per matvec).

## 6. Migration path (start small)

This is the corrected path after the packet‑routing study (§10) and the
scatter/broadcast audit (§11). The canonical target is:

* `o_gemv_ffn`: **8 → 4 devices**: D1{og,a1}, D2{rm},
  D3{gg,ug,sw}, D4{dg,a2}.
* `rms_gemv_rope`: **6 → 2 devices**: RGR1{r_rms},
  RGR2{q,k,v + in‑core rope}.

1. **Measure first with a dispatch microbenchmark.** Host timing around
   `KernelCache.load_and_run` measures the outer XRT launch, not just
   inner `aiex.configure` / `aiex.run` PDI cost. Build dispatcher
   variants with 0, 1, 2, 4, 8, and 16 lightweight inner runs under
   `--expand-load-pdis`, then fit the slope. If the added cost is
   <5 µs/run, packing is likely noise; if it is >20 µs/run, the
   rewrite has a clear target.

2. **First POC: D1 `og_matvec + a1_eltwise_add`.** og runs on row 2,
   a1_add on row 3, and the `proj` hand‑off stays in the per‑column
   memtile L2 buffer. This avoids the `proj_buf` DDR write+read, keeps
   the hand‑off element‑wise, and validates packet‑routed shim inputs
   plus co‑resident core programs.

3. **Second POC: D3 `gg_matvec ‖ ug_matvec → sw_silu_mul`.** This is
   the dense routing case: two matvec rows feed a SiLU/mul row through
   two L2 chains (`gate`, `up`) while weight streams are packet‑muxed.

4. **Fill in the easy pieces.** D4 `dg_matvec + a2_eltwise_add` is the
   same element‑wise L2 pattern as D1. D2 remains `rm_rms` alone because
   both adjacent hand‑offs are scatter/gather or broadcast boundaries.

5. **Then apply the same principle to `rms_gemv_rope`.** Keep RGR1
   `r_rms` as a boundary at the single→broadcast transition, then pack
   q/k/v matvecs into RGR2 and fold RoPE into the matvec cores.

Stretch goals — 8→2 or 8→1 FFN, 6→1 RGR, cross‑column memtile packet
routing, or relying on L2 contents surviving PDI swaps — should wait
until the 4‑device FFN target works and is measured.

## 7. What this does **not** help

* **lm_head_gemv (8 partition segs).** They’re already structurally
  identical 40‑flow matvecs differing only in weight partition; the
  reconfig between them is mostly BD swap, not switchbox change.
  Packing all 8 partitions into one device would need 8 row‑bank
  matvecs sharing one shim‑col‑0 broadcast — which exceeds the memtile
  6+6 budget unless we packet‑mux. Probably skip.
* **The AWQ `_run_o_ffn_awq_experimental` direct path.** It already
  decomposes into 4 standalone awq_gemv kernels + CPU glue. Different
  optimization axis (data movement, not reconfig).
* **Prefill.** Prefill phases use the full 32‑core 4×8 herd already,
  so there are no spare rows to pack into.

## 8. Risk summary

| risk                                          | mitigation                                              |
| --------------------------------------------- | ------------------------------------------------------- |
| Memtile DMA channel exhaustion (>6+6)         | Chain phases through L2; never have >6 streams in flight |
| Shim DMA channel exhaustion (>2+2)            | Drop intermediates; only load weights + final result    |
| Switchbox source/dest collision               | Disjoint (tile,bundle,channel) tuples per phase         |
| Increased PDI size                            | Larger ELF + more BDs per device (~2–3× current per device, but fewer devices) |
| Compile‑time routing failure                  | Start with 2‑phase pack, grow incrementally             |
| BD pool exhaustion (16 BDs/channel)           | Cap at ~3 phases per shim channel; rebalance otherwise  |

## 9. Recommendation

**Superseded by §10.10 / §11.10.** This section's recommendation was
written before the packet‑routing study (§10) and the scatter‑gather
audit (§11). Read §10.10 for the realistic POC and §10.8 / §11.9 for
the realistic dispatch and DDR savings.

Short version of the current recommendation: start with the
**`og + a1_add` 2‑phase pack** (§10.10 / §11.10) — element‑wise L2
chain on proj, packet‑routed shim inputs, no scatter‑gather. If that
proves out, build the **D3 `gg ‖ ug → sw_silu_mul` 3‑phase pack** —
the harder routing test (parallel matvecs, two intra‑device L2 chains,
packet‑muxed weights). Then fill in D2 (rm alone) and D4 (dg + a2),
yielding the 8→4 device pack for o_gemv_ffn. Stretch goals (8→2 or
8→1 via on‑chip scatter‑gather, or via memtile surviving PDI swap)
are §11.7 / §11.8 — investigate only after the 4‑device pack lands.

---

## 10. Packet routing on the shim — "phased routing"

§3 declared the shim's 2 MM2S + 2 S2MM DMA channel budget the killer
constraint for row‑per‑phase packing. That accounting assumed **circuit
routing**, where one physical switchbox channel ties exactly one source
port to one destination port for the lifetime of the PDI. AIE2/npu2
also supports **packet routing**, which dissolves that limit. This
section studies whether packet routing on the shim removes the §3
blocker.

### 10.1 What packet routing actually is (npu2 specifics)

Per `aie/Dialect/AIE/IR/AIEOps.td` and `AIETargetModel.cpp`:

* Each packet carries a 5‑bit ID (`pkt_id`) in a 4‑byte header.
* Each switchbox has **6 arbiters × 4 master‑select (msel) values =
  24 (arbiter, msel) slots**.
* Each **source port** in a switchbox can host up to **4 packet rules**
  (`aie.rule`). A rule is `(mask, value) → (arbiter, msel)`, mapping a
  packet ID class to a route.
* Each **master port** is configured by an `aie.masterset(dest, amsels...)`
  selecting one arbiter and one or more msels — different msels from the
  same arbiter can activate the same master port.
* Packet header can be **stripped** at the destination (default) or
  preserved (`keep_pkt_header`). For ≥256‑byte payloads the 4‑byte
  overhead is <2%.

So one physical shim DMA channel can fan out to up to **4 different
destination switchbox endpoints** keyed by packet ID, with no PDI
reconfiguration between fan‑outs.

### 10.2 Prior art in this repo

This is not theoretical — the awq_gemv reference kernels in
`reference_mlir/awq_gemv_k{2048,8192}_*_vecdeq.npu.air.mlir` already use
packet routing on the shim:

```mlir
aie.packet_flow(0) { aie.packet_source<%shim_0_0, DMA:0> aie.packet_dest<%tile_0_2, DMA:0> }
aie.packet_flow(1) { aie.packet_source<%shim_0_0, DMA:0> aie.packet_dest<%tile_0_2, DMA:0> }
aie.packet_flow(2) { aie.packet_source<%shim_0_0, DMA:0> aie.packet_dest<%tile_0_2, DMA:0> }
aie.flow(%tile_0_2, DMA:0, %shim_0_0, DMA:0)
aie.shim_dma_allocation @air_channel_3 (%shim_0_0, S2MM, 0)
aie.shim_dma_allocation @air_channel_0 (%shim_0_0, MM2S, 0)  // pkt_id 0
aie.shim_dma_allocation @air_channel_1 (%shim_0_0, MM2S, 0)  // pkt_id 1
aie.shim_dma_allocation @air_channel_2 (%shim_0_0, MM2S, 0)  // pkt_id 2
```

Three logically distinct shim streams (x vector, qweight, params) share
**one physical shim MM2S 0 channel**, distinguished only by `pkt_id` in
the BD descriptor:

```mlir
aie.dma_bd(%arg0 ...) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}  // x
aie.dma_bd(%arg1 ...) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}  // q
aie.dma_bd(%arg2 ...) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}  // p
```

`builders/awq_matvec.py:304‑343` emits this directly via the
`packetflow` helper. So both the lowering machinery and the runtime path
are exercised in production code in this codebase.

### 10.3 What packet routing changes for row‑packing

The §3 conflict was:

> Phase A → shim col c S2MM 0, Phase B → shim col c S2MM 0 — conflict!
> Both write to same shim channel.

With packet routing this stops being a conflict. Reframed:

| budget                              | with circuit routing       | with packet routing                          |
| ----------------------------------- | -------------------------- | -------------------------------------------- |
| Distinct logical streams per shim DMA channel | 1                  | **up to 4** (limited by 4 packet rules / port) |
| Phases sharing one shim channel     | 1                          | up to 4 (1:1 with pkt_id)                    |
| Switchbox reconfig between phases   | required if dst differs    | **none** — packet routing is static          |

A row‑per‑phase packing with packet flows then needs:

* **Per shim MM2S channel**: ≤ 4 packet IDs → ≤ 4 phases fanning out
  from that channel. The 4‑phase FFN sub‑pipelines fit exactly.
* **Per shim S2MM channel**: ≤ 4 packet IDs → ≤ 4 phases fanning **into**
  that channel from different sources.
* **Per switchbox**: ≤ 24 amsel slots. Even densely‑packet‑routed designs
  rarely come close. The current awq_gemv kernel uses 3 amsel slots; a
  4‑row‑packed FFN device would use ~12–16.

### 10.4 Concrete packing with packet routing

Re‑doing §4a's 4‑matvec stretch goal with packet routing on the shim:

```
           col 0 ... col 7
row 5  [og og og og og og og og]    ← phase: og_matvec (pkt_id 0)
row 4  [gg gg gg gg gg gg gg gg]    ← phase: gg_matvec (pkt_id 1)
row 3  [ug ug ug ug ug ug ug ug]    ← phase: ug_matvec (pkt_id 2)
row 2  [dg dg dg dg dg dg dg dg]    ← phase: dg_matvec (pkt_id 3)
row 1  [M  M  M  M  M  M  M  M ]    ← memtiles
row 0  [S  S  S  S  S  S  S  S ]    ← shim
```

Shim packet flows per column c:

```mlir
// Per‑column weight stream: shim col c MM2S 0 fans out to 4 rows by pkt_id
aie.packet_flow(0) { src %shim_c_0,DMA:0 → dst %tile_c_5,DMA:0 }  // og weight
aie.packet_flow(1) { src %shim_c_0,DMA:0 → dst %tile_c_4,DMA:0 }  // gg weight
aie.packet_flow(2) { src %shim_c_0,DMA:0 → dst %tile_c_3,DMA:0 }  // ug weight
aie.packet_flow(3) { src %shim_c_0,DMA:0 → dst %tile_c_2,DMA:0 }  // dg weight

// Output stream: 4 rows → shim col c S2MM 0 (only one fires at a time)
aie.packet_flow(0) { src %tile_c_5,DMA:0 → dst %shim_c_0,DMA:0 }  // og output
aie.packet_flow(1) { src %tile_c_4,DMA:0 → dst %shim_c_0,DMA:0 }  // gg output
aie.packet_flow(2) { src %tile_c_3,DMA:0 → dst %shim_c_0,DMA:0 }  // ug output
aie.packet_flow(3) { src %tile_c_2,DMA:0 → dst %shim_c_0,DMA:0 }  // dg output

// Vector broadcast: shim col 0 MM2S 1 fans out to 4 rows × 8 cols
// (multicast packet flow with multiple packet_dest ops per pkt_id)
```

Runtime sequence picks the phase by selecting which BD chain to fire
and tagging its dma_bd with the matching `pkt_id`:

```mlir
// Phase: og_matvec
%bd_og = aiex.dma_configure_task_for @col0_weight_channel {
  aie.dma_bd(%arg_w_og : ...) {packet = #aie.packet_info<pkt_id = 0>}
  aie.end
}
aiex.dma_start_task(%bd_og)
... (await og output)

// Phase: gg_matvec — same channel, different pkt_id
%bd_gg = aiex.dma_configure_task_for @col0_weight_channel {
  aie.dma_bd(%arg_w_gg : ...) {packet = #aie.packet_info<pkt_id = 1>}
  aie.end
}
aiex.dma_start_task(%bd_gg)
```

**The switchbox is configured once at xclbin load and never touched
again between phases. The only thing that changes is which BD descriptor
fires.** This is the "phased routing" the user is asking about.

### 10.5 The constraints that *don't* go away

Packet routing is not free, and these are still real:

1. **4 packet rules per source port.** Hard ceiling on phases sharing a
   single shim channel. For >4 phases, use the other physical channel
   (MM2S 1, S2MM 1) — gets you to 8 phases per col, which covers the
   o_gemv_ffn 8‑phase pipeline exactly.

2. **6 arbiters × 4 msels per switchbox.** With 8 phases × 2 streams
   per phase × per‑column fan‑out, the per‑switchbox amsel demand grows
   roughly with `phases × concurrent_streams`. Tight but feasible —
   exact count needs to be checked against the placer/router output.

3. **Memtile DMA budget (6+6) still applies.** Packet routing does not
   add memtile channels. If every phase wants its own memtile staging,
   you still hit the §3 wall. Workaround: time‑multiplex memtile
   channels (different BD chains on same channel, gated by run order)
   or chain via L2 (§4a) so only 1 phase's memtile traffic is in flight
   at a time.

4. **Per‑channel BD pool (~16 BDs on AIE2 shim).** 4 phases × ~4 BDs
   per phase ≈ 16. Exactly the budget for the 4‑row pack. 8 phases
   sharing one channel would exceed; would need to either reuse BDs
   (configure between phases — partial reconfig cost returns) or split
   across multiple channels.

5. **Packet header overhead (4 B / packet).** Negligible at decode
   matvec sizes — K=2048 bf16 = 4 KB per packet → 0.1% overhead.
   Worth noting for very short bursts; not a concern here.

6. **Determinism / ordering.** Packet routing introduces arbiter
   contention. For serial phase execution it's fine (only one BD chain
   in flight). If you ever wanted to overlap two phases on the same
   physical channel, the arbiter‑fair scheduling could interleave them
   — but the row‑pack design already assumes serial.

### 10.6 What this unlocks vs §4 recommendations

§4 recommended three workable variants:
* **4a (memtile‑chained)**: fuse element‑wise hand‑offs by keeping
  intermediates in L2 instead of writing them to DDR.
* **4b (packet flow)**: dismissed as insufficient alone.
* **4c (BD pool multiplex)**: limited by 16 BDs/channel.

This study elevates **4b** from a footnote to the **primary mechanism**
for row‑per‑phase packing. It directly solves the §3 shim conflict
that ruled out the naive layout. The realistic combined story:

* **Use packet routing on the shim** to coexist 4 phases' streams on
  one channel with no switchbox reconfig — this is what makes the
  "one phase per row" layout statically routable.
* **Use memtile chaining (§4a)** for intermediates between phases so
  shim S2MM doesn't have to write out then re‑read every transient.
  This keeps the per‑channel packet‑rule count low (≤4 per shim
  channel) and the memtile DMA channel count manageable.
* **Use BD‑pool multiplex (§4c)** only when going beyond 4 phases per
  channel — i.e. if you want a single 8‑phase device rather than two
  4‑phase devices.

### 10.7 Revised packing target

§11 establishes that three hand‑offs in `o_gemv_ffn` are not simple
element‑wise L2 chains but **scatter→gather / scatter→broadcast**
transitions (a1→rm, rm→gg/ug, sw→dg). On‑chip realization of those
requires cross‑col memtile transfers that blow out the memtile DMA /
amsel budget. The pragmatic choice is to **place device boundaries
at the scatter‑gather points** so DDR continues to do the cross‑col
work (as it does today), and pack only the element‑wise chains
inside each device.

**`o_gemv_ffn` 8 → 4 devices:**

| device | phases                            | rows used | intra‑device L2 chain   | boundary type (out → next) |
| ------ | --------------------------------- | --------- | ----------------------- | -------------------------- |
| `D1`   | og(r2) → a1_add(r3)               | 2, 3      | proj                    | scatter→gather (res1 → rm) |
| `D2`   | rm_rms (col 0, r2)                | 2         | —                       | single→broadcast (normed2 → gg/ug) |
| `D3`   | gg(r2) ‖ ug(r3) → sw_silu_mul(r4) | 2, 3, 4   | gate, up                | scatter→broadcast (swiglu → dg) |
| `D4`   | dg(r2) → a2_add(r3)               | 2, 3      | down                    | element‑wise (output → host) |

→ **4 device dispatches per FFN layer instead of 8** (2× reduction).

For `rms_gemv_rope` (6 phases): r_rms's normed output must broadcast to
all 8 cols × 3 matvec rows (24 destinations). That's beyond memtile
fan‑out budget without packet routing on memtile (untested in this
codebase). So device boundary stays at the broadcast.

**`rms_gemv_rope` 6 → 2 devices:**

| device  | phases                                   | rows used | intra‑device L2 chain | boundary type |
| ------- | ---------------------------------------- | --------- | --------------------- | ------------- |
| `RGR1`  | r_rms (col 0, r2)                        | 2         | —                     | single→broadcast (normed → q/k/v) |
| `RGR2`  | q_matvec(r2) ‖ k_matvec(r3) ‖ v_matvec(r4) + in‑core rope | 2, 3, 4 | (matvec output → in‑core rope, no memtile) | element‑wise (q/k/v → host) |

→ **2 device dispatches per attn‑pre layer instead of 6** (3× reduction).

### 10.8 Token‑level math (revised, realistic)

| kernel                    | current segs/token | revised (§10.7) | savings |
| ------------------------- | ------------------ | --------------- | ------- |
| `rms_gemv_rope` × 16      | 96                 | 32              | 64      |
| `o_gemv_ffn` × 16         | 128                | 64              | 64      |
| `lm_head_gemv` × 1        | 8                  | 8 (unchanged)   | 0       |
| **total**                 | **232**            | **104**         | **55% reduction** |

This is below the 76% claimed in an earlier draft (which incorrectly
assumed all hand‑offs were element‑wise L2 chains), and also below the
earlier §4a ~60% estimate. The revised number is the conservative
target after keeping the scatter/gather and broadcast hand‑offs at DDR
boundaries.

A stretch target of ~75% reduction is reachable only if (a) we solve
the cross‑col scatter→broadcast hand‑offs on‑chip via memtile packet
routing (8 → 2 devices for `o_gemv_ffn`, 6 → 1 for `rms_gemv_rope`),
or (b) memtile contents survive PDI swap (§11.9), letting us treat
adjacent devices as a single L2 namespace.

### 10.9 Risks specific to packet routing

| risk                                          | mitigation                                              |
| --------------------------------------------- | ------------------------------------------------------- |
| Packet rule count > 4 per source port         | Use both MM2S 0 and MM2S 1; split phases across channels |
| Arbiter / msel exhaustion (>24 per switchbox) | Cap concurrent packet flows per switchbox; route some via N/S/E/W to neighbor switchboxes |
| BD pool < total phase BDs                     | Stay at 4 phases per channel, not 8                     |
| Packet ordering at S2MM with multiple sources | Phases run serially in this design; single‑source‑at‑a‑time, so order is trivial |
| Header parsing / addressing of fanned‑out streams | `keep_pkt_header=false` strips header; destination BD config handles address |
| Router pass fails to find a valid arbiter assignment | Fall back to splitting into 2 devices instead of 1; placer is the constraint |

### 10.10 Recommended next concrete step

The minimal proof‑of‑concept to validate packet‑routed phased
dispatch end‑to‑end:

**Build a 2‑phase packed device combining `og_matvec` (row 2) +
`a1_eltwise_add` (row 3)** using packet routing on the col‑0 shim
MM2S channels:

* `shim col c MM2S 0` carries weight (pkt_id 0) for og and operand B
  (pkt_id 1) for a1_add.
* `shim col 0 MM2S 1` carries the attn_out broadcast (pkt_id 0) for og
  and the x residual (pkt_id 1) for a1_add.
* Intermediate proj output stays in memtile (chained from row 2 → row
  3 via memtile DMA), not back to DDR.
* `shim col c S2MM 0` carries final res1 (pkt_id 1, from a1_add).

This exercises:
1. Multi‑pkt‑id shim packet flows for inputs ✓
2. Memtile L2 chaining between rows ✓
3. Single shim S2MM output for the packed device ✓
4. Co‑resident core programs on rows 2 and 3 in one device ✓

Compared to the cached AIR output (separate og + a1 devices), success
criteria:
* 1 XRT call (was 2), 1 `aiex.run` (was 2) — confirmed via emitted IR.
* Correctness within bf16 tolerance vs the BF16 reference.
* Measured token‑time reduction matches predicted savings from §10.8.

If the POC validates, scale to the 4‑phase `D1_AT`/`D2_UV` packs
(§10.7).

### 10.11 What I don't know yet

* **Whether the mlir‑aie router/placer correctly handles 4 packet
  flows on one shim channel with disjoint destinations.** The
  awq_gemv kernel does 3 flows to *the same* destination — easier
  case. The 4‑destination fan‑out is the routability question. Worth
  a smoke test before committing to the bigger pack.
* **Whether `aie-expand-load-pdis` correctly preserves packet‑flow
  amsel/masterset configuration in the inlined write32/blockwrite
  stream.** Need to inspect post‑expansion IR for the awq_gemv kernel
  and confirm the switchbox config writes are present.
* **Real per‑seg reconfig cost.** §6 measurement methodology still
  applies — packet routing only saves time if reconfig is actually
  the bottleneck. Measure first.

---

*Sources for §10: `aie/Dialect/AIE/IR/AIEOps.td` (PacketFlowOp,
AMSelOp, MasterSetOp, PacketRulesOp), `reference_mlir/awq_gemv_*`
(working shim packet‑flow IR), `builders/awq_matvec.py:304‑343`
(Python emitter), `mlir-aie/programming_examples/basic/packet_switch/
aie_add_placed.py` (multi‑destination packet flow reference).*

## 11. Memtile‑chained pipeline — per‑intermediate audit

§10.10 floated the idea that intermediates should live in L2 rather than
round‑trip through DDR, but never specified *which* intermediates and
*where* they live in the packed design. This section does that audit.

### 11.1 Current state — every intermediate is DDR‑resident

From `llama32_1b_decode.py:281‑302`, the `o_gemv_ffn` call passes 8
intermediate buffers to XRT:

| arg | name        | size (bf16)    | producer phase | consumer phase(s)     |
| --- | ----------- | -------------- | -------------- | ---------------------- |
| 2   | proj_buf    | 2048 → 4 KB    | og_matvec      | a1_eltwise_add         |
| 4   | res1_buf    | 2048 → 4 KB    | a1_eltwise_add | rm_rms **AND** a2_eltwise_add |
| 6   | normed2_buf | 2048 → 4 KB    | rm_rms         | gg_matvec **AND** ug_matvec   |
| 8   | gate_buf    | 8192 → 16 KB   | gg_matvec      | sw_silu_mul            |
| 10  | up_buf      | 8192 → 16 KB   | ug_matvec      | sw_silu_mul            |
| 11  | swiglu_buf  | 8192 → 16 KB   | sw_silu_mul    | dg_matvec              |
| 13  | down_buf    | 2048 → 4 KB    | dg_matvec      | a2_eltwise_add         |
| 14  | output_buf  | 2048 → 4 KB    | a2_eltwise_add | host (final)           |

The `intermediate_indices={2,4,6,8,10,11,13,14}` flag tells the cache
layer to skip the host‑side copy‑back, but the *NPU‑side* DMA still
writes every one of these to a DDR‑mapped BO and reads it back into
the next phase. Per‑call DDR traffic for intermediates alone:

`(4 + 4 + 4 + 16 + 16 + 16 + 4 + 4) KB × 2 (write+read) = 136 KB`

×16 layers × N tokens = **~2.2 MB per token of pure intermediate
shuffle**. Plus weights (read‑only, much larger) and the residual/x
inputs.

### 11.2 Hand‑off taxonomy — element‑wise vs scatter‑gather

The herd partitions outputs across 8 columns. Whether a hand‑off
fits a simple L2 chain depends on whether the producer's per‑col
output partition matches the consumer's per‑col input partition.

* **Element‑wise (same partition).** Producer col c writes its slice
  to mem_tile_c_1; consumer col c reads its slice from the same
  memtile. No cross‑col data movement. **L2 chain is trivial.**
* **Scatter→gather (8 cols → 1).** Producer writes 8 col‑slices;
  consumer needs the assembled full vector on a single tile. Today
  this happens by all cols writing to a DDR buffer and the single
  consumer reading the full DDR range.
* **Single→broadcast (1 → 8 cols).** Producer is single‑tile and
  writes a full vector; every consumer col needs the full vector
  replicated in its L1. Today this is shim col‑0 MM2S 1 with a
  multicast flow reading from a DDR buffer.
* **Scatter→broadcast (8 cols of M‑slice → 8 cols of full‑K vector).**
  Producer writes per‑col M‑slices; consumer is a matvec whose K‑axis
  needs the *full* assembled vector broadcast to every col. Today
  this combines a gather and a re‑broadcast through DDR (each col
  writes its slice; shim col‑0 reads the full vector back and
  multicasts).

On‑chip realization of the last three patterns requires cross‑col
memtile traffic (col 0 memtile reading from cols 1..7 memtiles via
east‑west switchbox flows, or broadcasting in the other direction).
That adds amsel pressure and memtile DMA channels well beyond the
6+6 budget unless memtile packet routing carries it — which exists
syntactically but isn't exercised in this codebase, so its actual
limits are unverified.

The pragmatic position: keep scatter‑gather hand‑offs at *device
boundaries* (DDR continues to do the work it does today) and only
fuse the element‑wise hand‑offs intra‑device.

### 11.3 Per‑intermediate audit under the §10.7 packing

§10.7 (revised) splits `o_gemv_ffn` into four devices: D1{og,a1},
D2{rm}, D3{gg,ug,sw}, D4{dg,a2}.

| intermediate | producer (dev/row) | consumer(s)               | class                | size      | proposed location |
| ------------ | ------------------ | ------------------------- | -------------------- | --------- | ----------------- |
| proj         | D1 / r2 (og)       | D1 / r3 (a1_add)          | element‑wise         | 4 KB tot  | L2 memtile (intra) |
| res1         | D1 / r3 (a1_add)   | D2 (rm) **+ D4 (a2_add)** | scatter→gather **+** later element‑wise | 4 KB | DDR (both consumers cross device boundary) |
| normed2      | D2 (rm, col 0)     | D3 (gg, ug)               | single→broadcast     | 4 KB tot  | DDR (boundary is the broadcast) |
| gate         | D3 / r2 (gg)       | D3 / r4 (sw_silu_mul)     | element‑wise         | 16 KB tot | L2 memtile (intra) |
| up           | D3 / r3 (ug)       | D3 / r4 (sw_silu_mul)     | element‑wise         | 16 KB tot | L2 memtile (intra) |
| swiglu       | D3 / r4 (sw)       | D4 (dg)                   | scatter→broadcast    | 16 KB tot | DDR (boundary is the scatter→broadcast) |
| down         | D4 / r2 (dg)       | D4 / r3 (a2_add)          | element‑wise         | 4 KB tot  | L2 memtile (intra) |
| output       | D4 / r3 (a2_add)   | host                      | external             | 4 KB tot  | DDR (final write) |

Four element‑wise hand‑offs go L2 (proj, gate, up, down).
Three scatter/broadcast hand‑offs stay on DDR (res1, normed2, swiglu).

DDR traffic comparison (intermediates only, per FFN call):

| flow                                    | current | with §11.3 packing |
| --------------------------------------- | ------- | ------------------ |
| proj write + read                       | 8 KB    | 0                  |
| res1 write + read (a1 → rm)             | 8 KB    | 8 KB               |
| res1 read (a2_add second consumer)      | 4 KB    | 4 KB               |
| normed2 write + read (rm → gg/ug)       | 8 KB    | 8 KB               |
| gate write + read                       | 32 KB   | 0                  |
| up write + read                         | 32 KB   | 0                  |
| swiglu write + read (sw → dg broadcast) | 32 KB   | 32 KB              |
| down write + read                       | 8 KB    | 0                  |
| output write                            | 4 KB    | 4 KB               |
| **total**                               | **136 KB** | **56 KB (~59% drop)** |

Per token: 16 layers × (136 → 56) ≈ 1.3 MB intermediate DDR traffic
saved. Less than the 1.4 MB I previously claimed; close enough that
the bandwidth argument still holds.

### 11.4 L2 footprint — multi‑tile and pipelining

I previously wrote "single buffer is enough" for the L2 chains. That
assumes strict serial execution with no producer/consumer overlap.
For real pipelined execution within a device you need at minimum
**double buffering** (ping‑pong) — 2× the per‑col tile in L2 — to let
the producer write tile N+1 while the consumer reads tile N.

Per‑col L2 footprint for the element‑wise chains (worst case
double‑buffered):

| chain               | per‑col tile | double‑buffered | when both tiles live in L2 |
| ------------------- | ------------ | --------------- | -------------------------- |
| proj (og → a1)      | 256 B        | 512 B           | yes (pipeline)             |
| gate (gg → sw)      | 2 KB         | 4 KB            | yes                        |
| up (ug → sw)        | 2 KB         | 4 KB            | concurrent with gate buffer |
| down (dg → a2)      | 256 B        | 512 B           | yes                        |

Concurrent peak per‑col L2 use during D3 (gg ‖ ug → sw):

* gate double‑buffer: 4 KB
* up double‑buffer: 4 KB
* weight staging buffers for gg and ug (each ~32 KB for K=2048 ×
  bf16 × tile‑factor): up to 64 KB
* total ≈ 72 KB per memtile — fits comfortably in the 512 KB budget.

If consumer order *doesn't* match producer order — which it does for
the element‑wise chains here — you'd need to buffer the full
intermediate, not just two tiles. Not a concern in this packing
because consumer order matches producer order for every kept L2
chain. Worth re‑checking if the kernel tile schedules ever shift.

### 11.5 Routing budget impact of L2 chaining

D3 is the densest packing (3 phases, rows 2/3/4 active). Memtile DMA
channels per col c during D3:

| stream                                              | dir at memtile | count |
| --------------------------------------------------- | -------------- | ----- |
| shim → memtile (gg weight, pkt_id 0)                | S2MM           | 1     |
| shim → memtile (ug weight, pkt_id 1)                | S2MM           | 1 (shared phys ch via packet) |
| memtile → r2 (gg weight stream)                     | MM2S           | 1     |
| memtile → r3 (ug weight stream)                     | MM2S           | 1     |
| r2 → memtile (gate, gg output)                      | S2MM           | 1     |
| r3 → memtile (up, ug output)                        | S2MM           | 1     |
| memtile → r4 (gate, sw input 0)                     | MM2S           | 1     |
| memtile → r4 (up, sw input 1)                       | MM2S           | 1     |
| r4 → memtile (swiglu, sw output)                    | S2MM           | 1     |
| memtile → shim (swiglu DDR spill, boundary)         | MM2S           | 1     |
| shim → memtile (normed2 broadcast for gg + ug input vector) | S2MM | 1 (shim‑side packet handles broadcast) |
| memtile → r2 (vector broadcast for gg)              | MM2S           | 1     |
| memtile → r3 (vector broadcast for ug)              | MM2S           | 1     |

Per col: **6 MM2S + 5 S2MM** if we count packet‑muxed shim arrivals
as one physical S2MM each. Memtile budget is 6+6. **Fits, tight.**

Col 0 might need 1 extra MM2S for col‑0‑only flows; would push to 7
MM2S → overflow. Mitigation: split the vector broadcast off the
memtile (let shim col‑0 MM2S 1 packet‑broadcast directly to both r2
and r3 across cols, bypassing memtile for that stream). That drops
2 memtile MM2S, comfortably fitting.

D1, D2, D4 are all simpler (fewer concurrent phases) so memtile
budget is not a concern.

### 11.6 rms_gemv_rope — also has the broadcast problem

I previously claimed rms_gemv_rope packs into 1 device with all
intermediates L2‑resident. That was wrong: r_rms's `normed` output
must broadcast to q/k/v matvec inputs as the K‑axis vector, replicated
across all 8 cols × 3 rows (24 destinations). Memtile MM2S only has 6
physical channels; the 24‑destination fan‑out exceeds the budget
unless memtile packet routing carries it. Memtile packet routing
exists in the dialect but isn't exercised in this codebase, so its
amsel/rule budget at memtile endpoints is unverified.

Honest packing for rms_gemv_rope (revised, §10.7 above): two devices.

**RGR1 (r_rms alone, single tile col 0 row 2):** writes normed to DDR
via shim S2MM 0 (same as today). 3 flows.

**RGR2 (q ‖ k ‖ v matvec on rows 2/3/4 + in‑core rope post):**
* Read normed from DDR via shim col‑0 MM2S 1, packet‑broadcast to all
  3 rows × 8 cols (today's pattern works as‑is — this is what shim
  already does well).
* Each row's matvec writes its M‑slice per col to its own memtile
  region.
* Rope post folds into the matvec kernel (no extra memtile hop).
* Each compute writes the final per‑col q/k/v slice out via shim S2MM.

Intra‑device L2 hand‑offs in RGR2: **none** (rope is in‑core, matvec
output goes straight to shim). The "L2 chain" benefit is zero here —
the saving comes purely from collapsing 3 matvec dispatches into 1
device, which packet routing on the shim makes possible (§10).

→ 6 → 2 devices, 3× dispatch reduction.

### 11.7 Single‑device stretch (revised)

The cleanest way to eliminate scatter‑gather DDR cost is collapsing
o_gemv_ffn into a **single** aie.device so all hand‑offs become
intra‑device. That requires on‑chip realization of:

* **a1 → rm gather** (8 col‑slices of res1 → full 2048 vector on col 0)
* **rm → gg / ug broadcast** (full 2048 vector on col 0 → all 8 cols
  on rows 2 and 3)
* **sw → dg scatter→broadcast** (8 col‑slices of swiglu → full 8192
  vector replicated to all 8 cols on the dg row)

Each requires cross‑col memtile flows that are not used anywhere in
the existing AIR‑lowered code. Specifically:

| pattern              | flow shape needed                                       |
| -------------------- | ------------------------------------------------------- |
| gather to col 0      | memtile_c_1 (c=1..7) → memtile_0_1, east/west switchbox |
| broadcast from col 0 | memtile_0_1 → tile_c_r (c=0..7), 8‑way multicast        |
| scatter→broadcast    | each memtile_c_1 → all memtile_*_1, then standard memtile→compute fan‑out |

The first two would consume up to 7 cross‑col flows each, going
through the east‑west switchbox arbitration. The third is a full
all‑to‑all and is unlikely to route cleanly. So the single‑device
form is feasible *only* if memtile packet routing scales to 24+
destinations per source — which is unverified, and likely runs into
the per‑switchbox amsel budget (24 slots) before getting there.

Stretch value is real (~90% dispatch reduction, ~95% DDR
intermediate traffic reduction) but the routing risk is high. The
§10.7 4‑device pack is the realistic target.

### 11.8 Can L2 contents survive a PDI swap? — open question

The 4‑device pack still has 3 DDR boundaries (D1→D2, D2→D3, D3→D4).
If memtile *contents* (RAM) survive a PDI swap and only the *control
state* (locks, BDs, switchbox) is reset, then the next device's first
action could be: configure new BDs that *assume* the memtile already
holds the prior phase's output at known offsets, set the locks to
"consumer‑ready", and stream them into the next device's compute rows
without any DDR detour.

This would drop res1, normed2, and swiglu boundary spills (44 KB of
the remaining 56 KB DDR traffic), leaving only the 4 KB final output
plus 8 KB of res1's second consumption — **~92% intermediate DDR
reduction** versus today.

I don't know whether this works on npu2. Things to check:

1. Whether the `--expand-load-pdis` configuration sequence writes to
   memtile data RAM addresses (would clobber contents) or only to
   memtile control/lock/BD registers.
2. Whether the device reset triggered by the `@empty_N` load_pdi
   clears tile memory.
3. Whether the host can pre‑initialize lock states in the next
   device's load sequence to claim "this data is already in your
   memtile."

Investigation plan: build a minimal 2‑device xclbin where D1 writes a
known pattern to a memtile buffer at known offset, D2 reads it back
(via shim) without D1 writing it to DDR first. Diff output against
expected pattern. If it matches, memtile survives. If not, expected.

Low‑priority follow‑on; the §11.3 ~59% reduction is worth getting
first.

### 11.9 Summary — corrected L2 chain map

| pipeline                | element‑wise L2 chains (kept)             | scatter‑gather / broadcast (DDR boundary) |
| ----------------------- | ------------------------------------------ | ------------------------------------------ |
| `rms_gemv_rope` D1{rm}  | none                                       | normed → q/k/v (single→broadcast)         |
| `rms_gemv_rope` D2{q‖k‖v + rope} | matvec → in‑core rope (no memtile) | q/k/v → host                              |
| `o_gemv_ffn` D1{og,a1}  | proj (og → a1)                             | res1 → rm (scatter→gather)                |
| `o_gemv_ffn` D2{rm}     | none                                       | normed2 → gg/ug (single→broadcast)        |
| `o_gemv_ffn` D3{gg,ug,sw} | gate (gg → sw), up (ug → sw)             | swiglu → dg (scatter→broadcast)           |
| `o_gemv_ffn` D4{dg,a2}  | down (dg → a2)                             | output → host                              |

**Four element‑wise L2 hand‑offs are real wins (proj, gate, up, down).
Three scatter/broadcast hand‑offs stay on DDR by design — putting the
device boundaries there is the realistic compromise.**

### 11.10 POC update (refines §10.10)

The 2‑phase pack POC (`og + a1_add` → device D1) is unchanged and
still the right first step: both phases share the same column
partitioning, so the proj hand‑off is element‑wise and L2 chaining is
trivial. Validates:

* Multi‑pkt‑id shim packet flows for inputs (§10).
* Memtile L2 chaining between rows (§11) for an element‑wise hand‑off.
* Single shim S2MM output for the packed device.
* Co‑resident core programs on rows 2 and 3 in one device.

Success criteria from §10.10 stand; add:

* No `shim_dma_allocation` for proj_buf in emitted MLIR; only memtile
  DMA carries it.
* DDR transaction count drops by ~8 KB per FFN call (4 KB write + 4 KB
  read of proj_buf) versus baseline.
* Token‑time improvement consistent with one fewer XRT call plus the
  DDR bandwidth contribution.

After D1 ships, the next packing step is D3{gg, ug, sw_silu_mul} —
that's the test of the harder case: two parallel matvecs on rows
2/3 feeding sw_silu_mul on row 4, two intra‑device element‑wise L2
chains, packet‑muxed weight streams via shim, normed2 broadcast input
via shim col‑0 MM2S 1 (today's pattern unchanged).

## 12. Initial dispatch microbenchmark results (2026-05-27)

Implemented `tools/measure_dispatch_overhead.py` to generate and time
inner `aiex.configure` / `aiex.run` count variants under the same aiecc
`--expand-load-pdis` path used by `KernelCache`. Measurements below are
medians on this machine with warm BOs and static inputs skipped after the
warmup calls.

**No-op PDI swap control** (`--distinct-devices 2`, 200 timed iters):

| inner runs | median ms | delta vs 0, µs/run |
| ---------- | --------- | ------------------ |
| 0          | 0.146     | —                  |
| 1          | 0.154     | 7.49               |
| 2          | 0.249     | 51.47              |
| 4          | 0.319     | 43.29              |
| 8          | 0.473     | 40.83              |
| 16         | 0.781     | 39.69              |

Least-squares fit on medians: **~40.1 µs per alternating empty PDI
run**. Same-device repeats (`--distinct-devices 1`) fit to ~0 µs/run,
which confirms the benchmark is mostly seeing PDI swap/configure cost,
not Python/XRT loop overhead.

**Production-shaped add-only FFN dispatcher** (real `o_gemv_ffn`
subdevices, alternating `a1_eltwise_add_seg` / `a2_eltwise_add_seg`,
50 timed iters):

| inner add runs | median ms | delta vs 0, µs/run |
| -------------- | --------- | ------------------ |
| 0              | 0.162     | —                  |
| 1              | 0.228     | 66.00              |
| 2              | 0.366     | 102.00             |
| 4              | 0.597     | 108.75             |
| 8              | 1.056     | 111.75             |
| 16             | 1.986     | 114.00             |

Least-squares fit on medians: **~115 µs per real add-segment inner
run**. This includes the add kernel's DMA/core work as well as the PDI
configure/run overhead, so it is an upper bound for pure reconfig cost
but a useful bound for the D1/D4 style packs.

Current cached full `o_gemv_ffn` baseline from the same harness
(`--include-current-o-gemv-ffn`, 20 timed iters): **3.41 ms median**.
The dispatch side is therefore large enough to justify D1/D3/D4
packing; the measured empty-PDI swap cost alone is already above the
>20 µs/run threshold from §6.

## 13. D1/D3/D4 packing trial results (2026-05-28)

Implemented experimental placed-builder pack modes in
`builders/o_gemv_ffn.py`:

* `pack_mode="d1d4"`: D1 packs `og_matvec_bf16_0 + a1_eltwise_add_seg`,
  D4 packs `dg_matvec_bf16_0 + a2_eltwise_add_seg`; dispatcher drops from
  8 inner runs to 6.
* `pack_mode="d1d3d4"`: additionally packs D3
  `gg_matvec_bf16_0 + ug_matvec_bf16_0 + sw_silu_mul_seg`; dispatcher drops
  from 8 inner runs to 4.
* `PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE={d1d4,d1d3d4}` now selects these modes
  through `kernel_builder.aie_ir_gen.build_o_gemv_ffn_ir`; unset keeps the
  production baseline.

Both packed variants compile with aiecc. The D1/D4-only variant also runs and
matches the cached baseline exactly for the comparable outputs (`arg4`, `arg6`,
`arg11`, `arg14`). The first D1/D3/D4 attempt compiled but timed out on NPU
execution because D3 used one shim MM2S stream for a 16-destination `normed2`
broadcast into both gate and up rows. Splitting that into two conventional
8-way broadcasts (`air_channel_12` from shim column 0 for gate, `air_channel_17`
from shim column 1 for up) fixes the hang. After that change, D1/D3/D4 also
matches baseline exactly on `arg4`, `arg6`, `arg11`, and `arg14`.

Longer BO-reuse timing run, rotated order across 3 rounds, each round using
10 warmup + 100 timed iterations per variant, final output read back only:

| variant | inner runs | kernel median ms | total median ms | correctness |
| ------- | ---------- | ---------------- | --------------- | ----------- |
| baseline | 8 | 3.303 | 3.418 | reference |
| D1/D4 | 6 | 3.335 | 3.509 | exact match |
| D1/D3/D4 | 4 | 3.125 | 3.238 | exact match |

Round medians were somewhat noisy for D1/D4, but the overall result is clear
enough for prioritization: D1/D4 alone is neutral to slightly slower, while the
full D1/D3/D4 pack saves about 0.18 ms kernel time and 0.18 ms end-to-end time
versus the cached baseline under this harness. The D3 split-input fix is
therefore required for a profitable pack; without it, the first D1/D3/D4 build
compiled but timed out at runtime.

Next D3-specific experiments:

* Trace D3 row 2, row 3, row 4, and memtile channels to identify the remaining
  bottleneck now that correctness and runtime liveness are established.
* Try a D3 weight-routing variant that separates gate/up weights across shim
  MM2S channels where possible instead of packet-muxing both large streams on
  shim MM2S0 for every column.
* Validate D1/D3/D4 inside the full decode loop before promoting it beyond the
  explicit `PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=d1d3d4` experiment flag.

---

*Source data: `reference_mlir/o_gemv_ffn.npu.air.mlir`,
`reference_mlir/rms_gemv_rope.npu.air.mlir`,
`mlir-aie/lib/Dialect/AIE/IR/AIETargetModel.cpp` (npu2 channel
budgets), and `llama32_1b_decode.py` / `llama32_1b_inference.py`
(token‑loop dispatch counts).*
