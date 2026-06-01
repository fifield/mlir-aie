# Phase H — Device stitching (adapted from llama32_1b methodology)

User correctly pointed out I'd dismissed real device stitching too
quickly. This is a plan after reading the llama32_1b approach end-to-end.

## What llama32_1b actually does

Reading `DEVICE_PACKING_ANALYSIS.md`, `builders/o_gemv_ffn.py`, and
`reference_mlir/o_gemv_ffn.npu.air.mlir`:

Each MLIR module contains N `aie.device(@phase_name)` blocks (each is a
self-contained physical NPU device program — switchbox config, memtile
DMA, core ELF, runtime_sequence) followed by ONE dispatcher
`aie.device(npu2)` with `aie.runtime_sequence @main(...)` containing
a sequence of `aiex.configure @phase { aiex.run @phase_sequence(args) }`
pairs.

This is **the exact same pattern our `build_merged.py` already emits.**
Our merged ELFs (Phase A.1, C, E, F, G) are structurally identical to
llama's reference MLIRs. The dispatcher's @main sequences sub-device
runs via configure/run pairs; `--expand-load-pdis` lowers each
configure to an explicit PDI load of that sub-device's switchbox /
BD / core configuration.

What llama32_1b does that we *haven't* done is **pack multiple different
kernels onto different physical rows within ONE aie.device**. Their
"D1" pack:

```
aie.device(@d1_og_a1_pack) {
  // og_matvec on row 2 cores (8 cores, columns 0-7)
  // a1_eltwise_add on row 3 cores (8 cores, columns 0-7)
  // memtile carries proj (og's output) → a1's input via L2 fifo
  // ONE runtime_sequence covers both
}
```

One PDI configures the switchbox + BOTH kernel programs (og on row 2,
a1 on row 3) at load time. The dispatcher fires ONE `aiex.run` for the
whole packed device. Intermediate `proj` never reaches DDR — it lives
in memtile L2.

Their measured win: **8 PDI swaps → 4 per FFN layer, -6 ms/token,
+6.7% throughput** end-to-end. Packed AWQ path: +8.6%.

## What carries over to MDV6 (and what doesn't)

### What carries over

1. **The merged-MLIR / dispatcher / @main pattern** — we already use it.
2. **The `aiex.configure / aiex.run` sequencing in @main** — same code path.
3. **`--expand-load-pdis` + `--generate-full-elf`** — exactly the aiecc flags we use.
4. **Memtile L2 as a chained pipe between kernels in one device** — we don't
   do this yet; it's the new mechanism Phase H would add.
5. **Splitting work into element-wise hand-offs (L2-chainable) vs
   scatter-gather hand-offs (DDR-bound)** — directly applicable to our
   conv layer chains.

### What does NOT carry over directly

1. **"Different kernels on different rows" requires the kernels to fit on
   a subset of cores.** Llama's matvec uses 8 cores (1 row × 8 cols);
   they have 3 free rows to fill with other kernels. Our conv kernels
   use 32 cores (all 4 rows × 8 cols) — **there are no free rows**.
   To pack two convs in one device we'd need either:
   - **Half-array conv variants** (16 cores = 2 rows × 8 cols). Each conv
     at ~half throughput. Per-conv compute time roughly doubles, but the
     per-conv PDI/dispatch overhead disappears. Whether net positive
     depends on the overhead/compute ratio (overhead dominates → win).
   - **Sequential reuse of the same 32 cores with BD chain swaps.** Same
     kernel program, different weights, intermediate via memtile L2.
     This is what we want for the bottleneck conv1+conv2 (both use
     `conv3x3_fused_packed_bf16` — same kernel, different weights). The
     `aie2_multicore_ocb.py` OCB-unroll already does this for N iterations
     of the same conv with different weights via DDR-replicated input;
     extending it to chain through memtile L2 instead is straightforward
     and is the *direct mechanical analog* of llama's D1 pack.

2. **Packet routing on shim to mux 4 logical streams onto 1 physical channel.**
   Our conv kernel runtime sequence currently uses circuit routing; adding
   packet routing is a real change. Probably not needed yet — the current
   shim BD budget isn't blocking on us (we already collapsed n_ocb iterations
   into single 4D-strided BDs in Phase F).

### Where this lands relative to OCB-unroll

OCB-unroll is essentially **a tighter version of llama's intra-device
packing for the case of repeating the same kernel N times**. We unroll
n_ocb iterations of one conv inside one aie.device runtime_sequence
with one PDI load and one aiex.run.

What's missing is the **multi-different-operation** case — bottleneck
`conv1 → memtile → conv2` where conv1 and conv2 are the same kernel
but with different weights and a memtile-routed intermediate.
That's the Phase E v1 design originally sketched, now reframed as
direct llama-style device packing.

## Concrete targets for MDV6

### Target 1: bottleneck conv1+conv2 packing (the original Phase E v1)

Per RepNCSP inner block (Phase D):
```python
repconv_out = rt(mc_rn3, sc_rn3, current, fuse_repconv(bn_block.conv1),
                 H, W, neck, trn3, trn3, orn3, 1, 3, 1)
conv2_out = rt(mc_rn3, sc_rn3, repconv_out, fuse_bn(bn_block.conv2),
               H, W, neck, trn3, trn3, orn3, 1, 3, 1)
current = (residual + conv2_out) if bn_block.residual else conv2_out
```

Two `mc_rn3` dispatches per inner block. 102 bottleneck iterations × 2
dispatches = 204 dispatches today (post-OCB collapse).

**Pack into one aie.device per layer-call** (one per inner block):
- Same 32 cores reused for conv1 then conv2 (BD chain).
- Memtile holds the intermediate (HWC layout, sized for one tile of
  output between conv1 and conv2).
- One PDI load per layer-call instead of two PDIs.
- One xrt.run dispatches both convs.

**Per-bottleneck-call DDR savings:**
| layer | output shape | DDR write+read of intermediate today |
|---|---|---|
| re6_rn3 | 40×40×48 | 154 KB × 2 = 308 KB |
| re8_rn3 | 20×20×64 | 51 KB × 2 = 102 KB |
| re4_rn3 | 80×80×32 | 410 KB × 2 = 820 KB |

Across the model: ~50 MB/frame intermediate DDR traffic eliminated.
At 50 GB/s effective DDR: ~1 ms wall savings. Modest.

**Dispatch overhead savings:**
- 102 saved PDI swaps × ~40-115 µs (llama's measured cost) ≈ **4-12 ms wall**.
- Plus eliminating 102 launch_gap (~600 µs each = ~60 ms wall).

**Total estimated wall savings: ~60-75 ms** (mostly from launch_gap,
some from on-NPU PDI swap).

### Target 2: rnm + rn3 chain (RepNCSP merge)

`run_rn_mc` ends with:
```python
concat = torch.cat([current, x2], dim=2)
return rt(mc_rnm, sc_rnm, concat, fuse_bn(repncsp.conv3), H, W, oc, ...)
```

The last bottleneck `conv2_out` feeds into `concat` → `rnm`. If the
last `conv2_out` and `rnm` shared a device, we could skip the DDR
roundtrip there too. But the host has to do `torch.cat([current, x2])`
between them, so this needs the concat folded into NPU-side DMA
(possible but more complex).

Defer this to Target 1 first.

### Target 3: rn1_pair + first-rn3 in one device

`run_rn_mc` does:
```python
x1, x2 = run_gemm_pair(...)
current = x1
for bn_block in repncsp.bottleneck:
    repconv_out = rt(mc_rn3, ..., current, ...)  # uses x1 → current
    ...
```

The first rn3 takes `x1` (= gemm_rn1's first output) as input. Could
pack `gemm_rn1_pair + first_rn3` into one device with x1 routed via
memtile L2.

But gemm is a 1×1 conv with different kernel (`gemm_conv1x1_fused_packed_bf16`)
than rn3 (`conv3x3_fused_packed_bf16`). DIFFERENT KERNELS — needs llama-style
row partitioning (gemm on rows 2-3, conv on rows 4-5) or kernel-program
swap between phases.

Either of those is significantly more surface than Target 1. Defer.

### Target 4: full run_re_mc chain

`run_re_mc` is: `c1 → split → rn_mc (3+ dispatches each) → c3 → rn_mc → c3 → cat → c4`.

Many phases, many data dependencies. Some hand-offs are element-wise
(c3 output → c4 input via cat), some are scatter (split). The llama-style
analysis would identify the element-wise boundaries and pack them.
Multi-week surface area.

## Concrete plan: Target 1 prototype

### Phase H step 1: build a bottleneck-pair OCB-extension for re8_rn3

Smallest shape, easiest L1/L2 budget. Mechanics:

1. **Fork `aie2_multicore_ocb.py` → `aie2_multicore_ocb_pair.py`.**
   New parameter: `n_pair_phases=2` instead of `n_ocb`. Each phase has
   its own weight slot (concatenated in big_W) and its own output BD
   set, but consecutive phases share data through a memtile L2 fifo
   instead of refilling input from DDR.

2. **Two halves of the runtime_sequence:**
   - Phase 0 (conv1): host fills `current` (DDR) into col_in_fifo via shim
     DMA, cores compute, write output to `intermediate_l2` memtile fifo.
     Memtile holds the tile-format output (no DDR write).
   - Phase 1 (conv2): memtile drains `intermediate_l2` into a halo-extracted
     patch layout in core_in_fifo (the conv2 cores see standard patch-format
     input), cores compute, write final output to shim → DDR.

3. **Memtile L2 sizing:**
   - re8_rn3 intermediate: 20×20×64 bf16 = 51 KB per OCB × n_ocb=4 → 204 KB.
     Fits in one memtile's 512 KB budget. Double-buffered for pipelining =
     up to 408 KB. Still fits.
   - re6_rn3: 40×40×48 bf16 = 154 KB × 3 = 462 KB. Barely fits.
   - re4_rn3: 80×80×32 bf16 = 410 KB × 1 = 410 KB. Fits.

4. **Halo extraction in memtile:**
   - Output of conv1 is tile-format `(n_cores, ppc, tile_h, tile_w, oc)`.
     Adjacent tiles' halo for conv2 needs to come from neighbors.
   - Memtile TAP reads from L2 buffer with halo offsets per output tile.
     Memtile has 4D tensor address generation; should be feasible.
   - **This is the hardest piece.** Need a TAP that for each conv2 output
     tile at position (tr, tc) reads from memtile L2 region
     `[tr*tile_h-1 : (tr+1)*tile_h+1, tc*tile_w-1 : (tc+1)*tile_w+1, :]`.
     Boundary tiles need zero-padded halo (matching conv padding=1).

5. **Same-kernel reuse:**
   - Both phases use `conv3x3_fused_packed_bf16`. The same compute tile
     program runs both phases. Only BDs and weight pointer change between
     phases.
   - Cores' `core_fn` becomes:
     ```python
     for _ in range(n_ocb):
       # Phase 0: conv1
       elem_wt = of_wt_p0.acquire(1)
       for ppc patches: acquire_in_p0, compute, release_in_p0, release_out_l2
       release wt_p0
       # Phase 1: conv2 (consumes L2)
       elem_wt = of_wt_p1.acquire(1)
       for ppc patches: acquire_in_l2_with_halo, compute, release_in_l2, release_out_p1
       release wt_p1
     ```

6. **Host BO layout:**
   - input: same as current (HWC tensor)
   - big_W: concatenated [conv1_weights, conv2_weights, conv1_w_ocb1, conv2_w_ocb1, ...]
   - output: tile-format conv2 output (current OCB-unroll layout)
   - intermediate is INVISIBLE to host (lives in memtile L2)

### Phase H step 2: bytewise correctness vs separate-dispatch baseline

Standalone test comparing:
- The new packed ELF (one xrt.run, both convs internal) vs
- Two sequential OCB-unrolled rn3 calls (current Phase G)

Same inputs, same weights, expect bytewise match. If not, the memtile halo
extraction is wrong.

### Phase H step 3: wire into model bottleneck loop

Replace the two `rt(mc_rn3, ...)` calls in `run_rn_mc` with a single
`rt_bottleneck_pair(...)` dispatch that uses the packed ELF.

### Phase H step 4: scale to re6_rn3 and re4_rn3

Same template, larger dimensions. re4_rn3 might hit L2 budget (410 KB).
If so, fall back to single-ELF dispatch for re4.

## What this won't do (and shouldn't be claimed)

The original modeling (PHASE_E_BOTTLENECK_MODEL.md §11) said memtile-
resident intermediate saves ~2 ms wall from DMA elimination alone.
Most of the win is **dispatch consolidation** (saved PDI swaps +
launch_gap), which OCB-unroll already partially captured for within-
layer iterations.

Honest projection for Phase H:
- ~50-70 ms wall savings (102 saved dispatches × ~500-700 µs each on
  the wall side).
- That would land the model at ~1200 ms (Phase G is at ~1265 ms).

Above the bottleneck pair, packing `run_re_mc` chains gets multi-week
risky and is genuinely Phase E v1+ work.

## Risks specific to this design

| risk | mitigation |
|---|---|
| Memtile halo TAP doesn't lower cleanly in aiecc | Validate with a minimal 2-tile prototype before scaling |
| Memtile L2 budget exceeded for re4_rn3 | Sized per-layer; re4 might need single-buffered or partial spill |
| BD count per memtile exceeds 48 across both phases | Already an issue in Phase F; same 4D-strided BD pattern applies |
| Tile-format → halo-format conversion at memtile is non-trivial | This is the technical heart of the design; allocate time |
| Per-phase kernel code share might break (e.g. weight slot type differs) | Both phases use same kernel + same weight slot shape; should be fine |
| Phase 0's memtile output and Phase 1's memtile input can't share fifo cleanly | Use two memtile fifos with sync ObjectFifo + lock primitives |

## Why this is right NOW (and wasn't before)

I claimed earlier this was "multi-week risky" when comparing memtile-
resident intermediate to OCB-unroll. That comparison was apples-to-
oranges:

- OCB-unroll collapses *same-layer* iterations (works for any conv
  layer with multiple OCBs).
- Phase H collapses *cross-layer* dispatches (works for the bottleneck
  pair specifically, but also as a template for any back-to-back same-
  kernel layer pair).

After Phase F validated the 4D-strided BD pattern and 4D TAP layouts in
production (with bytewise correctness), the memtile L2 chain is the
natural next step. The infrastructure (`build_ocb.py`, `aie2_multicore_ocb.py`)
is already 90% of what's needed; the new pieces are:

1. Memtile ObjectFifo with TAP-based halo extraction
2. Two-phase core_fn loop
3. Big_W as `[conv1_w, conv2_w]` per OCB iteration
4. Host wiring (no intermediate BO, single dispatch per bottleneck inner)

That's a 2-3 day prototype on re8_rn3 (smallest), then a few more days
for re6/re4 if it works.

## Recommendation

Start with Phase H step 1+2 (re8_rn3 prototype + standalone correctness).
If bytewise PASS, step 3 (model wiring) is a few hours. If the prototype
exposes a memtile-TAP-lowering issue, we learn something concrete about
mlir-aie's TAP support and can decide whether the design is feasible
at all.

Decision point after step 2: if standalone PASS + wall savings >
prediction, do steps 3-4. If standalone PASS but wall doesn't improve,
declare the dispatch overhead truly is per-xrt.run not per-aiex.configure
and stop. If standalone FAILS, the design needs rework or abandonment.

## Update: 2026-05-31 — Step 1 quick spike hit the predicted blocker

Tried the cheap shortcut first: build a merged ELF with two stock
`mc_re8_rn3` sub-devices, chain_links to alias sub_1's input BO to
sub_0's output BO. See `conv/build_pair_rn3.py` (committed for the
record).

Result: `build_merged._make_dispatcher_block` rejects the chain_link
with:

```
ValueError: chain_links type mismatch:
  sub0.arg2 (memref<32768xui16>)        <- conv1 output, tile-format
  vs sub1.arg0 (memref<204800xui16>)    <- conv2 input, patch-format-with-halo
```

This is the exact format mismatch the plan flagged. The two BOs have
fundamentally different sizes and layouts:

| BO | shape | elements | layout |
|---|---|---|---|
| conv1 output | (32 cores × 1 ppc × 4 × 4 × 16) | 32,768 u16 | tile-format, no halo |
| conv2 input | (32 cores × 1 ppc × 6 × 6 × 64) | 204,800 u16 | patch-format with halo |

`chain_links` requires type match — so this approach is structurally
closed. No type-coercion shortcut exists in the merged-ELF builder.

### What the spike confirmed (and what's still ahead)

1. **The merged-ELF builder is type-safe.** Can't paper over format
   mismatches with chain_links — by design.
2. **The format conversion IS the technical core of Phase H.** Not a
   detail to defer.
3. **Conv1's output IC=16 vs conv2's input IC=64 is a per-OCB issue.**
   conv2 reads ALL 64 OC of conv1's combined output as its 64 IC. So
   the BO shared between phases must carry all OCBs of conv1's output
   stacked into an HWC tensor (40×40×64 for re8_rn3 = 51,200 elements
   = 102,400 bytes raw, but the patch-format expansion with halo is
   204,800 elements). The 4x size difference is the halo replication.

### Three concrete paths forward (none are quick)

**Path A — Modify conv2's IRON to consume tile-format input directly.**
Change conv2's `_run_tiled_mc_inner_*` host-side packing to pass
tile-format input (without halo) and have its kernel do internal halo
gather from neighboring cores' tiles. Requires either:
- C kernel modification (`conv3x3_fused_packed_bf16` reads from a
  bigger L1 region that includes neighbors), or
- Memtile DMA pattern that reformats tile-format → patch-format
  on the memtile side (memtile-resident reformat as described in the
  original plan §3).

**Path B — Modify conv1's IRON to produce patch-format-with-halo output.**
Each core writes its tile output PLUS the halo region required by
neighboring tiles, redundantly. The shared BO is then in patch-format
and conv2 reads as-is. Wasteful (4x output BO) but technically simple.
Conv1's `core_fn` would write n_neighbors × tile_size output instead
of 1 × tile_size.

**Path C — Add a "reformat" sub-device between conv1 and conv2.**
Three subs in the merged ELF: conv1, reformat, conv2. The reformat
sub is a single-tile sequencer whose only job is rearranging conv1's
output BO into patch-format conv2-input BO via memtile DMA. Conv1 →
intermediate (tile-format, small) → reformat-sub does the gather/halo
→ intermediate2 (patch-format, big) → conv2.

This is the cleanest separation; it also lets us measure exactly what
the reformat costs. But "single-tile DMA sequencer" is a new pattern
we don't have in the codebase.

### Step 1 verdict

Cheap chain_links shortcut doesn't work. To make any progress on
Phase H we have to commit to one of paths A/B/C. None are <1 day.

Recommendation: try Path C first (single-tile reformat sub). It's the
most diagnostically valuable — we can measure its cost separately
from the convs. If reformat-only cost is significant (say >100 µs per
call), we'll know to invest in Path A. If it's negligible, we keep
the reformat sub and call Phase H done.

But this should be a real planning conversation before spending
multi-day on it. Phase G already beats the pre-Phase-A baseline; the
remaining work to chase below 1200 ms is high-risk multi-day surgery.
