# Decode kernel buffers — W and X sizes, ping-pong state

Snapshot at current HEAD.

All sizes are **per single DMA BD execution** (one consume or one produce).
Y is omitted because it's tiny everywhere (16 B or 4 B) and never ping-ponged.

**Important:** X never has an L2 buffer. The X path is shim → compute direct via the
AXI stream switch. "L2 X" rows are intentionally absent.

## BF16 path

| Kernel | Sub-device | K | W L1 | W L2 | X L1 | W L1 pp | W L2 pp | X L1 pp |
|---|---|---|---|---|---|---|---|---|
| rms_gemv_rope | **v_matvec_bf16_0** | 2048 | 16 KB | 32 KB | 4 KB | **ON** | off | off |
| rms_gemv_rope | k_matvec_bf16_0 | 2048 | 16 KB | 32 KB | 4 KB | off | off | off |
| rms_gemv_rope | q_matvec_bf16_0 | 2048 | 16 KB | 32 KB | 4 KB | off | off | off |
| o_gemv_ffn | **og_matvec_bf16_0** (O-proj) | 2048 | 16 KB | 32 KB | 4 KB | **ON** | off | off |
| o_gemv_ffn | gg_matvec_bf16_0 (gate) | 2048 | 16 KB | 32 KB | 4 KB | off | off | off |
| o_gemv_ffn | ug_matvec_bf16_0 (up) | 2048 | 16 KB | 32 KB | 4 KB | off | off | off |
| o_gemv_ffn | **dg_matvec_bf16_0** (down) | 8192 | 16 KB | 32 KB | 16 KB | off (L1 cap) | **ON** | off |
| lm_head_gemv | LM head partitions | 2048 | 16 KB | 32 KB | 4 KB | off | off | off |

## AWQ path

| Kernel | Sub-device | K | W L1 | W L2 | X L1 | W L1 pp | W L2 pp | X L1 pp |
|---|---|---|---|---|---|---|---|---|
| rms_gemv_rope_awq | v_matvec_awq_bf16_0 | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| rms_gemv_rope_awq | k_matvec_awq_bf16_0 | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| rms_gemv_rope_awq | q_matvec_awq_bf16_0 | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | og_awq_matvec_0 (O-proj) | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | gg_awq_matvec_0 (gate) | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | ug_awq_matvec_0 (up) | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | **dg_awq_matvec_0** (down) | 8192 | **8.5 KB** (K_TILE=2) | **17 KB** (M_TILE=2 × 8.5 KB) | 16 KB | off (infra not plumbed) | off (infra not plumbed) | off (replaced by K_TILE=2) |
| lm_head_gemv_awq | LM head AWQ | 2048 | 4.25 KB | 8.5 KB | 4 KB | off | off | off |

## Summary of what's ON

Currently active across both paths:

| Kernel | Sub-device | What's on | Commit |
|---|---|---|---|
| rms_gemv_rope | v_matvec_bf16_0 | W L1 pp | `6cfb1db03` |
| o_gemv_ffn | og_matvec_bf16_0 | W L1 pp | `c0396143b` |
| o_gemv_ffn | dg_matvec_bf16_0 | W L2 pp | `bb8ddd4ab` |
| o_gemv_ffn_awq | dg_awq_matvec_0 | **K_TILE_K8192 = 2** (bigger tile) | `b9d5a515d` |
| rms_gemv_rope_awq, o_gemv_ffn_awq (K=2048) | V/Q/K, og/gg/ug | **K_TILE = 8** (bigger tile) | `09b583ea6` |

Everything else is single-buffered at both L1 and L2.

`o_gemv_ffn_awq:dg_awq_matvec_0` had X L1 PP from `7a221447b` then reverted in
favor of `K_TILE_K8192=2` (bigger W tile) after an A/B that showed equivalent
tok/sec with less L1 footprint and lower DMA contention.

## Infrastructure that's plumbed (function-parameter exists, just not wired on)

- `pingpong_w` + `pingpong_w_l2` on `_emit_matvec_seg` (rms_gemv_rope)
- `pingpong_w` on `_emit_matvec_seg_k2048` (o_gemv_ffn — L2 still not plumbed here)
- `pingpong_w` + `pingpong_w_l2` on `_emit_matvec_seg_k8192` (o_gemv_ffn)
- `pingpong_w` + `pingpong_w_l2` on `_emit_awq_matvec_seg` (rms_gemv_rope_awq)
- `pingpong_x` on `_emit_awq_matvec_seg_k8192` (o_gemv_ffn_awq) — **only X pp anywhere**

## Where infrastructure is **not** plumbed

- `pingpong_x` not on any BF16 builder
- `pingpong_x` not on the AWQ K=2048 builders (`_emit_awq_matvec_seg`, `_emit_awq_matvec_seg_k2048`)
- `pingpong_w` / `pingpong_w_l2` not on any AWQ kernel in `o_gemv_ffn_awq` (the AWQ counterpart of `o_gemv_ffn`)
- Nothing on `lm_head_gemv` / `lm_head_gemv_awq` (those builders haven't been touched)

## Where the data suggests the next wins live

Based on the per-BD size + trace patterns observed:

| Target | Lever | Expected leverage |
|---|---|---|
| **BF16 dg X L1 pp** | infra not plumbed, but same pattern as AWQ dg X | High — BF16 dg lock_stall was 76% after the W L2 pp landed; X is 16 KB on the K=8192 path. Direct mirror of what just worked for AWQ. |
| BF16 og/gg/ug L2 W pp | infra not plumbed for L2 on K=2048 | Medium — L1 W pp on og raised starv1 to 59%, indicating L2 W is the new constraint, but og span is small relative to dg. |
| AWQ V/og/gg/ug W pp | infra plumbed, all off | Low — already showed it doesn't help when AWQ lock_stall is low. |

If extending, BF16 dg X L1 ping-pong is the obvious next experiment — same code shape
as the AWQ dg X change.

---

## Bigger tiles vs ping-pong on AWQ

AWQ has by far the most L1 headroom of any path, so it's worth asking whether to
spend the headroom on ping-pong vs on bigger tiles.

### L1 headroom inventory

| Path | Current L1 use | L1 cap | Headroom |
|---|---|---|---|
| BF16 K=2048 | ~20 KB | 64 KB | ~44 KB |
| BF16 K=8192 (dg) | ~32 KB | 64 KB | ~32 KB |
| **AWQ K=2048** | **~8.3 KB** | 64 KB | **~55 KB** |
| AWQ K=8192 (dg) | ~20 KB | 64 KB | ~44 KB |

55 KB free on AWQ K=2048 is enormous. So the question is real.

### What each lever actually attacks

**Ping-pong** attacks `lock_stall` (compute idling while waiting for a buffer to be
filled). Doesn't reduce total work; just overlaps DMA with compute.

**Bigger tiles** attack *per-iteration fixed overhead* (lock acquire/release cycles,
BD setup, software-pipeline drain at the boundary of each K-loop iter). Each L1 fill
produces more output, so the per-fill overhead amortizes over more compute. Doesn't
help with DMA-wait stalls.

They're partially complementary — but for *any given buffer* you choose between making
it bigger or making it ping-pong'd, since 2× the size of the slot eats the same L1
budget either way.

### What the AWQ trace data says about which is the bottleneck

| AWQ tile | lock_stall | vec_util | What ping-pong-able | What tile-size-able |
|---|---|---|---|---|
| V/og/gg/ug AWQ | 17-25% | 17-19% | small lock_stall to claim back | K_TILE=4 → 8 cuts K-loop iters in half |
| dg AWQ | 40% (PP'd to 18%) | 14.5% → 19.7% | already X-PP'd | K_TILE_K8192=1 → 2 would also remove the K-loop |

For dg AWQ we already proved X PP works: lock_stall went 40 → 18%, span −26.6%,
+0.46 tok/sec.

For AWQ K=2048 (V/og/gg/ug) the lock_stall budget to attack is much smaller (17-25%),
and the W PP experiment showed PP doesn't claim it cleanly because there's not enough
to amortize the BD overhead.

### Where to actually spend the headroom

**1. Grow K_TILE on AWQ K=2048 first** — single-knob change, no PP infrastructure needed.

Current `K_TILE = 4` (4 output rows per matvec_fn call, 2 K-iters per outer-for-iter).

Bumping to `K_TILE = 8`: each call does 8 rows in one shot, K-loop becomes 1 iter.
Halves the lock acquire/release cycles AND removes the K-loop branch/prolog. W L1
grows from 4.25 KB → 8.5 KB (still tiny). No need to touch X or Y.

Expected: 5-10% kernel-local cycle reduction. Tok/sec impact depends on whether the
kernel matters dispatch-wise.

**2. For AWQ dg, evaluate K_TILE_K8192 = 1 → 2 *instead of* X PP.**

Right now we ping-pong X across 2 K-iters. If K_TILE_K8192 = 2 (matching
M_TILE_K8192), there's only 1 K-iter and only 1 matvec call per outer iter. The X PP
becomes moot — we'd be hiding nothing because there's nothing to hide behind. The
single bigger call uses:
- 1 W fill (8.5 KB)
- 1 X fill (16 KB)
- 1 matvec_fn call processing 2 rows × 8192 K

L1: W 8.5 + X 16 + Y 4B = ~25 KB. *Less* L1 than the X PP version (which used
W 4.25 + 2×X 32 = ~36 KB), because doubling K_TILE costs less than doubling X.

This is an A/B test against the +0.46 tok/sec we got from X PP.

**3. Don't grow X** — there's nothing to grow. X is the full input vec, dictated by
the model. The only X-side lever is double-buffering (which is what PP does).

### Order to try

1. **A/B AWQ dg `K_TILE_K8192 = 2` vs the current X PP committed state.** Same
   correctness check + 5-run benchmark + trace. Two outcomes:
   - K_TILE=2 wins → revert X PP, take the bigger-tile path.
   - X PP wins → keep current state. Either way we now know which.

2. **AWQ K=2048 `K_TILE = 8` everywhere** (V/Q/K, og/gg/ug). Single-knob change, but
   it touches the runtime sequence dimensions too (M_TILE/K_TILE = 1 instead of 2
   changes how the host issues DMAs). Probably another +0.1-0.3 tok/sec.

3. **Skip W PP on AWQ for good** unless step 2 shows lock_stall growing again — the
   trace says it's not worth it on AWQ K=2048.
