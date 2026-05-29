# Decode kernel buffers — W and X sizes, ping-pong state

Snapshot at current HEAD.

All sizes are **per single DMA BD execution** (one consume or one produce).
Y is omitted because it's tiny everywhere (16 B or 4 B) and never ping-ponged.

**Important:** X never has an L2 buffer. The X path is shim → compute direct via the
AXI stream switch. "L2 X" rows are intentionally absent.

## BF16 path

| Kernel | Sub-device | K | W L1 | W L2 | X L1 | W L1 pp | W L2 pp | X L1 pp |
|---|---|---|---|---|---|---|---|---|
| rms_gemv_rope | v_matvec_bf16_0 | 2048 | **32 KB** (K_TILE=8) | **32 KB** | 4 KB | off (reverted) | off | off |
| rms_gemv_rope | k_matvec_bf16_0 | 2048 | **32 KB** (K_TILE=8) | **32 KB** | 4 KB | off | off | off |
| rms_gemv_rope | q_matvec_bf16_0 | 2048 | **32 KB** (K_TILE=8) | **32 KB** | 4 KB | off | off | off |
| o_gemv_ffn | og_matvec_bf16_0 (O-proj) | 2048 | **32 KB** (K_TILE=8) | **32 KB** | 4 KB | off (reverted) | off | off |
| o_gemv_ffn | gg_matvec_bf16_0 (gate) | 2048 | **32 KB** (K_TILE=8) | **32 KB** | 4 KB | off | off | off |
| o_gemv_ffn | ug_matvec_bf16_0 (up) | 2048 | **32 KB** (K_TILE=8) | **32 KB** | 4 KB | off | off | off |
| o_gemv_ffn | **dg_matvec_bf16_0** (down) | 8192 | **32 KB** (K_TILE_K8192=2) | **32 KB** | 16 KB | off (L1 cap) | **ON** | off |
| lm_head_gemv | LM head partitions | 2048 | **32 KB** (K_TILE=8) | **32 KB** | 4 KB | off | off | off |

## AWQ path

| Kernel | Sub-device | K | W L1 | W L2 | X L1 | W L1 pp | W L2 pp | X L1 pp |
|---|---|---|---|---|---|---|---|---|
| rms_gemv_rope_awq | v_matvec_awq_bf16_0 | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| rms_gemv_rope_awq | k_matvec_awq_bf16_0 | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| rms_gemv_rope_awq | q_matvec_awq_bf16_0 | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | og_awq_matvec_0 (O-proj) | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | gg_awq_matvec_0 (gate) | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | ug_awq_matvec_0 (up) | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |
| o_gemv_ffn_awq | **dg_awq_matvec_0** (down) | 8192 | **8.5 KB** (K_TILE_K8192=2) | **8.5 KB** (M_TILE_K8192=2 × 4.25 KB row) | 16 KB | off (no L1 W infra) | off (`pingpong_w_l2` plumbed; tested, not worth it) | off (`pingpong_x` plumbed; K_TILE_K8192=2 collapsed K-loop) |
| lm_head_gemv_awq | LM head AWQ | 2048 | **8.5 KB** (K_TILE=8) | **8.5 KB** | 4 KB | off | off | off |

## Summary of what's ON

Currently active across both paths:

| Kernel | Sub-device | What's on | Commit |
|---|---|---|---|
| rms_gemv_rope (BF16, K=2048) | V/Q/K | **K_TILE = 8** (bigger tile) | `70431541f` |
| o_gemv_ffn (BF16, K=2048) | og/gg/ug | **K_TILE = 8** (bigger tile) | `70431541f` |
| o_gemv_ffn (BF16, K=8192) | dg_matvec_bf16_0 | **K_TILE_K8192 = 2** + W L2 PP | `70431541f` + `bb8ddd4ab` |
| lm_head_gemv (BF16, K=2048) | partitions | **K_TILE = 8** (bigger tile) | `fcab5c0fc` |
| rms_gemv_rope_awq, o_gemv_ffn_awq (K=2048) | V/Q/K, og/gg/ug | **K_TILE = 8** (bigger tile) | `09b583ea6` |
| o_gemv_ffn_awq (K=8192) | dg_awq_matvec_0 | **K_TILE_K8192 = 2** (bigger tile) | `b9d5a515d` |
| lm_head_gemv_awq (K=2048) | LM head AWQ | **K_TILE = 8** (bigger tile) | `fcab5c0fc` |

BF16 V/og L1 W PP (commits `6cfb1db03`, `c0396143b`) were reverted by `70431541f`
in favor of the bigger-tile approach; they were superseded by K_TILE=8 which gave
a larger gain on the same axes.

Everything else is single-buffered at both L1 and L2.

`o_gemv_ffn_awq:dg_awq_matvec_0` had X L1 PP from `7a221447b` then reverted in
favor of `K_TILE_K8192=2` (bigger W tile) after an A/B that showed equivalent
tok/sec with less L1 footprint and lower DMA contention.

## Infrastructure that's plumbed (function-parameter exists, just not wired on)

- `pingpong_w` + `pingpong_w_l2` on `_emit_matvec_seg` (rms_gemv_rope)
- `pingpong_w` on `_emit_matvec_seg_k2048` (o_gemv_ffn — L2 still not plumbed here)
- `pingpong_w` + `pingpong_w_l2` on `_emit_matvec_seg_k8192` (o_gemv_ffn)
- `pingpong_w` + `pingpong_w_l2` on `_emit_awq_matvec_seg` (rms_gemv_rope_awq)
- `pingpong_x` **+** `pingpong_w_l2` on `_emit_awq_matvec_seg_k8192` (o_gemv_ffn_awq) — this is the **only X pp anywhere**; both default off, dg call site (`o_gemv_ffn_awq.py:2840`) passes neither

## Where infrastructure is **not** plumbed

- `pingpong_x` not on any BF16 builder
- `pingpong_x` not on the AWQ K=2048 builders (`_emit_awq_matvec_seg`, `_emit_awq_matvec_seg_k2048`)
- `pingpong_w` (L1 W) not on **any** AWQ builder
- In `o_gemv_ffn_awq`: `_emit_awq_matvec_seg_k8192` **does** have `pingpong_w_l2` (and `pingpong_x`), but `_emit_awq_matvec_seg_k2048` has no pp params at all
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
| V/og/gg/ug AWQ | 17-25% | 17-19% | small lock_stall to claim back | K_TILE 4 → 8 (**landed**, `09b583ea6`) |
| dg AWQ | 40% → 18% | 14.5% → 19.7% | X PP works but was reverted | K_TILE_K8192 1 → 2 (**landed**, `b9d5a515d`) |

For dg AWQ X PP was proven to work (lock_stall 40 → 18%, span −26.6%, +0.46 tok/sec)
**but was then reverted** in favor of `K_TILE_K8192 = 2`, which collapses the K-loop
to a single iter and gives equivalent tok/sec with less L1 footprint and lower DMA
contention. With one K-iter there is nothing for X PP to hide behind, so `pingpong_x`
stays plumbed-but-off (the unroll assert at `o_gemv_ffn_awq.py:747` would in fact fire
if it were re-enabled against the current `M_TILE_K8192/K_TILE_K8192 == 1`).

For AWQ K=2048 (V/og/gg/ug) the lock_stall budget to attack is much smaller (17-25%),
and the W PP experiment showed PP doesn't claim it cleanly because there's not enough
to amortize the BD overhead. The win there came from `K_TILE = 8` (bigger tile)
instead — **landed**.

### What landed (the bigger-tile bet paid off on both)

Both candidate experiments below have since been merged; bigger tiles beat ping-pong
on AWQ in both cases:

**1. AWQ K=2048 `K_TILE = 8` — LANDED (`09b583ea6`).** Each `matvec_fn` call now does
8 rows in one shot (`K_TILE = M_TILE = 8`, `o_gemv_ffn_awq.py:91-92`), so the K-loop is
a single iter — halves the lock acquire/release cycles and removes the K-loop
branch/prolog. W L1 is 8.5 KB (still tiny). No PP infrastructure was needed. Applied
to V/Q/K and og/gg/ug.

**2. AWQ dg `K_TILE_K8192 = 2` — LANDED (`b9d5a515d`), X PP reverted.** With
`K_TILE_K8192 = M_TILE_K8192 = 2` (`o_gemv_ffn_awq.py:97-98`) there is one K-iter and
one matvec call per outer iter, so X PP has nothing to hide behind. The single bigger
call uses W 8.5 KB + X 16 KB + Y 4 B ≈ 25 KB — *less* L1 than the X-PP version
(W 4.25 + 2×X 32 ≈ 36 KB), because doubling K_TILE costs less than doubling X. Measured
equivalent tok/sec to X PP with lower L1 footprint and DMA contention, so X PP was
reverted (`7a221447b` → reverted).

**3. X is not a tile-size lever** — X is the full input vec, fixed by the model. The
only X-side lever is double-buffering (`pingpong_x`), which is now moot on dg per #2.

### What's left

- **W PP on AWQ: skip for good.** The trace says it's not worth it on AWQ K=2048
  (lock_stall budget too small to amortize BD overhead), and on dg `pingpong_w_l2`
  was tested (starv1 22%→12%, dma_in1_eff 30%→44%) for only a 0.4% span change — W is
  not on the critical path. Both PP infras stay plumbed but off.
- The remaining ~50% unaccounted dg cycles are likely **memory_stall in the dequant
  chain** (uint4→bf16), which DMA/buffer changes cannot attack — see the AWQ-dequant
  codegen-gap note. That, not buffering, is the higher-leverage AWQ target.
