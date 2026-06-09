# Halo redistribution — chained rn3-pair iterations in one launch

Goal: run all 4 bottleneck iterations of a re6 RepNCSP block in ONE
launch (4 fused pairs + residual adds), eliminating per-iteration host
pack/scatter (~2 ms each) and 3 launches per block.

## Where the cost is today (fused pair path, measured 2026-06-09)

Per pair call: 0.76 ms NPU kernel + ~2 ms host (extract halo patches,
bf16↔f32 conversions, scatter). Per re6 block: 4 calls = ~11 ms wall.
Roofline: 4×0.76 NPU + 1 launch + 1 pack/scatter ≈ 5.5 ms — half.

## Redistribution problem statement

Between iterations, the 25 8×8×48 tile outputs must redistribute as 25
12×12×48 halo patches. Memtiles have no E/W ports, tiles are striped 4
per column over 7 columns, halos cross columns → DDR bounce (measured
22 GB/s pass-through; ~30 µs/hop) is the right substrate for v1.

## Design v1 (single launch, DDR bounce, no host)

Sequence per iteration (compile-time unrolled, taskgroup-serialized):
1. fill arenas: per worker, 3D TAP [12,12,48]/[44·48,48,1] from padded
   44×44×48 HWC image in DDR; mask filled once (static).
2. cores: conv1 ×3 (mode 0) → conv2 ×3 (mode 1) → residual add ×3
   (mode 2, NEW — final += center 8×8 of input patch; landed).
3. drain finals: 4D TAP [3,8,8,16]/[16,44·48,48,1] scatter to image.

Iteration weights: 24 slots streamed (12/iter), stride TAP as today.

## Open layout constraint (the design fork)

Finals live mid-arena (offset 4800); FIFO joins can't slice subregions,
so a drain consumes whole arenas (8272 elems) — junk scratch can't be
TAP'd around. Options:
- A: out fifo elem = 3072 finals; conv2 scratch moves to a worker-local
  buffer (LocalBuffer ~9.6 KB; stack/L1 fits). Kernel mode1/mode2 base
  becomes the fifo elem base. PREFERRED.
- B: kernel writes finals at arena base, scratch at +4800; drain whole
  arena with junk to a DDR junk region (extra 5200/arena traffic, OK at
  22 GB/s; needs second image-sized junk BO).

BD budget per col per iter: 4 fills + 4 drains + 1 weight ≈ 9 tasks via
4D BDs; ×4 iters via task groups (queue, not slots). Within 48.

## Layout decision (locked, 2026-06-09)

Row-major tile→worker striping breaks per-column single-BD fills (patch
offsets wrap grid rows). Assign each NPU column one 40×40 GRID COLUMN:
5 tiles per column at uniform stride 8·44·48 → fill TAP [5,12,12,48],
drain TAP [5,3,8,8·16], both single BDs. 5 columns × 4 workers; worker 0
processes 2 tiles. Kernels landed in kernels/rn3_chain_pythoc.py
(chain_conv1/mask/conv2/residual — mode-free; mode-branch variant
measured 4-5× slower and was reverted). Scratch = per-worker Buffer
(auto-placed); finals at FIFO base.

## Step plan

1. Kernel mode 2 — DONE (compile check in flight).
2. Fork aie2_rn3_pair_vector_chain.py: option A topology, n_iters=2.
3. Standalone test vs run_re6_rn3_pair applied twice + residual.
4. Scale to n_iters=4 + wire into run_rn_mc.
5. Re-profile model: ~−27 launches, ~−24 ms/block × 3 re6 blocks.
