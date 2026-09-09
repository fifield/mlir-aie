"""Whole re4/re15 final-projection spatial command sequence.

Logical shape: H=W=80, IC=256, OC=128. Preserve the deployed K-blocked
geometry: 32 cores, tile_m=68, one patch/core, KB16, three spatial batches.
Each batch has 2176 pixels; the final batch has 2048 real pixels.

ABI (bf16 bits in contiguous uint16):
  I: [batch=3, core=32, patch=1, tile_m=68, IC=256] (3342336 bytes).
  W: [K-block=16, chunk=2304] (73728 bytes). Each chunk contains packed
     [KB/8, OC/8, innerIC=8, innerOC=8] weights, then 128 BN scales/biases.
  O: [batch=3, core=32, patch=1, tile_m=68, OC=128] (1671168 bytes).

One submission contains three bounded spatial DMA groups, awaiting every
column's output before freeing/reusing descriptors. All 16 weight chunks
are replayed per patch per batch; this does NOT retain full weights on-chip.
The original K order and bf16 partial-output rounding are unchanged.
"""

from aie.iron.device import NPU2
from aie2_gemm_conv1x1 import gemm_conv1x1


if __name__ == "__main__":
    print(gemm_conv1x1(
        NPU2(), tile_m=68, ic=256, oc=128, n_cores=32,
        patches_per_core=1, k_block=16, spatial_batches=3,
    ))
