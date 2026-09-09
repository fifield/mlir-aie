"""Experimental whole ELAN2 conv4 spatial-batch command sequence.

Logical shape: H=W=160, IC=128, OC=64. Preserve the deployed non-K-blocked
GEMM geometry: 32 cores, tile_m=104, two patches/core, four spatial batches.
The last batch is padded to 6656 pixels; only the first 25600 outputs are real.

ABI (bf16 bit patterns in contiguous uint16):
  I: [batch=4, core=32, patch=2, tile_m=104, IC=128].
  W: [8320], packed [IC/8, OC/8, innerIC=8, innerOC=8], then BN scale/bias.
  O: [batch=4, core=32, patch=2, tile_m=104, OC=64].

One submission, four sequential spatial DMA groups, all eight column outputs
awaited per group. Weights are sent once and retained by each worker across
all eight patches. No inter-operator residency or cross-invocation weight
retention is implied. Original mmul/BN/SiLU kernels and rounding are unchanged.
"""

from aie.iron.device import NPU2
from aie2_gemm_conv1x1 import gemm_conv1x1


if __name__ == "__main__":
    print(gemm_conv1x1(
        NPU2(), tile_m=104, ic=128, oc=64, n_cores=32,
        patches_per_core=2, k_block=0, spatial_batches=4,
    ))
