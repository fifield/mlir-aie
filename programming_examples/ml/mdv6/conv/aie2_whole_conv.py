"""Experimental whole re8 RN 3x3 convolution command sequence.

Fixed logical shape: H=W=20, IC=OC=64, stride=1, pad=1.
Preserves mc_re8_rn3 geometry: 32 cores, 8x8 output tiles, OC block 16,
one patch/core. Nine real spatial patches are padded to 32 patches.

ABI (bf16 bit patterns in contiguous uint16):
  I: [32, 10, 10, 64], reused by four device DMA groups.
  W: [4, 9248], each [packed 16x64x3x3 weights, 16 scales, 16 biases].
  O: [4, 32, 8, 8, 16], OC-block-major, then core-major.

One host submission, four sequential OC-block DMA groups, all eight column
outputs awaited per group. This batches dispatches; it does not remove the
four external-memory patch reads or make an adjacent operator resident.
"""

from aie.iron.device import NPU2
from aie2_multicore import multicore_conv


if __name__ == "__main__":
    print(multicore_conv(
        NPU2(), tile_h=8, tile_w=8, ic=64, oc=16,
        kernel_size=3, stride_val=1, padding_val=1,
        n_cores=32, patches_per_core=1, output_blocks=4,
    ))
