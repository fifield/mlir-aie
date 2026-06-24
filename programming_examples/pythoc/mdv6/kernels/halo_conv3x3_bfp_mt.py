"""Tiles-per-core halo conv kernel variant (re4-scaling exploration).

NEW FILE — does NOT touch kernels/halo_conv3x3_bfp.py (the proven re8/re6 fusion
kernel a sibling agent is measuring). The compute math here is BYTE-IDENTICAL to
the proven halo_conv3x3_bfp: one 8x8 output tile per call, input halo-gathered
from a padded-HWC window, BFP576-MAC'd, BN+SiLU-stored to a tiled-C buffer.

The tiles-per-core scaling does NOT live in the kernel — it lives in the
generator (conv/aie2_halo_conv_mt.py): each Worker LOOPS `tpc` calls to this
kernel, acquiring one window/output objectfifo element per tile (exactly the
rn3 raster chain's per-(worker,round) structure). So one tile's window + one
tile's C are L1-resident at a time, independent of tpc. The kernel itself need
not change at all; this file simply RE-EXPORTS the proven single-tile kernel so
the existing file stays untouched and the new generator imports from here.

Re-exported symbols are the proven, bit-exact ones."""
from __future__ import annotations

from kernels.halo_conv3x3_bfp import (
    halo_conv3x3_bfp as halo_conv3x3_bfp_mt,
    halo_conv3x3_bfp_ocb1 as halo_conv3x3_bfp_mt_ocb1,
    _build_a64_halo,
    _store_bn_silu_4x8_f32,
    KERNEL_EXTRA_GLOBALS,
    HALO_CONV_HELPERS,
)

__all__ = [
    "halo_conv3x3_bfp_mt",
    "halo_conv3x3_bfp_mt_ocb1",
    "KERNEL_EXTRA_GLOBALS",
    "HALO_CONV_HELPERS",
]
