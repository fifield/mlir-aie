"""Regime contracts for mlir-aie-1jg xclbin collapse.

Sizes are in bf16/uint16 elements, not bytes. The contract separates the
regime envelope used by MLIR ObjectFifo/BD sizing from the active layer shape
written into RTP constants.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeContract:
    name: str
    description: str
    spatial: str
    kernel_kinds: tuple[str, ...]
    input_patch_size: int
    weight_buffer_size: int
    output_tile_size: int
    tile_h: int
    tile_w: int
    ic_max: int
    oc_max: int
    oc_block_max: int
    patches_per_core: int
    members: tuple[str, ...]


@dataclass(frozen=True)
class ConvRegimeArtifact:
    regime: str
    xclbin_name: str
    kernel_size: int
    stride: int
    tile_h: int
    tile_w: int
    ic: int
    oc_block: int
    patches_per_core: int
    input_depth: int
    members: dict[str, tuple[int, int, int, int, int, int, int]]


@dataclass(frozen=True)
class GemmRegimeArtifact:
    regime: str
    xclbin_name: str
    tile_m: int
    ic: int
    oc: int
    patches_per_core: int
    k_block: int
    members: dict[str, tuple[int, int, int, int]]


# Current target grouping. Some spatial regimes contain multiple kernel
# families in today's implementation; those families are migrated separately
# while sharing the same top-level regime accounting.
REGIME_CONTRACTS = {
    "R1": RegimeContract(
        name="R1",
        description="80x80, IC <= 128 non-K spatial work",
        spatial="80x80",
        kernel_kinds=("conv3x3", "conv1x1_gemm"),
        input_patch_size=14 * 14 * 64,
        weight_buffer_size=16 * 64 * 9 + 2 * 16,
        output_tile_size=12 * 12 * 16,
        tile_h=12,
        tile_w=12,
        ic_max=128,
        oc_max=128,
        oc_block_max=64,
        patches_per_core=4,
        members=("elan2", "re4", "re15"),
    ),
    "R2": RegimeContract(
        name="R2",
        description="40x40, IC <= 192 non-K spatial work",
        spatial="40x40",
        kernel_kinds=("conv3x3", "conv1x1_gemm"),
        input_patch_size=10 * 10 * 96,
        weight_buffer_size=16 * 96 * 9 + 2 * 16,
        output_tile_size=8 * 8 * 16,
        tile_h=8,
        tile_w=8,
        ic_max=192,
        oc_max=192,
        oc_block_max=48,
        patches_per_core=2,
        members=("re6", "re12", "re18"),
    ),
    "R3": RegimeContract(
        name="R3",
        description="20x20, IC <= 256 non-K spatial work",
        spatial="20x20",
        kernel_kinds=("conv3x3", "conv1x1_gemm"),
        input_patch_size=6 * 6 * 128,
        weight_buffer_size=16 * 128 * 9 + 2 * 16,
        output_tile_size=4 * 4 * 16,
        tile_h=4,
        tile_w=4,
        ic_max=256,
        oc_max=256,
        oc_block_max=64,
        patches_per_core=1,
        members=("re8", "re21", "spp9"),
    ),
    "R4": RegimeContract(
        name="R4",
        description="K-blocked concat merges and high-IC 1x1 layers",
        spatial="mixed",
        kernel_kinds=("kblocked_gemm",),
        input_patch_size=68 * 256,
        weight_buffer_size=64 * 256 + 2 * 256,
        output_tile_size=68 * 128,
        tile_h=68,
        tile_w=1,
        ic_max=512,
        oc_max=256,
        oc_block_max=256,
        patches_per_core=2,
        members=("re4_c4", "re6_c4", "re8_c1", "re8_c4", "spp9_c1",
                 "re12_c1", "re15_c1", "re15_c4", "re18_c1", "re21_c1"),
    ),
    "R5": RegimeContract(
        name="R5",
        description="stride-2 stems",
        spatial="stride2",
        kernel_kinds=("conv3x3",),
        input_patch_size=41 * 41 * 128,
        weight_buffer_size=32 * 128 * 9 + 2 * 32,
        output_tile_size=20 * 20 * 32,
        tile_h=20,
        tile_w=20,
        ic_max=128,
        oc_max=256,
        oc_block_max=32,
        patches_per_core=4,
        members=("conv0", "conv1", "aconv3", "aconv5", "aconv7", "aconv16", "aconv19"),
    ),
}


# First migrated artifact: R3 conv3x3. The xclbin/ObjectFifo envelope is
# 4x4, IC=128, OC block=16. Active layer constants are baked into the .bin.
# mc_re8_rn3 previously used an 8x8 tile, but 8x8/IC128 does not fit L1 with
# the current full-weight broadcast. 4x4 keeps calls_per_ocb at one for 20x20.
CONV_REGIME_ARTIFACTS = {
    "regime_r3_conv3x3": ConvRegimeArtifact(
        regime="R3",
        xclbin_name="regime_r3_conv3x3",
        kernel_size=3,
        stride=1,
        tile_h=4,
        tile_w=4,
        ic=128,
        oc_block=16,
        patches_per_core=1,
        input_depth=1,
        members={
            # name: active tile_h, tile_w, ic, oc_block, stride, padding, ppc
            "mc_re8_c3": (4, 4, 128, 16, 1, 1, 1),
            "mc_re8_rn3": (4, 4, 64, 16, 1, 1, 1),
        },
    ),
}


GEMM_REGIME_ARTIFACTS = {
    "regime_r3_gemm_non_k": GemmRegimeArtifact(
        regime="R3",
        xclbin_name="regime_r3_gemm_non_k",
        tile_m=44,
        ic=128,
        oc=128,
        patches_per_core=1,
        k_block=0,
        members={
            # name: active tile_m, ic, oc, ppc
            "gemm_re8_rn1": (44, 128, 64, 1),
            "gemm_re8_rnm": (44, 128, 128, 1),
        },
    ),
}


def conv_regime_for_layer(layer_name: str):
    for artifact in CONV_REGIME_ARTIFACTS.values():
        if layer_name in artifact.members:
            return artifact
    return None


def gemm_regime_for_layer(layer_name: str):
    for artifact in GEMM_REGIME_ARTIFACTS.values():
        if layer_name in artifact.members:
            return artifact
    return None
