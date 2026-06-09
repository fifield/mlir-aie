#!/usr/bin/env python3
"""Scalar PythoC fused rn3 3x3+3x3 prototype kernel.

This is intentionally a correctness/bring-up kernel, not the final optimized
rn3 implementation. It uses a simple contiguous OIHW weight layout and
recomputes conv1 values needed by conv2 rather than using vectorized mmul or a
scratch tile. Use tiny shapes for first NPU smoke tests.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from aie.iron.pythoc import PythocKernel, aie_kernel
from pythoc import ptr, i32, bf16, f32, void
from pythoc.builtin_entities import array
from pythoc.aie.profiling import event0, event1

# Reuse the vectorizer-disable patch installed by the existing kernel module.
try:
    import rep_elan_bf16_pythoc  # noqa: F401
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = HERE / "build"


@aie_kernel
def rn3_pair_fused_bf16(
    input: ptr[bf16, True],
    weights: ptr[bf16, True],
    output: ptr[bf16, True],
    tile_h: i32,
    tile_w: i32,
    ic: i32,
    mid: i32,
    ocb: i32,
) -> void:
    event0()

    patch_w: i32 = tile_w + 4
    mid_w: i32 = tile_w + 2
    w1_size: i32 = mid * ic * 9
    bn1_w_off: i32 = w1_size
    bn1_b_off: i32 = bn1_w_off + mid
    w2_off: i32 = bn1_b_off + mid
    w2_size: i32 = ocb * mid * 9
    bn2_w_off: i32 = w2_off + w2_size
    bn2_b_off: i32 = bn2_w_off + ocb

    if ocb == 4:
        # Row-band shared-intermediate correctness path for the re6_oc4 path.
        # Keep only one final output row's four-channel accumulators live at a
        # time (8 pixels × 4 channels = 32 f32 values), and compute only the
        # 3×10 conv1 intermediate band needed by that output row. This keeps
        # live state below the larger-accumulator NaN threshold while avoiding
        # the previous row-sliced path's wasted 10×10 midplane work per row.
        midband4: array[f32, 30] = array[f32, 30]()
        accrow4: array[f32, 32] = array[f32, 32]()

        r4: i32 = 0
        while r4 < tile_h:
            z4: i32 = 0
            while z4 < 32:
                accrow4[z4] = 0.0
                z4 = z4 + 1

            o14: i32 = 0
            while o14 < mid:
                # Compute only rows r4..r4+2 and columns 0..9 of the conv1
                # intermediate plane, which are exactly the values conv2 needs
                # for final output row r4.
                br4: i32 = 0
                while br4 < 3:
                    mc4: i32 = 0
                    while mc4 < 10:
                        acc14: f32 = 0.0
                        kh14: i32 = 0
                        while kh14 < 3:
                            kw14: i32 = 0
                            while kw14 < 3:
                                i4: i32 = 0
                                while i4 < ic:
                                    in_idx4: i32 = ((r4 + br4 + kh14) * patch_w + (mc4 + kw14)) * ic + i4
                                    w1_idx4: i32 = ((o14 * ic + i4) * 3 + kh14) * 3 + kw14
                                    acc14 = acc14 + f32(input[in_idx4]) * f32(weights[w1_idx4])
                                    i4 = i4 + 1
                                kw14 = kw14 + 1
                            kh14 = kh14 + 1
                        x14: f32 = acc14 * f32(weights[bn1_w_off + o14]) + f32(weights[bn1_b_off + o14])
                        ax14: f32 = x14
                        if x14 < 0.0:
                            ax14 = -x14
                        midband4[br4 * 10 + mc4] = x14 * (0.5 + x14 / (2.0 + 2.0 * ax14))
                        mc4 = mc4 + 1
                    br4 = br4 + 1

                c4: i32 = 0
                while c4 < tile_w:
                    kh24: i32 = 0
                    while kh24 < 3:
                        kw24: i32 = 0
                        while kw24 < 3:
                            y14: f32 = midband4[kh24 * 10 + (c4 + kw24)]
                            w2_base4: i32 = w2_off + ((o14 * 3 + kh24) * 3 + kw24)
                            aidx4: i32 = c4 * 4
                            accrow4[aidx4] = accrow4[aidx4] + y14 * f32(weights[w2_base4])
                            accrow4[aidx4 + 1] = accrow4[aidx4 + 1] + y14 * f32(weights[w2_base4 + mid * 9])
                            accrow4[aidx4 + 2] = accrow4[aidx4 + 2] + y14 * f32(weights[w2_base4 + 2 * mid * 9])
                            accrow4[aidx4 + 3] = accrow4[aidx4 + 3] + y14 * f32(weights[w2_base4 + 3 * mid * 9])
                            kw24 = kw24 + 1
                        kh24 = kh24 + 1
                    c4 = c4 + 1
                o14 = o14 + 1

            c24: i32 = 0
            while c24 < tile_w:
                o24: i32 = 0
                while o24 < 4:
                    idx24: i32 = c24 * 4 + o24
                    x24: f32 = accrow4[idx24] * f32(weights[bn2_w_off + o24]) + f32(weights[bn2_b_off + o24])
                    ax24: f32 = x24
                    if x24 < 0.0:
                        ax24 = -x24
                    output[(r4 * tile_w + c24) * 4 + o24] = bf16(x24 * (0.5 + x24 / (2.0 + 2.0 * ax24)))
                    o24 = o24 + 1
                c24 = c24 + 1
            r4 = r4 + 1
    else:
        r: i32 = 0
        while r < tile_h:
            c: i32 = 0
            while c < tile_w:
                o2: i32 = 0
                while o2 < ocb:
                    acc2: f32 = 0.0

                    kh2: i32 = 0
                    while kh2 < 3:
                        kw2: i32 = 0
                        while kw2 < 3:
                            mid_r: i32 = r + kh2
                            mid_c: i32 = c + kw2
                            o1: i32 = 0
                            while o1 < mid:
                                acc1: f32 = 0.0
                                kh1: i32 = 0
                                while kh1 < 3:
                                    kw1: i32 = 0
                                    while kw1 < 3:
                                        i: i32 = 0
                                        while i < ic:
                                            in_idx: i32 = ((mid_r + kh1) * patch_w + (mid_c + kw1)) * ic + i
                                            w1_idx: i32 = ((o1 * ic + i) * 3 + kh1) * 3 + kw1
                                            acc1 = acc1 + f32(input[in_idx]) * f32(weights[w1_idx])
                                            i = i + 1
                                        kw1 = kw1 + 1
                                    kh1 = kh1 + 1

                                x1: f32 = acc1 * f32(weights[bn1_w_off + o1]) + f32(weights[bn1_b_off + o1])
                                ax1: f32 = x1
                                if x1 < 0.0:
                                    ax1 = -x1
                                sig1: f32 = 0.5 + x1 / (2.0 + 2.0 * ax1)
                                y1: f32 = x1 * sig1

                                w2_idx: i32 = w2_off + ((o2 * mid + o1) * 3 + kh2) * 3 + kw2
                                acc2 = acc2 + y1 * f32(weights[w2_idx])
                                o1 = o1 + 1
                            kw2 = kw2 + 1
                        kh2 = kh2 + 1

                    x2: f32 = acc2 * f32(weights[bn2_w_off + o2]) + f32(weights[bn2_b_off + o2])
                    ax2: f32 = x2
                    if x2 < 0.0:
                        ax2 = -x2
                    sig2: f32 = 0.5 + x2 / (2.0 + 2.0 * ax2)
                    output[(r * tile_w + c) * ocb + o2] = bf16(x2 * sig2)
                    o2 = o2 + 1
                c = c + 1
            r = r + 1

    event1()


def make_rn3_pair_fused_bf16(input_ty, weight_ty, out_ty) -> PythocKernel:
    return PythocKernel(
        rn3_pair_fused_bf16,
        [input_ty, weight_ty, out_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
        extra_globals={"array": array},
    )


def build_obj(build_dir: Optional[Path] = None) -> Path:
    build_dir = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    input_ty = np.ndarray[(16384,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(81920,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(8192,), np.dtype[np.uint16]]
    k = make_rn3_pair_fused_bf16(input_ty, weight_ty, out_ty)
    dst = build_dir / "rn3_pair_fused_bf16.o"
    shutil.copyfile(Path(k.object_file_name), dst)
    return dst


if __name__ == "__main__":
    print(build_obj())
