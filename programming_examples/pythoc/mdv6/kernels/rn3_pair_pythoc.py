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
