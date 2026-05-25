# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed-AWQ GEMV AIR builder with the same tiling shape as bf16/matvec.py.

C[M] = dequant_u4_awq(W[M, K/2 + 4*groups]) @ B[K]

W is a single combined uint8 buffer per linear. Each row is laid out as:
  [qweight_bytes (K/2)] [params_bytes (2*groups bf16 = 4*groups bytes)]

The two-buffer (qweight, params) ABI exceeded the device's shim DMA channel
budget once stitched into the fused O+FFN decode kernel; folding them into a
single per-row byte buffer cuts the weight DMAs per GEMV launch in half.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

range_ = for_


def combined_row_bytes(k, group_size):
    """Bytes per combined-row uint8 buffer for one AWQ linear row."""
    if k % 2 != 0:
        raise ValueError(f"K ({k}) must be even for uint4 packing")
    if k % group_size != 0:
        raise ValueError(f"K ({k}) must be divisible by group_size ({group_size})")
    groups = k // group_size
    return k // 2 + 4 * groups  # 2 * groups bf16 == 4 * groups bytes


@module_builder
def build_module(m, k, group_size, tile_m, m_input, herd_m):
    assert m % (tile_m * herd_m) == 0, (
        f"M ({m}) must be divisible by tile_m * herd_m ({tile_m * herd_m})"
    )
    assert tile_m % m_input == 0, (
        f"tile_m ({tile_m}) must be divisible by m_input ({m_input})"
    )
    assert k % 2 == 0, f"K ({k}) must be even for uint4 packing"
    assert k % group_size == 0, f"K ({k}) must be divisible by group_size ({group_size})"

    w_cols = combined_row_bytes(k, group_size)

    bf16_ty = type_mapper(bfloat16)
    u8_ty = type_mapper(np.uint8)

    w_l3_ty = MemRefType.get([m, w_cols], u8_ty)
    x_l3_ty = MemRefType.get([k], bf16_ty)
    y_l3_ty = MemRefType.get([m], bf16_ty)

    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    w_l2_ty = MemRefType.get([herd_m, tile_m, w_cols], u8_ty, memory_space=l2_space)
    y_l2_ty = MemRefType.get([herd_m, tile_m], bf16_ty, memory_space=l2_space)

    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    w_l1_ty = MemRefType.get([m_input, w_cols], u8_ty, memory_space=l1_space)
    x_l1_ty = MemRefType.get([k], bf16_ty, memory_space=l1_space)
    y_l1_ty = MemRefType.get([tile_m], bf16_ty, memory_space=l1_space)

    awq_func = FuncOp(
        "awq_matvec_vectorized_u4_bf16",
        ([T.i32(), T.i32(), T.i32(), w_l1_ty, x_l1_ty, y_l1_ty], []),
        visibility="private",
    )
    fill_func = FuncOp(
        "awq_linalg_fill_bf16",
        ([bf16_ty, y_l1_ty], []),
        visibility="private",
    )
    for func in [awq_func, fill_func]:
        func.attributes["link_with"] = StringAttr.get("awq_mv.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(w_l3_ty, x_l3_ty, y_l3_ty)
    def awq_matvec(arg0, arg1, arg2):
        launch_size = [m // tile_m // herd_m, 1]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_w_data,
            l3_x_data,
            l3_y_data,
        ):
            @segment(
                name="awq_matvec_0",
                operands=[launch_ivx, l3_w_data, l3_x_data, l3_y_data],
            )
            def segment_body(launch_ivx_s, l3_w_s, l3_x_s, l3_y_s):
                launch_ivx_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                launch_offset_m = affine_apply(launch_ivx_map, [launch_ivx_s])

                l2_w = AllocOp(w_l2_ty, [], [])
                l2_y = AllocOp(y_l2_ty, [], [])
                l1_w = AllocOp(w_l1_ty, [], [])
                l1_x = AllocOp(x_l1_ty, [], [])
                l1_y = AllocOp(y_l1_ty, [], [])

                dma_memcpy_nd(
                    l2_w,
                    l3_w_s,
                    src_offsets=[0, launch_offset_m, 0],
                    src_sizes=[herd_m, tile_m, w_cols],
                    src_strides=[tile_m * w_cols, w_cols, 1],
                )

                @herd(
                    name="herd_0",
                    sizes=[herd_m, 1],
                    operands=[l1_w, l1_x, l1_y, l2_w, l3_x_s, l2_y],
                )
                def herd_body(_tx, _ty, _sx, _sy, _l1_w, _l1_x, _l1_y, _l2_w, _l3_x, _l2_y):
                    zero = ConstantOp(FloatAttr.get(bf16_ty, 0), None)
                    CallOp(fill_func, [zero, _l1_y])

                    for j_m in range_(0, tile_m // m_input):
                        j_m_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(m_input),
                                )
                            ],
                        )
                        j_m_offset = affine_apply(j_m_map, [j_m])

                        dma_memcpy_nd(_l1_x, _l3_x, src_offsets=[], src_sizes=[k], src_strides=[1])
                        dma_memcpy_nd(
                            _l1_w,
                            _l2_w,
                            src_offsets=[_tx, j_m_offset, 0],
                            src_sizes=[1, m_input, w_cols],
                            src_strides=[tile_m * w_cols, w_cols, 1],
                        )

                        row_offset_i32 = arith.index_cast(T.i32(), j_m_offset)
                        m_const = ConstantOp(IntegerAttr.get(T.i32(), m_input), None)
                        k_const = ConstantOp(IntegerAttr.get(T.i32(), k), None)
                        CallOp(awq_func, [m_const, k_const, row_offset_i32, _l1_w, _l1_x, _l1_y])
                        yield_([])

                    dma_memcpy_nd(
                        _l2_y,
                        _l1_y,
                        dst_offsets=[_tx, 0],
                        dst_sizes=[1, tile_m],
                        dst_strides=[tile_m, 1],
                        src_offsets=[],
                        src_sizes=[tile_m],
                        src_strides=[1],
                    )

                herd_body.attributes["link_with"] = StringAttr.get("awq_mv.o")

                dma_memcpy_nd(
                    l3_y_s,
                    l2_y,
                    dst_offsets=[launch_offset_m],
                    dst_sizes=[herd_m * tile_m],
                    dst_strides=[1],
                    src_offsets=[0, 0],
                    src_sizes=[herd_m, tile_m],
                    src_strides=[tile_m, 1],
                )

                DeallocOp(l2_w)
                DeallocOp(l2_y)
                DeallocOp(l1_w)
                DeallocOp(l1_x)
                DeallocOp(l1_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=8)
    parser.add_argument("--m-input", type=int, default=4)
    parser.add_argument("--herd-m", type=int, default=8)
    args = parser.parse_args()
    print(build_module(args.m, args.k, args.group_size, args.tile_m, args.m_input, args.herd_m))
