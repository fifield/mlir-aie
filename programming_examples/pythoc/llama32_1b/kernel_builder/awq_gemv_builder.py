# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWQ uint4 GEMV kernel-builder entry points.

The first fused-AWQ primitive is dimension-specialized and kept separate from
existing BF16 GEMV cache entries. It consumes packed uint4 weights and AWQ params
through a scalar external AIE kernel first; vectorization/tiling can replace the
external kernel internals without changing the Python/runtime ABI.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


def _validate_dims(k: int, m: int, group_size: int) -> tuple[int, int, int]:
    k = int(k)
    m = int(m)
    group_size = int(group_size)
    if k <= 0 or m <= 0 or group_size <= 0:
        raise ValueError(
            f"AWQ GEMV dimensions must be positive, got k={k}, m={m}, group_size={group_size}"
        )
    if k % 2 != 0:
        raise ValueError(f"AWQ GEMV K must be even for uint4 byte packing, got {k}")
    if k % group_size != 0:
        raise ValueError(
            f"AWQ GEMV K={k} must be divisible by group_size={group_size} for the initial kernel"
        )
    return k, m, group_size


_AWQ_GEMV_VARIANTS = {"scalar", "vecdeq"}


def _validate_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in _AWQ_GEMV_VARIANTS:
        raise ValueError(
            f"Unsupported AWQ GEMV variant {variant!r}; expected one of {sorted(_AWQ_GEMV_VARIANTS)}"
        )
    return variant


def awq_gemv_kernel_name(k: int, m: int, group_size: int, *, variant: str = "scalar") -> str:
    """Return a cache-safe name for a specialized packed-AWQ GEMV kernel."""
    k, m, group_size = _validate_dims(k, m, group_size)
    variant = _validate_variant(variant)
    return f"awq_gemv_k{k}_m{m}_g{group_size}_{variant}"


def awq_gemv_object_name(k: int, m: int, group_size: int, *, variant: str = "scalar") -> str:
    """Return the object filename linked by the specialized AWQ GEMV IR.

    Stage 2 ports to PythoC: object names get the ``_pythoc`` suffix to
    mirror the other PythoC-built ``.o`` outputs (mv_pythoc.o, attn_pythoc.o,
    ...). This matches the link_with strings in the cached MLIR after the
    Stage-2 sed-swap.
    """
    return f"{awq_gemv_kernel_name(k, m, group_size, variant=variant)}_pythoc.o"


def build_awq_gemv_air_module(k: int, m: int, group_size: int, *, variant: str = "scalar"):
    """Build the AIR module for one packed uint4 AWQ GEMV primitive.

    ABI:
      x:       bf16[K]
      qweight: uint8[M * (K / 2)], row-major, low nibble = even K,
               high nibble = odd K
      params:  bf16[M * 2 * (K / group_size)], row-major/interleaved [scale, zero]
      y:       bf16[M]
    """
    k, m, group_size = _validate_dims(k, m, group_size)
    variant = _validate_variant(variant)
    groups = k // group_size
    q_cols = k // 2
    obj_name = awq_gemv_object_name(k, m, group_size, variant=variant)

    from air.backend.xrt_runner import type_mapper
    from air.dialects.air import (
        IntegerAttr,
        MemorySpace,
        MemRefType,
        StringAttr,
        T,
        UnitAttr,
        dma_memcpy_nd,
        herd,
        launch,
        module_builder,
        segment,
    )
    from air.dialects.func import CallOp, FuncOp
    from air.dialects.memref import AllocOp, DeallocOp

    @module_builder
    def _module():
        bf16_ty = type_mapper(bfloat16)
        u8_ty = type_mapper(np.uint8)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        x_l3_ty = MemRefType.get([k], bf16_ty)
        q_l3_ty = MemRefType.get([m * q_cols], u8_ty)
        p_l3_ty = MemRefType.get([m * 2 * groups], bf16_ty)
        y_l3_ty = MemRefType.get([m], bf16_ty)

        x_l1_ty = MemRefType.get([k], bf16_ty, memory_space=l1_space)
        q_l1_ty = MemRefType.get([m * q_cols], u8_ty, memory_space=l1_space)
        p_l1_ty = MemRefType.get([m * 2 * groups], bf16_ty, memory_space=l1_space)
        y_l1_ty = MemRefType.get([m], bf16_ty, memory_space=l1_space)

        awq_func = FuncOp(
            "awq_gemv_u4_bf16",
            ([x_l1_ty, q_l1_ty, p_l1_ty, y_l1_ty], []),
            visibility="private",
        )
        awq_func.attributes["link_with"] = StringAttr.get(obj_name)
        awq_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(x_l3_ty, q_l3_ty, p_l3_ty, y_l3_ty)
        def awq_gemv(x, qweight, params, y):
            @launch(operands=[x, qweight, params, y])
            def awq_launch(l_x, l_q, l_p, l_y):
                @segment(name="awq_gemv_seg", operands=[l_x, l_q, l_p, l_y])
                def awq_seg(s_x, s_q, s_p, s_y):
                    @herd(name="awq_gemv_herd", sizes=[1, 1], operands=[s_x, s_q, s_p, s_y])
                    def herd_body(_tx, _ty, _sx, _sy, h_x, h_q, h_p, h_y):
                        l1_x = AllocOp(x_l1_ty, [], [])
                        l1_q = AllocOp(q_l1_ty, [], [])
                        l1_p = AllocOp(p_l1_ty, [], [])
                        l1_y = AllocOp(y_l1_ty, [], [])

                        dma_memcpy_nd(l1_x, h_x)
                        dma_memcpy_nd(l1_q, h_q)
                        dma_memcpy_nd(l1_p, h_p)

                        CallOp(awq_func, [l1_x, l1_q, l1_p, l1_y])

                        dma_memcpy_nd(h_y, l1_y)

                        DeallocOp(l1_x)
                        DeallocOp(l1_q)
                        DeallocOp(l1_p)
                        DeallocOp(l1_y)

                    herd_body.attributes["link_with"] = StringAttr.get(obj_name)

    return _module()


def build_awq_gemv_ir(
    k: int,
    m: int,
    group_size: int,
    *,
    variant: str = "scalar",
    verbose: bool = False,
) -> str:
    """Build post-stitched npu.air.mlir for a packed uint4 AWQ GEMV primitive."""
    k, m, group_size = _validate_dims(k, m, group_size)
    variant = _validate_variant(variant)
    try:
        from .external_kernels import compile_awq_gemv
        from .aie_ir_gen import lower_air_to_npu_air_mlir
    except ImportError:  # Support direct script-style imports from kernel_builder/.
        from external_kernels import compile_awq_gemv
        from aie_ir_gen import lower_air_to_npu_air_mlir

    compile_awq_gemv(k, m, group_size, variant=variant)
    mod = build_awq_gemv_air_module(k, m, group_size, variant=variant)
    return lower_air_to_npu_air_mlir(
        str(mod),
        device="npu2",
        num_cols=1,
        omit_pingpong="all",
        runtime_loop_tiling_sizes=[1, 1],
        use_lock_race_condition_fix=False,
        verbose=verbose,
    )
