# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed-AWQ O GEMV + FFN multi-launch ELF for decode.

This mirrors o_gemv_ffn_multi.py but replaces the four BF16 GEMVs with packed
uint4 AWQ GEMVs:
  O/Gate/Up:   K=2048, group=128, awq_mv.o
  Down:        K=8192, group=128, awq_mv_k8192.o with renamed externs

Each AWQ linear is consumed as a single combined uint8 buffer per row:
  [qweight_bytes (K/2)] [params_bytes (2*groups bf16 = 4*groups bytes)]
which halves the weight DMA channels relative to the separate (qweight,
params) ABI. That brought the fused module's shim DMA usage back inside the
device's channel budget.

Combined ABI:
  0 wo_w        u8[emb_dim, emb_dim/2 + 4*(emb_dim/group_size)]
  1 attn_out    bf16[emb_dim]
  2 proj        bf16[emb_dim]
  3 x_residual  bf16[emb_dim]
  4 res1        bf16[emb_dim]
  5 ffn_norm_w  bf16[emb_dim]
  6 normed2     bf16[emb_dim]
  7 wgate_w     u8[hidden_dim, emb_dim/2 + 4*(emb_dim/group_size)]
  8 gate        bf16[hidden_dim]
  9 wup_w       u8[hidden_dim, emb_dim/2 + 4*(emb_dim/group_size)]
 10 up          bf16[hidden_dim]
 11 swiglu      bf16[hidden_dim]
 12 wdown_w     u8[emb_dim, hidden_dim/2 + 4*(hidden_dim/group_size)]
 13 down        bf16[emb_dim]
 14 output      bf16[emb_dim]
"""

import os
import re
import sys

from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _build_rms_1d_ir(emb_dim, vector_size=16):
    """Inlined 1D RMSNorm AIR builder for the AWQ stitched decode module.

    Stage-1 scaffolding only -- this is a verbatim copy of
    `multi_launch_builder.o_gemv_ffn_multi._build_rms_1d_ir` from the AIR
    worktree.  The pythoc tree intentionally does not ship the BF16 AIR-tree
    stitching module (it uses placed-IRON for that path today); we inline the
    one helper the AWQ stitcher needs.  Stage 2 retires this whole module.
    """
    from air.ir import (
        Context,
        Module,
        MemRefType,
        VectorType,
        IntegerAttr,
        AffineMap,
        AffineMapAttr,
        F32Type,
    )
    from air.dialects.air import (
        module_builder,
        MemorySpace,
        launch,
        segment,
        herd,
        dma_memcpy_nd,
    )
    from air.dialects import arith, math as math_dialect
    from air.dialects.memref import (
        AllocOp,
        DeallocOp,
        subview,
        expand_shape as memref_expand_shape,
    )
    from air.dialects.vector import (
        transfer_read,
        transfer_write,
        BroadcastOp,
        reduction as vector_reduction,
    )
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_, yield_
    from air.backend.xrt_runner import type_mapper

    n = emb_dim

    @module_builder
    def _build():
        from air.dialects.air import T

        xrt_dtype = type_mapper(bfloat16)
        N = n
        EPS = 1e-5

        vecTy_g = VectorType.get([vector_size], xrt_dtype)
        identity_map_g = AffineMapAttr.get(AffineMap.get_identity(1))

        l3_1d_ty = MemRefType.get([N], xrt_dtype)
        l3_2d_ty = MemRefType.get([1, N], xrt_dtype)
        l3_weight_ty = MemRefType.get([N], xrt_dtype)

        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_row_ty = MemRefType.get([N], xrt_dtype, memory_space=l1_space)
        l1_vec_ty = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_space)

        @FuncOp.from_py_func(l3_1d_ty, l3_weight_ty, l3_1d_ty)
        def weighted_rms_norm_1d(arg0, arg1, arg2):
            @launch(operands=[arg0, arg1, arg2])
            def rms_launch(l_in, l_weight, l_out):
                in_2d = memref_expand_shape(l3_2d_ty, l_in, [[0, 1]], [], [1, n])
                out_2d = memref_expand_shape(l3_2d_ty, l_out, [[0, 1]], [], [1, n])

                @segment(name="rms_seg", operands=[in_2d, l_weight, out_2d])
                def rms_seg(s_in, s_weight, s_out):
                    @herd(
                        name="herd_0",
                        sizes=[1, 1],
                        operands=[s_in, s_weight, s_out],
                    )
                    def herd_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
                        l1_row = AllocOp(l1_row_ty, [], [])
                        l1_out = AllocOp(l1_row_ty, [], [])
                        l1_weight = AllocOp(l1_row_ty, [], [])
                        l1_acc = AllocOp(l1_vec_ty, [], [])

                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        n_f = arith.ConstantOp(xrt_dtype, float(N))
                        eps_f = arith.ConstantOp(xrt_dtype, EPS)

                        v_zero = BroadcastOp(vecTy_g, cst0)

                        dma_memcpy_nd(l1_weight, l3_weight)

                        dma_memcpy_nd(
                            l1_row,
                            l3_in,
                            src_offsets=[0, 0],
                            src_sizes=[1, N],
                            src_strides=[N, 1],
                        )

                        transfer_write(None, v_zero, l1_acc, [c0], identity_map_g, [True])
                        for j in for_(0, N, vector_size):
                            sub_row = subview(l1_row.result, [j], [vector_size], [1])
                            sub_tmp = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(vecTy_g, sub_row, [c0], identity_map_g, cst0, [True])
                            v_sq = arith.mulf(v_x, v_x)
                            transfer_write(None, v_sq, sub_tmp, [c0], identity_map_g, [True])
                            v_sq_rd = transfer_read(vecTy_g, sub_tmp, [c0], identity_map_g, cst0, [True])
                            v_acc = transfer_read(vecTy_g, l1_acc, [c0], identity_map_g, cst0, [True])
                            v_sum = arith.addf(v_acc, v_sq_rd)
                            transfer_write(None, v_sum, l1_acc, [c0], identity_map_g, [True])
                            yield_([])

                        v_final = transfer_read(vecTy_g, l1_acc, [c0], identity_map_g, cst0, [True])
                        total_sum = vector_reduction(xrt_dtype, "add", v_final)
                        rms = arith.divf(total_sum, n_f)

                        f32 = F32Type.get()
                        rms_eps = arith.addf(rms, eps_f)
                        rms_eps_f32 = arith.extf(f32, rms_eps)
                        rstd_f32 = math_dialect.rsqrt(rms_eps_f32)
                        rstd = arith.truncf(xrt_dtype, rstd_f32)

                        v_rstd = BroadcastOp(vecTy_g, rstd)
                        for j in for_(0, N, vector_size):
                            sub_row = subview(l1_row.result, [j], [vector_size], [1])
                            sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_x = transfer_read(vecTy_g, sub_row, [c0], identity_map_g, cst0, [True])
                            v_w = transfer_read(vecTy_g, sub_w, [c0], identity_map_g, cst0, [True])
                            v_normed = arith.mulf(v_x, v_rstd)
                            v_weighted = arith.mulf(v_normed, v_w)
                            transfer_write(None, v_weighted, sub_out, [c0], identity_map_g, [True])
                            yield_([])

                        dma_memcpy_nd(
                            l3_out,
                            l1_out,
                            dst_offsets=[0, 0],
                            dst_sizes=[1, N],
                            dst_strides=[N, 1],
                        )

                        DeallocOp(l1_row)
                        DeallocOp(l1_out)
                        DeallocOp(l1_weight)
                        DeallocOp(l1_acc)

    return str(_build())


def build_o_gemv_ffn_awq_module(
    emb_dim=2048,
    hidden_dim=8192,
    group_size=128,
    tile_m=8,
    m_input=4,
    down_tile_m=2,
    down_m_input=1,
    herd_m=8,
):
    """Build an 8-launch fused O+FFN decode module using packed-AWQ GEMVs."""
    # Lazy-import stitching helpers -- the pythoc tree does not vendor the AIR
    # stitching module by default; these AWQ scaffolding paths are Stage-1 only.
    from kernel_builder.stitching import (
        _extract_affine_maps,
        _extract_between_func_and_return,
        _extract_private_funcs,
        _fix_launch_func_args,
        _rename_all_with_externs,
        _wrap_ir_in_launch,
    )
    if emb_dim % group_size != 0 or hidden_dim % group_size != 0:
        raise ValueError(
            f"emb_dim={emb_dim} and hidden_dim={hidden_dim} must be divisible by group_size={group_size}"
        )
    if emb_dim % 2 != 0 or hidden_dim % 2 != 0:
        raise ValueError("AWQ packed uint4 dimensions must be even")

    from kernel_builder.awq_matvec import build_module as build_awq_matvec, combined_row_bytes
    from eltwise_add.eltwise_add import build_module as build_add
    from kernel_builder.ffn_swiglu.silu_and_mul import build_module as build_silu

    print("  [1/8] O AWQ GEMV...")
    o_gemv_ir = str(build_awq_matvec(emb_dim, emb_dim, group_size, tile_m, m_input, herd_m))

    print("  [2/8] Eltwise Add (post-attn residual)...")
    add1_ir = _wrap_ir_in_launch(
        str(build_add(emb_dim, emb_dim // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1))
    )

    print("  [3/8] RMSNorm (1D decode)...")
    rms_ir = _build_rms_1d_ir(emb_dim, vector_size=16)

    print("  [4/8] Gate AWQ GEMV...")
    gate_ir = str(build_awq_matvec(hidden_dim, emb_dim, group_size, tile_m, m_input, herd_m))

    print("  [5/8] Up AWQ GEMV...")
    up_ir = str(build_awq_matvec(hidden_dim, emb_dim, group_size, tile_m, m_input, herd_m))

    print("  [6/8] SiLU x mul...")
    silu_ir = _wrap_ir_in_launch(
        str(build_silu(hidden_dim, hidden_dim // 8, bfloat16, herd_x=8, herd_y=1))
    )

    print("  [7/8] Down AWQ GEMV...")
    down_ir = str(
        build_awq_matvec(emb_dim, hidden_dim, group_size, down_tile_m, down_m_input, herd_m)
    )

    print("  [8/8] Eltwise Add (FFN residual)...")
    add2_ir = _wrap_ir_in_launch(
        str(build_add(emb_dim, emb_dim // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1))
    )

    # Combined ABI: AWQ matvec is now (weight, x, y) -- 3 args per GEMV launch.
    stitch_specs = [
        (o_gemv_ir, "og", {0: 0, 1: 1, 2: 2}),         # O: wo_w, attn_out, proj
        (add1_ir, "a1", {0: 2, 1: 3, 2: 4}),           # proj + x_residual -> res1
        (rms_ir, "rm", {0: 4, 1: 5, 2: 6}),            # rms(res1, ffn_norm_w) -> normed2
        (gate_ir, "gg", {0: 7, 1: 6, 2: 8}),           # gate: wgate_w, normed2, gate
        (up_ir, "ug", {0: 9, 1: 6, 2: 10}),            # up:   wup_w,   normed2, up
        (silu_ir, "sw", {0: 8, 1: 10, 2: 11}),         # silu(gate) * up -> swiglu
        (down_ir, "dg", {0: 12, 1: 11, 2: 13}),        # down: wdown_w, swiglu, down
        (add2_ir, "a2", {0: 13, 1: 4, 2: 14}),         # down + res1 -> output
    ]

    extern_shared = {
        "@awq_matvec_vectorized_u4_bf16",
        "@awq_linalg_fill_bf16",
        "@silu_and_mul_bf16",
    }
    # For down, do not preserve AWQ matvec/fill symbols: prefix to dg_* and link
    # against awq_mv_k8192.o, whose C symbols are compiled with matching names.
    extern_down = {"@silu_and_mul_bf16"}

    bodies, maps_all = [], []
    for ir, prefix, arg_map in stitch_specs:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        externs = extern_down if prefix == "dg" else extern_shared
        body = _rename_all_with_externs(body, prefix, externs)
        maps = [_rename_all_with_externs(m, prefix, externs) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        if prefix == "dg":
            body = body.replace('link_with = "awq_mv.o"', 'link_with = "awq_mv_k8192.o"')
        bodies.append(body)
        maps_all.extend(maps)

    shared_privates = _extract_private_funcs(o_gemv_ir) + _extract_private_funcs(silu_ir)
    down_privates = []
    for p in _extract_private_funcs(down_ir):
        p = _rename_all_with_externs(p, "dg", extern_down)
        p = p.replace('link_with = "awq_mv.o"', 'link_with = "awq_mv_k8192.o"')
        down_privates.append(p.strip())

    seen_funcs = set()
    all_privates = []
    for p in shared_privates + down_privates:
        fname = re.search(r"@(\w+)", p)
        if fname and fname.group(1) not in seen_funcs:
            seen_funcs.add(fname.group(1))
            all_privates.append(p.strip())

    wcols_emb_in = combined_row_bytes(emb_dim, group_size)
    wcols_hidden_in = combined_row_bytes(hidden_dim, group_size)
    combined = "\n".join(maps_all) + f"""
module {{
  {chr(10).join('  ' + p for p in all_privates)}
  func.func @o_gemv_ffn_awq(
    %arg0: memref<{emb_dim}x{wcols_emb_in}xui8>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}xbf16>,
    %arg4: memref<{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}xbf16>,
    %arg6: memref<{emb_dim}xbf16>,
    %arg7: memref<{hidden_dim}x{wcols_emb_in}xui8>,
    %arg8: memref<{hidden_dim}xbf16>,
    %arg9: memref<{hidden_dim}x{wcols_emb_in}xui8>,
    %arg10: memref<{hidden_dim}xbf16>,
    %arg11: memref<{hidden_dim}xbf16>,
    %arg12: memref<{emb_dim}x{wcols_hidden_in}xui8>,
    %arg13: memref<{emb_dim}xbf16>,
    %arg14: memref<{emb_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Context, Module

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, 15 args, 8 launches")
        return module
