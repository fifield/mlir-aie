"""BFP576 conv2res: 3x3 conv + BN + SiLU + residual, drop-in for chain_conv2res_bf16.

Same signature. Replaces the emulated 4x8x8 mmul with the BFP576 8x8x8 MAC.
conv2 output is 8x8 (64 pos) and OW=8 DIVIDES the 8-pixel block -> NO row-aligned
padding needed (unlike conv1's OW=10): every M-block is exactly one output row, so
the on-the-fly im2col's per-pixel o is constant per block. ic=16 (K_MICRO=18, 2 ic
groups x 9 taps), 16 oc (N_BLOCKS=2), MB=8 rows, 2x2 register-blocked m/n loops.
A from the conv1 scratch (10x10x16), B = chain weights [oc_block,kk,ic,oc]+BN, MAC,
then _store_bn_silu_res (BN+SiLU+residual) -- same store as the emulated kernel.
"""
from __future__ import annotations

from pythoc import ptr, i32, bf16, f32, void
from pythoc.aie import (
    aie_vector, load_v, store_v, concat, vector_cast, vector_extract,
    vshuffle, zeros, set_ctrl_reg, v32bf16_to_v32accfloat,
    v32accfloat_to_v32bf16, v64accfloat_to_v64bfp16ebs8,
    BFP576_BFP576_ACC2048_mac_conf,
)
from aie.iron.pythoc import aie_kernel
from kernels.rn3_chain_pythoc import _store_bn_silu_res_4x8_rows
from pythoc.aie.profiling import event0, event1

MAC_CONF = 780


@aie_kernel
def _build_a64_3x3_c2(input: ptr[bf16, True], sp: i32, tile_w: i32, stride: i32,
                      patch_w: i32, ic: i32, ic_blk: i32, kh: i32,
                      kw: i32) -> aie_vector[bf16, 64]:
    """Distinct-symbol copy of _build_a64_3x3 (avoids a duplicate symbol when
    both conv1 and conv2res BFP kernels are compiled in the same design)."""
    p: i32 = ic_blk * 8
    o0: i32 = (sp + 0) // tile_w
    a0: aie_vector[bf16, 8] = load_v(input + (((o0 * stride + kh) * patch_w) + ((sp + 0) - o0 * tile_w) * stride + kw) * ic + p, 8)
    o1: i32 = (sp + 1) // tile_w
    a1: aie_vector[bf16, 8] = load_v(input + (((o1 * stride + kh) * patch_w) + ((sp + 1) - o1 * tile_w) * stride + kw) * ic + p, 8)
    o2: i32 = (sp + 2) // tile_w
    a2: aie_vector[bf16, 8] = load_v(input + (((o2 * stride + kh) * patch_w) + ((sp + 2) - o2 * tile_w) * stride + kw) * ic + p, 8)
    o3: i32 = (sp + 3) // tile_w
    a3: aie_vector[bf16, 8] = load_v(input + (((o3 * stride + kh) * patch_w) + ((sp + 3) - o3 * tile_w) * stride + kw) * ic + p, 8)
    o4: i32 = (sp + 4) // tile_w
    a4: aie_vector[bf16, 8] = load_v(input + (((o4 * stride + kh) * patch_w) + ((sp + 4) - o4 * tile_w) * stride + kw) * ic + p, 8)
    o5: i32 = (sp + 5) // tile_w
    a5: aie_vector[bf16, 8] = load_v(input + (((o5 * stride + kh) * patch_w) + ((sp + 5) - o5 * tile_w) * stride + kw) * ic + p, 8)
    o6: i32 = (sp + 6) // tile_w
    a6: aie_vector[bf16, 8] = load_v(input + (((o6 * stride + kh) * patch_w) + ((sp + 6) - o6 * tile_w) * stride + kw) * ic + p, 8)
    o7: i32 = (sp + 7) // tile_w
    a7: aie_vector[bf16, 8] = load_v(input + (((o7 * stride + kh) * patch_w) + ((sp + 7) - o7 * tile_w) * stride + kw) * ic + p, 8)
    a01: aie_vector[bf16, 16] = concat(a0, a1)
    a23: aie_vector[bf16, 16] = concat(a2, a3)
    a45: aie_vector[bf16, 16] = concat(a4, a5)
    a67: aie_vector[bf16, 16] = concat(a6, a7)
    a0123: aie_vector[bf16, 32] = concat(a01, a23)
    a4567: aie_vector[bf16, 32] = concat(a45, a67)
    return concat(a0123, a4567)


@aie_kernel
def chain_conv2res_bfp(
    scratch: ptr[bf16, True],
    weight: ptr[bf16, True],
    finals_base: ptr[bf16, True],
    arena_base: ptr[bf16, True],
    block: i32,
    t: i32,
    ic: i32,
    grow: i32,
    gcol: i32,
    gbound: i32,
) -> void:
    """Conv2 + BN + SiLU + residual via BFP576 2x2 gemm (ic=16, 8x8 out)."""
    event0()
    set_ctrl_reg(9, 1)
    set_ctrl_reg(1, 12)
    outp: ptr[bf16] = finals_base + t * (64 * ic)
    arena_in: ptr[bf16] = arena_base + t * (96 * ic)
    woff_b: i32 = (ic // 8) * 9 * 64           # 1152 for ic=16
    bn_w: ptr[bf16] = weight + 16 * ic * 9
    bn_b: ptr[bf16] = bn_w + 16
    kkmax: i32 = (ic // 8) * 9                  # 18

    m: i32 = 0
    while m < 8:                                # MB = 8 output rows
        n: i32 = 0
        while n < 2:
            acc00: aie_vector[f32, 64] = zeros(f32, 64)   # row m, oc-blk n
            acc01: aie_vector[f32, 64] = zeros(f32, 64)   # row m, oc-blk n+1
            acc10: aie_vector[f32, 64] = zeros(f32, 64)   # row m+1, oc-blk n
            acc11: aie_vector[f32, 64] = zeros(f32, 64)   # row m+1, oc-blk n+1
            sp0: i32 = m * 8
            sp1: i32 = sp0 + 8
            b0_off: i32 = n * woff_b
            b1_off: i32 = (n + 1) * woff_b
            k: i32 = 0
            while k < kkmax:
                icb: i32 = k // 9
                tp: i32 = k - icb * 9
                kh: i32 = tp // 3
                kw: i32 = tp - kh * 3
                va0: aie_vector[bf16, 64] = _build_a64_3x3_c2(scratch, sp0, 8, 1, 10, ic, icb, kh, kw)
                a0l: aie_vector[bf16, 32] = vector_extract(va0, 0, 32)
                a0h: aie_vector[bf16, 32] = vector_extract(va0, 32, 32)
                a0al: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0l)
                a0ah: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0h)
                a0ac: aie_vector[f32, 64] = concat(a0al, a0ah)
                a0m, a0e = v64accfloat_to_v64bfp16ebs8(a0ac)
                vb0: aie_vector[bf16, 64] = load_v(weight + b0_off, 64)
                b0_off = b0_off + 64
                b0i: aie_vector[i32, 32] = vector_cast(vb0, i32, 32)
                b0lo: aie_vector[i32, 16] = vector_extract(b0i, 0, 16)
                b0hi: aie_vector[i32, 16] = vector_extract(b0i, 16, 16)
                b0ev: aie_vector[i32, 16] = vshuffle(b0lo, b0hi, 52)
                b0od: aie_vector[i32, 16] = vshuffle(b0lo, b0hi, 53)
                b0cat: aie_vector[i32, 32] = concat(b0ev, b0od)
                vb0s: aie_vector[bf16, 64] = vector_cast(b0cat, bf16, 64)
                b0sl: aie_vector[bf16, 32] = vector_extract(vb0s, 0, 32)
                b0sh: aie_vector[bf16, 32] = vector_extract(vb0s, 32, 32)
                b0al: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b0sl)
                b0ah: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b0sh)
                b0ac: aie_vector[f32, 64] = concat(b0al, b0ah)
                b0m, b0e = v64accfloat_to_v64bfp16ebs8(b0ac)
                i00: aie_vector[i32, 64] = vector_cast(acc00, i32, 64)
                r00: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(a0m, a0e, b0m, b0e, i00, MAC_CONF)
                acc00 = vector_cast(r00, f32, 64)
                vb1: aie_vector[bf16, 64] = load_v(weight + b1_off, 64)
                b1_off = b1_off + 64
                b1i: aie_vector[i32, 32] = vector_cast(vb1, i32, 32)
                b1lo: aie_vector[i32, 16] = vector_extract(b1i, 0, 16)
                b1hi: aie_vector[i32, 16] = vector_extract(b1i, 16, 16)
                b1ev: aie_vector[i32, 16] = vshuffle(b1lo, b1hi, 52)
                b1od: aie_vector[i32, 16] = vshuffle(b1lo, b1hi, 53)
                b1cat: aie_vector[i32, 32] = concat(b1ev, b1od)
                vb1s: aie_vector[bf16, 64] = vector_cast(b1cat, bf16, 64)
                b1sl: aie_vector[bf16, 32] = vector_extract(vb1s, 0, 32)
                b1sh: aie_vector[bf16, 32] = vector_extract(vb1s, 32, 32)
                b1al: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b1sl)
                b1ah: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b1sh)
                b1ac: aie_vector[f32, 64] = concat(b1al, b1ah)
                b1m, b1e = v64accfloat_to_v64bfp16ebs8(b1ac)
                i01: aie_vector[i32, 64] = vector_cast(acc01, i32, 64)
                r01: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(a0m, a0e, b1m, b1e, i01, MAC_CONF)
                acc01 = vector_cast(r01, f32, 64)
                va1: aie_vector[bf16, 64] = _build_a64_3x3_c2(scratch, sp1, 8, 1, 10, ic, icb, kh, kw)
                a1l: aie_vector[bf16, 32] = vector_extract(va1, 0, 32)
                a1h: aie_vector[bf16, 32] = vector_extract(va1, 32, 32)
                a1al: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1l)
                a1ah: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1h)
                a1ac: aie_vector[f32, 64] = concat(a1al, a1ah)
                a1m, a1e = v64accfloat_to_v64bfp16ebs8(a1ac)
                i10: aie_vector[i32, 64] = vector_cast(acc10, i32, 64)
                r10: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(a1m, a1e, b0m, b0e, i10, MAC_CONF)
                acc10 = vector_cast(r10, f32, 64)
                i11: aie_vector[i32, 64] = vector_cast(acc11, i32, 64)
                r11: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(a1m, a1e, b1m, b1e, i11, MAC_CONF)
                acc11 = vector_cast(r11, f32, 64)
                k = k + 1
            # ── BN + SiLU + residual store (4 pixels per call) ──
            r0t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc00, 0, 32))
            r0b: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc00, 32, 32))
            r1t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc01, 0, 32))
            r1b: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc01, 32, 32))
            r2t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc10, 0, 32))
            r2b: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc10, 32, 32))
            r3t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc11, 0, 32))
            r3b: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc11, 32, 32))
            _store_bn_silu_res_4x8_rows(r0t, outp, bn_w, bn_b, sp0, block * 2 + n, arena_in, ic, grow, gcol, gbound)
            _store_bn_silu_res_4x8_rows(r0b, outp, bn_w, bn_b, sp0 + 4, block * 2 + n, arena_in, ic, grow, gcol, gbound)
            _store_bn_silu_res_4x8_rows(r1t, outp, bn_w, bn_b, sp0, block * 2 + n + 1, arena_in, ic, grow, gcol, gbound)
            _store_bn_silu_res_4x8_rows(r1b, outp, bn_w, bn_b, sp0 + 4, block * 2 + n + 1, arena_in, ic, grow, gcol, gbound)
            _store_bn_silu_res_4x8_rows(r2t, outp, bn_w, bn_b, sp1, block * 2 + n, arena_in, ic, grow, gcol, gbound)
            _store_bn_silu_res_4x8_rows(r2b, outp, bn_w, bn_b, sp1 + 4, block * 2 + n, arena_in, ic, grow, gcol, gbound)
            _store_bn_silu_res_4x8_rows(r3t, outp, bn_w, bn_b, sp1, block * 2 + n + 1, arena_in, ic, grow, gcol, gbound)
            _store_bn_silu_res_4x8_rows(r3b, outp, bn_w, bn_b, sp1 + 4, block * 2 + n + 1, arena_in, ic, grow, gcol, gbound)
            n = n + 2
        m = m + 2
    event1()


KERNEL_EXTRA_GLOBALS = {
    "load_v": load_v, "store_v": store_v, "concat": concat,
    "vector_cast": vector_cast, "vector_extract": vector_extract, "vshuffle": vshuffle,
    "zeros": zeros, "set_ctrl_reg": set_ctrl_reg,
    "v32bf16_to_v32accfloat": v32bf16_to_v32accfloat,
    "v32accfloat_to_v32bf16": v32accfloat_to_v32bf16,
    "v64accfloat_to_v64bfp16ebs8": v64accfloat_to_v64bfp16ebs8,
    "BFP576_BFP576_ACC2048_mac_conf": BFP576_BFP576_ACC2048_mac_conf,
    "aie_vector": aie_vector, "event0": event0, "event1": event1, "MAC_CONF": 780,
}
CONV2RES_BFP_HELPERS = [_build_a64_3x3_c2, _store_bn_silu_res_4x8_rows]
