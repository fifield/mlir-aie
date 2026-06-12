"""BFP576 conv1: 3x3 conv via the hardware block-float matmul (512 multipliers).

DROP-IN for chain_conv1_bf16 (same signature). Replaces the emulated 4x8x8 mmul
with the BFP576 8x8x8 hardware MAC (5.6x denser). Validated standalone vs numpy
conv1+BN+SiLU at max 0.044 (test_conv1_dropin_hw.py).

Recipe (proven):
  - ROW-ALIGNED blocking: output rows padded to 16 wide (OWP=16, 10 real + 6 pad
    cols) -> 10 rows x 2 blocks = MB=20 M-blocks; every 8-pixel block stays in ONE
    output row so the on-core im2col's per-pixel o=pixel//16 is CONSTANT per block
    (the condition the inline accfloat quant needs; OW=10 not dividing 8 was the
    odd-cols-zero root cause). 2x2 register-blocked m/n step-2 loops.
  - A (activations): on-the-fly 3x3 im2col, tile_w=16, patch_w=12 (chain's input
    width) -> real pixels correct, pad-pixel reads land in the last input rows
    (arena must have >=2 safety rows beyond the 12 patch rows).
  - B (weights): chain layout [oc_block, kk, 8(ic), 8(oc)] row-major == the BFP
    vshuffle B layout; oc-block stride = woff_b = (ic//8)*9*64. No repacking.
  - reference inline bf16->bfp quant (accfloat + vshuffle), zero-acc, MAC all taps.
  - on-core BN/SiLU + bf16 scratch[block*1600 + pos*16 + oc], per-block stores with
    spatial_out guards (second block of each row writes only its 2 real cols 8,9).
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

MAC_CONF = 780


@aie_kernel
def _build_a64_3x3(input: ptr[bf16, True], sp: i32, tile_w: i32, stride: i32,
                   patch_w: i32, ic: i32, ic_blk: i32, kh: i32,
                   kw: i32) -> aie_vector[bf16, 64]:
    """Load 8 contiguous (sp+0..sp+7) 8-wide activation patches -> v64 bf16.
    [pixel(row), ic(col)] row-major = the A operand (block r = row r = pixel r)."""
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


from kernels.rep_elan_bf16_pythoc import _store_bn_silu_4x8_rows  # noqa: E402
from pythoc.aie.profiling import event0, event1  # noqa: E402


@aie_kernel
def chain_conv1_bfp(
    arena_base: ptr[bf16, True],
    weight: ptr[bf16, True],
    scratch: ptr[bf16, True],
    block: i32,
    t: i32,
    ic: i32,
) -> void:
    """Conv1 (3x3, ic->16) via BFP576 row-aligned 2x2 gemm + on-core BN/SiLU."""
    event0()
    set_ctrl_reg(9, 1)
    set_ctrl_reg(1, 12)
    arena_in: ptr[bf16] = arena_base + t * (96 * ic)
    scr: ptr[bf16] = scratch + block * 1600
    woff_b: i32 = (ic // 8) * 9 * 64          # oc-block stride (B_N_STRIDE)
    bn_w: ptr[bf16] = weight + 16 * ic * 9
    bn_b: ptr[bf16] = bn_w + 16
    kkmax: i32 = (ic // 8) * 9

    m: i32 = 0
    while m < 20:                              # MB = 10 rows x 2 blocks
        n: i32 = 0
        while n < 2:
            acc00: aie_vector[f32, 64] = zeros(f32, 64)
            acc01: aie_vector[f32, 64] = zeros(f32, 64)
            acc10: aie_vector[f32, 64] = zeros(f32, 64)
            acc11: aie_vector[f32, 64] = zeros(f32, 64)
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
                va0: aie_vector[bf16, 64] = _build_a64_3x3(arena_in, sp0, 16, 1, 12, ic, icb, kh, kw)
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
                va1: aie_vector[bf16, 64] = _build_a64_3x3(arena_in, sp1, 16, 1, 12, ic, icb, kh, kw)
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
            # ── on-core BN/SiLU + store; rr = output row, first/second block ──
            rr: i32 = m // 2
            a00t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc00, 0, 32))
            a00b: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc00, 32, 32))
            a01t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc01, 0, 32))
            a01b: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc01, 32, 32))
            a10t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc10, 0, 32))
            a11t: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(vector_extract(acc11, 0, 32))
            _store_bn_silu_4x8_rows(a00t, scr, bn_w, bn_b, rr * 10, 100, 16, n)
            _store_bn_silu_4x8_rows(a00b, scr, bn_w, bn_b, rr * 10 + 4, 100, 16, n)
            _store_bn_silu_4x8_rows(a01t, scr, bn_w, bn_b, rr * 10, 100, 16, n + 1)
            _store_bn_silu_4x8_rows(a01b, scr, bn_w, bn_b, rr * 10 + 4, 100, 16, n + 1)
            _store_bn_silu_4x8_rows(a10t, scr, bn_w, bn_b, rr * 10 + 8, rr * 10 + 10, 16, n)
            _store_bn_silu_4x8_rows(a11t, scr, bn_w, bn_b, rr * 10 + 8, rr * 10 + 10, 16, n + 1)
            n = n + 2
        m = m + 2
    event1()


# Extra-globals + helpers for PythocKernel wiring (mirror chain_conv1_bf16 setup).
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
CONV1_BFP_HELPERS = [_build_a64_3x3, _store_bn_silu_4x8_rows]
