"""BFP576 3x3 conv that reads halo'd windows directly from a padded-HWC tile.

KEYSTONE kernel for on-device fusion: the conv's input is a contiguous
PAD-padded HWC window (the format every on-device producer emits -- the rn3
chain, a prior conv), NOT host im2col patch-packed input. Per output 8x8 tile
the kernel halo-gathers the overlapping (8+2)x(8+2)xIC source window with the
same on-the-fly im2col `_build_a64_3x3` indexing the chain's conv2res already
uses. Only the INPUT delivery changes vs the host-im2col 3x3 conv.

Signature: (in_win, weight, c_out, ic, oc) where
  in_win : padded-HWC window for ONE tile, [PATCH_H, PATCH_W, IC] row-major
           (PATCH_H = PATCH_W = TILE + 2 = 10 for an 8x8 stride-1 pad-1 tile)
  weight : BFP B layout [oc_block, kk, 8(ic), 8(oc)] row-major (same as the
           chain / conv3x3_ref) -- (oc//8) blocks * (ic//8)*9 taps * 64
  c_out  : f32 [N_BLOCKS, M_BLOCKS, 8, 8] tiled accumulator (untiled on host)

Computes a 3x3 stride-1 conv over the 8x8 output tile. M_BLOCKS = 8 (one block
per output row of 8 pixels), so the per-pixel output row `o = sp//8` is constant
per block (the inline accfloat quant requires this).
"""
from __future__ import annotations

from pythoc import ptr, i32, bf16, f32, void
from pythoc.aie import (
    aie_vector, load_v, store_v, concat, vector_cast, vector_extract,
    vshuffle, zeros, set_ctrl_reg, v32bf16_to_v32accfloat,
    v64accfloat_to_v64bfp16ebs8, BFP576_BFP576_ACC2048_mac_conf,
)
from aie.iron.pythoc import aie_kernel

MAC_CONF = 780


@aie_kernel
def _build_a64_halo(input: ptr[bf16, True], sp: i32, tile_w: i32, stride: i32,
                    patch_w: i32, ic: i32, ic_blk: i32, kh: i32,
                    kw: i32) -> aie_vector[bf16, 64]:
    """Halo gather: 8 contiguous output pixels (sp..sp+7) -> v64 bf16 [pix,ic8].

    Reads the overlapping window straight from the padded-HWC image: for output
    pixel sp+j, source addr = ((o*stride+kh)*patch_w + ((sp+j)-o*tile_w)*stride
    + kw)*ic + ic_blk*8, where o = (sp+j)//tile_w is the output row. patch_w is
    the SOURCE row stride (= padded window width), so overlapping tiles just
    read overlapping source rows -- exactly the keystone source gather."""
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
def halo_conv3x3_bfp(in_win: ptr[bf16, True], weight: ptr[bf16, True],
                     c_buf: ptr[f32, True], ic: i32, oc: i32) -> void:
    """3x3 stride-1 conv over an 8x8 tile, input halo-gathered from padded HWC.

    A operand = on-the-fly im2col from the padded window (patch_w=10). B = BFP
    weights [oc_block,kk,8,8]. Output C[N_BLOCKS, M_BLOCKS=8, 8, 8] f32, MACd
    over (ic//8)*9 taps with the BFP576 hw matmul, 2x2 register-blocked."""
    set_ctrl_reg(9, 1)
    set_ctrl_reg(1, 12)
    PATCH_W: i32 = 10                 # 8x8 tile + 2 halo cols
    N_BLOCKS: i32 = oc // 8
    M_BLOCKS: i32 = 8                 # 64 pixels / 8 = 8 rows
    KKMAX: i32 = (ic // 8) * 9
    woff_b: i32 = KKMAX * 64          # per oc-block stride in weights

    # zero the tiled C accumulator
    z: aie_vector[f32, 64] = zeros(f32, 64)
    nz: i32 = 0
    while nz < N_BLOCKS * M_BLOCKS:
        store_v(c_buf + nz * 64, z)
        nz = nz + 1

    m: i32 = 0
    while m < M_BLOCKS:               # one block per output row of 8 pixels
        n: i32 = 0
        while n < N_BLOCKS:
            c00_off: i32 = (n * M_BLOCKS + m) * 64
            c01_off: i32 = ((n + 1) * M_BLOCKS + m) * 64
            c10_off: i32 = (n * M_BLOCKS + (m + 1)) * 64
            c11_off: i32 = ((n + 1) * M_BLOCKS + (m + 1)) * 64
            acc00: aie_vector[f32, 64] = load_v(c_buf + c00_off, 64)
            acc01: aie_vector[f32, 64] = load_v(c_buf + c01_off, 64)
            acc10: aie_vector[f32, 64] = load_v(c_buf + c10_off, 64)
            acc11: aie_vector[f32, 64] = load_v(c_buf + c11_off, 64)
            sp0: i32 = m * 8
            sp1: i32 = sp0 + 8
            b0_off: i32 = n * woff_b
            b1_off: i32 = (n + 1) * woff_b
            k: i32 = 0
            while k < KKMAX:
                icb: i32 = k // 9
                tp: i32 = k - icb * 9
                kh: i32 = tp // 3
                kw: i32 = tp - kh * 3
                va0: aie_vector[bf16, 64] = _build_a64_halo(in_win, sp0, 8, 1, PATCH_W, ic, icb, kh, kw)
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
                va1: aie_vector[bf16, 64] = _build_a64_halo(in_win, sp1, 8, 1, PATCH_W, ic, icb, kh, kw)
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
            store_v(c_buf + c00_off, acc00)
            store_v(c_buf + c01_off, acc01)
            store_v(c_buf + c10_off, acc10)
            store_v(c_buf + c11_off, acc11)
            n = n + 2
        m = m + 2


KERNEL_EXTRA_GLOBALS = {
    "load_v": load_v, "store_v": store_v, "concat": concat,
    "vector_cast": vector_cast, "vector_extract": vector_extract, "vshuffle": vshuffle,
    "zeros": zeros, "set_ctrl_reg": set_ctrl_reg,
    "v32bf16_to_v32accfloat": v32bf16_to_v32accfloat,
    "v64accfloat_to_v64bfp16ebs8": v64accfloat_to_v64bfp16ebs8,
    "BFP576_BFP576_ACC2048_mac_conf": BFP576_BFP576_ACC2048_mac_conf,
    "aie_vector": aie_vector, "MAC_CONF": 780,
}
HALO_CONV_HELPERS = [_build_a64_halo]
