from aie.iron.pythoc import aie_kernel
from pythoc import ptr, bf16, f32, i32, void
from pythoc.aie import (
    aie_vector,
    load_v,
    store_v,
    vector_cast,
    vshuffle,
    vector_extract,
    set_ctrl_reg,
    T16_4x8,
    T16_8x4,
    extract_v4bfloat16_broadcast_to_v32bfloat16,
    extract_v8bfloat16_broadcast_to_v32bfloat16,
    I512_I512_ACC1024_bf_mac_conf,
)


@aie_kernel
def laya_gemm_native(a: ptr[bf16, True], b: ptr[bf16, True], c: ptr[f32, True]) -> void:
    set_ctrl_reg(1, 12)
    m: i32 = 0
    while m < 4:
        half: i32 = 0
        while half < 2:
            n0: i32 = 0
            while n0 < 8:
                acc0: aie_vector[f32, 32] = load_v(
                    c + (m * 8 + n0 + 0) * 64 + half * 32, 32, 32
                )
                acc1: aie_vector[f32, 32] = load_v(
                    c + (m * 8 + n0 + 1) * 64 + half * 32, 32, 32
                )
                acc2: aie_vector[f32, 32] = load_v(
                    c + (m * 8 + n0 + 2) * 64 + half * 32, 32, 32
                )
                acc3: aie_vector[f32, 32] = load_v(
                    c + (m * 8 + n0 + 3) * 64 + half * 32, 32, 32
                )
                k: i32 = 0
                while k < 4:
                    av: aie_vector[bf16, 32] = load_v(
                        a + (m * 4 + k) * 64 + half * 32, 32, 32
                    )
                    ai: aie_vector[i32, 16] = vector_cast(av, i32, 16)
                    at: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ai, ai, T16_4x8), bf16, 32
                    )
                    ax0: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 0)
                    )
                    ix0: aie_vector[i32, 16] = vector_cast(ax0, i32, 16)
                    xx0: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix0, ix0, T16_8x4), bf16, 32
                    )
                    ax1: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 1)
                    )
                    ix1: aie_vector[i32, 16] = vector_cast(ax1, i32, 16)
                    xx1: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix1, ix1, T16_8x4), bf16, 32
                    )
                    ax2: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 2)
                    )
                    ix2: aie_vector[i32, 16] = vector_cast(ax2, i32, 16)
                    xx2: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix2, ix2, T16_8x4), bf16, 32
                    )
                    ax3: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 3)
                    )
                    ix3: aie_vector[i32, 16] = vector_cast(ax3, i32, 16)
                    xx3: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix3, ix3, T16_8x4), bf16, 32
                    )
                    ax4: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 4)
                    )
                    ix4: aie_vector[i32, 16] = vector_cast(ax4, i32, 16)
                    xx4: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix4, ix4, T16_8x4), bf16, 32
                    )
                    ax5: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 5)
                    )
                    ix5: aie_vector[i32, 16] = vector_cast(ax5, i32, 16)
                    xx5: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix5, ix5, T16_8x4), bf16, 32
                    )
                    ax6: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 6)
                    )
                    ix6: aie_vector[i32, 16] = vector_cast(ax6, i32, 16)
                    xx6: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix6, ix6, T16_8x4), bf16, 32
                    )
                    ax7: aie_vector[bf16, 32] = (
                        extract_v4bfloat16_broadcast_to_v32bfloat16(at, 7)
                    )
                    ix7: aie_vector[i32, 16] = vector_cast(ax7, i32, 16)
                    xx7: aie_vector[bf16, 32] = vector_cast(
                        vshuffle(ix7, ix7, T16_8x4), bf16, 32
                    )
                    bl0: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 0) * 64, 32, 32
                    )
                    bh0: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 0) * 64 + 32, 32, 32
                    )
                    bl1: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 1) * 64, 32, 32
                    )
                    bh1: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 1) * 64 + 32, 32, 32
                    )
                    bl2: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 2) * 64, 32, 32
                    )
                    bh2: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 2) * 64 + 32, 32, 32
                    )
                    bl3: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 3) * 64, 32, 32
                    )
                    bh3: aie_vector[bf16, 32] = load_v(
                        b + (k * 8 + n0 + 3) * 64 + 32, 32, 32
                    )
                    y0_0: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl0, 0)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx0, y0_0, acc0, 60)
                    y1_0: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl1, 0)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx0, y1_0, acc1, 60)
                    y2_0: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl2, 0)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx0, y2_0, acc2, 60)
                    y3_0: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl3, 0)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx0, y3_0, acc3, 60)
                    y0_1: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl0, 1)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx1, y0_1, acc0, 60)
                    y1_1: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl1, 1)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx1, y1_1, acc1, 60)
                    y2_1: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl2, 1)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx1, y2_1, acc2, 60)
                    y3_1: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl3, 1)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx1, y3_1, acc3, 60)
                    y0_2: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl0, 2)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx2, y0_2, acc0, 60)
                    y1_2: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl1, 2)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx2, y1_2, acc1, 60)
                    y2_2: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl2, 2)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx2, y2_2, acc2, 60)
                    y3_2: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl3, 2)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx2, y3_2, acc3, 60)
                    y0_3: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl0, 3)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx3, y0_3, acc0, 60)
                    y1_3: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl1, 3)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx3, y1_3, acc1, 60)
                    y2_3: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl2, 3)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx3, y2_3, acc2, 60)
                    y3_3: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bl3, 3)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx3, y3_3, acc3, 60)
                    y0_4: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh0, 0)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx4, y0_4, acc0, 60)
                    y1_4: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh1, 0)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx4, y1_4, acc1, 60)
                    y2_4: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh2, 0)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx4, y2_4, acc2, 60)
                    y3_4: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh3, 0)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx4, y3_4, acc3, 60)
                    y0_5: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh0, 1)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx5, y0_5, acc0, 60)
                    y1_5: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh1, 1)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx5, y1_5, acc1, 60)
                    y2_5: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh2, 1)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx5, y2_5, acc2, 60)
                    y3_5: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh3, 1)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx5, y3_5, acc3, 60)
                    y0_6: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh0, 2)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx6, y0_6, acc0, 60)
                    y1_6: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh1, 2)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx6, y1_6, acc1, 60)
                    y2_6: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh2, 2)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx6, y2_6, acc2, 60)
                    y3_6: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh3, 2)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx6, y3_6, acc3, 60)
                    y0_7: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh0, 3)
                    )
                    acc0 = I512_I512_ACC1024_bf_mac_conf(xx7, y0_7, acc0, 60)
                    y1_7: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh1, 3)
                    )
                    acc1 = I512_I512_ACC1024_bf_mac_conf(xx7, y1_7, acc1, 60)
                    y2_7: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh2, 3)
                    )
                    acc2 = I512_I512_ACC1024_bf_mac_conf(xx7, y2_7, acc2, 60)
                    y3_7: aie_vector[bf16, 32] = (
                        extract_v8bfloat16_broadcast_to_v32bfloat16(bh3, 3)
                    )
                    acc3 = I512_I512_ACC1024_bf_mac_conf(xx7, y3_7, acc3, 60)
                    k = k + 1
                store_v(c + (m * 8 + n0 + 0) * 64 + half * 32, acc0, 32)
                store_v(c + (m * 8 + n0 + 1) * 64 + half * 32, acc1, 32)
                store_v(c + (m * 8 + n0 + 2) * 64 + half * 32, acc2, 32)
                store_v(c + (m * 8 + n0 + 3) * 64 + half * 32, acc3, 32)
                n0 = n0 + 4
            half = half + 1
        m = m + 1
