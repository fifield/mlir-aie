from pythoc import ptr, f32, bf16, i16, i32, i64, u16, u32, void
from pythoc.decorators.inline import inline
from pythoc.aie import (
    set_ctrl_reg,
    fast_exp2_i32_to_f32,
    bitcast_i32_to_f32,
    aie_vector,
    load_v,
    store_v,
    broadcast,
    vector_cast,
)
from pythoc.aie import (
    vector_add,
    vector_sub,
    vector_and,
    vector_or,
    vector_xor,
    ACC2048_accfloat_add_conf,
    ACC2048_accfloat_sub_conf,
)
from pythoc.aie import (
    I512_I512_ACC1024_mul_conf,
    I512_I512_ACC1024_mac_conf,
    I512_v16_acc64_srs,
    vshuffle,
    vlt32,
    vge32,
    veqz32,
    vsel32,
)


def exp_fallback(x: f32) -> f32:
    if x < f32(-80.0):
        return f32(0.0)
    n: i32 = i32(x * f32(1.4426950408889634))
    r: f32 = x - f32(n) * f32(0.6931471805599453)
    p: f32 = f32(0.001388888888888889) + r * f32(0.0001984126984126984)
    p = f32(0.008333333333333333) + r * p
    p = f32(0.041666666666666664) + r * p
    p = f32(0.16666666666666666) + r * p
    p = f32(0.5) + r * p
    p = f32(1.0) + r * p
    p = f32(1.0) + r * p
    return p * fast_exp2_i32_to_f32(n)


def exp_v64(
    xs: ptr[u32, True],
    ys: ptr[u32, True],
    tab: ptr[u32, True],
    wk: ptr[u32, True],
    ng: i32,
) -> void:
    wk[320] = u32(985008993)
    wk[321] = u32(1007192201)
    wk[322] = u32(1026206379)
    wk[323] = u32(1042983595)
    wk[324] = u32(1056964608)
    wk[325] = u32(1065353216)
    wk[326] = u32(1065353216)
    zv: aie_vector[i32, 16] = broadcast(i32, 16, i32(0))
    g: i32 = 0
    while g < ng:
        xb: aie_vector[i32, 16] = load_v(ptr[i32](xs + g * 16), 16, 32)
        ax: aie_vector[i32, 16] = vector_and(xb, broadcast(i32, 16, i32(2147483647)))
        neg: i32 = vlt32(xb, zv, 1)
        fast: i32 = veqz32(xb) | (
            neg & vge32(broadcast(i32, 16, i32(1117782016)), ax, 0)
        )
        axf: aie_vector[i32, 16] = vsel32(zv, ax, fast)
        tb: aie_vector[i32, 16] = broadcast(i32, 16, i32(tab[115]))
        k23: aie_vector[i32, 16] = zv
        j: i32 = 0
        kk: i32 = 0
        while j < 115:
            m: i32 = vge32(axf, broadcast(i32, 16, i32(tab[j])), 0)
            if m == 0:
                break
            kk = kk + 8388608
            tb = vsel32(tb, broadcast(i32, 16, i32(tab[116 + j])), m)
            k23 = vsel32(k23, broadcast(i32, 16, kk), m)
            j = j + 1
        store_v(ptr[i32](wk + g * 16), tb, 32)
        store_v(ptr[i32](wk + 64 + g * 16), k23, 32)
        wk[256 + g] = u32(fast)
        wk[268 + g] = u32(
            neg
            & vlt32(broadcast(i32, 16, i32(1117782016)), ax, 0)
            & vge32(broadcast(i32, 16, i32(2139095040)), ax, 0)
        )
        g = g + 1
    store_v(
        ptr[f32](wk + 128),
        ACC2048_accfloat_sub_conf(
            load_v(ptr[f32](xs), 64, 32), load_v(ptr[f32](wk), 64, 32), 60
        ),
        32,
    )
    s: i32 = 0
    while s < 7:
        g = 0
        while g < ng:
            rb: aie_vector[i32, 16] = load_v(ptr[i32](wk + 128 + g * 16), 16, 32)
            pb: aie_vector[i32, 16] = broadcast(i32, 16, i32(961547521))
            if s > 0:
                pb = load_v(ptr[i32](wk + 192 + g * 16), 16, 32)
            pr: aie_vector[i32, 16] = rb
            m_sg: aie_vector[i32, 16] = vector_and(
                vector_xor(rb, pb), broadcast(i32, 16, i32(-2147483648))
            )
            m_ex: aie_vector[i32, 16] = vector_sub(
                vector_add(
                    vector_and(rb, broadcast(i32, 16, i32(2139095040))),
                    vector_and(pb, broadcast(i32, 16, i32(2139095040))),
                ),
                broadcast(i32, 16, i32(1065353216)),
            )
            m_ma: aie_vector[i32, 16] = vector_or(
                vector_and(rb, broadcast(i32, 16, i32(8388607))),
                broadcast(i32, 16, i32(8388608)),
            )
            m_mb: aie_vector[i32, 16] = vector_or(
                vector_and(pb, broadcast(i32, 16, i32(8388607))),
                broadcast(i32, 16, i32(8388608)),
            )
            m_a0: aie_vector[i32, 16] = vshuffle(m_ma, m_mb, 2)
            m_a1: aie_vector[i32, 16] = vshuffle(m_ma, m_mb, 3)
            m_b0: aie_vector[i32, 16] = vshuffle(m_mb, m_ma, 2)
            m_b1: aie_vector[i32, 16] = vshuffle(m_mb, m_ma, 3)
            m_pp: aie_vector[i64, 16] = I512_I512_ACC1024_mul_conf(
                vector_cast(m_a1, i16, 32), vector_cast(m_b1, i16, 32), 90
            )
            m_pp = I512_I512_ACC1024_mac_conf(
                vector_cast(m_a0, i16, 32), vector_cast(m_b1, i16, 32), m_pp, 1114
            )
            m_pp = I512_I512_ACC1024_mac_conf(
                vector_cast(m_a1, i16, 32), vector_cast(m_b0, i16, 32), m_pp, 90
            )
            m_pp = I512_I512_ACC1024_mac_conf(
                vector_cast(m_a0, i16, 32), vector_cast(m_b0, i16, 32), m_pp, 1114
            )
            m_s3: aie_vector[i32, 16] = I512_v16_acc64_srs(m_pp, 23, 0)
            m_s4: aie_vector[i32, 16] = I512_v16_acc64_srs(m_pp, 24, 0)
            m_nm: i32 = vge32(m_s3, broadcast(i32, 16, i32(16777216)), 0)
            m_sm: aie_vector[i32, 16] = vsel32(
                vector_sub(m_s3, broadcast(i32, 16, i32(8388608))), m_s4, m_nm
            )
            pr = vector_or(m_sg, vector_add(m_ex, m_sm))
            pr = vsel32(
                pr,
                vector_and(vector_xor(rb, pb), broadcast(i32, 16, i32(-2147483648))),
                veqz32(vector_and(rb, broadcast(i32, 16, i32(2147483647)))),
            )
            store_v(ptr[i32](wk + 192 + g * 16), pr, 32)
            g = g + 1
        store_v(
            ptr[f32](wk + 192),
            ACC2048_accfloat_add_conf(
                load_v(ptr[f32](wk + 192), 64, 32),
                broadcast(f32, 64, bitcast_i32_to_f32(wk[320 + s], f32)),
                60,
            ),
            32,
        )
        s = s + 1
    g = 0
    while g < ng:
        pf: aie_vector[i32, 16] = load_v(ptr[i32](wk + 192 + g * 16), 16, 32)
        k2: aie_vector[i32, 16] = load_v(ptr[i32](wk + 64 + g * 16), 16, 32)
        rt: aie_vector[i32, 16] = load_v(ptr[i32](wk + 128 + g * 16), 16, 32)
        e: aie_vector[i32, 16] = vector_and(pf, broadcast(i32, 16, i32(2139095040)))
        ok: i32 = vlt32(k2, e, 0) & (
            ~veqz32(vector_xor(e, broadcast(i32, 16, i32(2139095040))))
        )
        ra: aie_vector[i32, 16] = vector_and(rt, broadcast(i32, 16, i32(2147483647)))
        tiny: i32 = vlt32(ra, broadcast(i32, 16, i32(268435456)), 0) & (~veqz32(ra))
        lo80: i32 = i32(wk[268 + g])
        store_v(ptr[i32](ys + g * 16), vsel32(vector_sub(pf, k2), zv, lo80), 32)
        fb: u32 = ((u32(~(i32(wk[256 + g]) & ok)) | u32(tiny)) & u32(~lo80)) & u32(
            65535
        )
        l: u32 = u32(0)
        while fb != u32(0):
            if (fb & u32(1)) != u32(0):
                ys[u32(g * 16) + l] = bitcast_i32_to_f32(
                    exp_fallback(bitcast_i32_to_f32(xs[u32(g * 16) + l], f32)), u32
                )
            fb = fb >> u32(1)
            l = l + u32(1)
        g = g + 1
