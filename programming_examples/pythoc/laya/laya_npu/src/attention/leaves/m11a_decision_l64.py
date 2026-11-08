from pythoc import ptr, f32, bf16, i32, u16, u32, void
from pythoc.decorators.inline import inline
from pythoc.aie import (
    set_ctrl_reg,
    fast_exp2_i32_to_f32,
    bitcast_i32_to_f32,
    aie_vector,
    load_v,
    store_v,
    broadcast,
    zeros,
    concat,
    ACC2048_accfloat_add_conf,
    vshuffle,
    vector_cast,
)
from pythoc.aie.mmul import mmul_bf16
from pythoc.aie import (
    aie_vector,
    load_v,
    store_v,
    broadcast,
    vector_cast,
    vector_extract,
    concat,
    vshuffle,
)
from pythoc.aie import (
    vector_add,
    vector_sub,
    vector_and,
    vector_or,
    ACC2048_accfloat_sub_conf,
    ACC2048_accfloat_add_conf,
)
from pythoc.aie import (
    I512_I512_ACC1024_mul_conf,
    I512_I512_ACC1024_mac_conf,
    I512_I512_ACC1024_addmac_conf,
    I512_v16_acc64_srs,
    acc64_v16_I512_ups,
)
from pythoc.aie import vlt32, vge32, veqz32, vsel32, v16bf16_to_v16i32


@inline
def ieee_add_bits(a: u32, b: u32) -> u32:
    aa: u32 = a & u32(2147483647)
    bb: u32 = b & u32(2147483647)
    if aa > u32(2139095040):
        return a | u32(4194304)
    if bb > u32(2139095040):
        return b | u32(4194304)
    if aa == u32(2139095040):
        if bb == aa and ((a ^ b) & u32(2147483648)) != u32(0):
            return u32(2143289344)
        return a
    if bb == u32(2139095040):
        return b
    if aa == u32(0):
        if bb == u32(0):
            return a & b
        return b
    if bb == u32(0):
        return a
    if bb > aa:
        tmp: u32 = a
        a = b
        b = tmp
    ae: i32 = i32((a >> 23) & u32(255))
    be: i32 = i32((b >> 23) & u32(255))
    am: u32 = a & u32(8388607)
    bm: u32 = b & u32(8388607)
    if ae == 0:
        ae = 1
        while am < u32(8388608):
            am = am << 1
            ae = ae - 1
    if be == 0:
        be = 1
        while bm < u32(8388608):
            bm = bm << 1
            be = be - 1
    am = (am | u32(8388608)) << 3
    bm = (bm | u32(8388608)) << 3
    shift: i32 = ae - be
    if shift > 0:
        if shift < 32:
            sticky: u32 = u32((bm & ((u32(1) << shift) - u32(1))) != u32(0))
            bm = (bm >> shift) | sticky
        else:
            bm = u32(1)
    sign: u32 = a & u32(2147483648)
    if ((a ^ b) & u32(2147483648)) != u32(0):
        am = am - bm
        if am == u32(0):
            return u32(0)
        while am < u32(67108864):
            am = am << 1
            ae = ae - 1
    else:
        am = am + bm
        if (am & u32(134217728)) != u32(0):
            am = (am >> 1) | (am & u32(1))
            ae = ae + 1
    if ae >= 255:
        return sign | u32(2139095040)
    if ae <= 0:
        shift = 1 - ae
        if shift < 32:
            sticky = u32((am & ((u32(1) << shift) - u32(1))) != u32(0))
            am = (am >> shift) | sticky
        else:
            am = u32(1)
        ae = 0
    low: u32 = am & u32(7)
    result: u32 = ((am >> 3) & u32(8388607)) | (u32(ae) << 23) | sign
    if low > u32(4) or (low == u32(4) and (result & u32(1)) != u32(0)):
        result = result + u32(1)
    return result


@inline
def fast_add_bits(a: u32, b: u32) -> u32:
    ea: u32 = (a >> u32(23)) & u32(255)
    eb: u32 = (b >> u32(23)) & u32(255)
    if ea - u32(1) < u32(254) and eb - u32(1) < u32(254):
        x: u32 = a
        y: u32 = b
        ex: u32 = ea
        ey: u32 = eb
        if (b & u32(2147483647)) > (a & u32(2147483647)):
            x = b
            y = a
            ex = eb
            ey = ea
        xm: u32 = ((x & u32(8388607)) | u32(8388608)) << u32(3)
        ym: u32 = ((y & u32(8388607)) | u32(8388608)) << u32(3)
        d: u32 = ex - ey
        if d != u32(0):
            if d < u32(27):
                st: u32 = u32((ym & ((u32(1) << d) - u32(1))) != u32(0))
                ym = (ym >> d) | st
            else:
                ym = u32(1)
        sg: u32 = x & u32(2147483648)
        e: i32 = i32(ex)
        if ((x ^ y) & u32(2147483648)) != u32(0):
            xm = xm - ym
            if xm == u32(0):
                return u32(0)
            while xm < u32(67108864):
                xm = xm << u32(1)
                e = e - 1
        else:
            xm = xm + ym
            if (xm & u32(134217728)) != u32(0):
                xm = (xm >> u32(1)) | (xm & u32(1))
                e = e + 1
        if e >= 255:
            return sg | u32(2139095040)
        if e <= 0:
            return ieee_add_bits(a, b)
        lo: u32 = xm & u32(7)
        rs: u32 = ((xm >> u32(3)) & u32(8388607)) | (u32(e) << u32(23)) | sg
        if lo > u32(4) or (lo == u32(4) and (rs & u32(1)) != u32(0)):
            rs = rs + u32(1)
        return rs
    return ieee_add_bits(a, b)


@inline
def attention_exp(x: f32) -> f32:
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


def m6_merge_stats(state: ptr[f32, True], incoming: ptr[f32, True]) -> void:
    row: i32 = 0
    while row < 16:
        az: f32 = state[row * 2 + 1]
        bz: f32 = incoming[row * 2 + 1]
        if bz > f32(0.0):
            if az == f32(0.0):
                state[row * 2] = incoming[row * 2]
                state[row * 2 + 1] = bz
            else:
                am: f32 = state[row * 2]
                bm: f32 = incoming[row * 2]
                maximum: f32 = am
                if bm > maximum:
                    maximum = bm
                state[row * 2 + 1] = az * attention_exp(
                    am - maximum
                ) + bz * attention_exp(bm - maximum)
                state[row * 2] = maximum
        row = row + 1


def m6_finish_stats(state: ptr[f32, True], output: ptr[f32, True]) -> void:
    i: i32 = 0
    while i < 32:
        output[i] = state[i]
        i = i + 1


def m6_finish_output(state: ptr[f32, True], output: ptr[bf16, True]) -> void:
    set_ctrl_reg(1, 12)
    i: i32 = 0
    while i < 1024:
        output[i] = bf16(state[i])
        i = i + 1


def s2_initialize(state: ptr[f32, True], phase: i32) -> void:
    n: i32 = 1024
    if phase == 1:
        n = 32
    i: i32 = 0
    while i < n:
        state[i] = f32(0.0)
        i = i + 1

    if phase == 1:
        i = 0
        while i < 16:
            state[i * 2] = f32(-3.402823466e38)
            i = i + 1


@inline
def s2_fold(state: ptr[f32, True], incoming: ptr[f32, True]) -> void:
    offset: i32 = 0
    while offset < 1024:
        sv: aie_vector[f32, 64] = load_v(state + offset, 64, 32)
        sv = ACC2048_accfloat_add_conf(sv, load_v(incoming + offset, 64, 32), 60)
        store_v(state + offset, sv, 32)
        offset = offset + 64


def s2_header(obj: ptr[u16, True], unit: ptr[f32, True], loops: ptr[i32, True]) -> void:
    partials: ptr[f32] = unit
    hdr: ptr[u16] = ptr[u16](unit + 128)
    loops[0] = 4
    i: i32 = 0
    while i < 96:
        hdr[i] = obj[i]
        i = i + 1


def s2_begin(
    qsrc: ptr[u16, True], unit: ptr[f32, True], rnd: i32, col: i32, role: i32
) -> void:
    partials: ptr[f32] = unit
    hdr: ptr[u16] = ptr[u16](unit + 128)
    block: i32 = (rnd * 32 + 4 * col) & 3
    hdr[0] = u16((block + role) * 16)


def s2_pass1(
    chunk_obj: ptr[u16, True],
    qsrc: ptr[u16, True],
    unit: ptr[f32, True],
    sall: ptr[f32, True],
    em: ptr[f32, True],
    i: i32,
    mka: ptr[i32, True],
    qx: ptr[u32, True],
) -> void:
    set_ctrl_reg(1, 12)
    partials: ptr[f32] = unit
    hdr: ptr[u16] = ptr[u16](unit + 128)
    stage: i32 = i
    chunk: i32 = 0
    qb: i32 = i32(hdr[0])
    kb: i32 = (stage * 1 + chunk) * 16
    local: i32 = i32(hdr[2])
    mk: ptr[bf16] = ptr[bf16](hdr + 32 + kb)
    kd: ptr[bf16] = ptr[bf16](chunk_obj)
    eall: ptr[f32] = em
    m1: ptr[f32] = em + 1024
    qd: ptr[bf16] = ptr[bf16](qsrc)
    state: ptr[f32] = partials + stage * 32
    if chunk == 0:
        s2_initialize(state, 1)
    kn: i32 = 0
    ek: i32 = 0
    while ek < 16:
        mka[ek] = i32(mk[ek])
        if mka[ek] != 0 and (
            local == 0 or (qb + 15 - kb - ek >= -64 and qb - kb - ek <= 64)
        ):
            kn = 16
        ek = ek + 1
    qt: ptr[bf16] = ptr[bf16](qx)
    wk: ptr[i32] = ptr[i32](qx + 512)
    if i == 0:
        tr: i32 = 0
        while tr < 1024:
            qt[(tr & 63) * 16 + (tr >> 6)] = qd[tr]
            tr = tr + 1
    if kn > 0:
        gi: i32 = 0
        while gi < kn:
            k0: i32 = gi * 1
            acc: aie_vector[i32, 16] = broadcast(i32, 16, i32(0))
            dd: i32 = 0
            while dd < 64:
                qv: aie_vector[bf16, 16] = load_v(qt + dd * 16, 16, 32)
                kv: aie_vector[bf16, 32] = broadcast(bf16, 32, kd[k0 * 64 + dd])
                store_v(ptr[f32](qx + 528), mmul_bf16(concat(qv, qv), kv), 32)
                pb: aie_vector[i32, 16] = load_v(ptr[i32](qx + 528), 16, 32)
                v_aa: aie_vector[i32, 16] = vector_and(
                    acc, broadcast(i32, 16, i32(2147483647))
                )
                v_bb: aie_vector[i32, 16] = vector_and(
                    pb, broadcast(i32, 16, i32(2147483647))
                )
                v_sw: i32 = vlt32(v_aa, v_bb, 0)
                v_x: aie_vector[i32, 16] = vsel32(acc, pb, v_sw)
                v_xx: aie_vector[i32, 16] = vsel32(v_aa, v_bb, v_sw)
                v_yy: aie_vector[i32, 16] = vsel32(v_bb, v_aa, v_sw)
                v_sb: i32 = vlt32(acc, broadcast(i32, 16, i32(0)), 1) ^ vlt32(
                    pb, broadcast(i32, 16, i32(0)), 1
                )
                v_ef: aie_vector[i32, 16] = vector_and(
                    v_xx, broadcast(i32, 16, i32(2139095040))
                )
                v_gf: aie_vector[i32, 16] = vector_and(
                    v_yy, broadcast(i32, 16, i32(2139095040))
                )
                v_em: aie_vector[i32, 16] = vector_sub(
                    vsel32(v_ef, broadcast(i32, 16, i32(8388608)), veqz32(v_ef)),
                    broadcast(i32, 16, i32(8388608)),
                )
                v_gm: aie_vector[i32, 16] = vector_sub(
                    vsel32(v_gf, broadcast(i32, 16, i32(8388608)), veqz32(v_gf)),
                    broadcast(i32, 16, i32(8388608)),
                )
                v_X: aie_vector[i32, 16] = vector_sub(v_xx, v_em)
                v_Y: aie_vector[i32, 16] = vector_sub(v_yy, v_gm)
                v_dF: aie_vector[i32, 16] = vector_sub(v_em, v_gm)
                v_DF: aie_vector[i32, 16] = vsel32(
                    v_dF,
                    broadcast(i32, 16, i32(218103808)),
                    vlt32(broadcast(i32, 16, i32(218103808)), v_dF, 0),
                )
                v_hw: aie_vector[i32, 16] = vshuffle(
                    vector_sub(broadcast(i32, 16, i32(1283457024)), v_DF),
                    broadcast(i32, 16, i32(0)),
                    3,
                )
                v_P: aie_vector[i32, 16] = v16bf16_to_v16i32(
                    vector_extract(vector_cast(v_hw, bf16, 32), 0, 16), 0
                )
                v_Ys: aie_vector[i32, 16] = vsel32(
                    v_Y, vector_sub(broadcast(i32, 16, i32(0)), v_Y), v_sb
                )
                v_yh: aie_vector[i32, 16] = vshuffle(
                    v_Ys, broadcast(i32, 16, i32(0)), 3
                )
                v_yl: aie_vector[i32, 16] = vshuffle(
                    v_Ys, broadcast(i32, 16, i32(0)), 2
                )
                v_ph: aie_vector[i32, 16] = vshuffle(v_P, broadcast(i32, 16, i32(0)), 3)
                v_pl: aie_vector[i32, 16] = vshuffle(v_P, broadcast(i32, 16, i32(0)), 2)
                v_T: aie_vector[i64, 16] = I512_I512_ACC1024_mul_conf(
                    vector_cast(v_yh, i16, 32), vector_cast(v_ph, i16, 32), 602
                )
                v_T = I512_I512_ACC1024_mac_conf(
                    vector_cast(v_yl, i16, 32), vector_cast(v_ph, i16, 32), v_T, 1114
                )
                v_T = I512_I512_ACC1024_mac_conf(
                    vector_cast(v_yh, i16, 32), vector_cast(v_pl, i16, 32), v_T, 602
                )
                v_T = I512_I512_ACC1024_addmac_conf(
                    vector_cast(v_yl, i16, 32),
                    vector_cast(v_pl, i16, 32),
                    v_T,
                    acc64_v16_I512_ups(v_X, 26, 0),
                    1114,
                )
                v_s5: aie_vector[i32, 16] = I512_v16_acc64_srs(v_T, 25, 1)
                v_s6: aie_vector[i32, 16] = I512_v16_acc64_srs(v_T, 26, 1)
                v_s7: aie_vector[i32, 16] = I512_v16_acc64_srs(v_T, 27, 1)
                v_c7: i32 = vge32(v_s6, broadcast(i32, 16, i32(16777216)), 0)
                v_c6: i32 = vge32(v_s5, broadcast(i32, 16, i32(16777216)), 0)
                v_far: i32 = v_c6 | (
                    vge32(v_s5, broadcast(i32, 16, i32(8388608)), 0)
                    & (veqz32(v_em) ^ 65535)
                )
                v_fb: aie_vector[i32, 16] = vsel32(
                    vsel32(
                        vector_sub(
                            vector_add(v_em, v_s5), broadcast(i32, 16, i32(8388608))
                        ),
                        vector_add(v_em, v_s6),
                        v_c6,
                    ),
                    vector_add(
                        vector_add(v_em, v_s7), broadcast(i32, 16, i32(8388608))
                    ),
                    v_c7,
                )
                v_fb = vsel32(
                    v_fb,
                    broadcast(i32, 16, i32(2139095040)),
                    vge32(v_fb, broadcast(i32, 16, i32(2139095040)), 0),
                )
                v_fz: aie_vector[f32, 16] = vector_cast(
                    broadcast(i32, 16, i32(1258291200)), f32, 16
                )
                v_fm: aie_vector[f32, 64] = ACC2048_accfloat_sub_conf(
                    concat(
                        concat(
                            vector_cast(
                                vector_add(v_s5, broadcast(i32, 16, i32(1258291200))),
                                f32,
                                16,
                            ),
                            v_fz,
                        ),
                        concat(v_fz, v_fz),
                    ),
                    concat(concat(v_fz, v_fz), concat(v_fz, v_fz)),
                    60,
                )
                v_F: aie_vector[i32, 16] = vector_add(
                    vector_cast(vector_extract(v_fm, 0, 16), i32, 16), v_em
                )
                v_nb: aie_vector[i32, 16] = vector_sub(
                    v_F, broadcast(i32, 16, i32(1258291200))
                )
                v_gm2: aie_vector[f32, 64] = ACC2048_accfloat_add_conf(
                    concat(
                        concat(
                            vector_cast(
                                vector_sub(v_F, broadcast(i32, 16, i32(8388608))),
                                f32,
                                16,
                            ),
                            v_fz,
                        ),
                        concat(v_fz, v_fz),
                    ),
                    concat(concat(v_fz, v_fz), concat(v_fz, v_fz)),
                    60,
                )
                v_mg: aie_vector[i32, 16] = vsel32(
                    v_nb,
                    vector_sub(
                        vector_cast(vector_extract(v_gm2, 0, 16), i32, 16),
                        broadcast(i32, 16, i32(1258291200)),
                    ),
                    vlt32(v_nb, broadcast(i32, 16, i32(8388608)), 1),
                )
                v_mg = vsel32(vsel32(v_mg, v_s6, veqz32(v_em)), v_fb, v_far)
                v_rz: i32 = veqz32(v_s5) & (v_far ^ 65535)
                v_g: aie_vector[i32, 16] = vector_or(
                    vsel32(v_mg, broadcast(i32, 16, i32(0)), v_rz),
                    vector_and(
                        vsel32(v_x, vector_and(acc, pb), v_rz),
                        broadcast(i32, 16, i32(-2147483648)),
                    ),
                )
                if vge32(v_xx, broadcast(i32, 16, i32(2139095040)), 0) != 0:
                    v_an: i32 = vlt32(broadcast(i32, 16, i32(2139095040)), v_aa, 0)
                    v_bn: i32 = vlt32(broadcast(i32, 16, i32(2139095040)), v_bb, 0)
                    v_ii: i32 = (
                        veqz32(vector_sub(v_aa, broadcast(i32, 16, i32(2139095040))))
                        & veqz32(vector_sub(v_bb, broadcast(i32, 16, i32(2139095040))))
                        & v_sb
                    )
                    v_g = vsel32(
                        v_g, v_x, vge32(v_xx, broadcast(i32, 16, i32(2139095040)), 0)
                    )
                    v_g = vsel32(v_g, broadcast(i32, 16, i32(2143289344)), v_ii)
                    v_g = vsel32(
                        v_g, vector_or(pb, broadcast(i32, 16, i32(4194304))), v_bn
                    )
                    v_g = vsel32(
                        v_g, vector_or(acc, broadcast(i32, 16, i32(4194304))), v_an
                    )
                acc = v_g
                dd = dd + 1
            store_v(ptr[i32](qx + 576 + gi * 16), acc, 32)
            gi = gi + 1
    row: i32 = 0
    while row < 16:
        scores: ptr[f32] = sall + row * 64 + kb
        exps: ptr[f32] = eall + row * 64 + kb
        maximum: f32 = f32(-3.402823466e38)
        key: i32 = 0
        while key < kn:
            delta: i32 = qb + row - kb - key
            allowed: i32 = mka[key]
            if local != 0:
                if delta < -64 or delta > 64:
                    allowed = 0
            score: f32 = f32(-3.402823466e38)
            if allowed != 0:
                score = f32(0.0)
                score_bits: u32 = qx[576 + key * 16 + row]
                score = bitcast_i32_to_f32(i32(score_bits), f32) * f32(0.125)
            scores[key] = score
            if score > maximum:
                maximum = score
            key = key + 1
        previous: f32 = state[row * 2]
        denom: f32 = state[row * 2 + 1]
        if previous > maximum:
            maximum = previous
        scale: f32 = f32(0.0)
        if denom > f32(0.0):
            scale = attention_exp(previous - maximum)
        denom = denom * scale
        key = 0
        while key < kn:
            delta = qb + row - kb - key
            if mka[key] != 0 and (local == 0 or (delta >= -64 and delta <= 64)):
                ex: f32 = attention_exp(scores[key] - maximum)
                exps[key] = ex
                denom = denom + ex
            key = key + 1
        state[row * 2] = maximum
        state[row * 2 + 1] = denom
        m1[row * 4 + stage * 1 + chunk] = maximum
        row = row + 1


def s2_merge(unit: ptr[f32, True]) -> void:
    partials: ptr[f32] = unit
    m6_merge_stats(partials + 64, partials + 96)
    m6_merge_stats(partials + 32, partials + 64)
    m6_merge_stats(partials + 0, partials + 32)


def s2_pass2(
    chunk_obj: ptr[u16, True],
    unit: ptr[f32, True],
    acc: ptr[f32, True],
    sall: ptr[f32, True],
    em: ptr[f32, True],
    i: i32,
    mka: ptr[i32, True],
) -> void:
    set_ctrl_reg(1, 12)
    partials: ptr[f32] = unit
    hdr: ptr[u16] = ptr[u16](unit + 128)
    stage: i32 = 3 - i
    chunk: i32 = 0
    qb: i32 = i32(hdr[0])
    kb: i32 = (stage * 1 + chunk) * 16
    local: i32 = i32(hdr[2])
    mk: ptr[bf16] = ptr[bf16](hdr + 32 + kb)
    vd: ptr[bf16] = ptr[bf16](chunk_obj)
    stats: ptr[f32] = partials
    eall: ptr[f32] = em
    m1: ptr[f32] = em + 1024
    A: ptr[f32] = acc + 1024
    B: ptr[f32] = acc + 0
    state: ptr[f32] = B
    if stage == 3:
        state = A
    if stage == 2:
        state = B
    if stage == 1:
        state = A
    if chunk == 0:
        s2_initialize(state, 2)
    kn: i32 = 0
    ek: i32 = 0
    while ek < 16:
        mka[ek] = i32(mk[ek])
        if mka[ek] != 0 and (
            local == 0 or (qb + 15 - kb - ek >= -64 and qb - kb - ek <= 64)
        ):
            kn = 16
        ek = ek + 1
    m1b: ptr[u32] = ptr[u32](m1)
    mb: ptr[u32] = ptr[u32](stats)
    row: i32 = 0
    while row < kn:
        scores: ptr[f32] = sall + row * 64 + kb
        exps: ptr[f32] = eall + row * 64 + kb
        same: i32 = 0
        if m1b[row * 4 + stage * 1 + chunk] == mb[row * 2]:
            same = 1
        denom: f32 = stats[row * 2 + 1]
        dpos: i32 = 0
        if denom > f32(0.0):
            dpos = 1
        key: i32 = 0
        while key < 16:
            delta: i32 = qb + row - kb - key
            if (
                dpos != 0
                and mka[key] != 0
                and (local == 0 or (delta >= -64 and delta <= 64))
            ):
                ex: f32 = f32(0.0)
                if same != 0:
                    ex = exps[key]
                else:
                    ex = attention_exp(scores[key] - stats[row * 2])
                probability: bf16 = bf16(ex / denom)
                pb: aie_vector[bf16, 32] = broadcast(bf16, 32, probability)
                lo: aie_vector[f32, 32] = mmul_bf16(pb, load_v(vd + key * 64, 32, 32))
                hi: aie_vector[f32, 32] = mmul_bf16(
                    pb, load_v(vd + key * 64 + 32, 32, 32)
                )
                sv: aie_vector[f32, 64] = load_v(state + row * 64, 64, 32)
                sv = ACC2048_accfloat_add_conf(sv, concat(lo, hi), 60)
                store_v(state + row * 64, sv, 32)
            key = key + 1
        row = row + 1
    if chunk == 0 and stage < 3:
        dst: ptr[f32] = B
        src: ptr[f32] = A
        if stage == 1:
            dst = A
            src = B
        if stage == 2:
            dst = B
            src = A
        s2_fold(dst, src)
