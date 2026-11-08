from pythoc import ptr, f32, bf16, i32, u16, u32, void
from pythoc.decorators.inline import inline
from pythoc.aie import (
    set_ctrl_reg,
    fast_exp2_i32_to_f32,
    aie_vector,
    load_v,
    store_v,
    broadcast,
    zeros,
    concat,
    ACC2048_accfloat_add_conf,
    vshuffle,
    vector_cast,
    bitcast_i32_to_f32,
)
from pythoc.aie.mmul import mmul_bf16

from pythoc import extern
from pythoc.aie import ACC2048_accfloat_sub_conf


@extern
def exp_v64(
    xs: ptr[u32, True],
    ys: ptr[u32, True],
    tab: ptr[u32, True],
    wk: ptr[u32, True],
    ng: i32,
) -> void:
    pass


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


def exp_e1c(x: f32, tab: ptr[u32]) -> f32:
    xb: u32 = bitcast_i32_to_f32(x, u32)
    ax: u32 = xb & u32(2147483647)
    n: i32 = 0
    t: f32 = f32(0.0)
    k: u32 = u32(0)
    fast: i32 = 0
    if xb == u32(0) or ((xb & u32(2147483648)) != u32(0) and ax <= u32(1117782016)):
        fast = 1
        if k + u32(64) <= u32(115):
            if tab[k + u32(63)] <= ax:
                k = k + u32(64)
        if k + u32(32) <= u32(115):
            if tab[k + u32(31)] <= ax:
                k = k + u32(32)
        if k + u32(16) <= u32(115):
            if tab[k + u32(15)] <= ax:
                k = k + u32(16)
        if k + u32(8) <= u32(115):
            if tab[k + u32(7)] <= ax:
                k = k + u32(8)
        if k + u32(4) <= u32(115):
            if tab[k + u32(3)] <= ax:
                k = k + u32(4)
        if k + u32(2) <= u32(115):
            if tab[k + u32(1)] <= ax:
                k = k + u32(2)
        if k + u32(1) <= u32(115):
            if tab[k + u32(0)] <= ax:
                k = k + u32(1)
        n = i32(0) - i32(k)
        t = bitcast_i32_to_f32(tab[u32(115) + k], f32)
    else:
        if x < f32(-80.0):
            return f32(0.0)
        n = i32(x * f32(1.4426950408889634))
        t = f32(n) * f32(0.6931471805599453)
    r: f32 = x - t
    p: f32 = f32(0.001388888888888889) + r * f32(0.0001984126984126984)
    p = f32(0.008333333333333333) + r * p
    p = f32(0.041666666666666664) + r * p
    p = f32(0.16666666666666666) + r * p
    p = f32(0.5) + r * p
    p = f32(1.0) + r * p
    p = f32(1.0) + r * p
    if fast != 0:
        pb: u32 = bitcast_i32_to_f32(p, u32)
        e: u32 = pb & u32(2139095040)
        sh: u32 = k << u32(23)
        if e > sh and e != u32(2139095040):
            return bitcast_i32_to_f32(pb - sh, f32)
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
    qsrc: ptr[u16, True],
    qk: ptr[u16, True],
    unit: ptr[f32, True],
    rnd: i32,
    col: i32,
    role: i32,
) -> void:
    partials: ptr[f32] = unit
    hdr: ptr[u16] = ptr[u16](unit + 128)
    block: i32 = (rnd * 32 + 4 * col) & 3
    hdr[0] = u16((block + role) * 16)
    qt: ptr[u16] = qk
    if rnd >= 0:
        r: i32 = 0
        while r < 16:
            dd: i32 = 0
            while dd < 64:
                qt[dd * 16 + r] = qsrc[(r & 15) * 64 + dd]
                dd = dd + 1
            r = r + 1


@extern
def m12a_vscale(p: ptr[u32, True], ng: i32) -> void:
    pass


@extern
def m12a_vdiv(
    src: ptr[u32, True],
    dst: ptr[u32, True],
    dstb: ptr[u16, True],
    db: u32,
    ng: i32,
    mk: ptr[i32, True],
) -> void:
    pass


def s2_pass1(
    chunk_obj: ptr[u16, True],
    qk: ptr[u16, True],
    unit: ptr[f32, True],
    sall: ptr[f32, True],
    em: ptr[f32, True],
    i: i32,
    mka: ptr[i32, True],
    tab: ptr[u32, True],
    vx: ptr[u32, True],
    vw: ptr[u32, True],
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
    qt: ptr[u16] = qk
    smat: ptr[f32] = ptr[f32](qk + 1024)
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
    qtb: ptr[bf16] = ptr[bf16](qt)
    kk: i32 = 0
    while kk < kn:
        if (
            mka[kk + 0] != 0 or mka[kk + 1] != 0 or mka[kk + 2] != 0 or mka[kk + 3] != 0
        ) and (local == 0 or (qb + 15 - kb - kk >= -64 and qb - kb - kk - 3 <= 64)):
            acc: aie_vector[f32, 64] = zeros(f32, 64)
            e: i32 = 0
            while e < 64:
                qh: aie_vector[bf16, 16] = load_v(qtb + e * 16, 16, 32)
                qv: aie_vector[bf16, 32] = concat(qh, qh)
                b0: aie_vector[bf16, 32] = vector_cast(
                    vshuffle(
                        vector_cast(broadcast(bf16, 32, kd[kk * 64 + e]), i32, 16),
                        vector_cast(broadcast(bf16, 32, kd[kk * 64 + 64 + e]), i32, 16),
                        10,
                    ),
                    bf16,
                    32,
                )
                b1: aie_vector[bf16, 32] = vector_cast(
                    vshuffle(
                        vector_cast(
                            broadcast(bf16, 32, kd[kk * 64 + 128 + e]), i32, 16
                        ),
                        vector_cast(
                            broadcast(bf16, 32, kd[kk * 64 + 192 + e]), i32, 16
                        ),
                        10,
                    ),
                    bf16,
                    32,
                )
                acc = ACC2048_accfloat_add_conf(
                    acc, concat(mmul_bf16(qv, b0), mmul_bf16(qv, b1)), 60
                )
                e = e + 1
            store_v(smat + kk * 16, acc, 32)
            m12a_vscale(ptr[u32](smat + kk * 16), 4)
        kk = kk + 4
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
                score = smat[key * 16 + row]
            scores[key] = score
            sb: i32 = bitcast_i32_to_f32(score, i32)
            mb: i32 = bitcast_i32_to_f32(maximum, i32)
            sa: u32 = u32(sb) & u32(2147483647)
            ma: u32 = u32(mb) & u32(2147483647)
            if (
                i32(sa <= u32(2139095040))
                & i32(ma <= u32(2139095040))
                & i32((sa | ma) != u32(0))
                & i32(
                    (sb ^ ((sb >> 31) & i32(2147483647)))
                    > (mb ^ ((mb >> 31) & i32(2147483647)))
                )
            ) != i32(0):
                maximum = score
            key = key + 1
        previous: f32 = state[row * 2]
        denom: f32 = state[row * 2 + 1]
        if previous > maximum:
            maximum = previous
        scale: f32 = f32(0.0)
        if denom > f32(0.0):
            store_v(ptr[i32](vx), broadcast(i32, 16, i32(0)), 32)
            vx[0] = bitcast_i32_to_f32(previous - maximum, u32)
            exp_v64(vx, vx + 64, tab, vw, 1)
            scale = bitcast_i32_to_f32(vx[64], f32)
        denom = denom * scale
        if kn > 0:
            store_v(ptr[f32](vx), load_v(scores, 16, 32), 32)
            store_v(
                ptr[f32](vx),
                ACC2048_accfloat_sub_conf(
                    load_v(ptr[f32](vx), 64, 32), broadcast(f32, 64, maximum), 60
                ),
                32,
            )
            exp_v64(vx, vx + 64, tab, vw, 1)
        key = 0
        while key < kn:
            delta = qb + row - kb - key
            if mka[key] != 0 and (local == 0 or (delta >= -64 and delta <= 64)):
                ex: f32 = bitcast_i32_to_f32(vx[64 + key], f32)
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
    tab: ptr[u32, True],
    vx: ptr[u32, True],
    vw: ptr[u32, True],
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
        db: u32 = bitcast_i32_to_f32(denom, u32)
        dpos: i32 = 0
        if (db > u32(0)) and (db <= u32(2139095040)):
            dpos = 1
        if dpos != 0 and same == 0:
            store_v(ptr[f32](vx), load_v(scores, 16, 32), 32)
            store_v(
                ptr[f32](vx),
                ACC2048_accfloat_sub_conf(
                    load_v(ptr[f32](vx), 64, 32), broadcast(f32, 64, stats[row * 2]), 60
                ),
                32,
            )
            exp_v64(vx, vx + 64, tab, vw, 1)
        if dpos != 0:
            sp: ptr[u32] = vx + 64
            if same != 0:
                sp = ptr[u32](exps)
            m12a_vdiv(sp, vx + 64, ptr[u16](vx), bitcast_i32_to_f32(denom, u32), 1, mka)
        key: i32 = 0
        while key < 16:
            delta: i32 = qb + row - kb - key
            if (
                dpos != 0
                and mka[key] != 0
                and (local == 0 or (delta >= -64 and delta <= 64))
            ):
                probability: bf16 = ptr[bf16](vx)[key]
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
