"""M35 native-float backbone attention leaf for the l64_fast template (fast B1/B2/B4 L64).

One core: a 16-query block of one head against 64 keys, in the template's fixed call order
(s2_header, s2_begin, exp_v64, s2_pass1 x4, s2_merge, s2_pass2 x4 with stage = 3 - i,
m6_finish_output); s2_header and s2_begin are the production leaves. The arithmetic is the
AIE2P's native float path (bf16 products, f32 accumulator, NLF exp2 and inv), not emulated IEEE:

* s2_pass1: Q K^T as in the fast leaf (bf16 x bf16 products accumulated in f32), stored
  key-major (sall[key * 16 + row]) so one 16-lane vector is one key for the 16 rows.
* s2_pass1(i == 3): softmax of the whole 16 x 64 block (no online rescale, no merge): row max
  over the ordered-integer image of the f32 scores; e = 2^((s - max) * 0.125 * log2 e) in f32
  (exp2_16); f32 row sums; p = bf16(e * inv(rowsum)), the bf16 probabilities the fast leaf
  also feeds to P.V. Every f32 x f32 product is split into bf16 hi + lo partial products.
* s2_pass2: acc[row, :] += p[row, key] * V[key, :], bf16 x bf16 into the f32 accumulator.
* m6_finish_output: acc rounded to bf16 (vector conversion).
* s2_merge, exp_v64: no-ops.

On AIE2P the ACC512 bf mul/mac with conf 60 is lane-wise over the low 16 lanes of each
operand (the upper 16 do not contribute; measured on the NPU in M35), so mul16/mac16 zero them.
Masking keeps the fast leaf's semantics: a key counts iff i32(mask) != 0 and, for local layers,
|query - key| <= 64 (never false at L64, but kept lane-exact). Rounding: crRnd = 12 (as the fast
leaf). Scratch: unit[0:64] key classes, sall exponentials, em[0:512] bf16 p, em[1024:1040] 1/rowsum.
"""

from pythoc import ptr, f32, bf16, i32, u16, u32, void
from pythoc.decorators.inline import inline
from pythoc.aie import (
    set_ctrl_reg,
    aie_vector,
    load_v,
    store_v,
    broadcast,
    zeros,
    concat,
    vshuffle,
    vector_cast,
    vector_xor,
    vector_max,
    vsel32,
    vlt32,
    exp2,
    inv,
    v16accfloat_to_v16bf16,
    I512_I512_ACC512_bf_mul_conf,
    I512_I512_ACC512_bf_mac_conf,
    ACC2048_accfloat_add_conf,
    ACC2048_accfloat_sub_conf,
)
from pythoc.aie.mmul import mmul_bf16, mmul_bf16_mac


@inline
def ordered16(s: aie_vector[i32, 16]) -> aie_vector[i32, 16]:
    # Order-preserving integer image of f32 bits (an involution): negatives get the
    # magnitude bits flipped. vlt32/vsel32, not ashr/and: <16 x i1> selects do not lower.
    return vsel32(
        s,
        vector_xor(s, broadcast(i32, 16, i32(2147483647))),
        vlt32(s, broadcast(i32, 16, i32(0)), 1),
    )


@inline
def mul16(a: aie_vector[bf16, 16], b: aie_vector[bf16, 16]) -> aie_vector[f32, 16]:
    # a * b, one bf16 x bf16 product per lane (exact in f32). On AIE2P the ACC512 bf
    # mul/mac with conf 60 is lane-wise over the low 16 lanes: the upper 16 of each
    # operand do not contribute (probe_nlf), so they are zero here.
    z: aie_vector[bf16, 16] = zeros(bf16, 16)
    return I512_I512_ACC512_bf_mul_conf(concat(a, z), concat(b, z), 60)


@inline
def mac16(
    a: aie_vector[bf16, 16], b: aie_vector[bf16, 16], acc: aie_vector[f32, 16]
) -> aie_vector[f32, 16]:
    # acc + a * b in the f32 accumulator.
    z: aie_vector[bf16, 16] = zeros(bf16, 16)
    return I512_I512_ACC512_bf_mac_conf(concat(a, z), concat(b, z), acc, 60)


@inline
def lo16(x: aie_vector[f32, 16], hi: aie_vector[bf16, 16]) -> aie_vector[bf16, 16]:
    # bf16(x - hi): x - hi is exact in the f32 accumulator.
    return v16accfloat_to_v16bf16(mac16(hi, broadcast(bf16, 16, bf16(-1.0)), x))


@inline
def mul_scale16(d: aie_vector[f32, 16]) -> aie_vector[f32, 16]:
    # d * 0.125 * log2(e) ~ dh*ch + dl*ch + dh*cl (ch + cl = c to 2^-17).
    dh: aie_vector[bf16, 16] = v16accfloat_to_v16bf16(d)
    dl: aie_vector[bf16, 16] = lo16(d, dh)
    ch: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(0.1806640625))
    cl: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(-0.00032806396484375))
    return mac16(dh, cl, mac16(dl, ch, mul16(dh, ch)))


@inline
def mul_step16(
    a: aie_vector[f32, 16],
    fh: aie_vector[bf16, 16],
    fl: aie_vector[bf16, 16],
    c: f32,
) -> aie_vector[f32, 16]:
    # c + a * f with a and f split into bf16 hi + lo.
    ah: aie_vector[bf16, 16] = v16accfloat_to_v16bf16(a)
    al: aie_vector[bf16, 16] = lo16(a, ah)
    return mac16(ah, fl, mac16(al, fh, mac16(ah, fh, broadcast(f32, 16, c))))


@inline
def exp2_16(t: aie_vector[f32, 16]) -> aie_vector[f32, 16]:
    # 2^t in the f32 accumulator: 2^t = 2^(n + 64) * (2^-64 * 2^f), n = round(t), |f| <= 1/2.
    # 2^-64 * 2^f is a degree-4 minimax polynomial (relative error 2.8e-6) with coefficients
    # pre-scaled by 2^-64, in split-bf16 arithmetic; 2^(n + 64) comes from the NLF unit
    # (llvm.aie2p.exp2), which is exact at non-negative integers (it is piecewise linear in
    # between, and one bf16 ulp low below zero). t < -64 is clamped to -64: such terms
    # (< 5.5e-20) are below every row sum's ulp.
    one: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(1.0))
    m1: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(-1.0))
    c64: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(64.0))
    magic: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(12582912.0))
    ti: aie_vector[i32, 16] = vector_cast(t, i32, 16)
    low: i32 = vlt32(ordered16(ti), broadcast(i32, 16, i32(-1115684865)), 1)
    tc: aie_vector[f32, 16] = vector_cast(
        vsel32(ti, broadcast(i32, 16, i32(-1031798784)), low), f32, 16
    )
    y: aie_vector[f32, 16] = mac16(magic, one, tc)
    n64: aie_vector[f32, 16] = mac16(c64, one, mac16(magic, m1, y))
    nb: aie_vector[bf16, 16] = v16accfloat_to_v16bf16(n64)
    f: aie_vector[f32, 16] = mac16(c64, one, mac16(nb, m1, tc))
    fh: aie_vector[bf16, 16] = v16accfloat_to_v16bf16(f)
    fl: aie_vector[bf16, 16] = lo16(f, fh)
    c4h: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(5.194694246833014e-22))
    c4l: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(1.8074381060423822e-26))
    a: aie_vector[f32, 16] = mac16(
        c4h, fl, mac16(c4l, fh, mac16(c4h, fh, broadcast(f32, 16, f32(3.030693435011435e-21))))
    )
    a = mul_step16(a, fh, fl, f32(1.3023490224378592e-20))
    a = mul_step16(a, fh, fl, f32(3.7574336820748897e-20))
    a = mul_step16(a, fh, fl, f32(5.421010862427522e-20))
    s: aie_vector[bf16, 16] = exp2(n64)
    ph: aie_vector[bf16, 16] = v16accfloat_to_v16bf16(a)
    pl: aie_vector[bf16, 16] = lo16(a, ph)
    return mac16(pl, s, mul16(ph, s))


@inline
def mask_on(hdr: ptr[u16], k: i32) -> i32:
    # i32(mask) != 0 for a bf16 mask word, as an integer test: |mask| >= 1.0.
    if (u32(hdr[32 + k]) & u32(32767)) >= u32(16256):
        return 1
    return 0


@inline
def key_class(hdr: ptr[u16], qb: i32, local: i32, k: i32) -> i32:
    # 0: key masked for every row; 1: allowed for every row; 2: allowed for some rows.
    if mask_on(hdr, k) == 0:
        return 0
    if local == 0:
        return 1
    if qb + 15 - k < -64 or qb - k > 64:
        return 0
    if qb - k >= -64 and qb + 15 - k <= 64:
        return 1
    return 2


@inline
def row_allowed(qb: i32, row: i32, k: i32) -> i32:
    d: i32 = qb + row - k
    if d >= -64 and d <= 64:
        return 1
    return 0


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
    hdr: ptr[u16] = ptr[u16](unit + 128)
    qb: i32 = i32(hdr[0])
    local: i32 = i32(hdr[2])
    kb: i32 = i * 16
    kd: ptr[bf16] = ptr[bf16](chunk_obj)
    qtb: ptr[bf16] = ptr[bf16](qk)
    kn: i32 = 0
    ek: i32 = 0
    while ek < 16:
        mka[ek] = mask_on(hdr, kb + ek)
        if mka[ek] != 0 and (
            local == 0 or (qb + 15 - kb - ek >= -64 and qb - kb - ek <= 64)
        ):
            kn = 16
        ek = ek + 1
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
            store_v(sall + (kb + kk) * 16, acc, 32)
        kk = kk + 4
    if i == 3:
        neg: aie_vector[i32, 16] = broadcast(i32, 16, i32(-2139095040))
        mx: aie_vector[i32, 16] = neg
        sl: ptr[i32] = ptr[i32](sall)
        kc: ptr[i32] = ptr[i32](unit)
        k: i32 = 0
        while k < 64:
            kc[k] = key_class(hdr, qb, local, k)
            k = k + 1
        k = 0
        while k < 64:
            cls: i32 = kc[k]
            if cls != 0:
                s: aie_vector[i32, 16] = load_v(sl + k * 16, 16, 32)
                o: aie_vector[i32, 16] = ordered16(s)
                if cls == 2:
                    bits: i32 = 0
                    r: i32 = 0
                    while r < 16:
                        if row_allowed(qb, r, k) == 0:
                            bits = bits | (1 << r)
                        r = r + 1
                    o = vsel32(o, neg, bits)
                mx = vector_max(mx, o)
            k = k + 1
        mxf: aie_vector[i32, 16] = ordered16(mx)
        mv: ptr[i32] = ptr[i32](vx)
        store_v(mv, mxf, 32)
        store_v(mv + 16, mxf, 32)
        store_v(mv + 32, mxf, 32)
        store_v(mv + 48, mxf, 32)
        m64: aie_vector[f32, 64] = load_v(ptr[f32](vx), 64, 32)
        dv: ptr[f32] = ptr[f32](vx + 64)
        eb: ptr[bf16] = ptr[bf16](em)
        su: ptr[u32] = ptr[u32](sall)
        z: aie_vector[bf16, 16] = zeros(bf16, 16)
        rs64: aie_vector[f32, 64] = zeros(f32, 64)
        g: i32 = 0
        while g < 16:
            if (kc[g * 4] | kc[g * 4 + 1] | kc[g * 4 + 2] | kc[g * 4 + 3]) != 0:
                store_v(
                    dv,
                    ACC2048_accfloat_sub_conf(load_v(sall + g * 64, 64, 32), m64, 60),
                    32,
                )
                # e = 2^t in f32, written over the scores of this group (0 where masked)
                j: i32 = 0
                while j < 4:
                    kj: i32 = g * 4 + j
                    cj: i32 = kc[kj]
                    if cj == 0:
                        store_v(sall + kj * 16, zeros(f32, 16), 32)
                    else:
                        d16: aie_vector[f32, 16] = load_v(dv + j * 16, 16, 32)
                        store_v(sall + kj * 16, exp2_16(mul_scale16(d16)), 32)
                        if cj == 2:
                            r2: i32 = 0
                            while r2 < 16:
                                if row_allowed(qb, r2, kj) == 0:
                                    su[kj * 16 + r2] = u32(0)
                                r2 = r2 + 1
                    j = j + 1
                rs64 = ACC2048_accfloat_add_conf(rs64, load_v(sall + g * 64, 64, 32), 60)
            else:
                store_v(sall + g * 64, zeros(f32, 64), 32)
            g = g + 1
        # row sums: fold the four 16-row quarters; inverse per row (0 for an empty row)
        rv: ptr[f32] = ptr[f32](vx)
        store_v(rv, rs64, 32)
        iv: ptr[f32] = em + 1024
        rb: ptr[u32] = ptr[u32](vx + 64)
        r3: i32 = 0
        while r3 < 16:
            zr: f32 = (rv[r3] + rv[16 + r3]) + (rv[32 + r3] + rv[48 + r3])
            rv[64 + r3] = zr
            ivr: f32 = f32(0.0)
            if rb[r3] > u32(0) and rb[r3] < u32(2139095040):
                ivr = inv(zr)
            iv[r3] = ivr
            r3 = r3 + 1
        ivv: aie_vector[f32, 16] = load_v(iv, 16, 32)
        ivh: aie_vector[bf16, 16] = v16accfloat_to_v16bf16(ivv)
        ivl: aie_vector[bf16, 16] = lo16(ivv, ivh)
        # p = bf16(e * (1 / rowsum)), the probabilities P.V consumes (as the fast leaf)
        k = 0
        while k < 64:
            if kc[k] == 0:
                store_v(eb + k * 16, z, 32)
            else:
                e16: aie_vector[f32, 16] = load_v(sall + k * 16, 16, 32)
                eh: aie_vector[bf16, 16] = v16accfloat_to_v16bf16(e16)
                el: aie_vector[bf16, 16] = lo16(e16, eh)
                store_v(eb + k * 16, v16accfloat_to_v16bf16(mac16(eh, ivl, mac16(el, ivh, mul16(eh, ivh)))), 32)
            k = k + 1


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
    stage: i32 = 3 - i
    kb: i32 = stage * 16
    vd: ptr[bf16] = ptr[bf16](chunk_obj)
    eb: ptr[bf16] = ptr[bf16](em)
    if i == 0:
        zf: aie_vector[f32, 16] = zeros(f32, 16)
        o: i32 = 0
        while o < 1024:
            store_v(acc + o, zf, 32)
            o = o + 16
    # key classes, written by s2_pass1(i == 3) into the free partials area of unit
    kc: ptr[i32] = ptr[i32](unit)
    anyk: i32 = 0
    key: i32 = 0
    while key < 16:
        c: i32 = kc[kb + key]
        mka[key] = c
        anyk = anyk | c
        key = key + 1
    if anyk != 0:
        row: i32 = 0
        while row < 16:
            lo: aie_vector[f32, 32] = load_v(acc + row * 64, 32, 32)
            hi: aie_vector[f32, 32] = load_v(acc + row * 64 + 32, 32, 32)
            key = 0
            while key < 16:
                if mka[key] != 0:
                    pb: aie_vector[bf16, 32] = broadcast(
                        bf16, 32, eb[(kb + key) * 16 + row]
                    )
                    lo = mmul_bf16_mac(pb, load_v(vd + key * 64, 32, 32), lo)
                    hi = mmul_bf16_mac(pb, load_v(vd + key * 64 + 32, 32, 32), hi)
                key = key + 1
            store_v(acc + row * 64, lo, 32)
            store_v(acc + row * 64 + 32, hi, 32)
            row = row + 1


def m6_finish_output(state: ptr[f32, True], output: ptr[bf16, True]) -> void:
    # acc already holds the normalised P.V: round to bf16 (vector conversion).
    set_ctrl_reg(1, 12)
    o: i32 = 0
    while o < 1024:
        store_v(output + o, v16accfloat_to_v16bf16(load_v(state + o, 16, 32)), 32)
        o = o + 16


# No-ops bound to the template's fixed symbols; last in the file so no leaf compiles them as helpers.
def exp_v64(
    xs: ptr[u32, True],
    ys: ptr[u32, True],
    tab: ptr[u32, True],
    wk: ptr[u32, True],
    ng: i32,
) -> void:
    pass


def s2_merge(unit: ptr[f32, True]) -> void:
    pass
