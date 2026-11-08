from pythoc import ptr, f32, bf16, i32, u16, u32, void
from pythoc.aie import (
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
    vlt32,
    vge32,
    veqz32,
    vsel32,
    accfloat_to_bf16,
)


def m12a_vdiv(
    src: ptr[u32, True],
    dst: ptr[u32, True],
    dstb: ptr[u16, True],
    db: u32,
    ng: i32,
    mk: ptr[i32, True],
) -> void:
    zero: aie_vector[i32, 16] = broadcast(i32, 16, i32(0))
    one: aie_vector[i32, 16] = broadcast(i32, 16, i32(1))
    hid: aie_vector[i32, 16] = broadcast(i32, 16, i32(8388608))
    infv: aie_vector[i32, 16] = broadcast(i32, 16, i32(2139095040))
    dbv: aie_vector[i32, 16] = broadcast(i32, 16, i32(db))
    dab: aie_vector[i32, 16] = vector_and(dbv, broadcast(i32, 16, i32(2147483647)))
    dnan: i32 = vlt32(infv, dab, 1)
    dinf: i32 = veqz32(vector_sub(dab, infv))
    dzero: i32 = veqz32(dab)
    dsub: i32 = veqz32(vector_and(dbv, infv)) & (dzero ^ 65535)
    dm: i32 = i32(db & u32(8388607))
    dfe: i32 = i32(db & u32(2139095040))
    if dsub != 0:
        dfe = 8388608
        while dm < 8388608:
            dm = dm + dm
            dfe = dfe - 8388608
    else:
        dm = dm | 8388608
    rofs: i32 = 1065353216 - dfe
    thr: i32 = 2147483647
    if rofs > -16777215:
        thr = 2130706432 - rofs
    dmv: aie_vector[i32, 16] = broadcast(i32, 16, dm)
    g: i32 = 0
    while g < ng:
        eb: aie_vector[i32, 16] = load_v(ptr[i32](src) + g * 16, 16, 32)
        live: i32 = veqz32(load_v(mk + g * 16, 16, 32)) ^ 65535
        eab: aie_vector[i32, 16] = vector_and(eb, broadcast(i32, 16, i32(2147483647)))
        sg: aie_vector[i32, 16] = vector_and(
            vector_sub(vector_or(eb, dbv), vector_and(eb, dbv)),
            broadcast(i32, 16, i32(-2147483648)),
        )
        enan: i32 = vlt32(infv, eab, 1)
        einf: i32 = veqz32(vector_sub(eab, infv))
        ezero: i32 = veqz32(eab)
        ef: aie_vector[i32, 16] = vector_and(eb, infv)
        esub: i32 = veqz32(ef) & (ezero ^ 65535)
        me: aie_vector[i32, 16] = vector_or(
            vector_and(eb, broadcast(i32, 16, i32(8388607))), vsel32(hid, zero, esub)
        )
        ef = vsel32(ef, hid, esub)
        nm: i32 = vlt32(me, hid, 1) & esub & live
        while nm != 0:
            me = vsel32(me, vector_add(me, me), nm)
            ef = vsel32(ef, vector_sub(ef, hid), nm)
            nm = vlt32(me, hid, 1) & esub & live
        small: i32 = vlt32(me, dmv, 1)
        sm8: aie_vector[i32, 16] = vsel32(zero, hid, small)
        r: aie_vector[i32, 16] = vsel32(me, vector_add(me, me), small)
        efs: aie_vector[i32, 16] = vector_sub(ef, sm8)
        ev: aie_vector[i32, 16] = vector_add(efs, broadcast(i32, 16, rofs))
        ovf: i32 = vlt32(broadcast(i32, 16, thr), efs, 1)
        ec: aie_vector[i32, 16] = vsel32(ev, hid, vlt32(hid, ev, 1))
        q: aie_vector[i32, 16] = zero
        gm: i32 = 0
        st: i32 = 0
        kt: i32 = -184549376
        it: i32 = 0
        while it < 25:
            t: aie_vector[i32, 16] = vector_sub(r, dmv)
            b: i32 = vge32(t, zero, 1)
            r = vsel32(r, t, b)
            kp: i32 = vge32(ec, broadcast(i32, 16, kt), 1)
            gq: i32 = veqz32(vector_sub(ec, broadcast(i32, 16, kt - 8388608)))
            q = vsel32(q, vector_add(vector_add(q, q), vsel32(zero, one, b)), kp)
            gm = gm | (gq & b)
            st = st | (((kp | gq) ^ 65535) & b)
            r = vector_add(r, r)
            kt = kt + 8388608
            it = it + 1
        st = st | (veqz32(r) ^ 65535)
        inc: i32 = gm & (st | (veqz32(vector_and(q, one)) ^ 65535))
        qi: aie_vector[i32, 16] = vector_add(q, vsel32(zero, one, inc))
        res: aie_vector[i32, 16] = vsel32(
            qi, vector_add(ev, vector_sub(qi, hid)), vge32(ev, hid, 1)
        )
        res = vsel32(res, infv, ovf)
        res = vector_or(res, sg)
        res = vsel32(res, vector_or(infv, sg), dzero)
        res = vsel32(res, sg, ezero)
        res = vsel32(res, sg, dinf)
        res = vsel32(res, vector_or(infv, sg), einf)
        res = vsel32(
            res, broadcast(i32, 16, i32(2143289344)), (einf & dinf) | (ezero & dzero)
        )
        res = vsel32(res, vector_or(dbv, broadcast(i32, 16, i32(4194304))), dnan)
        res = vsel32(res, vector_or(eb, broadcast(i32, 16, i32(4194304))), enan)
        store_v(ptr[i32](dst) + g * 16, res, 32)
        store_v(
            ptr[bf16](dstb) + g * 16, accfloat_to_bf16(vector_cast(res, f32, 16)), 32
        )
        g = g + 1
