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


def m12a_vscale(p: ptr[u32, True], ng: i32) -> void:
    g: i32 = 0
    while g < ng:
        x: aie_vector[i32, 16] = load_v(ptr[i32](p) + g * 16, 16, 32)
        ef: aie_vector[i32, 16] = vector_and(x, broadcast(i32, 16, i32(2139095040)))
        fast: i32 = vge32(ef, broadcast(i32, 16, i32(33554432)), 1) & vlt32(
            ef, broadcast(i32, 16, i32(2139095040)), 1
        )
        zl: i32 = veqz32(vector_and(x, broadcast(i32, 16, i32(2147483647))))
        big: i32 = vge32(ef, broadcast(i32, 16, i32(2139095040)), 1)
        y: aie_vector[i32, 16] = vsel32(
            x, vector_sub(x, broadcast(i32, 16, i32(25165824))), fast
        )
        y = vsel32(
            y,
            vector_or(
                x,
                vsel32(
                    broadcast(i32, 16, i32(0)),
                    broadcast(i32, 16, i32(4194304)),
                    vlt32(
                        broadcast(i32, 16, i32(2139095040)),
                        vector_and(x, broadcast(i32, 16, i32(2147483647))),
                        1,
                    ),
                ),
            ),
            big,
        )
        store_v(ptr[i32](p) + g * 16, y, 32)
        slow: i32 = (fast | zl | big) ^ 65535
        if slow != 0:
            l: i32 = 0
            while l < 16:
                if ((slow >> l) & 1) != 0:
                    p[g * 16 + l] = bitcast_i32_to_f32(
                        bitcast_i32_to_f32(p[g * 16 + l], f32) * f32(0.125), u32
                    )
                l = l + 1
        g = g + 1
