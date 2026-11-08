from aie.iron.pythoc import aie_kernel
from pythoc import ptr, i32, void
from pythoc.aie import aie_vector, zeros, vshuffle, store_v


@aie_kernel
def bf16_zero_kernel(c_buf: ptr[i32, True]) -> void:
    # event0()
    i: i32 = 0
    z: aie_vector[i32, 16] = zeros(i32, 16)
    # vshuffle is opaque to opt, preventing memset conversion
    zero_vec: aie_vector[i32, 16] = vshuffle(z, z, 0)
    while i < C_ELEMS_CONST:
        store_v(c_buf + i, zero_vec)
        i = i + 16
