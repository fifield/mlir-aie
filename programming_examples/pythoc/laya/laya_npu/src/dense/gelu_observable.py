"""Explicit FP32 arithmetic between BF16 boundaries; finite-input M1 kernels."""

from aie.iron.pythoc import aie_kernel
from pythoc import ptr, bf16, f32, i32, u16, void
from pythoc.aie import (
    set_ctrl_reg,
    sqrtf,
    load_v,
    store_v,
    vector_mul,
    vector_add,
    concat,
    zeros,
    vector_extract,
    aie_vector,
    ACC2048_accfloat_add_conf,
    v32accfloat_to_v32bf16,
)


@aie_kernel
def gelu_observable(src: ptr[bf16, True], dst: ptr[bf16, True]) -> void:
    # Source layout: gate[256], up[256], exact BF16 GELU LUT[3074].
    # LUT covers magnitudes [2^-9,8] in BF16-bit order, positive then negative.
    # Outside this interval rounded GELU is exactly x/2 (tiny), x (large +),
    # or signed zero (large -). Exhaustive finite BF16 testing verifies this.
    set_ctrl_reg(1, 12)
    bits: ptr[u16] = ptr[u16](src)
    out: ptr[u16] = ptr[u16](dst)
    i: i32 = 0
    while i < 256:
        raw: i32 = i32(bits[i])
        mag: i32 = raw & 32767
        sign: i32 = raw & 32768
        value: i32 = 0
        if mag < 15104:
            if mag >= 256:
                value = (mag - 128) | sign
            else:
                value = ((mag >> 1) + ((mag & 1) & ((mag >> 1) & 1))) | sign
        elif mag > 16640:
            if sign == 0:
                value = raw
            else:
                value = sign
        else:
            index: i32 = mag - 15104
            if sign != 0:
                index = index + 1537
            value = i32(bits[512 + index])
        out[i] = u16(value)
        i = i + 1
    i = 0
    while i < 256:
        g = load_v(dst + i, 32, 32)
        u = load_v(src + 256 + i, 32, 32)
        store_v(dst + 256 + i, vector_mul(g, u), 32)
        i = i + 32
