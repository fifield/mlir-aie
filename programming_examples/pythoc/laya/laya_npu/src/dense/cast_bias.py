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
def cast_bias(src: ptr[f32, True], dst: ptr[bf16, True]) -> void:
    set_ctrl_reg(1, 12)
    i: i32 = 0
    while i < 256:
        a: aie_vector[f32, 32] = load_v(src + i, 32, 32)
        b: aie_vector[f32, 32] = load_v(src + 256 + i, 32, 32)
        z: aie_vector[f32, 32] = zeros(f32, 32)
        total: aie_vector[f32, 64] = ACC2048_accfloat_add_conf(
            concat(a, z), concat(b, z), 60
        )
        short: aie_vector[f32, 32] = vector_extract(total, 0, 32)
        store_v(dst + i, v32accfloat_to_v32bf16(short), 32)
        i = i + 32
