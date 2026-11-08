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
def residual(src: ptr[bf16, True], dst: ptr[bf16, True]) -> void:
    set_ctrl_reg(1, 12)
    i: i32 = 0
    while i < 256:
        a = load_v(src + i, 32, 32)
        b = load_v(src + 256 + i, 32, 32)
        store_v(dst + i, vector_add(a, b), 32)
        i = i + 32
