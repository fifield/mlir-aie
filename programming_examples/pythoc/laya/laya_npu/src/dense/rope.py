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
def rope(src: ptr[bf16, True], dst: ptr[f32, True]) -> void:
    set_ctrl_reg(1, 12)
    i: i32 = 0
    while i < 32:
        a: f32 = f32(src[i])
        b: f32 = f32(src[32 + i])
        c: f32 = f32(src[64 + i])
        s: f32 = f32(src[96 + i])
        dst[i] = a * c - b * s
        dst[32 + i] = a * s + b * c
        i = i + 1
