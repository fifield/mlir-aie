# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BF16 matrix-vector multiply + zero-fill kernels.

Replaces the AIR-tree reference `mv.cc`:
    c[i] = sum_j a[i*k + j] * b[j]   for i in [0, m)
with `row_offset` adding into c (c_out += row_offset).

Uses the 32-lane bf16 MAC into 32-lane f32-accumulator intrinsic
`I512_I512_ACC1024_bf_mac_conf`. This is the same kernel shape as the
.cc (`aie::mac` with template parameter r) but with r=32 instead of r=64
because PythoC's `reduce_add` doesn't support float vectors and we want
to convert acc -> bf16 vector via `v32accfloat_to_v32bf16` for the
horizontal sum.

The numeric drift vs the .cc's `aie::reduce_add(acc.to_vector<float>())`:
the .cc reduces in f32 then casts; we reduce in bf16. For BF16 weights
the difference is sub-ULP per row in expectation, and is bounded by
sum(|a*b|) * 2^-7 worst case (single bf16 rounding per partial sum).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u32, void
from pythoc.aie import (
    aie_vector,
    extract_elem,
    load_v,
    shuffle_down,
    store_v,
    vector_add,
    v32accfloat_to_v32bf16,
    zeros,
)

# Lazy AIE2P intrinsics.
from pythoc.aie import I512_I512_ACC1024_bf_mac_conf  # noqa: F401


# Output-row count for the zero-fill helper. Matches `-DDIM_M_OUTPUT=8`
# the AIR tree uses when compiling mv.cc as mv.o.
DIM_M_OUTPUT: i32 = 8


@aie_kernel
def linalg_fill_bf16(c: ptr[bf16, True]) -> void:
    """Zero DIM_M_OUTPUT bf16 elements at `c`.

    Defined FIRST in the source so when compile_pythoc_source runs with
    function_name="matvec_vectorized_bf16_bf16", linalg_fill_bf16 becomes
    a "helper" and gets compiled into the same .o (single mv_pythoc.o
    carries both symbols). Helpers that come AFTER the target function in
    source order are skipped by the AST walker (it `break`s on the match).
    """
    z: aie_vector[bf16, 16] = zeros(bf16, 16)
    store_v(c, z)


@aie_kernel
def matvec_vectorized_bf16_bf16(
    m: u32,
    k: u32,
    row_offset: u32,
    a: ptr[bf16, True],
    b: ptr[bf16, True],
    c: ptr[bf16, True],
) -> void:
    r: u32 = u32(32)

    p_c: ptr[bf16] = c + row_offset
    p_a_row: ptr[bf16] = a

    i: u32 = u32(0)
    while i < m:
        acc: aie_vector[f32, 32] = zeros(f32, 32)
        p_a: ptr[bf16] = p_a_row
        p_b: ptr[bf16] = b
        j: u32 = u32(0)
        while j < k:
            a_v: aie_vector[bf16, 32] = load_v(p_a, 32)
            b_v: aie_vector[bf16, 32] = load_v(p_b, 32)
            acc = I512_I512_ACC1024_bf_mac_conf(a_v, b_v, acc, i32(0))
            p_a = p_a + r
            p_b = p_b + r
            j = j + r

        # Convert v32 accfloat -> v32 bf16 and tree-reduce via shuffle_down,
        # staying at 32-lane width throughout (PythoC's bf16 vector_add only
        # supports 16/32-lane widths; shuffle_down with high lanes zeroed
        # gives correct partial sums in the lower lanes).
        s: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(acc)
        s = vector_add(s, shuffle_down(s, 16))  # lane 0..15 carry pair sums
        s = vector_add(s, shuffle_down(s, 8))   # lane 0..7  carry quad sums
        s = vector_add(s, shuffle_down(s, 4))   # lane 0..3  carry octa sums
        s = vector_add(s, shuffle_down(s, 2))   # lane 0..1  carry 16-sums
        s = vector_add(s, shuffle_down(s, 1))   # lane 0     = total sum
        p_c[0] = extract_elem(s, i32(0))

        p_c = p_c + u32(1)
        p_a_row = p_a_row + k
        i = i + u32(1)
