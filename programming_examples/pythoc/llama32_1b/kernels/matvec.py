# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BF16 matrix-vector multiply + zero-fill kernels.

Replaces the AIR-tree reference `mv.cc`. For each output row:
    c[i] = sum_j a[i*k + j] * b[j]   for i in [0, m)
with `row_offset` adding into c (c_out += row_offset).

Mirrors the .cc's precision discipline: each row's dot product is
accumulated in 32-lane accfloat via `I512_I512_ACC1024_bf_mac_conf`
(conf=60 -- per-lane bf16 MAC; conf=0 silently picks a different
sub-element pattern and produces garbage), then horizontally reduced to
scalar f32 via `reduce_add`, and only the final scalar is truncated to
bf16 at store. Reducing in bf16 earlier loses too much precision; the
HF answer-level gate catches it.

Wired into rms_gemv_rope, o_gemv_ffn, and lm_head_gemv as mv_pythoc.o
(see kernels/build.py:compile_matvec). The K=8192 FFN down-projection
variant with `dg_*` symbol names lives in matvec_k8192.py.

Depends on the PythoC fix at PythoC@09cf024 that adds float support to
`reduce_add` and to `extract_elem`'s type-hint mapping.
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u32, void
from pythoc.aie import (
    aie_vector,
    load_v,
    loop_range,
    prepare_for_pipelining,
    reduce_add_reassoc,
    store_v,
    zeros,
)

# Lazy AIE2P intrinsic.
from pythoc.aie import I1024_I1024_ACC2048_bf_mac_conf  # noqa: F401


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
    r: u32 = u32(64)
    # conf=60 selects per-lane bf16 MAC on AIE2P. Matches air's
    # aie::mac<accfloat, 64> which lowers to one I1024 MAC per iteration
    # (vs two I512 MACs the 32-lane form needs), doubling K-throughput.
    conf: i32 = i32(60)

    p_c: ptr[bf16] = c + row_offset
    p_a_row: ptr[bf16] = a

    i: u32 = u32(0)
    while i < m:
        acc: aie_vector[f32, 64] = zeros(f32, 64)
        p_a: ptr[bf16] = p_a_row
        p_b: ptr[bf16] = b
        j: u32 = u32(0)
        # K=2048, r=64 -> 32 inner iters in practice. Loop hints unlock
        # peano's zero-overhead `lc/le` hardware loop + software pipelining;
        # without them pythoc emits a manual ltu+jnz counter that costs
        # ~3 cycles/iter on top of the MAC.
        with prepare_for_pipelining():
            with loop_range(32):
                while j < k:
                    a_v: aie_vector[bf16, 64] = load_v(p_a, 64)
                    b_v: aie_vector[bf16, 64] = load_v(p_b, 64)
                    acc = I1024_I1024_ACC2048_bf_mac_conf(a_v, b_v, acc, conf)
                    p_a = p_a + r
                    p_b = p_b + r
                    j = j + r

        # Horizontal sum in f32, truncate to bf16 only at the final store.
        s: f32 = reduce_add_reassoc(acc)
        p_c[0] = bf16(s)

        p_c = p_c + u32(1)
        p_a_row = p_a_row + k
        i = i + u32(1)
