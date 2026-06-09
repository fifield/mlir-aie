# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mode-switched BF16 matrix-vector multiply for the packet-fed proj-engine.

One kernel that carries BOTH the K=2048 (loop_range 32) and K=8192
(loop_range 128) inner loops, selected at runtime by a `mode` argument the
core reads from an RTP. This is the "concatenate the different core bodies in
a switch statement with one mode RTP" strategy: each branch keeps its own
compile-time `loop_range` hint (the constant that actually unlocks Peano's
zero-overhead hardware loop + software pipelining), so neither body is
de-specialized. The math in each branch is byte-for-byte the same as
kernels/matvec.py (mode 0) and kernels/matvec_k8192.py (mode 1).

Why a `mode` switch rather than feeding K as a runtime value into a single
loop: K itself is *already* a runtime arg (`while j < k`), but the
`loop_range(N)` trip-count hint is a compile-time constant -- making it
dynamic loses the hardware loop. So we keep two hinted bodies and branch.

The branches use disjoint local names so the PythoC `if`/`else` lowering
never has to merge a loop-carried accumulator across the two arms.

Used by `o_gemv_ffn` (pack_mode `d1d3d4_rms_fmv`) as matvec_fused_pythoc.o;
the per-tile `mode` RTP is hard-coded by the runtime sequence (1 for the
K=8192 down projection, 0 for the K=2048 O/gate/up projections).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u32, void
from pythoc.aie import (
    aie_vector,
    load_v,
    loop_range,
    prepare_for_pipelining,
    reduce_add_reassoc,
    zeros,
)

# Lazy AIE2P intrinsic.
from pythoc.aie import I1024_I1024_ACC2048_bf_mac_conf  # noqa: F401


@aie_kernel
def mvf_linalg_fill_bf16(c: ptr[bf16, True]) -> void:
    """Zero 2 bf16 elements (down y buffer is M_TILE_K8192=2).

    Defined FIRST so compile_pythoc_source co-compiles it as a helper of
    matvec_fused_bf16 -- both land in matvec_fused_pythoc.o, so the fused
    core links ONLY that object (no mv_k8192.o, so its size is clean).
    The matvec overwrites every output row, so the fill width is not
    correctness-critical; 2 scalars is safe for both the K=8192 (m=2) and
    K=2048 (m=8, buffer overwritten) call sites.
    """
    c[0] = bf16(0.0)
    c[1] = bf16(0.0)


@aie_kernel
def matvec_fused_bf16(
    mode: u32,
    m: u32,
    k: u32,
    row_offset: u32,
    a: ptr[bf16, True],
    b: ptr[bf16, True],
    c: ptr[bf16, True],
) -> void:
    r: u32 = u32(64)
    # conf=60 selects per-lane bf16 MAC on AIE2P (see kernels/matvec.py).
    conf: i32 = i32(60)

    if mode == u32(0):
        # ---- mode 0: K=2048 role, 32 inner iters (== kernels/matvec.py) ----
        p_c: ptr[bf16] = c + row_offset
        p_a_row: ptr[bf16] = a
        i: u32 = u32(0)
        while i < m:
            acc: aie_vector[f32, 64] = zeros(f32, 64)
            p_a: ptr[bf16] = p_a_row
            p_b: ptr[bf16] = b
            j: u32 = u32(0)
            with prepare_for_pipelining():
                with loop_range(32):
                    while j < k:
                        a_v: aie_vector[bf16, 64] = load_v(p_a, 64)
                        b_v: aie_vector[bf16, 64] = load_v(p_b, 64)
                        acc = I1024_I1024_ACC2048_bf_mac_conf(a_v, b_v, acc, conf)
                        p_a = p_a + r
                        p_b = p_b + r
                        j = j + r
            s: f32 = reduce_add_reassoc(acc)
            p_c[0] = bf16(s)
            p_c = p_c + u32(1)
            p_a_row = p_a_row + k
            i = i + u32(1)
    else:
        # ---- mode 1: K=8192 role, 128 inner iters (== matvec_k8192.py) ----
        q_c: ptr[bf16] = c + row_offset
        q_a_row: ptr[bf16] = a
        ii: u32 = u32(0)
        while ii < m:
            acc2: aie_vector[f32, 64] = zeros(f32, 64)
            q_a: ptr[bf16] = q_a_row
            q_b: ptr[bf16] = b
            jj: u32 = u32(0)
            with prepare_for_pipelining():
                with loop_range(128):
                    while jj < k:
                        a_w: aie_vector[bf16, 64] = load_v(q_a, 64)
                        b_w: aie_vector[bf16, 64] = load_v(q_b, 64)
                        acc2 = I1024_I1024_ACC2048_bf_mac_conf(a_w, b_w, acc2, conf)
                        q_a = q_a + r
                        q_b = q_b + r
                        jj = jj + r
            s2: f32 = reduce_add_reassoc(acc2)
            q_c[0] = bf16(s2)
            q_c = q_c + u32(1)
            q_a_row = q_a_row + k
            ii = ii + u32(1)
