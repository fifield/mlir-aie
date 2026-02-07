# my_kernels.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""Reusable PythoC AIE kernel library.

A collection of element-wise vector kernels written in PythoC for AIE2/AIE2P.
Import any kernel and pass it to PythocKernel for use in IRON programs:

    from my_kernels import add_kernel
    kernel = PythocKernel(add_kernel, [tile_ty, tile_ty, tile_ty, np.int32])
"""

from aie.iron.pythoc import aie_kernel

from pythoc import ptr, i32, bf16, void
from pythoc.aie.operations import (
    load_v,
    store_v,
    vector_add,
    vector_sub,
    vector_mul,
)
from pythoc.aie.vector import aie_vector
from pythoc.aie.profiling import event0, event1


# ---------------------------------------------------------------------------
# Integer kernels (i32, 16-wide vectors)
# ---------------------------------------------------------------------------

@aie_kernel
def add_kernel(a: ptr[i32, True], b: ptr[i32, True], c: ptr[i32, True], N: i32):
    """Vectorized element-wise addition: C[i] = A[i] + B[i]."""
    event0()

    vec_size: i32 = 16
    iterations: i32 = N // vec_size

    pA: ptr[i32] = a
    pB: ptr[i32] = b
    pC: ptr[i32] = c

    i: i32 = 0
    while i < iterations:
        va: aie_vector[i32, 16] = load_v(pA, 16)
        vb: aie_vector[i32, 16] = load_v(pB, 16)
        vc: aie_vector[i32, 16] = vector_add(va, vb)
        store_v(pC, vc)

        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1

    event1()


@aie_kernel
def sub_kernel(a: ptr[i32, True], b: ptr[i32, True], c: ptr[i32, True], N: i32):
    """Vectorized element-wise subtraction: C[i] = A[i] - B[i]."""
    event0()

    vec_size: i32 = 16
    iterations: i32 = N // vec_size

    pA: ptr[i32] = a
    pB: ptr[i32] = b
    pC: ptr[i32] = c

    i: i32 = 0
    while i < iterations:
        va: aie_vector[i32, 16] = load_v(pA, 16)
        vb: aie_vector[i32, 16] = load_v(pB, 16)
        vc: aie_vector[i32, 16] = vector_sub(va, vb)
        store_v(pC, vc)

        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1

    event1()


# ---------------------------------------------------------------------------
# BFloat16 kernels (bf16, 16-wide vectors)
# ---------------------------------------------------------------------------

@aie_kernel
def add_bf16_kernel(
    a: ptr[bf16, True], b: ptr[bf16, True], c: ptr[bf16, True], N: i32
):
    """Vectorized element-wise bf16 addition: C[i] = A[i] + B[i]."""
    event0()

    vec_size: i32 = 16
    iterations: i32 = N // vec_size

    pA: ptr[bf16] = a
    pB: ptr[bf16] = b
    pC: ptr[bf16] = c

    i: i32 = 0
    while i < iterations:
        va: aie_vector[bf16, 16] = load_v(pA, 16)
        vb: aie_vector[bf16, 16] = load_v(pB, 16)
        vc: aie_vector[bf16, 16] = vector_add(va, vb)
        store_v(pC, vc)

        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1

    event1()


@aie_kernel
def mul_bf16_kernel(
    a: ptr[bf16, True], b: ptr[bf16, True], c: ptr[bf16, True], N: i32
):
    """Vectorized element-wise bf16 multiplication: C[i] = A[i] * B[i]."""
    event0()

    vec_size: i32 = 16
    iterations: i32 = N // vec_size

    pA: ptr[bf16] = a
    pB: ptr[bf16] = b
    pC: ptr[bf16] = c

    i: i32 = 0
    while i < iterations:
        va: aie_vector[bf16, 16] = load_v(pA, 16)
        vb: aie_vector[bf16, 16] = load_v(pB, 16)
        vc: aie_vector[bf16, 16] = vector_mul(va, vb)
        store_v(pC, vc)

        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1

    event1()
