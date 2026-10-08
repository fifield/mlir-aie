# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# aie.helpers.dialects.aievec_aie4 builds the flat and region matrix-state
# ops with plain-int slots and ports, inferred result types, and checked
# signedness.

from aie.extras import types as T
from aie.helpers.dialects.aievec_aie4 import (
    BF16,
    FP8,
    INT4,
    INT8,
    UINT8,
    commit,
    matrix_region,
    mmac,
    mmac_active,
    stage,
    stage_stream,
)
from aie.helpers.dialects.func import func
from aie.dialects import vector
from aie.ir import Module

from util import construct_and_print_module


def expect_error(exc, f):
    try:
        f()
    except exc as e:
        print(f"{exc.__name__}: {e}")
        return
    raise AssertionError("no error raised")


# CHECK-LABEL: TEST: test_flat
# CHECK: %[[C3:.*]] = arith.constant 3 : i32
# CHECK: aievec.aie4.stage int8 %{{.*}}, %[[C3]] {rhs_signed = true} : vector<128xi8>, i32
# CHECK: aievec.aie4.stage int8 %{{.*}}, %[[S:.*]] {rhs_signed = false} : vector<128xi8>, i32
# CHECK: %[[P1:.*]] = arith.constant 1 : i32
# CHECK: %[[C7:.*]] = arith.constant 7 : i32
# CHECK: aievec.aie4.stage_stream int8 %[[P1]], %[[C7]] {rhs_signed = true} : i32, i32
# CHECK: aievec.aie4.stage_stream bf16 %{{.*}}, %{{.*}} : i32, i32
# CHECK: aievec.aie4.stage_stream int4 %{{.*}}, %{{.*}} {rhs_signed = false} : i32, i32

# CHECK: aievec.aie4.commit int8 {rhs_signed = true}
# CHECK: aievec.aie4.commit fp8


# CHECK: aievec.aie4.mmac_active int8 %{{.*}}, %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
# CHECK: aievec.aie4.mmac_active int8 %{{.*}} {lhs_signed = false, rhs_signed = false} : vector<64xi8> -> vector<64xi32>
# CHECK: aievec.aie4.mmac_active int8 %{{.*}}, %{{.*}} zero_if %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
# CHECK: %[[T:.*]] = arith.constant true
# CHECK: aievec.aie4.mmac_active int8 %{{.*}}, %{{.*}} zero_if %[[T]]
# CHECK: aievec.aie4.mmac_active bf16 %{{.*}} : vector<32xbf16> -> vector<64xf32>
@construct_and_print_module
def test_flat(module):
    i8, i32 = T.i8(), T.i32()

    @func(emit=True)
    def flat(
        chunk: T.vector(128, i8),
        s: i32,
        a: T.vector(64, i8),
        c: T.vector(64, i32),
        first: T.bool(),
        abf: T.vector(32, T.bf16()),
    ):
        stage(INT8, chunk, 3, rhs_signed=True)
        stage(UINT8, chunk, s)
        stage_stream(INT8, 1, 7, rhs_signed=True)
        stage_stream(BF16, 0, s)
        stage_stream(INT4, 1, s, rhs_signed=False)
        commit(INT8, rhs_signed=True)
        commit(FP8)
        r = mmac_active(INT8, a, c, lhs_signed=True, rhs_signed=True)
        mmac_active(UINT8, a, lhs_signed=False)
        mmac_active(INT8, a, r, zero_if=first, lhs_signed=True, rhs_signed=True)
        mmac_active(INT8, a, r, zero_if=True, lhs_signed=True, rhs_signed=True)
        mmac_active(BF16, abf)

    return module


# CHECK-LABEL: TEST: test_region
# CHECK: %[[R:.*]]:2 = aievec.aie4.matrix_region int8 %{{.*}} : vector<64x64xi8>, %{{.*}}, %{{.*}} : vector<64xi32>, vector<64xi32> -> vector<64xi32>, vector<64xi32> attributes {rhs_signed = true} {
# CHECK: ^bb0(%[[A0:.*]]: vector<64xi32>, %[[A1:.*]]: vector<64xi32>):
# CHECK:   %[[M0:.*]] = aievec.aie4.mmac int8 %{{.*}}, %[[A0]] {lhs_signed = true, rhs_signed = true}
# CHECK:   %[[M1:.*]] = aievec.aie4.mmac int8 %{{.*}}, %[[A1]] {lhs_signed = true, rhs_signed = true}
# CHECK:   aievec.aie4.matrix_yield %[[M0]], %[[M1]] : vector<64xi32>, vector<64xi32>
# CHECK: return %[[R]]#0, %[[R]]#1
# CHECK: aievec.aie4.matrix_region bf16 %{{.*}} : vector<32x64xbf16>, %{{.*}} : vector<64xf32> -> vector<64xf32> {
# CHECK:   aievec.aie4.mmac bf16 %{{.*}} : vector<32xbf16> -> vector<64xf32>
@construct_and_print_module
def test_region(module):
    i8, i32, bf16 = T.i8(), T.i32(), T.bf16()

    @func(emit=True)
    def region(
        b: T.vector(64, 64, i8),
        a: T.vector(2, 64, i8),
        c0: T.vector(64, i32),
        c1: T.vector(64, i32),
    ) -> (T.vector(64, i32), T.vector(64, i32)):
        @matrix_region(b, [c0, c1], INT8, rhs_signed=True)
        def out(*accs):
            return [
                mmac(
                    INT8,
                    vector.extract(a, [], [r]),
                    accs[r],
                    lhs_signed=True,
                    rhs_signed=True,
                )
                for r in range(2)
            ]

        return out

    @func(emit=True)
    def region_bf16(
        b: T.vector(32, 64, bf16), a: T.vector(32, bf16), c: T.vector(64, T.f32())
    ) -> T.vector(64, T.f32()):
        @matrix_region(b, [c], BF16)
        def out(acc):
            return mmac(BF16, a)

        return out

    return module


# CHECK-LABEL: TEST: test_errors
# CHECK: ValueError: mmac_active: zero_if is integer-only; peel the first K tile
# CHECK: ValueError: mmac_active: zero_if requires acc
# CHECK: ValueError: stage: UINT8 implies rhs_signed=False
# CHECK: ValueError: commit: int8 requires rhs_signed=True or False
# CHECK: ValueError: mmac_active: lhs_signed applies only to integer formats
# CHECK: TypeError: stage: slot must be an int or a Value
# CHECK: ValueError: stage: slot must be in [0, 32), got 32
# CHECK: ValueError: stage_stream: slot must be in [0, 32), got -1
# CHECK: ValueError: stage_stream: port must be one of 0, 1, got 2
@construct_and_print_module
def test_errors(module):
    i8, i32 = T.i8(), T.i32()

    @func(emit=True)
    def errors(
        chunk: T.vector(128, i8),
        a: T.vector(32, T.bf16()),
        c: T.vector(64, T.f32()),
    ):
        expect_error(ValueError, lambda: mmac_active(BF16, a, c, zero_if=True))
        expect_error(
            ValueError,
            lambda: mmac_active(
                INT8, chunk, zero_if=True, lhs_signed=True, rhs_signed=True
            ),
        )
        expect_error(ValueError, lambda: stage(UINT8, chunk, 0, rhs_signed=True))
        expect_error(ValueError, lambda: commit(INT8))
        expect_error(ValueError, lambda: mmac_active(BF16, a, lhs_signed=True))
        expect_error(TypeError, lambda: stage(INT8, chunk, 1.5, rhs_signed=True))
        expect_error(ValueError, lambda: stage(INT8, chunk, 32, rhs_signed=True))
        expect_error(ValueError, lambda: stage_stream(INT8, 0, -1, rhs_signed=True))
        expect_error(ValueError, lambda: stage_stream(INT8, 2, 0, rhs_signed=True))

    return module
