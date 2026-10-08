# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Build int8 regions with row accumulators and explicit result types for a
# region without inits. Check signedness and row assembly in the generated IR.

from aie.extras import types as T
from aie.helpers.dialects.aievec_aie4 import INT8, matrix_region, mmac
from aie.helpers.dialects.func import func
from aie.dialects import ub, vector
from aie.ir import Context, InsertionPoint, Location, Module

M, N, K = 4, 64, 64
LHS_SIGNED, RHS_SIGNED = False, True


def build_with_helpers():
    i8, i32 = T.i8(), T.i32()
    acc_ty = T.vector(M, N, i32)
    module = Module.create()
    with InsertionPoint(module.body):

        @func(emit=True)
        def r1_r3(
            a0: T.vector(M, K, i8), bt0: T.vector(N, K, i8), c0: acc_ty
        ) -> (acc_ty, acc_ty):
            b0 = vector.transpose(T.vector(K, N, i8), bt0, [1, 0])

            def rows(acc_rows):
                # One row MMAC per output row; with acc_rows None the
                # hardware zeroes the accumulator.
                out = ub.poison(acc_ty) if acc_rows is None else acc_rows
                for r in range(M):
                    a = vector.extract(a0, [], [r])
                    c = None if acc_rows is None else vector.extract(acc_rows, [], [r])
                    x = mmac(INT8, a, c, lhs_signed=LHS_SIGNED, rhs_signed=RHS_SIGNED)
                    out = vector.insert(x, out, [], [r])
                return out

            @matrix_region(b0, [c0], INT8, rhs_signed=RHS_SIGNED)
            def r1(acc):
                return rows(acc)

            @matrix_region(b0, [], INT8, rhs_signed=RHS_SIGNED, results=[acc_ty])
            def r3():
                return rows(None)

            return r1, r3

    return module


# CHECK-LABEL: func.func @r1_r3
# CHECK: %[[BT:.*]] = vector.transpose
# CHECK: %[[R1:.*]] = aievec.aie4.matrix_region int8 %[[BT]] : vector<64x64xi8>, %{{.*}} : vector<4x64xi32> -> vector<4x64xi32> attributes {rhs_signed = true}
# CHECK-COUNT-4: aievec.aie4.mmac int8 %{{.*}}, %{{.*}} {lhs_signed = false, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
# CHECK: aievec.aie4.matrix_yield %{{.*}} : vector<4x64xi32>
# CHECK: %[[R3:.*]] = aievec.aie4.matrix_region int8 %[[BT]] : vector<64x64xi8> -> vector<4x64xi32> attributes {rhs_signed = true}
# CHECK: ub.poison : vector<4x64xi32>
# CHECK-COUNT-4: aievec.aie4.mmac int8 %{{.*}} {lhs_signed = false, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
# CHECK: aievec.aie4.matrix_yield %{{.*}} : vector<4x64xi32>
# CHECK: return %[[R1]], %[[R3]] : vector<4x64xi32>, vector<4x64xi32>
with Context(), Location.unknown():
    module = build_with_helpers()
    assert module.operation.verify()
    print(module)
