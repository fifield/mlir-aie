# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Check enum exports, attribute builders and op-builder round trips.

from aie.extras import types as T
from aie.helpers.dialects.func import func
from aie.ir import AttrBuilder, Attribute, InsertionPoint, Module

from aie.dialects import aievec
from aie.dialects.aievec import (
    AIE4MatrixFormat,
    AIE4ReduceKind,
    ShuffleMode,
)
from util import construct_and_print_module

# CHECK-LABEL: TEST: test_enum_values
# CHECK: int8 1
# CHECK: max 2
# CHECK: t16_4x8 29
print("\nTEST: test_enum_values")
print(AIE4MatrixFormat.Int8, int(AIE4MatrixFormat.Int8))
print(AIE4ReduceKind.Max, int(AIE4ReduceKind.Max))
print(ShuffleMode.T16_4X8, int(ShuffleMode.T16_4X8))


# CHECK-LABEL: TEST: test_attribute_builder
# CHECK: #aievec<aie4_matrix_format int8>
# CHECK: #aievec.aie4_reduce_kind<max>
@construct_and_print_module
def test_attribute_builder(module):
    fmt = AttrBuilder.get("aievec.AIEVec_AIE4MatrixFormatAttr")(
        AIE4MatrixFormat.Int8, None
    )
    assert Attribute.parse(str(fmt)) == fmt
    print(fmt)
    kind = AttrBuilder.get("aievec.AIEVec_AIE4ReduceKindAttr")(AIE4ReduceKind.Max, None)
    assert Attribute.parse(str(kind)) == kind
    print(kind)
    return None


# CHECK-LABEL: TEST: test_op_builders
# CHECK: aievec.aie4.matrix_region int8 %{{.*}} : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
# CHECK:   aievec.aie4.mmac int8 %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
# CHECK: aievec.aie4.reduce <add> %{{.*}} : vector<32xi32> -> i32
# CHECK: ROUND TRIP OK
@construct_and_print_module
def test_op_builders(module):
    i8, i32 = T.i8(), T.i32()

    @func(emit=True)
    def enums(a: T.vector(64, i8), b: T.vector(64, 64, i8), v: T.vector(32, i32)):
        region = aievec.AIE4MatrixRegionOp(
            [T.vector(64, i32)], b, [], AIE4MatrixFormat.Int8, rhs_signed=True
        )
        with InsertionPoint(region.body.blocks.append()):
            row = aievec.aie4_mmac(
                T.vector(64, i32),
                a,
                AIE4MatrixFormat.Int8,
                lhs_signed=True,
                rhs_signed=True,
            )
            aievec.AIE4MatrixYieldOp([row])
        aievec.aie4_reduce(i32, v, AIE4ReduceKind.Add)

    assert module.operation.verify()
    print(module)
    assert str(Module.parse(str(module))) == str(module)
    print("ROUND TRIP OK")
    return None
