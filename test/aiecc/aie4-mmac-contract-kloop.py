# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: peano

# RUN: %python %s > %t.mlir
# RUN: FileCheck %s --check-prefix=IN --input-file %t.mlir
# RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %t.mlir
# RUN: llvm-objdump -d %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
# RUN: %python %S/aie4-mmac-schedule-checker.py %t.dis --mmacs-per-tile 4,4 | FileCheck %s --check-prefix=ORDER
# RUN: FileCheck %s --check-prefix=ASM --input-file %t.dis --implicit-check-not=llvm.aie4

# IN: scf.for
# IN: arith.extf
# IN: vector.contract {fastmath = #arith.fastmath<afn>
# IN: scf.yield
# ORDER: ORDER OK: 2 tiles, 64 inserts, 8 MMACs, no accumulator spills (loop body checked twice)
# ASM-DAG: vmov{{[[:blank:]]+}}zb, za{{[[:blank:]]*(;|$)}}
# ASM-DAG: mmac.f{{[[:blank:]]+}}dm{{[ab][0-9]+}}, dm{{[ab][0-9]+}}, x{{[0-9]+}}, zb, r{{[0-9]+}}
# ASM-DAG: <.LBB0_{{[0-9]+}}>:

"""A bf16 K loop written as one vector.contract {fastmath<afn>} per K tile
over arith.extf operands, in programming_examples/aie4/matmul_bf16's layout
(A[ko][m][32], B[ko][n][32]): C[4 x 64] = A[4 x 128] * B[128 x 64]. The
Vector-to-AIEVec matcher converts the contraction inside the scf.for, giving
one 32-row insert group, one commit and four MMACs per iteration."""

from aie.dialects import arith, scf, ub, vector
from aie.dialects.aie import AIEDevice, Core, Device, buffer, end, tile
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.ir import Attribute, Block, InsertionPoint

M, K_TILES = 4, 4


def contraction(a, b, acc):
    op = vector.ContractionOp(
        T.vector(M, 64, T.f32()),
        a,
        b,
        acc,
        Attribute.parse(
            "[affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>,"
            " affine_map<(m, n, k) -> (m, n)>]"
        ),
        Attribute.parse(
            "[#vector.iterator_type<parallel>, #vector.iterator_type<parallel>,"
            " #vector.iterator_type<reduction>]"
        ),
    )
    op.operation.attributes["kind"] = Attribute.parse("#vector.kind<add>")
    op.operation.attributes["fastmath"] = Attribute.parse("#arith.fastmath<afn>")
    return op.result


with mlir_mod_ctx() as ctx:
    dev = Device(AIEDevice.npu3)
    with InsertionPoint(Block.create_at_start(dev.body_region)):
        t = tile(0, 2)
        bA = buffer(t, T.memref(K_TILES * M * 32, T.bf16()), name="bA")
        bB = buffer(t, T.memref(K_TILES * 64 * 32, T.bf16()), name="bB")
        bC = buffer(t, T.memref(M * 64, T.f32()), name="bC")
        core = Core(t)
        with InsertionPoint(Block.create_at_start(core.body)):
            c0 = arith.constant(T.index(), 0)
            c1 = arith.constant(T.index(), 1)
            cK = arith.constant(T.index(), K_TILES)
            cA = arith.constant(T.index(), M * 32)
            cB = arith.constant(T.index(), 64 * 32)
            row_offsets = [arith.constant(T.index(), i * 32) for i in range(64)]

            def load(buf, base, rows):
                value = ub.poison(T.vector(rows, 32, T.bf16()))
                for r in range(rows):
                    offset = arith.addi(base, row_offsets[r])
                    row = vector.load(T.vector(32, T.bf16()), buf, [offset])
                    value = vector.insert(row, value, [], [r])
                return value

            zero = arith.constant(
                T.vector(M, 64, T.f32()),
                Attribute.parse(f"dense<0.0> : vector<{M}x64xf32>"),
            )
            loop = scf.ForOp(c0, cK, c1, iter_args=[zero])
            with InsertionPoint(loop.body):
                a = load(bA, arith.muli(loop.induction_variable, cA), M)
                bt = load(bB, arith.muli(loop.induction_variable, cB), 64)
                b = vector.transpose(T.vector(32, 64, T.bf16()), bt, [1, 0])
                aw = arith.extf(T.vector(M, 32, T.f32()), a)
                bw = arith.extf(T.vector(32, 64, T.f32()), b)
                scf.yield_([contraction(aw, bw, loop.inner_iter_args[0])])

            for r in range(M):
                row = vector.extract(loop.result, [], [r])
                vector.store(row, bC, [arith.constant(T.index(), r * 64)])
            end()
        end()
    ctx.module.operation.verify()
    print(ctx.module)
