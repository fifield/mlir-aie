#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Expand matrix_region before SCF lowering: staging and commit must precede
# the row loop, and the resulting core must preserve MMAC schedule order.
# REQUIRES: peano
# RUN: %python %s -o %t.mlir
# RUN: %aiecc --dump-intermediates --get-core-elfs --tmpdir %t.prj --output-dir %t.out %t.mlir
# RUN: FileCheck %s --check-prefix=IR --implicit-check-not='aievec.aie4.matrix_region' --implicit-check-not='aievec.aie4.mmac ' --input-file %t.prj/input_with_addresses.mlir
# RUN: FileCheck %s --check-prefix=CANCEL --input-file %t.prj/lowered_main_core_0_2.mlir
# RUN: llvm-objdump -d %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
# RUN: %python %S/aie4-mmac-schedule-checker.py %t.dis --mmacs-per-tile 3 | FileCheck %s --check-prefix=ORDER

# IR-COUNT-32: aievec.aie4.stage bf16
# IR-NOT: aievec.aie4.stage
# IR: aievec.aie4.commit bf16
# IR: cf.br ^[[HEAD:bb[0-9]+]]
# IR: ^[[HEAD]]
# IR: cf.cond_br
# IR: aievec.aie4.mmac_active bf16
# IR: cf.br ^[[HEAD]]
# Count the peeled first MMAC plus two simulated loop-body iterations.
# ORDER: ORDER OK: 1 tiles, 32 inserts, 3 MMACs, no accumulator spills (loop body checked twice)

# CANCEL-NOT: vector.transpose
# CANCEL-NOT: vector<2048xbf16>
# CANCEL-COUNT-32: insert.staging.bf16
# CANCEL-NOT: vector.transpose
# CANCEL-NOT: vector<2048xbf16>

import argparse

from aie.dialects import arith, scf, ub, vector
from aie.dialects.aie import AIEDevice, buffer, core, device, tile
from aie.extras import types as T
from aie.helpers.dialects.aievec_aie4 import BF16, matrix_region, mmac
from aie.ir import Context, InsertionPoint, Location, Module


def emit(m_rows):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @device(AIEDevice.npu3)
            def device_body():
                compute = tile(0, 2)
                a_mem = buffer(compute, T.memref(m_rows * 32, T.bf16()), name="bA")
                b_mem = buffer(compute, T.memref(64 * 32, T.bf16()), name="bB")
                c_mem = buffer(compute, T.memref(m_rows * 64, T.f32()), name="bC")

                @core(compute)
                def core_body():
                    def cidx(value):
                        return arith.constant(T.index(), value)

                    columns = ub.poison(T.vector(64, 32, T.bf16()))
                    for i in range(64):
                        row = vector.load(T.vector(32, T.bf16()), b_mem, [cidx(i * 32)])
                        columns = vector.insert(row, columns, [], [i])
                    b = vector.transpose(T.vector(32, 64, T.bf16()), columns, [1, 0])

                    @matrix_region(b, [], BF16)
                    def region():
                        loop = scf.ForOp(cidx(0), cidx(m_rows), cidx(1))
                        with InsertionPoint(loop.body):
                            i = loop.induction_variable
                            ao = arith.muli(i, cidx(32))
                            co = arith.muli(i, cidx(64))
                            a = vector.load(T.vector(32, T.bf16()), a_mem, [ao])
                            c = vector.load(T.vector(64, T.f32()), c_mem, [co])
                            vector.store(mmac(BF16, a, c), c_mem, [co])
                            scf.YieldOp([])

        assert module.operation.verify()
        return str(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emit a binding-built bf16 row-loop design"
    )
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    with open(args.output, "w") as out:
        out.write(emit(args.m))
