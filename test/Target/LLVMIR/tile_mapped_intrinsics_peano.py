# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: peano
# RUN: %python %s aie2 > tm_aie2.mlir
# RUN: %aiecc --get-core-elfs --tmpdir tm_aie2.prj --output-dir tm_aie2.out tm_aie2.mlir
# RUN: llvm-objdump -dr tm_aie2.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s
# RUN: llvm-objdump -dr tm_aie2.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s --check-prefix=BUNDLE
# RUN: %python %s aie2p > tm_aie2p.mlir
# RUN: %aiecc --get-core-elfs --tmpdir tm_aie2p.prj --output-dir tm_aie2p.out tm_aie2p.mlir
# RUN: llvm-objdump -dr tm_aie2p.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s
# RUN: llvm-objdump -dr tm_aie2p.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s --check-prefix=BUNDLE
# RUN: %python %s aie4 > tm_aie4.mlir
# RUN: %aiecc --get-core-elfs --tmpdir tm_aie4.prj --output-dir tm_aie4.out tm_aie4.mlir
# RUN: llvm-objdump -dr tm_aie4.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s
# RUN: llvm-objdump -dr tm_aie4.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s --check-prefix=BUNDLE

# Actual instructions, with no unresolved intrinsic/helper relocations.
# The barriers keep each store and read on different bundle lines. FileCheck
# regexes do not cross lines, so this rejects any bundle containing both.
# BUNDLE-NOT: {{st\.tm.*lda\.tm|lda\.tm.*st\.tm}}
# CHECK-NOT: R_AIE{{.*}}llvm.
# CHECK: st.tm
# CHECK-NOT: R_AIE{{.*}}llvm.
# CHECK: lda.tm
# CHECK-NOT: R_AIE{{.*}}llvm.

"""Core-side TM access under a loop and conditional, through a device helper.

Compile-only: processor-bus setup and resource synchronization belong to the
runtime, so this design must not be dispatched as-is.
"""

import sys

from aie.dialects import aie, arith, func, memref, scf, xllvm
from aie.extras.context import mlir_mod_ctx
from aie.ir import InsertionPoint, IntegerType, MemRefType

arch = sys.argv[1]
dev = {
    "aie2": aie.AIEDevice.npu1_1col,
    "aie2p": aie.AIEDevice.npu2_1col,
    "aie4": aie.AIEDevice.npu3,
}[arch]
register = 0xA1060 if arch == "aie4" else 0x36060

with mlir_mod_ctx() as ctx:

    @aie.device(dev)
    def device():
        tile = aie.tile(0, 2)
        i32 = IntegerType.get_signless(32)
        result = aie.buffer(tile, MemRefType.get([4], i32), name="result")
        helper = func.FuncOp("set_direction", ([i32], [i32]), visibility="private")
        with InsertionPoint(helper.add_entry_block()):
            xllvm.sched_barrier(arch=arch)
            xllvm.write_tm(helper.arguments[0], register, arch=arch)
            xllvm.sched_barrier(arch=arch)
            readback = xllvm.read_tm(register, arch=arch)
            xllvm.sched_barrier(arch=arch)
            func.ReturnOp([readback])

        @aie.core(tile)
        def core():
            from aie.ir import IndexType

            idx = IndexType.get()
            c0 = arith.constant(idx, 0)
            c1 = arith.constant(idx, 1)
            c4 = arith.constant(idx, 4)
            loop = scf.ForOp(c0, c4, c1)
            with InsertionPoint(loop.body):
                value = arith.index_cast(i32, loop.induction_variable)
                pred = arith.cmpi(arith.CmpIPredicate.eq, value, arith.constant(i32, 0))
                branch = scf.IfOp(pred, has_else=True)
                with InsertionPoint(branch.then_block):
                    v = func.CallOp(helper, [value]).result
                    xllvm.sched_barrier(arch=arch)
                    xllvm.write_tm(v, register, arch=arch)
                    xllvm.sched_barrier(arch=arch)
                    memref.store(v, result, [loop.induction_variable])
                    scf.YieldOp([])
                with InsertionPoint(branch.else_block):
                    v = func.CallOp(helper, [arith.constant(i32, 3)]).result
                    xllvm.sched_barrier(arch=arch)
                    xllvm.write_tm(v, register, arch=arch)
                    xllvm.sched_barrier(arch=arch)
                    memref.store(v, result, [loop.induction_variable])
                    scf.YieldOp([])
                scf.YieldOp([])

    assert ctx.module.operation.verify()
    print(ctx.module)
