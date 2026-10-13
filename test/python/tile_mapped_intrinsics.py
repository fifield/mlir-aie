# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | aie-opt --canonicalize --cse | FileCheck %s

from aie.dialects import aie, arith, func, xllvm
from aie.extras.context import mlir_mod_ctx
from aie.ir import InsertionPoint, IntegerType, Module

with mlir_mod_ctx() as ctx:
    i32 = IntegerType.get_signless(32)
    for arch, register in (("aie2", 0x36060), ("aie2p", 0x36060), ("aie4", 0xA1060)):
        f = func.FuncOp(arch, ([i32, i32], [i32]))
        with InsertionPoint(f.add_entry_block()):
            value, address = f.arguments
            # Constant offsets fold to core-visible addresses. Dynamic ones
            # retain the offset calculation.
            xllvm.sched_barrier(arch=arch)
            xllvm.write_tm(value, register, arch=arch)
            xllvm.sched_barrier(arch=arch)
            xllvm.read_tm(register, arch=arch)  # unused, but observable
            xllvm.sched_barrier(arch=arch)
            xllvm.write_tm(3, address, arch=arch)
            xllvm.sched_barrier(arch=arch)
            result = xllvm.read_tm(address, arch=arch)
            xllvm.sched_barrier(arch=arch)
            func.ReturnOp([result])

    # Invalid helper arguments fail before emitting an access.
    f = func.FuncOp("bad_arguments", ([], []))
    with InsertionPoint(f.add_entry_block()):
        # Misaligned/out-of-range offsets, addresses outside the processor-bus
        # window (AIE4 0x98000 is Peano's aperture base, but not a register;
        # 0x58000 would map to address 0), a full array address with row bits,
        # and AIE2 addresses already translated into the aperture.
        for arch, address in (
            ("aie4", -4),
            ("aie4", 3),
            ("aie4", 1 << 32),
            ("aie4", 0x9FFFC),
            ("aie4", 0x58000),
            ("aie4", 0xB5000),
            ("aie4", 0x2A1060),
            ("aie2", 0x0),
            ("aie2", 0x80000),
            ("aie2p", 0xB6060),
        ):
            try:
                xllvm.read_tm(address, arch=arch)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {arch} register {address:#x}")
        try:
            xllvm.sched_barrier(arch="aie1")
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an unsupported target")
        bad = arith.constant(IntegerType.get_signless(64), 0)
        try:
            xllvm.read_tm(bad, arch="aie4")
        except TypeError:
            pass
        else:
            raise AssertionError("accepted an i64 address")
        try:
            xllvm.sched_barrier()
        except ValueError:
            pass
        else:
            raise AssertionError("inferred an arch outside aie.device")
        func.ReturnOp([])

    # Inside aie.device, arch defaults to the device target and must match it.
    # Single-result ops are accepted wherever an i32 value is.
    @aie.device(aie.AIEDevice.npu3)
    def device():
        f = func.FuncOp("in_device", ([], [i32]))
        with InsertionPoint(f.add_entry_block()):
            xllvm.write_tm(arith.ConstantOp(i32, 1), 0xA1060)
            result = xllvm.read_tm(0xA1060, arch="aie4")
            try:
                xllvm.read_tm(0x36060, arch="aie2")
            except ValueError:
                pass
            else:
                raise AssertionError("accepted an arch that mismatches npu3")
            func.ReturnOp([result])

    assert ctx.module.operation.verify()
    # Registered dialect and custom assembly must round-trip in Python too.
    assert Module.parse(str(ctx.module)).operation.verify()
    print(ctx.module)

# CHECK-LABEL: func.func @aie2(
# CHECK-DAG: arith.constant 745568 : i32
# CHECK: xllvm.intr.aie2.sched.barrier
# CHECK: xllvm.intr.aie2.write.tm
# CHECK: xllvm.intr.aie2.read.tm
# CHECK: xllvm.intr.aie2.write.tm
# CHECK: xllvm.intr.aie2.read.tm

# CHECK-LABEL: func.func @aie2p(
# CHECK-DAG: arith.constant 745568 : i32
# CHECK: xllvm.intr.aie2p.sched.barrier
# CHECK: xllvm.intr.aie2p.write.tm
# CHECK: xllvm.intr.aie2p.read.tm
# CHECK: xllvm.intr.aie2p.write.tm
# CHECK: xllvm.intr.aie2p.read.tm

# CHECK-LABEL: func.func @aie4(
# CHECK-DAG: arith.constant 299104 : i32
# CHECK: xllvm.intr.aie4.sched.barrier
# CHECK: xllvm.intr.aie4.write.tm
# CHECK: xllvm.intr.aie4.read.tm
# CHECK: xllvm.intr.aie4.write.tm
# CHECK: xllvm.intr.aie4.read.tm

# CHECK-LABEL: func.func @in_device(
# CHECK: xllvm.intr.aie4.write.tm
# CHECK: xllvm.intr.aie4.read.tm
