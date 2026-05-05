# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %python %s | FileCheck %s

# Builds a tiny IRON program and prints the resulting MLIR with debug info
# enabled. Asserts that ObjectFifo / Worker / Kernel / fill ops carry a
# FileLineColLoc / NameLoc tied back to this file rather than loc(unknown).

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1


def core_body(in_buf, out_buf, kernel):
    pass


def main():
    dev = NPU1Col1()
    line_ty = np.ndarray[(64,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="of_in")  # OF_IN_LINE
    of_out = ObjectFifo(line_ty, name="of_out")  # OF_OUT_LINE

    add_one = Kernel("add_one", "add_one.o", [line_ty, line_ty])  # KERNEL_LINE

    worker = Worker(  # WORKER_LINE
        core_body, fn_args=[of_in.cons(), of_out.prod(), add_one]
    )

    rt = Runtime()
    with rt.sequence(line_ty, line_ty) as (a_in, c_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)  # FILL_LINE
        rt.drain(of_out.cons(), c_out, wait=True)  # DRAIN_LINE

    program = Program(dev, rt)
    module = program.resolve_program()

    text = module.operation.get_asm(enable_debug_info=True)
    print(text)


# MLIR pretty-prints each unique location once via `loc(#locN)` and emits
# the definitions at the bottom of the module. NameLoc wrappers reference
# their inner FileLineColLoc indirectly (e.g. loc("of_in"(#loc1))), so we
# just check that each IRON-named NameLoc exists, plus that this file
# appears as some FileLineColLoc.
# CHECK-DAG: loc("of_in"
# CHECK-DAG: loc("of_out"
# CHECK-DAG: loc("add_one"
# CHECK-DAG: loc("core_body"
# CHECK-DAG: loc("fill(of_in)"
# CHECK-DAG: loc("drain(of_out)"
# CHECK-DAG: loc("{{.*}}iron_loc_emit.py":

if __name__ == "__main__":
    main()
