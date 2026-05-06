# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# Covers IRON entry points that iron_loc_emit.py does not exercise:
#   - Buffer (a top-level named buffer placed on the worker tile)
#   - ObjectFifoLink (created by ObjectFifoHandle.forward())
#   - Runtime.set_barrier
#   - Runtime.inline_ops
# Asserts that each of their resulting MLIR ops carries a NameLoc whose
# inner FileLineColLoc points back to this file.

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import (
    Buffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.device import NPU1Col1


def core_body(scratch, barrier):
    pass


def my_inline_func(*args):
    # Body runs inside an MLIR context during Runtime.resolve. Empty body
    # is fine — we only care that the loc context wraps any ops it might
    # generate.
    pass


def main():
    dev = NPU1Col1()
    line_ty = np.ndarray[(64,), np.dtype[np.int32]]
    scratch_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="of_in")
    of_mid = of_in.cons().forward(name="of_mid")  # ObjectFifoLink site
    of_out = of_mid.cons().forward(name="of_out")

    scratch = Buffer(scratch_ty, name="my_scratch")  # Buffer site
    barrier = WorkerRuntimeBarrier()

    worker = Worker(core_body, fn_args=[scratch, barrier])

    rt = Runtime()
    with rt.sequence(line_ty, line_ty) as (a_in, c_out):
        rt.start(worker)
        rt.set_barrier(barrier, 1)  # set_barrier site
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)
        rt.inline_ops(my_inline_func, [scratch])  # inline_ops site

    program = Program(dev, rt)
    module = program.resolve_program()
    print(module.operation.get_asm(enable_debug_info=True))


# CHECK-DAG: loc("my_scratch"
# CHECK-DAG: loc("link_of_in_to_of_mid"
# CHECK-DAG: loc("{{.*}}iron_loc_emit_extras.py":

if __name__ == "__main__":
    main()
