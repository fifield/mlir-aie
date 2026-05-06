# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# Builds a tiny IRON program and verifies, two ways:
#  PRE  — the printed MLIR carries `loc("of_in"(this_file:line))`-style
#         NameLoc wrappers on every IRON entry-point op.
#  POST — running that module through `aie-objectFifo-stateful-transform`
#         preserves those locations on the synthesized buffer / lock /
#         DMA / mem ops; i.e., the user's Python source position survives
#         end-to-end through the front-end and a representative pass.

# RUN: %python %s | FileCheck %s --check-prefix=PRE
# RUN: %python %s | aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform --mlir-print-debuginfo | FileCheck %s --check-prefix=POST

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1


def core_body(in_buf, out_buf, kernel):
    pass


def main():
    dev = NPU1Col1()
    line_ty = np.ndarray[(64,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="of_in")
    of_out = ObjectFifo(line_ty, name="of_out")

    add_one = Kernel("add_one", "add_one.o", [line_ty, line_ty])

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), add_one])

    rt = Runtime()
    with rt.sequence(line_ty, line_ty) as (a_in, c_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    program = Program(dev, rt)
    module = program.resolve_program()

    text = module.operation.get_asm(enable_debug_info=True)
    print(text)


# IRON entry points each get a NameLoc whose inner FileLineColLoc points
# back to this source file.
# PRE-DAG: loc("of_in"
# PRE-DAG: loc("of_out"
# PRE-DAG: loc("add_one"
# PRE-DAG: loc("core_body"
# PRE-DAG: loc("fill(of_in)"
# PRE-DAG: loc("drain(of_out)"
# PRE-DAG: loc("{{.*}}iron_loc_emit.py":

# After --aie-place-tiles + --aie-objectFifo-stateful-transform expand the
# objectfifos into buffers / locks / DMA descriptors, those synthesized ops
# still carry the user's source position via the original NameLoc.
# POST-DAG: aie.buffer({{.*}}) {sym_name = "of_in_cons_buff_0"} : memref<64xi32>{{ +}}loc(#[[OF_IN:loc[0-9]*]])
# POST-DAG: aie.buffer({{.*}}) {sym_name = "of_out_buff_0"} : memref<64xi32>{{ +}}loc(#[[OF_OUT:loc[0-9]*]])
# POST-DAG: aie.lock({{.*}}) {{.*}}sym_name = "of_in_cons_prod_lock_0"{{.*}}loc(#[[OF_IN]])
# POST-DAG: aie.lock({{.*}}) {{.*}}sym_name = "of_out_prod_lock_0"{{.*}}loc(#[[OF_OUT]])
# POST-DAG: aie.dma_bd(%of_in_cons_buff_0 : memref<64xi32>, 0, 64) loc(#[[OF_IN]])
# POST-DAG: aie.dma_bd(%of_out_buff_0 : memref<64xi32>, 0, 64) loc(#[[OF_OUT]])
# POST-DAG: #[[OF_IN]] = loc("of_in"(
# POST-DAG: #[[OF_OUT]] = loc("of_out"(

if __name__ == "__main__":
    main()
