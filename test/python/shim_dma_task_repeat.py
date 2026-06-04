# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# shim_dma_single_bd_task supports an outermost (iteration) dimension only as a
# pure (zero-stride) repeat, which lowers to a queue-level repeat_count. The AIE4
# shim hardware itself supports an advancing iteration, but through this
# objectfifo helper a strided outer dimension lowers to a self-repeating task
# that does not compose with the objectfifo's object/token model (the repeats
# re-read the first block instead of advancing). So a STRIDED outermost dimension
# is rejected here; a genuinely advancing iteration is available through the raw
# dma_configure_task / aie.dma_bd path.

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *

# Zero-stride outermost dimension (size 5, stride 0): a pure repeat, lowered to a
# queue repeat_count of 4 with the BD iteration zeroed.
with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu1)
    def dev():
        S = tile(0, 0)
        M = tile(0, 2)
        nty = T.memref(8, T.i32())
        Nty = T.memref(256, T.i32())
        of_b = object_fifo("b", S, M, 2, nty)

        @runtime_sequence(Nty)
        def seq(A):
            # CHECK: aiex.dma_configure_task_for @b {
            # CHECK: aie.dma_bd(%arg0 : memref<256xi32>, 0, 8, [<size = 5, stride = 0>,
            # CHECK: } {repeat_count = 4 : i32}
            shim_dma_single_bd_task(
                of_b, A, sizes=[5, 1, 1, 1, 8], strides=[0, 0, 0, 0, 1]
            )

    print(ctx.module)

# Strided outermost dimension (size 5, stride 48): rejected with a clear error.
# CHECK: REJECTED: {{.*}}strided outermost
try:
    with mlir_mod_ctx() as ctx2:

        @device(AIEDevice.npu1)
        def dev2():
            S = tile(0, 0)
            M = tile(0, 2)
            nty = T.memref(8, T.i32())
            Nty = T.memref(256, T.i32())
            of_a = object_fifo("a", S, M, 2, nty)

            @runtime_sequence(Nty)
            def seq(A):
                shim_dma_single_bd_task(
                    of_a, A, sizes=[5, 3, 2, 2, 4], strides=[48, 16, 8, 4, 1]
                )

    print("NO ERROR")
except ValueError as e:
    print("REJECTED:", e)
