# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Coverage for the aie dialect Python bindings building a 4D (D0..D3) static
# shim buffer descriptor. A four-entry dimensions list on a static aie.dma_bd is
# only legal on an AIE4 (npu3) shim tile -- DMABDOp::verify caps other tiles at
# three -- so this exercises both the dma_bd dimensions binding and that the
# resulting IR verifies on an AIE4 shim. The four entries map to the four data
# dimensions d0..d3 (outermost-first [d3, d2, d1, d0]); there is no iteration
# entry on the static path.

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu3)
    def device_body():
        shim = tile(0, 0)
        ext = external_buffer(np.ndarray[(256,), np.dtype[np.int32]])
        prod = lock(shim, init=1)
        cons = lock(shim, init=0)

        # CHECK: aie.shim_dma
        # CHECK: aie.dma_bd(%{{.+}} : memref<256xi32>, 0, 256, [<size = 4, stride = 64>, <size = 4, stride = 4>, <size = 4, stride = 16>, <size = 4, stride = 1>])
        @shim_mem(shim)
        def sd(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(prod, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(
                    ext,
                    offset=0,
                    len=256,
                    dimensions=[(4, 64), (4, 4), (4, 16), (4, 1)],
                )
                use_lock(cons, LockAction.Release, value=1)
                EndOp()
            with block[2]:
                EndOp()

    assert ctx.module.operation.verify() == True
    print(ctx.module)
