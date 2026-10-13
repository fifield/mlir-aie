# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | aie-opt -o /dev/null

"""Exercise core-programmed North/West cascade input flips with explicit traffic.

The runtime submits one word to each core and waits for the pivot output
before submitting the next. Each source sends exactly one cascade word per
input. Consuming that word drains the old direction before a flip, independently
of the compiler scheduling barriers.
"""

import sys
import numpy as np

from aie.dialects.aie import (
    AIEDevice,
    CascadeDir,
    ObjectFifoPort,
    configure_cascade,
    core,
    device,
    get_cascade,
    object_fifo,
    put_cascade,
    tile,
)
from aie.dialects.aiex import (
    dma_await_task,
    dma_start_task,
    npu_maskwrite32,
    runtime_sequence,
    shim_dma_single_bd_task,
)
from aie.dialects import arith, memref, vector
from aie.dialects.xllvm import read_tm, sched_barrier, write_tm
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_
from aie.ir import IndexType, IntegerType, VectorType
from aie.utils.regdb import AIEAddressDecoder

ITERATIONS = 8

# TM helpers and tile-relative runtime writes take register offsets.
core_registers = {
    reg["name"]: reg
    for reg in AIEAddressDecoder(arch="aie4").database["modules"]["core"]["registers"]
}
ACCUMULATOR_CONTROL = int(core_registers["Accumulator_Control"]["offset"], 0)
CORE_PROCESSOR_BUS = int(core_registers["Core_Processor_Bus"]["offset"], 0)

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu3)
    def design():
        shim = tile(0, 0)
        pivot = tile(1, 2)
        north = tile(1, 3)
        west = tile(0, 2)
        word = np.ndarray[(16,), np.dtype[np.int32]]
        record = np.ndarray[(64,), np.dtype[np.int32]]
        start = object_fifo("start", shim, [pivot, north, west], 2, word)
        out = object_fifo("out", pivot, shim, 2, record)
        configure_cascade(north, CascadeDir.North, CascadeDir.South)
        configure_cascade(west, CascadeDir.West, CascadeDir.East)
        configure_cascade(pivot, CascadeDir.North, CascadeDir.South)
        i32 = IntegerType.get_signless(32)
        v16 = VectorType.get([16], i32)

        def idx(n):
            return arith.constant(IndexType.get(), n)

        def source(t, offset):
            @core(t)
            def source_core():
                for _ in range_(sys.maxsize):
                    payload = start.acquire(ObjectFifoPort.Consume, 1)
                    v = vector.load(v16, payload, [idx(0)])
                    addend = vector.broadcast(v16, arith.constant(i32, offset))
                    put_cascade(arith.addi(v, addend))
                    start.release(ObjectFifoPort.Consume, 1)

        source(north, 0)
        source(west, 1000)

        def direction(value):
            sched_barrier(arch="aie4")
            write_tm(value, ACCUMULATOR_CONTROL, arch="aie4")
            sched_barrier(arch="aie4")
            got = read_tm(ACCUMULATOR_CONTROL, arch="aie4")
            sched_barrier(arch="aie4")
            return got

        @core(pivot)
        def pivot_core():
            for _ in range_(sys.maxsize):
                # Host enables the processor bus before submitting this input.
                start.acquire(ObjectFifoPort.Consume, 1)
                dst = out.acquire(ObjectFifoPort.Produce, 1)
                north_mode = direction(0)
                n = get_cascade(v16)
                vector.store(n, dst, [idx(0)])

                # The only requested North word has now been consumed.
                west_mode = direction(3)
                w = get_cascade(v16)
                vector.store(w, dst, [idx(16)])
                vector.store(arith.addi(n, w), dst, [idx(32)])

                # No more West traffic is outstanding. Restore before the
                # next iteration/host launch and check the register readback.
                restored = direction(0)
                memref.store(north_mode, dst, [idx(48)])
                memref.store(west_mode, dst, [idx(49)])
                memref.store(restored, dst, [idx(50)])
                out.release(ObjectFifoPort.Produce, 1)
                start.release(ObjectFifoPort.Consume, 1)

        @runtime_sequence(
            np.ndarray[(16 * ITERATIONS,), np.dtype[np.int32]],
            np.ndarray[(64 * ITERATIONS,), np.dtype[np.int32]],
        )
        def sequence(src, dst):
            npu_maskwrite32(
                column=1, row=2, address=CORE_PROCESSOR_BUS, value=1, mask=1
            )
            # Waiting for the output gates the next input to all three cores.
            for i in range(ITERATIONS):
                a = shim_dma_single_bd_task(
                    start, src, offset=i * 16, sizes=[1, 1, 1, 16], issue_token=True
                )
                b = shim_dma_single_bd_task(
                    out, dst, offset=i * 64, sizes=[1, 1, 1, 64], issue_token=True
                )
                dma_start_task(a, b)
                dma_await_task(a, b)

    assert ctx.module.operation.verify()
    print(ctx.module)
