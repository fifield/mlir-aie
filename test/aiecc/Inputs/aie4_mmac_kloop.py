#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build a bf16 design with the MMAC region inside an scf.for K loop.

C[M x 64] += sum over KT tiles of A_k[M x 32] * B_k[32 x 64]. Each iteration
loads A_k and B_k^T (column-major B) from buffers, and aievec.aie4.matmul
stages B_k inside the loop body. Used by aie4-mmac-codegen.test to check
staging/commit/MMAC order across the loop back edge.
"""

import argparse

from aie.dialects import aievec, arith, scf, ub, vector
from aie.dialects.aie import (
    AIEDevice,
    Core,
    Device,
    ObjectFifoPort,
    buffer,
    dma_bd,
    end,
    object_fifo,
    tile,
)
from aie.dialects.aiex import (
    dma_await_task,
    dma_configure_task_for,
    dma_start_task,
    runtime_sequence,
)
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.ir import Block, InsertionPoint


def build_module(m_rows, k_tiles, runtime=False):
    a_len, b_len, c_len = k_tiles * m_rows * 32, k_tiles * 64 * 32, m_rows * 64
    with mlir_mod_ctx() as ctx:
        dev = Device(AIEDevice.npu3)
        with InsertionPoint(Block.create_at_start(dev.body_region)):
            if runtime:
                shim = tile(0, 0)
            t = tile(0, 2)
            if runtime:
                in0 = object_fifo("in0", shim, t, 1, T.memref(a_len + b_len, T.bf16()))
                in1 = object_fifo("in1", shim, t, 1, T.memref(c_len, T.f32()))
                out = object_fifo("out", t, shim, 1, T.memref(c_len, T.f32()))
            else:
                bA = buffer(t, T.memref(a_len, T.bf16()), name="bA")
                bB = buffer(t, T.memref(b_len, T.bf16()), name="bB")
                bC = buffer(t, T.memref(c_len, T.f32()), name="bC")
            core = Core(t)
            with InsertionPoint(Block.create_at_start(core.body)):
                if runtime:
                    bA = bB = in0.acquire(ObjectFifoPort.Consume, 1)
                    bC = in1.acquire(ObjectFifoPort.Consume, 1)
                    bOut = out.acquire(ObjectFifoPort.Produce, 1)
                else:
                    bOut = bC
                c0 = arith.constant(T.index(), 0)
                c1 = arith.constant(T.index(), 1)
                cK = arith.constant(T.index(), k_tiles)
                cA = arith.constant(T.index(), m_rows * 32)
                cB = arith.constant(T.index(), 2048)
                if runtime:
                    b_offset = arith.constant(T.index(), a_len)
                row_offsets = [arith.constant(T.index(), i * 32) for i in range(64)]
                c_offsets = [arith.constant(T.index(), r * 64) for r in range(m_rows)]

                acc = ub.poison(T.vector(m_rows, 64, T.f32()))
                for r, offset in enumerate(c_offsets):
                    row = vector.load(T.vector(64, T.f32()), bC, [offset])
                    acc = vector.insert(row, acc, [], [r])

                def load(buf, base, rows):
                    value = ub.poison(T.vector(rows, 32, T.bf16()))
                    for r in range(rows):
                        offset = arith.addi(base, row_offsets[r])
                        row = vector.load(T.vector(32, T.bf16()), buf, [offset])
                        value = vector.insert(row, value, [], [r])
                    return value

                loop = scf.ForOp(c0, cK, c1, iter_args=[acc])
                with InsertionPoint(loop.body):
                    abase = arith.muli(loop.induction_variable, cA)
                    bbase = arith.muli(loop.induction_variable, cB)
                    if runtime:
                        bbase = arith.addi(bbase, b_offset)
                    a = load(bA, abase, m_rows)
                    bt = load(bB, bbase, 64)
                    b = vector.transpose(T.vector(32, 64, T.bf16()), bt, [1, 0])
                    result = aievec.aie4_matmul(a, b, loop.inner_iter_args[0])
                    scf.yield_([result])

                for r, offset in enumerate(c_offsets):
                    row = vector.extract(loop.result, [], [r])
                    vector.store(row, bOut, [offset])
                if runtime:
                    in0.release(ObjectFifoPort.Consume, 1)
                    in1.release(ObjectFifoPort.Consume, 1)
                    out.release(ObjectFifoPort.Produce, 1)
                end()
            if runtime:

                @runtime_sequence(in0.datatype, in1.datatype, out.datatype)
                def sequence(a, c, result):
                    tasks = []
                    for fifo, mem, length in (
                        (in0, a, a_len + b_len),
                        (in1, c, c_len),
                        (out, result, c_len),
                    ):
                        task = dma_configure_task_for(fifo, issue_token=True)
                        with InsertionPoint(Block.create_at_start(task.body)):
                            dma_bd(
                                mem,
                                offset=0,
                                transfer_len=length,
                                sizes=[1, 1, 1, length],
                                strides=[0, 0, 0, 1],
                            )
                            end()
                        tasks.append(task)
                    dma_start_task(*tasks)
                    dma_await_task(*tasks)

            end()
        ctx.module.operation.verify()
        return ctx.module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--k-tiles", type=int, default=4)
    parser.add_argument("--runtime", action="store_true", help="add host DMA transfers")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    module = build_module(args.m, args.k_tiles, args.runtime)
    with open(args.output, "w") as out:
        module.operation.print(file=out)
