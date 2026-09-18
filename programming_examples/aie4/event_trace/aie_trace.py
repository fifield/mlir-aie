# aie_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
# ===-----------------------------------------------------------------------===#
#
# AIE4 event trace example — Python equivalent of aie_trace.mlir.
# Vector-scalar multiply with trace collection on core, mem, and shim tiles.
#
# Usage:
#   python3 aie_trace.py > aie_trace_from_py.mlir
#
# ===-----------------------------------------------------------------------===#

import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def build_aie_trace():
    tensor_size = 4096
    tile_size = 1024
    num_sub_vectors = 4

    @device(AIEDevice.npu3)
    def device_body():
        tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
        scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]

        # External kernel function declaration
        scale = external_func(
            "vector_scalar_mul",
            inputs=[tile_ty, tile_ty, scalar_ty, np.int32],
            link_with="project/scale.o",
        )

        # Tile declarations
        shim_noc_tile_0_0 = tile(0, 0)
        tile_0_2 = tile(0, 2)

        # ObjectFIFOs for data movement
        of_in = object_fifo("in", shim_noc_tile_0_0, tile_0_2, 2, tile_ty)
        of_factor = object_fifo("infactor", shim_noc_tile_0_0, tile_0_2, 2, scalar_ty)
        of_out = object_fifo("out", tile_0_2, shim_noc_tile_0_0, 2, tile_ty)

        # Core computation
        @core(tile_0_2)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_factor = of_factor.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(num_sub_vectors):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    scale(elem_in, elem_out, elem_factor, tile_size)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_factor.release(ObjectFifoPort.Consume, 1)

        # ==================================================================
        # TRACE CONFIGURATION
        # ==================================================================

        # Trace configuration for compute tile (0,2) - core events (Trace0 unit)
        @trace(tile_0_2, "core_trace")
        def core_trace_body():
            trace_mode(TraceMode.EventTime)
            trace_packet(1, TracePacketType.Core)
            trace_event("INSTR_EVENT_0")
            trace_event("INSTR_EVENT_1")
            trace_event("INSTR_VECTOR")
            trace_event("MEMORY_STALL")
            trace_event("STREAM_STALL")
            trace_event("LOCK_STALL")
            trace_event("DMA_S2MM_0_RUNNING")
            trace_event("DMA_MM2S_0_RUNNING")
            trace_start(broadcast=3)
            trace_stop(broadcast=4)

        # Trace configuration for compute tile (0,2) - DMA/memory events (Trace1 unit)
        @trace(tile_0_2, "mem_trace")
        def mem_trace_body():
            trace_mode(TraceMode.EventTime)
            trace_packet(3, TracePacketType.Mem)
            trace_event("DMA_S2MM_0_START_TASK")
            trace_event("DMA_S2MM_1_START_TASK")
            trace_event("DMA_MM2S_0_START_TASK")
            trace_event("DMA_S2MM_0_FINISHED_TASK")
            trace_event("DMA_S2MM_1_FINISHED_TASK")
            trace_event("DMA_MM2S_0_FINISHED_TASK")
            trace_event("DMA_S2MM_0_STREAM_STARVATION")
            trace_event("DMA_S2MM_1_STREAM_STARVATION")
            trace_start(broadcast=3)
            trace_stop(broadcast=4)

        # Trace configuration for shim tile (0,0)
        @trace(shim_noc_tile_0_0, "shim_trace")
        def shim_trace_body():
            trace_packet(2, TracePacketType.ShimTile)
            trace_event("DMA_S2MM_0_START_TASK")
            trace_event("DMA_S2MM_1_START_TASK")
            trace_event("DMA_MM2S_0_START_TASK")
            trace_event("DMA_S2MM_0_FINISHED_TASK")
            trace_event("DMA_S2MM_1_FINISHED_TASK")
            trace_event("DMA_MM2S_0_FINISHED_TASK")
            trace_event("DMA_S2MM_0_STREAM_STARVATION")
            trace_event("DMA_S2MM_1_STREAM_STARVATION")
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        # ==================================================================
        # RUNTIME SEQUENCE WITH TRACE ACTIVATION
        # ==================================================================

        trace_size = 8192  # bytes

        # The trace buffer argument is appended to the runtime sequence by
        # trace_host_config lowering (reuse_output_buffer defaults to False),
        # so it is not declared here.
        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):
            # Configure trace output buffer
            trace_host_config(trace_size)

            # Declarative trace configuration lowered by passes
            trace_start_config("core_trace")
            trace_start_config("mem_trace")
            trace_start_config("shim_trace")

            # Data transfer configuration
            in_task = shim_dma_single_bd_task(
                of_in, A, sizes=[1, 1, 1, tensor_size], issue_token=True
            )
            factor_task = shim_dma_single_bd_task(
                of_factor, F, sizes=[1, 1, 1, 1], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, tensor_size], issue_token=True
            )

            dma_start_task(in_task, factor_task, out_task)
            dma_await_task(in_task, factor_task, out_task)


with mlir_mod_ctx() as ctx:
    build_aie_trace()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
