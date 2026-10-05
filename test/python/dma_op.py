# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
import numpy as np
from aie.dialects.aie import (
    AIEDevice,
    buffer,
    lock,
    WireBundle,
    flow,
    shim_dma_allocation,
    memtile_dma,
    DMAChannelDir,
    use_lock,
    LockAction,
    dma,
    dma_bd,
    tile,
    Device,
    end,
)
from aie.ir import InsertionPoint, Block

from util import construct_and_print_module

# CHECK: module {
# CHECK: aie.device(xcve2302) {
# CHECK: %[[SHIM:.*]] = aie.tile(2, 0)
# CHECK: %[[MEM:.*]] = aie.tile(2, 1)
# CHECK: %[[PROD_LOCK:.*]] = aie.lock(%[[MEM]], 0) {init = 1 : i32}
# CHECK: %[[CONS_LOCK:.*]] = aie.lock(%[[MEM]], 1) {init = 0 : i32}
# CHECK: %[[BUFFER:.*]] = aie.buffer(%[[MEM]]) {sym_name = "mem_A"} : memref<2x1xi16>
# CHECK: aie.flow(%[[MEM]], DMA : 0, %[[SHIM]], DMA : 0)
# CHECK: aie.shim_dma_allocation @mem_A_shim_alloc(%[[SHIM]], S2MM, 0)

# CHECK: aie.memtile_dma(%[[MEM]]) {
# CHECK: aie.dma(MM2S, 0)
# CHECK: aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual,
# CHECK: aie.dma_bd(%[[BUFFER]] : memref<2x1xi16> len = 2 sizes = [1, 1, 1, 1] strides = [1, 1, 1, 1])
# CHECK: aie.use_lock(%[[PROD_LOCK]], Release,


@construct_and_print_module
def objFifo_example():
    dev = Device(AIEDevice.xcve2302)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        # Use a shim-NoC column for memory DMA.
        shim_tile = tile(2, 0)
        mem_tile = tile(2, 1)

        prod_lock_mem_A = lock(mem_tile, lock_id=0, init=1)
        cons_lock_mem_A = lock(mem_tile, lock_id=1, init=0)
        tensor_ty_A = np.ndarray[(2, 1), np.dtype[np.int16]]
        buff_mem_A = buffer(tile=mem_tile, datatype=tensor_ty_A, name="mem_A")

        flow(mem_tile, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 0)

        shim_dma_allocation("mem_A_shim_alloc", shim_tile, DMAChannelDir.S2MM, 0)

        @memtile_dma(mem_tile)
        def memtile_dma_2_1():
            @dma(DMAChannelDir.MM2S, 0)
            def dma_in_A_to_compute():
                use_lock(cons_lock_mem_A, LockAction.AcquireGreaterEqual)
                dma_bd(
                    buff_mem_A,
                    transfer_len=2,
                    sizes=[1, 1, 1, 1],
                    strides=[1, 1, 1, 1],
                )
                use_lock(prod_lock_mem_A, LockAction.Release)

            end()

        end()
