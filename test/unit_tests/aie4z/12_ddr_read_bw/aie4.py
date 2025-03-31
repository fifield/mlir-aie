# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: mesimulator

# RUN: %python %s > aie4_py.mlir
# RUN: aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --aiesim aie4_py.mlir %test_utils_flags %S/test.cpp
# RUN: aie-translate -aie-npu-to-binary --aie-sequence-name=configure -o transaction.bin aie4_py.mlir.prj/txn.mlir
# RUN: aie4_py.mlir.prj/mesim.sh | FileCheck --check-prefix=MESIM %s

# MESIM: PASS!

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx


def generate_module():
    @device(AIEDevice.npu3z)
    def d():
        N = 64 * 1024
        n = 1024
        tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

        num_cols = 3
        num_shim_channels = 8

        for col in range(0, num_cols):
            ShimTile = tile(col, 0)
            CoreTile = [tile(col, i + 2) for i in range(4)]

            for core in CoreTile:
                ping_in_buffer = [aie.buffer(core, tile_ty) for _ in range(0, 2)]
                pong_in_buffer = [aie.buffer(core, tile_ty) for _ in range(0, 2)]

                @mem(core)
                def s(block):
                    with block[0]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        dma_bd(ping_in_buffer[0])
                        next_bd(block[2])
                    with block[2]:
                        dma_bd(pong_in_buffer[0])
                        next_bd(block[1])
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
                    with block[4]:
                        dma_bd(ping_in_buffer[1])
                        next_bd(block[5])
                    with block[5]:
                        dma_bd(pong_in_buffer[1])
                        next_bd(block[4])
                    with block[6]:
                        EndOp()

            shim_tile_input_write_lock = []
            shim_tile_input_read_lock = []
            ext_input_buffer = []
            for i in range(0, num_shim_channels):
                shim_tile_input_write_lock.append(lock(ShimTile, lock_id=i * 2, init=1))
                shim_tile_input_read_lock.append(
                    lock(ShimTile, lock_id=(i * 2) + 1, init=0)
                )
                ext_input_buffer.append(external_buffer(tensor_ty, address=0x80000000))

            for channel in range(num_shim_channels):
                flow(
                    ShimTile,
                    WireBundle.DMA,
                    channel,
                    CoreTile[channel // 2],
                    WireBundle.DMA,
                    channel % 2,
                )

            @shim_mem(ShimTile)
            def s(block):
                for i in range(0, num_shim_channels):
                    id = i * 2
                    with block[id]:
                        dma_start(
                            DMAChannelDir.MM2S,
                            i,
                            dest=block[id + 1],
                            chain=block[id + 2],
                        )
                    with block[id + 1]:
                        use_lock(
                            shim_tile_input_read_lock[i],
                            LockAction.AcquireGreaterEqual,
                            value=1,
                        )
                        dma_bd(ext_input_buffer[i], bd_id=i)
                        use_lock(
                            shim_tile_input_write_lock[i], LockAction.Release, value=1
                        )
                        next_bd(block[id + 1])
                with block[num_shim_channels * 2]:
                    EndOp()


with mlir_mod_ctx() as ctx:
    # generate mlir-aie module
    generate_module()
    print(ctx.module)
