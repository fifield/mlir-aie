//===- design_b.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Design B: a row permutation of 1024 x i32 seen as 32 rows of 32, core-free.
//   shim (0, 0) MM2S 0 -> memtile (0, 1) S2MM 2 -> MM2S 2 -> shim S2MM 0
// Same shim channels and memtile input port (South:0) as design A, but the
// memtile writes the stream with S2MM 2 (connection South:0 -> DMA:2), not
// S2MM 0. MM2S 2 reads the rows as an 8 x 4 grid transposed:
//   out row 4 * i + k = in row i + 8 * k   (i < 8, k < 4)
// (AIE4 DMAs have no innermost stride, so rows stay contiguous.) The memtile
// BD chains loop forever, like objectFifo chains.

module {
  aie.device(npu3) @stream_b {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) {sym_name = "b_buf"} : memref<1024xi32>
    %prod = aie.lock(%mem, 0) {init = 1 : i32, sym_name = "b_prod"}
    %cons = aie.lock(%mem, 1) {init = 0 : i32, sym_name = "b_cons"}

    %dma = aie.memtile_dma(%mem) {
      %c1 = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 2, ^in, ^next)
    ^in:
      aie.use_lock(%prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024)
      aie.use_lock(%cons, Release, %c1)
      aie.next_bd ^in
    ^next:
      %1 = aie.dma_start(MM2S, 2, ^out, ^end)
    ^out:
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024 sizes = [8, 4, 32] strides = [32, 256, 1])
      aie.use_lock(%prod, Release, %c1)
      aie.next_bd ^out
    ^end:
      aie.end
    }

    // Placed and routed by hand so that the stream always enters the memtile
    // on South:0, the port design A also uses.
    aie.switchbox(%shim) {
      aie.connect<DMA : 0, North : 0>
      aie.connect<North : 0, DMA : 0>
    }
    aie.switchbox(%mem) {
      aie.connect<South : 0, DMA : 2>
      aie.connect<DMA : 2, South : 0>
    }
    aie.shim_dma_allocation @b_in(%shim, MM2S, 0)
    aie.shim_dma_allocation @b_out(%shim, S2MM, 0)

    // AIE4 needs Control32 -> Controller32 packet flows on the shim tile so
    // dma_await_task has a return path.
    aie.packet_flow(0) {
      aie.packet_source<%shim, Control32 : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%shim, Control32 : 1>
      aie.packet_dest<%shim, Controller32 : 0>
    }

    aie.runtime_sequence @sequence(%in: memref<1024xi32>, %out: memref<1024xi32>) {
      %t_in = aiex.dma_configure_task_for @b_in {
        aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 1024)
        aie.end
      } {issue_token = true}
      %t_out = aiex.dma_configure_task_for @b_out {
        aie.dma_bd(%out : memref<1024xi32> offset = 0 len = 1024)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_in)
      aiex.dma_await_task(%t_out)
    }
  }
}
