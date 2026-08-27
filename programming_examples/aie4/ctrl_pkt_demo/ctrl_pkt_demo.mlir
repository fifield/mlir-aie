//===- ctrl_pkt_demo.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Minimal AIE4 (npu3) control-packet configuration demo.
//
// A single control packet writes 16 words into core tile (0,2) data memory
// over the DMA/TileControl network. The runtime sequence then releases a lock
// and a plain core-DMA -> shim-S2MM transfer streams that same memory back to
// a host buffer, so the host can verify byte-for-byte that the control packet
// landed (no polling, no timeout ambiguity).
//
// One column, one control-packet destination tile, one `.ctrlpkt` ELF section.
//
// Address notes (AIE4 tile-address encoding, see AIE4TargetModel):
//   * tile(0,2) is a core tile; its address row is 6, so its base address is
//     0x600000 and `cbuf` at buffer offset 0x400 lives at 0x600400.
//   * Do NOT target mem tile *data* memory (tile base + 0x100000) from a
//     control packet: the address row of that region decodes back to a core
//     tile row, so control-packet tile inference picks the wrong destination.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    // Destination of the control packet.
    %cbuf = aie.buffer(%core) {sym_name = "cbuf", address = 1024 : i32} : memref<16xi32>

    // Gate: released from the runtime sequence *after* the control packet has
    // landed, so the DMA only reads memory the control packet already wrote.
    %lock_go = aie.lock(%core, 0) {init = 0 : i32, sym_name = "lock_go"}
    %lock_done = aie.lock(%core, 1) {init = 0 : i32, sym_name = "lock_done"}

    %c1_lock = arith.constant 1 : i32
    aie.mem(%core) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_go, AcquireGreaterEqual, %c1_lock)
      aie.dma_bd(%cbuf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%lock_done, Release, %c1_lock)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    aie.flow(%core, DMA : 0, %shim, DMA : 0)
    aie.shim_dma_allocation @out (%shim, S2MM, 0)

    aie.runtime_sequence @sequence(%out: memref<16xi32>) {
      // 16 x 32-bit words written into cbuf by a single AIE4 control packet.
      // AIE4 allows up to 16 data beats per control packet, so this fits in
      // one packet and aie-legalize-control-packet does not have to split it.
      aiex.control_packet {address = 0x600400 : ui32, data = array<i32: 4660, 22136, 4661, 22137, 4662, 22138, 4663, 22139, 4664, 22140, 4665, 22141, 4666, 22142, 4667, 22143>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.set_lock(%lock_go, 1)
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
