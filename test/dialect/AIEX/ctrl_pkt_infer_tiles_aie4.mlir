//===- ctrl_pkt_infer_tiles_aie4.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-infer-tiles | FileCheck %s

// Regression test for the AIE4 tile-address row field width. The row field sits
// between getRowShift() (20) and getColumnShift() (25), i.e. it is 5 bits wide.
// Masking it with 0xff leaks the low column bits into the row, so every control
// packet aimed at a column > 0 used to resolve to a bogus (often out of range)
// row. Column 0 cannot catch this, hence the col >= 1 cases below. Expected
// decodes cross-checked against `python -m aie.utils.regdb --arch aie4 <addr>`.

// A MEM tile owns 5MiB of address space so on npu3 address rows 1-5 all
// belong to the single MEM tile and the core rows start at address row 6.
// Decoding only unwound the core-row offset, so the upper 4MiB of every
// MEM tile (address rows 2-5) resolved to core rows 2-5 instead -- silently
// retargeting L2 control packets at a core tile, which then got the wrong
// controller id and shim channel. Only address rows 0, 1, 6 and 7 were
// previously tested, and those are exactly the values that decode correctly
// either way.

// CHECK-LABEL: aie.device(npu3)
// CHECK-DAG: aie.tile(1, 3)
// CHECK-DAG: aie.tile(1, 1)
// CHECK-DAG: aie.tile(2, 2)
// CHECK-DAG: aie.tile(0, 1)
// CHECK-DAG: aie.tile(2, 1)
// The bogus core tiles the old decode produced for MEM tile addresses.
// CHECK-NOT: aie.tile(0, 2)
// CHECK-NOT: aie.tile(2, 4)
// CHECK-NOT: aie.tile(2, 5)

module {
  aie.device(npu3) {
    aie.runtime_sequence() {
      // 0x2700000: col 1, core tile row 3 (address row 7), data memory.
      // With the 0xff mask this decoded to row 35.
      aiex.control_packet {address = 0x2700000 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      // 0x2110000: col 1, mem tile row 1, Lock0_value.
      // With the 0xff mask this decoded to row 29.
      aiex.control_packet {address = 0x2110000 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      // 0x46AE000: col 2, core tile row 2 (address row 6), Lock0_value.
      // With the 0xff mask this decoded to row 66.
      aiex.control_packet {address = 0x46AE000 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      // The four MEM tile address rows above the first MiB. All of these are
      // offsets into the column's single MEM tile at logical row 1.
      // 0x0200000: col 0, address row 2. Previously decoded to core row 2.
      aiex.control_packet {address = 0x0200000 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      // 0x2300000: col 1, address row 3. Previously decoded to core row 3.
      aiex.control_packet {address = 0x2300000 : ui32, data = array<i32: 5>, opcode = 0 : i32, stream_id = 0 : i32}
      // 0x4400000: col 2, address row 4. Previously decoded to core row 4.
      aiex.control_packet {address = 0x4400000 : ui32, data = array<i32: 6>, opcode = 0 : i32, stream_id = 0 : i32}
      // 0x4500000: col 2, address row 5. Previously decoded to core row 5,
      // which also lands in a different shim channel group than the MEM tile.
      aiex.control_packet {address = 0x4500000 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
