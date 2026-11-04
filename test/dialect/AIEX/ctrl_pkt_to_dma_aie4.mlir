// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -aie-ctrl-packet-infer-tiles -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" -aie-ctrl-packet-to-dma | FileCheck %s

// AIE4 core tile row 3 is encoded as address row 7.

// CHECK-LABEL: aie.device(npu3) {
// CHECK: %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK: %[[CORE:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_id = 29>}
// CHECK: aie.runtime_sequence(%[[ARG0:.*]]: memref<?xi32>) {
// CHECK:   aiex.npu.dma_memcpy_nd(%[[ARG0]][0, 0, 0, 0][1, 1, 1, 6][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_ctrl_mm2s_chan1} : memref<?xi32>
// CHECK:   %[[ROW:.*]] = arith.constant 0 : i32
// CHECK:   %[[COL:.*]] = arith.constant 0 : i32
// CHECK:   %[[DIR:.*]] = arith.constant 2 : i32
// CHECK:   %[[CHAN:.*]] = arith.constant 1 : i32
// CHECK:   %[[COLNUM:.*]] = arith.constant 1 : i32
// CHECK:   %[[ROWNUM:.*]] = arith.constant 1 : i32
// CHECK:   aiex.npu.sync(%[[COL]], %[[ROW]], %[[DIR]], %[[CHAN]], %[[COLNUM]], %[[ROWNUM]])
// CHECK: }
// CHECK: aie.packet_flow(29) {
// CHECK:   aie.packet_source<%[[SHIM]], DMAControl : 1>
// CHECK:   aie.packet_dest<%[[CORE]], TileControl : 0>
// CHECK: } {keep_pkt_header = false, priority_route = true}
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_ctrl_mm2s_chan1(%[[SHIM]], ControlMM2S, 1)

aie.device(npu3) {
  aie.runtime_sequence() {
    aiex.control_packet {address = 7340032 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 7340036 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}
