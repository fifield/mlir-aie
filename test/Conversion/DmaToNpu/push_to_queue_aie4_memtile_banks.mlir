//===- push_to_queue_aie4_memtile_banks.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// MEM S2MM channels 4-7 use an upper register bank whose local channel index
// restarts at zero. Channel 7 is therefore upper-bank local channel 3.
// CHECK-DAG: %[[CTRL:.*]] = arith.constant 68460056 : i32
// CHECK: aiex.npu.maskwrite32(%[[CTRL]], %{{.*}}, %{{.*}}) : i32, i32, i32
// CHECK-DAG: %[[QUEUE:.*]] = arith.constant 68460060 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module {
  aie.device(npu3) {
    %mem = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
    aie.runtime_sequence @channel7() {
      %repeat = arith.constant 0 : i32
      %bd = arith.constant 0 : i32
      aiex.npu.push_queue(2, 1, S2MM : 7) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
    }
  }
}
