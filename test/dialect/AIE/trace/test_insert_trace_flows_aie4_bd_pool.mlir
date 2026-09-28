// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE4 trace uses its private four-BD pool. Even with distribute-channels,
// every trace shares TraceS2MM:0 and one host buffer of the requested size.
// RUN: aie-opt %s --aie-insert-trace-flows | FileCheck %s
// RUN: aie-opt %s --aie-insert-trace-flows="distribute-channels=true" | FileCheck %s
// RUN: aie-opt %s --aie-insert-trace-flows="default-bd-id=0 shim-channel=0" | FileCheck %s --check-prefix=BD0
// RUN: aie-opt %s --aie-insert-trace-flows="default-bd-id=2" | FileCheck %s --check-prefix=BD2
// RUN: aie-opt %s --aie-insert-trace-flows --aie-dma-to-npu | FileCheck %s --check-prefix=BLOCK

// CHECK: aie.runtime_sequence(%{{.*}}: memref<16xi32>, %{{.*}}: memref<8192xi8>)
// CHECK: aiex.npu.writebd {bd_id = 3 : i32,
// CHECK-SAME: channel = 0 : i32,
// CHECK-SAME: direction = 3 : i32,
// CHECK-SAME: enable_packet = 0 : i32,
// BD3 address: 0x18480 + 3 * 0x30 + 4 = 0x18514.
// CHECK: aiex.npu.address_patch({{.*}}) buffer %{{.*}} {addr = 99604 : ui32}
// Control register: DMA_Trace_S2MM_Ctrl = 0x18548.
// CHECK: %[[CTRL:.*]] = arith.constant 99656 : i32
// CHECK: aiex.npu.maskwrite32(%[[CTRL]],
// Task queue: DMA_Trace_S2MM_Task_Queue = 0x1854c; token issue | BD3.
// CHECK: %[[VALUE:.*]] = arith.constant -2147483645 : i32
// CHECK: %[[QUEUE:.*]] = arith.constant 99660 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %[[VALUE]])
// CHECK-NOT: aiex.npu.writebd
// CHECK: aie.packet_dest<%{{.*}}, DMATrace : 0>
// CHECK: aie.packet_dest<%{{.*}}, DMATrace : 0>
// CHECK-NOT: aie.packet_dest

// BD0: aiex.npu.writebd {bd_id = 0 : i32,
// BD0-SAME: channel = 0 : i32,
// BD0-SAME: direction = 3 : i32,
// BD0: aiex.npu.address_patch({{.*}}) buffer %{{.*}} {addr = 99460 : ui32}
// BD2: aiex.npu.writebd {bd_id = 2 : i32,
// BD2-SAME: direction = 3 : i32,
// BD2: aiex.npu.address_patch({{.*}}) buffer %{{.*}} {addr = 99556 : ui32}
// BLOCK: aiex.npu.blockwrite({{.*}}) {address = 99600 : ui32}

module @distribute_two_traces_aie4 {
  aie.device(npu3) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    aie.trace @trace_a(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=3
      aie.trace.stop broadcast=4
    }
    aie.trace @trace_b(%tile03) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=3
      aie.trace.stop broadcast=4
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }
  }
}
