//===- test_insert_trace_flows_aie4_bd_pool.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// `default-bd-id` is a channel local BD id. Its default value 15 names the last
// BD in a 16-entry S2MM pool. An AIE4 shim exposes those pools in a global BD
// namespace (channel 0 -> ids [0,16), channel 1 -> ids [16,32)), so local BD
// 15 on the default channel 1 maps to global BD 31.

// RUN: aie-opt %s --split-input-file --aie-insert-trace-flows | FileCheck %s
// RUN: aie-opt %s --split-input-file --aie-insert-trace-flows="distribute-channels=true" --aie-resolve-address-patch-buffers | FileCheck %s --check-prefix=DIST

// The target-aware local-to-global translation is routine and must not warn.
// RUN: aie-opt %s --split-input-file --aie-insert-trace-flows 2>&1 | FileCheck %s --check-prefix=SILENT
// SILENT-NOT: warning

// An explicit valid local BD id has the same translation semantics: local 12
// maps to global 28 on channel 1 and global 12 on channel 0.
// RUN: aie-opt %s --split-input-file --aie-insert-trace-flows="distribute-channels=true default-bd-id=12" | FileCheck %s --check-prefix=LOCAL12
// LOCAL12-DAG: aiex.npu.writebd {bd_id = 28 : i32, {{.*}}channel = 1 : i32,
// LOCAL12-DAG: aiex.npu.writebd {bd_id = 12 : i32, {{.*}}channel = 0 : i32,

// Single trace on the default channel 1: BD 31, the top of ch1's pool.
// Address patch targets 0x58000 + 15 * 0x30 + 0x4 = 0x582D4 = 361172.
// Queue push to DMA_S2MM_1_Task_Queue at 0x58544 = 361796 carries
// Enable_Token_Issue (bit 31) | Start_BD_ID = 15 == 0x8000000F.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.npu.writebd {bd_id = 31 : i32,
// CHECK-SAME: channel = 1 : i32,
// CHECK: aiex.npu.address_patch({{.*}}) buffer %{{.*}} {addr = 361172 : ui32}
// CHECK: %[[V:.*]] = arith.constant -2147483633 : i32
// CHECK: %[[A:.*]] = arith.constant 361796 : i32
// CHECK: aiex.npu.write32(%[[A]], %[[V]])

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.trace @core_trace(%tile) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id = 10 type = core
      aie.trace.event <"INSTR_EVENT_0">
      aie.trace.start broadcast = 3
      aie.trace.stop broadcast = 4
    }
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Two traces with distribute-channels: each channel must take the top of its
// OWN pool, not the AIE2 "step down by one"
//
// Local BD 15 maps independently into each channel's global pool:
//   ch1 -> BD 31, address patch 0x58000 + 15 * 0x30 + 0x4 = 0x582D4 = 361172,
//          queue at DMA_S2MM_1_Task_Queue 0x58544 = 361796
//   ch0 -> BD 15, address patch 0x18000 + 15 * 0x30 + 0x4 = 0x182D4 =  99028,
//          queue at DMA_S2MM_0_Task_Queue 0x18544 =  99652
//
// Both queue pushes carry the same bank-local Start_BD_ID of 15
// (0x8000000F == -2147483633), because 31 - 16 == 15 - 0 == 15.

// DIST-LABEL: aie.device(npu3)
// DIST-DAG: aiex.npu.writebd {bd_id = 31 : i32, {{.*}}channel = 1 : i32,
// DIST-DAG: aiex.npu.writebd {bd_id = 15 : i32, {{.*}}channel = 0 : i32,
// DIST-DAG: aiex.npu.address_patch({{.*}}) {addr = 361172 : ui32, arg_idx = 1 : i32}
// DIST-DAG: aiex.npu.address_patch({{.*}}) {addr = 99028 : ui32, arg_idx = 1 : i32}
// DIST-DAG: %[[QA1:.*]] = arith.constant 361796 : i32
// DIST-DAG: %[[QA0:.*]] = arith.constant 99652 : i32
// DIST-DAG: aiex.npu.write32(%[[QA1]], %{{.*}})
// DIST-DAG: aiex.npu.write32(%[[QA0]], %{{.*}})
// The two traces land on the two different channels.
// DIST-DAG: aie.packet_dest<%{{.*}}, DMA : 1>
// DIST-DAG: aie.packet_dest<%{{.*}}, DMA : 0>

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
