//===- aie4_teardown.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An AIE4 aiex.npu.write_teardown, written in line: it undoes exactly what the
// configuration of @ov enabled, on npu3's register layout. Each register is
// named as the AIE4 register database names it.

// RUN: aie-opt --aie-create-pathfinder-flows --aie-assign-bd-ids --aie-expand-load-pdi="inline-config=true" %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @seq
// The circuit routes shim -> memtile -> core: both ends of each connection.
// Shim Stream_Switch_Manager_Config_North_0 and Subordinate_Config_DMA_0.
// CHECK:      aiex.npu.write32(%c32800_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c32832_i32, %c0_i32)
// The packet route: shim Manager_Config_North_3, Subordinate_Config_DMA_1 and
// its slot 0.
// CHECK-NEXT: aiex.npu.write32(%c294948_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c32836_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c32912_i32, %c0_i32)
// Memtile Manager_Config_DMA_0, Subordinate_Config_South_0, Manager_Config_North_2,
// Subordinate_Config_DMA_0, then the packet hop: Manager_Config_North_5,
// Subordinate_Config_South_3 and its slot 0.
// CHECK-NEXT: aiex.npu.write32(%c1179648_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c1179740_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c1179688_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c1179712_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c1441828_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c1441888_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c1442048_i32, %c0_i32)
// Core tile Manager_Config_DMA_0, Subordinate_Config_South_2,
// Manager_Config_DMA_1, Subordinate_Config_South_5 and its slot 0.
// CHECK-NEXT: aiex.npu.write32(%c7028744_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c7029020_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c7028748_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c7029032_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.write32(%c7029408_i32, %c0_i32)
// Core (0, 2): Core_Control Enable cleared.
// CHECK-NEXT: aiex.npu.maskwrite32(%c6959104_i32, %c0_i32, %c1_i32)
// Each DMA channel the configuration used: Reset asserted, then released with
// its default mode. Core DMA_S2MM_0_Ctrl, memtile DMA_S2MM_0_Ctrl and
// DMA_MM2S_0_Ctrl.
// CHECK-NEXT: aiex.npu.maskwrite32(%c6999552_i32, %c2_i32, %c2_i32)
// CHECK-NEXT: aiex.npu.write32(%c6999552_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.maskwrite32(%c1089024_i32, %c2_i32, %c2_i32)
// CHECK-NEXT: aiex.npu.write32(%c1089024_i32, %c0_i32)
// CHECK-NEXT: aiex.npu.maskwrite32(%c1089120_i32, %c2_i32, %c2_i32)
// CHECK-NEXT: aiex.npu.write32(%c1089120_i32, %c0_i32)
// The memtile MM2S channel's DMA_MM2S_0_Constant_Pad_Value too, which its
// descriptor does not cover.
// CHECK-NEXT: aiex.npu.write32(%c1089392_i32, %c0_i32)
// CHECK-NOT:  aiex.npu

module {
  aie.device(npu3_1col) @ov {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %c = aie.tile(0, 2)
    %buf = aie.buffer(%m) {address = 0 : i32, sym_name = "mbuf"} : memref<16xi32>
    %l = aie.lock(%m, 0) {init = 1 : i32}
    %cbuf = aie.buffer(%c) {address = 4096 : i32, sym_name = "cbuf"} : memref<16xi32>
    %cl = aie.lock(%c, 0) {init = 0 : i32}
    aie.flow(%s, DMA : 0, %m, DMA : 0)
    aie.flow(%m, DMA : 0, %c, DMA : 0)
    aie.packet_flow(3) {
      aie.packet_source<%s, DMA : 1>
      aie.packet_dest<%c, DMA : 1>
    }
    %mt = aie.memtile_dma(%m) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^mm2s)
    ^bd0:
      %one = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^mm2s:
      %1 = aie.dma_start(MM2S, 0, ^bd1, ^end)
    ^bd1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd1
    ^end:
      aie.end
    }
    %md = aie.mem(%c) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %z = arith.constant 1 : i32
      aie.use_lock(%cl, AcquireGreaterEqual, %z)
      aie.dma_bd(%cbuf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    %core = aie.core(%c) {
      aie.end
    }
  }
  aie.device(npu3_1col) @main {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @seq() {
      aiex.npu.write_reset
      aiex.npu.write_config @ov
      aiex.npu.write_teardown @ov
    }
  }
}
