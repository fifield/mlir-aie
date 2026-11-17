//===- npu3_devices.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The npu3 devices parse, verify and print, including the AIE4 stream-switch
// bundles and core-tile DMAs.

// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: aie.device(npu3) @full
// CHECK:         %[[SHIM:.*]] = aie.tile(2, 0)
// CHECK:         %[[MEM:.*]] = aie.tile(2, 1)
// CHECK:         %[[CORE:.*]] = aie.tile(2, 5)
// CHECK:         aie.lock(%[[CORE]], 15) {init = 63 : i32}
// CHECK:         aie.lock(%[[MEM]], 63)
// CHECK:         aie.lock(%[[SHIM]], 31)
// CHECK:         aie.flow(%[[CORE]], DMA : 0, %[[MEM]], DMA : 0)
// CHECK:         aie.flow(%[[SHIM]], DMAControl : 1, %[[CORE]], TileControl : 0)
// CHECK:         aie.flow(%[[CORE]], Trace32 : 0, %[[SHIM]], DMATrace : 0)
// CHECK:         aie.switchbox(%[[CORE]]) {
// CHECK:           aie.connect<South32 : 0, North32 : 1>
// CHECK:           aie.connect<Switch32 : 0, DMA : 0>
// CHECK:         aie.mem(%[[CORE]]) {
// CHECK:           aie.dma_start(S2MM, 1, ^bb1, ^bb2)
// CHECK:           aie.dma_bd(%{{.*}} : memref<256xi32>{{.*}}) {bd_id = 15 : i32}
module {
  aie.device(npu3) @full {
    %shim = aie.tile(2, 0)
    %mem = aie.tile(2, 1)
    %core = aie.tile(2, 5)
    %core_lock = aie.lock(%core, 15) {init = 63 : i32}
    %mem_lock = aie.lock(%mem, 63)
    %shim_lock = aie.lock(%shim, 31)
    %buf = aie.buffer(%core) : memref<256xi32>
    aie.flow(%core, DMA : 0, %mem, DMA : 0)
    aie.flow(%shim, DMAControl : 1, %core, TileControl : 0)
    aie.flow(%core, Trace32 : 0, %shim, DMATrace : 0)
    aie.switchbox(%core) {
      aie.connect<South32 : 0, North32 : 1>
      aie.connect<Switch32 : 0, DMA : 0>
    }
    aie.mem(%core) {
      %0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%core_lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 15 : i32}
      aie.use_lock(%core_lock, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }

  // CHECK-LABEL: aie.device(npu3_1col) @one
  // CHECK:         aie.tile(0, 5)
  aie.device(npu3_1col) @one {
    %tile = aie.tile(0, 5)
  }

  // CHECK-LABEL: aie.device(npu3_2col) @two
  // CHECK:         aie.tile(1, 5)
  aie.device(npu3_2col) @two {
    %tile = aie.tile(1, 5)
  }
}
