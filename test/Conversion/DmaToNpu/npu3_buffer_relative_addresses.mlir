//===- npu3_buffer_relative_addresses.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// npu3 tile addresses: column << 25, memtile spans address rows 1-5, core row
// r is address row r + 4; memtile data memory starts 1 MiB into its window.

// Memtile (1, 1) at 0x02100000, data memory at +0x100000, buffer at 0x200000,
// word 2: 0x02100000 + 0x100000 + 0x200000 + 8.
// CHECK-LABEL: aie.runtime_sequence @memtile
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 37748744 : ui32} : memref<2xi32>
// Word 4: 0x02400010.
// CHECK: %[[A1:.*]] = arith.constant 37748752 : i32
// CHECK: aiex.npu.write32(%[[A1]], %{{.*}}) : i32, i32
// Memtile (0, 1) last data word: 0x00100000 + 0x100000 + 0xFFFFC.
// CHECK: %[[A2:.*]] = arith.constant 3145724 : i32
// CHECK: aiex.npu.maskwrite32(%[[A2]], %{{.*}}, %{{.*}}) : i32, i32, i32
// CHECK: %[[A3:.*]] = arith.constant 37748744 : i32
// CHECK: aiex.npu.maskpoll(%[[A3]], %{{.*}}, %{{.*}}) : i32, i32, i32

// Core tile (2, 5) at 0x04900000, buffer at 0x400, word 3: 0x0490040C.
// CHECK-LABEL: aie.runtime_sequence @core
// CHECK: %[[C1:.*]] = arith.constant 76547084 : i32
// CHECK: aiex.npu.write32(%[[C1]], %{{.*}}) : i32, i32
// CHECK: %[[C2:.*]] = arith.constant 76547084 : i32
// CHECK: aiex.npu.maskpoll(%[[C2]], %{{.*}}, %{{.*}}) : i32, i32, i32

module {
  aie.device(npu3) {
    %mem01 = aie.tile(0, 1)
    %mem11 = aie.tile(1, 1)
    %core25 = aie.tile(2, 5)
    %low = aie.buffer(%mem01) {address = 0 : i32, sym_name = "low"} : memref<262144xi32>
    %high = aie.buffer(%mem11) {address = 2097152 : i32, sym_name = "high"} : memref<64xi32>
    %core = aie.buffer(%core25) {address = 1024 : i32, sym_name = "core_buf"} : memref<64xi32>
    memref.global "private" constant @payload : memref<2xi32> = dense<[5, 6]>
    aie.runtime_sequence @memtile() {
      %data = memref.get_global @payload : memref<2xi32>
      aiex.npu.blockwrite(%data) {address = 2 : ui32, buffer = @high} : memref<2xi32>
      %c4 = arith.constant 4 : i32
      %v = arith.constant 7 : i32
      %m = arith.constant 255 : i32
      aiex.npu.write32(%c4, %v) {buffer = @high} : i32, i32
      %last = arith.constant 262143 : i32
      aiex.npu.maskwrite32(%last, %v, %m) {buffer = @low} : i32, i32, i32
      %c2 = arith.constant 2 : i32
      aiex.npu.maskpoll(%c2, %v, %m) {buffer = @high} : i32, i32, i32
    }
    aie.runtime_sequence @core() {
      %c3 = arith.constant 3 : i32
      %v = arith.constant 7 : i32
      %m = arith.constant 255 : i32
      aiex.npu.write32(%c3, %v) {buffer = @core_buf} : i32, i32
      aiex.npu.maskpoll(%c3, %v, %m) {buffer = @core_buf} : i32, i32, i32
    }
  }
}
