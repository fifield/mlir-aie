//===- npu3_tile_addresses.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s

// npu3 TXN addresses: column at bit 25, memtile spans address rows 1-5 (data
// memory at +1 MiB), core row r is address row r + 4.

// TXN header: device generation 42, 6 rows, 3 columns, 1 memtile row.
// CHECK: 062A0100
// CHECK-NEXT: 00000103
// write32
// CHECK: 04150000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000007
// maskwrite32
// CHECK: 006AE010
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000007
// CHECK-NEXT: 000000FF
// maskpoll
// CHECK: 029AE000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000007
// CHECK-NEXT: 000000FF
// blockwrite by column and row
// CHECK: 02110000
// CHECK-NEXT: 00000018
// CHECK-NEXT: 00000005
// CHECK-NEXT: 00000006
// blockwrite by buffer
// CHECK: 02400008
// CHECK-NEXT: 00000018
// CHECK-NEXT: 00000005
// CHECK-NEXT: 00000006

module {
  aie.device(npu3) {
    %mem11 = aie.tile(1, 1)
    %core02 = aie.tile(0, 2)
    %buf = aie.buffer(%mem11) {address = 2097152 : i32, sym_name = "buf"} : memref<64xi32>
    memref.global "private" constant @payload : memref<2xi32> = dense<[5, 6]>
    aie.runtime_sequence @seq() {
      %v = arith.constant 7 : i32
      %m = arith.constant 255 : i32
      // Memtile (2, 1) upper lock bank: 0x04100000 + 0x50000.
      %lock = arith.constant 0x50000 : i32
      aiex.npu.write32(%lock, %v) {column = 2 : i32, row = 1 : i32} : i32, i32
      // Core tile (0, 2) lock 1: address row 6, 0x00600000 + 0xAE010.
      %clock = arith.constant 0xAE010 : i32
      aiex.npu.maskwrite32(%clock, %v, %m) {column = 0 : i32, row = 2 : i32} : i32, i32, i32
      // Core tile (1, 5): address row 9, 0x02900000 + 0xAE000.
      %poll = arith.constant 0xAE000 : i32
      aiex.npu.maskpoll(%poll, %v, %m) {column = 1 : i32, row = 5 : i32} : i32, i32, i32
      // Memtile (1, 1) by column and row: 0x02100000 + 0x10000.
      %data = memref.get_global @payload : memref<2xi32>
      aiex.npu.blockwrite(%data) {address = 0x10000 : ui32, column = 1 : i32, row = 1 : i32} : memref<2xi32>
      // Memtile (1, 1) buffer word 2: 0x02100000 + 0x100000 + 0x200000 + 8.
      aiex.npu.blockwrite(%data) {address = 2 : ui32, buffer = @buf} : memref<2xi32>
    }
  }
}
