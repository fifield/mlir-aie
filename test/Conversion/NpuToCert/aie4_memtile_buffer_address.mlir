//===- aie4_memtile_buffer_address.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// Mem tile (0, 1), buffer at 64, word 1: 0x100000 + 0x100000 + 0x44 =
// 0x200044. Mem tile (1, 1), buffer at 0x180000, word 1: 0x2000000 +
// 0x100000 + 0x100000 + 0x180004 = 0x2380004. A core tile's data memory
// starts at its tile address: core tile (0, 2) is address row 6, so word 1 of
// a buffer at 1024 is 0x600404.

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// CHECK: aiex.cert.write32(2097220, 7)
// CHECK: aiex.cert.write32(37224452, 7)
// CHECK: aiex.cert.write32(6292484, 7)
aie.device(npu3) {
  %mem01 = aie.tile(0, 1)
  %mem11 = aie.tile(1, 1)
  %core = aie.tile(0, 2)
  %a = aie.buffer(%mem01) {address = 64 : i32, sym_name = "a"} : memref<16xi32>
  %b = aie.buffer(%mem11) {address = 0x180000 : i32, sym_name = "b"} : memref<16xi32>
  %c = aie.buffer(%core) {address = 1024 : i32, sym_name = "c"} : memref<16xi32>
  aie.runtime_sequence @seq() {
    %i1 = arith.constant 1 : i32
    %v = arith.constant 7 : i32
    aiex.npu.write32(%i1, %v) {buffer = @a} : i32, i32
    aiex.npu.write32(%i1, %v) {buffer = @b} : i32, i32
    aiex.npu.write32(%i1, %v) {buffer = @c} : i32, i32
  }
}
