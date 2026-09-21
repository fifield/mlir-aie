//===- aie4-vector-baseline-mul-bf16.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Compile 32- and 64-lane bf16 multiplication in separate cores so each ELF
// must contain exactly one vmul.f.64 using configuration 0xf2.
//
// REQUIRES: peano
//
// RUN: %aiecc -j 4 --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s
// RUN: llvm-objdump -dr %t.out/elfs_main_core_1_2/elfs_main_core_1_2.elf | FileCheck %s

// Track the configuration register and reject writes before its use.
// CHECK-NOT: vmul
// CHECK: mova{{[[:blank:]]+}}[[CONF:r[0-9]+]], #0xf2{{([[:blank:]]|;|$)}}
// CHECK-NOT: vmul
// CHECK-NOT: {{[a-zA-Z0-9_.]+[[:blank:]]+}}[[CONF]],
// CHECK: vmul.f.64{{[[:blank:]]+y[0-9]+, y[0-9]+, y[0-9]+, }}[[CONF]]{{([[:blank:]]|;|$)}}
// CHECK-NOT: vmul

module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_a32 = aie.buffer(%tile_0_2) {sym_name = "buf_a32"} : memref<32xbf16>
    %buf_b32 = aie.buffer(%tile_0_2) {sym_name = "buf_b32"} : memref<32xbf16>
    %buf_c32 = aie.buffer(%tile_0_2) {sym_name = "buf_c32"} : memref<32xbf16>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0.0 : bf16
      %a = vector.transfer_read %buf_a32[%c0], %pad {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
      %b = vector.transfer_read %buf_b32[%c0], %pad {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
      %m = arith.mulf %a, %b : vector<32xbf16>
      vector.transfer_write %m, %buf_c32[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16>
      aie.end
    }

    %tile_1_2 = aie.tile(1, 2)
    %buf_a64 = aie.buffer(%tile_1_2) {sym_name = "buf_a64"} : memref<64xbf16>
    %buf_b64 = aie.buffer(%tile_1_2) {sym_name = "buf_b64"} : memref<64xbf16>
    %buf_c64 = aie.buffer(%tile_1_2) {sym_name = "buf_c64"} : memref<64xbf16>

    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0.0 : bf16
      %a = vector.transfer_read %buf_a64[%c0], %pad {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %b = vector.transfer_read %buf_b64[%c0], %pad {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %m = arith.mulf %a, %b : vector<64xbf16>
      vector.transfer_write %m, %buf_c64[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
      aie.end
    }
  }
}
