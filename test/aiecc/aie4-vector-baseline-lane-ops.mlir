//===- aie4-vector-baseline-lane-ops.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check generic AIE4 broadcasts and constant-index lane operations.
// Runtime indices are covered by aie4-vector-baseline-dynamic-lane-ops.mlir.
// The upper-half extraction can fold to the concatenation's original input.
//
// REQUIRES: peano
//
// RUN: aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// CHECK-DAG: vbcst.8
// CHECK-DAG: vbcst.16{{.*}}y{{[0-9]}}
// CHECK-DAG: vbcst.32{{.*}}y{{[0-9]}}
// CHECK-DAG: vextbcst.32
// CHECK-DAG: vextract.32
// CHECK-DAG: vinsert.32

module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)

    %buf_i8 = aie.buffer(%tile_0_2) {sym_name = "buf_i8"} : memref<64xi8>
    %buf_i8_bcast = aie.buffer(%tile_0_2) {sym_name = "buf_i8_bcast"} : memref<64xi8>

    %buf_bf16 = aie.buffer(%tile_0_2) {sym_name = "buf_bf16"} : memref<64xbf16>
    %buf_bf16_bcast1024 = aie.buffer(%tile_0_2) {sym_name = "buf_bf16_bcast1024"} : memref<64xbf16>

    %buf_i32_a = aie.buffer(%tile_0_2) {sym_name = "buf_i32_a"} : memref<16xi32>
    %buf_i32_bcast1024 = aie.buffer(%tile_0_2) {sym_name = "buf_i32_bcast1024"} : memref<32xi32>
    %buf_i32_lanebcast = aie.buffer(%tile_0_2) {sym_name = "buf_i32_lanebcast"} : memref<16xi32>
    %buf_i32_extract_c = aie.buffer(%tile_0_2) {sym_name = "buf_i32_extract_c"} : memref<1xi32>
    %buf_i32_extract_d = aie.buffer(%tile_0_2) {sym_name = "buf_i32_extract_d"} : memref<1xi32>
    %buf_i32_insert_c = aie.buffer(%tile_0_2) {sym_name = "buf_i32_insert_c"} : memref<16xi32>
    %buf_i32_insert_d = aie.buffer(%tile_0_2) {sym_name = "buf_i32_insert_d"} : memref<16xi32>

    %buf_lo = aie.buffer(%tile_0_2) {sym_name = "buf_lo"} : memref<16xi32>
    %buf_hi = aie.buffer(%tile_0_2) {sym_name = "buf_hi"} : memref<16xi32>
    %buf_concat = aie.buffer(%tile_0_2) {sym_name = "buf_concat"} : memref<32xi32>
    %buf_exthi = aie.buffer(%tile_0_2) {sym_name = "buf_exthi"} : memref<16xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c0_i8 = arith.constant 0 : i8
      %c0_i32 = arith.constant 0 : i32
      %c0_bf16 = arith.constant 0.0 : bf16
      %c5 = arith.constant 5 : index

      // Broadcast i8 to 512 bits.
      %a8 = vector.transfer_read %buf_i8[%c0], %c0_i8 {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %x8 = vector.extract %a8[0] : i8 from vector<64xi8>
      %b8 = vector.broadcast %x8 : i8 to vector<64xi8>
      vector.transfer_write %b8, %buf_i8_bcast[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>

      // Broadcast bf16 to 1024 bits.
      %abf = vector.transfer_read %buf_bf16[%c0], %c0_bf16 {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %xbf = vector.extract %abf[0] : bf16 from vector<64xbf16>
      %bbf = vector.broadcast %xbf : bf16 to vector<64xbf16>
      vector.transfer_write %bbf, %buf_bf16_bcast1024[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>

      %a = vector.transfer_read %buf_i32_a[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<16xi32>

      // Broadcast i32 to a 1024-bit y register.
      %x0 = vector.extract %a[0] : i32 from vector<16xi32>
      %bcast1024 = vector.broadcast %x0 : i32 to vector<32xi32>
      vector.transfer_write %bcast1024, %buf_i32_bcast1024[%c0] {in_bounds = [true]} : vector<32xi32>, memref<32xi32>

      // Broadcast constant lane 3.
      %lane = vector.extract %a[3] : i32 from vector<16xi32>
      %lanebcast = vector.broadcast %lane : i32 to vector<16xi32>
      vector.transfer_write %lanebcast, %buf_i32_lanebcast[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>

      // Extract with an immediate index.
      %ec = vector.extract %a[3] : i32 from vector<16xi32>
      memref.store %ec, %buf_i32_extract_c[%c0] : memref<1xi32>

      // Extract with an SSA constant index.
      %ed = vector.extract %a[%c5] : i32 from vector<16xi32>
      memref.store %ed, %buf_i32_extract_d[%c0] : memref<1xi32>

      // Insert with an immediate index.
      %ic = vector.insert %x0, %a [3] : i32 into vector<16xi32>
      vector.transfer_write %ic, %buf_i32_insert_c[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>

      // Insert with an SSA constant index.
      %idyn = vector.insert %x0, %a [%c5] : i32 into vector<16xi32>
      vector.transfer_write %idyn, %buf_i32_insert_d[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>

      // Concatenate two 512-bit vectors and extract the upper half.
      %lo = vector.transfer_read %buf_lo[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %hi = vector.transfer_read %buf_hi[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %z = arith.constant dense<0> : vector<32xi32>
      %t1 = vector.insert_strided_slice %lo, %z offsets = [0], strides = [1] : vector<16xi32> into vector<32xi32>
      %t2 = vector.insert_strided_slice %hi, %t1 offsets = [16], strides = [1] : vector<16xi32> into vector<32xi32>
      vector.transfer_write %t2, %buf_concat[%c0] {in_bounds = [true]} : vector<32xi32>, memref<32xi32>

      %exthi = vector.extract_strided_slice %t2 offsets = [16], sizes = [16], strides = [1] : vector<32xi32> to vector<16xi32>
      vector.transfer_write %exthi, %buf_exthi[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>

      aie.end
    }
  }
}
