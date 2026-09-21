// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc -j 4 --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefixes=CHECK,I8 --implicit-check-not='extractelement <64 x i1>' --input-file=%t.prj/peano-compat_main_core_0_2.ll
// RUN: FileCheck %s --check-prefixes=CHECK,I16 --implicit-check-not='extractelement <64 x i1>' --input-file=%t.prj/peano-compat_main_core_0_3.ll

// Extract constant and runtime-indexed bits after combining two native masks.
// Both 64-lane shapes must compile through to ELF.

// I8-LABEL: define void @core_0_2()
// I16-LABEL: define void @core_0_3()
// I8: call { <64 x i8>, <2 x i32> } @llvm.aie4.vadd8.512(
// I16: call { <64 x i16>, <2 x i32> } @llvm.aie4.vadd16.1024(
// I8: call { <64 x i8>, <2 x i32> } @llvm.aie4.vadd8.512(
// I16: call { <64 x i16>, <2 x i32> } @llvm.aie4.vadd16.1024(
// CHECK: store i8 {{.*}}, ptr @last_{{(8|16)}}
// CHECK: store i8 {{.*}}, ptr @dynamic_{{(8|16)}}

module {
  aie.device(npu3) {
    %tile_8 = aie.tile(0, 2)
    %a_8 = aie.buffer(%tile_8) {sym_name = "a_8"} : memref<64xi8>
    %b_8 = aie.buffer(%tile_8) {sym_name = "b_8"} : memref<64xi8>
    %c_8 = aie.buffer(%tile_8) {sym_name = "c_8"} : memref<64xi8>
    %index_8 = aie.buffer(%tile_8) {sym_name = "index_8"} : memref<1xi32>
    %last_8 = aie.buffer(%tile_8) {sym_name = "last_8"} : memref<1xi8>
    %dynamic_8 = aie.buffer(%tile_8) {sym_name = "dynamic_8"} : memref<1xi8>
    %core_8 = aie.core(%tile_8) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0 : i8
      %limit = arith.constant 63 : i32
      %a = vector.transfer_read %a_8[%c0], %pad {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %b = vector.transfer_read %b_8[%c0], %pad {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %c = vector.transfer_read %c_8[%c0], %pad {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %m = arith.cmpi slt, %a, %b : vector<64xi8>
      %n = arith.cmpi slt, %c, %b : vector<64xi8>
      %mask = arith.xori %m, %n : vector<64xi1>
      %last = vector.extract %mask[63] : i1 from vector<64xi1>
      %last_value = arith.extui %last : i1 to i8
      memref.store %last_value, %last_8[%c0] : memref<1xi8>
      %raw = memref.load %index_8[%c0] : memref<1xi32>
      %bounded = arith.andi %raw, %limit : i32
      %index = arith.index_castui %bounded : i32 to index
      %bit = vector.extract %mask[%index] : i1 from vector<64xi1>
      %value = arith.extui %bit : i1 to i8
      memref.store %value, %dynamic_8[%c0] : memref<1xi8>
      aie.end
    }
    %tile_16 = aie.tile(0, 3)
    %a_16 = aie.buffer(%tile_16) {sym_name = "a_16"} : memref<64xi16>
    %b_16 = aie.buffer(%tile_16) {sym_name = "b_16"} : memref<64xi16>
    %c_16 = aie.buffer(%tile_16) {sym_name = "c_16"} : memref<64xi16>
    %index_16 = aie.buffer(%tile_16) {sym_name = "index_16"} : memref<1xi32>
    %last_16 = aie.buffer(%tile_16) {sym_name = "last_16"} : memref<1xi8>
    %dynamic_16 = aie.buffer(%tile_16) {sym_name = "dynamic_16"} : memref<1xi8>
    %core_16 = aie.core(%tile_16) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0 : i16
      %limit = arith.constant 63 : i32
      %a = vector.transfer_read %a_16[%c0], %pad {in_bounds = [true]} : memref<64xi16>, vector<64xi16>
      %b = vector.transfer_read %b_16[%c0], %pad {in_bounds = [true]} : memref<64xi16>, vector<64xi16>
      %c = vector.transfer_read %c_16[%c0], %pad {in_bounds = [true]} : memref<64xi16>, vector<64xi16>
      %m = arith.cmpi slt, %a, %b : vector<64xi16>
      %n = arith.cmpi slt, %c, %b : vector<64xi16>
      %mask = arith.xori %m, %n : vector<64xi1>
      %last = vector.extract %mask[63] : i1 from vector<64xi1>
      %last_value = arith.extui %last : i1 to i8
      memref.store %last_value, %last_16[%c0] : memref<1xi8>
      %raw = memref.load %index_16[%c0] : memref<1xi32>
      %bounded = arith.andi %raw, %limit : i32
      %index = arith.index_castui %bounded : i32 to index
      %bit = vector.extract %mask[%index] : i1 from vector<64xi1>
      %value = arith.extui %bit : i1 to i8
      memref.store %value, %dynamic_16[%c0] : memref<1xi8>
      aie.end
    }
  }
}
