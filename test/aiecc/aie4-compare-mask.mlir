// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=I32 --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: FileCheck %s --check-prefix=I8 --input-file=%t.prj/opted_main_core_0_3.ll
// RUN: FileCheck %s --check-prefix=I16 --input-file=%t.prj/opted_main_core_0_4.ll
// RUN: FileCheck %s --check-prefix=SCALAR --input-file=%t.prj/opted_main_core_0_5.ll

// Compile native comparison masks with constant/dynamic extraction and a
// simultaneous select consumer, plus a scalar condition selecting all lanes.

// I32-LABEL: define void @core_0_2()
// I32: call { {{.*}} } @llvm.aie4.vadd32.512(
// I32: store i8 {{.*}}, ptr @bit_32
// I32: store i8 {{.*}}, ptr @dynamic_bit_32
// I32: call <16 x i32> @llvm.aie4.vadd32.mask.512(
// I32: store <16 x i32> {{.*}}, ptr @selected_32

// I8-LABEL: define void @core_0_3()
// I8: call { {{.*}} } @llvm.aie4.vadd8.512(
// I8: store i8 {{.*}}, ptr @bit_8
// I8: store i8 {{.*}}, ptr @dynamic_bit_8
// I8: call <64 x i8> @llvm.aie4.vadd8.mask.512(
// I8: store <64 x i8> {{.*}}, ptr @selected_8

// I16-LABEL: define void @core_0_4()
// I16: call { {{.*}} } @llvm.aie4.vadd16.1024(
// I16: store i8 {{.*}}, ptr @bit_16
// I16: store i8 {{.*}}, ptr @dynamic_bit_16
// I16: call <64 x i16> @llvm.aie4.vadd16.mask.1024(
// I16: store <64 x i16> {{.*}}, ptr @selected_16

// SCALAR-LABEL: define void @core_0_5()
// SCALAR: %[[CONDITION:[0-9]+]] = load i32, ptr @condition
// SCALAR: %[[BIT:[0-9]+]] = and i32 %[[CONDITION]], 1
// SCALAR: %[[MASK:[0-9]+]] = sub nsw i32 0, %[[BIT]]
// SCALAR: call <16 x i32> @llvm.aie4.vadd32.mask.512({{.*}}i32 %[[MASK]], i32 0)

module {
  aie.device(npu3) {
    %tile_32 = aie.tile(0, 2)
    %a_32 = aie.buffer(%tile_32) {sym_name = "a_32"} : memref<16xi32>
    %b_32 = aie.buffer(%tile_32) {sym_name = "b_32"} : memref<16xi32>
    %t_32 = aie.buffer(%tile_32) {sym_name = "t_32"} : memref<16xi32>
    %f_32 = aie.buffer(%tile_32) {sym_name = "f_32"} : memref<16xi32>
    %selected_32 = aie.buffer(%tile_32) {sym_name = "selected_32"} : memref<16xi32>
    %index_32 = aie.buffer(%tile_32) {sym_name = "index_32"} : memref<1xi32>
    %bit_32 = aie.buffer(%tile_32) {sym_name = "bit_32"} : memref<1xi8>
    %dynamic_bit_32 = aie.buffer(%tile_32) {sym_name = "dynamic_bit_32"} : memref<1xi8>
    %core_32 = aie.core(%tile_32) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0 : i32
      %limit = arith.constant 15 : i32
      %a = vector.transfer_read %a_32[%c0], %pad {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %b = vector.transfer_read %b_32[%c0], %pad {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %mask = arith.cmpi slt, %a, %b : vector<16xi32>
      %last = vector.extract %mask[15] : i1 from vector<16xi1>
      %bit = arith.extui %last : i1 to i8
      memref.store %bit, %bit_32[%c0] : memref<1xi8>
      %raw = memref.load %index_32[%c0] : memref<1xi32>
      %bounded = arith.andi %raw, %limit : i32
      %idx = arith.index_castui %bounded : i32 to index
      %dynamic = vector.extract %mask[%idx] : i1 from vector<16xi1>
      %dyn_bit = arith.extui %dynamic : i1 to i8
      memref.store %dyn_bit, %dynamic_bit_32[%c0] : memref<1xi8>
      %t = vector.transfer_read %t_32[%c0], %pad {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %f = vector.transfer_read %f_32[%c0], %pad {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %selected = arith.select %mask, %t, %f : vector<16xi1>, vector<16xi32>
      vector.transfer_write %selected, %selected_32[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>
      aie.end
    }
    %tile_8 = aie.tile(0, 3)
    %a_8 = aie.buffer(%tile_8) {sym_name = "a_8"} : memref<64xi8>
    %b_8 = aie.buffer(%tile_8) {sym_name = "b_8"} : memref<64xi8>
    %t_8 = aie.buffer(%tile_8) {sym_name = "t_8"} : memref<64xi8>
    %f_8 = aie.buffer(%tile_8) {sym_name = "f_8"} : memref<64xi8>
    %selected_8 = aie.buffer(%tile_8) {sym_name = "selected_8"} : memref<64xi8>
    %index_8 = aie.buffer(%tile_8) {sym_name = "index_8"} : memref<1xi32>
    %bit_8 = aie.buffer(%tile_8) {sym_name = "bit_8"} : memref<1xi8>
    %dynamic_bit_8 = aie.buffer(%tile_8) {sym_name = "dynamic_bit_8"} : memref<1xi8>
    %core_8 = aie.core(%tile_8) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0 : i8
      %limit = arith.constant 63 : i32
      %a = vector.transfer_read %a_8[%c0], %pad {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %b = vector.transfer_read %b_8[%c0], %pad {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %mask = arith.cmpi slt, %a, %b : vector<64xi8>
      %last = vector.extract %mask[63] : i1 from vector<64xi1>
      %bit = arith.extui %last : i1 to i8
      memref.store %bit, %bit_8[%c0] : memref<1xi8>
      %raw = memref.load %index_8[%c0] : memref<1xi32>
      %bounded = arith.andi %raw, %limit : i32
      %idx = arith.index_castui %bounded : i32 to index
      %dynamic = vector.extract %mask[%idx] : i1 from vector<64xi1>
      %dyn_bit = arith.extui %dynamic : i1 to i8
      memref.store %dyn_bit, %dynamic_bit_8[%c0] : memref<1xi8>
      %t = vector.transfer_read %t_8[%c0], %pad {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %f = vector.transfer_read %f_8[%c0], %pad {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
      %selected = arith.select %mask, %t, %f : vector<64xi1>, vector<64xi8>
      vector.transfer_write %selected, %selected_8[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
      aie.end
    }
    %tile_16 = aie.tile(0, 4)
    %a_16 = aie.buffer(%tile_16) {sym_name = "a_16"} : memref<64xi16>
    %b_16 = aie.buffer(%tile_16) {sym_name = "b_16"} : memref<64xi16>
    %t_16 = aie.buffer(%tile_16) {sym_name = "t_16"} : memref<64xi16>
    %f_16 = aie.buffer(%tile_16) {sym_name = "f_16"} : memref<64xi16>
    %selected_16 = aie.buffer(%tile_16) {sym_name = "selected_16"} : memref<64xi16>
    %index_16 = aie.buffer(%tile_16) {sym_name = "index_16"} : memref<1xi32>
    %bit_16 = aie.buffer(%tile_16) {sym_name = "bit_16"} : memref<1xi8>
    %dynamic_bit_16 = aie.buffer(%tile_16) {sym_name = "dynamic_bit_16"} : memref<1xi8>
    %core_16 = aie.core(%tile_16) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0 : i16
      %limit = arith.constant 63 : i32
      %a = vector.transfer_read %a_16[%c0], %pad {in_bounds = [true]} : memref<64xi16>, vector<64xi16>
      %b = vector.transfer_read %b_16[%c0], %pad {in_bounds = [true]} : memref<64xi16>, vector<64xi16>
      %mask = arith.cmpi slt, %a, %b : vector<64xi16>
      %last = vector.extract %mask[63] : i1 from vector<64xi1>
      %bit = arith.extui %last : i1 to i8
      memref.store %bit, %bit_16[%c0] : memref<1xi8>
      %raw = memref.load %index_16[%c0] : memref<1xi32>
      %bounded = arith.andi %raw, %limit : i32
      %idx = arith.index_castui %bounded : i32 to index
      %dynamic = vector.extract %mask[%idx] : i1 from vector<64xi1>
      %dyn_bit = arith.extui %dynamic : i1 to i8
      memref.store %dyn_bit, %dynamic_bit_16[%c0] : memref<1xi8>
      %t = vector.transfer_read %t_16[%c0], %pad {in_bounds = [true]} : memref<64xi16>, vector<64xi16>
      %f = vector.transfer_read %f_16[%c0], %pad {in_bounds = [true]} : memref<64xi16>, vector<64xi16>
      %selected = arith.select %mask, %t, %f : vector<64xi1>, vector<64xi16>
      vector.transfer_write %selected, %selected_16[%c0] {in_bounds = [true]} : vector<64xi16>, memref<64xi16>
      aie.end
    }
    %scalar_tile = aie.tile(0, 5)
    %condition = aie.buffer(%scalar_tile) {sym_name = "condition"} : memref<1xi32>
    %true = aie.buffer(%scalar_tile) {sym_name = "true_value"} : memref<16xi32>
    %false = aie.buffer(%scalar_tile) {sym_name = "false_value"} : memref<16xi32>
    %result = aie.buffer(%scalar_tile) {sym_name = "result"} : memref<16xi32>
    %scalar_core = aie.core(%scalar_tile) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0 : i32
      %raw = memref.load %condition[%c0] : memref<1xi32>
      %mask = arith.trunci %raw : i32 to i1
      %t = vector.transfer_read %true[%c0], %pad {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %f = vector.transfer_read %false[%c0], %pad {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %selected = arith.select %mask, %t, %f : vector<16xi32>
      vector.transfer_write %selected, %result[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>
      aie.end
    }
  }
}
