// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercise extract/insert with an index loaded from memory and bounded to 0..15.
// Check the runtime index in LLVM IR and the native lane instructions.
//
// REQUIRES: peano
// RUN: %aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// CHECK-DAG: vextract.32{{[[:blank:]]+r[0-9]+, x[0-9]+, r[0-9]+, vaddsign1}}
// CHECK-DAG: vinsert.32{{[[:blank:]]+x[0-9]+, x[0-9]+, r[0-9]+, r[0-9]+}}

// LLVM-LABEL: define void @core_0_2()
// LLVM: %[[RAW:[0-9]+]] = load i32, ptr @lane_index
// LLVM: %[[BOUNDED:[0-9]+]] = and i32 %[[RAW]], 15
// LLVM: %[[IDX:[0-9]+]] = zext{{( nneg)?}} i32 %[[BOUNDED]] to i64
// LLVM: %[[VALUE:[0-9]+]] = extractelement <16 x i32> %{{[0-9]+}}, i64 %[[IDX]]
// LLVM: store i32 %[[VALUE]], ptr @extracted
// LLVM: %[[INSERTED:[0-9]+]] = insertelement <16 x i32> %{{[0-9]+}}, i32 42, i64 %[[IDX]]
// LLVM: store <16 x i32> %[[INSERTED]], ptr @inserted

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %src = aie.buffer(%tile) {sym_name = "src"} : memref<16xi32>
    %lane_index = aie.buffer(%tile) {sym_name = "lane_index"} : memref<1xi32>
    %extracted = aie.buffer(%tile) {sym_name = "extracted"} : memref<1xi32>
    %inserted = aie.buffer(%tile) {sym_name = "inserted"} : memref<16xi32>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %pad = arith.constant 0 : i32
      %mask = arith.constant 15 : i32
      %replacement = arith.constant 42 : i32
      %raw = memref.load %lane_index[%c0] : memref<1xi32>
      %bounded = arith.andi %raw, %mask : i32
      %idx = arith.index_cast %bounded : i32 to index
      %v = vector.transfer_read %src[%c0], %pad {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %value = vector.extract %v[%idx] : i32 from vector<16xi32>
      memref.store %value, %extracted[%c0] : memref<1xi32>
      %updated = vector.insert %replacement, %v[%idx] : i32 into vector<16xi32>
      vector.transfer_write %updated, %inserted[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>
      aie.end
    }
  }
}
