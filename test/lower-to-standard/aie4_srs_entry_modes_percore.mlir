//===- aie4_srs_entry_modes_percore.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aiecc's per-core flow lowers one core per compile (tilecol/tilerow). A
// helper that holds an explicit-mode aievec.aie4.srs must still get its
// aie4.entry_modes from the cores filtered out of this compile.
//
// RUN: split-file %s %t
// RUN: aie-opt --aie-standard-lowering="tilecol=1 tilerow=2" --split-input-file %t/filtered.mlir | FileCheck %s
//
// merge.mlir: a helper reached from two cores keeps the modes both give,
// whichever cores this compile selects.
// RUN: aie-opt --aie-standard-lowering %t/merge.mlir | FileCheck %s --check-prefix=MERGE
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %t/merge.mlir | FileCheck %s --check-prefix=MERGE
// RUN: aie-opt --aie-standard-lowering="tilecol=1 tilerow=2" %t/merge.mlir | FileCheck %s --check-prefix=MERGE
// The helper then lowers: its entry modes give the crSat it needs.
// RUN: aie-opt --aie-standard-lowering %t/merge.mlir | aie-opt --convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s --check-prefix=LOWER
// LOWER: "xllvm.intr.aie4.I512.v64.acc32.srs"

//--- filtered.mlir

// Per-core flow (tilecol=1 tilerow=2): core (0, 2) is filtered out, but the
// helper only it calls still gets its entry modes, so the per-core compile of
// a module that shares helpers can lower them.
// CHECK-LABEL: module @two_cores
// CHECK-NOT:   func.func @core_0_2
// CHECK:       func.func @narrow_a({{.*}}) -> vector<64xi8> attributes {aie4.entry_modes = {rnd = 9 : i32, sat = 1 : i32}}
// CHECK:       func.func @narrow_b({{.*}}) -> vector<64xi8> attributes {aie4.entry_modes = {rnd = 12 : i32, sat = 3 : i32}}
// CHECK-NOT:   func.func @core_0_2
// CHECK:       func.func @core_1_2() {
// CHECK:         arith.constant 46 : i32
// CHECK-NEXT:    arith.constant 12 : i32
// CHECK:         arith.constant 45 : i32
// CHECK-NEXT:    arith.constant 3 : i32
module @two_cores {
  aie.device(npu3) {
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %a0 = aie.buffer(%t0) {sym_name = "a0"} : memref<64xi32>
    %b0 = aie.buffer(%t0) {sym_name = "b0"} : memref<64xi8>
    %a1 = aie.buffer(%t1) {sym_name = "a1"} : memref<64xi32>
    %b1 = aie.buffer(%t1) {sym_name = "b1"} : memref<64xi8>
    func.func @narrow_a(%v: vector<64xi32>) -> vector<64xi8> {
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      return %y : vector<64xi8>
    }
    func.func @narrow_b(%v: vector<64xi32>) -> vector<64xi8> {
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<conv_even>, saturation = #aievec.aie4_sat<symmetric>} : vector<64xi32> -> vector<64xi8>
      return %y : vector<64xi8>
    }
    aie.core(%t0) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a0[%c0] : memref<64xi32>, vector<64xi32>
      %y = func.call @narrow_a(%v) : (vector<64xi32>) -> vector<64xi8>
      vector.store %y, %b0[%c0] : memref<64xi8>, vector<64xi8>
      aie.end
    }
    aie.core(%t1) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a1[%c0] : memref<64xi32>, vector<64xi32>
      %y = func.call @narrow_b(%v) : (vector<64xi32>) -> vector<64xi8>
      vector.store %y, %b1[%c0] : memref<64xi8>, vector<64xi8>
      aie.end
    }
  }
}

// -----

// A filtered core with conflicting modes contributes no entry guarantee.
// Its shared helper retains the modes established by the selected core.
// CHECK-LABEL: module @filtered_scan_fails
// CHECK:       func.func @shared({{.*}}) -> vector<64xi8> attributes {aie4.entry_modes = {rnd = 9 : i32}}
// CHECK:       func.func @core_1_2() {
module @filtered_scan_fails {
  aie.device(npu3) {
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %a0 = aie.buffer(%t0) {sym_name = "a0"} : memref<64xi32>
    %b0 = aie.buffer(%t0) {sym_name = "b0"} : memref<64xi8>
    %c0b = aie.buffer(%t0) {sym_name = "c0b"} : memref<32xi16>
    %a1 = aie.buffer(%t1) {sym_name = "a1"} : memref<64xi32>
    %b1 = aie.buffer(%t1) {sym_name = "b1"} : memref<64xi8>
    func.func @shared(%v: vector<64xi32>) -> vector<64xi8> {
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      return %y : vector<64xi8>
    }
    aie.core(%t0) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a0[%c0] : memref<64xi32>, vector<64xi32>
      %y = func.call @shared(%v) : (vector<64xi32>) -> vector<64xi8>
      vector.store %y, %b0[%c0] : memref<64xi8>, vector<64xi8>
      %h = vector.extract_strided_slice %v offsets = [0], sizes = [32], strides = [1] : vector<64xi32> to vector<32xi32>
      %n = aievec.aie4.srs_i32_i16 %h {shift = 2 : i32} : vector<32xi32>, vector<32xi16>
      vector.store %n, %c0b[%c0] : memref<32xi16>, vector<32xi16>
      aie.end
    }
    aie.core(%t1) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a1[%c0] : memref<64xi32>, vector<64xi32>
      %y = func.call @shared(%v) : (vector<64xi32>) -> vector<64xi8>
      vector.store %y, %b1[%c0] : memref<64xi8>, vector<64xi8>
      aie.end
    }
  }
}

//--- merge.mlir

// Both cores preserve ambient crRnd and set crSat = 1, so their shared
// helper receives only {sat = 1}.
// MERGE-LABEL: module @merge
// MERGE:       func.func @narrow_sat({{.*}}) -> vector<64xi8> attributes {aie4.entry_modes = {sat = 1 : i32}}
module @merge {
  aie.device(npu3) {
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %a0 = aie.buffer(%t0) {sym_name = "a0"} : memref<64xi32>
    %b0 = aie.buffer(%t0) {sym_name = "b0"} : memref<64xi8>
    %a1 = aie.buffer(%t1) {sym_name = "a1"} : memref<64xi32>
    %b1 = aie.buffer(%t1) {sym_name = "b1"} : memref<64xi8>
    %f1 = aie.buffer(%t1) {sym_name = "f1"} : memref<32xf32>
    %h1 = aie.buffer(%t1) {sym_name = "h1"} : memref<32xbf16>
    func.func @narrow_sat(%v: vector<64xi32>) -> vector<64xi8> {
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      return %y : vector<64xi8>
    }
    aie.core(%t0) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a0[%c0] : memref<64xi32>, vector<64xi32>
      %y = func.call @narrow_sat(%v) : (vector<64xi32>) -> vector<64xi8>
      vector.store %y, %b0[%c0] : memref<64xi8>, vector<64xi8>
      aie.end
    }
    aie.core(%t1) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a1[%c0] : memref<64xi32>, vector<64xi32>
      %y = func.call @narrow_sat(%v) : (vector<64xi32>) -> vector<64xi8>
      vector.store %y, %b1[%c0] : memref<64xi8>, vector<64xi8>
      %x = vector.load %f1[%c0] : memref<32xf32>, vector<32xf32>
      %e = aievec.aie4.conv_f32_bf16 %x : vector<32xf32>, vector<32xbf16>
      vector.store %e, %h1[%c0] : memref<32xbf16>, vector<32xbf16>
      aie.end
    }
  }
}
