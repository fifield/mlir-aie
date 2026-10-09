// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// aievec.aie4.srs through LLVM IR: shift and sign are the trailing i32
// arguments, and aie4.entry_modes leaves no trace in the LLVM IR.
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:9,sat:1" --convert-func-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts | aie-translate --mlir-to-llvmir > %t.ll
// RUN: FileCheck %s --input-file=%t.ll
// RUN: %if peano %{ llc -mtriple=aie4 -mcpu=aie4b -O2 %t.ll -o - | FileCheck %s --check-prefix=ASM %}

// CHECK-NOT: aie4.entry_modes
// CHECK-LABEL: define <64 x i8> @srs_v64
// CHECK: call <64 x i8> @llvm.aie4.I512.v64.acc32.srs(<64 x i32> %{{.*}}, i32 12, i32 1)
// ASM-LABEL: srs_v64:
// ASM: srs.4x
func.func @srs_v64(%acc : vector<64xi32>) -> vector<64xi8>
    attributes {aie4.entry_modes = {rnd = 9 : i32, sat = 1 : i32}} {
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: define <32 x i16> @srs_v32
// CHECK: call <32 x i16> @llvm.aie4.I512.v32.acc32.srs(<32 x i32> %{{.*}}, i32 59, i32 0)
// CHECK-NOT: aie4.entry_modes
// ASM-LABEL: srs_v32:
// ASM: srs.2x
func.func @srs_v32(%acc : vector<32xi32>) -> vector<32xi16> {
  %0 = aievec.aie4.srs %acc {shift = 59 : i32, isSigned = false, rounding = #aievec.aie4_rnd<pos_inf>} : vector<32xi32> -> vector<32xi16>
  return %0 : vector<32xi16>
}
