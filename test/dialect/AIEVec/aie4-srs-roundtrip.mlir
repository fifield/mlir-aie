// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Round-trip aievec.aie4.srs: both shapes, every rounding and saturation
// mode, each attribute alone, both and neither, and the shift range ends.
//
// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: func.func @shapes
// CHECK: aievec.aie4.srs %{{.*}} {isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>, shift = 12 : i32} : vector<64xi32> -> vector<64xi8>
// CHECK: aievec.aie4.srs %{{.*}} {isSigned = false, shift = 0 : i32} : vector<32xi32> -> vector<32xi16>
func.func @shapes(%a : vector<64xi32>, %b : vector<32xi32>)
    -> (vector<64xi8>, vector<32xi16>) {
  %0 = aievec.aie4.srs %a {shift = 12 : i32, isSigned = true,
         rounding = #aievec.aie4_rnd<pos_inf>,
         saturation = #aievec.aie4_sat<saturate>}
       : vector<64xi32> -> vector<64xi8>
  %1 = aievec.aie4.srs %b {shift = 0 : i32, isSigned = false}
       : vector<32xi32> -> vector<32xi16>
  return %0, %1 : vector<64xi8>, vector<32xi16>
}

// Each attribute alone; shift 59.
// CHECK-LABEL: func.func @one_attribute
// CHECK: aievec.aie4.srs %{{.*}} {isSigned = true, rounding = #aievec.aie4_rnd<conv_even>, shift = 59 : i32} : vector<64xi32> -> vector<64xi8>
// CHECK: aievec.aie4.srs %{{.*}} {isSigned = true, saturation = #aievec.aie4_sat<symmetric>, shift = 59 : i32} : vector<32xi32> -> vector<32xi16>
func.func @one_attribute(%a : vector<64xi32>, %b : vector<32xi32>)
    -> (vector<64xi8>, vector<32xi16>) {
  %0 = aievec.aie4.srs %a {shift = 59 : i32, isSigned = true,
         rounding = #aievec.aie4_rnd<conv_even>}
       : vector<64xi32> -> vector<64xi8>
  %1 = aievec.aie4.srs %b {shift = 59 : i32, isSigned = true,
         saturation = #aievec.aie4_sat<symmetric>}
       : vector<32xi32> -> vector<32xi16>
  return %0, %1 : vector<64xi8>, vector<32xi16>
}

// CHECK-LABEL: func.func @rounding_modes
// CHECK: rounding = #aievec.aie4_rnd<floor>
// CHECK: rounding = #aievec.aie4_rnd<ceil>
// CHECK: rounding = #aievec.aie4_rnd<sym_floor>
// CHECK: rounding = #aievec.aie4_rnd<sym_ceil>
// CHECK: rounding = #aievec.aie4_rnd<neg_inf>
// CHECK: rounding = #aievec.aie4_rnd<pos_inf>
// CHECK: rounding = #aievec.aie4_rnd<sym_zero>
// CHECK: rounding = #aievec.aie4_rnd<sym_inf>
// CHECK: rounding = #aievec.aie4_rnd<conv_even>
// CHECK: rounding = #aievec.aie4_rnd<conv_odd>
func.func @rounding_modes(%a : vector<64xi32>) {
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<floor>} : vector<64xi32> -> vector<64xi8>
  %1 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<ceil>} : vector<64xi32> -> vector<64xi8>
  %2 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<sym_floor>} : vector<64xi32> -> vector<64xi8>
  %3 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<sym_ceil>} : vector<64xi32> -> vector<64xi8>
  %4 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<neg_inf>} : vector<64xi32> -> vector<64xi8>
  %5 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
  %6 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<sym_zero>} : vector<64xi32> -> vector<64xi8>
  %7 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<sym_inf>} : vector<64xi32> -> vector<64xi8>
  %8 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<conv_even>} : vector<64xi32> -> vector<64xi8>
  %9 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<conv_odd>} : vector<64xi32> -> vector<64xi8>
  return
}

// CHECK-LABEL: func.func @saturation_modes
// CHECK: saturation = #aievec.aie4_sat<truncate>
// CHECK: saturation = #aievec.aie4_sat<saturate>
// CHECK: saturation = #aievec.aie4_sat<symmetric>
func.func @saturation_modes(%a : vector<32xi32>) {
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = false, saturation = #aievec.aie4_sat<truncate>} : vector<32xi32> -> vector<32xi16>
  %1 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = false, saturation = #aievec.aie4_sat<saturate>} : vector<32xi32> -> vector<32xi16>
  %2 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = false, saturation = #aievec.aie4_sat<symmetric>} : vector<32xi32> -> vector<32xi16>
  return
}

// The generic form prints in the custom form.
// CHECK-LABEL: func.func @generic_form
// CHECK: aievec.aie4.srs %{{.*}} {isSigned = true, rounding = #aievec.aie4_rnd<sym_zero>, saturation = #aievec.aie4_sat<truncate>, shift = 7 : i32} : vector<64xi32> -> vector<64xi8>
func.func @generic_form(%a : vector<64xi32>) -> vector<64xi8> {
  %0 = "aievec.aie4.srs"(%a) <{shift = 7 : i32, isSigned = true,
         rounding = #aievec.aie4_rnd<sym_zero>,
         saturation = #aievec.aie4_sat<truncate>}>
       : (vector<64xi32>) -> vector<64xi8>
  return %0 : vector<64xi8>
}
