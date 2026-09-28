// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s --implicit-check-not=aievec.aie4.fminmax --implicit-check-not=aievec.aie4.fcmp

// Native integer min/max, comparisons, and selects support five fixed
// rank-one shapes. Comparisons and selects pack or unpack Boolean lanes;
// unsupported shapes and equality predicates use generic lowering.

// -- MIN/MAX --

// CHECK-LABEL: @minsi_i8x64
func.func @minsi_i8x64(%a: vector<64xi8>, %b: vector<64xi8>) -> vector<64xi8> {
  // CHECK: aievec.aie4.minmax %{{.*}}, %{{.*}}[kind = <min>, signed = true] : vector<64xi8>
  %0 = arith.minsi %a, %b : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @maxui_i32x16
func.func @maxui_i32x16(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.aie4.minmax %{{.*}}, %{{.*}}[kind = <max>, signed = false] : vector<16xi32>
  %0 = arith.maxui %a, %b : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @minui_i16x64
func.func @minui_i16x64(%a: vector<64xi16>, %b: vector<64xi16>) -> vector<64xi16> {
  // CHECK: aievec.aie4.minmax %{{.*}}, %{{.*}}[kind = <min>, signed = false] : vector<64xi16>
  %0 = arith.minui %a, %b : vector<64xi16>
  return %0 : vector<64xi16>
}

// Unsupported shapes retain generic min/max.
// CHECK-LABEL: @minsi_out_of_scope
func.func @minsi_out_of_scope(%a: vector<8xi32>, %b: vector<8xi32>) -> vector<8xi32> {
  // CHECK: arith.minsi
  // CHECK-NOT: aievec.aie4.minmax
  %0 = arith.minsi %a, %b : vector<8xi32>
  return %0 : vector<8xi32>
}

// -----

// -- CMP --

// CHECK-LABEL: @cmpi_slt_i16x32
func.func @cmpi_slt_i16x32(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi1> {
  // CHECK: %[[C:.*]] = aievec.aie4.cmp %{{.*}}, %{{.*}}[pred = <lt>, signed = true] : vector<32xi16>, ui32
  // CHECK: %[[S:.*]] = builtin.unrealized_conversion_cast %[[C]] : ui32 to i32
  // CHECK: %[[V:.*]] = vector.broadcast %[[S]] : i32 to vector<1xi32>
  // CHECK: vector.bitcast %[[V]] : vector<1xi32> to vector<32xi1>
  %0 = arith.cmpi slt, %a, %b : vector<32xi16>
  return %0 : vector<32xi1>
}

// gt(a, b) lowers to lt(b, a).
// CHECK-LABEL: @cmpi_sgt_i16x32
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
func.func @cmpi_sgt_i16x32(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi1> {
  // CHECK: aievec.aie4.cmp %[[B]], %[[A]][pred = <lt>, signed = true] : vector<32xi16>, ui32
  %0 = arith.cmpi sgt, %a, %b : vector<32xi16>
  return %0 : vector<32xi1>
}

// Unsigned le(a, b) lowers to ge(b, a).
// CHECK-LABEL: @cmpi_ule_i32x16
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
func.func @cmpi_ule_i32x16(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi1> {
  // CHECK: aievec.aie4.cmp %[[B]], %[[A]][pred = <ge>, signed = false] : vector<16xi32>, ui32
  %0 = arith.cmpi ule, %a, %b : vector<16xi32>
  return %0 : vector<16xi1>
}

// Equality predicates retain generic comparison.
// CHECK-LABEL: @cmpi_eq_i16x32
func.func @cmpi_eq_i16x32(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi1> {
  // CHECK: arith.cmpi eq
  // CHECK-NOT: aievec.aie4.cmp
  %0 = arith.cmpi eq, %a, %b : vector<32xi16>
  return %0 : vector<32xi1>
}

// -----

// -- SELECT --

// CHECK-LABEL: @select_i32x32
// CHECK-SAME: %[[COND:.*]]: vector<32xi1>, %[[T:.*]]: vector<32xi32>, %[[F:.*]]: vector<32xi32>
func.func @select_i32x32(%cond: vector<32xi1>, %t: vector<32xi32>, %f: vector<32xi32>) -> vector<32xi32> {
  // CHECK: %[[V:.*]] = vector.bitcast %[[COND]] : vector<32xi1> to vector<1xi32>
  // CHECK: %[[S:.*]] = vector.extract %[[V]][0] : i32 from vector<1xi32>
  // CHECK: %[[SEL:.*]] = builtin.unrealized_conversion_cast %[[S]] : i32 to ui32
  // Native select takes the false value first and the true value second.
  // CHECK: aievec.aie4.sel %[[F]], %[[T]], %[[SEL]] : vector<32xi32>, ui32, vector<32xi32>
  %0 = arith.select %cond, %t, %f : vector<32xi1>, vector<32xi32>
  return %0 : vector<32xi32>
}

// Unsupported shapes retain generic select.
// CHECK-LABEL: @select_out_of_scope
func.func @select_out_of_scope(%cond: vector<8xi1>, %t: vector<8xi32>, %f: vector<8xi32>) -> vector<8xi32> {
  // CHECK: arith.select
  // CHECK-NOT: aievec.aie4.sel
  %0 = arith.select %cond, %t, %f : vector<8xi1>, vector<8xi32>
  return %0 : vector<8xi32>
}

// -----

// Float source operations use integer encodings, independent of crFPOrder.

// CHECK-LABEL: @minimumf_bf16x32
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<32641> : vector<32xi16>
// CHECK-DAG: %[[ONES:.*]] = arith.constant dense<-1> : vector<32xi16>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi16>
// CHECK-DAG: %[[MAG:.*]] = arith.constant dense<32767> : vector<32xi16>
// CHECK-DAG: %[[SIGN:.*]] = arith.constant dense<-32768> : vector<32xi16>
// CHECK: %[[AI:.*]] = vector.bitcast %[[A]] : vector<32xbf16> to vector<32xi16>
// CHECK: %[[BI:.*]] = vector.bitcast %[[B]] : vector<32xbf16> to vector<32xi16>
// CHECK: %[[AM:.*]] = arith.andi %[[AI]], %[[MAG]]
// CHECK: %[[BM:.*]] = arith.andi %[[BI]], %[[MAG]]
// Negative encodings are complemented; positive encodings flip only the sign.
// CHECK: %[[ANEG:.*]] = aievec.aie4.cmp %[[AI]], %[[ZERO]][pred = <lt>, signed = true]
// CHECK: %[[APOS:.*]] = arith.xori %[[AI]], %[[SIGN]]
// CHECK: %[[AINV:.*]] = arith.xori %[[AI]], %[[ONES]]
// CHECK: %[[AK:.*]] = aievec.aie4.sel %[[APOS]], %[[AINV]], %[[ANEG]]
// CHECK: %[[BNEG:.*]] = aievec.aie4.cmp %[[BI]], %[[ZERO]][pred = <lt>, signed = true]
// CHECK: %[[BPOS:.*]] = arith.xori %[[BI]], %[[SIGN]]
// CHECK: %[[BINV:.*]] = arith.xori %[[BI]], %[[ONES]]
// CHECK: %[[BK:.*]] = aievec.aie4.sel %[[BPOS]], %[[BINV]], %[[BNEG]]
// Classify each magnitude, then propagate a NaN from either input.
// CHECK: %[[ANUM:.*]] = aievec.aie4.cmp %[[AM]], %[[NAN]][pred = <lt>, signed = false]
// CHECK: %[[BNUM:.*]] = aievec.aie4.cmp %[[BM]], %[[NAN]][pred = <lt>, signed = false]
// CHECK: %[[LT:.*]] = aievec.aie4.cmp %[[AK]], %[[BK]][pred = <lt>, signed = false]
// CHECK: %[[MIN:.*]] = aievec.aie4.sel %[[BI]], %[[AI]], %[[LT]]
// CHECK: %[[BN:.*]] = aievec.aie4.sel %[[BI]], %[[MIN]], %[[BNUM]]
// CHECK: %[[AN:.*]] = aievec.aie4.sel %[[AI]], %[[BN]], %[[ANUM]]
// CHECK: vector.bitcast %[[AN]] : vector<32xi16> to vector<32xbf16>
func.func @minimumf_bf16x32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.minimumf %a, %b : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @maximumf_fp32x32
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: aievec.aie4.sel
func.func @maximumf_fp32x32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.maximumf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @minimumf_out_of_scope
// CHECK: arith.minimumf
func.func @minimumf_out_of_scope(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
  %0 = arith.minimumf %a, %b : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: @cmpf_olt_fp32x16
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: vector.bitcast {{.*}} to vector<{{[0-9]+}}xi1>
func.func @cmpf_olt_fp32x16(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xi1> {
  %0 = arith.cmpf olt, %a, %b : vector<16xf32>
  return %0 : vector<16xi1>
}

// CHECK-LABEL: @cmpf_ogt_bf16x32
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: vector.bitcast {{.*}} to vector<{{[0-9]+}}xi1>
func.func @cmpf_ogt_bf16x32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xi1> {
  %0 = arith.cmpf ogt, %a, %b : vector<32xbf16>
  return %0 : vector<32xi1>
}

// CHECK-LABEL: @cmpf_oeq_fp16x32
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: vector.bitcast {{.*}} to vector<{{[0-9]+}}xi1>
func.func @cmpf_oeq_fp16x32(%a: vector<32xf16>, %b: vector<32xf16>) -> vector<32xi1> {
  %0 = arith.cmpf oeq, %a, %b : vector<32xf16>
  return %0 : vector<32xi1>
}

// CHECK-LABEL: @cmpf_ord_fp32x32
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: vector.bitcast {{.*}} to vector<{{[0-9]+}}xi1>
func.func @cmpf_ord_fp32x32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xi1> {
  %0 = arith.cmpf ord, %a, %b : vector<32xf32>
  return %0 : vector<32xi1>
}

// CHECK-LABEL: @cmpf_uno_bf16x64
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: vector.bitcast {{.*}} to vector<{{[0-9]+}}xi1>
func.func @cmpf_uno_bf16x64(%a: vector<64xbf16>, %b: vector<64xbf16>) -> vector<64xi1> {
  %0 = arith.cmpf uno, %a, %b : vector<64xbf16>
  return %0 : vector<64xi1>
}

// CHECK-LABEL: @cmpf_ult_fp16x64
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: vector.bitcast {{.*}} to vector<{{[0-9]+}}xi1>
func.func @cmpf_ult_fp16x64(%a: vector<64xf16>, %b: vector<64xf16>) -> vector<64xi1> {
  %0 = arith.cmpf ult, %a, %b : vector<64xf16>
  return %0 : vector<64xi1>
}

// CHECK-LABEL: @cmpf_out_of_scope
// CHECK: arith.cmpf
func.func @cmpf_out_of_scope(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xi1> {
  %0 = arith.cmpf olt, %a, %b : vector<8xf32>
  return %0 : vector<8xi1>
}

// CHECK-LABEL: @minimumf_and_olt_not_fused
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: aievec.aie4.sel
// CHECK: vector.bitcast {{.*}} to vector<{{[0-9]+}}xi1>
func.func @minimumf_and_olt_not_fused(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>, vector<16xi1>) {
  %0 = arith.minimumf %a, %b : vector<16xf32>
  %1 = arith.cmpf olt, %a, %b : vector<16xf32>
  return %0, %1 : vector<16xf32>, vector<16xi1>
}

// CHECK-LABEL: @select_float_fp32x16
// CHECK-SAME: %[[COND:.*]]: vector<16xi1>, %[[T:.*]]: vector<16xf32>, %[[F:.*]]: vector<16xf32>
func.func @select_float_fp32x16(%cond: vector<16xi1>, %t: vector<16xf32>, %f: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[V:.*]] = vector.bitcast %[[COND]] : vector<16xi1> to vector<1xi16>
  // CHECK: %[[S:.*]] = vector.extract %[[V]][0] : i16 from vector<1xi16>
  // CHECK: %[[E:.*]] = arith.extui %[[S]] : i16 to i32
  // CHECK: %[[SEL:.*]] = builtin.unrealized_conversion_cast %[[E]] : i32 to ui32
  // CHECK: %[[FI:.*]] = vector.bitcast %[[F]] : vector<16xf32> to vector<16xi32>
  // CHECK: %[[TI:.*]] = vector.bitcast %[[T]] : vector<16xf32> to vector<16xi32>
  // Native select takes the false value first and the true value second.
  // CHECK: %[[RI:.*]] = aievec.aie4.sel %[[FI]], %[[TI]], %[[SEL]] : vector<16xi32>, ui32, vector<16xi32>
  // CHECK: vector.bitcast %[[RI]] : vector<16xi32> to vector<16xf32>
  %0 = arith.select %cond, %t, %f : vector<16xi1>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @minnumf_nnan
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: aievec.aie4.sel
func.func @minnumf_nnan(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.minnumf %a, %b fastmath<nnan> : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @minnumf_general
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: aievec.aie4.sel
func.func @minnumf_general(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.minnumf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @maxnumf_general
// CHECK: vector.bitcast
// CHECK: arith.andi
// CHECK: aievec.aie4.cmp
// CHECK: aievec.aie4.sel
func.func @maxnumf_general(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.maxnumf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @minnumf_out_of_scope
// CHECK: arith.minnumf
func.func @minnumf_out_of_scope(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
  %0 = arith.minnumf %a, %b : vector<8xf32>
  return %0 : vector<8xf32>
}
