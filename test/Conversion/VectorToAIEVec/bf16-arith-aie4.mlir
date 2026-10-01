// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// bf16 add and subtract run in f32: native widening, f32 op, generic truncf.
// fma is computed in f32 and rounded once.

// CHECK-LABEL: @addf_bf16_32
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>)
// CHECK: %[[WA:.*]] = aievec.aie4.conv_bf16_f32 %[[A]]
// CHECK: %[[WB:.*]] = aievec.aie4.conv_bf16_f32 %[[B]]
// CHECK: %[[S:.*]] = arith.addf %[[WA]], %[[WB]] : vector<32xf32>
// CHECK: %[[R:.*]] = arith.truncf %[[S]] : vector<32xf32> to vector<32xbf16>
// CHECK: return %[[R]]
func.func @addf_bf16_32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %r = arith.addf %a, %b : vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

// CHECK-LABEL: @subf_bf16_32
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>)
// CHECK: %[[WA:.*]] = aievec.aie4.conv_bf16_f32 %[[A]]
// CHECK: %[[WB:.*]] = aievec.aie4.conv_bf16_f32 %[[B]]
// CHECK: %[[D:.*]] = arith.subf %[[WA]], %[[WB]] : vector<32xf32>
// CHECK: arith.truncf %[[D]] : vector<32xf32> to vector<32xbf16>
func.func @subf_bf16_32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %r = arith.subf %a, %b : vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

// Fastmath flags carry over to the f32 operation.
// CHECK-LABEL: @addf_bf16_fastmath
// CHECK: arith.addf {{.*}} fastmath<fast> : vector<32xf32>
func.func @addf_bf16_fastmath(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %r = arith.addf %a, %b fastmath<fast> : vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

// A chain widens and narrows at every step, as the bf16 source rounds at
// every step.
// CHECK-LABEL: @chain
// CHECK-COUNT-2: arith.truncf
// CHECK-NOT: arith.addf {{.*}} : vector<32xbf16>
// CHECK-NOT: arith.subf {{.*}} : vector<32xbf16>
func.func @chain(%a: vector<32xbf16>, %b: vector<32xbf16>, %c: vector<32xbf16>) -> vector<32xbf16> {
  %s = arith.addf %a, %b : vector<32xbf16>
  %d = arith.subf %s, %c : vector<32xbf16>
  return %d : vector<32xbf16>
}

// -----

// fma rounds once: the f32 sum is rounded to odd using TwoSum's exact
// error, then narrowed.
// CHECK-LABEL: @fma_bf16_32
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>, %[[C:.*]]: vector<32xbf16>)
// CHECK-DAG: %[[ONE:.*]] = arith.constant dense<1> : vector<32xi32>
// CHECK-DAG: %[[MAXF:.*]] = arith.constant dense<2139095039> : vector<32xi32>
// CHECK-DAG: %[[ABS:.*]] = arith.constant dense<2147483647> : vector<32xi32>
// CHECK: %[[P:.*]] = aievec.aie4.mul_ewise_bf16_f32 %[[A]], %[[B]]
// CHECK: %[[WC:.*]] = aievec.aie4.conv_bf16_f32 %[[C]]
// CHECK: %[[S:.*]] = arith.addf %[[P]], %[[WC]] : vector<32xf32>
// CHECK: %[[BV:.*]] = arith.subf %[[S]], %[[P]]
// CHECK: %[[AV:.*]] = arith.subf %[[S]], %[[BV]]
// CHECK: %[[BR:.*]] = arith.subf %[[WC]], %[[BV]]
// CHECK: %[[AR:.*]] = arith.subf %[[P]], %[[AV]]
// CHECK: %[[E:.*]] = arith.addf %[[AR]], %[[BR]]
// CHECK: %[[SB:.*]] = arith.bitcast %[[S]] : vector<32xf32> to vector<32xi32>
// CHECK: %[[EB:.*]] = arith.bitcast %[[E]] : vector<32xf32> to vector<32xi32>
// CHECK: %[[EM:.*]] = arith.andi %[[EB]], %[[ABS]]
// CHECK: %[[EM1:.*]] = arith.subi %[[EM]], %[[ONE]]
// CHECK: %[[INEXACT:.*]] = aievec.aie4.cmp %[[EM1]], %[[MAXF]][pred = <lt>, signed = false]
// CHECK: arith.xori %[[SB]], %[[EB]]
// CHECK: %[[DOWN:.*]] = arith.subi %[[SB]], %{{.*}}
// CHECK: %[[ODD:.*]] = arith.ori %[[DOWN]], %[[ONE]]
// CHECK: %[[RB:.*]] = aievec.aie4.sel %[[SB]], %[[ODD]], %[[INEXACT]]
// CHECK: %[[R:.*]] = arith.bitcast %[[RB]] : vector<32xi32> to vector<32xf32>
// CHECK: arith.truncf %[[R]] : vector<32xf32> to vector<32xbf16>
// CHECK-NOT: vector.fma
// CHECK-NOT: mac_ewise
func.func @fma_bf16_32(%a: vector<32xbf16>, %b: vector<32xbf16>, %c: vector<32xbf16>) -> vector<32xbf16> {
  %r = vector.fma %a, %b, %c : vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

// Other shapes are not claimed.
// CHECK-LABEL: @fma_unclaimed
// CHECK: vector.fma {{.*}} : vector<16xbf16>
// CHECK: vector.fma {{.*}} : vector<32xf32>
func.func @fma_unclaimed(%a: vector<16xbf16>, %f: vector<32xf32>) -> (vector<16xbf16>, vector<32xf32>) {
  %0 = vector.fma %a, %a, %a : vector<16xbf16>
  %1 = vector.fma %f, %f, %f : vector<32xf32>
  return %0, %1 : vector<16xbf16>, vector<32xf32>
}

// -----

// Only the 32-lane rank-1 shape has the native conversions.
// CHECK-LABEL: @addsub_unclaimed
// CHECK: arith.addf {{.*}} : vector<64xbf16>
// CHECK: arith.subf {{.*}} : vector<2x32xbf16>
// CHECK: arith.addf {{.*}} : bf16
// CHECK-NOT: aievec.aie4.conv_bf16_f32
func.func @addsub_unclaimed(%a: vector<64xbf16>, %b: vector<2x32xbf16>, %s: bf16)
    -> (vector<64xbf16>, vector<2x32xbf16>, bf16) {
  %0 = arith.addf %a, %a : vector<64xbf16>
  %1 = arith.subf %b, %b : vector<2x32xbf16>
  %2 = arith.addf %s, %s : bf16
  return %0, %1, %2 : vector<64xbf16>, vector<2x32xbf16>, bf16
}

// -----

// An f32 MAC chain whose operands are widened bf16 still fuses; the bf16
// rewrite only claims adds whose result is bf16.
// CHECK-LABEL: @bf16_mac_chain_still_fuses
// CHECK: aievec.aie4.mac_ewise_bf16_f32
// CHECK-NOT: arith.truncf
func.func @bf16_mac_chain_still_fuses(%a: vector<32xbf16>, %b: vector<32xbf16>,
                                      %acc: vector<32xf32>) -> vector<32xf32> {
  %wa = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %wb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %wa, %wb fastmath<contract> : vector<32xf32>
  %r = arith.addf %acc, %m fastmath<contract> : vector<32xf32>
  return %r : vector<32xf32>
}
