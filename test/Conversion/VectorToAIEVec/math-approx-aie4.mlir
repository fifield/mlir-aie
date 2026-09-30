// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// afn exp2/tanh preserve result types. exp2 adds the result-specific bias
// with a flag-free addf: 895 + 2^-12 for f32, vendor 895.001 for bf16.

// CHECK-LABEL: @exp2_f32
// CHECK-SAME: (%[[X:.*]]: vector<32xf32>)
// CHECK: %[[BIAS:.*]] = arith.constant dense<895.000244> : vector<32xf32>
// CHECK: %[[B:.*]] = arith.addf %[[X]], %[[BIAS]] : vector<32xf32>
// CHECK: %[[R:.*]] = aievec.aie4.biased_exp2 %[[B]] : vector<32xf32>, vector<32xf32>
// CHECK: return %[[R]] : vector<32xf32>
func.func @exp2_f32(%x: vector<32xf32>) -> vector<32xf32> {
  %0 = math.exp2 %x fastmath<afn> : vector<32xf32>
  return %0 : vector<32xf32>
}

// A bf16 source widens exactly, adds the bf16-form bias in f32, and uses the
// bf16 form.
// CHECK-LABEL: @exp2_bf16
// CHECK-SAME: (%[[X:.*]]: vector<32xbf16>)
// CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<895.000976> : vector<32xf32>
// CHECK-DAG: %[[W:.*]] = aievec.aie4.conv_bf16_f32 %[[X]] : vector<32xbf16>, vector<32xf32>
// CHECK: %[[B:.*]] = arith.addf %[[W]], %[[BIAS]] : vector<32xf32>
// CHECK: %[[R:.*]] = aievec.aie4.biased_exp2 %[[B]] : vector<32xf32>, vector<32xbf16>
// CHECK: return %[[R]] : vector<32xbf16>
func.func @exp2_bf16(%x: vector<32xbf16>) -> vector<32xbf16> {
  %0 = math.exp2 %x fastmath<afn> : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// VTANH produces bf16; an f32 result widens it back exactly.
// CHECK-LABEL: @tanh_f32
// CHECK-SAME: (%[[X:.*]]: vector<32xf32>)
// CHECK: %[[T:.*]] = aievec.aie4.tanh %[[X]] : vector<32xf32>, vector<32xbf16>
// CHECK: %[[R:.*]] = aievec.aie4.conv_bf16_f32 %[[T]] : vector<32xbf16>, vector<32xf32>
// CHECK: return %[[R]] : vector<32xf32>
func.func @tanh_f32(%x: vector<32xf32>) -> vector<32xf32> {
  %0 = math.tanh %x fastmath<afn> : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @tanh_bf16
// CHECK-SAME: (%[[X:.*]]: vector<32xbf16>)
// CHECK: %[[W:.*]] = aievec.aie4.conv_bf16_f32 %[[X]] : vector<32xbf16>, vector<32xf32>
// CHECK: %[[R:.*]] = aievec.aie4.tanh %[[W]] : vector<32xf32>, vector<32xbf16>
// CHECK: return %[[R]] : vector<32xbf16>
func.func @tanh_bf16(%x: vector<32xbf16>) -> vector<32xbf16> {
  %0 = math.tanh %x fastmath<afn> : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// `fast` includes afn; afn does not require nnan/ninf. The bias add does not
// inherit the source flags.
// CHECK-LABEL: @exp2_fast
// CHECK: arith.addf %{{[[:alnum:]_]+}}, %{{[[:alnum:]_]+}} : vector<32xf32>
// CHECK: aievec.aie4.biased_exp2
func.func @exp2_fast(%x: vector<32xf32>) -> vector<32xf32> {
  %0 = math.exp2 %x fastmath<fast> : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @tanh_afn_only
// CHECK: aievec.aie4.tanh
func.func @tanh_afn_only(%x: vector<32xf32>) -> vector<32xf32> {
  %0 = math.tanh %x fastmath<afn,contract> : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// Wider vectors split into 32-lane chunks and are reassembled in order.

// CHECK-LABEL: @exp2_f32_64
// CHECK-SAME: (%[[X:.*]]: vector<64xf32>)
// CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<895.000244> : vector<32xf32>
// CHECK-DAG: %[[P:.*]] = ub.poison : vector<64xf32>
// CHECK: %[[X0:.*]] = vector.extract_strided_slice %[[X]] offsets = [0], sizes = [32], strides = [1] : vector<64xf32> to vector<32xf32>
// CHECK: %[[B0:.*]] = arith.addf %[[X0]], %[[BIAS]]
// CHECK: %[[E0:.*]] = aievec.aie4.biased_exp2 %[[B0]] : vector<32xf32>, vector<32xf32>
// CHECK: %[[I0:.*]] = vector.insert_strided_slice %[[E0]], %[[P]] offsets = [0], strides = [1] : vector<32xf32> into vector<64xf32>
// CHECK: %[[X1:.*]] = vector.extract_strided_slice %[[X]] offsets = [32], sizes = [32], strides = [1] : vector<64xf32> to vector<32xf32>
// CHECK: %[[B1:.*]] = arith.addf %[[X1]], %[[BIAS]]
// CHECK: %[[E1:.*]] = aievec.aie4.biased_exp2 %[[B1]] : vector<32xf32>, vector<32xf32>
// CHECK: %[[I1:.*]] = vector.insert_strided_slice %[[E1]], %[[I0]] offsets = [32], strides = [1] : vector<32xf32> into vector<64xf32>
// CHECK: return %[[I1]] : vector<64xf32>
func.func @exp2_f32_64(%x: vector<64xf32>) -> vector<64xf32> {
  %0 = math.exp2 %x fastmath<afn> : vector<64xf32>
  return %0 : vector<64xf32>
}

// CHECK-LABEL: @tanh_bf16_96
// CHECK-COUNT-3: aievec.aie4.tanh {{.*}} : vector<32xf32>, vector<32xbf16>
// CHECK-NOT: aievec.aie4.tanh
// CHECK: return {{.*}} : vector<96xbf16>
func.func @tanh_bf16_96(%x: vector<96xbf16>) -> vector<96xbf16> {
  %0 = math.tanh %x fastmath<afn> : vector<96xbf16>
  return %0 : vector<96xbf16>
}
