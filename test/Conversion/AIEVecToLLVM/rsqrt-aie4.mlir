// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// afn rsqrt is INVSQRT (fast includes afn). The ISA has no vector invsqrt,
// so the scalar case is the direct one and every vector unrolls into it.

// CHECK-LABEL: @rsqrt_f32
func.func @rsqrt_f32(%a: f32) -> f32 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.invsqrt"(%{{.*}}) : (f32) -> f32
  // CHECK: return %[[R]]
  %0 = math.rsqrt %a fastmath<afn> : f32
  return %0 : f32
}

// -----

// Four lanes, short enough to spell out every step of the unrolling.

// CHECK-LABEL: @rsqrt_v4f32
func.func @rsqrt_v4f32(%a: vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.poison : vector<4xf32>
  // CHECK: %[[I0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[E0:.*]] = llvm.extractelement %{{.*}}[%[[I0]] : i64]
  // CHECK: %[[R0:.*]] = "xllvm.intr.aie4.invsqrt"(%[[E0]]) : (f32) -> f32
  // CHECK: %[[V0:.*]] = llvm.insertelement %[[R0]], %[[UNDEF]][%[[I0]] : i64]
  // CHECK: %[[I1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: "xllvm.intr.aie4.invsqrt"
  // CHECK: %[[I2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: "xllvm.intr.aie4.invsqrt"
  // CHECK: %[[I3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: "xllvm.intr.aie4.invsqrt"
  %0 = math.rsqrt %a fastmath<afn> : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// AIE4's f32 vector width, and the lane count to check does not blow up.

// CHECK-LABEL: @rsqrt_v32f32
// CHECK-COUNT-32: "xllvm.intr.aie4.invsqrt"
// CHECK-NOT: "xllvm.intr.aie4.invsqrt"
func.func @rsqrt_v32f32(%a: vector<32xf32>) -> vector<32xf32> {
  %0 = math.rsqrt %a fastmath<fast> : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// No AIE4 intrinsic takes bf16, so the op is left alone rather than lowered
// to the f32 one.

// CHECK-LABEL: @rsqrt_v32bf16_untouched
// CHECK: math.rsqrt
// CHECK-NOT: xllvm.intr.aie4.invsqrt
func.func @rsqrt_v32bf16_untouched(%a: vector<32xbf16>) -> vector<32xbf16> {
  %0 = math.rsqrt %a fastmath<afn> : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// The type converter keeps multidimensional vector types, and the LLVM ops
// below take only flat ones, so the unrolling is bracketed by shape casts.

// CHECK-LABEL: @rsqrt_v2x4f32
// CHECK: %[[FLAT:.*]] = vector.shape_cast %{{.*}} : vector<2x4xf32> to vector<8xf32>
// CHECK: llvm.mlir.poison : vector<8xf32>
// CHECK-COUNT-8: "xllvm.intr.aie4.invsqrt"
// CHECK: vector.shape_cast %{{.*}} : vector<8xf32> to vector<2x4xf32>
func.func @rsqrt_v2x4f32(%a: vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = math.rsqrt %a fastmath<afn> : vector<2x4xf32>
  return %0 : vector<2x4xf32>
}
