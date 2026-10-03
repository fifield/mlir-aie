// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Placement and ordering after lowering, including under the generic
// transforms that run after convert-aievec-to-llvm in aiecc.
//
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" -canonicalize -cse -loop-invariant-code-motion | FileCheck %s

// Staging sits inside the K loop (one B tile per K step) and outside the
// row loop that reuses B; the commit precedes the row loop and its MMACs.
// B is loop-invariant here, yet nothing is hoisted out of the K loop.
// CHECK-LABEL: func.func @k_loop_row_loop
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK-COUNT-32: "xllvm.intr.aie4.insert.staging.bf16.I1024"
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I32768"
// CHECK-NEXT: scf.for
// CHECK-NOT: insert.staging
// CHECK-NOT: staging.to.matrix
// CHECK: "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK-NOT: insert.staging
// CHECK: return
func.func @k_loop_row_loop(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc_k = %c) -> (vector<4x64xf32>) {
    %out = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %acc_k : vector<4x64xf32> -> vector<4x64xf32> {
    ^bb0(%acc0: vector<4x64xf32>):
      %rows = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %acc0) -> (vector<4x64xf32>) {
        %a_i = vector.extract %a[%i] : vector<32xbf16> from vector<4x32xbf16>
        %c_i = vector.extract %acc[%i] : vector<64xf32> from vector<4x64xf32>
        %r_i = aievec.aie4.mmac bf16 %a_i, %c_i : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
        %next = vector.insert %r_i, %acc [%i] : vector<64xf32> into vector<4x64xf32>
        scf.yield %next : vector<4x64xf32>
      }
      aievec.aie4.matrix_yield %rows : vector<4x64xf32>
    }
    scf.yield %out : vector<4x64xf32>
  }
  return %res : vector<4x64xf32>
}

// -----

// Two identical regions stage B twice and commit twice; identical MMACs stay.
// CHECK-LABEL: func.func @no_cse
// CHECK-COUNT-32: insert.staging.int.I1024
// CHECK-NEXT: staging.to.matrix.I32768
// CHECK-NEXT: "xllvm.intr.aie4.I512.ACC2048.mmac"
// CHECK-COUNT-32: insert.staging.int.I1024
// CHECK-NEXT: staging.to.matrix.I32768
// CHECK-NEXT: "xllvm.intr.aie4.I512.ACC2048.mmac"
// CHECK-NOT: insert.staging
// CHECK: return
func.func @no_cse(%a: vector<64xi8>, %b: vector<64x64xi8>, %c: vector<64xi32>) -> (vector<64xi32>, vector<64xi32>) {
  %r0 = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = aievec.aie4.mmac int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  %r1 = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = aievec.aie4.mmac int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r0, %r1 : vector<64xi32>, vector<64xi32>
}

// -----

// Unused results do not make staging, commit or MMACs dead.
// CHECK-LABEL: func.func @no_dce
// CHECK-COUNT-32: insert.staging.fp8.I1024
// CHECK-NEXT: staging.to.matrix.FP32768
// CHECK: "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"
// CHECK: return
func.func @no_dce(%a: vector<64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>) {
  %r = aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> -> vector<64xf32> {
    %x = aievec.aie4.mmac fp8 %a : vector<64xf8E4M3FN> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return
}
