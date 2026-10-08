// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -aievec-aie4-expand-matrix-region 2>/dev/null | FileCheck %s
// RUN: aie-opt %s -split-input-file -aievec-aie4-expand-matrix-region 2>/dev/null > %t.once
// RUN: aie-opt %t.once -split-input-file -aievec-aie4-expand-matrix-region > %t.twice
// RUN: diff %t.once %t.twice
// RUN: aie-opt %s -split-input-file -aievec-aie4-expand-matrix-region -verify-diagnostics -o /dev/null

// matrix_region expands in place into 32 stages of two B columns each, the
// commit, and the body with each mmac replaced by mmac_active.

// CHECK-LABEL: func.func @bf16
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32x64xbf16>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK-NEXT: %[[L0:.*]] = vector.extract %[[BT]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK-NEXT: %[[H0:.*]] = vector.extract %[[BT]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK-NEXT: %[[K0:.*]] = vector.shuffle %[[L0]], %[[H0]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: %[[S0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: aievec.aie4.stage bf16 %[[K0]], %[[S0]] : vector<64xbf16>, i32
// CHECK-COUNT-30: aievec.aie4.stage bf16
// CHECK: vector.extract %[[BT]][62]
// CHECK-NEXT: vector.extract %[[BT]][63]
// CHECK-NEXT: vector.shuffle
// CHECK-NEXT: %[[S31:.*]] = arith.constant 31 : i32
// CHECK-NEXT: aievec.aie4.stage bf16 %{{.*}}, %[[S31]] : vector<64xbf16>, i32
// CHECK-NEXT: aievec.aie4.commit bf16
// CHECK-NEXT: %[[R:.*]] = aievec.aie4.mmac_active bf16 %[[A]], %[[C]] : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
// CHECK-NEXT: return %[[R]]
// CHECK-NOT: aievec.aie4.matrix_region
// CHECK-NOT: aievec.aie4.mmac{{ }}
func.func @bf16(%a: vector<32xbf16>, %b: vector<32x64xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  // expected-warning @+1 {{B tile is not produced by a vector.transpose [1, 0] of a column-major tile}}
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    %x = aievec.aie4.mmac bf16 %a, %acc : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// int8: rhs_signed goes on every stage and the commit; row MMAC attributes
// are preserved and no zero_if is added. A column-major B gives no warning.
// CHECK-LABEL: func.func @int8_unsigned_b
// CHECK: vector.transpose
// CHECK: vector.shuffle {{.*}} : vector<64xi8>, vector<64xi8>
// CHECK-NEXT: arith.constant 0 : i32
// CHECK-NEXT: aievec.aie4.stage int8 %{{.*}}, %{{.*}} {rhs_signed = false} : vector<128xi8>, i32
// CHECK-COUNT-31: aievec.aie4.stage int8 %{{.*}}, %{{.*}} {rhs_signed = false} : vector<128xi8>, i32
// CHECK-NEXT: aievec.aie4.commit int8 {rhs_signed = false}
// CHECK-NEXT: aievec.aie4.mmac_active int8 %{{.*}}, %{{.*}} {lhs_signed = true, rhs_signed = false} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
// CHECK-NOT: zero_if
func.func @int8_unsigned_b(%a: vector<64xi8>, %bt: vector<64x64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %b = vector.transpose %bt, [1, 0] : vector<64x64xi8> to vector<64x64xi8>
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = false} {
    %x = aievec.aie4.mmac int8 %a, %c {lhs_signed = true, rhs_signed = false} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// int4 and fp8 columns are bitcast to bytes before the shuffle, and the
// stage takes the vector<128xi8> byte image.
// CHECK-LABEL: func.func @int4
// CHECK: %[[BT:.*]] = vector.transpose
// CHECK-NEXT: %[[L0:.*]] = vector.extract %[[BT]][0] : vector<128xi4> from vector<64x128xi4>
// CHECK-NEXT: %[[L0B:.*]] = vector.bitcast %[[L0]] : vector<128xi4> to vector<64xi8>
// CHECK-NEXT: %[[H0:.*]] = vector.extract %[[BT]][1] : vector<128xi4> from vector<64x128xi4>
// CHECK-NEXT: %[[H0B:.*]] = vector.bitcast %[[H0]] : vector<128xi4> to vector<64xi8>
// CHECK-NEXT: %[[K0:.*]] = vector.shuffle %[[L0B]], %[[H0B]]
// CHECK-NEXT: %[[S0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: aievec.aie4.stage int4 %[[K0]], %[[S0]] {rhs_signed = true} : vector<128xi8>, i32
// CHECK-COUNT-31: aievec.aie4.stage int4
// CHECK-NEXT: aievec.aie4.commit int4 {rhs_signed = true}
// Zero-acc row MMAC (no acc) stays without acc.
// CHECK-NEXT: aievec.aie4.mmac_active int4 %{{.*}} {lhs_signed = false, rhs_signed = true} : vector<128xi4> -> vector<64xi32>
func.func @int4(%a: vector<128xi4>, %b: vector<128x64xi4>) -> vector<64xi32> {
  // expected-warning @+1 {{B tile is not produced by a vector.transpose}}
  %r = aievec.aie4.matrix_region int4 %b : vector<128x64xi4> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = aievec.aie4.mmac int4 %a {lhs_signed = false, rhs_signed = true} : vector<128xi4> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @fp8
// CHECK: vector.bitcast %{{.*}} : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: vector.bitcast %{{.*}} : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: aievec.aie4.stage fp8 %{{.*}}, %{{.*}} : vector<128xi8>, i32
// CHECK-COUNT-31: aievec.aie4.stage fp8
// CHECK-NEXT: aievec.aie4.commit fp8
// CHECK-NEXT: aievec.aie4.mmac_active fp8 %{{.*}}, %{{.*}} : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
func.func @fp8(%a: vector<64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c: vector<64xf32>) -> vector<64xf32> {
  // expected-warning @+1 {{B tile is not produced by a vector.transpose}}
  %r = aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> -> vector<64xf32> {
    %x = aievec.aie4.mmac fp8 %a, %c : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// A yielded block argument becomes the corresponding init.
// CHECK-LABEL: func.func @yield_block_arg
// CHECK-SAME: (%{{.*}}: vector<32x64xbf16>, %[[C:.*]]: vector<64xf32>)
// CHECK: aievec.aie4.commit bf16
// CHECK-NEXT: return %[[C]]
func.func @yield_block_arg(%b: vector<32x64xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  // expected-warning @+1 {{B tile is not produced by a vector.transpose}}
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    aievec.aie4.matrix_yield %acc : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// Staging stays inside the K loop; the body's row loop is inlined after the
// commit with its MMACs rewritten in place.
// CHECK-LABEL: func.func @k_loop_row_loop
// CHECK: scf.for %{{.*}} iter_args(%[[ACCK:.*]] = %{{.*}})
// CHECK-COUNT-32: aievec.aie4.stage bf16
// CHECK-NEXT: aievec.aie4.commit bf16
// CHECK-NEXT: %[[ROWS:.*]] = scf.for %{{.*}} iter_args(%[[ACC:.*]] = %[[ACCK]])
// CHECK: aievec.aie4.mmac_active bf16
// CHECK: scf.yield
// CHECK: scf.yield %[[ROWS]]
func.func @k_loop_row_loop(%a: vector<4x32xbf16>, %bt: vector<64x32xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %b = vector.transpose %bt, [1, 0] : vector<64x32xbf16> to vector<32x64xbf16>
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

// matmul is left alone; it is expanded by aievec-aie4-expand-matmul.
// CHECK-LABEL: func.func @matmul_untouched
// CHECK: aievec.aie4.matmul
// CHECK-NOT: aievec.aie4.stage
func.func @matmul_untouched(%a: vector<2x32xbf16>, %b: vector<32x64xbf16>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %r = aievec.aie4.matmul %a, %b, %c : vector<2x32xbf16>, vector<32x64xbf16> into vector<2x64xf32>
  return %r : vector<2x64xf32>
}
