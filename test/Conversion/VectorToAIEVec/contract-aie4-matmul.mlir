// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Full pipeline: canonicalize-vector-for-aievec, then lowering.
// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s
// Lowering only, without ExtractTransposeFromContractionOp.
// RUN: aie-opt %s -split-input-file --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s --check-prefix=LOWER

// vector.contract -> aievec.aie4.matmul. A contraction
// converts only when it is exactly one native B tile (K x 64) with plain
// matmul maps, its operands are exact widenings of the native element type,
// and, for floating formats, its fastmath flags include `afn`. Everything
// else stays vector.contract.

//===----------------------------------------------------------------------===//
// bf16: permission
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @bf16_afn(
// CHECK-SAME: %[[A:.*]]: vector<4x32xbf16>, %[[B:.*]]: vector<32x64xbf16>, %[[C:.*]]: vector<4x64xf32>
// CHECK: %[[R:.*]] = aievec.aie4.matmul %[[A]], %[[B]], %[[C]] : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
// CHECK-NOT: vector.contract
// CHECK: return %[[R]]
func.func @bf16_afn(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x32xbf16> to vector<4x32xf32>
  %bw = arith.extf %b : vector<32x64xbf16> to vector<32x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<4x32xf32>, vector<32x64xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// `fast` includes `afn`. M = 1 is one row.
// CHECK-LABEL: func.func @bf16_fast_one_row(
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} : vector<1x32xbf16>, vector<32x64xbf16> into vector<1x64xf32>
// CHECK-NOT: vector.contract
func.func @bf16_fast_one_row(%a: vector<1x32xbf16>, %b: vector<32x64xbf16>, %c: vector<1x64xf32>) -> vector<1x64xf32> {
  %aw = arith.extf %a : vector<1x32xbf16> to vector<1x32xf32>
  %bw = arith.extf %b : vector<32x64xbf16> to vector<32x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<fast>}
       %aw, %bw, %c : vector<1x32xf32>, vector<32x64xf32> into vector<1x64xf32>
  return %r : vector<1x64xf32>
}

// -----

// MMAC flushes subnormal f32 accumulators, so a strict contraction cannot
// convert based on shape alone.
// CHECK-LABEL: func.func @bf16_strict(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @bf16_strict(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x32xbf16> to vector<4x32xf32>
  %bw = arith.extf %b : vector<32x64xbf16> to vector<32x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<4x32xf32>, vector<32x64xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// `contract` (and `reassoc`) without `afn` is not the permission.
// CHECK-LABEL: func.func @bf16_contract_only(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @bf16_contract_only(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x32xbf16> to vector<4x32xf32>
  %bw = arith.extf %b : vector<32x64xbf16> to vector<32x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<reassoc,contract>}
       %aw, %bw, %c : vector<4x32xf32>, vector<32x64xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// Operands already in bf16, without an extf, are not matched.
// CHECK-LABEL: func.func @bf16_unextended(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @bf16_unextended(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Integer formats: exact products, i32 wraparound; no permission needed.
// The extension kind gives each operand's signedness.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @i8_signed_signed(
// CHECK-SAME: %[[A:.*]]: vector<2x64xi8>, %[[B:.*]]: vector<64x64xi8>, %[[C:.*]]: vector<2x64xi32>
// CHECK: aievec.aie4.matmul %[[A]], %[[B]], %[[C]] {lhs_signed = true, rhs_signed = true} : vector<2x64xi8>, vector<64x64xi8> into vector<2x64xi32>
// CHECK-NOT: vector.contract
func.func @i8_signed_signed(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// CHECK-LABEL: func.func @i8_signed_unsigned(
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} {lhs_signed = true, rhs_signed = false} : vector<2x64xi8>, vector<64x64xi8> into vector<2x64xi32>
// CHECK-NOT: vector.contract
func.func @i8_signed_unsigned(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extui %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// CHECK-LABEL: func.func @i8_unsigned_signed(
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} {lhs_signed = false, rhs_signed = true} : vector<2x64xi8>, vector<64x64xi8> into vector<2x64xi32>
// CHECK-NOT: vector.contract
func.func @i8_unsigned_signed(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extui %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// CHECK-LABEL: func.func @i8_unsigned_unsigned(
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} {lhs_signed = false, rhs_signed = false} : vector<2x64xi8>, vector<64x64xi8> into vector<2x64xi32>
// CHECK-NOT: vector.contract
func.func @i8_unsigned_unsigned(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extui %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extui %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// A signless i8 contraction has no signedness to give the MMAC.
// CHECK-LABEL: func.func @i8_unextended(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @i8_unextended(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %a, %b, %c : vector<2x64xi8>, vector<64x64xi8> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// The matcher requires operands widened directly to i32; widening through
// i16 first is a conservative restriction, not a correctness requirement
// (signed i8 x i8 products are exact in i16 too).
// CHECK-LABEL: func.func @i8_via_i16(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @i8_via_i16(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi16>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi16>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi16>, vector<64x64xi16> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// Only kind = add is a matrix multiplication.
// CHECK-LABEL: func.func @i8_kind_maxsi(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @i8_kind_maxsi(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<maxsi>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// CHECK-LABEL: func.func @i4_signed_unsigned(
// CHECK-SAME: %[[A:.*]]: vector<3x128xi4>, %[[B:.*]]: vector<128x64xi4>, %[[C:.*]]: vector<3x64xi32>
// CHECK: aievec.aie4.matmul %[[A]], %[[B]], %[[C]] {lhs_signed = true, rhs_signed = false} : vector<3x128xi4>, vector<128x64xi4> into vector<3x64xi32>
// CHECK-NOT: vector.contract
func.func @i4_signed_unsigned(%a: vector<3x128xi4>, %b: vector<128x64xi4>, %c: vector<3x64xi32>) -> vector<3x64xi32> {
  %aw = arith.extsi %a : vector<3x128xi4> to vector<3x128xi32>
  %bw = arith.extui %b : vector<128x64xi4> to vector<128x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<3x128xi32>, vector<128x64xi32> into vector<3x64xi32>
  return %r : vector<3x64xi32>
}

// -----

// int4 with the int8 K (64) is not a native tile.
// CHECK-LABEL: func.func @i4_wrong_k(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @i4_wrong_k(%a: vector<3x64xi4>, %b: vector<64x64xi4>, %c: vector<3x64xi32>) -> vector<3x64xi32> {
  %aw = arith.extsi %a : vector<3x64xi4> to vector<3x64xi32>
  %bw = arith.extsi %b : vector<64x64xi4> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<3x64xi32>, vector<64x64xi32> into vector<3x64xi32>
  return %r : vector<3x64xi32>
}

// -----

//===----------------------------------------------------------------------===//
// fp8 (E4M3FN)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fp8_afn(
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} : vector<2x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<2x64xf32>
// CHECK-NOT: vector.contract
func.func @fp8_afn(%a: vector<2x64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %aw = arith.extf %a : vector<2x64xf8E4M3FN> to vector<2x64xf32>
  %bw = arith.extf %b : vector<64x64xf8E4M3FN> to vector<64x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<2x64xf32>, vector<64x64xf32> into vector<2x64xf32>
  return %r : vector<2x64xf32>
}

// -----

// CHECK-LABEL: func.func @fp8_strict(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @fp8_strict(%a: vector<2x64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %aw = arith.extf %a : vector<2x64xf8E4M3FN> to vector<2x64xf32>
  %bw = arith.extf %b : vector<64x64xf8E4M3FN> to vector<64x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xf32>, vector<64x64xf32> into vector<2x64xf32>
  return %r : vector<2x64xf32>
}

// -----

// AIE4 fp8 MMAC requires E4M3FN.
// CHECK-LABEL: func.func @fp8_e5m2(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @fp8_e5m2(%a: vector<2x64xf8E5M2>, %b: vector<64x64xf8E5M2>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %aw = arith.extf %a : vector<2x64xf8E5M2> to vector<2x64xf32>
  %bw = arith.extf %b : vector<64x64xf8E5M2> to vector<64x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<2x64xf32>, vector<64x64xf32> into vector<2x64xf32>
  return %r : vector<2x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Shapes and maps
//===----------------------------------------------------------------------===//

// K = 64 is two bf16 tiles; producers must tile first.
// CHECK-LABEL: func.func @bf16_wrong_k(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @bf16_wrong_k(%a: vector<4x64xbf16>, %b: vector<64x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x64xbf16> to vector<4x64xf32>
  %bw = arith.extf %b : vector<64x64xbf16> to vector<64x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<4x64xf32>, vector<64x64xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// N = 32 is half a tile.
// CHECK-LABEL: func.func @bf16_wrong_n(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @bf16_wrong_n(%a: vector<4x32xbf16>, %b: vector<32x32xbf16>, %c: vector<4x32xf32>) -> vector<4x32xf32> {
  %aw = arith.extf %a : vector<4x32xbf16> to vector<4x32xf32>
  %bw = arith.extf %b : vector<32x32xbf16> to vector<32x32xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<4x32xf32>, vector<32x32xf32> into vector<4x32xf32>
  return %r : vector<4x32xf32>
}

// -----

// A transposed A map is rejected.
// CHECK-LABEL: func.func @bf16_transposed_a(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
func.func @bf16_transposed_a(%a: vector<32x4xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<32x4xbf16> to vector<32x4xf32>
  %bw = arith.extf %b : vector<32x64xbf16> to vector<32x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (k, m)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<32x4xf32>, vector<32x64xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// A transposed B map: the matcher rejects it (LOWER), and the full
// pipeline first makes the transpose explicit (keeping afn), after which
// the contraction matches with a transposed rhs.
// CHECK-LABEL: func.func @bf16_transposed_b(
// CHECK-SAME: %[[A:.*]]: vector<4x32xbf16>, %[[BT:.*]]: vector<64x32xbf16>, %[[C:.*]]: vector<4x64xf32>
// CHECK: %[[B:.*]] = vector.transpose %[[BT]], [1, 0] : vector<64x32xbf16> to vector<32x64xbf16>
// CHECK: aievec.aie4.matmul %[[A]], %[[B]], %[[C]] : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
// CHECK-NOT: vector.contract
// LOWER-LABEL: func.func @bf16_transposed_b(
// LOWER-NOT: aievec.aie4.matmul
// LOWER: vector.contract
// LOWER-NOT: aievec.aie4.matmul
func.func @bf16_transposed_b(%a: vector<4x32xbf16>, %bt: vector<64x32xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x32xbf16> to vector<4x32xf32>
  %bw = arith.extf %bt : vector<64x32xbf16> to vector<64x32xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<4x32xf32>, vector<64x32xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// Transposing B does not grant permission.
// CHECK-LABEL: func.func @bf16_transposed_b_strict(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @bf16_transposed_b_strict(%a: vector<4x32xbf16>, %bt: vector<64x32xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x32xbf16> to vector<4x32xf32>
  %bw = arith.extf %bt : vector<64x32xbf16> to vector<64x32xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<4x32xf32>, vector<64x32xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// Square int8 tiles: a transposed map has native shapes, so only the map
// check rejects it. For B, the full pipeline makes the transpose explicit
// (above the extension) and then matches.
// CHECK-LABEL: func.func @i8_transposed_b_square(
// CHECK-SAME: %[[A:.*]]: vector<2x64xi8>, %[[BT:.*]]: vector<64x64xi8>, %[[C:.*]]: vector<2x64xi32>
// CHECK: %[[B:.*]] = vector.transpose %[[BT]], [1, 0] : vector<64x64xi8> to vector<64x64xi8>
// CHECK: aievec.aie4.matmul %[[A]], %[[B]], %[[C]] {lhs_signed = true, rhs_signed = false}
// CHECK-NOT: vector.contract
// LOWER-LABEL: func.func @i8_transposed_b_square(
// LOWER-NOT: aievec.aie4.matmul
// LOWER: vector.contract
// LOWER-NOT: aievec.aie4.matmul
func.func @i8_transposed_b_square(%a: vector<2x64xi8>, %bt: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extui %bt : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// CHECK-LABEL: func.func @i8_transposed_a_square(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
// LOWER-LABEL: func.func @i8_transposed_a_square(
// LOWER-NOT: aievec.aie4.matmul
// LOWER: vector.contract
// LOWER-NOT: aievec.aie4.matmul
func.func @i8_transposed_a_square(%at: vector<64x64xi8>, %b: vector<64x64xi8>, %c: vector<64x64xi32>) -> vector<64x64xi32> {
  %aw = arith.extsi %at : vector<64x64xi8> to vector<64x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (k, m)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<64x64xi32>, vector<64x64xi32> into vector<64x64xi32>
  return %r : vector<64x64xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Other rejections
//===----------------------------------------------------------------------===//

// f16 is not a native MMAC input format: only bf16,
// f8E4M3FN, i8 and i4 are. `fast` (which includes `afn`) does not change that.
// CHECK-LABEL: func.func @f16_not_native(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @f16_not_native(%a: vector<4x32xf16>, %b: vector<32x64xf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x32xf16> to vector<4x32xf32>
  %bw = arith.extf %b : vector<32x64xf16> to vector<32x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<fast>}
       %aw, %bw, %c : vector<4x32xf32>, vector<32x64xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// A scalable K is rejected: the matcher only handles static native tiles.
// CHECK-LABEL: func.func @scalable_k(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @scalable_k(%a: vector<4x[64]xi8>, %b: vector<[64]x64xi8>, %c: vector<4x64xi32>) -> vector<4x64xi32> {
  %aw = arith.extsi %a : vector<4x[64]xi8> to vector<4x[64]xi32>
  %bw = arith.extsi %b : vector<[64]x64xi8> to vector<[64]x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<4x[64]xi32>, vector<[64]x64xi32> into vector<4x64xi32>
  return %r : vector<4x64xi32>
}

// -----

// extsi i8->i16 then extui i16->i32: the matcher only peels the single
// widening op that feeds the contraction directly, so this stops at i16,
// which is not a native MMAC element type.
// CHECK-LABEL: func.func @i8_via_i16_mixed_chain(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @i8_via_i16_mixed_chain(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %a16 = arith.extsi %a : vector<2x64xi8> to vector<2x64xi16>
  %aw = arith.extui %a16 : vector<2x64xi16> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// Operands are block arguments with no widening op at all.
// CHECK-LABEL: func.func @blockarg_no_widening(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @blockarg_no_widening(%a: vector<4x64xi32>, %b: vector<64x64xi32>, %c: vector<4x64xi32>) -> vector<4x64xi32> {
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %a, %b, %c : vector<4x64xi32>, vector<64x64xi32> into vector<4x64xi32>
  return %r : vector<4x64xi32>
}

// -----

// The matcher alone accepts the plain form (LOWER).
// LOWER-LABEL: func.func @lower_only_accepts(
// LOWER: aievec.aie4.matmul
// LOWER-NOT: vector.contract
// CHECK-LABEL: func.func @lower_only_accepts(
// CHECK: aievec.aie4.matmul
func.func @lower_only_accepts(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %aw = arith.extf %a : vector<4x32xbf16> to vector<4x32xf32>
  %bw = arith.extf %b : vector<32x64xbf16> to vector<32x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
       %aw, %bw, %c : vector<4x32xf32>, vector<32x64xf32> into vector<4x64xf32>
  return %r : vector<4x64xf32>
}

// -----

// M is bounded by the accumulator rows the lowering keeps live: 64 rows
// convert, 65 stay vector.contract (kAIE4MatMulMaxRows).
// LOWER-LABEL: func.func @i8_max_rows(
// CHECK-LABEL: func.func @i8_max_rows(
// CHECK: aievec.aie4.matmul {{.*}} : vector<64x64xi8>, vector<64x64xi8> into vector<64x64xi32>
// CHECK-NOT: vector.contract
func.func @i8_max_rows(%a: vector<64x64xi8>, %b: vector<64x64xi8>, %c: vector<64x64xi32>) -> vector<64x64xi32> {
  %aw = arith.extsi %a : vector<64x64xi8> to vector<64x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<64x64xi32>, vector<64x64xi32> into vector<64x64xi32>
  return %r : vector<64x64xi32>
}

// -----

// CHECK-LABEL: func.func @i8_too_many_rows(
// CHECK-NOT: aievec.aie4.matmul
// CHECK: vector.contract
// CHECK-NOT: aievec.aie4.matmul
func.func @i8_too_many_rows(%a: vector<65x64xi8>, %b: vector<64x64xi8>, %c: vector<65x64xi32>) -> vector<65x64xi32> {
  %aw = arith.extsi %a : vector<65x64xi8> to vector<65x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<65x64xi32>, vector<64x64xi32> into vector<65x64xi32>
  return %r : vector<65x64xi32>
}
