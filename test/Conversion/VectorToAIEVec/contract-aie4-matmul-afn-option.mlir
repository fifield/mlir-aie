// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The pass-level opt-in writes `afn` into the IR; it is off by default.
// RUN: aie-opt %s -split-input-file -aievec-grant-contract-afn | FileCheck %s --check-prefix=STAMP
// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie4b contract-afn=true" | FileCheck %s --check-prefix=GRANT
// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s --check-prefix=DEFAULT

// STAMP-LABEL: func.func @bf16_strict(
// STAMP: vector.contract {fastmath = #arith.fastmath<afn>,
// GRANT-LABEL: func.func @bf16_strict(
// GRANT: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
// GRANT-NOT: vector.contract
// DEFAULT-LABEL: func.func @bf16_strict(
// DEFAULT-NOT: aievec.aie4.matmul
// DEFAULT: vector.contract
// DEFAULT-NOT: aievec.aie4.matmul
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

// Existing flags are kept.
// STAMP-LABEL: func.func @fp8_contract(
// STAMP: vector.contract {fastmath = #arith.fastmath<contract,afn>,
// GRANT-LABEL: func.func @fp8_contract(
// GRANT: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} : vector<2x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<2x64xf32>
// DEFAULT-LABEL: func.func @fp8_contract(
// DEFAULT-NOT: aievec.aie4.matmul
// DEFAULT: vector.contract
func.func @fp8_contract(%a: vector<2x64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %aw = arith.extf %a : vector<2x64xf8E4M3FN> to vector<2x64xf32>
  %bw = arith.extf %b : vector<64x64xf8E4M3FN> to vector<64x64xf32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>, fastmath = #arith.fastmath<contract>}
       %aw, %bw, %c : vector<2x64xf32>, vector<64x64xf32> into vector<2x64xf32>
  return %r : vector<2x64xf32>
}

// -----

// The permission is written before the transpose is made explicit, and
// survives it.
// GRANT-LABEL: func.func @bf16_strict_transposed_b(
// GRANT: vector.transpose
// GRANT: aievec.aie4.matmul
// GRANT-NOT: vector.contract
// DEFAULT-LABEL: func.func @bf16_strict_transposed_b(
// DEFAULT-NOT: aievec.aie4.matmul
// DEFAULT: vector.contract
func.func @bf16_strict_transposed_b(%a: vector<4x32xbf16>, %bt: vector<64x32xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
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

// Integer contractions need no permission and are not stamped.
// STAMP-LABEL: func.func @i8_untouched(
// STAMP-NOT: fastmath
// STAMP: return
func.func @i8_untouched(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}
