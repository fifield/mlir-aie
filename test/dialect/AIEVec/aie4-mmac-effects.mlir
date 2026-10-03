// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// matrix_region and row MMACs retain unknown effects, preventing generic
// CSE, DCE and LICM from merging, erasing or moving them.
// aievec.aie4.matmul is Pure and serves as the positive control.

// RUN: aie-opt %s -cse | FileCheck %s --check-prefix=CSE
// RUN: aie-opt %s -canonicalize | FileCheck %s --check-prefix=CANON
// RUN: aie-opt %s -loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

// Identical MMACs read the same matrix but are not merged.
// CSE-LABEL: func.func @cse_mmac
// CSE-COUNT-2: aievec.aie4.mmac
// CSE-NOT: aievec.aie4.mmac
// CSE: return
func.func @cse_mmac(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) -> (vector<64xf32>, vector<64xf32>) {
  %r:2 = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32>, vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    %y = aievec.aie4.mmac bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x, %y : vector<64xf32>, vector<64xf32>
  }
  return %r#0, %r#1 : vector<64xf32>, vector<64xf32>
}

// Identical regions each restage their matrix.
// CSE-LABEL: func.func @cse_region
// CSE-COUNT-2: aievec.aie4.matrix_region
// CSE-NOT: aievec.aie4.matrix_region
// CSE: return
func.func @cse_region(%b: vector<32x64xbf16>, %a: vector<32xbf16>) -> (vector<64xf32>, vector<64xf32>) {
  %r0 = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  %r1 = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r0, %r1 : vector<64xf32>, vector<64xf32>
}

// CSE-LABEL: func.func @cse_matmul
// CSE-COUNT-1: aievec.aie4.matmul
// CSE-NOT: aievec.aie4.matmul
// CSE: return
func.func @cse_matmul(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> (vector<4x64xf32>, vector<4x64xf32>) {
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
  %1 = aievec.aie4.matmul %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
  return %0, %1 : vector<4x64xf32>, vector<4x64xf32>
}

// Unused results do not make the ops dead.
// CANON-LABEL: func.func @dead_mmac_and_region
// CANON-NOT: aievec.aie4.matmul
// CANON: aievec.aie4.matrix_region
// CANON: aievec.aie4.mmac
// CANON: return
func.func @dead_mmac_and_region(%a4: vector<4x32xbf16>, %b: vector<32x64xbf16>, %a: vector<32xbf16>, %c4: vector<4x64xf32>) {
  %unused = aievec.aie4.matmul %a4, %b, %c4 : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return
}

// The region's own staging and commit are effects even when its body is
// empty, so the region is not derived from its body's effects.
// CANON-LABEL: func.func @empty_region
// CANON-NEXT: aievec.aie4.matrix_region
// CANON-NEXT: aievec.aie4.matrix_yield
// CANON-NEXT: }
// CANON-NEXT: return
func.func @empty_region(%b: vector<32x64xbf16>) {
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    aievec.aie4.matrix_yield
  }
  return
}

// Even with loop-invariant operands, a region retains its staging effects.
// LICM-LABEL: func.func @licm_region
// LICM: scf.for
// LICM: aievec.aie4.matrix_region
// LICM: aievec.aie4.mmac
// LICM: scf.yield
func.func @licm_region(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<64xf32>) {
    %n = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
      %y = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
      aievec.aie4.matrix_yield %y : vector<64xf32>
    }
    %s = arith.addf %acc, %n : vector<64xf32>
    scf.yield %s : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// An invariant MMAC stays in the row loop inside its region.
// LICM-LABEL: func.func @licm_mmac
// LICM: aievec.aie4.matrix_region
// LICM: scf.for
// LICM: aievec.aie4.mmac
// LICM: scf.yield
func.func @licm_mmac(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %out = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%c_in: vector<64xf32>):
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c_in) -> (vector<64xf32>) {
      %y = aievec.aie4.mmac bf16 %a, %c_in : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
      %s = arith.addf %acc, %y : vector<64xf32>
      scf.yield %s : vector<64xf32>
    }
    aievec.aie4.matrix_yield %r : vector<64xf32>
  }
  return %out : vector<64xf32>
}

// The Pure semantic op is hoisted.
// LICM-LABEL: func.func @licm_matmul
// LICM: aievec.aie4.matmul
// LICM: scf.for
// LICM-NOT: aievec.aie4.matmul
// LICM: return
func.func @licm_matmul(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<4x64xf32>) {
    %m = aievec.aie4.matmul %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
    %s = arith.addf %acc, %m : vector<4x64xf32>
    scf.yield %s : vector<4x64xf32>
  }
  return %r : vector<4x64xf32>
}
