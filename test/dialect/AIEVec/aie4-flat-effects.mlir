// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The flat matrix-state ops have unknown effects, like matrix_region and
// row MMACs: generic CSE, DCE and LICM must not merge, erase or move them.

// RUN: aie-opt %s -cse | FileCheck %s --check-prefix=CSE
// RUN: aie-opt %s -canonicalize | FileCheck %s --check-prefix=CANON
// RUN: aie-opt %s -loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

// Identical MMACs read the same matrix but are not merged.
// CSE-LABEL: func.func @cse_mmac_active
// CSE-COUNT-2: aievec.aie4.mmac_active
// CSE-NOT: aievec.aie4.mmac_active
// CSE: return
func.func @cse_mmac_active(%a: vector<64xi8>, %c: vector<64xi32>) -> (vector<64xi32>, vector<64xi32>) {
  %x = aievec.aie4.mmac_active int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  %y = aievec.aie4.mmac_active int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  return %x, %y : vector<64xi32>, vector<64xi32>
}

// Repeated commits remain observable even with identical operands.
// CSE-LABEL: func.func @cse_commit
// CSE-COUNT-2: aievec.aie4.commit bf16
// CSE-NOT: aievec.aie4.commit
// CSE: return
func.func @cse_commit() {
  aievec.aie4.commit bf16
  aievec.aie4.commit bf16
  return
}

// Repeated identical stages and stream pops are kept.
// CSE-LABEL: func.func @cse_stage
// CSE-COUNT-2: aievec.aie4.stage bf16
// CSE-COUNT-2: aievec.aie4.stage_stream bf16
// CSE: return
func.func @cse_stage(%chunk: vector<64xbf16>, %p: i32, %i: i32) {
  aievec.aie4.stage bf16 %chunk, %i : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %i : vector<64xbf16>, i32
  aievec.aie4.stage_stream bf16 %p, %i : i32, i32
  aievec.aie4.stage_stream bf16 %p, %i : i32, i32
  return
}

// Unused results do not make the ops dead.
// CANON-LABEL: func.func @dead_flat
// CANON: aievec.aie4.stage int8
// CANON: aievec.aie4.stage_stream int8
// CANON: aievec.aie4.commit int8
// CANON: aievec.aie4.mmac_active int8
// CANON: aievec.aie4.mmac_active int8 {{.*}} zero_if
// CANON: return
func.func @dead_flat(%chunk: vector<128xi8>, %p: i32, %i: i32, %a: vector<64xi8>, %c: vector<64xi32>) {
  %f = arith.constant false
  aievec.aie4.stage int8 %chunk, %i {rhs_signed = true} : vector<128xi8>, i32
  aievec.aie4.stage_stream int8 %p, %i {rhs_signed = true} : i32, i32
  aievec.aie4.commit int8 {rhs_signed = true}
  %x = aievec.aie4.mmac_active int8 %a {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
  %y = aievec.aie4.mmac_active int8 %a, %c zero_if %f {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  return
}

// Loop-invariant stage, commit and mmac_active stay in the loop.
// LICM-LABEL: func.func @licm_flat
// LICM: scf.for
// LICM-NEXT: aievec.aie4.stage bf16
// LICM-NEXT: aievec.aie4.stage_stream bf16
// LICM-NEXT: aievec.aie4.commit bf16
// LICM-NEXT: aievec.aie4.mmac_active bf16
// LICM: scf.yield
func.func @licm_flat(%chunk: vector<64xbf16>, %p: i32, %i: i32, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<64xf32>) {
    aievec.aie4.stage bf16 %chunk, %i : vector<64xbf16>, i32
    aievec.aie4.stage_stream bf16 %p, %i : i32, i32
    aievec.aie4.commit bf16
    %y = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    %s = arith.addf %acc, %y : vector<64xf32>
    scf.yield %s : vector<64xf32>
  }
  return %r : vector<64xf32>
}
