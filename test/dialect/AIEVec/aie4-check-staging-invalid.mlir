// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -aievec-aie4-check-staging -verify-diagnostics -allow-unregistered-dialect

// Programs the staging lint rejects, one per split.

// An unprimed mmac_active inside scf.while: only scf.for, scf.if and
// matrix_region may surround matrix-state ops.
func.func @mmac_in_while(%a: vector<32xbf16>, %c: vector<64xf32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-error @+1 {{'scf.while' contains matrix-state operations; aievec-aie4-check-staging supports only scf.for, scf.if and aievec.aie4.matrix_region around them}}
  %w = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %n : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    // expected-note @+1 {{matrix-state operation here}}
    %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    %i1 = arith.addi %i, %c1 : i32
    scf.yield %i1 : i32
  }
  return
}

// -----

func.func @commit_in_execute_region() {
  // expected-error @+1 {{'scf.execute_region' contains matrix-state operations}}
  scf.execute_region {
    // expected-note @+1 {{matrix-state operation here}}
    aievec.aie4.commit bf16
    scf.yield
  }
  return
}

// -----

func.func @mmac_in_index_switch(%a: vector<32xbf16>, %c: vector<64xf32>, %sel: index) {
  // expected-error @+1 {{'scf.index_switch' contains matrix-state operations}}
  scf.index_switch %sel
  case 0 {
    // expected-note @+1 {{matrix-state operation here}}
    %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// -----

func.func @stage_in_while(%chunk: vector<64xbf16>, %s: i32, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-error @+1 {{'scf.while' contains matrix-state operations}}
  %w = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %n : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    // expected-note @+1 {{matrix-state operation here}}
    aievec.aie4.stage bf16 %chunk, %s : vector<64xbf16>, i32
    %i1 = arith.addi %i, %c1 : i32
    scf.yield %i1 : i32
  }
  return
}

// -----

// A commit in one branch only.
func.func @one_branch_commit(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %cond: i1) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  scf.if %cond {
    aievec.aie4.commit bf16
  }
  // expected-error @+1 {{'aievec.aie4.mmac_active' may read an undefined matrix: no commit of a fully staged bf16 tile reaches it on every path}}
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

// Branches commit different keys: the matrix is Conflict.
func.func @if_different_keys(%chunk: vector<64xbf16>, %chunk8: vector<128xi8>, %a: vector<32xbf16>, %c: vector<64xf32>, %cond: i1) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.if %cond {
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
    // expected-note @+1 {{committed here}}
    aievec.aie4.commit bf16
  } else {
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage int8 %chunk8, %pslot {rhs_signed = true} : vector<128xi8>, i32
  }
    // expected-note @+1 {{committed here}}
    aievec.aie4.commit int8 {rhs_signed = true}
  }
  // expected-error @+1 {{'aievec.aie4.mmac_active' may read an undefined matrix}}
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

// Zero-trip loop: the commit inside may not run.
func.func @zero_trip_commit(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %n: index) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %k = %pc0 to %n step %pc1 {
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
    aievec.aie4.commit bf16
  }
  // expected-error @+1 {{'aievec.aie4.mmac_active' may read an undefined matrix}}
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

// Zero-trip prime: the slots may be unwritten; no clobber note.
func.func @zero_trip_prime(%chunk: vector<64xbf16>, %n: index) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %k = %pc0 to %n step %pc1 {
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  }
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// A call at the end of the K loop clobbers staging for the next iteration.
func.func private @kernel()
func.func @back_edge_clobber(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %n: index) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  scf.for %k = %pc0 to %n step %pc1 {
    // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0 after a call that may clobber it}}
    aievec.aie4.commit bf16
    %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    // expected-note @+1 {{call here}}
    func.call @kernel() : () -> ()
  }
  return
}

// -----

func.func private @kernel()
func.func @call_before_commit(%chunk: vector<64xbf16>) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  // expected-note @+1 {{call here}}
  func.call @kernel() : () -> ()
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0 after a call that may clobber it}}
  aievec.aie4.commit bf16
  return
}

// -----

func.func @inline_asm_before_commit(%chunk: vector<64xbf16>) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  // expected-note @+1 {{clobbered here}}
  llvm.inline_asm has_side_effects "nop", "" : () -> ()
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

func.func @unregistered_before_commit(%chunk: vector<64xbf16>) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  // expected-note @+1 {{clobbered here}}
  "test.unknown"() : () -> ()
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// A raw XLLVM insert writes an unknown slot.
func.func @raw_insert_after_prime(%chunk: vector<128xi8>, %i: i32) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage int8 %chunk, %pslot {rhs_signed = true} : vector<128xi8>, i32
  }
  // expected-note @+1 {{clobbered here}}
  "xllvm.intr.aie4.insert.staging.int.I1024"(%chunk, %i) {mode = 3 : i32} : (vector<128xi8>, i32) -> ()
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit int8 {rhs_signed = true}
  return
}

// -----

// A raw XLLVM commit replaces the matrix with unknown contents.
func.func @raw_commit_after_commit(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  aievec.aie4.commit bf16
  // expected-note @+1 {{clobbered here}}
  "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
  // expected-error @+1 {{'aievec.aie4.mmac_active' may read an undefined matrix}}
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

// A dynamic-slot prime does not define every slot.
func.func @dynamic_prime(%chunk: vector<64xbf16>, %dyn: i32) {
  // expected-note @+1 {{written with a dynamic index here}}
  aievec.aie4.stage bf16 %chunk, %dyn : vector<64xbf16>, i32
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// A dynamic rewrite with a different key leaves every slot Conflict.
func.func @dynamic_different_key(%chunk: vector<64xbf16>, %chunk8: vector<128xi8>, %dyn: i32) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    // expected-note @+1 {{written here}}
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  // expected-note @+1 {{written here}}
  aievec.aie4.stage int8 %chunk8, %dyn {rhs_signed = true} : vector<128xi8>, i32
  // expected-error @+1 {{'aievec.aie4.commit' expects bf16 but staging slot 0 holds different formats on different paths}}
  aievec.aie4.commit bf16
  return
}

// -----

// Mismatched induction loop: 0 to 31.
func.func @induction_upper_31(%chunk: vector<64xbf16>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c31 = arith.constant 31 : index
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : i32
  scf.for %i = %c0 to %c31 step %c1 {
    %slot = arith.index_cast %i : index to i32
    // expected-note @+1 {{written with a dynamic index here}}
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// Mismatched induction loop: step 2.
func.func @induction_step_2(%chunk: vector<64xbf16>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c31 = arith.constant 31 : index
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : i32
  scf.for %i = %c0 to %c32 step %c2 {
    %slot = arith.index_cast %i : index to i32
    // expected-note @+1 {{written with a dynamic index here}}
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// Mismatched induction loop: slot is %i + 1.
func.func @induction_plus_one(%chunk: vector<64xbf16>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c31 = arith.constant 31 : index
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : i32
  scf.for %i = %c0 to %c32 step %c1 {
    %s0 = arith.index_cast %i : index to i32
    %slot = arith.addi %s0, %one : i32
    // expected-note @+1 {{written with a dynamic index here}}
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// Mismatched induction loop: the stage is conditional.
func.func @induction_conditional(%chunk: vector<64xbf16>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c31 = arith.constant 31 : index
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : i32
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    scf.if %cond {
      // expected-note @+1 {{written with a dynamic index here}}
      aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
    }
  }
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// A call in the induction loop clobbers staging before the commit.
func.func private @kernel()
func.func @induction_extra_call(%chunk: vector<64xbf16>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c31 = arith.constant 31 : index
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : i32
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    // expected-note @+1 {{call here}}
    func.call @kernel() : () -> ()
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0 after a call that may clobber it}}
  aievec.aie4.commit bf16
  return
}

// -----

// int4 keys include the staging signedness.
func.func @int4_sign_mismatch(%chunk: vector<128xi8>) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    // expected-note @+1 {{written here}}
    aievec.aie4.stage int4 %chunk, %pslot {rhs_signed = true} : vector<128xi8>, i32
  }
  // expected-error @+1 {{'aievec.aie4.commit' expects int4 (rhs_signed = false) but staging slot 0 holds int4 (rhs_signed = true)}}
  aievec.aie4.commit int4 {rhs_signed = false}
  return
}

// -----

func.func @format_mismatch(%chunk: vector<64xbf16>) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    // expected-note @+1 {{written here}}
    aievec.aie4.stage bf16 %chunk, %pslot : vector<64xbf16>, i32
  }
  // expected-error @+1 {{'aievec.aie4.commit' expects int8 but staging slot 0 holds bf16}}
  aievec.aie4.commit int8 {rhs_signed = true}
  return
}

// -----

func.func @mmac_format_mismatch(%chunk: vector<128xi8>, %a: vector<32xbf16>, %c: vector<64xf32>) {
  %pc0 = arith.constant 0 : index
  %pc1 = arith.constant 1 : index
  %pc32 = arith.constant 32 : index
  scf.for %pi = %pc0 to %pc32 step %pc1 {
    %pslot = arith.index_cast %pi : index to i32
    aievec.aie4.stage int8 %chunk, %pslot {rhs_signed = true} : vector<128xi8>, i32
  }
  // expected-note @+1 {{committed here}}
  aievec.aie4.commit int8 {rhs_signed = true}
  // expected-error @+1 {{'aievec.aie4.mmac_active' expects a committed bf16 matrix but the committed matrix holds int8}}
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

// Unstructured control flow around flat ops.
// expected-error @+1 {{region with multiple blocks contains matrix-state operations; unstructured control flow is not supported here}}
func.func @multi_block(%a: vector<32xbf16>, %c: vector<64xf32>) {
  cf.br ^bb1
^bb1:
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

// One bad commit is reported once, not again at each later mmac_active.
func.func @cascade(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) {
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  %y = aievec.aie4.mmac_active bf16 %a, %x : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  %z = aievec.aie4.mmac_active bf16 %a, %y : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

// An i4 truncation of the induction variable writes only slots 0-15, so the
// loop is not an induction-loop prime.
func.func @induction_trunc_i4(%chunk: vector<64xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    %w = arith.index_cast %i : index to i32
    %t = arith.trunci %w : i32 to i4
    %slot = arith.extui %t : i4 to i32
    // expected-note @+1 {{written with a dynamic index here}}
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  // expected-error @+1 {{'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path}}
  aievec.aie4.commit bf16
  return
}

// -----

// The commit runs only on some iterations; an unreported fixed-point pass
// must not mark the matrix as already diagnosed.
func.func @conditional_commit_in_loop(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %n: index, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  scf.for %k = %c0 to %n step %c1 {
    scf.if %cond {
      aievec.aie4.commit bf16
    }
    // expected-error @+1 {{'aievec.aie4.mmac_active' may read an undefined matrix}}
    %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  }
  return
}

// -----

// No commit anywhere: the MMAC in the loop is reported.
func.func @mmac_only_loop(%a: vector<32xbf16>, %c: vector<64xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %k = %c0 to %n step %c1 {
    // expected-error @+1 {{'aievec.aie4.mmac_active' may read an undefined matrix}}
    %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  }
  return
}

// -----

// A K loop whose prefetch stages one slot with the wrong format: the notes
// name both formats, and later commits of the same image are not reported
// again.
func.func @prefetch_wrong_format(%c8: vector<128xi8>, %cbf: vector<64xbf16>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %s5 = arith.constant 5 : i32
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    // expected-note @+1 {{written here}}
    aievec.aie4.stage int8 %c8, %slot {rhs_signed = true} : vector<128xi8>, i32
  }
  scf.for %k = %c0 to %n step %c1 {
    // expected-error @+1 {{'aievec.aie4.commit' expects int8 but staging slot 5 holds different formats on different paths}}
    aievec.aie4.commit int8 {rhs_signed = true}
    // expected-note @+1 {{written here}}
    aievec.aie4.stage bf16 %cbf, %s5 : vector<64xbf16>, i32
  }
  aievec.aie4.commit int8 {rhs_signed = true}
  aievec.aie4.commit int8 {rhs_signed = true}
  return
}

// -----

// A later commit skips slots already reported and still reports a
// different bad write.
func.func @second_bad_write(%c8: vector<128xi8>, %cbf: vector<64xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %s0 = arith.constant 0 : i32
  %s5 = arith.constant 5 : i32
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    aievec.aie4.stage int8 %c8, %slot {rhs_signed = true} : vector<128xi8>, i32
  }
  // expected-note @+1 {{written here}}
  aievec.aie4.stage bf16 %cbf, %s0 : vector<64xbf16>, i32
  // expected-error @+1 {{'aievec.aie4.commit' expects int8 but staging slot 0 holds bf16}}
  aievec.aie4.commit int8 {rhs_signed = true}
  // expected-note @+1 {{written here}}
  aievec.aie4.stage bf16 %cbf, %s5 : vector<64xbf16>, i32
  // expected-error @+1 {{'aievec.aie4.commit' expects int8 but staging slot 5 holds bf16}}
  aievec.aie4.commit int8 {rhs_signed = true}
  return
}
