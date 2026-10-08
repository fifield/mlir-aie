// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -aievec-aie4-check-staging -allow-unregistered-dialect | FileCheck %s

// Programs the staging lint accepts. The pass changes nothing.

// A matrix_region defines the matrix even when its row MMAC is in scf.while.
// Unrelated control flow preserves that state for the following mmac_active.
// CHECK-LABEL: func.func @compat_355
// CHECK: aievec.aie4.matrix_region
// CHECK: scf.while
// CHECK: scf.index_switch
// CHECK: scf.execute_region
// CHECK: aievec.aie4.mmac_active
func.func @compat_355(%chunk: vector<64xbf16>, %b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %n: i32, %sel: index) -> vector<64xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %s0 = arith.constant 0 : i32
  %s1 = arith.constant 1 : i32
  %s2 = arith.constant 2 : i32
  %s3 = arith.constant 3 : i32
  %s4 = arith.constant 4 : i32
  %s5 = arith.constant 5 : i32
  %s6 = arith.constant 6 : i32
  %s7 = arith.constant 7 : i32
  %s8 = arith.constant 8 : i32
  %s9 = arith.constant 9 : i32
  %s10 = arith.constant 10 : i32
  %s11 = arith.constant 11 : i32
  %s12 = arith.constant 12 : i32
  %s13 = arith.constant 13 : i32
  %s14 = arith.constant 14 : i32
  %s15 = arith.constant 15 : i32
  %s16 = arith.constant 16 : i32
  %s17 = arith.constant 17 : i32
  %s18 = arith.constant 18 : i32
  %s19 = arith.constant 19 : i32
  %s20 = arith.constant 20 : i32
  %s21 = arith.constant 21 : i32
  %s22 = arith.constant 22 : i32
  %s23 = arith.constant 23 : i32
  %s24 = arith.constant 24 : i32
  %s25 = arith.constant 25 : i32
  %s26 = arith.constant 26 : i32
  %s27 = arith.constant 27 : i32
  %s28 = arith.constant 28 : i32
  %s29 = arith.constant 29 : i32
  %s30 = arith.constant 30 : i32
  %s31 = arith.constant 31 : i32
  aievec.aie4.stage bf16 %chunk, %s0 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s1 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s2 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s3 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s4 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s5 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s6 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s7 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s8 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s9 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s10 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s11 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s12 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s13 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s14 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s15 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s16 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s17 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s18 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s19 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s20 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s21 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s22 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s23 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s24 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s25 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s26 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s27 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s28 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s29 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s30 : vector<64xbf16>, i32
  aievec.aie4.stage bf16 %chunk, %s31 : vector<64xbf16>, i32
  aievec.aie4.commit bf16
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %x : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc0: vector<64xf32>):
    %w:2 = scf.while (%i = %c0, %acc = %acc0) : (i32, vector<64xf32>) -> (i32, vector<64xf32>) {
      %cond = arith.cmpi slt, %i, %n : i32
      scf.condition(%cond) %i, %acc : i32, vector<64xf32>
    } do {
    ^bb0(%i: i32, %acc: vector<64xf32>):
      %y = aievec.aie4.mmac bf16 %a, %acc : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
      %i1 = arith.addi %i, %c1 : i32
      scf.yield %i1, %y : i32, vector<64xf32>
    }
    aievec.aie4.matrix_yield %w#1 : vector<64xf32>
  }
  %sw = scf.index_switch %sel -> vector<64xf32>
  case 0 {
    %z = arith.addf %r, %r : vector<64xf32>
    scf.yield %z : vector<64xf32>
  }
  default {
    scf.yield %r : vector<64xf32>
  }
  %ex = scf.execute_region -> vector<64xf32> {
    %z = arith.mulf %sw, %sw : vector<64xf32>
    scf.yield %z : vector<64xf32>
  }
  %out = aievec.aie4.mmac_active bf16 %a, %ex : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return %out : vector<64xf32>
}

// -----

// A subtree without matrix-state ops preserves the state.
// CHECK-LABEL: func.func @state_preserving_while
func.func @state_preserving_while(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %n: i32) -> vector<64xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    aievec.aie4.matrix_yield
  }
  %w = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %n : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %i1 = arith.addi %i, %c1 : i32
    scf.yield %i1 : i32
  }
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return %x : vector<64xf32>
}

// -----

// Streaming GEMM: prime, commit/MMAC/prefetch loop, then a final tile.
// Staging must remain defined across the loop back edge.
// CHECK-LABEL: func.func @streaming_k_loop
func.func @streaming_k_loop(%a: vector<4x64xi8>, %c: vector<64xi32>, %kt: index) -> vector<64xi32> {
  %p = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %s0 = arith.constant 0 : i32
  %s1 = arith.constant 1 : i32
  %s2 = arith.constant 2 : i32
  %s3 = arith.constant 3 : i32
  %s4 = arith.constant 4 : i32
  %s5 = arith.constant 5 : i32
  %s6 = arith.constant 6 : i32
  %s7 = arith.constant 7 : i32
  %s8 = arith.constant 8 : i32
  %s9 = arith.constant 9 : i32
  %s10 = arith.constant 10 : i32
  %s11 = arith.constant 11 : i32
  %s12 = arith.constant 12 : i32
  %s13 = arith.constant 13 : i32
  %s14 = arith.constant 14 : i32
  %s15 = arith.constant 15 : i32
  %s16 = arith.constant 16 : i32
  %s17 = arith.constant 17 : i32
  %s18 = arith.constant 18 : i32
  %s19 = arith.constant 19 : i32
  %s20 = arith.constant 20 : i32
  %s21 = arith.constant 21 : i32
  %s22 = arith.constant 22 : i32
  %s23 = arith.constant 23 : i32
  %s24 = arith.constant 24 : i32
  %s25 = arith.constant 25 : i32
  %s26 = arith.constant 26 : i32
  %s27 = arith.constant 27 : i32
  %s28 = arith.constant 28 : i32
  %s29 = arith.constant 29 : i32
  %s30 = arith.constant 30 : i32
  %s31 = arith.constant 31 : i32
  aievec.aie4.stage_stream int8 %p, %s0 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s1 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s2 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s3 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s4 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s5 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s6 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s7 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s8 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s9 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s10 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s11 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s12 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s13 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s14 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s15 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s16 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s17 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s18 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s19 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s20 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s21 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s22 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s23 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s24 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s25 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s26 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s27 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s28 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s29 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s30 {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int8 %p, %s31 {rhs_signed = true} : i32, i32
  %kt1 = arith.subi %kt, %c1 : index
  %acc = scf.for %k = %c0 to %kt1 step %c1 iter_args(%acc_k = %c) -> (vector<64xi32>) {
    aievec.aie4.commit int8 {rhs_signed = true}
    %first = arith.cmpi eq, %k, %c0 : index
    %rows = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc_i = %acc_k) -> (vector<64xi32>) {
      %a_i = vector.extract %a[%i] : vector<64xi8> from vector<4x64xi8>
      %y = aievec.aie4.mmac_active int8 %a_i, %acc_i zero_if %first {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
      scf.yield %y : vector<64xi32>
    }
    aievec.aie4.stage_stream int8 %p, %s0 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s1 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s2 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s3 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s4 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s5 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s6 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s7 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s8 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s9 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s10 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s11 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s12 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s13 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s14 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s15 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s16 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s17 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s18 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s19 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s20 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s21 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s22 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s23 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s24 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s25 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s26 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s27 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s28 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s29 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s30 {rhs_signed = true} : i32, i32
    aievec.aie4.stage_stream int8 %p, %s31 {rhs_signed = true} : i32, i32
    scf.yield %rows : vector<64xi32>
  }
  aievec.aie4.commit int8 {rhs_signed = true}
  %a0 = vector.extract %a[0] : vector<64xi8> from vector<4x64xi8>
  %last = aievec.aie4.mmac_active int8 %a0, %acc {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  return %last : vector<64xi32>
}

// -----

// Induction-loop primes, index and i32 forms.
// CHECK-LABEL: func.func @induction_index
func.func @induction_index(%chunk: vector<128xi8>, %a: vector<128xi4>, %c: vector<64xi32>) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    aievec.aie4.stage int4 %chunk, %slot {rhs_signed = true} : vector<128xi8>, i32
  }
  aievec.aie4.commit int4 {rhs_signed = true}
  %x = aievec.aie4.mmac_active int4 %a, %c {lhs_signed = true, rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
  return %x : vector<64xi32>
}

// CHECK-LABEL: func.func @induction_i32
func.func @induction_i32(%p: i32, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %i = %c0 to %c32 step %c1 : i32 {
    aievec.aie4.stage_stream bf16 %p, %i : i32, i32
  }
  aievec.aie4.commit bf16
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return %x : vector<64xf32>
}

// -----

// int8 keys ignore rhs_signed: the MMAC sign comes from its own conf.
// CHECK-LABEL: func.func @int8_sign_agnostic
func.func @int8_sign_agnostic(%chunk: vector<128xi8>, %a: vector<64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    aievec.aie4.stage int8 %chunk, %slot {rhs_signed = false} : vector<128xi8>, i32
  }
  aievec.aie4.commit int8 {rhs_signed = true}
  %x = aievec.aie4.mmac_active int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  return %x : vector<64xi32>
}

// -----

// Locks, events and stream reads between the prime and the commit are not
// clobbers.
// CHECK-LABEL: aie.device(npu3)
aie.device(npu3) {
  %t = aie.tile(0, 2)
  %l = aie.lock(%t, 0)
  %core = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %p = arith.constant 0 : i32
    %z = arith.constant dense<0> : vector<64xi8>
    %acc = arith.constant dense<0> : vector<64xi32>
    scf.for %i = %c0 to %c32 step %c1 {
      %slot = arith.index_cast %i : index to i32
      aievec.aie4.stage_stream int8 %p, %slot {rhs_signed = true} : i32, i32
    }
    %one = arith.constant 1 : i32
    aie.use_lock(%l, AcquireGreaterEqual, %one)
    aie.event(0)
    %w = aie.get_stream(%p : i32) : vector<32xi32>
    aievec.aie4.commit int8 {rhs_signed = true}
    %x = aievec.aie4.mmac_active int8 %z, %acc {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aie.use_lock(%l, Release, %one)
    aie.end
  }
}

// -----

// Both branches commit the same key, so the join keeps it.
// CHECK-LABEL: func.func @if_same_key
func.func @if_same_key(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %cond: i1) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  scf.if %cond {
    aievec.aie4.commit bf16
  } else {
    aievec.aie4.commit bf16
  }
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return %x : vector<64xf32>
}

// -----

// A dynamic-slot rewrite with the same key after a full prime: join(K, K).
// CHECK-LABEL: func.func @dynamic_same_key
func.func @dynamic_same_key(%chunk: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %dyn: i32) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %i : index to i32
    aievec.aie4.stage bf16 %chunk, %slot : vector<64xbf16>, i32
  }
  aievec.aie4.stage bf16 %chunk, %dyn : vector<64xbf16>, i32
  aievec.aie4.commit bf16
  %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return %x : vector<64xf32>
}

// -----

// Scopes without flat ops are not analysed: a raw XLLVM MMAC alone, and a
// matrix_region inside scf.while.
// CHECK-LABEL: func.func @raw_mmac_only
func.func @raw_mmac_only(%a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %x = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%c, %a) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
  return %x : vector<64xf32>
}

// CHECK-LABEL: func.func @region_in_while
func.func @region_in_while(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>, %n: i32) -> vector<64xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %w:2 = scf.while (%i = %c0, %acc = %c) : (i32, vector<64xf32>) -> (i32, vector<64xf32>) {
    %cond = arith.cmpi slt, %i, %n : i32
    scf.condition(%cond) %i, %acc : i32, vector<64xf32>
  } do {
  ^bb0(%i: i32, %acc: vector<64xf32>):
    %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %acc : vector<64xf32> -> vector<64xf32> {
    ^bb0(%x: vector<64xf32>):
      %y = aievec.aie4.mmac bf16 %a, %x : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
      aievec.aie4.matrix_yield %y : vector<64xf32>
    }
    %i1 = arith.addi %i, %c1 : i32
    scf.yield %i1, %r : i32, vector<64xf32>
  }
  return %w#1 : vector<64xf32>
}
