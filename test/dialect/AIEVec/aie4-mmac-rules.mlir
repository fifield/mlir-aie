// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics -allow-unregistered-dialect

// Verify matrix residency, format agreement and prohibited region contents.
// Side effects are tested in aie4-mmac-effects.mlir.

//===----------------------------------------------------------------------===//
// an MMAC is nested in a region of the same format.
//===----------------------------------------------------------------------===//

func.func @mmac_outside_region(%a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  // expected-error @+1 {{must be nested in an 'aievec.aie4.matrix_region' that provides the active matrix}}
  %r = aievec.aie4.mmac bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return %r : vector<64xf32>
}

// -----

// A loop is not a region, even when a region precedes it.
func.func @mmac_after_region(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    aievec.aie4.matrix_yield
  }
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<64xf32>) {
    // expected-error @+1 {{must be nested in an 'aievec.aie4.matrix_region'}}
    %n = aievec.aie4.mmac bf16 %a, %acc : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    scf.yield %n : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// Code under an isolated-from-above op can run outside the region.
func.func @mmac_in_nested_module(%b: vector<32x64xbf16>) {
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    builtin.module {
      func.func @inner(%a: vector<32xbf16>) -> vector<64xf32> {
        // expected-error @+1 {{must be nested in an 'aievec.aie4.matrix_region'}}
        %r = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
        return %r : vector<64xf32>
      }
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @mmac_format_mismatch(%b: vector<64x64xi8>, %a: vector<64xf8E4M3FN>) {
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> attributes {rhs_signed = true} {
    // expected-error @+1 {{format fp8 does not match the enclosing region's format int8}}
    %r = aievec.aie4.mmac fp8 %a : vector<64xf8E4M3FN> -> vector<64xf32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// int8 and int4 rows share the I512 intrinsic but not the matrix layout.
func.func @mmac_int4_in_int8_region(%b: vector<64x64xi8>, %a: vector<128xi4>) {
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> attributes {rhs_signed = true} {
    // expected-error @+1 {{format int4 does not match the enclosing region's format int8}}
    %r = aievec.aie4.mmac int4 %a {lhs_signed = true, rhs_signed = true} : vector<128xi4> -> vector<64xi32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// Format mismatch at depth, inside a row loop.
func.func @mmac_format_mismatch_in_loop(%b: vector<32x64xbf16>, %a: vector<64xf8E4M3FN>, %c: vector<64xf32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %out = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc0: vector<64xf32>):
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %acc0) -> (vector<64xf32>) {
      // expected-error @+1 {{format fp8 does not match the enclosing region's format bf16}}
      %n = aievec.aie4.mmac fp8 %a, %acc : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
      scf.yield %n : vector<64xf32>
    }
    aievec.aie4.matrix_yield %r : vector<64xf32>
  }
  return %out : vector<64xf32>
}

// -----

// The Z signedness is fixed when B is staged.
func.func @mmac_rhs_sign_mismatch(%b: vector<128x64xi4>, %a: vector<128xi4>) {
  aievec.aie4.matrix_region int4 %b : vector<128x64xi4> attributes {rhs_signed = true} {
    // expected-error @+1 {{'rhs_signed' must match the enclosing region's 'rhs_signed'}}
    %r = aievec.aie4.mmac int4 %a {lhs_signed = true, rhs_signed = false} : vector<128xi4> -> vector<64xi32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// regions do not nest.
//===----------------------------------------------------------------------===//

func.func @nested_region(%b0: vector<32x64xbf16>, %b1: vector<32x64xbf16>) {
  aievec.aie4.matrix_region bf16 %b0 : vector<32x64xbf16> {
    // expected-error @+1 {{must not be nested in another 'aievec.aie4.matrix_region'; there is one active matrix}}
    aievec.aie4.matrix_region bf16 %b1 : vector<32x64xbf16> {
      aievec.aie4.matrix_yield
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @nested_region_in_loop(%b0: vector<32x64xbf16>, %b1: vector<64x64xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  aievec.aie4.matrix_region bf16 %b0 : vector<32x64xbf16> {
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-error @+1 {{must not be nested in another 'aievec.aie4.matrix_region'}}
      aievec.aie4.matrix_region int8 %b1 : vector<64x64xi8> attributes {rhs_signed = true} {
        aievec.aie4.matrix_yield
      }
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// no calls, staging or commit in the body, at any depth.
//===----------------------------------------------------------------------===//

func.func private @helper()

func.func @call_in_region(%b: vector<32x64xbf16>) {
  // expected-error @+1 {{body must not contain calls; it may replace or clobber the active matrix}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-note @+1 {{call here}}
    func.call @helper() : () -> ()
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func private @helper()

func.func @call_in_region_loop(%b: vector<32x64xbf16>, %p: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{body must not contain calls}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    scf.for %i = %c0 to %c4 step %c1 {
      scf.if %p {
        // expected-note @+1 {{call here}}
        func.call @helper() : () -> ()
      }
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @indirect_call_in_region(%b: vector<32x64xbf16>, %f: () -> ()) {
  // expected-error @+1 {{body must not contain calls}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-note @+1 {{call here}}
    func.call_indirect %f() : () -> ()
    aievec.aie4.matrix_yield
  }
  return
}

// -----

llvm.func @llvm_helper()

func.func @llvm_call_in_region(%b: vector<32x64xbf16>) {
  // expected-error @+1 {{body must not contain calls}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-note @+1 {{call here}}
    llvm.call @llvm_helper() : () -> ()
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @staging_in_region(%b: vector<64x64xi8>, %chunk: vector<128xi8>, %i: i32) {
  // expected-error @+1 {{body must not contain matrix staging or commit operations}}
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> attributes {rhs_signed = true} {
    // expected-note @+1 {{matrix-state write here}}
    "xllvm.intr.aie4.insert.staging.int.I1024"(%chunk, %i) {mode = 3 : i32} : (vector<128xi8>, i32) -> ()
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @commit_in_region_loop(%b: vector<64x64xf8E4M3FN>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{body must not contain matrix staging or commit operations}}
  aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> {
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-note @+1 {{matrix-state write here}}
      "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// Another matrix-state operation: lowering a matmul stages its own B tile.
func.func @matmul_in_region(%b: vector<32x64xbf16>, %a: vector<4x32xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  // expected-error @+1 {{body must not contain 'aievec.aie4.matmul', which stages its own matrix}}
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<4x64xf32> {
    // expected-note @+1 {{matmul here}}
    %m = aievec.aie4.matmul %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
    aievec.aie4.matrix_yield %m : vector<4x64xf32>
  }
  return %r : vector<4x64xf32>
}

// -----

func.func @matmul_in_region_loop(%b: vector<64x64xi8>, %a: vector<1x64xi8>, %c: vector<1x64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{body must not contain 'aievec.aie4.matmul'}}
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> attributes {rhs_signed = true} {
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-note @+1 {{matmul here}}
      %m = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<1x64xi8>, vector<64x64xi8> into vector<1x64xi32>
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// Inline assembly can write the matrix directly.
func.func @inline_asm_in_region(%b: vector<32x64xbf16>) {
  // expected-error @+1 {{body must not contain inline assembly; it may replace or clobber the active matrix}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-note @+1 {{inline assembly here}}
    llvm.inline_asm has_side_effects "vmov zb, za", "" : () -> ()
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// llvm.call_intrinsic is not a CallOpInterface but can name any intrinsic.
func.func @call_intrinsic_in_region(%b: vector<32x64xbf16>) {
  // expected-error @+1 {{body must not contain calls}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-note @+1 {{call here}}
    llvm.call_intrinsic "llvm.aie4.staging.to.matrix.I32768"() : () -> ()
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// Nothing is known about an unregistered op; it may write matrix state.
func.func @unregistered_in_region(%b: vector<32x64xbf16>) {
  // expected-error @+1 {{body must not contain unregistered operations}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-note @+1 {{unregistered operation here}}
    "unknown.clobber"() : () -> ()
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// An isolated-from-above body need not execute inside the region.
func.func @module_in_region(%b: vector<32x64xbf16>) {
  // expected-error @+1 {{body must not contain isolated-from-above operations}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-note @+1 {{isolated-from-above operation here}}
    builtin.module {
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// Accepted forms
//===----------------------------------------------------------------------===//

// XLLVM MMACs only read the matrix. They are allowed so that the body can
// be lowered in place before the region is erased.
func.func @xllvm_mmac_in_region(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%c, %a) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// Staging and commit before the region, and one region per K tile in a
// loop, are the intended structure.
func.func @regions_in_k_loop(%bs: vector<4x32x64xbf16>, %a: vector<4x32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<64xf32>) {
    %b = vector.extract %bs[%k] : vector<32x64xbf16> from vector<4x32x64xbf16>
    %a_k = vector.extract %a[%k] : vector<32xbf16> from vector<4x32xbf16>
    %n = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %acc : vector<64xf32> -> vector<64xf32> {
    ^bb0(%x: vector<64xf32>):
      %y = aievec.aie4.mmac bf16 %a_k, %x : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
      aievec.aie4.matrix_yield %y : vector<64xf32>
    }
    scf.yield %n : vector<64xf32>
  }
  return %r : vector<64xf32>
}
