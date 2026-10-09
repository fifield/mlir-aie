// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Entry-mode assumptions require the function, its callers and their callees
// to preserve modes, regardless of instruction order.
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:9,sat:1" -verify-diagnostics

// The entry assertion is true, but this write changes crRnd before SRS.
func.func private @llvm.aie4.set.ctrl.reg(i32, i32)
func.func @direct_write(%acc: vector<64xi32>) -> vector<64xi8> {
  %reg = arith.constant 46 : i32
  %floor = arith.constant 0 : i32
  // expected-note@+1 {{may write crRnd and crSat}}
  func.call @llvm.aie4.set.ctrl.reg(%reg, %floor) : (i32, i32) -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// The intrinsic operation must not bypass the same check.
func.func @intrinsic_write(%acc: vector<64xi32>) -> vector<64xi8> {
  %reg = arith.constant 45 : i32
  %none = arith.constant 0 : i32
  // expected-note@+1 {{may write crRnd and crSat}}
  llvm.call_intrinsic "llvm.aie4.set.ctrl.reg"(%reg, %none) : (i32, i32) -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// Calls after SRS also count as clobbers.
func.func private @external()
func.func @external_call(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  // expected-note@+1 {{may write crRnd and crSat}}
  func.call @external() : () -> ()
  return %y : vector<64xi8>
}

// -----

// Follow multiple internal calls before reaching the clobber.
func.func private @external()
func.func private @leaf() {
  // expected-note@+1 {{may write crRnd and crSat}}
  func.call @external() : () -> ()
  return
}
func.func private @middle() {
  func.call @leaf() : () -> ()
  return
}
func.func @transitive(%acc: vector<64xi32>) -> vector<64xi8> {
  func.call @middle() : () -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// A harmless SRS helper cannot rely on assumptions after its caller clobbers.
func.func private @external()
func.func private @srs_helper(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}
func.func @caller(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-note@+1 {{may write crRnd and crSat}}
  func.call @external() : () -> ()
  %y = func.call @srs_helper(%acc) : (vector<64xi32>) -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// An unknown indirect target cannot be inspected.
func.func @indirect(%acc: vector<64xi32>, %f: () -> ()) -> vector<64xi8> {
  // expected-note@+1 {{may write crRnd and crSat}}
  func.call_indirect %f() : () -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// LLVM calls conservatively count as clobbers.
llvm.func @external()
func.func @llvm_call(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-note@+1 {{may write crRnd and crSat}}
  llvm.call @external() : () -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// Inline assembly is a mode clobber even without declared side effects.
func.func @inline_asm(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-note@+1 {{may write crRnd and crSat}}
  llvm.inline_asm "movx crrnd, #0", "" : () -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// A standalone aie.debug has not yet lowered to its external call.
func.func @debug(%acc: vector<64xi32>) -> vector<64xi8> {
  %value = arith.constant 1 : i32
  // expected-note@+1 {{may write crRnd and crSat}}
  aie.debug(%value : i32)
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// -----

// func-to-llvm can run first: an LLVM function needs the same protection.
llvm.func @llvm_function(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-note@+1 {{may write crRnd and crSat}}
  llvm.inline_asm "movx crrnd, #0", "" : () -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  llvm.return %y : vector<64xi8>
}

// -----

// The option states a function-entry contract, not an unoutlined core entry.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) : memref<64xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %acc = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      // expected-error@+1 {{aie4-assume-modes requires an enclosing function; outline AIE cores before converting explicit-mode SRS}}
      %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      aie.end
    }
  }
}

// -----

// LLVM callers also reach assumption-backed SRS helpers. LLVM calls retain
// the conservative core classification, and their internal bodies are visited.
llvm.func @llvm_srs_helper(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  llvm.return %y : vector<64xi8>
}
llvm.func @llvm_srs_caller(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-note@+1 {{may write crRnd and crSat}}
  llvm.inline_asm "movx crrnd, #0", "" : () -> ()
  %y = llvm.call @llvm_srs_helper(%acc) : (vector<64xi32>) -> vector<64xi8>
  llvm.return %y : vector<64xi8>
}

// -----

// Only a dictionary is an entry guarantee. A malformed attribute must not
// bypass the preservation check when the option supplies the real guarantee.
func.func @invalid_entry_attr(%acc: vector<64xi32>) -> vector<64xi8>
    attributes {aie4.entry_modes = "invalid"} {
  // expected-note@+1 {{may write crRnd and crSat}}
  llvm.inline_asm "movx crrnd, #0", "" : () -> ()
  // expected-error@+1 {{relies on aie4-assume-modes at function entry, but its function or a caller and their callees contain an operation that may change crRnd or crSat (D19)}}
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}
