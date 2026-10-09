// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit-mode aievec.aie4.srs needs its modes guaranteed at function entry.
//
// RUN: aie-opt %s -split-input-file -verify-diagnostics -convert-aievec-to-llvm="aie-target=aie4b"
//
// With the option, it applies only where no attribute exists (@no_guarantee).
// RUN: not aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:12" 2>&1 | FileCheck %s --check-prefix=E3
// E3: 'aievec.aie4.srs' op needs crRnd = 9 (pos_inf), but aie4-assume-modes gives crRnd = 12 (conv_even)
// E3: 'aievec.aie4.srs' op needs crSat = 1 (saturate), but aie4-assume-modes does not set crSat
// E3: the entry modes of '@wrong_rnd' give crRnd = 12 (conv_even)
//
// Option errors, each in its own RUN.
// RUN: not aie-opt %s -convert-aievec-to-llvm="aie-target=aie2p aie4-assume-modes=rnd:9" 2>&1 | FileCheck %s --check-prefix=TARGET
// TARGET: aie4-assume-modes requires an AIE4 target, got aie2p
// RUN: not aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd=9" 2>&1 | FileCheck %s --check-prefix=SYNTAX
// SYNTAX: invalid aie4-assume-modes 'rnd=9': expected rnd:<mode> and/or sat:<mode>, comma-separated
// RUN: not aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:4" 2>&1 | FileCheck %s --check-prefix=RND4
// RND4: invalid aie4-assume-modes: crRnd mode 4 is not a valid encoding (0-3, 8-13)
// RUN: not aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=sat:2" 2>&1 | FileCheck %s --check-prefix=SAT2
// SAT2: invalid aie4-assume-modes: crSat mode 2 is not a valid encoding (0, 1, 3)
// RUN: not aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:9,rnd:9" 2>&1 | FileCheck %s --check-prefix=TWICE
// TWICE: invalid aie4-assume-modes: 'rnd' given twice

// No entry attribute or assumption option.
func.func @no_guarantee(%acc : vector<64xi32>) -> vector<64xi8> {
  // expected-error @+1 {{'aievec.aie4.srs' op needs crRnd = 9 (pos_inf) and crSat = 1 (saturate) at entry to '@no_guarantee', which is not reached from an AIE4 core (no aie4.entry_modes); pass aie4-assume-modes=rnd:9,sat:1 to convert-aievec-to-llvm to assert the caller's modes}}
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// The entry modes give another crRnd.
func.func @wrong_rnd(%acc : vector<64xi32>) -> vector<64xi8>
    attributes {aie4.entry_modes = {rnd = 12 : i32}} {
  // expected-error @+1 {{'aievec.aie4.srs' op needs crRnd = 9 (pos_inf), but the entry modes of '@wrong_rnd' give crRnd = 12 (conv_even)}}
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// The entry modes do not fix crSat.
func.func @missing_sat(%acc : vector<64xi32>) -> vector<64xi8>
    attributes {aie4.entry_modes = {rnd = 9 : i32}} {
  // expected-error @+1 {{'aievec.aie4.srs' op needs crSat = 1 (saturate), but the entry modes of '@missing_sat' do not fix crSat (its callers disagree or read it ambient)}}
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// An empty dictionary guarantees nothing.
func.func @empty_entry_modes(%acc : vector<32xi32>) -> vector<32xi16>
    attributes {aie4.entry_modes = {}} {
  // expected-error @+1 {{'aievec.aie4.srs' op needs crRnd = 12 (conv_even), but the entry modes of '@empty_entry_modes' do not fix crRnd}}
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<conv_even>} : vector<32xi32> -> vector<32xi16>
  return %0 : vector<32xi16>
}
