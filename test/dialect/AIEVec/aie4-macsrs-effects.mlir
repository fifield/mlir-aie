//
// aie4-macsrs-effects.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Check CSE and LICM behavior for AIE4 control-register effects.
// SRS reads mode registers and writes overflow status; MAC only reads crVecSat.
//
// RUN: aie-opt %s -split-input-file -cse | FileCheck %s --check-prefix=CSE
// RUN: aie-opt %s -split-input-file -loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

//===----------------------------------------------------------------------===//
// CSE
//===----------------------------------------------------------------------===//

// Unknown side effects prevent CSE of identical SRS operations.
//
// CSE-LABEL: func.func @srs_pair_not_merged
// CSE: %[[S0:.*]] = aievec.aie4.srs_i32_i16
// CSE: %[[S1:.*]] = aievec.aie4.srs_i32_i16
// CSE: return %[[S0]], %[[S1]]
func.func @srs_pair_not_merged(%acc : vector<32xi32>)
    -> (vector<32xi16>, vector<32xi16>) {
  %0 = aievec.aie4.srs_i32_i16 %acc {shift = 4 : i32} : vector<32xi32>, vector<32xi16>
  %1 = aievec.aie4.srs_i32_i16 %acc {shift = 4 : i32} : vector<32xi32>, vector<32xi16>
  return %0, %1 : vector<32xi16>, vector<32xi16>
}

// -----

//===----------------------------------------------------------------------===//
// UPS mirrors SRS: same mechanism (purity by omission), same CSE/LICM proof
//===----------------------------------------------------------------------===//

// UPS reads `crUPSMode` and the `upsSign*` control state (see
// `AIE4GenInstrInfo.td`, `VUPS_2x_*`: `Uses = [crUPSMode, upsSign0]` /
// `Uses = [crUPSMode, upsSign1]`), so two UPS ops with identical operands are
// not interchangeable for exactly the same reason two SRS ops are not: the
// mode may differ between them. UPS carries no `MemoryEffects` at all
// (purity by omission, same as SRS), so this is the same proof, re-run for
// the new op rather than assumed to transfer from SRS's result.
//
// CSE-LABEL: func.func @ups_pair_not_merged
// CSE: %[[U0:.*]] = aievec.aie4.ups_i16_i32
// CSE: %[[U1:.*]] = aievec.aie4.ups_i16_i32
// CSE: return %[[U0]], %[[U1]]
func.func @ups_pair_not_merged(%src : vector<32xi16>)
    -> (vector<32xi32>, vector<32xi32>) {
  %0 = aievec.aie4.ups_i16_i32 %src {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
  %1 = aievec.aie4.ups_i16_i32 %src {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
  return %0, %1 : vector<32xi32>, vector<32xi32>
}

// -----

// A UPS whose operand is loop-invariant. Nothing but the omission stops this
// being hoisted, which would evaluate it once under whatever sign/mode
// happened to be live at the loop preheader.
//
// LICM-LABEL: func.func @ups_not_hoisted
// LICM: scf.for
// LICM: aievec.aie4.ups_i16_i32
func.func @ups_not_hoisted(%src : vector<32xi16>, %n : index,
                           %buf : memref<1024xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.aie4.ups_i16_i32 %src {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
    vector.transfer_write %0, %buf[%i] {in_bounds = [true]} : vector<32xi32>, memref<1024xi32>
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// AIE4 bf16<->f32 accumulator conversion (the bf16 add/sub unblock)
//===----------------------------------------------------------------------===//
//
// Exact widening is pure and can be merged or hoisted. Narrowing reads
// ambient control modes and has unknown side effects, preventing both.

// CSE-LABEL: func.func @conv_widen_pair_is_merged
// CSE: %[[C:.*]] = aievec.aie4.conv_bf16_f32
// CSE-NOT: aievec.aie4.conv_bf16_f32
// CSE: return %[[C]], %[[C]]
func.func @conv_widen_pair_is_merged(%src : vector<32xbf16>)
    -> (vector<32xf32>, vector<32xf32>) {
  %0 = aievec.aie4.conv_bf16_f32 %src : vector<32xbf16>, vector<32xf32>
  %1 = aievec.aie4.conv_bf16_f32 %src : vector<32xbf16>, vector<32xf32>
  return %0, %1 : vector<32xf32>, vector<32xf32>
}

// -----

// LICM-LABEL: func.func @conv_widen_is_hoisted
// LICM: aievec.aie4.conv_bf16_f32
// LICM: scf.for
// LICM-NOT: aievec.aie4.conv_bf16_f32
func.func @conv_widen_is_hoisted(%src : vector<32xbf16>, %n : index,
                                 %buf : memref<1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.aie4.conv_bf16_f32 %src : vector<32xbf16>, vector<32xf32>
    vector.transfer_write %0, %buf[%i] {in_bounds = [true]} : vector<32xf32>, memref<1024xf32>
  }
  return
}

// -----

// CSE-LABEL: func.func @conv_narrow_pair_not_merged
// CSE: %[[N0:.*]] = aievec.aie4.conv_f32_bf16
// CSE: %[[N1:.*]] = aievec.aie4.conv_f32_bf16
// CSE: return %[[N0]], %[[N1]]
func.func @conv_narrow_pair_not_merged(%acc : vector<32xf32>)
    -> (vector<32xbf16>, vector<32xbf16>) {
  %0 = aievec.aie4.conv_f32_bf16 %acc : vector<32xf32>, vector<32xbf16>
  %1 = aievec.aie4.conv_f32_bf16 %acc : vector<32xf32>, vector<32xbf16>
  return %0, %1 : vector<32xbf16>, vector<32xbf16>
}

// -----

// LICM-LABEL: func.func @conv_narrow_not_hoisted
// LICM: scf.for
// LICM: aievec.aie4.conv_f32_bf16
func.func @conv_narrow_not_hoisted(%acc : vector<32xf32>, %n : index,
                                   %buf : memref<1024xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.aie4.conv_f32_bf16 %acc : vector<32xf32>, vector<32xbf16>
    vector.transfer_write %0, %buf[%i] {in_bounds = [true]} : vector<32xbf16>, memref<1024xbf16>
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// LICM
//===----------------------------------------------------------------------===//

// SRS effects prevent LICM even when its operands are loop-invariant.
//
// LICM-LABEL: func.func @srs_not_hoisted
// LICM: scf.for
// LICM: aievec.aie4.srs_i32_i16
func.func @srs_not_hoisted(%acc : vector<32xi32>, %n : index,
                           %buf : memref<1024xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.aie4.srs_i32_i16 %acc {shift = 4 : i32} : vector<32xi32>, vector<32xi16>
    vector.transfer_write %0, %buf[%i] {in_bounds = [true]} : vector<32xi16>, memref<1024xi16>
  }
  return
}

// -----

// The crVecSat read effect prevents this LICM pass from hoisting MAC.
//
// LICM-LABEL: func.func @mac_not_hoisted
// LICM: scf.for
// LICM: aievec.aie4.mac_ewise_i16_i32
func.func @mac_not_hoisted(%a : vector<32xi16>, %b : vector<32xi16>,
                           %acc : vector<32xi32>, %n : index,
                           %buf : memref<1024xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.aie4.mac_ewise_i16_i32 %a, %b, %acc : vector<32xi16>, vector<32xi32>
    vector.transfer_write %0, %buf[%i] {in_bounds = [true]} : vector<32xi32>, memref<1024xi32>
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// MAC CSE
//===----------------------------------------------------------------------===//

// Identical MACs can be merged because there is no intervening write to crVecSat.
//
// CSE-LABEL: func.func @mac_pair_is_merged
// CSE: %[[M:.*]] = aievec.aie4.mac_ewise_i16_i32
// CSE-NOT: aievec.aie4.mac_ewise_i16_i32
// CSE: return %[[M]], %[[M]]
func.func @mac_pair_is_merged(%a : vector<32xi16>, %b : vector<32xi16>,
                              %acc : vector<32xi32>)
    -> (vector<32xi32>, vector<32xi32>) {
  %0 = aievec.aie4.mac_ewise_i16_i32 %a, %b, %acc : vector<32xi16>, vector<32xi32>
  %1 = aievec.aie4.mac_ewise_i16_i32 %a, %b, %acc : vector<32xi16>, vector<32xi32>
  return %0, %1 : vector<32xi32>, vector<32xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Pure-operation controls
//===----------------------------------------------------------------------===//

// Verify that CSE merges and LICM hoists the pure aievec.add_elem operation.

// CSE-LABEL: func.func @pure_pair_is_merged
// CSE: %[[P:.*]] = aievec.add_elem
// CSE-NOT: aievec.add_elem
// CSE: return %[[P]], %[[P]]
func.func @pure_pair_is_merged(%a : vector<32xi32>, %b : vector<32xi32>)
    -> (vector<32xi32>, vector<32xi32>) {
  %0 = aievec.add_elem %a, %b : vector<32xi32>
  %1 = aievec.add_elem %a, %b : vector<32xi32>
  return %0, %1 : vector<32xi32>, vector<32xi32>
}

// -----

// LICM-LABEL: func.func @pure_is_hoisted
// LICM: aievec.add_elem
// LICM: scf.for
// LICM-NOT: aievec.add_elem
func.func @pure_is_hoisted(%a : vector<32xi32>, %b : vector<32xi32>, %n : index,
                           %buf : memref<1024xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.add_elem %a, %b : vector<32xi32>
    vector.transfer_write %0, %buf[%i] {in_bounds = [true]} : vector<32xi32>, memref<1024xi32>
  }
  return
}
