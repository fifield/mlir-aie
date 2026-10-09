// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// aievec.aie4.srs reads crRnd and crSat and writes the SRS overflow status,
// so identical ops are not merged and loop-invariant ones are not hoisted.
//
// RUN: aie-opt %s -split-input-file -cse | FileCheck %s --check-prefix=CSE
// RUN: aie-opt %s -split-input-file -loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

// CSE-LABEL: func.func @srs_pair_not_merged
// CSE: %[[S0:.*]] = aievec.aie4.srs
// CSE: %[[S1:.*]] = aievec.aie4.srs
// CSE: return %[[S0]], %[[S1]]
func.func @srs_pair_not_merged(%acc : vector<64xi32>)
    -> (vector<64xi8>, vector<64xi8>) {
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  %1 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %0, %1 : vector<64xi8>, vector<64xi8>
}

// -----

// LICM-LABEL: func.func @srs_not_hoisted
// LICM: scf.for
// LICM: aievec.aie4.srs
func.func @srs_not_hoisted(%acc : vector<32xi32>, %n : index,
                           %buf : memref<1024xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.aie4.srs %acc {shift = 4 : i32, isSigned = true} : vector<32xi32> -> vector<32xi16>
    vector.transfer_write %0, %buf[%i] {in_bounds = [true]} : vector<32xi16>, memref<1024xi16>
  }
  return
}
