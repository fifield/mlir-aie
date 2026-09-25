// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pure shuffles can be CSE'd and hoisted out of loops. Different modes
// must remain distinct.

// RUN: aie-opt %s -split-input-file -cse | FileCheck %s --check-prefix=CSE
// RUN: aie-opt %s -split-input-file -loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

// CSE-LABEL: func.func @shuffle_4x4_pair_is_merged
// CSE: %[[S:.*]] = aievec.aie4.shuffle
// CSE-NOT: aievec.aie4.shuffle
// CSE: return %[[S]], %[[S]]
func.func @shuffle_4x4_pair_is_merged(%a : vector<16xi32>)
    -> (vector<16xi32>, vector<16xi32>) {
  %0 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<16xi32>
  %1 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<16xi32>
  return %0, %1 : vector<16xi32>, vector<16xi32>
}

// -----

// CSE-LABEL: func.func @shuffle_2x16lo_pair_is_merged
// CSE: %[[S:.*]] = aievec.aie4.shuffle
// CSE-NOT: aievec.aie4.shuffle
// CSE: return %[[S]], %[[S]]
func.func @shuffle_2x16lo_pair_is_merged(%a : vector<16xi32>, %b : vector<16xi32>)
    -> (vector<16xi32>, vector<16xi32>) {
  %0 = aievec.aie4.shuffle %a, %b [mode = <t32_2x16_lo>] : vector<16xi32>
  %1 = aievec.aie4.shuffle %a, %b [mode = <t32_2x16_lo>] : vector<16xi32>
  return %0, %1 : vector<16xi32>, vector<16xi32>
}

// -----

// CSE-LABEL: func.func @shuffle_different_mode_not_merged
// CSE: %[[S0:.*]] = aievec.aie4.shuffle %{{.*}}[mode = <t32_4x4>]
// CSE: %[[S1:.*]] = aievec.aie4.shuffle %{{.*}}[mode = <t8_8x8>]
// CSE: return %[[S0]], %[[S1]]
func.func @shuffle_different_mode_not_merged(%a : vector<16xi32>)
    -> (vector<16xi32>, vector<64xi8>) {
  %ac = vector.bitcast %a : vector<16xi32> to vector<64xi8>
  %0 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<16xi32>
  %1 = aievec.aie4.shuffle %ac [mode = <t8_8x8>] : vector<64xi8>
  return %0, %1 : vector<16xi32>, vector<64xi8>
}

// -----

// LICM-LABEL: func.func @shuffle_4x4_is_hoisted
// LICM: aievec.aie4.shuffle
// LICM: scf.for
// LICM-NOT: aievec.aie4.shuffle
func.func @shuffle_4x4_is_hoisted(%a : vector<16xi32>, %n : index,
                                  %buf : memref<1024xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<16xi32>
    %ext = vector.extract_strided_slice %0 offsets = [0], sizes = [16], strides = [1] : vector<16xi32> to vector<16xi32>
    vector.transfer_write %ext, %buf[%i] {in_bounds = [true]} : vector<16xi32>, memref<1024xi32>
  }
  return
}
