//===- target-parsing.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins the behavior of the centralized `aie-target` parsing in
// xilinx::aievec::parseAIEVecTarget (Utils/TargetInfo.h): accepted aliases
// must keep producing identical output, and the deliberate AIE2P asymmetry
// (AIE2P-specific lowering in convert-vector-to-aievec, but collapsed to the
// AIE2 patterns in optimize-aievec) must not regress.
//
//===----------------------------------------------------------------------===//

// --- absent aie-target and explicit "aie" agree (both hit the "aie"     ---
// --- literal via cl::init("aie"); this does NOT exercise the            ---
// --- `spelling.empty()` branch of parseAIEVecTarget -- see below).      ---

// RUN: aie-opt %s --canonicalize-vector-for-aievec -o %t.default.mlir
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie" -o %t.explicit-aie.mlir
// RUN: diff %t.default.mlir %t.explicit-aie.mlir

func.func @trivial_vector_op(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  %r = arith.addi %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// --- explicit "aie-target=" (an *empty* value) reaches `spelling.empty()`--
// --- directly and must fall back to AIE1, agreeing with explicit "aie"  ---
// --- and genuinely diverging from "aie2": AIE1's transfer-read alignment--
// --- threshold in `SplitUnalignedTransferReadPattern` is 128 bits (4    ---
// --- i32s), so a read at a 4-element-aligned offset is already aligned  ---
// --- and left untouched, while AIE2's 256-bit (8 i32) threshold still   ---
// --- considers it unaligned and splits it via an extra 16-wide read     ---
// --- plus `vector.extract_strided_slice`. If the empty-fallback ever    ---
// --- silently mapped to AIE2 instead of AIE1, this would fail.          ---

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=" -o %t.empty.mlir
// RUN: diff %t.empty.mlir %t.explicit-aie.mlir
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie2" -o %t.empty-vs-aie2.mlir
// RUN: not diff %t.empty.mlir %t.empty-vs-aie2.mlir

func.func @maybe_unaligned_read(%m: memref<64xi32>) -> vector<8xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %0 = vector.transfer_read %m[%c4], %c0_i32 : memref<64xi32>, vector<8xi32>
  return %0 : vector<8xi32>
}

// --- "aieml" and "aie2" are the same alias: identical output. -------------

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" -o %t.aieml.mlir
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" -o %t.aie2.mlir
// RUN: diff %t.aieml.mlir %t.aie2.mlir

func.func @conv2d(%arg0: memref<18x288xi16>, %arg1: memref<9xi16>, %arg2: memref<16x256xi16>) {
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 256 step 16 {
      %0 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi16>, vector<32xi16>
      %sbh = aievec.ext %0 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
      %sth = aievec.ext %0 {index = 1 : i8} : vector<32xi16>, vector<16xi16>
      %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<9xi16>, vector<16xi16>
      %2 = aievec.broadcast %1 {idx = 0 : i8} : vector<16xi16>, vector<16xi16>
      %3 = arith.muli %sbh, %2 : vector<16xi16>
      %4 = aievec.shift %sbh, %sth, %c2_i32 {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
      %5 = aievec.broadcast %1 {idx = 1 : i8} : vector<16xi16>, vector<16xi16>
      %6 = arith.muli %4, %5 : vector<16xi16>
      %7 = arith.addi %3, %6 : vector<16xi16>
      %8 = aievec.shift %sbh, %sth, %c4_i32 {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
      %9 = aievec.broadcast %1 {idx = 2 : i8} : vector<16xi16>, vector<16xi16>
      %10 = arith.muli %8, %9 : vector<16xi16>
      %11 = arith.addi %7, %10 : vector<16xi16>
      vector.transfer_write %11, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi16>, memref<16x256xi16>
    }
  }
  return
}

// --- "aie2p" diverges from "aie2" in convert-vector-to-aievec: math.exp ----
// --- lowers to `aievec.exp` only on AIE2P.                              ----

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s --check-prefix=EXP-AIE2
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s --check-prefix=EXP-AIE2P

// EXP-AIE2-LABEL: func.func @test_exp
// EXP-AIE2-NOT: aievec.exp
// EXP-AIE2: call @getExpBf16
func.func @test_exp(%a: vector<16xbf16>) -> vector<16xbf16> {
  %0 = math.exp %a : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// EXP-AIE2P-LABEL: func.func @test_exp
// EXP-AIE2P: aievec.exp
// EXP-AIE2P-NOT: call @getExpBf16

// --- ...but "aie2p" collapses back to "aie2" in optimize-aievec: this is ---
// --- the deliberate asymmetry documented on `isAIE2Family()` and on the  ---
// --- `isAIE1() ? AIEv1 : AIE2` branch in VectorToVectorConversions.cpp. ---

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie2" \
// RUN:   --lower-vector-to-aievec="aie-target=aie2" -o %t.pre-opt.mlir
// RUN: aie-opt %t.pre-opt.mlir --optimize-aievec="aie-target=aie2 shift=10" -o %t.opt-aie2.mlir
// RUN: aie-opt %t.pre-opt.mlir --optimize-aievec="aie-target=aie2p shift=10" -o %t.opt-aie2p.mlir
// RUN: diff %t.opt-aie2.mlir %t.opt-aie2p.mlir
