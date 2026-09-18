//===- lower_cascade_put_get_aie4.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIE4 cascade lowering. The sibling lower_cascade_put_get.mlir covers AIE2;
// this file exists because AIE4 used to fall through to AIE2P's answer in two
// separate places, and each failure mode is pinned below:
//
//  1. getAIE4Intrinsics() declared only debug/event/acquire/release, and the
//     funcName selection ended in `else -> "llvm.aie2p.mcd.write.vec"`. On an
//     AIE4 target that aborted with "Could not find the intrinsic function".
//     The CHECKs here name llvm.aie4.* so a regression cannot pass by
//     emitting the AIE2P name.
//  2. The enable operand was appended only for isa<AIE2TargetModel>, which
//     AIE4's target model is not derived from, so the call came out one
//     operand short: "incorrect number of operands for callee". Every CHECK
//     below therefore spells the full (vector, i32) signature rather than
//     using {{.*}} for the operand list.
//
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering --split-input-file %s | FileCheck %s

// bf16 needs a bitcast to the i32 accumulator vector on the way out.
// CHECK-LABEL: aie4_put_cascade_bf16
// CHECK: %[[VECREAD:.*]] = vector.transfer_read
// CHECK: %[[BITCAST:.*]] = vector.bitcast %[[VECREAD]] : vector<32xbf16> to vector<16xi32>
// CHECK: call @llvm.aie4.mcd.write.vec(%[[BITCAST]], %c1{{.*}}) : (vector<16xi32>, i32) -> ()
// CHECK-NOT: llvm.aie2p.
module @aie4_put_cascade_bf16 {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  %l0 = aie.lock(%t02, 0) {init = 1 : i32}
  %buf = aie.buffer(%t02) { sym_name = "a" } : memref<32xbf16, 2 : i32>
  aie.core(%t02) {
    %c1 = arith.constant 1 : i32
    aie.use_lock(%l0, AcquireGreaterEqual, %c1)
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %vec = vector.transfer_read %buf[%c0], %cst {in_bounds = [true]} : memref<32xbf16, 2 : i32>, vector<32xbf16>
    aie.put_cascade(%vec : vector<32xbf16>)
    aie.end
  }
 }
}

// -----

// i32 is already the accumulator vector type: no bitcast, but still an enable.
// CHECK-LABEL: aie4_put_cascade_i32
// CHECK: %[[VECREAD:.*]] = vector.transfer_read
// CHECK-NOT: vector.bitcast
// CHECK: call @llvm.aie4.mcd.write.vec(%[[VECREAD]], %c1{{.*}}) : (vector<16xi32>, i32) -> ()
module @aie4_put_cascade_i32 {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  %l0 = aie.lock(%t02, 0) {init = 1 : i32}
  %buf = aie.buffer(%t02) { sym_name = "b" } : memref<16xi32, 2 : i32>
  aie.core(%t02) {
    %c1 = arith.constant 1 : i32
    aie.use_lock(%l0, AcquireGreaterEqual, %c1)
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %vec = vector.transfer_read %buf[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32, 2 : i32>, vector<16xi32>
    aie.put_cascade(%vec : vector<16xi32>)
    aie.end
  }
 }
}

// -----

// The read side takes the enable as its only operand.
// CHECK-LABEL: aie4_get_cascade_bf16
// CHECK: %[[GET:.*]] = call @llvm.aie4.scd.read.vec(%c1{{.*}}) : (i32) -> vector<16xi32>
// CHECK: %[[BITCAST:.*]] = vector.bitcast %[[GET]] : vector<16xi32> to vector<32xbf16>
// CHECK: vector.transfer_write %[[BITCAST]]
// CHECK-NOT: llvm.aie2p.
module @aie4_get_cascade_bf16 {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  %l0 = aie.lock(%t02, 0) {init = 1 : i32}
  %buf = aie.buffer(%t02) { sym_name = "c" } : memref<32xbf16, 2 : i32>
  aie.core(%t02) {
    %c1 = arith.constant 1 : i32
    aie.use_lock(%l0, AcquireGreaterEqual, %c1)
    %c0 = arith.constant 0 : index
    %vec = aie.get_cascade() : vector<32xbf16>
    vector.transfer_write %vec, %buf[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, 2 : i32>
    aie.end
  }
 }
}

// -----

// CHECK-LABEL: aie4_get_cascade_i32
// CHECK: %[[GET:.*]] = call @llvm.aie4.scd.read.vec(%c1{{.*}}) : (i32) -> vector<16xi32>
// CHECK-NOT: vector.bitcast
// CHECK: vector.transfer_write %[[GET]]
module @aie4_get_cascade_i32 {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  %l0 = aie.lock(%t02, 0) {init = 1 : i32}
  %buf = aie.buffer(%t02) { sym_name = "d" } : memref<16xi32, 2 : i32>
  aie.core(%t02) {
    %c1 = arith.constant 1 : i32
    aie.use_lock(%l0, AcquireGreaterEqual, %c1)
    %c0 = arith.constant 0 : index
    %vec = aie.get_cascade() : vector<16xi32>
    vector.transfer_write %vec, %buf[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32, 2 : i32>
    aie.end
  }
 }
}

// -----

// AIE2P must keep its own name. Without this the AIE4 arm could be written as
// an unconditional rename and still pass everything above.
// CHECK-LABEL: aie2p_put_cascade_unchanged
// CHECK: call @llvm.aie2p.mcd.write.vec(%{{.*}}, %c1{{.*}}) : (vector<16xi32>, i32) -> ()
// CHECK-NOT: llvm.aie4.
module @aie2p_put_cascade_unchanged {
 aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  %l0 = aie.lock(%t02, 0) {init = 1 : i32}
  %buf = aie.buffer(%t02) { sym_name = "e" } : memref<16xi32, 2 : i32>
  aie.core(%t02) {
    %c1 = arith.constant 1 : i32
    aie.use_lock(%l0, AcquireGreaterEqual, %c1)
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %vec = vector.transfer_read %buf[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32, 2 : i32>, vector<16xi32>
    aie.put_cascade(%vec : vector<16xi32>)
    aie.end
  }
 }
}

// -----

// AIE1 must NOT get an enable operand. This is the case the arch check has to
// get right: `isa<AIE2TargetModel>` excluded AIE1 and AIE4 together, and the
// replacement `arch != AIE1` has to keep excluding AIE1 while including AIE4.
// Writing it as an unconditional append passes every other case in this file
// and breaks here.
// CHECK-LABEL: aie1_cascade_no_enable
// CHECK: %[[GET:.*]] = call @llvm.aie.get.scd() : () -> i384
// CHECK: call @llvm.aie.put.mcd(%[[GET]]) : (i384) -> ()
// CHECK-NOT: llvm.aie4.
// CHECK-NOT: llvm.aie2
module @aie1_cascade_no_enable {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  aie.core(%t33) {
    %val = aie.get_cascade() : i384
    aie.put_cascade(%val : i384)
    aie.end
  }
 }
}

// -----

// AIE2 (xcve2802) keeps its own symbols and its enable.
// CHECK-LABEL: aie2_cascade_unchanged
// CHECK: call @llvm.aie2.mcd.write.vec(%{{.*}}, %c1{{.*}}) : (vector<16xi32>, i32) -> ()
// CHECK-NOT: llvm.aie4.
module @aie2_cascade_unchanged {
 aie.device(xcve2802) {
  %t33 = aie.tile(3, 3)
  %l0 = aie.lock(%t33, 0) {init = 1 : i32}
  %buf = aie.buffer(%t33) { sym_name = "g" } : memref<16xi32, 2 : i32>
  aie.core(%t33) {
    %c1 = arith.constant 1 : i32
    aie.use_lock(%l0, AcquireGreaterEqual, %c1)
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %vec = vector.transfer_read %buf[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32, 2 : i32>, vector<16xi32>
    aie.put_cascade(%vec : vector<16xi32>)
    aie.end
  }
 }
}
