//===- lower_stream_aie4.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check architecture-specific stream intrinsic names and types.
// Invalid reads are covered in lower_stream_aie4_invalid.mlir.
//
// RUN: aie-opt --aie-standard-lowering --split-input-file %s | FileCheck %s

// CHECK-LABEL: aie4_put_stream
// CHECK: call @llvm.aie4.put.ms(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
// CHECK-NOT: llvm.aie2p.
module @aie4_put_stream {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    %val = arith.constant 16 : i32
    aie.put_stream(%c0 : i32, %val : i32)
    aie.end
  }
 }
}

// -----

// CHECK-LABEL: aie4_put_stream_vectors
// CHECK-DAG: func.func private @llvm.aie4.put.ms.v8(vector<8xi32>, i32)
// CHECK-DAG: func.func private @llvm.aie4.put.ms.v16(vector<16xi32>, i32)
// CHECK: arith.constant dense<2> : vector<16xi32>
// CHECK: %[[LAST8:.*]] = arith.constant 0 : i32
// CHECK-NEXT: call @llvm.aie4.put.ms.v8(%{{.*}}, %[[LAST8]]) : (vector<8xi32>, i32) -> ()
// CHECK: %[[LAST16:.*]] = arith.constant 0 : i32
// CHECK-NEXT: call @llvm.aie4.put.ms.v16(%{{.*}}, %[[LAST16]]) : (vector<16xi32>, i32) -> ()
module @aie4_put_stream_vectors {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    %v8 = arith.constant dense<1> : vector<8xi32>
    %v16 = arith.constant dense<2> : vector<16xi32>
    aie.put_stream(%c0 : i32, %v8 : vector<8xi32>)
    aie.put_stream(%c0 : i32, %v16 : vector<16xi32>)
    aie.end
  }
 }
}

// -----

// Check both read widths and zero-extension of an i8 stream index.
// CHECK-LABEL: aie4_get_stream
// CHECK-DAG: func.func private @llvm.aie4.get.ss(i32) -> vector<32xi32>
// CHECK-DAG: func.func private @llvm.aie4.getl.wss(i32) -> vector<16xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: call @llvm.aie4.get.ss(%[[C0]]) : (i32) -> vector<32xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : i8
// CHECK: %[[EXT:.*]] = arith.extui %[[C1]] : i8 to i32
// CHECK: call @llvm.aie4.getl.wss(%[[EXT]]) : (i32) -> vector<16xi32>
// CHECK-NOT: llvm.aie2p.
module @aie4_get_stream {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    %0 = aie.get_stream(%c0 : i32) : vector<32xi32>
    %c1 = arith.constant 1 : i8
    %1 = aie.get_stream(%c1 : i8) : vector<16xi32>
    aie.end
  }
 }
}

// -----

// Preserve AIE2P stream-write lowering.
// CHECK-LABEL: aie2p_put_stream_unchanged
// CHECK: call @llvm.aie2p.put.ms(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
// CHECK-NOT: llvm.aie4.
module @aie2p_put_stream_unchanged {
 aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    %val = arith.constant 16 : i32
    aie.put_stream(%c0 : i32, %val : i32)
    aie.end
  }
 }
}
