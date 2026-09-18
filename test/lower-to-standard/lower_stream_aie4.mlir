//===- lower_stream_aie4.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIE4 stream put/get lowering.
//
// Same class of bug as the cascade ops in lower_cascade_put_get_aie4.mlir:
// the funcName selection for aie.put_stream / aie.get_stream ended in
// `else -> "llvm.aie2p.put."`, so an AIE4 target asked for the AIE2P symbol.
// The AIE4 intrinsic table did not declare either name, so the lookup failed.
// These CHECKs name llvm.aie4.* explicitly.
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

// CHECK-LABEL: aie4_get_stream
// CHECK: call @llvm.aie4.get.ss() : () -> (i32, i32)
// CHECK-NOT: llvm.aie2p.
module @aie4_get_stream {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    %0 = aie.get_stream(%c0 : i32) : i32
    aie.end
  }
 }
}

// -----

// AIE2P keeps its own symbol, so the AIE4 arm cannot be an unconditional
// rename.
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
