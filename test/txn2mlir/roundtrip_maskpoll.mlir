//===- roundtrip_maskpoll.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: rm -rf roundtrip_maskpoll && mkdir -p roundtrip_maskpoll
// RUN: aie-translate -aie-npu-to-binary %s -o roundtrip_maskpoll/cfg
// RUN: %python txn2mlir.py -f roundtrip_maskpoll/cfg | FileCheck %s

// A mask poll survives the binary round trip: it is parsed back from the
// transaction (same word layout as maskwrite32) and re-emitted as
// aiex.npu.maskpoll, in order with the surrounding writes.
// CHECK: aie.device(npu1_1col)
// CHECK: %[[P_ADDR:.*]] = arith.constant 2224128 : i32
// CHECK: %[[P_VAL:.*]] = arith.constant 2 : i32
// CHECK: %[[P_MASK:.*]] = arith.constant 6 : i32
// CHECK: aiex.npu.maskpoll(%[[P_ADDR]], %[[P_VAL]], %[[P_MASK]]) : i32, i32, i32
// CHECK: %[[W_ADDR:.*]] = arith.constant 2224132 : i32
// CHECK: aiex.npu.write32(%[[W_ADDR]], %[[P_VAL]]) : i32, i32
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence() {
      %addr = arith.constant 2224128 : i32
      %val = arith.constant 2 : i32
      %mask = arith.constant 6 : i32
      aiex.npu.maskpoll(%addr, %val, %mask) : i32, i32, i32
      %waddr = arith.constant 2224132 : i32
      aiex.npu.write32(%waddr, %val) : i32, i32
    }
  }
}
