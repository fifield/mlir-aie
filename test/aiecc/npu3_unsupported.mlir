//===- npu3_unsupported.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc rejects AIE4 devices up front.

// RUN: rm -rf npu3_unsupported && mkdir -p npu3_unsupported && cd npu3_unsupported && not aiecc %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.device' op aiecc does not yet support the AIE4 architecture

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
  }
}
