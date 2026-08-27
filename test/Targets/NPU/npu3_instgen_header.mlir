//===- npu3_instgen_header.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s

// CHECK: 062A0100
// CHECK: 00000103
// CHECK: 00000000
// CHECK: 00000010

module {
  aie.device(npu3) {
    aie.runtime_sequence() {
    }
  }
}
