//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator
// RUN: %python aiecc.py --no-compile --aiesim %s %S/test.cpp
// RUN: aie.mlir.prj/aiesim.sh | FileCheck --check-prefix=AIESIM %s
// RUN: aie.mlir.prj/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// AIESIM: AIE4{{.*}}sim
// AIESIM: Hello, world.
// MESIM: device model: {{.*}}aie4
// MESIM: Hello, world.
//
module @test00_itsalive {
  aie.device(npu3) {
    %tile20 = aie.tile(2, 0)
    %tile22 = aie.tile(2, 2)

    aie.flow(%tile20, DMA : 0, %tile22, DMA : 0)
    aie.flow(%tile22, DMA : 0, %tile20, DMA : 0)
  }
}