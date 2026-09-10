//===- base.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: mesimulator
//
// Build kernel and generate cert ELF:
// RUN: mkdir -p project0
// RUN: cp %aie_obj_root/third_party/cert/main/cert_medusa.elf project0/cert_medusa.elf
// RUN: aiecc --tmpdir=project0 --xchesscc --xbridge --aie-generate-txn --txn-name=transaction.bin %s
// RUN: aiecc --tmpdir=project1 --xchesscc --xbridge --no-compile-host --generate-full-elf --full-elf-name=aie4.elf --aiesim %S/aie_trace.mlir %test_utils_flags %S/test.cpp
// RUN: project1/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: PASS!
//
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) { elf_file = "cert_medusa.elf" }
    aie.flow(%tile_0_0, Control32 : 0, %tile_0_0, Controller32 : 0)
  }
}
