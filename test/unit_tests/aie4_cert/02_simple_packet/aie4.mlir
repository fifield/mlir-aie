// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator
//
// RUN: aiecc -v --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile --no-compile-host --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-opt --convert-aie-to-transaction='elf-dir=%aie_obj_root/third_party/cert/main' %s -o transaction.mlir 
// RUN: aie-translate -aie-npu-to-binary -o transaction.bin transaction.mlir
// RUN: aie4.mlir.prj/mesim.sh | FileCheck %s
//
// CHECK: PASS!
//
module {

  aie.device(npu3) {
    %t00 = aie.tile(0, 0) { elf_file = "cert_medusa.elf" }
  }

}