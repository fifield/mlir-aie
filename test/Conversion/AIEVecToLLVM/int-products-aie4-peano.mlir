// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Ensure every surviving widened product reaches native instructions, including
// the accumulator product in a sum, shared results, and mixed-sign operands.
// REQUIRES: peano
// RUN: aie-opt %S/../VectorToAIEVec/aie4-int-products.mlir --convert-vector-to-aievec="aie-target=aie4b" \
// RUN:   --convert-aievec-to-llvm="aie-target=aie4b" --convert-func-to-llvm \
// RUN:   --convert-to-llvm --reconcile-unrealized-casts -o %t.mlir
// RUN: aie-translate --mlir-to-llvmir %t.mlir -o %t.ll
// RUN: %PEANO_INSTALL_DIR/bin/opt -S -mtriple=aie4 %t.ll -o %t.opt.ll
// RUN: %PEANO_INSTALL_DIR/bin/llc -O2 -mtriple=aie4 -mcpu=aie4b -filetype=obj %t.opt.ll -o %t.o
// RUN: %PEANO_INSTALL_DIR/bin/llvm-objdump -d --no-show-raw-insn %t.o | FileCheck %s
// CHECK-LABEL: <two_products>:
// CHECK: vmac
// CHECK-LABEL: <shared_product>:
// CHECK: vmac
// CHECK-LABEL: <doubled_product>:
// CHECK: vmac
// CHECK-LABEL: <mixed_product>:
// CHECK-DAG: vunpack{{.*}}unpacksign0
// CHECK-DAG: vunpack{{.*}}unpacksign1
// CHECK-DAG: vmac
// CHECK-LABEL: <unsigned_product>:
// CHECK-DAG: vunpack{{.*}}unpacksign0
// CHECK-DAG: vmac
// CHECK-LABEL: <i16_product>:
// CHECK: vmac
// CHECK-LABEL: <two_fused_uses>:
// CHECK: vmac
