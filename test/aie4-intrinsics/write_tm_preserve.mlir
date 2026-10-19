// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Identical tile-mapped writes can enqueue separate hardware tasks. Preserve
// both through MLIR and Peano optimization, without a barrier masking incorrect
// memory effects. Peano models write.tm as argmem read/write for this reason.
//
// REQUIRES: peano
// RUN: aie-opt %s --canonicalize --cse | aie-translate --mlir-to-llvmir > %t.ll
// RUN: FileCheck %s < %t.ll
// RUN: %PEANO_INSTALL_DIR/bin/opt -S -mtriple=aie4 -passes=early-cse %t.ll | FileCheck %s
// RUN: %PEANO_INSTALL_DIR/bin/opt -S -mtriple=aie4 -passes='default<O2>' %t.ll | FileCheck %s
//
// CHECK-LABEL: define void @write_tm_twice(
// CHECK: call void @llvm.aie4.write.tm(i32 16711680, ptr %{{.*}})
// CHECK-NEXT: {{.*}}call void @llvm.aie4.write.tm(i32 16711680, ptr %{{.*}})
// CHECK-NEXT: ret void
llvm.func @write_tm_twice(%address: !llvm.ptr) {
  %task = llvm.mlir.constant(16711680 : i32) : i32
  xllvm.intr.aie4.write.tm %task, %address : !llvm.ptr
  xllvm.intr.aie4.write.tm %task, %address : !llvm.ptr
  llvm.return
}
