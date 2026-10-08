// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b" -convert-scf-to-cf -convert-to-llvm | aie-translate --mlir-to-llvmir -o %t.ll
// RUN: FileCheck %s < %t.ll
// RUN: FileCheck %s --check-prefix=DECL < %t.ll
// RUN: grep -c 'declare <32 x i32> @llvm.aie4.get.ss(i32)' %t.ll | FileCheck %s --check-prefix=ONE

// Flat matrix-state ops through to LLVM IR.

// A stream pop feeds the int8 insert through a bitcast.
// CHECK-LABEL: define void @stage_stream
// CHECK: %[[RAW:[0-9]+]] = call <32 x i32> @llvm.aie4.get.ss(i32 1)
// CHECK: %[[IMG:[0-9]+]] = bitcast <32 x i32> %[[RAW]] to <128 x i8>
// CHECK: call void @llvm.aie4.insert.staging.int.I1024(<128 x i8> %[[IMG]], i32 %{{[0-9]+}}, i32 3)
// CHECK: call void @llvm.aie4.staging.to.matrix.I32768()
func.func @stage_stream(%slot: i32) {
  %p = arith.constant 1 : i32
  aievec.aie4.stage_stream int8 %p, %slot {rhs_signed = true} : i32, i32
  aievec.aie4.commit int8 {rhs_signed = true}
  return
}

// A loop-derived zero_if becomes zext + or with the static conf 776, and
// calls the same MMAC intrinsic as the static form.
// CHECK-LABEL: define <64 x i32> @mmac_zero_if_loop
// CHECK: %[[Z:[0-9]+]] = zext i1 %{{[0-9]+}} to i32
// CHECK: %[[CONF:[0-9]+]] = or i32 %[[Z]], 776
// CHECK: call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 %[[CONF]])
func.func @mmac_zero_if_loop(%a: vector<64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<64xi32>) {
    %z = arith.cmpi eq, %k, %c0 : index
    %n = aievec.aie4.mmac_active int8 %a, %acc zero_if %z {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    scf.yield %n : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// The aie.get_stream lowering declares @llvm.aie4.get.ss as <32 x i32> (i32);
// stage_stream in the same function reuses that declaration.
// CHECK-LABEL: define void @get_stream_and_stage_stream
// CHECK: call <32 x i32> @llvm.aie4.get.ss(i32 0)
// CHECK: call <32 x i32> @llvm.aie4.get.ss(i32 0)
// CHECK: call void @llvm.aie4.insert.staging.bf16.I1024(<64 x bfloat> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 2)
func.func private @llvm.aie4.get.ss(i32) -> vector<32xi32>
func.func @get_stream_and_stage_stream(%slot: i32, %out: memref<32xi32>) {
  %p = arith.constant 0 : i32
  %v = func.call @llvm.aie4.get.ss(%p) : (i32) -> vector<32xi32>
  aievec.aie4.stage_stream bf16 %p, %slot : i32, i32
  %c0 = arith.constant 0 : index
  vector.store %v, %out[%c0] : memref<32xi32>, vector<32xi32>
  return
}

// Exactly one declaration of the stream read and of the MMAC.
// ONE: {{^}}1{{$}}
// DECL-COUNT-1: declare <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32>, <64 x i8>, i32)
// DECL-NOT: @llvm.aie4.I512.ACC2048.mmac(<64 x i32>, <64 x i8>, i32)
