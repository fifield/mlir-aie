; Copyright (C) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; Reduced Peano reproducer: int8 staging of tile B0, B1, B0 (one chunk each),
; each followed by a commit and an MMAC. B1 is inserted straight from a load.
; The post-legalizer custom combiner fuses that load and insert into a staging load
; (vldb za[i], [p]) at the load's position, before tile B0's insert and
; commit, so tile B1 is staged too early.
target triple = "aie4"
declare void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8>, i32)
declare void @llvm.aie4.insert.staging.int.I1024(<128 x i8>, i32, i32)
declare void @llvm.aie4.staging.to.matrix.FP32768()
declare void @llvm.aie4.staging.to.matrix.I32768()
declare <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float>, <64 x i8>, i32)
declare <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32>, <64 x i8>, i32)
define void @main(ptr %pb0, ptr %pb1, <64 x i8> %a, ptr %out) {
  %b0p0 = getelementptr <128 x i8>, ptr %pb0, i32 0
  %b0v0 = load <128 x i8>, ptr %b0p0
  %b1p0 = getelementptr <128 x i8>, ptr %pb1, i32 0
  %b1v0 = load <128 x i8>, ptr %b1p0
  call void @llvm.aie4.insert.staging.int.I1024(<128 x i8> %b0v0, i32 0, i32 3)
  call void @llvm.aie4.staging.to.matrix.I32768()
  %r0 = call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> poison, <64 x i8> %a, i32 777)
  %o0 = getelementptr <64 x i32>, ptr %out, i32 0
  store <64 x i32> %r0, ptr %o0
  call void @llvm.aie4.insert.staging.int.I1024(<128 x i8> %b1v0, i32 0, i32 3)
  call void @llvm.aie4.staging.to.matrix.I32768()
  %r1 = call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> poison, <64 x i8> %a, i32 777)
  %o1 = getelementptr <64 x i32>, ptr %out, i32 1
  store <64 x i32> %r1, ptr %o1
  call void @llvm.aie4.insert.staging.int.I1024(<128 x i8> %b0v0, i32 0, i32 3)
  call void @llvm.aie4.staging.to.matrix.I32768()
  %r2 = call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> poison, <64 x i8> %a, i32 777)
  %o2 = getelementptr <64 x i32>, ptr %out, i32 2
  store <64 x i32> %r2, ptr %o2
  ret void
}
