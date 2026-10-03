; Copyright (C) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; Reduced Peano reproducer: fp8 staging inserts of tile B0, B1, then B0 again.
; MachineCSE deletes a repeated (value, index) insert of the third tile even
; though the second tile overwrote those staging chunks. Six inserts are
; expected; -disable-machine-cse keeps all six.
target triple = "aie4"
declare void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8>, i32)
declare void @llvm.aie4.staging.to.matrix.FP32768()
declare <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float>, <64 x i8>, i32)
define void @f(<128 x i8> %b0v0, <128 x i8> %b0v1, <128 x i8> %b1v0, <128 x i8> %b1v1, <64 x i8> %a, ptr %out) {
  call void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8> %b0v0, i32 0)
  call void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8> %b0v1, i32 1)
  call void @llvm.aie4.staging.to.matrix.FP32768()
  %r0 = call <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float> poison, <64 x i8> %a, i32 1)
  %o0 = getelementptr <64 x float>, ptr %out, i32 0
  store <64 x float> %r0, ptr %o0
  call void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8> %b1v0, i32 0)
  call void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8> %b1v1, i32 1)
  call void @llvm.aie4.staging.to.matrix.FP32768()
  %r1 = call <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float> poison, <64 x i8> %a, i32 1)
  %o1 = getelementptr <64 x float>, ptr %out, i32 1
  store <64 x float> %r1, ptr %o1
  call void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8> %b0v0, i32 0)
  call void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8> %b0v1, i32 1)
  call void @llvm.aie4.staging.to.matrix.FP32768()
  %r2 = call <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float> poison, <64 x i8> %a, i32 1)
  %o2 = getelementptr <64 x float>, ptr %out, i32 2
  store <64 x float> %r2, ptr %o2
  ret void
}
