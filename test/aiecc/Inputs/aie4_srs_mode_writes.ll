; Copyright (C) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; Input for aie4-srs-mode-writes-fused.test.

target triple = "aie4"

declare void @llvm.aie4.set.ctrl.reg(i32, i32)
declare <64 x i8> @llvm.aie4.I512.v64.acc32.srs(<64 x i32>, i32, i32)

; Entry writes, then a load / srs / store loop.
define void @entry_fused(ptr noalias %in, ptr noalias %out, i32 %n) {
entry:
  call void @llvm.aie4.set.ctrl.reg(i32 46, i32 9)
  call void @llvm.aie4.set.ctrl.reg(i32 45, i32 1)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr <64 x i32>, ptr %in, i32 %i
  %pb = getelementptr <64 x i8>, ptr %out, i32 %i
  %a = load <64 x i32>, ptr %pa, align 64
  %y = call <64 x i8> @llvm.aie4.I512.v64.acc32.srs(<64 x i32> %a, i32 12, i32 1)
  store <64 x i8> %y, ptr %pb, align 64
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Mid-function writes around one fused SRS.
define void @switch_around_fused(ptr noalias %in, ptr noalias %out) {
entry:
  %a = load <64 x i32>, ptr %in, align 64
  call void @llvm.aie4.set.ctrl.reg(i32 46, i32 9)
  %y = call <64 x i8> @llvm.aie4.I512.v64.acc32.srs(<64 x i32> %a, i32 12, i32 1)
  store <64 x i8> %y, ptr %out, align 64
  call void @llvm.aie4.set.ctrl.reg(i32 46, i32 12)
  ret void
}
