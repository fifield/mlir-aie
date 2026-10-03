; Copyright (C) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; A staging insert at constant index 1 must not replace the unrelated
; loop increment k + 1. The faulty IndexedToAutoInc combine rewrites
; %next to the insert's post-increment result (2), so the PHI gets stuck.
target triple = "aie4"
declare void @llvm.aie4.insert.staging.bf16.I1024(<64 x bfloat>, i32, i32)
define void @main(ptr %a, ptr %b, ptr %out) {
entry:
  br label %loop
loop:
  %k = phi i32 [0, %entry], [%next, %loop]
  %ap = getelementptr <32 x bfloat>, ptr %a, i32 %k
  %av = load <32 x bfloat>, ptr %ap, align 64
  %bv = load <64 x bfloat>, ptr %b, align 64
  call void @llvm.aie4.insert.staging.bf16.I1024(<64 x bfloat> %bv, i32 1, i32 2)
  %op = getelementptr <32 x bfloat>, ptr %out, i32 %k
  store <32 x bfloat> %av, ptr %op, align 64
  %next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
