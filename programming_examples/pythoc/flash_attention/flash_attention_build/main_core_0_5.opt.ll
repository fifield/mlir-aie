; ModuleID = '/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/main_core_0_5.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@g_stage3 = external global [6144 x bfloat]
@k_stage3 = external global [64 x [96 x bfloat]]
@q_stage3 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @matmul_a_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @zero_fill_g_bf16(ptr) local_unnamed_addr

define void @core_0_5() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %8
  %2 = phi i64 [ 0, %0 ], [ %9, %8 ]
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %3

3:                                                ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %6, %3 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_stage3, ptr nonnull @k_stage3, ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_stage3, ptr nonnull @k_stage3, ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_stage3, ptr nonnull @k_stage3, ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %5 = or disjoint i64 %4, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_stage3, ptr nonnull @k_stage3, ptr nonnull @g_stage3)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %6 = add nuw nsw i64 %4, 4
  %7 = icmp samesign ult i64 %5, 31
  br i1 %7, label %3, label %8

8:                                                ; preds = %3
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %9 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %9, 9223372036854775807
  br i1 %.not, label %10, label %1

10:                                               ; preds = %8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
