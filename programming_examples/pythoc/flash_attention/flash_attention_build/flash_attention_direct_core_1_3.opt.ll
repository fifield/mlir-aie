; ModuleID = '/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_1_3.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@v_stage1 = external global [64 x [96 x bfloat]]
@g_in_stage1 = external global [6144 x bfloat]
@gp_in_stage1 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @matmul_g_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @zero_fill_gp_bf16(ptr) local_unnamed_addr

define void @core_1_3() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %7
  %1 = phi i64 [ 0, %0 ], [ %8, %7 ]
  br label %2

2:                                                ; preds = %2, %.preheader
  %3 = phi i64 [ 0, %.preheader ], [ %5, %2 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_in_stage1, ptr nonnull @v_stage1, ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_in_stage1, ptr nonnull @v_stage1, ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_in_stage1, ptr nonnull @v_stage1, ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %4 = or disjoint i64 %3, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_in_stage1, ptr nonnull @v_stage1, ptr nonnull @gp_in_stage1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %5 = add nuw nsw i64 %3, 4
  %6 = icmp samesign ult i64 %4, 31
  br i1 %6, label %2, label %7

7:                                                ; preds = %2
  %8 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %8, 9223372036854775807
  br i1 %.not, label %9, label %.preheader

9:                                                ; preds = %7
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
