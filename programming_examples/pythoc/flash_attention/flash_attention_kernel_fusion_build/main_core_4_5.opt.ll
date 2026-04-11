; ModuleID = '/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_kernel_fusion_build/main_core_4_5.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@r_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@s_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@sp_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@up_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@gp_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@g_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@v_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@q_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@qk_seg1_s3_q0 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #1

declare void @zero_fill_g_bf16_pythoc(ptr) local_unnamed_addr

declare void @zero_fill_gp_bf16_pythoc(ptr) local_unnamed_addr

declare void @zero_fill_sp_bf16_pythoc(ptr) local_unnamed_addr

declare void @neg_inf_fill_up_bf16_pythoc(ptr) local_unnamed_addr

declare void @copy_tile_pythoc(ptr, ptr) local_unnamed_addr

declare void @matmul_a_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @fused_softmax(ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @mul_r_gp(ptr, ptr) local_unnamed_addr

declare void @matmul_g_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @accum_sp_r_s(ptr, ptr, ptr) local_unnamed_addr

declare void @vector_copy_32elems_pythoc(i32, ptr, ptr) local_unnamed_addr

define void @core_4_5() local_unnamed_addr {
  br label %.preheader6.preheader

.preheader6.preheader:                            ; preds = %0, %20
  %1 = phi i64 [ 0, %0 ], [ %21, %20 ]
  tail call void @zero_fill_gp_bf16_pythoc(ptr nonnull @gp_seg1_s3_q0)
  tail call void @zero_fill_sp_bf16_pythoc(ptr nonnull @sp_seg1_s3_q0)
  tail call void @neg_inf_fill_up_bf16_pythoc(ptr nonnull @up_seg1_s3_q0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @copy_tile_pythoc(ptr nonnull @qk_seg1_s3_q0, ptr nonnull @q_seg1_s3_q0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @zero_fill_g_bf16_pythoc(ptr nonnull @g_seg1_s3_q0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_seg1_s3_q0, ptr nonnull @qk_seg1_s3_q0, ptr nonnull @g_seg1_s3_q0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @fused_softmax(ptr nonnull @g_seg1_s3_q0, ptr nonnull @up_seg1_s3_q0, ptr nonnull @s_seg1_s3_q0, ptr nonnull @r_seg1_s3_q0)
  tail call void @mul_r_gp(ptr nonnull @r_seg1_s3_q0, ptr nonnull @gp_seg1_s3_q0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_seg1_s3_q0, ptr nonnull @v_seg1_s3_q0, ptr nonnull @gp_seg1_s3_q0)
  tail call void @accum_sp_r_s(ptr nonnull @sp_seg1_s3_q0, ptr nonnull @r_seg1_s3_q0, ptr nonnull @s_seg1_s3_q0)
  tail call void @vector_copy_32elems_pythoc(i32 0, ptr nonnull @s_seg1_s3_q0, ptr nonnull @sp_seg1_s3_q0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16_pythoc(ptr nonnull @g_seg1_s3_q0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_seg1_s3_q0, ptr nonnull @qk_seg1_s3_q0, ptr nonnull @g_seg1_s3_q0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @fused_softmax(ptr nonnull @g_seg1_s3_q0, ptr nonnull @up_seg1_s3_q0, ptr nonnull @s_seg1_s3_q0, ptr nonnull @r_seg1_s3_q0)
  tail call void @mul_r_gp(ptr nonnull @r_seg1_s3_q0, ptr nonnull @gp_seg1_s3_q0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_seg1_s3_q0, ptr nonnull @v_seg1_s3_q0, ptr nonnull @gp_seg1_s3_q0)
  tail call void @accum_sp_r_s(ptr nonnull @sp_seg1_s3_q0, ptr nonnull @r_seg1_s3_q0, ptr nonnull @s_seg1_s3_q0)
  tail call void @vector_copy_32elems_pythoc(i32 0, ptr nonnull @s_seg1_s3_q0, ptr nonnull @sp_seg1_s3_q0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  br label %.preheader6

.preheader6:                                      ; preds = %.preheader6.preheader, %.preheader6
  %2 = phi i64 [ %6, %.preheader6 ], [ 0, %.preheader6.preheader ]
  %3 = trunc nuw i64 %2 to i20
  %4 = getelementptr bfloat, ptr @gp_seg1_s3_q0, i20 %3
  %5 = load <16 x i32>, ptr %4, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %5, i32 1)
  %6 = add nuw nsw i64 %2, 32
  %7 = icmp samesign ult i64 %2, 4064
  br i1 %7, label %.preheader6, label %.preheader5

.preheader5:                                      ; preds = %.preheader6, %.preheader5
  %8 = phi i64 [ %12, %.preheader5 ], [ 0, %.preheader6 ]
  %9 = trunc nuw i64 %8 to i20
  %10 = getelementptr bfloat, ptr @up_seg1_s3_q0, i20 %9
  %11 = load <16 x i32>, ptr %10, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %11, i32 1)
  %12 = add nuw nsw i64 %8, 32
  %13 = icmp eq i64 %8, 0
  br i1 %13, label %.preheader5, label %.preheader

.preheader:                                       ; preds = %.preheader5, %.preheader
  %14 = phi i64 [ %18, %.preheader ], [ 0, %.preheader5 ]
  %15 = trunc nuw i64 %14 to i20
  %16 = getelementptr bfloat, ptr @sp_seg1_s3_q0, i20 %15
  %17 = load <16 x i32>, ptr %16, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %17, i32 1)
  %18 = add nuw nsw i64 %14, 32
  %19 = icmp eq i64 %14, 0
  br i1 %19, label %.preheader, label %20

20:                                               ; preds = %.preheader
  %21 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %21, 9223372036854775807
  br i1 %.not, label %22, label %.preheader6.preheader

22:                                               ; preds = %20
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
