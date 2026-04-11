; ModuleID = '/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/main_core_2_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@tmp_sp_stage0 = external global [64 x [1 x bfloat]]
@r_from_local_stage0 = external global [64 x [1 x bfloat]]
@r_from_cascade_stage0 = external global [64 x [1 x bfloat]]
@prev_up_stage0 = external global [64 x [1 x bfloat]]
@cascade_sp_stage0 = external global [64 x [1 x bfloat]]
@cascade_up_stage0 = external global [64 x [1 x bfloat]]
@softmax_out_stage0 = external global [64 x [64 x bfloat]]
@softmax_gv_stage0 = external global [64 x [64 x bfloat]]
@softmax_g_copy_stage0 = external global [6144 x bfloat]
@softmax_g_stage0 = external global [6144 x bfloat]
@softmax_s_stage0 = external global [64 x [1 x bfloat]]
@softmax_r_stage0 = external global [64 x [1 x bfloat]]
@softmax_u_stage0 = external global [64 x [1 x bfloat]]
@softmax_up_stage0 = external global [64 x [1 x bfloat]]
@softmax_sp_stage0 = external global [64 x [1 x bfloat]]
@softmax_gp_stage0 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #1

declare void @zero_fill_gp_bf16(ptr) local_unnamed_addr

declare void @zero_fill_sp_bf16(ptr) local_unnamed_addr

declare void @neg_inf_fill_up_bf16(ptr) local_unnamed_addr

declare void @max_g_bf16(ptr, ptr) local_unnamed_addr

declare void @maximum_up_u_bf16(ptr, ptr) local_unnamed_addr

declare void @exp_g_minus_u(ptr, ptr) local_unnamed_addr

declare void @exp_up_minus_u(ptr, ptr, ptr) local_unnamed_addr

declare void @mul_r_gp(ptr, ptr) local_unnamed_addr

declare void @sum_g(ptr, ptr) local_unnamed_addr

declare void @accum_sp_r_s(ptr, ptr, ptr) local_unnamed_addr

declare void @vector_copy_32elems(i32, ptr, ptr) local_unnamed_addr

declare void @vector_copy_32x96elems(i32, ptr, ptr) local_unnamed_addr

declare void @vector_accum_32x64elems(ptr, ptr) local_unnamed_addr

declare void @div_gp_sp(ptr, ptr) local_unnamed_addr

declare void @add_gp_g(ptr, ptr) local_unnamed_addr

define void @core_2_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %26
  %2 = phi i64 [ 0, %0 ], [ %27, %26 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @softmax_gp_stage0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @softmax_sp_stage0)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @softmax_up_stage0)
  br label %3

3:                                                ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %6, %3 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @max_g_bf16(ptr nonnull @softmax_g_stage0, ptr nonnull @softmax_u_stage0)
  tail call void @maximum_up_u_bf16(ptr nonnull @softmax_up_stage0, ptr nonnull @softmax_u_stage0)
  tail call void @exp_g_minus_u(ptr nonnull @softmax_u_stage0, ptr nonnull @softmax_g_stage0)
  tail call void @exp_up_minus_u(ptr nonnull @softmax_up_stage0, ptr nonnull @softmax_u_stage0, ptr nonnull @softmax_r_stage0)
  tail call void @mul_r_gp(ptr nonnull @softmax_r_stage0, ptr nonnull @softmax_gp_stage0)
  tail call void @vector_copy_32x96elems(i32 0, ptr nonnull @softmax_g_stage0, ptr nonnull @softmax_g_copy_stage0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 54, i32 -1)
  tail call void @vector_accum_32x64elems(ptr nonnull @softmax_gv_stage0, ptr nonnull @softmax_gp_stage0)
  tail call void @llvm.aie2p.release(i32 55, i32 1)
  tail call void @sum_g(ptr nonnull @softmax_g_stage0, ptr nonnull @softmax_s_stage0)
  tail call void @accum_sp_r_s(ptr nonnull @softmax_sp_stage0, ptr nonnull @softmax_r_stage0, ptr nonnull @softmax_s_stage0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_s_stage0, ptr nonnull @softmax_sp_stage0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_u_stage0, ptr nonnull @softmax_up_stage0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %5 = or disjoint i64 %4, 1
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @max_g_bf16(ptr nonnull @softmax_g_stage0, ptr nonnull @softmax_u_stage0)
  tail call void @maximum_up_u_bf16(ptr nonnull @softmax_up_stage0, ptr nonnull @softmax_u_stage0)
  tail call void @exp_g_minus_u(ptr nonnull @softmax_u_stage0, ptr nonnull @softmax_g_stage0)
  tail call void @exp_up_minus_u(ptr nonnull @softmax_up_stage0, ptr nonnull @softmax_u_stage0, ptr nonnull @softmax_r_stage0)
  tail call void @mul_r_gp(ptr nonnull @softmax_r_stage0, ptr nonnull @softmax_gp_stage0)
  tail call void @vector_copy_32x96elems(i32 0, ptr nonnull @softmax_g_stage0, ptr nonnull @softmax_g_copy_stage0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 54, i32 -1)
  tail call void @vector_accum_32x64elems(ptr nonnull @softmax_gv_stage0, ptr nonnull @softmax_gp_stage0)
  tail call void @llvm.aie2p.release(i32 55, i32 1)
  tail call void @sum_g(ptr nonnull @softmax_g_stage0, ptr nonnull @softmax_s_stage0)
  tail call void @accum_sp_r_s(ptr nonnull @softmax_sp_stage0, ptr nonnull @softmax_r_stage0, ptr nonnull @softmax_s_stage0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_s_stage0, ptr nonnull @softmax_sp_stage0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_u_stage0, ptr nonnull @softmax_up_stage0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %6 = add nuw nsw i64 %4, 2
  %7 = icmp samesign ult i64 %5, 31
  br i1 %7, label %3, label %.preheader6

.preheader6:                                      ; preds = %3, %.preheader6
  %8 = phi i64 [ %12, %.preheader6 ], [ 0, %3 ]
  %9 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %10 = trunc nuw i64 %8 to i20
  %11 = getelementptr bfloat, ptr @softmax_out_stage0, i20 %10
  store <16 x i32> %9, ptr %11, align 64
  %12 = add nuw nsw i64 %8, 32
  %13 = icmp samesign ult i64 %8, 4064
  br i1 %13, label %.preheader6, label %.preheader5

.preheader5:                                      ; preds = %.preheader6, %.preheader5
  %14 = phi i64 [ %18, %.preheader5 ], [ 0, %.preheader6 ]
  %15 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %16 = trunc nuw i64 %14 to i20
  %17 = getelementptr bfloat, ptr @cascade_up_stage0, i20 %16
  store <16 x i32> %15, ptr %17, align 64
  %18 = add nuw nsw i64 %14, 32
  %19 = icmp eq i64 %14, 0
  br i1 %19, label %.preheader5, label %.preheader

.preheader:                                       ; preds = %.preheader5, %.preheader
  %20 = phi i64 [ %24, %.preheader ], [ 0, %.preheader5 ]
  %21 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %22 = trunc nuw i64 %20 to i20
  %23 = getelementptr bfloat, ptr @cascade_sp_stage0, i20 %22
  store <16 x i32> %21, ptr %23, align 64
  %24 = add nuw nsw i64 %20, 32
  %25 = icmp eq i64 %20, 0
  br i1 %25, label %.preheader, label %26

26:                                               ; preds = %.preheader
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_up_stage0, ptr nonnull @prev_up_stage0)
  tail call void @maximum_up_u_bf16(ptr nonnull @cascade_up_stage0, ptr nonnull @softmax_up_stage0)
  tail call void @exp_up_minus_u(ptr nonnull @cascade_up_stage0, ptr nonnull @softmax_up_stage0, ptr nonnull @r_from_cascade_stage0)
  tail call void @exp_up_minus_u(ptr nonnull @prev_up_stage0, ptr nonnull @softmax_up_stage0, ptr nonnull @r_from_local_stage0)
  tail call void @mul_r_gp(ptr nonnull @r_from_cascade_stage0, ptr nonnull @softmax_out_stage0)
  tail call void @mul_r_gp(ptr nonnull @r_from_local_stage0, ptr nonnull @softmax_gp_stage0)
  tail call void @add_gp_g(ptr nonnull @softmax_gp_stage0, ptr nonnull @softmax_out_stage0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @tmp_sp_stage0)
  tail call void @accum_sp_r_s(ptr nonnull @cascade_sp_stage0, ptr nonnull @r_from_cascade_stage0, ptr nonnull @tmp_sp_stage0)
  tail call void @accum_sp_r_s(ptr nonnull @softmax_sp_stage0, ptr nonnull @r_from_local_stage0, ptr nonnull @tmp_sp_stage0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @tmp_sp_stage0, ptr nonnull @cascade_sp_stage0)
  tail call void @div_gp_sp(ptr nonnull @cascade_sp_stage0, ptr nonnull @softmax_out_stage0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %27 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %27, 9223372036854775807
  br i1 %.not, label %28, label %1

28:                                               ; preds = %26
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
