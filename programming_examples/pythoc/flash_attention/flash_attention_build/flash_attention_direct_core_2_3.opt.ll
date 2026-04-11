; ModuleID = '/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_2_3.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@tmp_sp_stage1 = external global [64 x [1 x bfloat]]
@r_from_local_stage1 = external global [64 x [1 x bfloat]]
@r_from_cascade_stage1 = external global [64 x [1 x bfloat]]
@prev_up_stage1 = external global [64 x [1 x bfloat]]
@cascade_sp_stage1 = external global [64 x [1 x bfloat]]
@cascade_up_stage1 = external global [64 x [1 x bfloat]]
@cascade_gp_stage1 = external global [64 x [64 x bfloat]]
@softmax_gv_stage1 = external global [64 x [64 x bfloat]]
@softmax_g_copy_stage1 = external global [6144 x bfloat]
@softmax_g_stage1 = external global [6144 x bfloat]
@softmax_s_stage1 = external global [64 x [1 x bfloat]]
@softmax_r_stage1 = external global [64 x [1 x bfloat]]
@softmax_u_stage1 = external global [64 x [1 x bfloat]]
@softmax_up_stage1 = external global [64 x [1 x bfloat]]
@softmax_sp_stage1 = external global [64 x [1 x bfloat]]
@softmax_gp_stage1 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32) #0

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

declare void @add_gp_g(ptr, ptr) local_unnamed_addr

define void @core_2_3() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %46
  %2 = phi i64 [ 0, %0 ], [ %47, %46 ]
  tail call void @zero_fill_gp_bf16(ptr nonnull @softmax_gp_stage1)
  tail call void @zero_fill_sp_bf16(ptr nonnull @softmax_sp_stage1)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @softmax_up_stage1)
  br label %3

3:                                                ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %6, %3 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @max_g_bf16(ptr nonnull @softmax_g_stage1, ptr nonnull @softmax_u_stage1)
  tail call void @maximum_up_u_bf16(ptr nonnull @softmax_up_stage1, ptr nonnull @softmax_u_stage1)
  tail call void @exp_g_minus_u(ptr nonnull @softmax_u_stage1, ptr nonnull @softmax_g_stage1)
  tail call void @exp_up_minus_u(ptr nonnull @softmax_up_stage1, ptr nonnull @softmax_u_stage1, ptr nonnull @softmax_r_stage1)
  tail call void @mul_r_gp(ptr nonnull @softmax_r_stage1, ptr nonnull @softmax_gp_stage1)
  tail call void @vector_copy_32x96elems(i32 0, ptr nonnull @softmax_g_stage1, ptr nonnull @softmax_g_copy_stage1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @vector_accum_32x64elems(ptr nonnull @softmax_gv_stage1, ptr nonnull @softmax_gp_stage1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @sum_g(ptr nonnull @softmax_g_stage1, ptr nonnull @softmax_s_stage1)
  tail call void @accum_sp_r_s(ptr nonnull @softmax_sp_stage1, ptr nonnull @softmax_r_stage1, ptr nonnull @softmax_s_stage1)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_s_stage1, ptr nonnull @softmax_sp_stage1)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_u_stage1, ptr nonnull @softmax_up_stage1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %5 = or disjoint i64 %4, 1
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @max_g_bf16(ptr nonnull @softmax_g_stage1, ptr nonnull @softmax_u_stage1)
  tail call void @maximum_up_u_bf16(ptr nonnull @softmax_up_stage1, ptr nonnull @softmax_u_stage1)
  tail call void @exp_g_minus_u(ptr nonnull @softmax_u_stage1, ptr nonnull @softmax_g_stage1)
  tail call void @exp_up_minus_u(ptr nonnull @softmax_up_stage1, ptr nonnull @softmax_u_stage1, ptr nonnull @softmax_r_stage1)
  tail call void @mul_r_gp(ptr nonnull @softmax_r_stage1, ptr nonnull @softmax_gp_stage1)
  tail call void @vector_copy_32x96elems(i32 0, ptr nonnull @softmax_g_stage1, ptr nonnull @softmax_g_copy_stage1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @vector_accum_32x64elems(ptr nonnull @softmax_gv_stage1, ptr nonnull @softmax_gp_stage1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @sum_g(ptr nonnull @softmax_g_stage1, ptr nonnull @softmax_s_stage1)
  tail call void @accum_sp_r_s(ptr nonnull @softmax_sp_stage1, ptr nonnull @softmax_r_stage1, ptr nonnull @softmax_s_stage1)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_s_stage1, ptr nonnull @softmax_sp_stage1)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_u_stage1, ptr nonnull @softmax_up_stage1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %6 = add nuw nsw i64 %4, 2
  %7 = icmp samesign ult i64 %5, 31
  br i1 %7, label %3, label %.preheader11

.preheader11:                                     ; preds = %3, %.preheader11
  %8 = phi i64 [ %12, %.preheader11 ], [ 0, %3 ]
  %9 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %10 = trunc nuw i64 %8 to i20
  %11 = getelementptr bfloat, ptr @cascade_gp_stage1, i20 %10
  store <16 x i32> %9, ptr %11, align 64
  %12 = add nuw nsw i64 %8, 32
  %13 = icmp samesign ult i64 %8, 4064
  br i1 %13, label %.preheader11, label %.preheader10

.preheader10:                                     ; preds = %.preheader11, %.preheader10
  %14 = phi i64 [ %18, %.preheader10 ], [ 0, %.preheader11 ]
  %15 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %16 = trunc nuw i64 %14 to i20
  %17 = getelementptr bfloat, ptr @cascade_up_stage1, i20 %16
  store <16 x i32> %15, ptr %17, align 64
  %18 = add nuw nsw i64 %14, 32
  %19 = icmp eq i64 %14, 0
  br i1 %19, label %.preheader10, label %.preheader9

.preheader9:                                      ; preds = %.preheader10, %.preheader9
  %20 = phi i64 [ %24, %.preheader9 ], [ 0, %.preheader10 ]
  %21 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %22 = trunc nuw i64 %20 to i20
  %23 = getelementptr bfloat, ptr @cascade_sp_stage1, i20 %22
  store <16 x i32> %21, ptr %23, align 64
  %24 = add nuw nsw i64 %20, 32
  %25 = icmp eq i64 %20, 0
  br i1 %25, label %.preheader9, label %26

26:                                               ; preds = %.preheader9
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @softmax_up_stage1, ptr nonnull @prev_up_stage1)
  tail call void @maximum_up_u_bf16(ptr nonnull @cascade_up_stage1, ptr nonnull @softmax_up_stage1)
  tail call void @exp_up_minus_u(ptr nonnull @cascade_up_stage1, ptr nonnull @softmax_up_stage1, ptr nonnull @r_from_cascade_stage1)
  tail call void @exp_up_minus_u(ptr nonnull @prev_up_stage1, ptr nonnull @softmax_up_stage1, ptr nonnull @r_from_local_stage1)
  tail call void @mul_r_gp(ptr nonnull @r_from_cascade_stage1, ptr nonnull @cascade_gp_stage1)
  tail call void @mul_r_gp(ptr nonnull @r_from_local_stage1, ptr nonnull @softmax_gp_stage1)
  tail call void @add_gp_g(ptr nonnull @softmax_gp_stage1, ptr nonnull @cascade_gp_stage1)
  tail call void @zero_fill_sp_bf16(ptr nonnull @tmp_sp_stage1)
  tail call void @accum_sp_r_s(ptr nonnull @cascade_sp_stage1, ptr nonnull @r_from_cascade_stage1, ptr nonnull @tmp_sp_stage1)
  tail call void @accum_sp_r_s(ptr nonnull @softmax_sp_stage1, ptr nonnull @r_from_local_stage1, ptr nonnull @tmp_sp_stage1)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @tmp_sp_stage1, ptr nonnull @cascade_sp_stage1)
  br label %27

27:                                               ; preds = %26, %27
  %28 = phi i64 [ 0, %26 ], [ %32, %27 ]
  %29 = trunc nuw i64 %28 to i20
  %30 = getelementptr bfloat, ptr @cascade_gp_stage1, i20 %29
  %31 = load <16 x i32>, ptr %30, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %31, i32 1)
  %32 = add nuw nsw i64 %28, 32
  %33 = icmp samesign ult i64 %28, 4064
  br i1 %33, label %27, label %.preheader8

.preheader8:                                      ; preds = %27, %.preheader8
  %34 = phi i64 [ %38, %.preheader8 ], [ 0, %27 ]
  %35 = trunc nuw i64 %34 to i20
  %36 = getelementptr bfloat, ptr @softmax_up_stage1, i20 %35
  %37 = load <16 x i32>, ptr %36, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %37, i32 1)
  %38 = add nuw nsw i64 %34, 32
  %39 = icmp eq i64 %34, 0
  br i1 %39, label %.preheader8, label %.preheader

.preheader:                                       ; preds = %.preheader8, %.preheader
  %40 = phi i64 [ %44, %.preheader ], [ 0, %.preheader8 ]
  %41 = trunc nuw i64 %40 to i20
  %42 = getelementptr bfloat, ptr @cascade_sp_stage1, i20 %41
  %43 = load <16 x i32>, ptr %42, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %43, i32 1)
  %44 = add nuw nsw i64 %40, 32
  %45 = icmp eq i64 %40, 0
  br i1 %45, label %.preheader, label %46

46:                                               ; preds = %.preheader
  %47 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %47, 9223372036854775807
  br i1 %.not, label %48, label %1

48:                                               ; preds = %46
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
