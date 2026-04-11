; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@softmax_gv_stage3 = external global [64 x [64 x bfloat]]
@softmax_g_copy_stage3 = external global [6144 x bfloat]
@softmax_g_stage3 = external global [6144 x bfloat]
@softmax_s_stage3 = external global [64 x [1 x bfloat]]
@softmax_r_stage3 = external global [64 x [1 x bfloat]]
@softmax_u_stage3 = external global [64 x [1 x bfloat]]
@softmax_up_stage3 = external global [64 x [1 x bfloat]]
@softmax_sp_stage3 = external global [64 x [1 x bfloat]]
@softmax_gp_stage3 = external global [64 x [64 x bfloat]]
@v_stage3 = external global [64 x [96 x bfloat]]
@g_in_stage3 = external global [6144 x bfloat]
@gp_in_stage3 = external global [64 x [64 x bfloat]]
@g_stage3 = external global [6144 x bfloat]
@k_stage3 = external global [64 x [96 x bfloat]]
@q_stage3 = external global [64 x [64 x bfloat]]
@tmp_sp_stage2 = external global [64 x [1 x bfloat]]
@r_from_local_stage2 = external global [64 x [1 x bfloat]]
@r_from_cascade_stage2 = external global [64 x [1 x bfloat]]
@prev_up_stage2 = external global [64 x [1 x bfloat]]
@cascade_sp_stage2 = external global [64 x [1 x bfloat]]
@cascade_up_stage2 = external global [64 x [1 x bfloat]]
@cascade_gp_stage2 = external global [64 x [64 x bfloat]]
@softmax_gv_stage2 = external global [64 x [64 x bfloat]]
@softmax_g_copy_stage2 = external global [6144 x bfloat]
@softmax_g_stage2 = external global [6144 x bfloat]
@softmax_s_stage2 = external global [64 x [1 x bfloat]]
@softmax_r_stage2 = external global [64 x [1 x bfloat]]
@softmax_u_stage2 = external global [64 x [1 x bfloat]]
@softmax_up_stage2 = external global [64 x [1 x bfloat]]
@softmax_sp_stage2 = external global [64 x [1 x bfloat]]
@softmax_gp_stage2 = external global [64 x [64 x bfloat]]
@v_stage2 = external global [64 x [96 x bfloat]]
@g_in_stage2 = external global [6144 x bfloat]
@gp_in_stage2 = external global [64 x [64 x bfloat]]
@g_stage2 = external global [6144 x bfloat]
@k_stage2 = external global [64 x [96 x bfloat]]
@q_stage2 = external global [64 x [64 x bfloat]]
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
@v_stage1 = external global [64 x [96 x bfloat]]
@g_in_stage1 = external global [6144 x bfloat]
@gp_in_stage1 = external global [64 x [64 x bfloat]]
@g_stage1 = external global [6144 x bfloat]
@k_stage1 = external global [64 x [96 x bfloat]]
@q_stage1 = external global [64 x [64 x bfloat]]
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
@v_stage0 = external global [64 x [96 x bfloat]]
@g_in_stage0 = external global [6144 x bfloat]
@gp_in_stage0 = external global [64 x [64 x bfloat]]
@g_stage0 = external global [6144 x bfloat]
@k_stage0 = external global [64 x [96 x bfloat]]
@q_stage0 = external global [64 x [64 x bfloat]]
@out_l2 = external global [64 x [64 x bfloat]]
@v_l2_stage3 = external global [96 x [64 x bfloat]]
@v_l2_stage2 = external global [96 x [64 x bfloat]]
@v_l2_stage1 = external global [96 x [64 x bfloat]]
@v_l2_stage0 = external global [96 x [64 x bfloat]]
@k_l2_stage3 = external global [64 x [96 x bfloat]]
@q_l2_stage3 = external global [64 x [64 x bfloat]]
@k_l2_stage2 = external global [64 x [96 x bfloat]]
@q_l2_stage2 = external global [64 x [64 x bfloat]]
@k_l2_stage1 = external global [64 x [96 x bfloat]]
@q_l2_stage1 = external global [64 x [64 x bfloat]]
@k_l2_stage0 = external global [64 x [96 x bfloat]]
@q_l2_stage0 = external global [64 x [64 x bfloat]]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @matmul_a_b_bf16(ptr, ptr, ptr)

declare void @matmul_g_b_bf16(ptr, ptr, ptr)

declare void @zero_fill_gp_bf16(ptr)

declare void @zero_fill_sp_bf16(ptr)

declare void @zero_fill_g_bf16(ptr)

declare void @neg_inf_fill_up_bf16(ptr)

declare void @max_g_bf16(ptr, ptr)

declare void @maximum_up_u_bf16(ptr, ptr)

declare void @exp_g_minus_u(ptr, ptr)

declare void @exp_up_minus_u(ptr, ptr, ptr)

declare void @mul_r_gp(ptr, ptr)

declare void @sum_g(ptr, ptr)

declare void @accum_sp_r_s(ptr, ptr, ptr)

declare void @vector_copy_32elems(i32, ptr, ptr)

declare void @vector_copy_32x96elems(i32, ptr, ptr)

declare void @vector_accum_32x64elems(ptr, ptr)

declare void @div_gp_sp(ptr, ptr)

declare void @add_gp_g(ptr, ptr)

define void @core_2_4() {
  br label %1

1:                                                ; preds = %59, %0
  %2 = phi i64 [ %60, %59 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %61

4:                                                ; preds = %1
  call void @zero_fill_gp_bf16(ptr @softmax_gp_stage2)
  call void @zero_fill_sp_bf16(ptr @softmax_sp_stage2)
  call void @neg_inf_fill_up_bf16(ptr @softmax_up_stage2)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %9, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @max_g_bf16(ptr @softmax_g_stage2, ptr @softmax_u_stage2)
  call void @maximum_up_u_bf16(ptr @softmax_up_stage2, ptr @softmax_u_stage2)
  call void @exp_g_minus_u(ptr @softmax_u_stage2, ptr @softmax_g_stage2)
  call void @exp_up_minus_u(ptr @softmax_up_stage2, ptr @softmax_u_stage2, ptr @softmax_r_stage2)
  call void @mul_r_gp(ptr @softmax_r_stage2, ptr @softmax_gp_stage2)
  call void @vector_copy_32x96elems(i32 0, ptr @softmax_g_stage2, ptr @softmax_g_copy_stage2)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @vector_accum_32x64elems(ptr @softmax_gv_stage2, ptr @softmax_gp_stage2)
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @sum_g(ptr @softmax_g_stage2, ptr @softmax_s_stage2)
  call void @accum_sp_r_s(ptr @softmax_sp_stage2, ptr @softmax_r_stage2, ptr @softmax_s_stage2)
  call void @vector_copy_32elems(i32 0, ptr @softmax_s_stage2, ptr @softmax_sp_stage2)
  call void @vector_copy_32elems(i32 0, ptr @softmax_u_stage2, ptr @softmax_up_stage2)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %9 = add i64 %6, 1
  br label %5

10:                                               ; preds = %13, %5
  %11 = phi i64 [ %17, %13 ], [ 0, %5 ]
  %12 = icmp slt i64 %11, 4096
  br i1 %12, label %13, label %18

13:                                               ; preds = %10
  %14 = call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %15 = bitcast <16 x i32> %14 to <32 x bfloat>
  %16 = getelementptr bfloat, ptr @cascade_gp_stage2, i64 %11
  store <32 x bfloat> %15, ptr %16, align 2
  %17 = add i64 %11, 32
  br label %10

18:                                               ; preds = %21, %10
  %19 = phi i64 [ %25, %21 ], [ 0, %10 ]
  %20 = icmp slt i64 %19, 64
  br i1 %20, label %21, label %26

21:                                               ; preds = %18
  %22 = call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %23 = bitcast <16 x i32> %22 to <32 x bfloat>
  %24 = getelementptr bfloat, ptr @cascade_up_stage2, i64 %19
  store <32 x bfloat> %23, ptr %24, align 2
  %25 = add i64 %19, 32
  br label %18

26:                                               ; preds = %29, %18
  %27 = phi i64 [ %33, %29 ], [ 0, %18 ]
  %28 = icmp slt i64 %27, 64
  br i1 %28, label %29, label %34

29:                                               ; preds = %26
  %30 = call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %31 = bitcast <16 x i32> %30 to <32 x bfloat>
  %32 = getelementptr bfloat, ptr @cascade_sp_stage2, i64 %27
  store <32 x bfloat> %31, ptr %32, align 2
  %33 = add i64 %27, 32
  br label %26

34:                                               ; preds = %26
  call void @vector_copy_32elems(i32 0, ptr @softmax_up_stage2, ptr @prev_up_stage2)
  call void @maximum_up_u_bf16(ptr @cascade_up_stage2, ptr @softmax_up_stage2)
  call void @exp_up_minus_u(ptr @cascade_up_stage2, ptr @softmax_up_stage2, ptr @r_from_cascade_stage2)
  call void @exp_up_minus_u(ptr @prev_up_stage2, ptr @softmax_up_stage2, ptr @r_from_local_stage2)
  call void @mul_r_gp(ptr @r_from_cascade_stage2, ptr @cascade_gp_stage2)
  call void @mul_r_gp(ptr @r_from_local_stage2, ptr @softmax_gp_stage2)
  call void @add_gp_g(ptr @softmax_gp_stage2, ptr @cascade_gp_stage2)
  call void @zero_fill_sp_bf16(ptr @tmp_sp_stage2)
  call void @accum_sp_r_s(ptr @cascade_sp_stage2, ptr @r_from_cascade_stage2, ptr @tmp_sp_stage2)
  call void @accum_sp_r_s(ptr @softmax_sp_stage2, ptr @r_from_local_stage2, ptr @tmp_sp_stage2)
  call void @vector_copy_32elems(i32 0, ptr @tmp_sp_stage2, ptr @cascade_sp_stage2)
  br label %35

35:                                               ; preds = %38, %34
  %36 = phi i64 [ %42, %38 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, 4096
  br i1 %37, label %38, label %43

38:                                               ; preds = %35
  %39 = getelementptr bfloat, ptr @cascade_gp_stage2, i64 %36
  %40 = load <32 x bfloat>, ptr %39, align 2
  %41 = bitcast <32 x bfloat> %40 to <16 x i32>
  call void @llvm.aie2p.mcd.write.vec(<16 x i32> %41, i32 1)
  %42 = add i64 %36, 32
  br label %35

43:                                               ; preds = %46, %35
  %44 = phi i64 [ %50, %46 ], [ 0, %35 ]
  %45 = icmp slt i64 %44, 64
  br i1 %45, label %46, label %51

46:                                               ; preds = %43
  %47 = getelementptr bfloat, ptr @softmax_up_stage2, i64 %44
  %48 = load <32 x bfloat>, ptr %47, align 2
  %49 = bitcast <32 x bfloat> %48 to <16 x i32>
  call void @llvm.aie2p.mcd.write.vec(<16 x i32> %49, i32 1)
  %50 = add i64 %44, 32
  br label %43

51:                                               ; preds = %54, %43
  %52 = phi i64 [ %58, %54 ], [ 0, %43 ]
  %53 = icmp slt i64 %52, 64
  br i1 %53, label %54, label %59

54:                                               ; preds = %51
  %55 = getelementptr bfloat, ptr @cascade_sp_stage2, i64 %52
  %56 = load <32 x bfloat>, ptr %55, align 2
  %57 = bitcast <32 x bfloat> %56 to <16 x i32>
  call void @llvm.aie2p.mcd.write.vec(<16 x i32> %57, i32 1)
  %58 = add i64 %52, 32
  br label %51

59:                                               ; preds = %51
  %60 = add i64 %2, 1
  br label %1

61:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
