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

define void @core_0_4() {
  br label %1

1:                                                ; preds = %10, %0
  %2 = phi i64 [ %11, %10 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %12

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %9, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @zero_fill_g_bf16(ptr @g_stage2)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @matmul_a_b_bf16(ptr @q_stage2, ptr @k_stage2, ptr @g_stage2)
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %9 = add i64 %6, 1
  br label %5

10:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 51, i32 1)
  %11 = add i64 %2, 1
  br label %1

12:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
