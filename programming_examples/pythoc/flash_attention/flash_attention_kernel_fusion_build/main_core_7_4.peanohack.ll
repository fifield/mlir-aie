; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@r_seg1_s3_q3 = external global [64 x [1 x bfloat]]
@s_seg1_s3_q3 = external global [64 x [1 x bfloat]]
@sp_seg1_s3_q3 = external global [64 x [1 x bfloat]]
@up_seg1_s3_q3 = external global [64 x [1 x bfloat]]
@gp_seg1_s3_q3 = external global [64 x [64 x bfloat]]
@g_seg1_s3_q3 = external global [64 x [64 x bfloat]]
@v_seg1_s3_q3 = external global [64 x [64 x bfloat]]
@q_seg1_s3_q3 = external global [64 x [64 x bfloat]]
@qk_seg1_s3_q3 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@r_local_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s2_q3 = external global [64 x [64 x bfloat]]
@r_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@s_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@sp_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@up_seg1_s2_q3 = external global [64 x [1 x bfloat]]
@gp_seg1_s2_q3 = external global [64 x [64 x bfloat]]
@g_seg1_s2_q3 = external global [64 x [64 x bfloat]]
@v_seg1_s2_q3 = external global [64 x [64 x bfloat]]
@q_seg1_s2_q3 = external global [64 x [64 x bfloat]]
@qk_seg1_s2_q3 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@r_local_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s1_q3 = external global [64 x [64 x bfloat]]
@r_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@s_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@sp_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@up_seg1_s1_q3 = external global [64 x [1 x bfloat]]
@gp_seg1_s1_q3 = external global [64 x [64 x bfloat]]
@g_seg1_s1_q3 = external global [64 x [64 x bfloat]]
@v_seg1_s1_q3 = external global [64 x [64 x bfloat]]
@q_seg1_s1_q3 = external global [64 x [64 x bfloat]]
@qk_seg1_s1_q3 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_q3 = external global [64 x [1 x bfloat]]
@r_local_seg1_q3 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_q3 = external global [64 x [1 x bfloat]]
@prev_up_seg1_q3 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_q3 = external global [64 x [1 x bfloat]]
@merged_up_seg1_q3 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_q3 = external global [64 x [64 x bfloat]]
@r_seg1_s0_q3 = external global [64 x [1 x bfloat]]
@s_seg1_s0_q3 = external global [64 x [1 x bfloat]]
@sp_seg1_s0_q3 = external global [64 x [1 x bfloat]]
@up_seg1_s0_q3 = external global [64 x [1 x bfloat]]
@gp_seg1_s0_q3 = external global [64 x [64 x bfloat]]
@g_seg1_s0_q3 = external global [64 x [64 x bfloat]]
@v_seg1_s0_q3 = external global [64 x [64 x bfloat]]
@q_seg1_s0_q3 = external global [64 x [64 x bfloat]]
@qk_seg1_s0_q3 = external global [64 x [64 x bfloat]]
@r_seg1_s3_q2 = external global [64 x [1 x bfloat]]
@s_seg1_s3_q2 = external global [64 x [1 x bfloat]]
@sp_seg1_s3_q2 = external global [64 x [1 x bfloat]]
@up_seg1_s3_q2 = external global [64 x [1 x bfloat]]
@gp_seg1_s3_q2 = external global [64 x [64 x bfloat]]
@g_seg1_s3_q2 = external global [64 x [64 x bfloat]]
@v_seg1_s3_q2 = external global [64 x [64 x bfloat]]
@q_seg1_s3_q2 = external global [64 x [64 x bfloat]]
@qk_seg1_s3_q2 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@r_local_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s2_q2 = external global [64 x [64 x bfloat]]
@r_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@s_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@sp_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@up_seg1_s2_q2 = external global [64 x [1 x bfloat]]
@gp_seg1_s2_q2 = external global [64 x [64 x bfloat]]
@g_seg1_s2_q2 = external global [64 x [64 x bfloat]]
@v_seg1_s2_q2 = external global [64 x [64 x bfloat]]
@q_seg1_s2_q2 = external global [64 x [64 x bfloat]]
@qk_seg1_s2_q2 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@r_local_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s1_q2 = external global [64 x [64 x bfloat]]
@r_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@s_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@sp_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@up_seg1_s1_q2 = external global [64 x [1 x bfloat]]
@gp_seg1_s1_q2 = external global [64 x [64 x bfloat]]
@g_seg1_s1_q2 = external global [64 x [64 x bfloat]]
@v_seg1_s1_q2 = external global [64 x [64 x bfloat]]
@q_seg1_s1_q2 = external global [64 x [64 x bfloat]]
@qk_seg1_s1_q2 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_q2 = external global [64 x [1 x bfloat]]
@r_local_seg1_q2 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_q2 = external global [64 x [1 x bfloat]]
@prev_up_seg1_q2 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_q2 = external global [64 x [1 x bfloat]]
@merged_up_seg1_q2 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_q2 = external global [64 x [64 x bfloat]]
@r_seg1_s0_q2 = external global [64 x [1 x bfloat]]
@s_seg1_s0_q2 = external global [64 x [1 x bfloat]]
@sp_seg1_s0_q2 = external global [64 x [1 x bfloat]]
@up_seg1_s0_q2 = external global [64 x [1 x bfloat]]
@gp_seg1_s0_q2 = external global [64 x [64 x bfloat]]
@g_seg1_s0_q2 = external global [64 x [64 x bfloat]]
@v_seg1_s0_q2 = external global [64 x [64 x bfloat]]
@q_seg1_s0_q2 = external global [64 x [64 x bfloat]]
@qk_seg1_s0_q2 = external global [64 x [64 x bfloat]]
@r_seg1_s3_q1 = external global [64 x [1 x bfloat]]
@s_seg1_s3_q1 = external global [64 x [1 x bfloat]]
@sp_seg1_s3_q1 = external global [64 x [1 x bfloat]]
@up_seg1_s3_q1 = external global [64 x [1 x bfloat]]
@gp_seg1_s3_q1 = external global [64 x [64 x bfloat]]
@g_seg1_s3_q1 = external global [64 x [64 x bfloat]]
@v_seg1_s3_q1 = external global [64 x [64 x bfloat]]
@q_seg1_s3_q1 = external global [64 x [64 x bfloat]]
@qk_seg1_s3_q1 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@r_local_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s2_q1 = external global [64 x [64 x bfloat]]
@r_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@s_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@sp_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@up_seg1_s2_q1 = external global [64 x [1 x bfloat]]
@gp_seg1_s2_q1 = external global [64 x [64 x bfloat]]
@g_seg1_s2_q1 = external global [64 x [64 x bfloat]]
@v_seg1_s2_q1 = external global [64 x [64 x bfloat]]
@q_seg1_s2_q1 = external global [64 x [64 x bfloat]]
@qk_seg1_s2_q1 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@r_local_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s1_q1 = external global [64 x [64 x bfloat]]
@r_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@s_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@sp_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@up_seg1_s1_q1 = external global [64 x [1 x bfloat]]
@gp_seg1_s1_q1 = external global [64 x [64 x bfloat]]
@g_seg1_s1_q1 = external global [64 x [64 x bfloat]]
@v_seg1_s1_q1 = external global [64 x [64 x bfloat]]
@q_seg1_s1_q1 = external global [64 x [64 x bfloat]]
@qk_seg1_s1_q1 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_q1 = external global [64 x [1 x bfloat]]
@r_local_seg1_q1 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_q1 = external global [64 x [1 x bfloat]]
@prev_up_seg1_q1 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_q1 = external global [64 x [1 x bfloat]]
@merged_up_seg1_q1 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_q1 = external global [64 x [64 x bfloat]]
@r_seg1_s0_q1 = external global [64 x [1 x bfloat]]
@s_seg1_s0_q1 = external global [64 x [1 x bfloat]]
@sp_seg1_s0_q1 = external global [64 x [1 x bfloat]]
@up_seg1_s0_q1 = external global [64 x [1 x bfloat]]
@gp_seg1_s0_q1 = external global [64 x [64 x bfloat]]
@g_seg1_s0_q1 = external global [64 x [64 x bfloat]]
@v_seg1_s0_q1 = external global [64 x [64 x bfloat]]
@q_seg1_s0_q1 = external global [64 x [64 x bfloat]]
@qk_seg1_s0_q1 = external global [64 x [64 x bfloat]]
@r_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@s_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@sp_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@up_seg1_s3_q0 = external global [64 x [1 x bfloat]]
@gp_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@g_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@v_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@q_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@qk_seg1_s3_q0 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@r_local_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s2_q0 = external global [64 x [64 x bfloat]]
@r_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@s_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@sp_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@up_seg1_s2_q0 = external global [64 x [1 x bfloat]]
@gp_seg1_s2_q0 = external global [64 x [64 x bfloat]]
@g_seg1_s2_q0 = external global [64 x [64 x bfloat]]
@v_seg1_s2_q0 = external global [64 x [64 x bfloat]]
@q_seg1_s2_q0 = external global [64 x [64 x bfloat]]
@qk_seg1_s2_q0 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@r_local_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@prev_up_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@merged_up_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_s1_q0 = external global [64 x [64 x bfloat]]
@r_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@s_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@sp_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@up_seg1_s1_q0 = external global [64 x [1 x bfloat]]
@gp_seg1_s1_q0 = external global [64 x [64 x bfloat]]
@g_seg1_s1_q0 = external global [64 x [64 x bfloat]]
@v_seg1_s1_q0 = external global [64 x [64 x bfloat]]
@q_seg1_s1_q0 = external global [64 x [64 x bfloat]]
@qk_seg1_s1_q0 = external global [64 x [64 x bfloat]]
@tmp_sp_seg1_q0 = external global [64 x [1 x bfloat]]
@r_local_seg1_q0 = external global [64 x [1 x bfloat]]
@r_cascade_seg1_q0 = external global [64 x [1 x bfloat]]
@prev_up_seg1_q0 = external global [64 x [1 x bfloat]]
@merged_sp_seg1_q0 = external global [64 x [1 x bfloat]]
@merged_up_seg1_q0 = external global [64 x [1 x bfloat]]
@merged_gp_seg1_q0 = external global [64 x [64 x bfloat]]
@r_seg1_s0_q0 = external global [64 x [1 x bfloat]]
@s_seg1_s0_q0 = external global [64 x [1 x bfloat]]
@sp_seg1_s0_q0 = external global [64 x [1 x bfloat]]
@up_seg1_s0_q0 = external global [64 x [1 x bfloat]]
@gp_seg1_s0_q0 = external global [64 x [64 x bfloat]]
@g_seg1_s0_q0 = external global [64 x [64 x bfloat]]
@v_seg1_s0_q0 = external global [64 x [64 x bfloat]]
@q_seg1_s0_q0 = external global [64 x [64 x bfloat]]
@qk_seg1_s0_q0 = external global [64 x [64 x bfloat]]
@r_seg0_s3_q3 = external global [64 x [1 x bfloat]]
@s_seg0_s3_q3 = external global [64 x [1 x bfloat]]
@sp_seg0_s3_q3 = external global [64 x [1 x bfloat]]
@up_seg0_s3_q3 = external global [64 x [1 x bfloat]]
@gp_seg0_s3_q3 = external global [64 x [64 x bfloat]]
@g_seg0_s3_q3 = external global [64 x [64 x bfloat]]
@v_seg0_s3_q3 = external global [64 x [64 x bfloat]]
@q_seg0_s3_q3 = external global [64 x [64 x bfloat]]
@qk_seg0_s3_q3 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@r_local_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s2_q3 = external global [64 x [64 x bfloat]]
@r_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@s_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@sp_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@up_seg0_s2_q3 = external global [64 x [1 x bfloat]]
@gp_seg0_s2_q3 = external global [64 x [64 x bfloat]]
@g_seg0_s2_q3 = external global [64 x [64 x bfloat]]
@v_seg0_s2_q3 = external global [64 x [64 x bfloat]]
@q_seg0_s2_q3 = external global [64 x [64 x bfloat]]
@qk_seg0_s2_q3 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@r_local_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s1_q3 = external global [64 x [64 x bfloat]]
@r_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@s_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@sp_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@up_seg0_s1_q3 = external global [64 x [1 x bfloat]]
@gp_seg0_s1_q3 = external global [64 x [64 x bfloat]]
@g_seg0_s1_q3 = external global [64 x [64 x bfloat]]
@v_seg0_s1_q3 = external global [64 x [64 x bfloat]]
@q_seg0_s1_q3 = external global [64 x [64 x bfloat]]
@qk_seg0_s1_q3 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_q3 = external global [64 x [1 x bfloat]]
@r_local_seg0_q3 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_q3 = external global [64 x [1 x bfloat]]
@prev_up_seg0_q3 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_q3 = external global [64 x [1 x bfloat]]
@merged_up_seg0_q3 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_q3 = external global [64 x [64 x bfloat]]
@r_seg0_s0_q3 = external global [64 x [1 x bfloat]]
@s_seg0_s0_q3 = external global [64 x [1 x bfloat]]
@sp_seg0_s0_q3 = external global [64 x [1 x bfloat]]
@up_seg0_s0_q3 = external global [64 x [1 x bfloat]]
@gp_seg0_s0_q3 = external global [64 x [64 x bfloat]]
@g_seg0_s0_q3 = external global [64 x [64 x bfloat]]
@v_seg0_s0_q3 = external global [64 x [64 x bfloat]]
@q_seg0_s0_q3 = external global [64 x [64 x bfloat]]
@qk_seg0_s0_q3 = external global [64 x [64 x bfloat]]
@r_seg0_s3_q2 = external global [64 x [1 x bfloat]]
@s_seg0_s3_q2 = external global [64 x [1 x bfloat]]
@sp_seg0_s3_q2 = external global [64 x [1 x bfloat]]
@up_seg0_s3_q2 = external global [64 x [1 x bfloat]]
@gp_seg0_s3_q2 = external global [64 x [64 x bfloat]]
@g_seg0_s3_q2 = external global [64 x [64 x bfloat]]
@v_seg0_s3_q2 = external global [64 x [64 x bfloat]]
@q_seg0_s3_q2 = external global [64 x [64 x bfloat]]
@qk_seg0_s3_q2 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@r_local_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s2_q2 = external global [64 x [64 x bfloat]]
@r_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@s_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@sp_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@up_seg0_s2_q2 = external global [64 x [1 x bfloat]]
@gp_seg0_s2_q2 = external global [64 x [64 x bfloat]]
@g_seg0_s2_q2 = external global [64 x [64 x bfloat]]
@v_seg0_s2_q2 = external global [64 x [64 x bfloat]]
@q_seg0_s2_q2 = external global [64 x [64 x bfloat]]
@qk_seg0_s2_q2 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@r_local_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s1_q2 = external global [64 x [64 x bfloat]]
@r_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@s_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@sp_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@up_seg0_s1_q2 = external global [64 x [1 x bfloat]]
@gp_seg0_s1_q2 = external global [64 x [64 x bfloat]]
@g_seg0_s1_q2 = external global [64 x [64 x bfloat]]
@v_seg0_s1_q2 = external global [64 x [64 x bfloat]]
@q_seg0_s1_q2 = external global [64 x [64 x bfloat]]
@qk_seg0_s1_q2 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_q2 = external global [64 x [1 x bfloat]]
@r_local_seg0_q2 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_q2 = external global [64 x [1 x bfloat]]
@prev_up_seg0_q2 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_q2 = external global [64 x [1 x bfloat]]
@merged_up_seg0_q2 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_q2 = external global [64 x [64 x bfloat]]
@r_seg0_s0_q2 = external global [64 x [1 x bfloat]]
@s_seg0_s0_q2 = external global [64 x [1 x bfloat]]
@sp_seg0_s0_q2 = external global [64 x [1 x bfloat]]
@up_seg0_s0_q2 = external global [64 x [1 x bfloat]]
@gp_seg0_s0_q2 = external global [64 x [64 x bfloat]]
@g_seg0_s0_q2 = external global [64 x [64 x bfloat]]
@v_seg0_s0_q2 = external global [64 x [64 x bfloat]]
@q_seg0_s0_q2 = external global [64 x [64 x bfloat]]
@qk_seg0_s0_q2 = external global [64 x [64 x bfloat]]
@r_seg0_s3_q1 = external global [64 x [1 x bfloat]]
@s_seg0_s3_q1 = external global [64 x [1 x bfloat]]
@sp_seg0_s3_q1 = external global [64 x [1 x bfloat]]
@up_seg0_s3_q1 = external global [64 x [1 x bfloat]]
@gp_seg0_s3_q1 = external global [64 x [64 x bfloat]]
@g_seg0_s3_q1 = external global [64 x [64 x bfloat]]
@v_seg0_s3_q1 = external global [64 x [64 x bfloat]]
@q_seg0_s3_q1 = external global [64 x [64 x bfloat]]
@qk_seg0_s3_q1 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@r_local_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s2_q1 = external global [64 x [64 x bfloat]]
@r_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@s_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@sp_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@up_seg0_s2_q1 = external global [64 x [1 x bfloat]]
@gp_seg0_s2_q1 = external global [64 x [64 x bfloat]]
@g_seg0_s2_q1 = external global [64 x [64 x bfloat]]
@v_seg0_s2_q1 = external global [64 x [64 x bfloat]]
@q_seg0_s2_q1 = external global [64 x [64 x bfloat]]
@qk_seg0_s2_q1 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@r_local_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s1_q1 = external global [64 x [64 x bfloat]]
@r_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@s_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@sp_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@up_seg0_s1_q1 = external global [64 x [1 x bfloat]]
@gp_seg0_s1_q1 = external global [64 x [64 x bfloat]]
@g_seg0_s1_q1 = external global [64 x [64 x bfloat]]
@v_seg0_s1_q1 = external global [64 x [64 x bfloat]]
@q_seg0_s1_q1 = external global [64 x [64 x bfloat]]
@qk_seg0_s1_q1 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_q1 = external global [64 x [1 x bfloat]]
@r_local_seg0_q1 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_q1 = external global [64 x [1 x bfloat]]
@prev_up_seg0_q1 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_q1 = external global [64 x [1 x bfloat]]
@merged_up_seg0_q1 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_q1 = external global [64 x [64 x bfloat]]
@r_seg0_s0_q1 = external global [64 x [1 x bfloat]]
@s_seg0_s0_q1 = external global [64 x [1 x bfloat]]
@sp_seg0_s0_q1 = external global [64 x [1 x bfloat]]
@up_seg0_s0_q1 = external global [64 x [1 x bfloat]]
@gp_seg0_s0_q1 = external global [64 x [64 x bfloat]]
@g_seg0_s0_q1 = external global [64 x [64 x bfloat]]
@v_seg0_s0_q1 = external global [64 x [64 x bfloat]]
@q_seg0_s0_q1 = external global [64 x [64 x bfloat]]
@qk_seg0_s0_q1 = external global [64 x [64 x bfloat]]
@r_seg0_s3_q0 = external global [64 x [1 x bfloat]]
@s_seg0_s3_q0 = external global [64 x [1 x bfloat]]
@sp_seg0_s3_q0 = external global [64 x [1 x bfloat]]
@up_seg0_s3_q0 = external global [64 x [1 x bfloat]]
@gp_seg0_s3_q0 = external global [64 x [64 x bfloat]]
@g_seg0_s3_q0 = external global [64 x [64 x bfloat]]
@v_seg0_s3_q0 = external global [64 x [64 x bfloat]]
@q_seg0_s3_q0 = external global [64 x [64 x bfloat]]
@qk_seg0_s3_q0 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@r_local_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s2_q0 = external global [64 x [64 x bfloat]]
@r_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@s_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@sp_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@up_seg0_s2_q0 = external global [64 x [1 x bfloat]]
@gp_seg0_s2_q0 = external global [64 x [64 x bfloat]]
@g_seg0_s2_q0 = external global [64 x [64 x bfloat]]
@v_seg0_s2_q0 = external global [64 x [64 x bfloat]]
@q_seg0_s2_q0 = external global [64 x [64 x bfloat]]
@qk_seg0_s2_q0 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@r_local_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@prev_up_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@merged_up_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_s1_q0 = external global [64 x [64 x bfloat]]
@r_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@s_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@sp_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@up_seg0_s1_q0 = external global [64 x [1 x bfloat]]
@gp_seg0_s1_q0 = external global [64 x [64 x bfloat]]
@g_seg0_s1_q0 = external global [64 x [64 x bfloat]]
@v_seg0_s1_q0 = external global [64 x [64 x bfloat]]
@q_seg0_s1_q0 = external global [64 x [64 x bfloat]]
@qk_seg0_s1_q0 = external global [64 x [64 x bfloat]]
@tmp_sp_seg0_q0 = external global [64 x [1 x bfloat]]
@r_local_seg0_q0 = external global [64 x [1 x bfloat]]
@r_cascade_seg0_q0 = external global [64 x [1 x bfloat]]
@prev_up_seg0_q0 = external global [64 x [1 x bfloat]]
@merged_sp_seg0_q0 = external global [64 x [1 x bfloat]]
@merged_up_seg0_q0 = external global [64 x [1 x bfloat]]
@merged_gp_seg0_q0 = external global [64 x [64 x bfloat]]
@r_seg0_s0_q0 = external global [64 x [1 x bfloat]]
@s_seg0_s0_q0 = external global [64 x [1 x bfloat]]
@sp_seg0_s0_q0 = external global [64 x [1 x bfloat]]
@up_seg0_s0_q0 = external global [64 x [1 x bfloat]]
@gp_seg0_s0_q0 = external global [64 x [64 x bfloat]]
@g_seg0_s0_q0 = external global [64 x [64 x bfloat]]
@v_seg0_s0_q0 = external global [64 x [64 x bfloat]]
@q_seg0_s0_q0 = external global [64 x [64 x bfloat]]
@qk_seg0_s0_q0 = external global [64 x [64 x bfloat]]
@out_l2_col7 = external global [64 x [64 x bfloat]]
@v_l2_col7 = external global [64 x [64 x bfloat]]
@qk_l2_col7 = external global [64 x [64 x bfloat]]
@out_l2_col6 = external global [64 x [64 x bfloat]]
@v_l2_col6 = external global [64 x [64 x bfloat]]
@qk_l2_col6 = external global [64 x [64 x bfloat]]
@out_l2_col5 = external global [64 x [64 x bfloat]]
@v_l2_col5 = external global [64 x [64 x bfloat]]
@qk_l2_col5 = external global [64 x [64 x bfloat]]
@out_l2_col4 = external global [64 x [64 x bfloat]]
@v_l2_col4 = external global [64 x [64 x bfloat]]
@qk_l2_col4 = external global [64 x [64 x bfloat]]
@out_l2_col3 = external global [64 x [64 x bfloat]]
@v_l2_col3 = external global [64 x [64 x bfloat]]
@qk_l2_col3 = external global [64 x [64 x bfloat]]
@out_l2_col2 = external global [64 x [64 x bfloat]]
@v_l2_col2 = external global [64 x [64 x bfloat]]
@qk_l2_col2 = external global [64 x [64 x bfloat]]
@out_l2_col1 = external global [64 x [64 x bfloat]]
@v_l2_col1 = external global [64 x [64 x bfloat]]
@qk_l2_col1 = external global [64 x [64 x bfloat]]
@out_l2_col0 = external global [64 x [64 x bfloat]]
@v_l2_col0 = external global [64 x [64 x bfloat]]
@qk_l2_col0 = external global [64 x [64 x bfloat]]

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

declare void @zero_fill_g_bf16_pythoc(ptr)

declare void @zero_fill_gp_bf16_pythoc(ptr)

declare void @zero_fill_sp_bf16_pythoc(ptr)

declare void @neg_inf_fill_up_bf16_pythoc(ptr)

declare void @copy_tile_pythoc(ptr, ptr)

declare void @matmul_a_b_bf16(ptr, ptr, ptr)

declare void @fused_softmax(ptr, ptr, ptr, ptr)

declare void @mul_r_gp(ptr, ptr)

declare void @matmul_g_b_bf16(ptr, ptr, ptr)

declare void @accum_sp_r_s(ptr, ptr, ptr)

declare void @vector_copy_32elems_pythoc(i32, ptr, ptr)

declare void @maximum_up_u_bf16(ptr, ptr)

declare void @exp_up_minus_u(ptr, ptr, ptr)

declare void @add_gp_g(ptr, ptr)

declare void @div_gp_sp(ptr, ptr)

define void @core_7_4() {
  br label %1

1:                                                ; preds = %59, %0
  %2 = phi i64 [ %60, %59 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %61

4:                                                ; preds = %1
  call void @zero_fill_gp_bf16_pythoc(ptr @gp_seg1_s2_q3)
  call void @zero_fill_sp_bf16_pythoc(ptr @sp_seg1_s2_q3)
  call void @neg_inf_fill_up_bf16_pythoc(ptr @up_seg1_s2_q3)
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @llvm.aie2p.release(i32 49, i32 1)
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @llvm.aie2p.release(i32 49, i32 1)
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @llvm.aie2p.release(i32 49, i32 1)
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @copy_tile_pythoc(ptr @qk_seg1_s2_q3, ptr @q_seg1_s2_q3)
  call void @llvm.aie2p.release(i32 49, i32 1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %9, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  call void @zero_fill_g_bf16_pythoc(ptr @g_seg1_s2_q3)
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @matmul_a_b_bf16(ptr @q_seg1_s2_q3, ptr @qk_seg1_s2_q3, ptr @g_seg1_s2_q3)
  call void @llvm.aie2p.release(i32 49, i32 1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @fused_softmax(ptr @g_seg1_s2_q3, ptr @up_seg1_s2_q3, ptr @s_seg1_s2_q3, ptr @r_seg1_s2_q3)
  call void @mul_r_gp(ptr @r_seg1_s2_q3, ptr @gp_seg1_s2_q3)
  call void @matmul_g_b_bf16(ptr @g_seg1_s2_q3, ptr @v_seg1_s2_q3, ptr @gp_seg1_s2_q3)
  call void @accum_sp_r_s(ptr @sp_seg1_s2_q3, ptr @r_seg1_s2_q3, ptr @s_seg1_s2_q3)
  call void @vector_copy_32elems_pythoc(i32 0, ptr @s_seg1_s2_q3, ptr @sp_seg1_s2_q3)
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
  %16 = getelementptr bfloat, ptr @merged_gp_seg1_s2_q3, i64 %11
  store <32 x bfloat> %15, ptr %16
  %17 = add i64 %11, 32
  br label %10

18:                                               ; preds = %21, %10
  %19 = phi i64 [ %25, %21 ], [ 0, %10 ]
  %20 = icmp slt i64 %19, 64
  br i1 %20, label %21, label %26

21:                                               ; preds = %18
  %22 = call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %23 = bitcast <16 x i32> %22 to <32 x bfloat>
  %24 = getelementptr bfloat, ptr @merged_up_seg1_s2_q3, i64 %19
  store <32 x bfloat> %23, ptr %24
  %25 = add i64 %19, 32
  br label %18

26:                                               ; preds = %29, %18
  %27 = phi i64 [ %33, %29 ], [ 0, %18 ]
  %28 = icmp slt i64 %27, 64
  br i1 %28, label %29, label %34

29:                                               ; preds = %26
  %30 = call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %31 = bitcast <16 x i32> %30 to <32 x bfloat>
  %32 = getelementptr bfloat, ptr @merged_sp_seg1_s2_q3, i64 %27
  store <32 x bfloat> %31, ptr %32
  %33 = add i64 %27, 32
  br label %26

34:                                               ; preds = %26
  call void @vector_copy_32elems_pythoc(i32 0, ptr @up_seg1_s2_q3, ptr @prev_up_seg1_s2_q3)
  call void @maximum_up_u_bf16(ptr @merged_up_seg1_s2_q3, ptr @up_seg1_s2_q3)
  call void @exp_up_minus_u(ptr @merged_up_seg1_s2_q3, ptr @up_seg1_s2_q3, ptr @r_cascade_seg1_s2_q3)
  call void @exp_up_minus_u(ptr @prev_up_seg1_s2_q3, ptr @up_seg1_s2_q3, ptr @r_local_seg1_s2_q3)
  call void @mul_r_gp(ptr @r_cascade_seg1_s2_q3, ptr @merged_gp_seg1_s2_q3)
  call void @mul_r_gp(ptr @r_local_seg1_s2_q3, ptr @gp_seg1_s2_q3)
  call void @add_gp_g(ptr @gp_seg1_s2_q3, ptr @merged_gp_seg1_s2_q3)
  call void @zero_fill_sp_bf16_pythoc(ptr @tmp_sp_seg1_s2_q3)
  call void @accum_sp_r_s(ptr @merged_sp_seg1_s2_q3, ptr @r_cascade_seg1_s2_q3, ptr @tmp_sp_seg1_s2_q3)
  call void @accum_sp_r_s(ptr @sp_seg1_s2_q3, ptr @r_local_seg1_s2_q3, ptr @tmp_sp_seg1_s2_q3)
  call void @vector_copy_32elems_pythoc(i32 0, ptr @tmp_sp_seg1_s2_q3, ptr @merged_sp_seg1_s2_q3)
  br label %35

35:                                               ; preds = %38, %34
  %36 = phi i64 [ %42, %38 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, 4096
  br i1 %37, label %38, label %43

38:                                               ; preds = %35
  %39 = getelementptr bfloat, ptr @merged_gp_seg1_s2_q3, i64 %36
  %40 = load <32 x bfloat>, ptr %39
  %41 = bitcast <32 x bfloat> %40 to <16 x i32>
  call void @llvm.aie2p.mcd.write.vec(<16 x i32> %41, i32 1)
  %42 = add i64 %36, 32
  br label %35

43:                                               ; preds = %46, %35
  %44 = phi i64 [ %50, %46 ], [ 0, %35 ]
  %45 = icmp slt i64 %44, 64
  br i1 %45, label %46, label %51

46:                                               ; preds = %43
  %47 = getelementptr bfloat, ptr @up_seg1_s2_q3, i64 %44
  %48 = load <32 x bfloat>, ptr %47
  %49 = bitcast <32 x bfloat> %48 to <16 x i32>
  call void @llvm.aie2p.mcd.write.vec(<16 x i32> %49, i32 1)
  %50 = add i64 %44, 32
  br label %43

51:                                               ; preds = %54, %43
  %52 = phi i64 [ %58, %54 ], [ 0, %43 ]
  %53 = icmp slt i64 %52, 64
  br i1 %53, label %54, label %59

54:                                               ; preds = %51
  %55 = getelementptr bfloat, ptr @merged_sp_seg1_s2_q3, i64 %52
  %56 = load <32 x bfloat>, ptr %55
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
