module {
  aie.device(npu2) @flash_attention_direct {
    %Q = aie.external_buffer {sym_name = "Q"} : memref<64x64xbf16>
    %K = aie.external_buffer {sym_name = "K"} : memref<64x12288xbf16>
    %V = aie.external_buffer {sym_name = "V"} : memref<12288x64xbf16>
    %Out = aie.external_buffer {sym_name = "Out"} : memref<64x64xbf16>
    func.func private @matmul_a_b_bf16(memref<64x64xbf16>, memref<64x96xbf16>, memref<6144xbf16>) attributes {link_with = "attn.o"}
    func.func private @matmul_g_b_bf16(memref<6144xbf16>, memref<64x96xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @zero_fill_gp_bf16(memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @zero_fill_sp_bf16(memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @zero_fill_g_bf16(memref<6144xbf16>) attributes {link_with = "attn.o"}
    func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @max_g_bf16(memref<6144xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @maximum_up_u_bf16(memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @exp_g_minus_u(memref<64x1xbf16>, memref<6144xbf16>) attributes {link_with = "attn.o"}
    func.func private @exp_up_minus_u(memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @mul_r_gp(memref<64x1xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @sum_g(memref<6144xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @accum_sp_r_s(memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @vector_copy_32elems(i32, memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @vector_copy_32x96elems(i32, memref<6144xbf16>, memref<6144xbf16>) attributes {link_with = "attn.o"}
    func.func private @vector_accum_32x64elems(memref<64x64xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @div_gp_sp(memref<64x1xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @add_gp_g(memref<64x64xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_4_1 = aie.tile(4, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_5_1 = aie.tile(5, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_6_1 = aie.tile(6, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_7_1 = aie.tile(7, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    aie.shim_dma_allocation @air_L2ToL3Chan1(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan1_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan1_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan1_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan1_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan3_0(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan3_1(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan3_2(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_L3ToL2Chan3_3(%shim_noc_tile_0_0, MM2S, 1)
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_0 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_1_1 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_2 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_3 = aie.lock(%mem_tile_0_1, 5) {init = 1 : i32}
    %lock_0_1_4 = aie.lock(%mem_tile_0_1, 4) {init = 0 : i32}
    %q_l2_stage0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "q_l2_stage0"} : memref<64x64xbf16> 
    %k_l2_stage0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "k_l2_stage0"} : memref<64x96xbf16> 
    %q_l2_stage1 = aie.buffer(%mem_tile_1_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "q_l2_stage1"} : memref<64x64xbf16> 
    %k_l2_stage1 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "k_l2_stage1"} : memref<64x96xbf16> 
    %lock_1_1 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_1_1_5 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_6 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_7 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %q_l2_stage2 = aie.buffer(%mem_tile_2_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "q_l2_stage2"} : memref<64x64xbf16> 
    %k_l2_stage2 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "k_l2_stage2"} : memref<64x96xbf16> 
    %lock_2_1 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_2_1_8 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_9 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_10 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %q_l2_stage3 = aie.buffer(%mem_tile_3_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "q_l2_stage3"} : memref<64x64xbf16> 
    %k_l2_stage3 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "k_l2_stage3"} : memref<64x96xbf16> 
    %lock_3_1 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_3_1_11 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_12 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_13 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %v_l2_stage0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "v_l2_stage0"} : memref<96x64xbf16> 
    %lock_5_1 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_5_1_14 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %v_l2_stage1 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "v_l2_stage1"} : memref<96x64xbf16> 
    %lock_6_1 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_6_1_15 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %v_l2_stage2 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "v_l2_stage2"} : memref<96x64xbf16> 
    %lock_7_1 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_7_1_16 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %v_l2_stage3 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "v_l2_stage3"} : memref<96x64xbf16> 
    %lock_4_1 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_4_1_17 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %out_l2 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "out_l2"} : memref<64x64xbf16> 
    %q_stage0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "q_stage0"} : memref<64x64xbf16> 
    %k_stage0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "k_stage0"} : memref<64x96xbf16> 
    %g_stage0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "g_stage0"} : memref<6144xbf16> 
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_18 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_19 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_20 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_21 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_2_22 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %gp_in_stage0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "gp_in_stage0"} : memref<64x64xbf16> 
    %g_in_stage0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "g_in_stage0"} : memref<6144xbf16> 
    %v_stage0 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "v_stage0"} : memref<64x96xbf16> 
    %lock_1_2 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_1_2_23 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_24 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_25 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_26 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_27 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %softmax_gp_stage0 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "softmax_gp_stage0"} : memref<64x64xbf16> 
    %softmax_sp_stage0 = aie.buffer(%tile_2_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "softmax_sp_stage0"} : memref<64x1xbf16> 
    %softmax_up_stage0 = aie.buffer(%tile_2_2) {address = 13312 : i32, mem_bank = 0 : i32, sym_name = "softmax_up_stage0"} : memref<64x1xbf16> 
    %softmax_u_stage0 = aie.buffer(%tile_2_2) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "softmax_u_stage0"} : memref<64x1xbf16> 
    %softmax_r_stage0 = aie.buffer(%tile_2_2) {address = 57472 : i32, mem_bank = 3 : i32, sym_name = "softmax_r_stage0"} : memref<64x1xbf16> 
    %softmax_s_stage0 = aie.buffer(%tile_2_2) {address = 13440 : i32, mem_bank = 0 : i32, sym_name = "softmax_s_stage0"} : memref<64x1xbf16> 
    %softmax_g_stage0 = aie.buffer(%tile_2_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "softmax_g_stage0"} : memref<6144xbf16> 
    %softmax_g_copy_stage0 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "softmax_g_copy_stage0"} : memref<6144xbf16> 
    %softmax_gv_stage0 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "softmax_gv_stage0"} : memref<64x64xbf16> 
    %softmax_out_stage0 = aie.buffer(%tile_2_2) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "softmax_out_stage0"} : memref<64x64xbf16> 
    %cascade_up_stage0 = aie.buffer(%tile_2_2) {address = 28800 : i32, mem_bank = 1 : i32, sym_name = "cascade_up_stage0"} : memref<64x1xbf16> 
    %cascade_sp_stage0 = aie.buffer(%tile_2_2) {address = 57600 : i32, mem_bank = 3 : i32, sym_name = "cascade_sp_stage0"} : memref<64x1xbf16> 
    %prev_up_stage0 = aie.buffer(%tile_2_2) {address = 13568 : i32, mem_bank = 0 : i32, sym_name = "prev_up_stage0"} : memref<64x1xbf16> 
    %r_from_cascade_stage0 = aie.buffer(%tile_2_2) {address = 28928 : i32, mem_bank = 1 : i32, sym_name = "r_from_cascade_stage0"} : memref<64x1xbf16> 
    %r_from_local_stage0 = aie.buffer(%tile_2_2) {address = 57728 : i32, mem_bank = 3 : i32, sym_name = "r_from_local_stage0"} : memref<64x1xbf16> 
    %tmp_sp_stage0 = aie.buffer(%tile_2_2) {address = 13696 : i32, mem_bank = 0 : i32, sym_name = "tmp_sp_stage0"} : memref<64x1xbf16> 
    %lock_2_2 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_2_2_28 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_29 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_30 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_31 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_32 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_33 = aie.lock(%tile_2_2, 7) {init = 1 : i32}
    %lock_2_2_34 = aie.lock(%tile_2_2, 6) {init = 0 : i32}
    %q_stage1 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "q_stage1"} : memref<64x64xbf16> 
    %k_stage1 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "k_stage1"} : memref<64x96xbf16> 
    %g_stage1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "g_stage1"} : memref<6144xbf16> 
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_35 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_36 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_37 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_38 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_3_39 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %gp_in_stage1 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "gp_in_stage1"} : memref<64x64xbf16> 
    %g_in_stage1 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "g_in_stage1"} : memref<6144xbf16> 
    %v_stage1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "v_stage1"} : memref<64x96xbf16> 
    %lock_1_3 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_1_3_40 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_41 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_42 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_43 = aie.lock(%tile_1_3, 5) {init = 1 : i32}
    %lock_1_3_44 = aie.lock(%tile_1_3, 4) {init = 0 : i32}
    %softmax_gp_stage1 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "softmax_gp_stage1"} : memref<64x64xbf16> 
    %softmax_sp_stage1 = aie.buffer(%tile_2_3) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "softmax_sp_stage1"} : memref<64x1xbf16> 
    %softmax_up_stage1 = aie.buffer(%tile_2_3) {address = 13312 : i32, mem_bank = 0 : i32, sym_name = "softmax_up_stage1"} : memref<64x1xbf16> 
    %softmax_u_stage1 = aie.buffer(%tile_2_3) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "softmax_u_stage1"} : memref<64x1xbf16> 
    %softmax_r_stage1 = aie.buffer(%tile_2_3) {address = 57472 : i32, mem_bank = 3 : i32, sym_name = "softmax_r_stage1"} : memref<64x1xbf16> 
    %softmax_s_stage1 = aie.buffer(%tile_2_3) {address = 13440 : i32, mem_bank = 0 : i32, sym_name = "softmax_s_stage1"} : memref<64x1xbf16> 
    %softmax_g_stage1 = aie.buffer(%tile_2_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "softmax_g_stage1"} : memref<6144xbf16> 
    %softmax_g_copy_stage1 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "softmax_g_copy_stage1"} : memref<6144xbf16> 
    %softmax_gv_stage1 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "softmax_gv_stage1"} : memref<64x64xbf16> 
    %cascade_gp_stage1 = aie.buffer(%tile_2_3) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "cascade_gp_stage1"} : memref<64x64xbf16> 
    %cascade_up_stage1 = aie.buffer(%tile_2_3) {address = 28800 : i32, mem_bank = 1 : i32, sym_name = "cascade_up_stage1"} : memref<64x1xbf16> 
    %cascade_sp_stage1 = aie.buffer(%tile_2_3) {address = 57600 : i32, mem_bank = 3 : i32, sym_name = "cascade_sp_stage1"} : memref<64x1xbf16> 
    %prev_up_stage1 = aie.buffer(%tile_2_3) {address = 13568 : i32, mem_bank = 0 : i32, sym_name = "prev_up_stage1"} : memref<64x1xbf16> 
    %r_from_cascade_stage1 = aie.buffer(%tile_2_3) {address = 28928 : i32, mem_bank = 1 : i32, sym_name = "r_from_cascade_stage1"} : memref<64x1xbf16> 
    %r_from_local_stage1 = aie.buffer(%tile_2_3) {address = 57728 : i32, mem_bank = 3 : i32, sym_name = "r_from_local_stage1"} : memref<64x1xbf16> 
    %tmp_sp_stage1 = aie.buffer(%tile_2_3) {address = 13696 : i32, mem_bank = 0 : i32, sym_name = "tmp_sp_stage1"} : memref<64x1xbf16> 
    %lock_2_3 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_2_3_45 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 5) {init = 1 : i32}
    %lock_2_3_49 = aie.lock(%tile_2_3, 4) {init = 0 : i32}
    %q_stage2 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "q_stage2"} : memref<64x64xbf16> 
    %k_stage2 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "k_stage2"} : memref<64x96xbf16> 
    %g_stage2 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "g_stage2"} : memref<6144xbf16> 
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_50 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_51 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_52 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_53 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_0_4_54 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %gp_in_stage2 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "gp_in_stage2"} : memref<64x64xbf16> 
    %g_in_stage2 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "g_in_stage2"} : memref<6144xbf16> 
    %v_stage2 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "v_stage2"} : memref<64x96xbf16> 
    %lock_1_4 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_1_4_55 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_56 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_57 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_58 = aie.lock(%tile_1_4, 5) {init = 1 : i32}
    %lock_1_4_59 = aie.lock(%tile_1_4, 4) {init = 0 : i32}
    %softmax_gp_stage2 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "softmax_gp_stage2"} : memref<64x64xbf16> 
    %softmax_sp_stage2 = aie.buffer(%tile_2_4) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "softmax_sp_stage2"} : memref<64x1xbf16> 
    %softmax_up_stage2 = aie.buffer(%tile_2_4) {address = 13312 : i32, mem_bank = 0 : i32, sym_name = "softmax_up_stage2"} : memref<64x1xbf16> 
    %softmax_u_stage2 = aie.buffer(%tile_2_4) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "softmax_u_stage2"} : memref<64x1xbf16> 
    %softmax_r_stage2 = aie.buffer(%tile_2_4) {address = 57472 : i32, mem_bank = 3 : i32, sym_name = "softmax_r_stage2"} : memref<64x1xbf16> 
    %softmax_s_stage2 = aie.buffer(%tile_2_4) {address = 13440 : i32, mem_bank = 0 : i32, sym_name = "softmax_s_stage2"} : memref<64x1xbf16> 
    %softmax_g_stage2 = aie.buffer(%tile_2_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "softmax_g_stage2"} : memref<6144xbf16> 
    %softmax_g_copy_stage2 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "softmax_g_copy_stage2"} : memref<6144xbf16> 
    %softmax_gv_stage2 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "softmax_gv_stage2"} : memref<64x64xbf16> 
    %cascade_gp_stage2 = aie.buffer(%tile_2_4) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "cascade_gp_stage2"} : memref<64x64xbf16> 
    %cascade_up_stage2 = aie.buffer(%tile_2_4) {address = 28800 : i32, mem_bank = 1 : i32, sym_name = "cascade_up_stage2"} : memref<64x1xbf16> 
    %cascade_sp_stage2 = aie.buffer(%tile_2_4) {address = 57600 : i32, mem_bank = 3 : i32, sym_name = "cascade_sp_stage2"} : memref<64x1xbf16> 
    %prev_up_stage2 = aie.buffer(%tile_2_4) {address = 13568 : i32, mem_bank = 0 : i32, sym_name = "prev_up_stage2"} : memref<64x1xbf16> 
    %r_from_cascade_stage2 = aie.buffer(%tile_2_4) {address = 28928 : i32, mem_bank = 1 : i32, sym_name = "r_from_cascade_stage2"} : memref<64x1xbf16> 
    %r_from_local_stage2 = aie.buffer(%tile_2_4) {address = 57728 : i32, mem_bank = 3 : i32, sym_name = "r_from_local_stage2"} : memref<64x1xbf16> 
    %tmp_sp_stage2 = aie.buffer(%tile_2_4) {address = 13696 : i32, mem_bank = 0 : i32, sym_name = "tmp_sp_stage2"} : memref<64x1xbf16> 
    %lock_2_4 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_2_4_60 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_61 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_62 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_63 = aie.lock(%tile_2_4, 5) {init = 1 : i32}
    %lock_2_4_64 = aie.lock(%tile_2_4, 4) {init = 0 : i32}
    %q_stage3 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "q_stage3"} : memref<64x64xbf16> 
    %k_stage3 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "k_stage3"} : memref<64x96xbf16> 
    %g_stage3 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "g_stage3"} : memref<6144xbf16> 
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_65 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_66 = aie.lock(%tile_0_5, 5) {init = 1 : i32}
    %lock_0_5_67 = aie.lock(%tile_0_5, 4) {init = 0 : i32}
    %lock_0_5_68 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_0_5_69 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %gp_in_stage3 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "gp_in_stage3"} : memref<64x64xbf16> 
    %g_in_stage3 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "g_in_stage3"} : memref<6144xbf16> 
    %v_stage3 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "v_stage3"} : memref<64x96xbf16> 
    %lock_1_5 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_1_5_70 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_71 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_72 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_73 = aie.lock(%tile_1_5, 5) {init = 1 : i32}
    %lock_1_5_74 = aie.lock(%tile_1_5, 4) {init = 0 : i32}
    %softmax_gp_stage3 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "softmax_gp_stage3"} : memref<64x64xbf16> 
    %softmax_sp_stage3 = aie.buffer(%tile_2_5) {address = 13312 : i32, mem_bank = 0 : i32, sym_name = "softmax_sp_stage3"} : memref<64x1xbf16> 
    %softmax_up_stage3 = aie.buffer(%tile_2_5) {address = 28672 : i32, mem_bank = 1 : i32, sym_name = "softmax_up_stage3"} : memref<64x1xbf16> 
    %softmax_u_stage3 = aie.buffer(%tile_2_5) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "softmax_u_stage3"} : memref<64x1xbf16> 
    %softmax_r_stage3 = aie.buffer(%tile_2_5) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "softmax_r_stage3"} : memref<64x1xbf16> 
    %softmax_s_stage3 = aie.buffer(%tile_2_5) {address = 13440 : i32, mem_bank = 0 : i32, sym_name = "softmax_s_stage3"} : memref<64x1xbf16> 
    %softmax_g_stage3 = aie.buffer(%tile_2_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "softmax_g_stage3"} : memref<6144xbf16> 
    %softmax_g_copy_stage3 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "softmax_g_copy_stage3"} : memref<6144xbf16> 
    %softmax_gv_stage3 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "softmax_gv_stage3"} : memref<64x64xbf16> 
    %lock_2_5 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_2_5_75 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_76 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_77 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_78 = aie.lock(%tile_2_5, 5) {init = 1 : i32}
    %lock_2_5_79 = aie.lock(%tile_2_5, 4) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_stage0 : memref<6144xbf16>, 0, 6144, [<size = 64, stride = 8>, <size = 12, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_stage0 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_2_18, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_2_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_stage0 : memref<64x96xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_0_2.elf", link_files = ["attn.o"]}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_stage1 : memref<6144xbf16>, 0, 6144, [<size = 64, stride = 8>, <size = 12, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_3_39, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_stage1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_3_35, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_3_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_stage1 : memref<64x96xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_3_37, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_0_3.elf", link_files = ["attn.o"]}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_stage2 : memref<6144xbf16>, 0, 6144, [<size = 64, stride = 8>, <size = 12, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_stage2 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_4_50, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_4_51, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_stage2 : memref<64x96xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_0_4.elf", link_files = ["attn.o"]}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_stage3 : memref<6144xbf16>, 0, 6144, [<size = 64, stride = 8>, <size = 12, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_stage3 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_stage3 : memref<64x96xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_5_67, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_0_5.elf", link_files = ["attn.o"]}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%gp_in_stage0 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_1_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_stage0 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_in_stage0 : memref<6144xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_1_2_27, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_1_2.elf", link_files = ["attn.o"]}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%gp_in_stage1 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_1_3_40, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_stage1 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_1_3_42, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_in_stage1 : memref<6144xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_1_3.elf", link_files = ["attn.o"]}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%gp_in_stage2 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_1_4_55, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_stage2 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_1_4_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_in_stage2 : memref<6144xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_1_4_59, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_1_4.elf", link_files = ["attn.o"]}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%gp_in_stage3 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_1_5_70, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_stage3 : memref<64x96xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_1_5_72, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_1_5_73, AcquireGreaterEqual, 1)
      aie.dma_bd(%g_in_stage3 : memref<6144xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_1_5_74, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_1_5 = aie.core(%tile_1_5) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_1_5.elf", link_files = ["attn.o"]}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_out_stage0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_2_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_copy_stage0 : memref<6144xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_stage0 : memref<6144xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_2_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_gv_stage0 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %core_2_2 = aie.core(%tile_2_2) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_2_2.elf", link_files = ["attn.o"]}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_copy_stage1 : memref<6144xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_2_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_stage1 : memref<6144xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_gv_stage1 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_2_3_49, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_2_3.elf", link_files = ["attn.o"]}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_copy_stage2 : memref<6144xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_4_61, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_stage2 : memref<6144xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_2_4_62, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_2_4_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_gv_stage2 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_2_4_64, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_2_4 = aie.core(%tile_2_4) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_2_4.elf", link_files = ["attn.o"]}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_copy_stage3 : memref<6144xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_2_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_5_76, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_g_stage3 : memref<6144xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_2_5_77, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_2_5_78, AcquireGreaterEqual, 1)
      aie.dma_bd(%softmax_gv_stage3 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_2_5_79, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_2_5 = aie.core(%tile_2_5) {
      aie.end
    } {elf_file = "/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/flash_attention_direct_core_2_5.elf", link_files = ["attn.o"]}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage0 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage0 : memref<64x96xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage3 : memref<96x64xbf16>, 0, 6144, [<size = 8, stride = 8>, <size = 96, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_1_3, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32}
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb9
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb11, repeat_count = 31)
    ^bb9:  // 3 preds: ^bb7, ^bb10, ^bb11
      aie.end
    ^bb10:  // pred: ^bb8
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage0 : memref<64x96xbf16>, 0, 6144) {bd_id = 3 : i32}
      aie.use_lock(%lock_0_1_2, Release, 1)
      aie.next_bd ^bb9
    ^bb11:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 1, ^bb12, ^bb9)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage3 : memref<96x64xbf16>, 0, 6144) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%lock_0_1_4, Release, 1)
      aie.next_bd ^bb12
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage1 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_1_1_5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage1 : memref<64x96xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%lock_1_1_7, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 0, ^bb5, ^bb6)
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb7
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb7, repeat_count = 31)
    ^bb7:  // 3 preds: ^bb5, ^bb6, ^bb8
      aie.end
    ^bb8:  // pred: ^bb6
      aie.use_lock(%lock_1_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage1 : memref<64x96xbf16>, 0, 6144) {bd_id = 2 : i32}
      aie.use_lock(%lock_1_1_6, Release, 1)
      aie.next_bd ^bb7
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage2 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage2 : memref<64x96xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%lock_2_1_10, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 0, ^bb5, ^bb6)
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage2 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb7
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb7, repeat_count = 31)
    ^bb7:  // 3 preds: ^bb5, ^bb6, ^bb8
      aie.end
    ^bb8:  // pred: ^bb6
      aie.use_lock(%lock_2_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage2 : memref<64x96xbf16>, 0, 6144) {bd_id = 2 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb7
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage3 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_3_1_11, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage3 : memref<64x96xbf16>, 0, 6144, [<size = 12, stride = 8>, <size = 64, stride = 96>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 0, ^bb5, ^bb6)
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%q_l2_stage3 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb7
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb8, ^bb7, repeat_count = 31)
    ^bb7:  // 3 preds: ^bb5, ^bb6, ^bb8
      aie.end
    ^bb8:  // pred: ^bb6
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_l2_stage3 : memref<64x96xbf16>, 0, 6144) {bd_id = 2 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb7
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage0 : memref<96x64xbf16>, 0, 6144, [<size = 8, stride = 8>, <size = 96, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_5_1_14, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_5_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage0 : memref<96x64xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage1 : memref<96x64xbf16>, 0, 6144, [<size = 8, stride = 8>, <size = 96, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_6_1_15, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_6_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage1 : memref<96x64xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage2 : memref<96x64xbf16>, 0, 6144, [<size = 8, stride = 8>, <size = 96, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_7_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_7_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_stage2 : memref<96x64xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_4_1_17, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_4_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.runtime_sequence @attention_bf16(%arg0: memref<64x64xbf16>, %arg1: memref<64x12288xbf16>, %arg2: memref<12288x64xbf16>, %arg3: memref<64x12288xbf16>, %arg4: memref<64x64xbf16>) {
      %0 = aiex.dma_configure_task_for @air_L3ToL2Chan1_0 {
        aie.dma_bd(%arg0 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_L3ToL2Chan1_1 {
        aie.dma_bd(%arg0 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_L3ToL2Chan1_2 {
        aie.dma_bd(%arg0 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_L3ToL2Chan1_3 {
        aie.dma_bd(%arg0 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_L3ToL2Chan1_0 {
        aie.dma_bd(%arg1 : memref<64x12288xbf16>, 0, 196608, [<size = 32, stride = 384>, <size = 64, stride = 12288>, <size = 96, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_L3ToL2Chan1_1 {
        aie.dma_bd(%arg1 : memref<64x12288xbf16>, 96, 196608, [<size = 32, stride = 384>, <size = 64, stride = 12288>, <size = 96, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_L3ToL2Chan1_2 {
        aie.dma_bd(%arg1 : memref<64x12288xbf16>, 192, 196608, [<size = 32, stride = 384>, <size = 64, stride = 12288>, <size = 96, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_L3ToL2Chan1_3 {
        aie.dma_bd(%arg1 : memref<64x12288xbf16>, 288, 196608, [<size = 32, stride = 384>, <size = 64, stride = 12288>, <size = 96, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_L3ToL2Chan3_0 {
        aie.dma_bd(%arg2 : memref<12288x64xbf16>, 0, 196608, [<size = 32, stride = 24576>, <size = 96, stride = 64>, <size = 64, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_L3ToL2Chan3_1 {
        aie.dma_bd(%arg2 : memref<12288x64xbf16>, 6144, 196608, [<size = 32, stride = 24576>, <size = 96, stride = 64>, <size = 64, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_L3ToL2Chan3_2 {
        aie.dma_bd(%arg2 : memref<12288x64xbf16>, 12288, 196608, [<size = 32, stride = 24576>, <size = 96, stride = 64>, <size = 64, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_L3ToL2Chan3_3 {
        aie.dma_bd(%arg2 : memref<12288x64xbf16>, 18432, 196608, [<size = 32, stride = 24576>, <size = 96, stride = 64>, <size = 64, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_L2ToL3Chan1 {
        aie.dma_bd(%arg4 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%3)
      aiex.dma_await_task(%2)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%6)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%8)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%9)
      aiex.npu.load_pdi {device_ref = @flash_attention_direct}
    }
    aie.configure_cascade(%tile_2_2, North, South)
    aie.configure_cascade(%tile_2_3, North, South)
    aie.configure_cascade(%tile_2_4, North, South)
    aie.configure_cascade(%tile_2_5, North, South)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<South : 3, North : 1>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, North : 1>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
    }
    %switchbox_5_0 = aie.switchbox(%shim_noc_tile_5_0) {
      aie.connect<South : 3, North : 1>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_5_0 = aie.shim_mux(%shim_noc_tile_5_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_5_1 = aie.switchbox(%mem_tile_5_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
    }
    %switchbox_6_0 = aie.switchbox(%shim_noc_tile_6_0) {
      aie.connect<South : 3, North : 1>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_6_0 = aie.shim_mux(%shim_noc_tile_6_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_6_1 = aie.switchbox(%mem_tile_6_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
    }
    %switchbox_7_0 = aie.switchbox(%shim_noc_tile_7_0) {
      aie.connect<South : 3, North : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_7_0 = aie.shim_mux(%shim_noc_tile_7_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_7_1 = aie.switchbox(%mem_tile_7_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, North : 1>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<East : 3, North : 5>
      aie.connect<East : 1, North : 4>
      aie.connect<South : 0, North : 3>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<East : 2, DMA : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<South : 3, North : 3>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 3>
      aie.connect<South : 5, North : 1>
      aie.connect<East : 3, West : 1>
      aie.connect<East : 2, DMA : 0>
      aie.connect<East : 0, North : 5>
      aie.connect<West : 0, East : 1>
      aie.connect<East : 1, DMA : 1>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 1, West : 2>
      aie.connect<East : 2, North : 3>
      aie.connect<East : 3, North : 5>
      aie.connect<East : 0, DMA : 0>
      aie.connect<South : 5, North : 4>
      aie.connect<West : 0, East : 1>
      aie.connect<East : 1, DMA : 1>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<East : 0, North : 3>
      aie.connect<South : 3, North : 2>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 1, West : 3>
      aie.connect<South : 5, North : 1>
      aie.connect<East : 0, West : 2>
      aie.connect<East : 2, West : 0>
      aie.connect<DMA : 0, East : 0>
      aie.connect<West : 1, DMA : 0>
      aie.connect<DMA : 1, West : 1>
      aie.connect<West : 0, DMA : 1>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 3, West : 3>
      aie.connect<East : 3, North : 0>
      aie.connect<South : 5, West : 0>
      aie.connect<South : 4, DMA : 0>
      aie.connect<West : 0, East : 1>
      aie.connect<East : 0, DMA : 1>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 1, West : 2>
      aie.connect<East : 2, West : 3>
      aie.connect<East : 1, West : 0>
      aie.connect<West : 1, DMA : 0>
      aie.connect<DMA : 0, West : 1>
      aie.connect<West : 0, DMA : 1>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<East : 3, DMA : 0>
      aie.connect<South : 3, DMA : 1>
      aie.connect<South : 2, East : 3>
      aie.connect<DMA : 0, East : 2>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 0, West : 3>
      aie.connect<West : 3, DMA : 0>
      aie.connect<West : 2, East : 2>
      aie.connect<East : 0, DMA : 1>
      aie.connect<DMA : 0, East : 3>
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<East : 1, West : 3>
      aie.connect<West : 1, DMA : 0>
      aie.connect<DMA : 0, West : 0>
      aie.connect<West : 0, DMA : 1>
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 1, North : 3>
      aie.connect<South : 5, North : 1>
      aie.connect<East : 3, West : 0>
      aie.connect<East : 0, North : 5>
      aie.connect<East : 2, West : 2>
      aie.connect<West : 0, East : 2>
    }
    %tile_3_3 = aie.tile(3, 3)
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 1, West : 2>
      aie.connect<South : 5, West : 1>
    }
    %tile_3_4 = aie.tile(3, 4)
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<South : 0, West : 1>
    }
    %tile_4_2 = aie.tile(4, 2)
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<East : 3, West : 3>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, West : 2>
      aie.connect<West : 2, South : 1>
    }
    %tile_5_2 = aie.tile(5, 2)
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<South : 1, West : 3>
      aie.connect<East : 3, West : 2>
      aie.connect<East : 2, West : 1>
    }
    %tile_6_2 = aie.tile(6, 2)
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<South : 1, West : 3>
      aie.connect<East : 1, West : 2>
    }
    %tile_7_2 = aie.tile(7, 2)
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<South : 1, West : 1>
    }
    %switchbox_4_1 = aie.switchbox(%mem_tile_4_1) {
      aie.connect<North : 1, DMA : 0>
      aie.connect<DMA : 0, South : 2>
    }
    %switchbox_4_0 = aie.switchbox(%shim_noc_tile_4_0) {
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_4_0 = aie.shim_mux(%shim_noc_tile_4_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<West : 2, DMA : 0>
      aie.connect<DMA : 0, West : 0>
      aie.connect<West : 3, DMA : 1>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
    aie.wire(%switchbox_2_4 : East, %switchbox_3_4 : West)
    aie.wire(%tile_3_4 : Core, %switchbox_3_4 : Core)
    aie.wire(%tile_3_4 : DMA, %switchbox_3_4 : DMA)
    aie.wire(%switchbox_3_3 : North, %switchbox_3_4 : South)
    aie.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
    aie.wire(%shim_mux_4_0 : North, %switchbox_4_0 : South)
    aie.wire(%shim_noc_tile_4_0 : DMA, %shim_mux_4_0 : DMA)
    aie.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
    aie.wire(%mem_tile_4_1 : Core, %switchbox_4_1 : Core)
    aie.wire(%mem_tile_4_1 : DMA, %switchbox_4_1 : DMA)
    aie.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
    aie.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
    aie.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
    aie.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
    aie.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
    aie.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
    aie.wire(%shim_mux_5_0 : North, %switchbox_5_0 : South)
    aie.wire(%shim_noc_tile_5_0 : DMA, %shim_mux_5_0 : DMA)
    aie.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
    aie.wire(%mem_tile_5_1 : Core, %switchbox_5_1 : Core)
    aie.wire(%mem_tile_5_1 : DMA, %switchbox_5_1 : DMA)
    aie.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
    aie.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
    aie.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
    aie.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
    aie.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
    aie.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
    aie.wire(%shim_mux_6_0 : North, %switchbox_6_0 : South)
    aie.wire(%shim_noc_tile_6_0 : DMA, %shim_mux_6_0 : DMA)
    aie.wire(%switchbox_5_1 : East, %switchbox_6_1 : West)
    aie.wire(%mem_tile_6_1 : Core, %switchbox_6_1 : Core)
    aie.wire(%mem_tile_6_1 : DMA, %switchbox_6_1 : DMA)
    aie.wire(%switchbox_6_0 : North, %switchbox_6_1 : South)
    aie.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
    aie.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
    aie.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
    aie.wire(%switchbox_6_1 : North, %switchbox_6_2 : South)
    aie.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)
    aie.wire(%shim_mux_7_0 : North, %switchbox_7_0 : South)
    aie.wire(%shim_noc_tile_7_0 : DMA, %shim_mux_7_0 : DMA)
    aie.wire(%switchbox_6_1 : East, %switchbox_7_1 : West)
    aie.wire(%mem_tile_7_1 : Core, %switchbox_7_1 : Core)
    aie.wire(%mem_tile_7_1 : DMA, %switchbox_7_1 : DMA)
    aie.wire(%switchbox_7_0 : North, %switchbox_7_1 : South)
    aie.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
    aie.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
    aie.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
    aie.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
  }
}
