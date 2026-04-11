module {
  aie.device(npu2) {
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
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_1_5, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%tile_2_5, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%tile_1_5, DMA : 0, %tile_2_5, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%tile_2_4, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%tile_1_4, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%tile_2_3, DMA : 0, %tile_1_3, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %tile_2_3, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%tile_2_2, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %tile_2_2, DMA : 1)
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @zero_fill_g_bf16(%g_stage0) : (memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%q_stage0, %k_stage0, %g_stage0) : (memref<64x64xbf16>, memref<64x96xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_2_19, Release, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%lock_0_2, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_3_35, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_3_39, AcquireGreaterEqual, 1)
      func.call @zero_fill_g_bf16(%g_stage1) : (memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_3_37, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%q_stage1, %k_stage1, %g_stage1) : (memref<64x64xbf16>, memref<64x96xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_3_36, Release, 1)
      aie.use_lock(%lock_0_3_38, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%lock_0_3, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @zero_fill_g_bf16(%g_stage2) : (memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%q_stage2, %k_stage2, %g_stage2) : (memref<64x64xbf16>, memref<64x96xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_4_51, Release, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%lock_0_4, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_5_69, AcquireGreaterEqual, 1)
      func.call @zero_fill_g_bf16(%g_stage3) : (memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_5_67, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%q_stage3, %k_stage3, %g_stage3) : (memref<64x64xbf16>, memref<64x96xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.use_lock(%lock_0_5_68, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%lock_0_5, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_2_23, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%gp_in_stage0) : (memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      func.call @matmul_g_b_bf16(%g_in_stage0, %v_stage0, %gp_in_stage0) : (memref<6144xbf16>, memref<64x96xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_2_24, Release, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%gp_in_stage1) : (memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_42, AcquireGreaterEqual, 1)
      func.call @matmul_g_b_bf16(%g_in_stage1, %v_stage1, %gp_in_stage1) : (memref<6144xbf16>, memref<64x96xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_3_41, Release, 1)
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.use_lock(%lock_1_3, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%gp_in_stage2) : (memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_4_59, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @matmul_g_b_bf16(%g_in_stage2, %v_stage2, %gp_in_stage2) : (memref<6144xbf16>, memref<64x96xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_58, Release, 1)
      aie.use_lock(%lock_1_4, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_1_5_70, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%gp_in_stage3) : (memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_5_74, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_72, AcquireGreaterEqual, 1)
      func.call @matmul_g_b_bf16(%g_in_stage3, %v_stage3, %gp_in_stage3) : (memref<6144xbf16>, memref<64x96xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_1_5_71, Release, 1)
      aie.use_lock(%lock_1_5_73, Release, 1)
      aie.use_lock(%lock_1_5, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %collapse_shape = memref.collapse_shape %softmax_out_stage0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_80 = memref.collapse_shape %cascade_up_stage0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      %collapse_shape_81 = memref.collapse_shape %cascade_sp_stage0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_2_28, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%softmax_gp_stage0) : (memref<64x64xbf16>) -> ()
      func.call @zero_fill_sp_bf16(%softmax_sp_stage0) : (memref<64x1xbf16>) -> ()
      func.call @neg_inf_fill_up_bf16(%softmax_up_stage0) : (memref<64x1xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @max_g_bf16(%softmax_g_stage0, %softmax_u_stage0) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @maximum_up_u_bf16(%softmax_up_stage0, %softmax_u_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_g_minus_u(%softmax_u_stage0, %softmax_g_stage0) : (memref<64x1xbf16>, memref<6144xbf16>) -> ()
      func.call @exp_up_minus_u(%softmax_up_stage0, %softmax_u_stage0, %softmax_r_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @mul_r_gp(%softmax_r_stage0, %softmax_gp_stage0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @vector_copy_32x96elems(%c0_i32, %softmax_g_stage0, %softmax_g_copy_stage0) : (i32, memref<6144xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_2_2_29, Release, 1)
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      func.call @vector_accum_32x64elems(%softmax_gv_stage0, %softmax_gp_stage0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_2_2_33, Release, 1)
      func.call @sum_g(%softmax_g_stage0, %softmax_s_stage0) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%softmax_sp_stage0, %softmax_r_stage0, %softmax_s_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_s_stage0, %softmax_sp_stage0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_u_stage0, %softmax_up_stage0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c4096 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %7 = aie.get_cascade() : vector<32xbf16>
      vector.store %7, %collapse_shape[%5] : memref<4096xbf16>, vector<32xbf16>
      %8 = arith.addi %5, %c32 : index
      cf.br ^bb6(%8 : index)
    ^bb8:  // pred: ^bb6
      cf.br ^bb9(%c0 : index)
    ^bb9(%9: index):  // 2 preds: ^bb8, ^bb10
      %10 = arith.cmpi slt, %9, %c64 : index
      cf.cond_br %10, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %11 = aie.get_cascade() : vector<32xbf16>
      vector.store %11, %collapse_shape_80[%9] : memref<64xbf16>, vector<32xbf16>
      %12 = arith.addi %9, %c32 : index
      cf.br ^bb9(%12 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%13: index):  // 2 preds: ^bb11, ^bb13
      %14 = arith.cmpi slt, %13, %c64 : index
      cf.cond_br %14, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      %15 = aie.get_cascade() : vector<32xbf16>
      vector.store %15, %collapse_shape_81[%13] : memref<64xbf16>, vector<32xbf16>
      %16 = arith.addi %13, %c32 : index
      cf.br ^bb12(%16 : index)
    ^bb14:  // pred: ^bb12
      func.call @vector_copy_32elems(%c0_i32, %softmax_up_stage0, %prev_up_stage0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @maximum_up_u_bf16(%cascade_up_stage0, %softmax_up_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_up_minus_u(%cascade_up_stage0, %softmax_up_stage0, %r_from_cascade_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_up_minus_u(%prev_up_stage0, %softmax_up_stage0, %r_from_local_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @mul_r_gp(%r_from_cascade_stage0, %softmax_out_stage0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @mul_r_gp(%r_from_local_stage0, %softmax_gp_stage0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @add_gp_g(%softmax_gp_stage0, %softmax_out_stage0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
      func.call @zero_fill_sp_bf16(%tmp_sp_stage0) : (memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%cascade_sp_stage0, %r_from_cascade_stage0, %tmp_sp_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%softmax_sp_stage0, %r_from_local_stage0, %tmp_sp_stage0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %tmp_sp_stage0, %cascade_sp_stage0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @div_gp_sp(%cascade_sp_stage0, %softmax_out_stage0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_2_2, Release, 1)
      %17 = arith.addi %0, %c1 : index
      cf.br ^bb1(%17 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %collapse_shape = memref.collapse_shape %cascade_gp_stage1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_80 = memref.collapse_shape %cascade_up_stage1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      %collapse_shape_81 = memref.collapse_shape %cascade_sp_stage1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      %collapse_shape_82 = memref.collapse_shape %softmax_up_stage1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb23
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb24
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%softmax_gp_stage1) : (memref<64x64xbf16>) -> ()
      func.call @zero_fill_sp_bf16(%softmax_sp_stage1) : (memref<64x1xbf16>) -> ()
      func.call @neg_inf_fill_up_bf16(%softmax_up_stage1) : (memref<64x1xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      func.call @max_g_bf16(%softmax_g_stage1, %softmax_u_stage1) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @maximum_up_u_bf16(%softmax_up_stage1, %softmax_u_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_g_minus_u(%softmax_u_stage1, %softmax_g_stage1) : (memref<64x1xbf16>, memref<6144xbf16>) -> ()
      func.call @exp_up_minus_u(%softmax_up_stage1, %softmax_u_stage1, %softmax_r_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @mul_r_gp(%softmax_r_stage1, %softmax_gp_stage1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @vector_copy_32x96elems(%c0_i32, %softmax_g_stage1, %softmax_g_copy_stage1) : (i32, memref<6144xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_2_3, Release, 1)
      aie.use_lock(%lock_2_3_49, AcquireGreaterEqual, 1)
      func.call @vector_accum_32x64elems(%softmax_gv_stage1, %softmax_gp_stage1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_2_3_48, Release, 1)
      func.call @sum_g(%softmax_g_stage1, %softmax_s_stage1) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%softmax_sp_stage1, %softmax_r_stage1, %softmax_s_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_s_stage1, %softmax_sp_stage1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_u_stage1, %softmax_up_stage1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      aie.use_lock(%lock_2_3_46, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c4096 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %7 = aie.get_cascade() : vector<32xbf16>
      vector.store %7, %collapse_shape[%5] : memref<4096xbf16>, vector<32xbf16>
      %8 = arith.addi %5, %c32 : index
      cf.br ^bb6(%8 : index)
    ^bb8:  // pred: ^bb6
      cf.br ^bb9(%c0 : index)
    ^bb9(%9: index):  // 2 preds: ^bb8, ^bb10
      %10 = arith.cmpi slt, %9, %c64 : index
      cf.cond_br %10, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %11 = aie.get_cascade() : vector<32xbf16>
      vector.store %11, %collapse_shape_80[%9] : memref<64xbf16>, vector<32xbf16>
      %12 = arith.addi %9, %c32 : index
      cf.br ^bb9(%12 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%13: index):  // 2 preds: ^bb11, ^bb13
      %14 = arith.cmpi slt, %13, %c64 : index
      cf.cond_br %14, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      %15 = aie.get_cascade() : vector<32xbf16>
      vector.store %15, %collapse_shape_81[%13] : memref<64xbf16>, vector<32xbf16>
      %16 = arith.addi %13, %c32 : index
      cf.br ^bb12(%16 : index)
    ^bb14:  // pred: ^bb12
      func.call @vector_copy_32elems(%c0_i32, %softmax_up_stage1, %prev_up_stage1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @maximum_up_u_bf16(%cascade_up_stage1, %softmax_up_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_up_minus_u(%cascade_up_stage1, %softmax_up_stage1, %r_from_cascade_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_up_minus_u(%prev_up_stage1, %softmax_up_stage1, %r_from_local_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @mul_r_gp(%r_from_cascade_stage1, %cascade_gp_stage1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @mul_r_gp(%r_from_local_stage1, %softmax_gp_stage1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @add_gp_g(%softmax_gp_stage1, %cascade_gp_stage1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
      func.call @zero_fill_sp_bf16(%tmp_sp_stage1) : (memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%cascade_sp_stage1, %r_from_cascade_stage1, %tmp_sp_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%softmax_sp_stage1, %r_from_local_stage1, %tmp_sp_stage1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %tmp_sp_stage1, %cascade_sp_stage1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      cf.br ^bb15(%c0 : index)
    ^bb15(%17: index):  // 2 preds: ^bb14, ^bb16
      %18 = arith.cmpi slt, %17, %c4096 : index
      cf.cond_br %18, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      %19 = vector.load %collapse_shape[%17] : memref<4096xbf16>, vector<32xbf16>
      aie.put_cascade(%19 : vector<32xbf16>)
      %20 = arith.addi %17, %c32 : index
      cf.br ^bb15(%20 : index)
    ^bb17:  // pred: ^bb15
      cf.br ^bb18(%c0 : index)
    ^bb18(%21: index):  // 2 preds: ^bb17, ^bb19
      %22 = arith.cmpi slt, %21, %c64 : index
      cf.cond_br %22, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      %23 = vector.load %collapse_shape_82[%21] : memref<64xbf16>, vector<32xbf16>
      aie.put_cascade(%23 : vector<32xbf16>)
      %24 = arith.addi %21, %c32 : index
      cf.br ^bb18(%24 : index)
    ^bb20:  // pred: ^bb18
      cf.br ^bb21(%c0 : index)
    ^bb21(%25: index):  // 2 preds: ^bb20, ^bb22
      %26 = arith.cmpi slt, %25, %c64 : index
      cf.cond_br %26, ^bb22, ^bb23
    ^bb22:  // pred: ^bb21
      %27 = vector.load %collapse_shape_81[%25] : memref<64xbf16>, vector<32xbf16>
      aie.put_cascade(%27 : vector<32xbf16>)
      %28 = arith.addi %25, %c32 : index
      cf.br ^bb21(%28 : index)
    ^bb23:  // pred: ^bb21
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb24:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %collapse_shape = memref.collapse_shape %cascade_gp_stage2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_80 = memref.collapse_shape %cascade_up_stage2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      %collapse_shape_81 = memref.collapse_shape %cascade_sp_stage2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      %collapse_shape_82 = memref.collapse_shape %softmax_up_stage2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb23
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb24
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%softmax_gp_stage2) : (memref<64x64xbf16>) -> ()
      func.call @zero_fill_sp_bf16(%softmax_sp_stage2) : (memref<64x1xbf16>) -> ()
      func.call @neg_inf_fill_up_bf16(%softmax_up_stage2) : (memref<64x1xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_62, AcquireGreaterEqual, 1)
      func.call @max_g_bf16(%softmax_g_stage2, %softmax_u_stage2) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @maximum_up_u_bf16(%softmax_up_stage2, %softmax_u_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_g_minus_u(%softmax_u_stage2, %softmax_g_stage2) : (memref<64x1xbf16>, memref<6144xbf16>) -> ()
      func.call @exp_up_minus_u(%softmax_up_stage2, %softmax_u_stage2, %softmax_r_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @mul_r_gp(%softmax_r_stage2, %softmax_gp_stage2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @vector_copy_32x96elems(%c0_i32, %softmax_g_stage2, %softmax_g_copy_stage2) : (i32, memref<6144xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_2_4, Release, 1)
      aie.use_lock(%lock_2_4_64, AcquireGreaterEqual, 1)
      func.call @vector_accum_32x64elems(%softmax_gv_stage2, %softmax_gp_stage2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_2_4_63, Release, 1)
      func.call @sum_g(%softmax_g_stage2, %softmax_s_stage2) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%softmax_sp_stage2, %softmax_r_stage2, %softmax_s_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_s_stage2, %softmax_sp_stage2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_u_stage2, %softmax_up_stage2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      aie.use_lock(%lock_2_4_61, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c4096 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %7 = aie.get_cascade() : vector<32xbf16>
      vector.store %7, %collapse_shape[%5] : memref<4096xbf16>, vector<32xbf16>
      %8 = arith.addi %5, %c32 : index
      cf.br ^bb6(%8 : index)
    ^bb8:  // pred: ^bb6
      cf.br ^bb9(%c0 : index)
    ^bb9(%9: index):  // 2 preds: ^bb8, ^bb10
      %10 = arith.cmpi slt, %9, %c64 : index
      cf.cond_br %10, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %11 = aie.get_cascade() : vector<32xbf16>
      vector.store %11, %collapse_shape_80[%9] : memref<64xbf16>, vector<32xbf16>
      %12 = arith.addi %9, %c32 : index
      cf.br ^bb9(%12 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%13: index):  // 2 preds: ^bb11, ^bb13
      %14 = arith.cmpi slt, %13, %c64 : index
      cf.cond_br %14, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      %15 = aie.get_cascade() : vector<32xbf16>
      vector.store %15, %collapse_shape_81[%13] : memref<64xbf16>, vector<32xbf16>
      %16 = arith.addi %13, %c32 : index
      cf.br ^bb12(%16 : index)
    ^bb14:  // pred: ^bb12
      func.call @vector_copy_32elems(%c0_i32, %softmax_up_stage2, %prev_up_stage2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @maximum_up_u_bf16(%cascade_up_stage2, %softmax_up_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_up_minus_u(%cascade_up_stage2, %softmax_up_stage2, %r_from_cascade_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_up_minus_u(%prev_up_stage2, %softmax_up_stage2, %r_from_local_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @mul_r_gp(%r_from_cascade_stage2, %cascade_gp_stage2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @mul_r_gp(%r_from_local_stage2, %softmax_gp_stage2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @add_gp_g(%softmax_gp_stage2, %cascade_gp_stage2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
      func.call @zero_fill_sp_bf16(%tmp_sp_stage2) : (memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%cascade_sp_stage2, %r_from_cascade_stage2, %tmp_sp_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%softmax_sp_stage2, %r_from_local_stage2, %tmp_sp_stage2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %tmp_sp_stage2, %cascade_sp_stage2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      cf.br ^bb15(%c0 : index)
    ^bb15(%17: index):  // 2 preds: ^bb14, ^bb16
      %18 = arith.cmpi slt, %17, %c4096 : index
      cf.cond_br %18, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      %19 = vector.load %collapse_shape[%17] : memref<4096xbf16>, vector<32xbf16>
      aie.put_cascade(%19 : vector<32xbf16>)
      %20 = arith.addi %17, %c32 : index
      cf.br ^bb15(%20 : index)
    ^bb17:  // pred: ^bb15
      cf.br ^bb18(%c0 : index)
    ^bb18(%21: index):  // 2 preds: ^bb17, ^bb19
      %22 = arith.cmpi slt, %21, %c64 : index
      cf.cond_br %22, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      %23 = vector.load %collapse_shape_82[%21] : memref<64xbf16>, vector<32xbf16>
      aie.put_cascade(%23 : vector<32xbf16>)
      %24 = arith.addi %21, %c32 : index
      cf.br ^bb18(%24 : index)
    ^bb20:  // pred: ^bb18
      cf.br ^bb21(%c0 : index)
    ^bb21(%25: index):  // 2 preds: ^bb20, ^bb22
      %26 = arith.cmpi slt, %25, %c64 : index
      cf.cond_br %26, ^bb22, ^bb23
    ^bb22:  // pred: ^bb21
      %27 = vector.load %collapse_shape_81[%25] : memref<64xbf16>, vector<32xbf16>
      aie.put_cascade(%27 : vector<32xbf16>)
      %28 = arith.addi %25, %c32 : index
      cf.br ^bb21(%28 : index)
    ^bb23:  // pred: ^bb21
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb24:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %collapse_shape = memref.collapse_shape %softmax_gp_stage3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
      %collapse_shape_80 = memref.collapse_shape %softmax_up_stage3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      %collapse_shape_81 = memref.collapse_shape %softmax_sp_stage3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      func.call @zero_fill_gp_bf16(%softmax_gp_stage3) : (memref<64x64xbf16>) -> ()
      func.call @zero_fill_sp_bf16(%softmax_sp_stage3) : (memref<64x1xbf16>) -> ()
      func.call @neg_inf_fill_up_bf16(%softmax_up_stage3) : (memref<64x1xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_2_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_77, AcquireGreaterEqual, 1)
      func.call @max_g_bf16(%softmax_g_stage3, %softmax_u_stage3) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @maximum_up_u_bf16(%softmax_up_stage3, %softmax_u_stage3) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @exp_g_minus_u(%softmax_u_stage3, %softmax_g_stage3) : (memref<64x1xbf16>, memref<6144xbf16>) -> ()
      func.call @exp_up_minus_u(%softmax_up_stage3, %softmax_u_stage3, %softmax_r_stage3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @mul_r_gp(%softmax_r_stage3, %softmax_gp_stage3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
      func.call @vector_copy_32x96elems(%c0_i32, %softmax_g_stage3, %softmax_g_copy_stage3) : (i32, memref<6144xbf16>, memref<6144xbf16>) -> ()
      aie.use_lock(%lock_2_5, Release, 1)
      aie.use_lock(%lock_2_5_79, AcquireGreaterEqual, 1)
      func.call @vector_accum_32x64elems(%softmax_gv_stage3, %softmax_gp_stage3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
      aie.use_lock(%lock_2_5_78, Release, 1)
      func.call @sum_g(%softmax_g_stage3, %softmax_s_stage3) : (memref<6144xbf16>, memref<64x1xbf16>) -> ()
      func.call @accum_sp_r_s(%softmax_sp_stage3, %softmax_r_stage3, %softmax_s_stage3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_s_stage3, %softmax_sp_stage3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %softmax_u_stage3, %softmax_up_stage3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
      aie.use_lock(%lock_2_5_76, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c4096 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %7 = vector.load %collapse_shape[%5] : memref<4096xbf16>, vector<32xbf16>
      aie.put_cascade(%7 : vector<32xbf16>)
      %8 = arith.addi %5, %c32 : index
      cf.br ^bb6(%8 : index)
    ^bb8:  // pred: ^bb6
      cf.br ^bb9(%c0 : index)
    ^bb9(%9: index):  // 2 preds: ^bb8, ^bb10
      %10 = arith.cmpi slt, %9, %c64 : index
      cf.cond_br %10, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %11 = vector.load %collapse_shape_80[%9] : memref<64xbf16>, vector<32xbf16>
      aie.put_cascade(%11 : vector<32xbf16>)
      %12 = arith.addi %9, %c32 : index
      cf.br ^bb9(%12 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%13: index):  // 2 preds: ^bb11, ^bb13
      %14 = arith.cmpi slt, %13, %c64 : index
      cf.cond_br %14, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      %15 = vector.load %collapse_shape_81[%13] : memref<64xbf16>, vector<32xbf16>
      aie.put_cascade(%15 : vector<32xbf16>)
      %16 = arith.addi %13, %c32 : index
      cf.br ^bb12(%16 : index)
    ^bb14:  // pred: ^bb12
      %17 = arith.addi %0, %c1 : index
      cf.br ^bb1(%17 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["attn.o"]}
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
      aiex.dma_await_task(%12)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%9)
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
  }
}
