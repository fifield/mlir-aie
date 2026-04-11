module {
  aie.device(npu2) {
    %Q = aie.external_buffer {sym_name = "Q"} : memref<2x512x64xbf16>
    %K = aie.external_buffer {sym_name = "K"} : memref<2x512x64xbf16>
    %V = aie.external_buffer {sym_name = "V"} : memref<2x512x64xbf16>
    %Out = aie.external_buffer {sym_name = "Out"} : memref<2x512x64xbf16>
    func.func private @zero_fill_g_bf16_pythoc(memref<4096xbf16>) attributes {link_with = "/tmp/pythoc_iron_9dxibvue/zero_fill_g_bf16_pythoc.o"}
    func.func private @zero_fill_gp_bf16_pythoc(memref<64x64xbf16>) attributes {link_with = "/tmp/pythoc_iron_tf76whxs/zero_fill_gp_bf16_pythoc.o"}
    func.func private @zero_fill_sp_bf16_pythoc(memref<64x1xbf16>) attributes {link_with = "/tmp/pythoc_iron_fvmho0qs/zero_fill_sp_bf16_pythoc.o"}
    func.func private @neg_inf_fill_up_bf16_pythoc(memref<64x1xbf16>) attributes {link_with = "/tmp/pythoc_iron_gvq3wbc0/neg_inf_fill_up_bf16_pythoc.o"}
    func.func private @copy_tile_pythoc(memref<64x64xbf16>, memref<64x64xbf16>) attributes {link_with = "/tmp/pythoc_iron_w4qe1xyv/copy_tile_pythoc.o"}
    func.func private @matmul_a_b_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) attributes {link_with = "attn.o"}
    func.func private @fused_softmax(memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @mul_r_gp(memref<64x1xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @matmul_g_b_bf16(memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @accum_sp_r_s(memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @vector_copy_32elems_pythoc(i32, memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "/tmp/pythoc_iron_m04puyht/vector_copy_32elems_pythoc.o"}
    func.func private @maximum_up_u_bf16(memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @exp_up_minus_u(memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) attributes {link_with = "attn.o"}
    func.func private @add_gp_g(memref<64x64xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    func.func private @div_gp_sp(memref<64x1xbf16>, memref<64x64xbf16>) attributes {link_with = "attn.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_2 = aie.tile(4, 2)
    %tile_4_3 = aie.tile(4, 3)
    %tile_4_4 = aie.tile(4, 4)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_2 = aie.tile(5, 2)
    %tile_5_3 = aie.tile(5, 3)
    %tile_5_4 = aie.tile(5, 4)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_3 = aie.tile(6, 3)
    %tile_6_4 = aie.tile(6, 4)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_2 = aie.tile(7, 2)
    %tile_7_3 = aie.tile(7, 3)
    %tile_7_4 = aie.tile(7, 4)
    %tile_7_5 = aie.tile(7, 5)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_3_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.cascade_flow(%tile_0_5, %tile_0_4)
    aie.cascade_flow(%tile_0_4, %tile_0_3)
    aie.cascade_flow(%tile_0_3, %tile_0_2)
    aie.cascade_flow(%tile_1_5, %tile_1_4)
    aie.cascade_flow(%tile_1_4, %tile_1_3)
    aie.cascade_flow(%tile_1_3, %tile_1_2)
    aie.cascade_flow(%tile_2_5, %tile_2_4)
    aie.cascade_flow(%tile_2_4, %tile_2_3)
    aie.cascade_flow(%tile_2_3, %tile_2_2)
    aie.cascade_flow(%tile_3_5, %tile_3_4)
    aie.cascade_flow(%tile_3_4, %tile_3_3)
    aie.cascade_flow(%tile_3_3, %tile_3_2)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %mem_tile_4_1, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %mem_tile_5_1, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %mem_tile_6_1, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %mem_tile_7_1, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_4_2, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_4_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_5_2, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_5_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_6_2, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_6_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_7_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_4_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_4_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_5_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_5_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_6_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_6_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_7_3, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 2, %tile_7_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_4_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_4_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_5_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_5_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_7_4, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_7_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_4_5, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_4_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_5_5, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_5_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_6_5, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_6_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_7_5, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_7_5, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 2)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 2)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 2)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 2)
    aie.cascade_flow(%tile_4_5, %tile_4_4)
    aie.cascade_flow(%tile_4_4, %tile_4_3)
    aie.cascade_flow(%tile_4_3, %tile_4_2)
    aie.cascade_flow(%tile_5_5, %tile_5_4)
    aie.cascade_flow(%tile_5_4, %tile_5_3)
    aie.cascade_flow(%tile_5_3, %tile_5_2)
    aie.cascade_flow(%tile_6_5, %tile_6_4)
    aie.cascade_flow(%tile_6_4, %tile_6_3)
    aie.cascade_flow(%tile_6_3, %tile_6_2)
    aie.cascade_flow(%tile_7_5, %tile_7_4)
    aie.cascade_flow(%tile_7_4, %tile_7_3)
    aie.cascade_flow(%tile_7_3, %tile_7_2)
    aie.shim_dma_allocation @air_QKIn_0_0_0_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_1_0_0_0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_2_0_0_0(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_3_0_0_0(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0_0_0_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_1_0_0_0(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_0_0_0(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_3_0_0_0(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_0_0_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_0_0_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_QKIn_0_1_0_0(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_1_1_0_0(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_2_1_0_0(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_QKIn_3_1_0_0(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0_1_0_0(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_1_1_0_0(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_1_0_0(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_3_1_0_0(%shim_noc_tile_7_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_0_1_0_0(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_1(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_2(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_3(%shim_noc_tile_7_0, S2MM, 0)
    %qk_l2_col0 = aie.buffer(%mem_tile_0_1) {sym_name = "qk_l2_col0"} : memref<64x64xbf16> 
    %v_l2_col0 = aie.buffer(%mem_tile_0_1) {sym_name = "v_l2_col0"} : memref<64x64xbf16> 
    %out_l2_col0 = aie.buffer(%mem_tile_0_1) {sym_name = "out_l2_col0"} : memref<64x64xbf16> 
    %lock_0_1 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_1_0 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_1 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_2 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_3 = aie.lock(%mem_tile_0_1, 4) {init = 0 : i32}
    %lock_0_1_4 = aie.lock(%mem_tile_0_1, 5) {init = 1 : i32}
    %qk_l2_col1 = aie.buffer(%mem_tile_1_1) {sym_name = "qk_l2_col1"} : memref<64x64xbf16> 
    %v_l2_col1 = aie.buffer(%mem_tile_1_1) {sym_name = "v_l2_col1"} : memref<64x64xbf16> 
    %out_l2_col1 = aie.buffer(%mem_tile_1_1) {sym_name = "out_l2_col1"} : memref<64x64xbf16> 
    %lock_1_1 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_1_1_5 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_6 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_7 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_8 = aie.lock(%mem_tile_1_1, 4) {init = 0 : i32}
    %lock_1_1_9 = aie.lock(%mem_tile_1_1, 5) {init = 1 : i32}
    %qk_l2_col2 = aie.buffer(%mem_tile_2_1) {sym_name = "qk_l2_col2"} : memref<64x64xbf16> 
    %v_l2_col2 = aie.buffer(%mem_tile_2_1) {sym_name = "v_l2_col2"} : memref<64x64xbf16> 
    %out_l2_col2 = aie.buffer(%mem_tile_2_1) {sym_name = "out_l2_col2"} : memref<64x64xbf16> 
    %lock_2_1 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_2_1_10 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_11 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_12 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_13 = aie.lock(%mem_tile_2_1, 4) {init = 0 : i32}
    %lock_2_1_14 = aie.lock(%mem_tile_2_1, 5) {init = 1 : i32}
    %qk_l2_col3 = aie.buffer(%mem_tile_3_1) {sym_name = "qk_l2_col3"} : memref<64x64xbf16> 
    %v_l2_col3 = aie.buffer(%mem_tile_3_1) {sym_name = "v_l2_col3"} : memref<64x64xbf16> 
    %out_l2_col3 = aie.buffer(%mem_tile_3_1) {sym_name = "out_l2_col3"} : memref<64x64xbf16> 
    %lock_3_1 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_3_1_15 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_16 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_17 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_18 = aie.lock(%mem_tile_3_1, 4) {init = 0 : i32}
    %lock_3_1_19 = aie.lock(%mem_tile_3_1, 5) {init = 1 : i32}
    %qk_l2_col4 = aie.buffer(%mem_tile_4_1) {sym_name = "qk_l2_col4"} : memref<64x64xbf16> 
    %v_l2_col4 = aie.buffer(%mem_tile_4_1) {sym_name = "v_l2_col4"} : memref<64x64xbf16> 
    %out_l2_col4 = aie.buffer(%mem_tile_4_1) {sym_name = "out_l2_col4"} : memref<64x64xbf16> 
    %lock_4_1 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_4_1_20 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_21 = aie.lock(%mem_tile_4_1, 2) {init = 0 : i32}
    %lock_4_1_22 = aie.lock(%mem_tile_4_1, 3) {init = 1 : i32}
    %lock_4_1_23 = aie.lock(%mem_tile_4_1, 4) {init = 0 : i32}
    %lock_4_1_24 = aie.lock(%mem_tile_4_1, 5) {init = 1 : i32}
    %qk_l2_col5 = aie.buffer(%mem_tile_5_1) {sym_name = "qk_l2_col5"} : memref<64x64xbf16> 
    %v_l2_col5 = aie.buffer(%mem_tile_5_1) {sym_name = "v_l2_col5"} : memref<64x64xbf16> 
    %out_l2_col5 = aie.buffer(%mem_tile_5_1) {sym_name = "out_l2_col5"} : memref<64x64xbf16> 
    %lock_5_1 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_5_1_25 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_26 = aie.lock(%mem_tile_5_1, 2) {init = 0 : i32}
    %lock_5_1_27 = aie.lock(%mem_tile_5_1, 3) {init = 1 : i32}
    %lock_5_1_28 = aie.lock(%mem_tile_5_1, 4) {init = 0 : i32}
    %lock_5_1_29 = aie.lock(%mem_tile_5_1, 5) {init = 1 : i32}
    %qk_l2_col6 = aie.buffer(%mem_tile_6_1) {sym_name = "qk_l2_col6"} : memref<64x64xbf16> 
    %v_l2_col6 = aie.buffer(%mem_tile_6_1) {sym_name = "v_l2_col6"} : memref<64x64xbf16> 
    %out_l2_col6 = aie.buffer(%mem_tile_6_1) {sym_name = "out_l2_col6"} : memref<64x64xbf16> 
    %lock_6_1 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_6_1_30 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %lock_6_1_31 = aie.lock(%mem_tile_6_1, 2) {init = 0 : i32}
    %lock_6_1_32 = aie.lock(%mem_tile_6_1, 3) {init = 1 : i32}
    %lock_6_1_33 = aie.lock(%mem_tile_6_1, 4) {init = 0 : i32}
    %lock_6_1_34 = aie.lock(%mem_tile_6_1, 5) {init = 1 : i32}
    %qk_l2_col7 = aie.buffer(%mem_tile_7_1) {sym_name = "qk_l2_col7"} : memref<64x64xbf16> 
    %v_l2_col7 = aie.buffer(%mem_tile_7_1) {sym_name = "v_l2_col7"} : memref<64x64xbf16> 
    %out_l2_col7 = aie.buffer(%mem_tile_7_1) {sym_name = "out_l2_col7"} : memref<64x64xbf16> 
    %lock_7_1 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_7_1_35 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %lock_7_1_36 = aie.lock(%mem_tile_7_1, 2) {init = 0 : i32}
    %lock_7_1_37 = aie.lock(%mem_tile_7_1, 3) {init = 1 : i32}
    %lock_7_1_38 = aie.lock(%mem_tile_7_1, 4) {init = 0 : i32}
    %lock_7_1_39 = aie.lock(%mem_tile_7_1, 5) {init = 1 : i32}
    %qk_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "qk_seg0_s0_q0"} : memref<64x64xbf16> 
    %q_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "q_seg0_s0_q0"} : memref<64x64xbf16> 
    %v_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "v_seg0_s0_q0"} : memref<64x64xbf16> 
    %g_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "g_seg0_s0_q0"} : memref<64x64xbf16> 
    %gp_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "gp_seg0_s0_q0"} : memref<64x64xbf16> 
    %up_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "up_seg0_s0_q0"} : memref<64x1xbf16> 
    %sp_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "sp_seg0_s0_q0"} : memref<64x1xbf16> 
    %s_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "s_seg0_s0_q0"} : memref<64x1xbf16> 
    %r_seg0_s0_q0 = aie.buffer(%tile_0_2) {sym_name = "r_seg0_s0_q0"} : memref<64x1xbf16> 
    %merged_gp_seg0_q0 = aie.buffer(%tile_0_2) {sym_name = "merged_gp_seg0_q0"} : memref<64x64xbf16> 
    %merged_up_seg0_q0 = aie.buffer(%tile_0_2) {sym_name = "merged_up_seg0_q0"} : memref<64x1xbf16> 
    %merged_sp_seg0_q0 = aie.buffer(%tile_0_2) {sym_name = "merged_sp_seg0_q0"} : memref<64x1xbf16> 
    %prev_up_seg0_q0 = aie.buffer(%tile_0_2) {sym_name = "prev_up_seg0_q0"} : memref<64x1xbf16> 
    %r_cascade_seg0_q0 = aie.buffer(%tile_0_2) {sym_name = "r_cascade_seg0_q0"} : memref<64x1xbf16> 
    %r_local_seg0_q0 = aie.buffer(%tile_0_2) {sym_name = "r_local_seg0_q0"} : memref<64x1xbf16> 
    %tmp_sp_seg0_q0 = aie.buffer(%tile_0_2) {sym_name = "tmp_sp_seg0_q0"} : memref<64x1xbf16> 
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_2_40 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_41 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_42 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_43 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_44 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %qk_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "qk_seg0_s1_q0"} : memref<64x64xbf16> 
    %q_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "q_seg0_s1_q0"} : memref<64x64xbf16> 
    %v_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "v_seg0_s1_q0"} : memref<64x64xbf16> 
    %g_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "g_seg0_s1_q0"} : memref<64x64xbf16> 
    %gp_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "gp_seg0_s1_q0"} : memref<64x64xbf16> 
    %up_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "up_seg0_s1_q0"} : memref<64x1xbf16> 
    %sp_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "sp_seg0_s1_q0"} : memref<64x1xbf16> 
    %s_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "s_seg0_s1_q0"} : memref<64x1xbf16> 
    %r_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "r_seg0_s1_q0"} : memref<64x1xbf16> 
    %merged_gp_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "merged_gp_seg0_s1_q0"} : memref<64x64xbf16> 
    %merged_up_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "merged_up_seg0_s1_q0"} : memref<64x1xbf16> 
    %merged_sp_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "merged_sp_seg0_s1_q0"} : memref<64x1xbf16> 
    %prev_up_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "prev_up_seg0_s1_q0"} : memref<64x1xbf16> 
    %r_cascade_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "r_cascade_seg0_s1_q0"} : memref<64x1xbf16> 
    %r_local_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "r_local_seg0_s1_q0"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s1_q0 = aie.buffer(%tile_0_3) {sym_name = "tmp_sp_seg0_s1_q0"} : memref<64x1xbf16> 
    %lock_0_3 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_45 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_3_46 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_47 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %qk_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "qk_seg0_s2_q0"} : memref<64x64xbf16> 
    %q_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "q_seg0_s2_q0"} : memref<64x64xbf16> 
    %v_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "v_seg0_s2_q0"} : memref<64x64xbf16> 
    %g_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "g_seg0_s2_q0"} : memref<64x64xbf16> 
    %gp_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "gp_seg0_s2_q0"} : memref<64x64xbf16> 
    %up_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "up_seg0_s2_q0"} : memref<64x1xbf16> 
    %sp_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "sp_seg0_s2_q0"} : memref<64x1xbf16> 
    %s_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "s_seg0_s2_q0"} : memref<64x1xbf16> 
    %r_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "r_seg0_s2_q0"} : memref<64x1xbf16> 
    %merged_gp_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "merged_gp_seg0_s2_q0"} : memref<64x64xbf16> 
    %merged_up_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "merged_up_seg0_s2_q0"} : memref<64x1xbf16> 
    %merged_sp_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "merged_sp_seg0_s2_q0"} : memref<64x1xbf16> 
    %prev_up_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "prev_up_seg0_s2_q0"} : memref<64x1xbf16> 
    %r_cascade_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "r_cascade_seg0_s2_q0"} : memref<64x1xbf16> 
    %r_local_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "r_local_seg0_s2_q0"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s2_q0 = aie.buffer(%tile_0_4) {sym_name = "tmp_sp_seg0_s2_q0"} : memref<64x1xbf16> 
    %lock_0_4 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_48 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_0_4_49 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_50 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %qk_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "qk_seg0_s3_q0"} : memref<64x64xbf16> 
    %q_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "q_seg0_s3_q0"} : memref<64x64xbf16> 
    %v_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "v_seg0_s3_q0"} : memref<64x64xbf16> 
    %g_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "g_seg0_s3_q0"} : memref<64x64xbf16> 
    %gp_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "gp_seg0_s3_q0"} : memref<64x64xbf16> 
    %up_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "up_seg0_s3_q0"} : memref<64x1xbf16> 
    %sp_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "sp_seg0_s3_q0"} : memref<64x1xbf16> 
    %s_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "s_seg0_s3_q0"} : memref<64x1xbf16> 
    %r_seg0_s3_q0 = aie.buffer(%tile_0_5) {sym_name = "r_seg0_s3_q0"} : memref<64x1xbf16> 
    %lock_0_5 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_51 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_0_5_52 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_53 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %qk_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "qk_seg0_s0_q1"} : memref<64x64xbf16> 
    %q_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "q_seg0_s0_q1"} : memref<64x64xbf16> 
    %v_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "v_seg0_s0_q1"} : memref<64x64xbf16> 
    %g_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "g_seg0_s0_q1"} : memref<64x64xbf16> 
    %gp_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "gp_seg0_s0_q1"} : memref<64x64xbf16> 
    %up_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "up_seg0_s0_q1"} : memref<64x1xbf16> 
    %sp_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "sp_seg0_s0_q1"} : memref<64x1xbf16> 
    %s_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "s_seg0_s0_q1"} : memref<64x1xbf16> 
    %r_seg0_s0_q1 = aie.buffer(%tile_1_2) {sym_name = "r_seg0_s0_q1"} : memref<64x1xbf16> 
    %merged_gp_seg0_q1 = aie.buffer(%tile_1_2) {sym_name = "merged_gp_seg0_q1"} : memref<64x64xbf16> 
    %merged_up_seg0_q1 = aie.buffer(%tile_1_2) {sym_name = "merged_up_seg0_q1"} : memref<64x1xbf16> 
    %merged_sp_seg0_q1 = aie.buffer(%tile_1_2) {sym_name = "merged_sp_seg0_q1"} : memref<64x1xbf16> 
    %prev_up_seg0_q1 = aie.buffer(%tile_1_2) {sym_name = "prev_up_seg0_q1"} : memref<64x1xbf16> 
    %r_cascade_seg0_q1 = aie.buffer(%tile_1_2) {sym_name = "r_cascade_seg0_q1"} : memref<64x1xbf16> 
    %r_local_seg0_q1 = aie.buffer(%tile_1_2) {sym_name = "r_local_seg0_q1"} : memref<64x1xbf16> 
    %tmp_sp_seg0_q1 = aie.buffer(%tile_1_2) {sym_name = "tmp_sp_seg0_q1"} : memref<64x1xbf16> 
    %lock_1_2 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_1_2_54 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_55 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_56 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_57 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_58 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %qk_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "qk_seg0_s1_q1"} : memref<64x64xbf16> 
    %q_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "q_seg0_s1_q1"} : memref<64x64xbf16> 
    %v_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "v_seg0_s1_q1"} : memref<64x64xbf16> 
    %g_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "g_seg0_s1_q1"} : memref<64x64xbf16> 
    %gp_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "gp_seg0_s1_q1"} : memref<64x64xbf16> 
    %up_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "up_seg0_s1_q1"} : memref<64x1xbf16> 
    %sp_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "sp_seg0_s1_q1"} : memref<64x1xbf16> 
    %s_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "s_seg0_s1_q1"} : memref<64x1xbf16> 
    %r_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "r_seg0_s1_q1"} : memref<64x1xbf16> 
    %merged_gp_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "merged_gp_seg0_s1_q1"} : memref<64x64xbf16> 
    %merged_up_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "merged_up_seg0_s1_q1"} : memref<64x1xbf16> 
    %merged_sp_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "merged_sp_seg0_s1_q1"} : memref<64x1xbf16> 
    %prev_up_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "prev_up_seg0_s1_q1"} : memref<64x1xbf16> 
    %r_cascade_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "r_cascade_seg0_s1_q1"} : memref<64x1xbf16> 
    %r_local_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "r_local_seg0_s1_q1"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s1_q1 = aie.buffer(%tile_1_3) {sym_name = "tmp_sp_seg0_s1_q1"} : memref<64x1xbf16> 
    %lock_1_3 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_59 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_1_3_60 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_61 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %qk_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "qk_seg0_s2_q1"} : memref<64x64xbf16> 
    %q_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "q_seg0_s2_q1"} : memref<64x64xbf16> 
    %v_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "v_seg0_s2_q1"} : memref<64x64xbf16> 
    %g_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "g_seg0_s2_q1"} : memref<64x64xbf16> 
    %gp_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "gp_seg0_s2_q1"} : memref<64x64xbf16> 
    %up_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "up_seg0_s2_q1"} : memref<64x1xbf16> 
    %sp_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "sp_seg0_s2_q1"} : memref<64x1xbf16> 
    %s_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "s_seg0_s2_q1"} : memref<64x1xbf16> 
    %r_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "r_seg0_s2_q1"} : memref<64x1xbf16> 
    %merged_gp_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "merged_gp_seg0_s2_q1"} : memref<64x64xbf16> 
    %merged_up_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "merged_up_seg0_s2_q1"} : memref<64x1xbf16> 
    %merged_sp_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "merged_sp_seg0_s2_q1"} : memref<64x1xbf16> 
    %prev_up_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "prev_up_seg0_s2_q1"} : memref<64x1xbf16> 
    %r_cascade_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "r_cascade_seg0_s2_q1"} : memref<64x1xbf16> 
    %r_local_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "r_local_seg0_s2_q1"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s2_q1 = aie.buffer(%tile_1_4) {sym_name = "tmp_sp_seg0_s2_q1"} : memref<64x1xbf16> 
    %lock_1_4 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_62 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_1_4_63 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_64 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %qk_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "qk_seg0_s3_q1"} : memref<64x64xbf16> 
    %q_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "q_seg0_s3_q1"} : memref<64x64xbf16> 
    %v_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "v_seg0_s3_q1"} : memref<64x64xbf16> 
    %g_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "g_seg0_s3_q1"} : memref<64x64xbf16> 
    %gp_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "gp_seg0_s3_q1"} : memref<64x64xbf16> 
    %up_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "up_seg0_s3_q1"} : memref<64x1xbf16> 
    %sp_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "sp_seg0_s3_q1"} : memref<64x1xbf16> 
    %s_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "s_seg0_s3_q1"} : memref<64x1xbf16> 
    %r_seg0_s3_q1 = aie.buffer(%tile_1_5) {sym_name = "r_seg0_s3_q1"} : memref<64x1xbf16> 
    %lock_1_5 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_65 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_1_5_66 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_67 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %qk_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "qk_seg0_s0_q2"} : memref<64x64xbf16> 
    %q_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "q_seg0_s0_q2"} : memref<64x64xbf16> 
    %v_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "v_seg0_s0_q2"} : memref<64x64xbf16> 
    %g_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "g_seg0_s0_q2"} : memref<64x64xbf16> 
    %gp_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "gp_seg0_s0_q2"} : memref<64x64xbf16> 
    %up_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "up_seg0_s0_q2"} : memref<64x1xbf16> 
    %sp_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "sp_seg0_s0_q2"} : memref<64x1xbf16> 
    %s_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "s_seg0_s0_q2"} : memref<64x1xbf16> 
    %r_seg0_s0_q2 = aie.buffer(%tile_2_2) {sym_name = "r_seg0_s0_q2"} : memref<64x1xbf16> 
    %merged_gp_seg0_q2 = aie.buffer(%tile_2_2) {sym_name = "merged_gp_seg0_q2"} : memref<64x64xbf16> 
    %merged_up_seg0_q2 = aie.buffer(%tile_2_2) {sym_name = "merged_up_seg0_q2"} : memref<64x1xbf16> 
    %merged_sp_seg0_q2 = aie.buffer(%tile_2_2) {sym_name = "merged_sp_seg0_q2"} : memref<64x1xbf16> 
    %prev_up_seg0_q2 = aie.buffer(%tile_2_2) {sym_name = "prev_up_seg0_q2"} : memref<64x1xbf16> 
    %r_cascade_seg0_q2 = aie.buffer(%tile_2_2) {sym_name = "r_cascade_seg0_q2"} : memref<64x1xbf16> 
    %r_local_seg0_q2 = aie.buffer(%tile_2_2) {sym_name = "r_local_seg0_q2"} : memref<64x1xbf16> 
    %tmp_sp_seg0_q2 = aie.buffer(%tile_2_2) {sym_name = "tmp_sp_seg0_q2"} : memref<64x1xbf16> 
    %lock_2_2 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_2_2_68 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_69 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_70 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_71 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_72 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %qk_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "qk_seg0_s1_q2"} : memref<64x64xbf16> 
    %q_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "q_seg0_s1_q2"} : memref<64x64xbf16> 
    %v_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "v_seg0_s1_q2"} : memref<64x64xbf16> 
    %g_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "g_seg0_s1_q2"} : memref<64x64xbf16> 
    %gp_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "gp_seg0_s1_q2"} : memref<64x64xbf16> 
    %up_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "up_seg0_s1_q2"} : memref<64x1xbf16> 
    %sp_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "sp_seg0_s1_q2"} : memref<64x1xbf16> 
    %s_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "s_seg0_s1_q2"} : memref<64x1xbf16> 
    %r_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "r_seg0_s1_q2"} : memref<64x1xbf16> 
    %merged_gp_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "merged_gp_seg0_s1_q2"} : memref<64x64xbf16> 
    %merged_up_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "merged_up_seg0_s1_q2"} : memref<64x1xbf16> 
    %merged_sp_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "merged_sp_seg0_s1_q2"} : memref<64x1xbf16> 
    %prev_up_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "prev_up_seg0_s1_q2"} : memref<64x1xbf16> 
    %r_cascade_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "r_cascade_seg0_s1_q2"} : memref<64x1xbf16> 
    %r_local_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "r_local_seg0_s1_q2"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s1_q2 = aie.buffer(%tile_2_3) {sym_name = "tmp_sp_seg0_s1_q2"} : memref<64x1xbf16> 
    %lock_2_3 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_73 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_2_3_74 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_75 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %qk_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "qk_seg0_s2_q2"} : memref<64x64xbf16> 
    %q_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "q_seg0_s2_q2"} : memref<64x64xbf16> 
    %v_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "v_seg0_s2_q2"} : memref<64x64xbf16> 
    %g_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "g_seg0_s2_q2"} : memref<64x64xbf16> 
    %gp_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "gp_seg0_s2_q2"} : memref<64x64xbf16> 
    %up_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "up_seg0_s2_q2"} : memref<64x1xbf16> 
    %sp_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "sp_seg0_s2_q2"} : memref<64x1xbf16> 
    %s_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "s_seg0_s2_q2"} : memref<64x1xbf16> 
    %r_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "r_seg0_s2_q2"} : memref<64x1xbf16> 
    %merged_gp_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "merged_gp_seg0_s2_q2"} : memref<64x64xbf16> 
    %merged_up_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "merged_up_seg0_s2_q2"} : memref<64x1xbf16> 
    %merged_sp_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "merged_sp_seg0_s2_q2"} : memref<64x1xbf16> 
    %prev_up_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "prev_up_seg0_s2_q2"} : memref<64x1xbf16> 
    %r_cascade_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "r_cascade_seg0_s2_q2"} : memref<64x1xbf16> 
    %r_local_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "r_local_seg0_s2_q2"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s2_q2 = aie.buffer(%tile_2_4) {sym_name = "tmp_sp_seg0_s2_q2"} : memref<64x1xbf16> 
    %lock_2_4 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_76 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_2_4_77 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_78 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %qk_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "qk_seg0_s3_q2"} : memref<64x64xbf16> 
    %q_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "q_seg0_s3_q2"} : memref<64x64xbf16> 
    %v_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "v_seg0_s3_q2"} : memref<64x64xbf16> 
    %g_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "g_seg0_s3_q2"} : memref<64x64xbf16> 
    %gp_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "gp_seg0_s3_q2"} : memref<64x64xbf16> 
    %up_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "up_seg0_s3_q2"} : memref<64x1xbf16> 
    %sp_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "sp_seg0_s3_q2"} : memref<64x1xbf16> 
    %s_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "s_seg0_s3_q2"} : memref<64x1xbf16> 
    %r_seg0_s3_q2 = aie.buffer(%tile_2_5) {sym_name = "r_seg0_s3_q2"} : memref<64x1xbf16> 
    %lock_2_5 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_79 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_2_5_80 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_81 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %qk_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "qk_seg0_s0_q3"} : memref<64x64xbf16> 
    %q_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "q_seg0_s0_q3"} : memref<64x64xbf16> 
    %v_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "v_seg0_s0_q3"} : memref<64x64xbf16> 
    %g_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "g_seg0_s0_q3"} : memref<64x64xbf16> 
    %gp_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "gp_seg0_s0_q3"} : memref<64x64xbf16> 
    %up_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "up_seg0_s0_q3"} : memref<64x1xbf16> 
    %sp_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "sp_seg0_s0_q3"} : memref<64x1xbf16> 
    %s_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "s_seg0_s0_q3"} : memref<64x1xbf16> 
    %r_seg0_s0_q3 = aie.buffer(%tile_3_2) {sym_name = "r_seg0_s0_q3"} : memref<64x1xbf16> 
    %merged_gp_seg0_q3 = aie.buffer(%tile_3_2) {sym_name = "merged_gp_seg0_q3"} : memref<64x64xbf16> 
    %merged_up_seg0_q3 = aie.buffer(%tile_3_2) {sym_name = "merged_up_seg0_q3"} : memref<64x1xbf16> 
    %merged_sp_seg0_q3 = aie.buffer(%tile_3_2) {sym_name = "merged_sp_seg0_q3"} : memref<64x1xbf16> 
    %prev_up_seg0_q3 = aie.buffer(%tile_3_2) {sym_name = "prev_up_seg0_q3"} : memref<64x1xbf16> 
    %r_cascade_seg0_q3 = aie.buffer(%tile_3_2) {sym_name = "r_cascade_seg0_q3"} : memref<64x1xbf16> 
    %r_local_seg0_q3 = aie.buffer(%tile_3_2) {sym_name = "r_local_seg0_q3"} : memref<64x1xbf16> 
    %tmp_sp_seg0_q3 = aie.buffer(%tile_3_2) {sym_name = "tmp_sp_seg0_q3"} : memref<64x1xbf16> 
    %lock_3_2 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_3_2_82 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_83 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_84 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_85 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_86 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %qk_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "qk_seg0_s1_q3"} : memref<64x64xbf16> 
    %q_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "q_seg0_s1_q3"} : memref<64x64xbf16> 
    %v_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "v_seg0_s1_q3"} : memref<64x64xbf16> 
    %g_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "g_seg0_s1_q3"} : memref<64x64xbf16> 
    %gp_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "gp_seg0_s1_q3"} : memref<64x64xbf16> 
    %up_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "up_seg0_s1_q3"} : memref<64x1xbf16> 
    %sp_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "sp_seg0_s1_q3"} : memref<64x1xbf16> 
    %s_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "s_seg0_s1_q3"} : memref<64x1xbf16> 
    %r_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "r_seg0_s1_q3"} : memref<64x1xbf16> 
    %merged_gp_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "merged_gp_seg0_s1_q3"} : memref<64x64xbf16> 
    %merged_up_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "merged_up_seg0_s1_q3"} : memref<64x1xbf16> 
    %merged_sp_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "merged_sp_seg0_s1_q3"} : memref<64x1xbf16> 
    %prev_up_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "prev_up_seg0_s1_q3"} : memref<64x1xbf16> 
    %r_cascade_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "r_cascade_seg0_s1_q3"} : memref<64x1xbf16> 
    %r_local_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "r_local_seg0_s1_q3"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s1_q3 = aie.buffer(%tile_3_3) {sym_name = "tmp_sp_seg0_s1_q3"} : memref<64x1xbf16> 
    %lock_3_3 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_87 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_3_3_88 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_89 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %qk_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "qk_seg0_s2_q3"} : memref<64x64xbf16> 
    %q_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "q_seg0_s2_q3"} : memref<64x64xbf16> 
    %v_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "v_seg0_s2_q3"} : memref<64x64xbf16> 
    %g_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "g_seg0_s2_q3"} : memref<64x64xbf16> 
    %gp_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "gp_seg0_s2_q3"} : memref<64x64xbf16> 
    %up_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "up_seg0_s2_q3"} : memref<64x1xbf16> 
    %sp_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "sp_seg0_s2_q3"} : memref<64x1xbf16> 
    %s_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "s_seg0_s2_q3"} : memref<64x1xbf16> 
    %r_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "r_seg0_s2_q3"} : memref<64x1xbf16> 
    %merged_gp_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "merged_gp_seg0_s2_q3"} : memref<64x64xbf16> 
    %merged_up_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "merged_up_seg0_s2_q3"} : memref<64x1xbf16> 
    %merged_sp_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "merged_sp_seg0_s2_q3"} : memref<64x1xbf16> 
    %prev_up_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "prev_up_seg0_s2_q3"} : memref<64x1xbf16> 
    %r_cascade_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "r_cascade_seg0_s2_q3"} : memref<64x1xbf16> 
    %r_local_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "r_local_seg0_s2_q3"} : memref<64x1xbf16> 
    %tmp_sp_seg0_s2_q3 = aie.buffer(%tile_3_4) {sym_name = "tmp_sp_seg0_s2_q3"} : memref<64x1xbf16> 
    %lock_3_4 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_90 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_3_4_91 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_92 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %qk_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "qk_seg0_s3_q3"} : memref<64x64xbf16> 
    %q_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "q_seg0_s3_q3"} : memref<64x64xbf16> 
    %v_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "v_seg0_s3_q3"} : memref<64x64xbf16> 
    %g_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "g_seg0_s3_q3"} : memref<64x64xbf16> 
    %gp_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "gp_seg0_s3_q3"} : memref<64x64xbf16> 
    %up_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "up_seg0_s3_q3"} : memref<64x1xbf16> 
    %sp_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "sp_seg0_s3_q3"} : memref<64x1xbf16> 
    %s_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "s_seg0_s3_q3"} : memref<64x1xbf16> 
    %r_seg0_s3_q3 = aie.buffer(%tile_3_5) {sym_name = "r_seg0_s3_q3"} : memref<64x1xbf16> 
    %lock_3_5 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_93 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %lock_3_5_94 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_95 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %qk_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "qk_seg1_s0_q0"} : memref<64x64xbf16> 
    %q_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "q_seg1_s0_q0"} : memref<64x64xbf16> 
    %v_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "v_seg1_s0_q0"} : memref<64x64xbf16> 
    %g_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "g_seg1_s0_q0"} : memref<64x64xbf16> 
    %gp_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "gp_seg1_s0_q0"} : memref<64x64xbf16> 
    %up_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "up_seg1_s0_q0"} : memref<64x1xbf16> 
    %sp_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "sp_seg1_s0_q0"} : memref<64x1xbf16> 
    %s_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "s_seg1_s0_q0"} : memref<64x1xbf16> 
    %r_seg1_s0_q0 = aie.buffer(%tile_4_2) {sym_name = "r_seg1_s0_q0"} : memref<64x1xbf16> 
    %merged_gp_seg1_q0 = aie.buffer(%tile_4_2) {sym_name = "merged_gp_seg1_q0"} : memref<64x64xbf16> 
    %merged_up_seg1_q0 = aie.buffer(%tile_4_2) {sym_name = "merged_up_seg1_q0"} : memref<64x1xbf16> 
    %merged_sp_seg1_q0 = aie.buffer(%tile_4_2) {sym_name = "merged_sp_seg1_q0"} : memref<64x1xbf16> 
    %prev_up_seg1_q0 = aie.buffer(%tile_4_2) {sym_name = "prev_up_seg1_q0"} : memref<64x1xbf16> 
    %r_cascade_seg1_q0 = aie.buffer(%tile_4_2) {sym_name = "r_cascade_seg1_q0"} : memref<64x1xbf16> 
    %r_local_seg1_q0 = aie.buffer(%tile_4_2) {sym_name = "r_local_seg1_q0"} : memref<64x1xbf16> 
    %tmp_sp_seg1_q0 = aie.buffer(%tile_4_2) {sym_name = "tmp_sp_seg1_q0"} : memref<64x1xbf16> 
    %lock_4_2 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_4_2_96 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_97 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_98 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_99 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_100 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %qk_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "qk_seg1_s1_q0"} : memref<64x64xbf16> 
    %q_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "q_seg1_s1_q0"} : memref<64x64xbf16> 
    %v_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "v_seg1_s1_q0"} : memref<64x64xbf16> 
    %g_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "g_seg1_s1_q0"} : memref<64x64xbf16> 
    %gp_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "gp_seg1_s1_q0"} : memref<64x64xbf16> 
    %up_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "up_seg1_s1_q0"} : memref<64x1xbf16> 
    %sp_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "sp_seg1_s1_q0"} : memref<64x1xbf16> 
    %s_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "s_seg1_s1_q0"} : memref<64x1xbf16> 
    %r_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "r_seg1_s1_q0"} : memref<64x1xbf16> 
    %merged_gp_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "merged_gp_seg1_s1_q0"} : memref<64x64xbf16> 
    %merged_up_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "merged_up_seg1_s1_q0"} : memref<64x1xbf16> 
    %merged_sp_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "merged_sp_seg1_s1_q0"} : memref<64x1xbf16> 
    %prev_up_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "prev_up_seg1_s1_q0"} : memref<64x1xbf16> 
    %r_cascade_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "r_cascade_seg1_s1_q0"} : memref<64x1xbf16> 
    %r_local_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "r_local_seg1_s1_q0"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s1_q0 = aie.buffer(%tile_4_3) {sym_name = "tmp_sp_seg1_s1_q0"} : memref<64x1xbf16> 
    %lock_4_3 = aie.lock(%tile_4_3, 1) {init = 1 : i32}
    %lock_4_3_101 = aie.lock(%tile_4_3, 0) {init = 0 : i32}
    %lock_4_3_102 = aie.lock(%tile_4_3, 3) {init = 1 : i32}
    %lock_4_3_103 = aie.lock(%tile_4_3, 2) {init = 0 : i32}
    %qk_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "qk_seg1_s2_q0"} : memref<64x64xbf16> 
    %q_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "q_seg1_s2_q0"} : memref<64x64xbf16> 
    %v_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "v_seg1_s2_q0"} : memref<64x64xbf16> 
    %g_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "g_seg1_s2_q0"} : memref<64x64xbf16> 
    %gp_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "gp_seg1_s2_q0"} : memref<64x64xbf16> 
    %up_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "up_seg1_s2_q0"} : memref<64x1xbf16> 
    %sp_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "sp_seg1_s2_q0"} : memref<64x1xbf16> 
    %s_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "s_seg1_s2_q0"} : memref<64x1xbf16> 
    %r_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "r_seg1_s2_q0"} : memref<64x1xbf16> 
    %merged_gp_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "merged_gp_seg1_s2_q0"} : memref<64x64xbf16> 
    %merged_up_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "merged_up_seg1_s2_q0"} : memref<64x1xbf16> 
    %merged_sp_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "merged_sp_seg1_s2_q0"} : memref<64x1xbf16> 
    %prev_up_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "prev_up_seg1_s2_q0"} : memref<64x1xbf16> 
    %r_cascade_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "r_cascade_seg1_s2_q0"} : memref<64x1xbf16> 
    %r_local_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "r_local_seg1_s2_q0"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s2_q0 = aie.buffer(%tile_4_4) {sym_name = "tmp_sp_seg1_s2_q0"} : memref<64x1xbf16> 
    %lock_4_4 = aie.lock(%tile_4_4, 1) {init = 1 : i32}
    %lock_4_4_104 = aie.lock(%tile_4_4, 0) {init = 0 : i32}
    %lock_4_4_105 = aie.lock(%tile_4_4, 3) {init = 1 : i32}
    %lock_4_4_106 = aie.lock(%tile_4_4, 2) {init = 0 : i32}
    %qk_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "qk_seg1_s3_q0"} : memref<64x64xbf16> 
    %q_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "q_seg1_s3_q0"} : memref<64x64xbf16> 
    %v_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "v_seg1_s3_q0"} : memref<64x64xbf16> 
    %g_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "g_seg1_s3_q0"} : memref<64x64xbf16> 
    %gp_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "gp_seg1_s3_q0"} : memref<64x64xbf16> 
    %up_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "up_seg1_s3_q0"} : memref<64x1xbf16> 
    %sp_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "sp_seg1_s3_q0"} : memref<64x1xbf16> 
    %s_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "s_seg1_s3_q0"} : memref<64x1xbf16> 
    %r_seg1_s3_q0 = aie.buffer(%tile_4_5) {sym_name = "r_seg1_s3_q0"} : memref<64x1xbf16> 
    %lock_4_5 = aie.lock(%tile_4_5, 1) {init = 1 : i32}
    %lock_4_5_107 = aie.lock(%tile_4_5, 0) {init = 0 : i32}
    %lock_4_5_108 = aie.lock(%tile_4_5, 3) {init = 1 : i32}
    %lock_4_5_109 = aie.lock(%tile_4_5, 2) {init = 0 : i32}
    %qk_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "qk_seg1_s0_q1"} : memref<64x64xbf16> 
    %q_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "q_seg1_s0_q1"} : memref<64x64xbf16> 
    %v_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "v_seg1_s0_q1"} : memref<64x64xbf16> 
    %g_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "g_seg1_s0_q1"} : memref<64x64xbf16> 
    %gp_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "gp_seg1_s0_q1"} : memref<64x64xbf16> 
    %up_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "up_seg1_s0_q1"} : memref<64x1xbf16> 
    %sp_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "sp_seg1_s0_q1"} : memref<64x1xbf16> 
    %s_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "s_seg1_s0_q1"} : memref<64x1xbf16> 
    %r_seg1_s0_q1 = aie.buffer(%tile_5_2) {sym_name = "r_seg1_s0_q1"} : memref<64x1xbf16> 
    %merged_gp_seg1_q1 = aie.buffer(%tile_5_2) {sym_name = "merged_gp_seg1_q1"} : memref<64x64xbf16> 
    %merged_up_seg1_q1 = aie.buffer(%tile_5_2) {sym_name = "merged_up_seg1_q1"} : memref<64x1xbf16> 
    %merged_sp_seg1_q1 = aie.buffer(%tile_5_2) {sym_name = "merged_sp_seg1_q1"} : memref<64x1xbf16> 
    %prev_up_seg1_q1 = aie.buffer(%tile_5_2) {sym_name = "prev_up_seg1_q1"} : memref<64x1xbf16> 
    %r_cascade_seg1_q1 = aie.buffer(%tile_5_2) {sym_name = "r_cascade_seg1_q1"} : memref<64x1xbf16> 
    %r_local_seg1_q1 = aie.buffer(%tile_5_2) {sym_name = "r_local_seg1_q1"} : memref<64x1xbf16> 
    %tmp_sp_seg1_q1 = aie.buffer(%tile_5_2) {sym_name = "tmp_sp_seg1_q1"} : memref<64x1xbf16> 
    %lock_5_2 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_5_2_110 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_111 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_112 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_113 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_114 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %qk_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "qk_seg1_s1_q1"} : memref<64x64xbf16> 
    %q_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "q_seg1_s1_q1"} : memref<64x64xbf16> 
    %v_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "v_seg1_s1_q1"} : memref<64x64xbf16> 
    %g_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "g_seg1_s1_q1"} : memref<64x64xbf16> 
    %gp_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "gp_seg1_s1_q1"} : memref<64x64xbf16> 
    %up_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "up_seg1_s1_q1"} : memref<64x1xbf16> 
    %sp_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "sp_seg1_s1_q1"} : memref<64x1xbf16> 
    %s_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "s_seg1_s1_q1"} : memref<64x1xbf16> 
    %r_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "r_seg1_s1_q1"} : memref<64x1xbf16> 
    %merged_gp_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "merged_gp_seg1_s1_q1"} : memref<64x64xbf16> 
    %merged_up_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "merged_up_seg1_s1_q1"} : memref<64x1xbf16> 
    %merged_sp_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "merged_sp_seg1_s1_q1"} : memref<64x1xbf16> 
    %prev_up_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "prev_up_seg1_s1_q1"} : memref<64x1xbf16> 
    %r_cascade_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "r_cascade_seg1_s1_q1"} : memref<64x1xbf16> 
    %r_local_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "r_local_seg1_s1_q1"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s1_q1 = aie.buffer(%tile_5_3) {sym_name = "tmp_sp_seg1_s1_q1"} : memref<64x1xbf16> 
    %lock_5_3 = aie.lock(%tile_5_3, 1) {init = 1 : i32}
    %lock_5_3_115 = aie.lock(%tile_5_3, 0) {init = 0 : i32}
    %lock_5_3_116 = aie.lock(%tile_5_3, 3) {init = 1 : i32}
    %lock_5_3_117 = aie.lock(%tile_5_3, 2) {init = 0 : i32}
    %qk_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "qk_seg1_s2_q1"} : memref<64x64xbf16> 
    %q_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "q_seg1_s2_q1"} : memref<64x64xbf16> 
    %v_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "v_seg1_s2_q1"} : memref<64x64xbf16> 
    %g_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "g_seg1_s2_q1"} : memref<64x64xbf16> 
    %gp_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "gp_seg1_s2_q1"} : memref<64x64xbf16> 
    %up_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "up_seg1_s2_q1"} : memref<64x1xbf16> 
    %sp_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "sp_seg1_s2_q1"} : memref<64x1xbf16> 
    %s_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "s_seg1_s2_q1"} : memref<64x1xbf16> 
    %r_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "r_seg1_s2_q1"} : memref<64x1xbf16> 
    %merged_gp_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "merged_gp_seg1_s2_q1"} : memref<64x64xbf16> 
    %merged_up_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "merged_up_seg1_s2_q1"} : memref<64x1xbf16> 
    %merged_sp_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "merged_sp_seg1_s2_q1"} : memref<64x1xbf16> 
    %prev_up_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "prev_up_seg1_s2_q1"} : memref<64x1xbf16> 
    %r_cascade_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "r_cascade_seg1_s2_q1"} : memref<64x1xbf16> 
    %r_local_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "r_local_seg1_s2_q1"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s2_q1 = aie.buffer(%tile_5_4) {sym_name = "tmp_sp_seg1_s2_q1"} : memref<64x1xbf16> 
    %lock_5_4 = aie.lock(%tile_5_4, 1) {init = 1 : i32}
    %lock_5_4_118 = aie.lock(%tile_5_4, 0) {init = 0 : i32}
    %lock_5_4_119 = aie.lock(%tile_5_4, 3) {init = 1 : i32}
    %lock_5_4_120 = aie.lock(%tile_5_4, 2) {init = 0 : i32}
    %qk_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "qk_seg1_s3_q1"} : memref<64x64xbf16> 
    %q_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "q_seg1_s3_q1"} : memref<64x64xbf16> 
    %v_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "v_seg1_s3_q1"} : memref<64x64xbf16> 
    %g_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "g_seg1_s3_q1"} : memref<64x64xbf16> 
    %gp_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "gp_seg1_s3_q1"} : memref<64x64xbf16> 
    %up_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "up_seg1_s3_q1"} : memref<64x1xbf16> 
    %sp_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "sp_seg1_s3_q1"} : memref<64x1xbf16> 
    %s_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "s_seg1_s3_q1"} : memref<64x1xbf16> 
    %r_seg1_s3_q1 = aie.buffer(%tile_5_5) {sym_name = "r_seg1_s3_q1"} : memref<64x1xbf16> 
    %lock_5_5 = aie.lock(%tile_5_5, 1) {init = 1 : i32}
    %lock_5_5_121 = aie.lock(%tile_5_5, 0) {init = 0 : i32}
    %lock_5_5_122 = aie.lock(%tile_5_5, 3) {init = 1 : i32}
    %lock_5_5_123 = aie.lock(%tile_5_5, 2) {init = 0 : i32}
    %qk_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "qk_seg1_s0_q2"} : memref<64x64xbf16> 
    %q_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "q_seg1_s0_q2"} : memref<64x64xbf16> 
    %v_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "v_seg1_s0_q2"} : memref<64x64xbf16> 
    %g_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "g_seg1_s0_q2"} : memref<64x64xbf16> 
    %gp_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "gp_seg1_s0_q2"} : memref<64x64xbf16> 
    %up_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "up_seg1_s0_q2"} : memref<64x1xbf16> 
    %sp_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "sp_seg1_s0_q2"} : memref<64x1xbf16> 
    %s_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "s_seg1_s0_q2"} : memref<64x1xbf16> 
    %r_seg1_s0_q2 = aie.buffer(%tile_6_2) {sym_name = "r_seg1_s0_q2"} : memref<64x1xbf16> 
    %merged_gp_seg1_q2 = aie.buffer(%tile_6_2) {sym_name = "merged_gp_seg1_q2"} : memref<64x64xbf16> 
    %merged_up_seg1_q2 = aie.buffer(%tile_6_2) {sym_name = "merged_up_seg1_q2"} : memref<64x1xbf16> 
    %merged_sp_seg1_q2 = aie.buffer(%tile_6_2) {sym_name = "merged_sp_seg1_q2"} : memref<64x1xbf16> 
    %prev_up_seg1_q2 = aie.buffer(%tile_6_2) {sym_name = "prev_up_seg1_q2"} : memref<64x1xbf16> 
    %r_cascade_seg1_q2 = aie.buffer(%tile_6_2) {sym_name = "r_cascade_seg1_q2"} : memref<64x1xbf16> 
    %r_local_seg1_q2 = aie.buffer(%tile_6_2) {sym_name = "r_local_seg1_q2"} : memref<64x1xbf16> 
    %tmp_sp_seg1_q2 = aie.buffer(%tile_6_2) {sym_name = "tmp_sp_seg1_q2"} : memref<64x1xbf16> 
    %lock_6_2 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_6_2_124 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_125 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_126 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_127 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_128 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %qk_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "qk_seg1_s1_q2"} : memref<64x64xbf16> 
    %q_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "q_seg1_s1_q2"} : memref<64x64xbf16> 
    %v_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "v_seg1_s1_q2"} : memref<64x64xbf16> 
    %g_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "g_seg1_s1_q2"} : memref<64x64xbf16> 
    %gp_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "gp_seg1_s1_q2"} : memref<64x64xbf16> 
    %up_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "up_seg1_s1_q2"} : memref<64x1xbf16> 
    %sp_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "sp_seg1_s1_q2"} : memref<64x1xbf16> 
    %s_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "s_seg1_s1_q2"} : memref<64x1xbf16> 
    %r_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "r_seg1_s1_q2"} : memref<64x1xbf16> 
    %merged_gp_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "merged_gp_seg1_s1_q2"} : memref<64x64xbf16> 
    %merged_up_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "merged_up_seg1_s1_q2"} : memref<64x1xbf16> 
    %merged_sp_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "merged_sp_seg1_s1_q2"} : memref<64x1xbf16> 
    %prev_up_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "prev_up_seg1_s1_q2"} : memref<64x1xbf16> 
    %r_cascade_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "r_cascade_seg1_s1_q2"} : memref<64x1xbf16> 
    %r_local_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "r_local_seg1_s1_q2"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s1_q2 = aie.buffer(%tile_6_3) {sym_name = "tmp_sp_seg1_s1_q2"} : memref<64x1xbf16> 
    %lock_6_3 = aie.lock(%tile_6_3, 1) {init = 1 : i32}
    %lock_6_3_129 = aie.lock(%tile_6_3, 0) {init = 0 : i32}
    %lock_6_3_130 = aie.lock(%tile_6_3, 3) {init = 1 : i32}
    %lock_6_3_131 = aie.lock(%tile_6_3, 2) {init = 0 : i32}
    %qk_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "qk_seg1_s2_q2"} : memref<64x64xbf16> 
    %q_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "q_seg1_s2_q2"} : memref<64x64xbf16> 
    %v_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "v_seg1_s2_q2"} : memref<64x64xbf16> 
    %g_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "g_seg1_s2_q2"} : memref<64x64xbf16> 
    %gp_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "gp_seg1_s2_q2"} : memref<64x64xbf16> 
    %up_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "up_seg1_s2_q2"} : memref<64x1xbf16> 
    %sp_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "sp_seg1_s2_q2"} : memref<64x1xbf16> 
    %s_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "s_seg1_s2_q2"} : memref<64x1xbf16> 
    %r_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "r_seg1_s2_q2"} : memref<64x1xbf16> 
    %merged_gp_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "merged_gp_seg1_s2_q2"} : memref<64x64xbf16> 
    %merged_up_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "merged_up_seg1_s2_q2"} : memref<64x1xbf16> 
    %merged_sp_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "merged_sp_seg1_s2_q2"} : memref<64x1xbf16> 
    %prev_up_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "prev_up_seg1_s2_q2"} : memref<64x1xbf16> 
    %r_cascade_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "r_cascade_seg1_s2_q2"} : memref<64x1xbf16> 
    %r_local_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "r_local_seg1_s2_q2"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s2_q2 = aie.buffer(%tile_6_4) {sym_name = "tmp_sp_seg1_s2_q2"} : memref<64x1xbf16> 
    %lock_6_4 = aie.lock(%tile_6_4, 1) {init = 1 : i32}
    %lock_6_4_132 = aie.lock(%tile_6_4, 0) {init = 0 : i32}
    %lock_6_4_133 = aie.lock(%tile_6_4, 3) {init = 1 : i32}
    %lock_6_4_134 = aie.lock(%tile_6_4, 2) {init = 0 : i32}
    %qk_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "qk_seg1_s3_q2"} : memref<64x64xbf16> 
    %q_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "q_seg1_s3_q2"} : memref<64x64xbf16> 
    %v_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "v_seg1_s3_q2"} : memref<64x64xbf16> 
    %g_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "g_seg1_s3_q2"} : memref<64x64xbf16> 
    %gp_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "gp_seg1_s3_q2"} : memref<64x64xbf16> 
    %up_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "up_seg1_s3_q2"} : memref<64x1xbf16> 
    %sp_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "sp_seg1_s3_q2"} : memref<64x1xbf16> 
    %s_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "s_seg1_s3_q2"} : memref<64x1xbf16> 
    %r_seg1_s3_q2 = aie.buffer(%tile_6_5) {sym_name = "r_seg1_s3_q2"} : memref<64x1xbf16> 
    %lock_6_5 = aie.lock(%tile_6_5, 1) {init = 1 : i32}
    %lock_6_5_135 = aie.lock(%tile_6_5, 0) {init = 0 : i32}
    %lock_6_5_136 = aie.lock(%tile_6_5, 3) {init = 1 : i32}
    %lock_6_5_137 = aie.lock(%tile_6_5, 2) {init = 0 : i32}
    %qk_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "qk_seg1_s0_q3"} : memref<64x64xbf16> 
    %q_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "q_seg1_s0_q3"} : memref<64x64xbf16> 
    %v_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "v_seg1_s0_q3"} : memref<64x64xbf16> 
    %g_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "g_seg1_s0_q3"} : memref<64x64xbf16> 
    %gp_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "gp_seg1_s0_q3"} : memref<64x64xbf16> 
    %up_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "up_seg1_s0_q3"} : memref<64x1xbf16> 
    %sp_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "sp_seg1_s0_q3"} : memref<64x1xbf16> 
    %s_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "s_seg1_s0_q3"} : memref<64x1xbf16> 
    %r_seg1_s0_q3 = aie.buffer(%tile_7_2) {sym_name = "r_seg1_s0_q3"} : memref<64x1xbf16> 
    %merged_gp_seg1_q3 = aie.buffer(%tile_7_2) {sym_name = "merged_gp_seg1_q3"} : memref<64x64xbf16> 
    %merged_up_seg1_q3 = aie.buffer(%tile_7_2) {sym_name = "merged_up_seg1_q3"} : memref<64x1xbf16> 
    %merged_sp_seg1_q3 = aie.buffer(%tile_7_2) {sym_name = "merged_sp_seg1_q3"} : memref<64x1xbf16> 
    %prev_up_seg1_q3 = aie.buffer(%tile_7_2) {sym_name = "prev_up_seg1_q3"} : memref<64x1xbf16> 
    %r_cascade_seg1_q3 = aie.buffer(%tile_7_2) {sym_name = "r_cascade_seg1_q3"} : memref<64x1xbf16> 
    %r_local_seg1_q3 = aie.buffer(%tile_7_2) {sym_name = "r_local_seg1_q3"} : memref<64x1xbf16> 
    %tmp_sp_seg1_q3 = aie.buffer(%tile_7_2) {sym_name = "tmp_sp_seg1_q3"} : memref<64x1xbf16> 
    %lock_7_2 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %lock_7_2_138 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_139 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_140 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_141 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_142 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %qk_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "qk_seg1_s1_q3"} : memref<64x64xbf16> 
    %q_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "q_seg1_s1_q3"} : memref<64x64xbf16> 
    %v_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "v_seg1_s1_q3"} : memref<64x64xbf16> 
    %g_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "g_seg1_s1_q3"} : memref<64x64xbf16> 
    %gp_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "gp_seg1_s1_q3"} : memref<64x64xbf16> 
    %up_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "up_seg1_s1_q3"} : memref<64x1xbf16> 
    %sp_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "sp_seg1_s1_q3"} : memref<64x1xbf16> 
    %s_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "s_seg1_s1_q3"} : memref<64x1xbf16> 
    %r_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "r_seg1_s1_q3"} : memref<64x1xbf16> 
    %merged_gp_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "merged_gp_seg1_s1_q3"} : memref<64x64xbf16> 
    %merged_up_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "merged_up_seg1_s1_q3"} : memref<64x1xbf16> 
    %merged_sp_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "merged_sp_seg1_s1_q3"} : memref<64x1xbf16> 
    %prev_up_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "prev_up_seg1_s1_q3"} : memref<64x1xbf16> 
    %r_cascade_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "r_cascade_seg1_s1_q3"} : memref<64x1xbf16> 
    %r_local_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "r_local_seg1_s1_q3"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s1_q3 = aie.buffer(%tile_7_3) {sym_name = "tmp_sp_seg1_s1_q3"} : memref<64x1xbf16> 
    %lock_7_3 = aie.lock(%tile_7_3, 1) {init = 1 : i32}
    %lock_7_3_143 = aie.lock(%tile_7_3, 0) {init = 0 : i32}
    %lock_7_3_144 = aie.lock(%tile_7_3, 3) {init = 1 : i32}
    %lock_7_3_145 = aie.lock(%tile_7_3, 2) {init = 0 : i32}
    %qk_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "qk_seg1_s2_q3"} : memref<64x64xbf16> 
    %q_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "q_seg1_s2_q3"} : memref<64x64xbf16> 
    %v_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "v_seg1_s2_q3"} : memref<64x64xbf16> 
    %g_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "g_seg1_s2_q3"} : memref<64x64xbf16> 
    %gp_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "gp_seg1_s2_q3"} : memref<64x64xbf16> 
    %up_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "up_seg1_s2_q3"} : memref<64x1xbf16> 
    %sp_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "sp_seg1_s2_q3"} : memref<64x1xbf16> 
    %s_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "s_seg1_s2_q3"} : memref<64x1xbf16> 
    %r_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "r_seg1_s2_q3"} : memref<64x1xbf16> 
    %merged_gp_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "merged_gp_seg1_s2_q3"} : memref<64x64xbf16> 
    %merged_up_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "merged_up_seg1_s2_q3"} : memref<64x1xbf16> 
    %merged_sp_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "merged_sp_seg1_s2_q3"} : memref<64x1xbf16> 
    %prev_up_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "prev_up_seg1_s2_q3"} : memref<64x1xbf16> 
    %r_cascade_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "r_cascade_seg1_s2_q3"} : memref<64x1xbf16> 
    %r_local_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "r_local_seg1_s2_q3"} : memref<64x1xbf16> 
    %tmp_sp_seg1_s2_q3 = aie.buffer(%tile_7_4) {sym_name = "tmp_sp_seg1_s2_q3"} : memref<64x1xbf16> 
    %lock_7_4 = aie.lock(%tile_7_4, 1) {init = 1 : i32}
    %lock_7_4_146 = aie.lock(%tile_7_4, 0) {init = 0 : i32}
    %lock_7_4_147 = aie.lock(%tile_7_4, 3) {init = 1 : i32}
    %lock_7_4_148 = aie.lock(%tile_7_4, 2) {init = 0 : i32}
    %qk_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "qk_seg1_s3_q3"} : memref<64x64xbf16> 
    %q_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "q_seg1_s3_q3"} : memref<64x64xbf16> 
    %v_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "v_seg1_s3_q3"} : memref<64x64xbf16> 
    %g_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "g_seg1_s3_q3"} : memref<64x64xbf16> 
    %gp_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "gp_seg1_s3_q3"} : memref<64x64xbf16> 
    %up_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "up_seg1_s3_q3"} : memref<64x1xbf16> 
    %sp_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "sp_seg1_s3_q3"} : memref<64x1xbf16> 
    %s_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "s_seg1_s3_q3"} : memref<64x1xbf16> 
    %r_seg1_s3_q3 = aie.buffer(%tile_7_5) {sym_name = "r_seg1_s3_q3"} : memref<64x1xbf16> 
    %lock_7_5 = aie.lock(%tile_7_5, 1) {init = 1 : i32}
    %lock_7_5_149 = aie.lock(%tile_7_5, 0) {init = 0 : i32}
    %lock_7_5_150 = aie.lock(%tile_7_5, 3) {init = 1 : i32}
    %lock_7_5_151 = aie.lock(%tile_7_5, 2) {init = 0 : i32}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col0 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_2, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col0 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_4, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_0_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1_3, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_1_5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col1 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_1_7, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_1_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col1 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_1_9, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_1_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_1_6, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_1_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_1_8, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_1_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col2 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_1_12, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_2_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col2 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_1_14, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_2_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_1_11, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_2_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_1_13, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_2_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_1_15, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_3_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col3 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_1_17, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_3_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col3 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_1_19, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_3_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_1_16, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_3_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_1_18, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_3_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col4 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_1_20, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_4_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col4 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_4_1_22, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_4_1_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col4 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_4_1_24, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_4_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col4 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_1_21, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_4_1_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col4 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_1_23, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_4_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col4 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col5 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_1_25, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_5_1_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col5 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_5_1_27, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_5_1_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col5 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_5_1_29, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_5_1_27, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col5 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_1_26, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_5_1_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col5 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_1_28, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_5_1_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col5 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col6 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_1_30, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_6_1_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col6 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_6_1_32, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_6_1_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col6 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_6_1_34, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_6_1_32, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col6 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_1_31, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_6_1_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col6 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_1_33, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_6_1_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col6 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col7 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_1_35, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_7_1_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col7 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_7_1_37, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_7_1_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col7 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_7_1_39, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%lock_7_1_37, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_l2_col7 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_1_36, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%lock_7_1_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_l2_col7 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_1_38, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%lock_7_1_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_l2_col7 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg0_q0 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_2_40, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_2_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s0_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_2_42, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s0_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_2_44, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_0_2_40, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s0_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s0_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s0_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_0_2_42, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s0_q0, %q_seg0_s0_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_0_2_41, Release, 1)
        aie.use_lock(%lock_0_2_42, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_41, Release, 1)
        aie.use_lock(%lock_0_2_42, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_41, Release, 1)
        aie.use_lock(%lock_0_2_42, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_41, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s0_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_2_42, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s0_q0, %qk_seg0_s0_q0, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_2_41, Release, 1)
          aie.use_lock(%lock_0_2_44, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s0_q0, %s_seg0_s0_q0, %r_seg0_s0_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s0_q0, %gp_seg0_s0_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s0_q0, %gp_seg0_s0_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s0_q0, %r_seg0_s0_q0, %s_seg0_s0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s0_q0, %sp_seg0_s0_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_0_2_43, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s0_q0, %prev_up_seg0_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_q0, %up_seg0_s0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_q0, %up_seg0_s0_q0, %r_cascade_seg0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_q0, %up_seg0_s0_q0, %r_local_seg0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_q0, %merged_gp_seg0_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_q0, %gp_seg0_s0_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s0_q0, %merged_gp_seg0_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_q0) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_q0, %r_cascade_seg0_q0, %tmp_sp_seg0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s0_q0, %r_local_seg0_q0, %tmp_sp_seg0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_q0, %merged_sp_seg0_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg0_q0, %merged_gp_seg0_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_0_2, Release, 1)
      }
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s1_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_3_46, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s1_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_3_47, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s1_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s1_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s1_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_0_3_45, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s1_q0, %q_seg0_s1_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
        aie.use_lock(%lock_0_3_45, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3, Release, 1)
        aie.use_lock(%lock_0_3_45, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3, Release, 1)
        aie.use_lock(%lock_0_3_45, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s1_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_3_45, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s1_q0, %qk_seg0_s1_q0, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_3, Release, 1)
          aie.use_lock(%lock_0_3_47, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s1_q0, %s_seg0_s1_q0, %r_seg0_s1_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s1_q0, %gp_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s1_q0, %gp_seg0_s1_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s1_q0, %r_seg0_s1_q0, %s_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s1_q0, %sp_seg0_s1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_0_3_46, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s1_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s1_q0, %prev_up_seg0_s1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s1_q0, %up_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s1_q0, %up_seg0_s1_q0, %r_cascade_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s1_q0, %up_seg0_s1_q0, %r_local_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s1_q0, %merged_gp_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s1_q0, %gp_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s1_q0, %merged_gp_seg0_s1_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s1_q0) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s1_q0, %r_cascade_seg0_s1_q0, %tmp_sp_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s1_q0, %r_local_seg0_s1_q0, %tmp_sp_seg0_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s1_q0, %merged_sp_seg0_s1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s1_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s2_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_4_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_4_49, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s2_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_4_50, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s2_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s2_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s2_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_0_4_48, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s2_q0, %q_seg0_s2_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_0_4, Release, 1)
        aie.use_lock(%lock_0_4_48, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4, Release, 1)
        aie.use_lock(%lock_0_4_48, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4, Release, 1)
        aie.use_lock(%lock_0_4_48, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s2_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_4_48, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s2_q0, %qk_seg0_s2_q0, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_4, Release, 1)
          aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s2_q0, %s_seg0_s2_q0, %r_seg0_s2_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s2_q0, %gp_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s2_q0, %gp_seg0_s2_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s2_q0, %r_seg0_s2_q0, %s_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s2_q0, %sp_seg0_s2_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_0_4_49, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s2_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s2_q0, %prev_up_seg0_s2_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s2_q0, %up_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s2_q0, %up_seg0_s2_q0, %r_cascade_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s2_q0, %up_seg0_s2_q0, %r_local_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s2_q0, %merged_gp_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s2_q0, %gp_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s2_q0, %merged_gp_seg0_s2_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s2_q0) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s2_q0, %r_cascade_seg0_s2_q0, %tmp_sp_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s2_q0, %r_local_seg0_s2_q0, %tmp_sp_seg0_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s2_q0, %merged_sp_seg0_s2_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s2_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s3_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_5_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_0_5_52, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s3_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_5_53, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s3_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s3_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s3_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_0_5_51, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s3_q0, %q_seg0_s3_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_0_5, Release, 1)
        aie.use_lock(%lock_0_5_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5, Release, 1)
        aie.use_lock(%lock_0_5_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5, Release, 1)
        aie.use_lock(%lock_0_5_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s3_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_5_51, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s3_q0, %qk_seg0_s3_q0, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_0_5, Release, 1)
          aie.use_lock(%lock_0_5_53, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s3_q0, %s_seg0_s3_q0, %r_seg0_s3_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s3_q0, %gp_seg0_s3_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s3_q0, %gp_seg0_s3_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s3_q0, %r_seg0_s3_q0, %s_seg0_s3_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s3_q0, %sp_seg0_s3_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_0_5_52, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg0_s3_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg0_s3_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg0_s3_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg0_q1 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_2_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s0_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_2_56, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_1_2_57, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s0_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_2_58, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_1_2_54, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s0_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s0_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s0_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_1_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_55, Release, 1)
        aie.use_lock(%lock_1_2_56, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s0_q1, %q_seg0_s0_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_1_2_55, Release, 1)
        aie.use_lock(%lock_1_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_55, Release, 1)
        aie.use_lock(%lock_1_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_55, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s0_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_2_56, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s0_q1, %qk_seg0_s0_q1, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_2_55, Release, 1)
          aie.use_lock(%lock_1_2_58, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s0_q1, %s_seg0_s0_q1, %r_seg0_s0_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s0_q1, %gp_seg0_s0_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s0_q1, %gp_seg0_s0_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s0_q1, %r_seg0_s0_q1, %s_seg0_s0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s0_q1, %sp_seg0_s0_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_1_2_57, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s0_q1, %prev_up_seg0_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_q1, %up_seg0_s0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_q1, %up_seg0_s0_q1, %r_cascade_seg0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_q1, %up_seg0_s0_q1, %r_local_seg0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_q1, %merged_gp_seg0_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_q1, %gp_seg0_s0_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s0_q1, %merged_gp_seg0_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_q1) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_q1, %r_cascade_seg0_q1, %tmp_sp_seg0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s0_q1, %r_local_seg0_q1, %tmp_sp_seg0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_q1, %merged_sp_seg0_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg0_q1, %merged_gp_seg0_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_1_2, Release, 1)
      }
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s1_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_3_59, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_3_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s1_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_3_61, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s1_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s1_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s1_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_1_3_59, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3, Release, 1)
        aie.use_lock(%lock_1_3_59, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s1_q1, %q_seg0_s1_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_1_3, Release, 1)
        aie.use_lock(%lock_1_3_59, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3, Release, 1)
        aie.use_lock(%lock_1_3_59, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s1_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_3_59, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s1_q1, %qk_seg0_s1_q1, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_3, Release, 1)
          aie.use_lock(%lock_1_3_61, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s1_q1, %s_seg0_s1_q1, %r_seg0_s1_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s1_q1, %gp_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s1_q1, %gp_seg0_s1_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s1_q1, %r_seg0_s1_q1, %s_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s1_q1, %sp_seg0_s1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_1_3_60, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s1_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s1_q1, %prev_up_seg0_s1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s1_q1, %up_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s1_q1, %up_seg0_s1_q1, %r_cascade_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s1_q1, %up_seg0_s1_q1, %r_local_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s1_q1, %merged_gp_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s1_q1, %gp_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s1_q1, %merged_gp_seg0_s1_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s1_q1) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s1_q1, %r_cascade_seg0_s1_q1, %tmp_sp_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s1_q1, %r_local_seg0_s1_q1, %tmp_sp_seg0_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s1_q1, %merged_sp_seg0_s1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s1_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s2_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_4_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_4_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s2_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_4_64, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s2_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s2_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s2_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_1_4_62, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4, Release, 1)
        aie.use_lock(%lock_1_4_62, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s2_q1, %q_seg0_s2_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_1_4, Release, 1)
        aie.use_lock(%lock_1_4_62, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4, Release, 1)
        aie.use_lock(%lock_1_4_62, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s2_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_4_62, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s2_q1, %qk_seg0_s2_q1, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_4, Release, 1)
          aie.use_lock(%lock_1_4_64, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s2_q1, %s_seg0_s2_q1, %r_seg0_s2_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s2_q1, %gp_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s2_q1, %gp_seg0_s2_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s2_q1, %r_seg0_s2_q1, %s_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s2_q1, %sp_seg0_s2_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_1_4_63, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s2_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s2_q1, %prev_up_seg0_s2_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s2_q1, %up_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s2_q1, %up_seg0_s2_q1, %r_cascade_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s2_q1, %up_seg0_s2_q1, %r_local_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s2_q1, %merged_gp_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s2_q1, %gp_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s2_q1, %merged_gp_seg0_s2_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s2_q1) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s2_q1, %r_cascade_seg0_s2_q1, %tmp_sp_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s2_q1, %r_local_seg0_s2_q1, %tmp_sp_seg0_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s2_q1, %merged_sp_seg0_s2_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s2_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s3_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_5_65, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_1_5_66, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s3_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s3_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s3_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s3_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5, Release, 1)
        aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s3_q1, %q_seg0_s3_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_1_5, Release, 1)
        aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5, Release, 1)
        aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s3_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s3_q1, %qk_seg0_s3_q1, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_1_5, Release, 1)
          aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s3_q1, %s_seg0_s3_q1, %r_seg0_s3_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s3_q1, %gp_seg0_s3_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s3_q1, %gp_seg0_s3_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s3_q1, %r_seg0_s3_q1, %s_seg0_s3_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s3_q1, %sp_seg0_s3_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_1_5_66, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg0_s3_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg0_s3_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg0_s3_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg0_q2 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_2_68, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_2_69, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s0_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_2_70, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_2_2_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s0_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_2_72, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_2_2_68, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s0_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s0_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s0_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_2_2_70, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_69, Release, 1)
        aie.use_lock(%lock_2_2_70, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_69, Release, 1)
        aie.use_lock(%lock_2_2_70, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s0_q2, %q_seg0_s0_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_2_2_69, Release, 1)
        aie.use_lock(%lock_2_2_70, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_69, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s0_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_2_70, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s0_q2, %qk_seg0_s0_q2, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_2_69, Release, 1)
          aie.use_lock(%lock_2_2_72, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s0_q2, %s_seg0_s0_q2, %r_seg0_s0_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s0_q2, %gp_seg0_s0_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s0_q2, %gp_seg0_s0_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s0_q2, %r_seg0_s0_q2, %s_seg0_s0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s0_q2, %sp_seg0_s0_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_2_2_71, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s0_q2, %prev_up_seg0_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_q2, %up_seg0_s0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_q2, %up_seg0_s0_q2, %r_cascade_seg0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_q2, %up_seg0_s0_q2, %r_local_seg0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_q2, %merged_gp_seg0_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_q2, %gp_seg0_s0_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s0_q2, %merged_gp_seg0_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_q2) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_q2, %r_cascade_seg0_q2, %tmp_sp_seg0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s0_q2, %r_local_seg0_q2, %tmp_sp_seg0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_q2, %merged_sp_seg0_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg0_q2, %merged_gp_seg0_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_2_2, Release, 1)
      }
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s1_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_3_73, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_3_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s1_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_3_75, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s1_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s1_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s1_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_2_3_73, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3, Release, 1)
        aie.use_lock(%lock_2_3_73, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3, Release, 1)
        aie.use_lock(%lock_2_3_73, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s1_q2, %q_seg0_s1_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_2_3, Release, 1)
        aie.use_lock(%lock_2_3_73, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s1_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_3_73, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s1_q2, %qk_seg0_s1_q2, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_3, Release, 1)
          aie.use_lock(%lock_2_3_75, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s1_q2, %s_seg0_s1_q2, %r_seg0_s1_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s1_q2, %gp_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s1_q2, %gp_seg0_s1_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s1_q2, %r_seg0_s1_q2, %s_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s1_q2, %sp_seg0_s1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_2_3_74, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s1_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s1_q2, %prev_up_seg0_s1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s1_q2, %up_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s1_q2, %up_seg0_s1_q2, %r_cascade_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s1_q2, %up_seg0_s1_q2, %r_local_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s1_q2, %merged_gp_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s1_q2, %gp_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s1_q2, %merged_gp_seg0_s1_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s1_q2) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s1_q2, %r_cascade_seg0_s1_q2, %tmp_sp_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s1_q2, %r_local_seg0_s1_q2, %tmp_sp_seg0_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s1_q2, %merged_sp_seg0_s1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s1_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s2_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_4_76, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_4_77, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s2_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_4_78, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s2_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s2_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s2_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4, Release, 1)
        aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4, Release, 1)
        aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s2_q2, %q_seg0_s2_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_2_4, Release, 1)
        aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s2_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_4_76, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s2_q2, %qk_seg0_s2_q2, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_4, Release, 1)
          aie.use_lock(%lock_2_4_78, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s2_q2, %s_seg0_s2_q2, %r_seg0_s2_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s2_q2, %gp_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s2_q2, %gp_seg0_s2_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s2_q2, %r_seg0_s2_q2, %s_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s2_q2, %sp_seg0_s2_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_2_4_77, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s2_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s2_q2, %prev_up_seg0_s2_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s2_q2, %up_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s2_q2, %up_seg0_s2_q2, %r_cascade_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s2_q2, %up_seg0_s2_q2, %r_local_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s2_q2, %merged_gp_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s2_q2, %gp_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s2_q2, %merged_gp_seg0_s2_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s2_q2) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s2_q2, %r_cascade_seg0_s2_q2, %tmp_sp_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s2_q2, %r_local_seg0_s2_q2, %tmp_sp_seg0_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s2_q2, %merged_sp_seg0_s2_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s2_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s3_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_5_79, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_2_5_80, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s3_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_2_5_81, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s3_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s3_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s3_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_2_5_79, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5, Release, 1)
        aie.use_lock(%lock_2_5_79, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5, Release, 1)
        aie.use_lock(%lock_2_5_79, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s3_q2, %q_seg0_s3_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_2_5, Release, 1)
        aie.use_lock(%lock_2_5_79, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s3_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_5_79, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s3_q2, %qk_seg0_s3_q2, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_2_5, Release, 1)
          aie.use_lock(%lock_2_5_81, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s3_q2, %s_seg0_s3_q2, %r_seg0_s3_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s3_q2, %gp_seg0_s3_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s3_q2, %gp_seg0_s3_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s3_q2, %r_seg0_s3_q2, %s_seg0_s3_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s3_q2, %sp_seg0_s3_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_2_5_80, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg0_s3_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg0_s3_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg0_s3_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg0_q3 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_2_82, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_3_2_83, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s0_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_2_84, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_3_2_85, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s0_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_2_86, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_3_2_82, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s0_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s0_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s0_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_3_2_84, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_83, Release, 1)
        aie.use_lock(%lock_3_2_84, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_83, Release, 1)
        aie.use_lock(%lock_3_2_84, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_83, Release, 1)
        aie.use_lock(%lock_3_2_84, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s0_q3, %q_seg0_s0_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_3_2_83, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s0_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_2_84, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s0_q3, %qk_seg0_s0_q3, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_2_83, Release, 1)
          aie.use_lock(%lock_3_2_86, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s0_q3, %s_seg0_s0_q3, %r_seg0_s0_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s0_q3, %gp_seg0_s0_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s0_q3, %gp_seg0_s0_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s0_q3, %r_seg0_s0_q3, %s_seg0_s0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s0_q3, %sp_seg0_s0_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_3_2_85, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s0_q3, %prev_up_seg0_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_q3, %up_seg0_s0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_q3, %up_seg0_s0_q3, %r_cascade_seg0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_q3, %up_seg0_s0_q3, %r_local_seg0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_q3, %merged_gp_seg0_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_q3, %gp_seg0_s0_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s0_q3, %merged_gp_seg0_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_q3) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_q3, %r_cascade_seg0_q3, %tmp_sp_seg0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s0_q3, %r_local_seg0_q3, %tmp_sp_seg0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_q3, %merged_sp_seg0_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg0_q3, %merged_gp_seg0_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_3_2, Release, 1)
      }
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s1_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_3_87, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_3_3_88, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s1_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_3_89, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s1_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s1_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s1_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_3_3_87, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3, Release, 1)
        aie.use_lock(%lock_3_3_87, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3, Release, 1)
        aie.use_lock(%lock_3_3_87, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3, Release, 1)
        aie.use_lock(%lock_3_3_87, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s1_q3, %q_seg0_s1_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_3_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s1_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_3_87, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s1_q3, %qk_seg0_s1_q3, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_3, Release, 1)
          aie.use_lock(%lock_3_3_89, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s1_q3, %s_seg0_s1_q3, %r_seg0_s1_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s1_q3, %gp_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s1_q3, %gp_seg0_s1_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s1_q3, %r_seg0_s1_q3, %s_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s1_q3, %sp_seg0_s1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_3_3_88, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s1_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s1_q3, %prev_up_seg0_s1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s1_q3, %up_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s1_q3, %up_seg0_s1_q3, %r_cascade_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s1_q3, %up_seg0_s1_q3, %r_local_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s1_q3, %merged_gp_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s1_q3, %gp_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s1_q3, %merged_gp_seg0_s1_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s1_q3) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s1_q3, %r_cascade_seg0_s1_q3, %tmp_sp_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s1_q3, %r_local_seg0_s1_q3, %tmp_sp_seg0_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s1_q3, %merged_sp_seg0_s1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s1_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s2_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_4_90, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_3_4_91, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s2_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_4_92, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s2_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s2_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s2_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_3_4_90, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4, Release, 1)
        aie.use_lock(%lock_3_4_90, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4, Release, 1)
        aie.use_lock(%lock_3_4_90, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4, Release, 1)
        aie.use_lock(%lock_3_4_90, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s2_q3, %q_seg0_s2_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_3_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg0_s2_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_4_90, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s2_q3, %qk_seg0_s2_q3, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_4, Release, 1)
          aie.use_lock(%lock_3_4_92, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg0_s2_q3, %s_seg0_s2_q3, %r_seg0_s2_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s2_q3, %gp_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg0_s2_q3, %gp_seg0_s2_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s2_q3, %r_seg0_s2_q3, %s_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s2_q3, %sp_seg0_s2_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_3_4_91, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg0_s2_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg0_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg0_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg0_s2_q3, %prev_up_seg0_s2_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg0_s2_q3, %up_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg0_s2_q3, %up_seg0_s2_q3, %r_cascade_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg0_s2_q3, %up_seg0_s2_q3, %r_local_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg0_s2_q3, %merged_gp_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg0_s2_q3, %gp_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg0_s2_q3, %merged_gp_seg0_s2_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg0_s2_q3) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg0_s2_q3, %r_cascade_seg0_s2_q3, %tmp_sp_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg0_s2_q3, %r_local_seg0_s2_q3, %tmp_sp_seg0_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg0_s2_q3, %merged_sp_seg0_s2_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg0_s2_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg0_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg0_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg0_s3_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_5_93, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_3_5_94, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg0_s3_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_3_5_95, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg0_s3_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg0_s3_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg0_s3_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_3_5_93, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5, Release, 1)
        aie.use_lock(%lock_3_5_93, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5, Release, 1)
        aie.use_lock(%lock_3_5_93, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5, Release, 1)
        aie.use_lock(%lock_3_5_93, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg0_s3_q3, %q_seg0_s3_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_3_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg0_s3_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_5_93, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg0_s3_q3, %qk_seg0_s3_q3, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_3_5, Release, 1)
          aie.use_lock(%lock_3_5_95, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg0_s3_q3, %s_seg0_s3_q3, %r_seg0_s3_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg0_s3_q3, %gp_seg0_s3_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg0_s3_q3, %gp_seg0_s3_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg0_s3_q3, %r_seg0_s3_q3, %s_seg0_s3_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg0_s3_q3, %sp_seg0_s3_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_3_5_94, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg0_s3_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg0_s3_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg0_s3_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg1_q0 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_4_2_96, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_4_2_97, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s0_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_2_98, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_4_2_99, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s0_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_2_100, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_4_2_96, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s0_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s0_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s0_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s0_q0, %q_seg1_s0_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_4_2_97, Release, 1)
        aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_97, Release, 1)
        aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_97, Release, 1)
        aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_97, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s0_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_2_98, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s0_q0, %qk_seg1_s0_q0, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_2_97, Release, 1)
          aie.use_lock(%lock_4_2_100, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s0_q0, %s_seg1_s0_q0, %r_seg1_s0_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s0_q0, %gp_seg1_s0_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s0_q0, %gp_seg1_s0_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s0_q0, %r_seg1_s0_q0, %s_seg1_s0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s0_q0, %sp_seg1_s0_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_4_2_99, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s0_q0, %prev_up_seg1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_q0, %up_seg1_s0_q0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_q0, %up_seg1_s0_q0, %r_cascade_seg1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_q0, %up_seg1_s0_q0, %r_local_seg1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_q0, %merged_gp_seg1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_q0, %gp_seg1_s0_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s0_q0, %merged_gp_seg1_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_q0) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_q0, %r_cascade_seg1_q0, %tmp_sp_seg1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s0_q0, %r_local_seg1_q0, %tmp_sp_seg1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_q0, %merged_sp_seg1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg1_q0, %merged_gp_seg1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_4_2, Release, 1)
      }
      aie.end
    }
    %mem_4_3 = aie.mem(%tile_4_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s1_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_3_101, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_4_3_102, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s1_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_3_103, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s1_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s1_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s1_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_4_3_101, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s1_q0, %q_seg1_s1_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_4_3, Release, 1)
        aie.use_lock(%lock_4_3_101, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_3, Release, 1)
        aie.use_lock(%lock_4_3_101, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_3, Release, 1)
        aie.use_lock(%lock_4_3_101, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s1_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_3_101, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s1_q0, %qk_seg1_s1_q0, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_3, Release, 1)
          aie.use_lock(%lock_4_3_103, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s1_q0, %s_seg1_s1_q0, %r_seg1_s1_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s1_q0, %gp_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s1_q0, %gp_seg1_s1_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s1_q0, %r_seg1_s1_q0, %s_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s1_q0, %sp_seg1_s1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_4_3_102, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s1_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s1_q0, %prev_up_seg1_s1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s1_q0, %up_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s1_q0, %up_seg1_s1_q0, %r_cascade_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s1_q0, %up_seg1_s1_q0, %r_local_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s1_q0, %merged_gp_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s1_q0, %gp_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s1_q0, %merged_gp_seg1_s1_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s1_q0) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s1_q0, %r_cascade_seg1_s1_q0, %tmp_sp_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s1_q0, %r_local_seg1_s1_q0, %tmp_sp_seg1_s1_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s1_q0, %merged_sp_seg1_s1_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s1_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s1_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_4_4 = aie.mem(%tile_4_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s2_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_4_104, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_4_4_105, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s2_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_4_106, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s2_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s2_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s2_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_4_4_104, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s2_q0, %q_seg1_s2_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_4_4, Release, 1)
        aie.use_lock(%lock_4_4_104, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_4, Release, 1)
        aie.use_lock(%lock_4_4_104, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_4, Release, 1)
        aie.use_lock(%lock_4_4_104, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s2_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_4_104, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s2_q0, %qk_seg1_s2_q0, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_4, Release, 1)
          aie.use_lock(%lock_4_4_106, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s2_q0, %s_seg1_s2_q0, %r_seg1_s2_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s2_q0, %gp_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s2_q0, %gp_seg1_s2_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s2_q0, %r_seg1_s2_q0, %s_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s2_q0, %sp_seg1_s2_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_4_4_105, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s2_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s2_q0, %prev_up_seg1_s2_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s2_q0, %up_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s2_q0, %up_seg1_s2_q0, %r_cascade_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s2_q0, %up_seg1_s2_q0, %r_local_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s2_q0, %merged_gp_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s2_q0, %gp_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s2_q0, %merged_gp_seg1_s2_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s2_q0) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s2_q0, %r_cascade_seg1_s2_q0, %tmp_sp_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s2_q0, %r_local_seg1_s2_q0, %tmp_sp_seg1_s2_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s2_q0, %merged_sp_seg1_s2_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s2_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s2_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_4_5 = aie.mem(%tile_4_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s3_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_5_107, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_4_5_108, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s3_q0 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_4_5_109, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s3_q0) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s3_q0) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s3_q0) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_4_5_107, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s3_q0, %q_seg1_s3_q0) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_4_5, Release, 1)
        aie.use_lock(%lock_4_5_107, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_5, Release, 1)
        aie.use_lock(%lock_4_5_107, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_5, Release, 1)
        aie.use_lock(%lock_4_5_107, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s3_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_5_107, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s3_q0, %qk_seg1_s3_q0, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_4_5, Release, 1)
          aie.use_lock(%lock_4_5_109, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s3_q0, %s_seg1_s3_q0, %r_seg1_s3_q0) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s3_q0, %gp_seg1_s3_q0) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s3_q0, %gp_seg1_s3_q0) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s3_q0, %r_seg1_s3_q0, %s_seg1_s3_q0) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s3_q0, %sp_seg1_s3_q0) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_4_5_108, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg1_s3_q0 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg1_s3_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg1_s3_q0 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg1_q1 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_5_2_110, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_5_2_111, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s0_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_2_112, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_5_2_113, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s0_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_2_114, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_5_2_110, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s0_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s0_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s0_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_5_2_112, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_111, Release, 1)
        aie.use_lock(%lock_5_2_112, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s0_q1, %q_seg1_s0_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_5_2_111, Release, 1)
        aie.use_lock(%lock_5_2_112, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_111, Release, 1)
        aie.use_lock(%lock_5_2_112, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_111, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s0_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_2_112, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s0_q1, %qk_seg1_s0_q1, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_2_111, Release, 1)
          aie.use_lock(%lock_5_2_114, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s0_q1, %s_seg1_s0_q1, %r_seg1_s0_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s0_q1, %gp_seg1_s0_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s0_q1, %gp_seg1_s0_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s0_q1, %r_seg1_s0_q1, %s_seg1_s0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s0_q1, %sp_seg1_s0_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_5_2_113, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s0_q1, %prev_up_seg1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_q1, %up_seg1_s0_q1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_q1, %up_seg1_s0_q1, %r_cascade_seg1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_q1, %up_seg1_s0_q1, %r_local_seg1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_q1, %merged_gp_seg1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_q1, %gp_seg1_s0_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s0_q1, %merged_gp_seg1_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_q1) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_q1, %r_cascade_seg1_q1, %tmp_sp_seg1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s0_q1, %r_local_seg1_q1, %tmp_sp_seg1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_q1, %merged_sp_seg1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg1_q1, %merged_gp_seg1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_5_2, Release, 1)
      }
      aie.end
    }
    %mem_5_3 = aie.mem(%tile_5_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s1_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_3_115, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_5_3_116, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s1_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_3_117, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s1_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s1_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s1_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_5_3_115, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_3, Release, 1)
        aie.use_lock(%lock_5_3_115, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s1_q1, %q_seg1_s1_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_5_3, Release, 1)
        aie.use_lock(%lock_5_3_115, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_3, Release, 1)
        aie.use_lock(%lock_5_3_115, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s1_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_3_115, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s1_q1, %qk_seg1_s1_q1, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_3, Release, 1)
          aie.use_lock(%lock_5_3_117, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s1_q1, %s_seg1_s1_q1, %r_seg1_s1_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s1_q1, %gp_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s1_q1, %gp_seg1_s1_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s1_q1, %r_seg1_s1_q1, %s_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s1_q1, %sp_seg1_s1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_5_3_116, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s1_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s1_q1, %prev_up_seg1_s1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s1_q1, %up_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s1_q1, %up_seg1_s1_q1, %r_cascade_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s1_q1, %up_seg1_s1_q1, %r_local_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s1_q1, %merged_gp_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s1_q1, %gp_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s1_q1, %merged_gp_seg1_s1_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s1_q1) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s1_q1, %r_cascade_seg1_s1_q1, %tmp_sp_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s1_q1, %r_local_seg1_s1_q1, %tmp_sp_seg1_s1_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s1_q1, %merged_sp_seg1_s1_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s1_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s1_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_5_4 = aie.mem(%tile_5_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s2_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_4_118, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_5_4_119, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s2_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_4_120, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s2_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s2_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s2_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_5_4_118, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_4, Release, 1)
        aie.use_lock(%lock_5_4_118, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s2_q1, %q_seg1_s2_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_5_4, Release, 1)
        aie.use_lock(%lock_5_4_118, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_4, Release, 1)
        aie.use_lock(%lock_5_4_118, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s2_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_4_118, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s2_q1, %qk_seg1_s2_q1, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_4, Release, 1)
          aie.use_lock(%lock_5_4_120, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s2_q1, %s_seg1_s2_q1, %r_seg1_s2_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s2_q1, %gp_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s2_q1, %gp_seg1_s2_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s2_q1, %r_seg1_s2_q1, %s_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s2_q1, %sp_seg1_s2_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_5_4_119, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s2_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s2_q1, %prev_up_seg1_s2_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s2_q1, %up_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s2_q1, %up_seg1_s2_q1, %r_cascade_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s2_q1, %up_seg1_s2_q1, %r_local_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s2_q1, %merged_gp_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s2_q1, %gp_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s2_q1, %merged_gp_seg1_s2_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s2_q1) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s2_q1, %r_cascade_seg1_s2_q1, %tmp_sp_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s2_q1, %r_local_seg1_s2_q1, %tmp_sp_seg1_s2_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s2_q1, %merged_sp_seg1_s2_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s2_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s2_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_5_5 = aie.mem(%tile_5_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s3_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_5_121, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_5_5_122, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s3_q1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_5_5_123, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s3_q1) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s3_q1) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s3_q1) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_5_5_121, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_5, Release, 1)
        aie.use_lock(%lock_5_5_121, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s3_q1, %q_seg1_s3_q1) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_5_5, Release, 1)
        aie.use_lock(%lock_5_5_121, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_5, Release, 1)
        aie.use_lock(%lock_5_5_121, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s3_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_5_121, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s3_q1, %qk_seg1_s3_q1, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_5_5, Release, 1)
          aie.use_lock(%lock_5_5_123, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s3_q1, %s_seg1_s3_q1, %r_seg1_s3_q1) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s3_q1, %gp_seg1_s3_q1) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s3_q1, %gp_seg1_s3_q1) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s3_q1, %r_seg1_s3_q1, %s_seg1_s3_q1) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s3_q1, %sp_seg1_s3_q1) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_5_5_122, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg1_s3_q1 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg1_s3_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg1_s3_q1 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg1_q2 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_6_2_124, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_6_2_125, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s0_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_2_126, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_6_2_127, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s0_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_2_128, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_6_2_124, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s0_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s0_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s0_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_6_2_126, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_125, Release, 1)
        aie.use_lock(%lock_6_2_126, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_125, Release, 1)
        aie.use_lock(%lock_6_2_126, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s0_q2, %q_seg1_s0_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_6_2_125, Release, 1)
        aie.use_lock(%lock_6_2_126, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_125, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s0_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_2_126, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s0_q2, %qk_seg1_s0_q2, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_2_125, Release, 1)
          aie.use_lock(%lock_6_2_128, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s0_q2, %s_seg1_s0_q2, %r_seg1_s0_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s0_q2, %gp_seg1_s0_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s0_q2, %gp_seg1_s0_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s0_q2, %r_seg1_s0_q2, %s_seg1_s0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s0_q2, %sp_seg1_s0_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_6_2_127, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s0_q2, %prev_up_seg1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_q2, %up_seg1_s0_q2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_q2, %up_seg1_s0_q2, %r_cascade_seg1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_q2, %up_seg1_s0_q2, %r_local_seg1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_q2, %merged_gp_seg1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_q2, %gp_seg1_s0_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s0_q2, %merged_gp_seg1_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_q2) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_q2, %r_cascade_seg1_q2, %tmp_sp_seg1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s0_q2, %r_local_seg1_q2, %tmp_sp_seg1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_q2, %merged_sp_seg1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg1_q2, %merged_gp_seg1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_6_2, Release, 1)
      }
      aie.end
    }
    %mem_6_3 = aie.mem(%tile_6_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s1_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_3_129, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_6_3_130, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s1_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_3_131, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s1_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s1_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s1_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_6_3_129, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_3, Release, 1)
        aie.use_lock(%lock_6_3_129, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_3, Release, 1)
        aie.use_lock(%lock_6_3_129, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s1_q2, %q_seg1_s1_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_6_3, Release, 1)
        aie.use_lock(%lock_6_3_129, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s1_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_3_129, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s1_q2, %qk_seg1_s1_q2, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_3, Release, 1)
          aie.use_lock(%lock_6_3_131, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s1_q2, %s_seg1_s1_q2, %r_seg1_s1_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s1_q2, %gp_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s1_q2, %gp_seg1_s1_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s1_q2, %r_seg1_s1_q2, %s_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s1_q2, %sp_seg1_s1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_6_3_130, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s1_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s1_q2, %prev_up_seg1_s1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s1_q2, %up_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s1_q2, %up_seg1_s1_q2, %r_cascade_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s1_q2, %up_seg1_s1_q2, %r_local_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s1_q2, %merged_gp_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s1_q2, %gp_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s1_q2, %merged_gp_seg1_s1_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s1_q2) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s1_q2, %r_cascade_seg1_s1_q2, %tmp_sp_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s1_q2, %r_local_seg1_s1_q2, %tmp_sp_seg1_s1_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s1_q2, %merged_sp_seg1_s1_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s1_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s1_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_6_4 = aie.mem(%tile_6_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s2_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_4_132, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_6_4_133, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s2_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_4_134, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s2_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s2_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s2_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_6_4_132, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_4, Release, 1)
        aie.use_lock(%lock_6_4_132, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_4, Release, 1)
        aie.use_lock(%lock_6_4_132, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s2_q2, %q_seg1_s2_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_6_4, Release, 1)
        aie.use_lock(%lock_6_4_132, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s2_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_4_132, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s2_q2, %qk_seg1_s2_q2, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_4, Release, 1)
          aie.use_lock(%lock_6_4_134, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s2_q2, %s_seg1_s2_q2, %r_seg1_s2_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s2_q2, %gp_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s2_q2, %gp_seg1_s2_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s2_q2, %r_seg1_s2_q2, %s_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s2_q2, %sp_seg1_s2_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_6_4_133, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s2_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s2_q2, %prev_up_seg1_s2_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s2_q2, %up_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s2_q2, %up_seg1_s2_q2, %r_cascade_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s2_q2, %up_seg1_s2_q2, %r_local_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s2_q2, %merged_gp_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s2_q2, %gp_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s2_q2, %merged_gp_seg1_s2_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s2_q2) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s2_q2, %r_cascade_seg1_s2_q2, %tmp_sp_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s2_q2, %r_local_seg1_s2_q2, %tmp_sp_seg1_s2_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s2_q2, %merged_sp_seg1_s2_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s2_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s2_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s3_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_5_135, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_6_5_136, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s3_q2 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_6_5_137, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s3_q2) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s3_q2) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s3_q2) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_6_5_135, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_5, Release, 1)
        aie.use_lock(%lock_6_5_135, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_5, Release, 1)
        aie.use_lock(%lock_6_5_135, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s3_q2, %q_seg1_s3_q2) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_6_5, Release, 1)
        aie.use_lock(%lock_6_5_135, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s3_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_5_135, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s3_q2, %qk_seg1_s3_q2, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_6_5, Release, 1)
          aie.use_lock(%lock_6_5_137, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s3_q2, %s_seg1_s3_q2, %r_seg1_s3_q2) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s3_q2, %gp_seg1_s3_q2) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s3_q2, %gp_seg1_s3_q2) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s3_q2, %r_seg1_s3_q2, %s_seg1_s3_q2) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s3_q2, %sp_seg1_s3_q2) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_6_5_136, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg1_s3_q2 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg1_s3_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg1_s3_q2 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%merged_gp_seg1_q3 : memref<64x64xbf16>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>])
      aie.use_lock(%lock_7_2_138, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_7_2_139, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s0_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_2_140, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_7_2_141, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s0_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_2_142, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_7_2_138, AcquireGreaterEqual, 1)
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s0_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s0_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s0_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_139, Release, 1)
        aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_139, Release, 1)
        aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_139, Release, 1)
        aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s0_q3, %q_seg1_s0_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_7_2_139, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s0_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_2_140, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s0_q3, %qk_seg1_s0_q3, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_2_139, Release, 1)
          aie.use_lock(%lock_7_2_142, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s0_q3, %s_seg1_s0_q3, %r_seg1_s0_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s0_q3, %gp_seg1_s0_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s0_q3, %gp_seg1_s0_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s0_q3, %r_seg1_s0_q3, %s_seg1_s0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s0_q3, %sp_seg1_s0_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_7_2_141, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s0_q3, %prev_up_seg1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_q3, %up_seg1_s0_q3) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_q3, %up_seg1_s0_q3, %r_cascade_seg1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_q3, %up_seg1_s0_q3, %r_local_seg1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_q3, %merged_gp_seg1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_q3, %gp_seg1_s0_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s0_q3, %merged_gp_seg1_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_q3) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_q3, %r_cascade_seg1_q3, %tmp_sp_seg1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s0_q3, %r_local_seg1_q3, %tmp_sp_seg1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_q3, %merged_sp_seg1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @div_gp_sp(%merged_sp_seg1_q3, %merged_gp_seg1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_7_2, Release, 1)
      }
      aie.end
    }
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s1_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_3_143, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_7_3_144, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s1_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_3_145, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s1_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s1_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s1_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_7_3_143, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_3, Release, 1)
        aie.use_lock(%lock_7_3_143, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_3, Release, 1)
        aie.use_lock(%lock_7_3_143, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_3, Release, 1)
        aie.use_lock(%lock_7_3_143, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s1_q3, %q_seg1_s1_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_7_3, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s1_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_3_143, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s1_q3, %qk_seg1_s1_q3, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_3, Release, 1)
          aie.use_lock(%lock_7_3_145, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s1_q3, %s_seg1_s1_q3, %r_seg1_s1_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s1_q3, %gp_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s1_q3, %gp_seg1_s1_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s1_q3, %r_seg1_s1_q3, %s_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s1_q3, %sp_seg1_s1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_7_3_144, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s1_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s1_q3, %prev_up_seg1_s1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s1_q3, %up_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s1_q3, %up_seg1_s1_q3, %r_cascade_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s1_q3, %up_seg1_s1_q3, %r_local_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s1_q3, %merged_gp_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s1_q3, %gp_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s1_q3, %merged_gp_seg1_s1_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s1_q3) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s1_q3, %r_cascade_seg1_s1_q3, %tmp_sp_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s1_q3, %r_local_seg1_s1_q3, %tmp_sp_seg1_s1_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s1_q3, %merged_sp_seg1_s1_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s1_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s1_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s2_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_4_146, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_7_4_147, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s2_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_4_148, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s2_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s2_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s2_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_7_4_146, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_4, Release, 1)
        aie.use_lock(%lock_7_4_146, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_4, Release, 1)
        aie.use_lock(%lock_7_4_146, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_4, Release, 1)
        aie.use_lock(%lock_7_4_146, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s2_q3, %q_seg1_s2_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_7_4, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_175 = memref.collapse_shape %g_seg1_s2_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_175) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_4_146, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s2_q3, %qk_seg1_s2_q3, %collapse_shape_175) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_4, Release, 1)
          aie.use_lock(%lock_7_4_148, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_175, %up_seg1_s2_q3, %s_seg1_s2_q3, %r_seg1_s2_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s2_q3, %gp_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_175, %v_seg1_s2_q3, %gp_seg1_s2_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s2_q3, %r_seg1_s2_q3, %s_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s2_q3, %sp_seg1_s2_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_7_4_147, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %merged_gp_seg1_s2_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_156 = memref.collapse_shape %merged_up_seg1_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        %collapse_shape_159 = memref.collapse_shape %merged_sp_seg1_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = aie.get_cascade() : vector<32xbf16>
          vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>
        }
        func.call @vector_copy_32elems_pythoc(%c0_i32, %up_seg1_s2_q3, %prev_up_seg1_s2_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @maximum_up_u_bf16(%merged_up_seg1_s2_q3, %up_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%merged_up_seg1_s2_q3, %up_seg1_s2_q3, %r_cascade_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @exp_up_minus_u(%prev_up_seg1_s2_q3, %up_seg1_s2_q3, %r_local_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @mul_r_gp(%r_cascade_seg1_s2_q3, %merged_gp_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @mul_r_gp(%r_local_seg1_s2_q3, %gp_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
        func.call @add_gp_g(%gp_seg1_s2_q3, %merged_gp_seg1_s2_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%tmp_sp_seg1_s2_q3) : (memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%merged_sp_seg1_s2_q3, %r_cascade_seg1_s2_q3, %tmp_sp_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @accum_sp_r_s(%sp_seg1_s2_q3, %r_local_seg1_s2_q3, %tmp_sp_seg1_s2_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        func.call @vector_copy_32elems_pythoc(%c0_i32, %tmp_sp_seg1_s2_q3, %merged_sp_seg1_s2_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
        %collapse_shape_163 = memref.collapse_shape %merged_gp_seg1_s2_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_164 = arith.constant 0 : index
        %c4096_165 = arith.constant 4096 : index
        %c32_166 = arith.constant 32 : index
        scf.for %arg1 = %c0_164 to %c4096_165 step %c32_166 {
          %subview = memref.subview %collapse_shape_163[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_167 = memref.collapse_shape %up_seg1_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_168 = arith.constant 0 : index
        %c64_169 = arith.constant 64 : index
        %c32_170 = arith.constant 32 : index
        scf.for %arg1 = %c0_168 to %c64_169 step %c32_170 {
          %subview = memref.subview %collapse_shape_167[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_171 = memref.collapse_shape %merged_sp_seg1_s2_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_172 = arith.constant 0 : index
        %c64_173 = arith.constant 64 : index
        %c32_174 = arith.constant 32 : index
        scf.for %arg1 = %c0_172 to %c64_173 step %c32_174 {
          %subview = memref.subview %collapse_shape_171[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    %mem_7_5 = aie.mem(%tile_7_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%qk_seg1_s3_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_5_149, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_7_5_150, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_seg1_s3_q3 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_7_5_151, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_152 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0_152 to %c9223372036854775807 step %c1 {
        func.call @zero_fill_gp_bf16_pythoc(%gp_seg1_s3_q3) : (memref<64x64xbf16>) -> ()
        func.call @zero_fill_sp_bf16_pythoc(%sp_seg1_s3_q3) : (memref<64x1xbf16>) -> ()
        func.call @neg_inf_fill_up_bf16_pythoc(%up_seg1_s3_q3) : (memref<64x1xbf16>) -> ()
        aie.use_lock(%lock_7_5_149, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_5, Release, 1)
        aie.use_lock(%lock_7_5_149, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_5, Release, 1)
        aie.use_lock(%lock_7_5_149, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_5, Release, 1)
        aie.use_lock(%lock_7_5_149, AcquireGreaterEqual, 1)
        func.call @copy_tile_pythoc(%qk_seg1_s3_q3, %q_seg1_s3_q3) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_7_5, Release, 1)
        %c0_153 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_154 = arith.constant 1 : index
        scf.for %arg1 = %c0_153 to %c2 step %c1_154 {
          %collapse_shape_163 = memref.collapse_shape %g_seg1_s3_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
          func.call @zero_fill_g_bf16_pythoc(%collapse_shape_163) : (memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_5_149, AcquireGreaterEqual, 1)
          func.call @matmul_a_b_bf16(%q_seg1_s3_q3, %qk_seg1_s3_q3, %collapse_shape_163) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<4096xbf16>) -> ()
          aie.use_lock(%lock_7_5, Release, 1)
          aie.use_lock(%lock_7_5_151, AcquireGreaterEqual, 1)
          func.call @fused_softmax(%collapse_shape_163, %up_seg1_s3_q3, %s_seg1_s3_q3, %r_seg1_s3_q3) : (memref<4096xbf16>, memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @mul_r_gp(%r_seg1_s3_q3, %gp_seg1_s3_q3) : (memref<64x1xbf16>, memref<64x64xbf16>) -> ()
          func.call @matmul_g_b_bf16(%collapse_shape_163, %v_seg1_s3_q3, %gp_seg1_s3_q3) : (memref<4096xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
          func.call @accum_sp_r_s(%sp_seg1_s3_q3, %r_seg1_s3_q3, %s_seg1_s3_q3) : (memref<64x1xbf16>, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          func.call @vector_copy_32elems_pythoc(%c0_i32, %s_seg1_s3_q3, %sp_seg1_s3_q3) : (i32, memref<64x1xbf16>, memref<64x1xbf16>) -> ()
          aie.use_lock(%lock_7_5_150, Release, 1)
        }
        %collapse_shape = memref.collapse_shape %gp_seg1_s3_q3 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
        %c0_155 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c32 = arith.constant 32 : index
        scf.for %arg1 = %c0_155 to %c4096 step %c32 {
          %subview = memref.subview %collapse_shape[%arg1] [32] [1] : memref<4096xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_156 = memref.collapse_shape %up_seg1_s3_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_157 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32_158 = arith.constant 32 : index
        scf.for %arg1 = %c0_157 to %c64 step %c32_158 {
          %subview = memref.subview %collapse_shape_156[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
        %collapse_shape_159 = memref.collapse_shape %sp_seg1_s3_q3 [[0, 1]] : memref<64x1xbf16> into memref<64xbf16>
        %c0_160 = arith.constant 0 : index
        %c64_161 = arith.constant 64 : index
        %c32_162 = arith.constant 32 : index
        scf.for %arg1 = %c0_160 to %c64_161 step %c32_162 {
          %subview = memref.subview %collapse_shape_159[%arg1] [32] [1] : memref<64xbf16> to memref<32xbf16, strided<[1], offset: ?>>
          %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>>, vector<32xbf16>
          aie.put_cascade(%0 : vector<32xbf16>)
        }
      }
      aie.end
    }
    aie.runtime_sequence @attention_bf16(%arg0: memref<2x512x64xbf16>, %arg1: memref<2x512x64xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x512x64xbf16>) {
      %0 = aiex.dma_configure_task_for @air_QKIn_0_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 0, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_QKIn_0_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_QKIn_1_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 0, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_QKIn_1_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 8192, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_QKIn_2_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 0, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_QKIn_2_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 16384, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_QKIn_3_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 0, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_QKIn_3_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 24576, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_VIn_0_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_VIn_1_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 8192, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_VIn_2_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 16384, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_VIn_3_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 24576, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_0_0_0_0 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_0_0_0_1 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 4096, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_0_0_0_2 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 8192, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_0_0_0_3 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 12288, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_QKIn_0_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 32768, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @air_QKIn_0_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 32768, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_QKIn_1_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 32768, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_QKIn_1_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 40960, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_QKIn_2_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 32768, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_QKIn_2_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 49152, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_QKIn_3_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 32768, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_QKIn_3_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 57344, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @air_VIn_0_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 32768, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @air_VIn_1_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 40960, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @air_VIn_2_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 49152, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @air_VIn_3_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 57344, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @air_channel_0_1_0_0 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 32768, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @air_channel_0_1_0_1 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 36864, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @air_channel_0_1_0_2 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 40960, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @air_channel_0_1_0_3 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 45056, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%11)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%15)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%27)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%31)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%28)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%24)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%12)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%8)
      %32 = aiex.dma_configure_task_for @air_QKIn_0_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 16384, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @air_QKIn_0_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @air_QKIn_1_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 16384, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @air_QKIn_1_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 8192, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @air_QKIn_2_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 16384, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @air_QKIn_2_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 16384, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @air_QKIn_3_0_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 16384, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @air_QKIn_3_0_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 24576, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @air_VIn_0_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @air_VIn_1_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 8192, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @air_VIn_2_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 16384, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @air_VIn_3_0_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 24576, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @air_channel_0_0_0_0 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 16384, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @air_channel_0_0_0_1 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 20480, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @air_channel_0_0_0_2 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 24576, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @air_channel_0_0_0_3 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 28672, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @air_QKIn_0_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 49152, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @air_QKIn_0_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 32768, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @air_QKIn_1_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 49152, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @air_QKIn_1_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 40960, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @air_QKIn_2_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 49152, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @air_QKIn_2_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 49152, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @air_QKIn_3_1_0_0 {
        aie.dma_bd(%arg0 : memref<2x512x64xbf16>, 49152, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @air_QKIn_3_1_0_0 {
        aie.dma_bd(%arg1 : memref<2x512x64xbf16>, 57344, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @air_VIn_0_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 32768, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @air_VIn_1_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 40960, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @air_VIn_2_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 49152, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @air_VIn_3_1_0_0 {
        aie.dma_bd(%arg2 : memref<2x512x64xbf16>, 57344, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @air_channel_0_1_0_0 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 49152, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @air_channel_0_1_0_1 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 53248, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @air_channel_0_1_0_2 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 57344, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @air_channel_0_1_0_3 {
        aie.dma_bd(%arg3 : memref<2x512x64xbf16>, 61440, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%43)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%47)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%59)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%63)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%48)
      aiex.dma_free_task(%49)
      aiex.dma_free_task(%50)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%60)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%56)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%44)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%40)
    }
  }
}

