#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @rk_rope_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf129 = aie.buffer(%tile_0_2) {sym_name = "buf129"} : memref<64xbf16, 2 : i32> 
    %buf128 = aie.buffer(%tile_0_2) {sym_name = "buf128"} : memref<64xbf16, 2 : i32> 
    %buf127 = aie.buffer(%tile_0_2) {sym_name = "buf127"} : memref<64xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<512xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<512xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<512xbf16>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf129 : memref<64xbf16, 2 : i32>, 0, 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf127 : memref<64xbf16, 2 : i32>, 0, 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf128 : memref<64xbf16, 2 : i32>, 0, 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c64_i32 = arith.constant 64 : i32
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      scf.for %arg0 = %c0 to %c8 step %c1 {
        aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
        func.call @rope(%buf127, %buf128, %buf129, %c64_i32) : (memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, i32) -> ()
        aie.use_lock(%lock_0_2_1, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
        aie.use_lock(%lock_0_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "rk_rope_herd", air.herd_size = array<i64: 1, 1>, link_with = "rope.o"}
    func.func private @rope(memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, i32) attributes {link_with = "rope.o", llvm.emit_c_interface}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_23(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_21(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_22(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @rk_rope_seg_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048x2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<512x2048xbf16>, %arg6: memref<512xbf16>, %arg7: memref<512x2048xbf16>, %arg8: memref<512xbf16>, %arg9: memref<2048xbf16>, %arg10: memref<512xbf16>, %arg11: memref<2048xbf16>, %arg12: memref<512xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_21 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 0, 512, [<size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_22 {
        aie.dma_bd(%arg10 : memref<512xbf16>, 0, 512, [<size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_23 {
        aie.dma_bd(%arg12 : memref<512xbf16>, 0, 512, [<size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @rq_rope_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf126 = aie.buffer(%tile_0_2) {sym_name = "buf126"} : memref<64xbf16, 2 : i32> 
    %buf125 = aie.buffer(%tile_0_2) {sym_name = "buf125"} : memref<64xbf16, 2 : i32> 
    %buf124 = aie.buffer(%tile_0_2) {sym_name = "buf124"} : memref<64xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf126 : memref<64xbf16, 2 : i32>, 0, 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf124 : memref<64xbf16, 2 : i32>, 0, 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf125 : memref<64xbf16, 2 : i32>, 0, 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c64_i32 = arith.constant 64 : i32
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      scf.for %arg0 = %c0 to %c32 step %c1 {
        aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
        func.call @rope(%buf124, %buf125, %buf126, %c64_i32) : (memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, i32) -> ()
        aie.use_lock(%lock_0_2_1, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
        aie.use_lock(%lock_0_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "rq_rope_herd", air.herd_size = array<i64: 1, 1>, link_with = "rope.o"}
    func.func private @rope(memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, i32) attributes {link_with = "rope.o", llvm.emit_c_interface}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_20(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_18(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_19(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @rq_rope_seg_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048x2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<512x2048xbf16>, %arg6: memref<512xbf16>, %arg7: memref<512x2048xbf16>, %arg8: memref<512xbf16>, %arg9: memref<2048xbf16>, %arg10: memref<512xbf16>, %arg11: memref<2048xbf16>, %arg12: memref<512xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_18 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_19 {
        aie.dma_bd(%arg9 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_20 {
        aie.dma_bd(%arg11 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @v_matvec_bf16_0 {
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
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %lock_7_1 = aie.lock(%mem_tile_7_1, 3) {init = 1 : i32}
    %lock_7_1_0 = aie.lock(%mem_tile_7_1, 2) {init = 0 : i32}
    %lock_7_1_1 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %lock_7_1_2 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_6_1 = aie.lock(%mem_tile_6_1, 3) {init = 1 : i32}
    %lock_6_1_3 = aie.lock(%mem_tile_6_1, 2) {init = 0 : i32}
    %lock_6_1_4 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %lock_6_1_5 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_5_1 = aie.lock(%mem_tile_5_1, 3) {init = 1 : i32}
    %lock_5_1_6 = aie.lock(%mem_tile_5_1, 2) {init = 0 : i32}
    %lock_5_1_7 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_8 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 3) {init = 1 : i32}
    %lock_4_1_9 = aie.lock(%mem_tile_4_1, 2) {init = 0 : i32}
    %lock_4_1_10 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_11 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_3_1 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_12 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_13 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_14 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_15 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_16 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_17 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_18 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_19 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_20 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_21 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_22 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_23 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_24 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_25 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_26 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_27 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_28 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_29 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_30 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_31 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_32 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_33 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_34 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_35 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_36 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_37 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_38 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_39 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_40 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_41 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_42 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_43 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_4_2 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_44 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_45 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_46 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_47 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_48 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_49 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_50 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_51 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_52 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_53 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_54 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_55 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_56 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_57 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_58 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_59 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_60 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_61 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_62 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_63 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %buf123 = aie.buffer(%mem_tile_0_1) {sym_name = "buf123"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf122 = aie.buffer(%mem_tile_1_1) {sym_name = "buf122"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf121 = aie.buffer(%mem_tile_2_1) {sym_name = "buf121"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf120 = aie.buffer(%mem_tile_3_1) {sym_name = "buf120"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf119 = aie.buffer(%mem_tile_4_1) {sym_name = "buf119"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf118 = aie.buffer(%mem_tile_5_1) {sym_name = "buf118"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf117 = aie.buffer(%mem_tile_6_1) {sym_name = "buf117"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf116 = aie.buffer(%mem_tile_7_1) {sym_name = "buf116"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf115 = aie.buffer(%mem_tile_0_1) {sym_name = "buf115"} : memref<1x8xbf16, 1 : i32> 
    %buf114 = aie.buffer(%mem_tile_1_1) {sym_name = "buf114"} : memref<1x8xbf16, 1 : i32> 
    %buf113 = aie.buffer(%mem_tile_2_1) {sym_name = "buf113"} : memref<1x8xbf16, 1 : i32> 
    %buf112 = aie.buffer(%mem_tile_3_1) {sym_name = "buf112"} : memref<1x8xbf16, 1 : i32> 
    %buf111 = aie.buffer(%mem_tile_4_1) {sym_name = "buf111"} : memref<1x8xbf16, 1 : i32> 
    %buf110 = aie.buffer(%mem_tile_5_1) {sym_name = "buf110"} : memref<1x8xbf16, 1 : i32> 
    %buf109 = aie.buffer(%mem_tile_6_1) {sym_name = "buf109"} : memref<1x8xbf16, 1 : i32> 
    %buf108 = aie.buffer(%mem_tile_7_1) {sym_name = "buf108"} : memref<1x8xbf16, 1 : i32> 
    %buf107 = aie.buffer(%tile_7_2) {sym_name = "buf107"} : memref<8xbf16, 2 : i32> 
    %buf106 = aie.buffer(%tile_7_2) {sym_name = "buf106"} : memref<4x2048xbf16, 2 : i32> 
    %buf105 = aie.buffer(%tile_7_2) {sym_name = "buf105"} : memref<2048xbf16, 2 : i32> 
    %buf104 = aie.buffer(%tile_6_2) {sym_name = "buf104"} : memref<8xbf16, 2 : i32> 
    %buf103 = aie.buffer(%tile_6_2) {sym_name = "buf103"} : memref<4x2048xbf16, 2 : i32> 
    %buf102 = aie.buffer(%tile_6_2) {sym_name = "buf102"} : memref<2048xbf16, 2 : i32> 
    %buf101 = aie.buffer(%tile_5_2) {sym_name = "buf101"} : memref<8xbf16, 2 : i32> 
    %buf100 = aie.buffer(%tile_5_2) {sym_name = "buf100"} : memref<4x2048xbf16, 2 : i32> 
    %buf99 = aie.buffer(%tile_5_2) {sym_name = "buf99"} : memref<2048xbf16, 2 : i32> 
    %buf98 = aie.buffer(%tile_4_2) {sym_name = "buf98"} : memref<8xbf16, 2 : i32> 
    %buf97 = aie.buffer(%tile_4_2) {sym_name = "buf97"} : memref<4x2048xbf16, 2 : i32> 
    %buf96 = aie.buffer(%tile_4_2) {sym_name = "buf96"} : memref<2048xbf16, 2 : i32> 
    %buf95 = aie.buffer(%tile_3_2) {sym_name = "buf95"} : memref<8xbf16, 2 : i32> 
    %buf94 = aie.buffer(%tile_3_2) {sym_name = "buf94"} : memref<4x2048xbf16, 2 : i32> 
    %buf93 = aie.buffer(%tile_3_2) {sym_name = "buf93"} : memref<2048xbf16, 2 : i32> 
    %buf92 = aie.buffer(%tile_2_2) {sym_name = "buf92"} : memref<8xbf16, 2 : i32> 
    %buf91 = aie.buffer(%tile_2_2) {sym_name = "buf91"} : memref<4x2048xbf16, 2 : i32> 
    %buf90 = aie.buffer(%tile_2_2) {sym_name = "buf90"} : memref<2048xbf16, 2 : i32> 
    %buf89 = aie.buffer(%tile_1_2) {sym_name = "buf89"} : memref<8xbf16, 2 : i32> 
    %buf88 = aie.buffer(%tile_1_2) {sym_name = "buf88"} : memref<4x2048xbf16, 2 : i32> 
    %buf87 = aie.buffer(%tile_1_2) {sym_name = "buf87"} : memref<2048xbf16, 2 : i32> 
    %buf86 = aie.buffer(%tile_0_2) {sym_name = "buf86"} : memref<8xbf16, 2 : i32> 
    %buf85 = aie.buffer(%tile_0_2) {sym_name = "buf85"} : memref<4x2048xbf16, 2 : i32> 
    %buf84 = aie.buffer(%tile_0_2) {sym_name = "buf84"} : memref<2048xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<512x2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<512xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf107 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf105 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_61, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf106 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_59, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_62, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf107) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_7_2_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_59, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf106, %buf105, %buf107) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_60, Release, 1)
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_63, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf104 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf102 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_56, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf103 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_54, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_57, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf104) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_6_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_54, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf103, %buf102, %buf104) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_55, Release, 1)
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_58, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf101 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf99 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_51, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf100 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_49, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_52, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf101) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_5_2_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_49, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf100, %buf99, %buf101) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_50, Release, 1)
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_53, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf98 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf96 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_46, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf97 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_44, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_47, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf98) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_4_2_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_44, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf97, %buf96, %buf98) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_45, Release, 1)
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_48, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf95 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf93 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_41, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf94 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_42, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf95) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_3_2_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf94, %buf93, %buf95) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_40, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_43, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf92 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_36, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf91 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_37, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf92) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_2_2_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf91, %buf90, %buf92) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_35, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_38, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf89 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf87 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf88 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_32, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf89) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_1_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf88, %buf87, %buf89) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_30, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_33, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf86 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_27, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf84 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_26, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf85 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_27, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf86) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_0_2_26, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf85, %buf84, %buf86) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_25, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_28, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "v_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    func.func private @linalg_fill_bf16(bf16, memref<8xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
    func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_5_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_6_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_5_2, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_2, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 1)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 1)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf115 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf123 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf123 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_21, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf115 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_23, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf114 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf122 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf122 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf114 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf113 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf121 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf121 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf113 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf112 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf120 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf120 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf112 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_14, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf111 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf119 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf119 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf111 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf110 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf118 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf118 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf110 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf109 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf117 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf117 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf109 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf108 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf116 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf116 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf108 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_29_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_24_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_24_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_24_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_24_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_24_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_24_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_24_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_24_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_14(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @v_matvec_bf16_0_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048x2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<512x2048xbf16>, %arg6: memref<512xbf16>, %arg7: memref<512x2048xbf16>, %arg8: memref<512xbf16>, %arg9: memref<2048xbf16>, %arg10: memref<512xbf16>, %arg11: memref<2048xbf16>, %arg12: memref<512xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_24_0 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 0, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_24_1 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 16384, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_24_2 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 32768, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_24_3 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 49152, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_24_4 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 65536, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_24_5 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 81920, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_24_6 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 98304, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_24_7 {
        aie.dma_bd(%arg7 : memref<512x2048xbf16>, 114688, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_14 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 15 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_29_0 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 0, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_29_1 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 8, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_29_2 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 16, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_29_3 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 24, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_29_4 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 32, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_29_5 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 40, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_29_6 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 48, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_29_7 {
        aie.dma_bd(%arg8 : memref<512xbf16>, 56, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%9)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%0)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @k_matvec_bf16_0 {
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
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %lock_7_1 = aie.lock(%mem_tile_7_1, 3) {init = 1 : i32}
    %lock_7_1_0 = aie.lock(%mem_tile_7_1, 2) {init = 0 : i32}
    %lock_7_1_1 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %lock_7_1_2 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_6_1 = aie.lock(%mem_tile_6_1, 3) {init = 1 : i32}
    %lock_6_1_3 = aie.lock(%mem_tile_6_1, 2) {init = 0 : i32}
    %lock_6_1_4 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %lock_6_1_5 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_5_1 = aie.lock(%mem_tile_5_1, 3) {init = 1 : i32}
    %lock_5_1_6 = aie.lock(%mem_tile_5_1, 2) {init = 0 : i32}
    %lock_5_1_7 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_8 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 3) {init = 1 : i32}
    %lock_4_1_9 = aie.lock(%mem_tile_4_1, 2) {init = 0 : i32}
    %lock_4_1_10 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_11 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_3_1 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_12 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_13 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_14 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_15 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_16 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_17 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_18 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_19 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_20 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_21 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_22 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_23 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_24 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_25 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_26 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_27 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_28 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_29 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_30 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_31 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_32 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_33 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_34 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_35 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_36 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_37 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_38 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_39 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_40 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_41 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_42 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_43 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_4_2 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_44 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_45 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_46 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_47 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_48 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_49 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_50 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_51 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_52 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_53 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_54 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_55 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_56 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_57 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_58 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_59 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_60 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_61 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_62 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_63 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %buf83 = aie.buffer(%mem_tile_0_1) {sym_name = "buf83"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf82 = aie.buffer(%mem_tile_1_1) {sym_name = "buf82"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf81 = aie.buffer(%mem_tile_2_1) {sym_name = "buf81"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf80 = aie.buffer(%mem_tile_3_1) {sym_name = "buf80"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf79 = aie.buffer(%mem_tile_4_1) {sym_name = "buf79"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf78 = aie.buffer(%mem_tile_5_1) {sym_name = "buf78"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf77 = aie.buffer(%mem_tile_6_1) {sym_name = "buf77"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf76 = aie.buffer(%mem_tile_7_1) {sym_name = "buf76"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf75 = aie.buffer(%mem_tile_0_1) {sym_name = "buf75"} : memref<1x8xbf16, 1 : i32> 
    %buf74 = aie.buffer(%mem_tile_1_1) {sym_name = "buf74"} : memref<1x8xbf16, 1 : i32> 
    %buf73 = aie.buffer(%mem_tile_2_1) {sym_name = "buf73"} : memref<1x8xbf16, 1 : i32> 
    %buf72 = aie.buffer(%mem_tile_3_1) {sym_name = "buf72"} : memref<1x8xbf16, 1 : i32> 
    %buf71 = aie.buffer(%mem_tile_4_1) {sym_name = "buf71"} : memref<1x8xbf16, 1 : i32> 
    %buf70 = aie.buffer(%mem_tile_5_1) {sym_name = "buf70"} : memref<1x8xbf16, 1 : i32> 
    %buf69 = aie.buffer(%mem_tile_6_1) {sym_name = "buf69"} : memref<1x8xbf16, 1 : i32> 
    %buf68 = aie.buffer(%mem_tile_7_1) {sym_name = "buf68"} : memref<1x8xbf16, 1 : i32> 
    %buf67 = aie.buffer(%tile_7_2) {sym_name = "buf67"} : memref<8xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_7_2) {sym_name = "buf66"} : memref<4x2048xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_7_2) {sym_name = "buf65"} : memref<2048xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_6_2) {sym_name = "buf64"} : memref<8xbf16, 2 : i32> 
    %buf63 = aie.buffer(%tile_6_2) {sym_name = "buf63"} : memref<4x2048xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_6_2) {sym_name = "buf62"} : memref<2048xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_5_2) {sym_name = "buf61"} : memref<8xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_5_2) {sym_name = "buf60"} : memref<4x2048xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_5_2) {sym_name = "buf59"} : memref<2048xbf16, 2 : i32> 
    %buf58 = aie.buffer(%tile_4_2) {sym_name = "buf58"} : memref<8xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_4_2) {sym_name = "buf57"} : memref<4x2048xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_4_2) {sym_name = "buf56"} : memref<2048xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_3_2) {sym_name = "buf55"} : memref<8xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_3_2) {sym_name = "buf54"} : memref<4x2048xbf16, 2 : i32> 
    %buf53 = aie.buffer(%tile_3_2) {sym_name = "buf53"} : memref<2048xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_2_2) {sym_name = "buf52"} : memref<8xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_2_2) {sym_name = "buf51"} : memref<4x2048xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_2_2) {sym_name = "buf50"} : memref<2048xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_1_2) {sym_name = "buf49"} : memref<8xbf16, 2 : i32> 
    %buf48 = aie.buffer(%tile_1_2) {sym_name = "buf48"} : memref<4x2048xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_1_2) {sym_name = "buf47"} : memref<2048xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_0_2) {sym_name = "buf46"} : memref<8xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_0_2) {sym_name = "buf45"} : memref<4x2048xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_0_2) {sym_name = "buf44"} : memref<2048xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<512x2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<512xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf67 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf65 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_61, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf66 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_59, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_62, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf67) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_7_2_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_59, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf66, %buf65, %buf67) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_60, Release, 1)
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_63, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf64 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf62 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_56, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf63 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_54, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_57, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf64) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_6_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_54, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf63, %buf62, %buf64) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_55, Release, 1)
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_58, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf61 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf59 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_51, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf60 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_49, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_52, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf61) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_5_2_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_49, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf60, %buf59, %buf61) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_50, Release, 1)
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_53, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf58 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf56 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_46, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf57 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_44, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_47, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf58) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_4_2_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_44, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf57, %buf56, %buf58) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_45, Release, 1)
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_48, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf55 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf53 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_41, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_42, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf55) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_3_2_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf54, %buf53, %buf55) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_40, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_43, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf52 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf50 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_36, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf51 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_37, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf52) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_2_2_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf51, %buf50, %buf52) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_35, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_38, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf49 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf47 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf48 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_32, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf49) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_1_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf48, %buf47, %buf49) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_30, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_33, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf46 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_27, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf44 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_26, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf45 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_27, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf46) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_0_2_26, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf45, %buf44, %buf46) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_25, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_28, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "k_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    func.func private @linalg_fill_bf16(bf16, memref<8xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
    func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_5_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_6_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_5_2, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_2, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 1)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 1)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf75 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf83 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf83 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_21, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf75 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_23, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf82 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf82 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf73 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf73 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf72 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf80 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf80 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf72 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_14, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf71 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf79 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf79 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf71 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf70 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf78 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf78 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf70 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf69 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf77 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf77 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf69 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf68 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf68 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_25_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_25_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_25_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_25_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_25_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_25_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_25_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_25_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_28_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_28_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_28_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_28_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_28_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_28_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_28_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_28_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_9(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @k_matvec_bf16_0_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048x2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<512x2048xbf16>, %arg6: memref<512xbf16>, %arg7: memref<512x2048xbf16>, %arg8: memref<512xbf16>, %arg9: memref<2048xbf16>, %arg10: memref<512xbf16>, %arg11: memref<2048xbf16>, %arg12: memref<512xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_28_0 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 0, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_28_1 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 16384, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_28_2 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 32768, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_28_3 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 49152, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_28_4 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 65536, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_28_5 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 81920, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_28_6 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 98304, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_28_7 {
        aie.dma_bd(%arg5 : memref<512x2048xbf16>, 114688, 131072, [<size = 8, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_9 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 15 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_25_0 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 0, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_25_1 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 8, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_25_2 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 16, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_25_3 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 24, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_25_4 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 32, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_25_5 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 40, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_25_6 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 48, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_25_7 {
        aie.dma_bd(%arg6 : memref<512xbf16>, 56, 64, [<size = 8, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%9)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%0)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @q_matvec_bf16_0 {
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
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %lock_7_1 = aie.lock(%mem_tile_7_1, 3) {init = 1 : i32}
    %lock_7_1_0 = aie.lock(%mem_tile_7_1, 2) {init = 0 : i32}
    %lock_7_1_1 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %lock_7_1_2 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_6_1 = aie.lock(%mem_tile_6_1, 3) {init = 1 : i32}
    %lock_6_1_3 = aie.lock(%mem_tile_6_1, 2) {init = 0 : i32}
    %lock_6_1_4 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %lock_6_1_5 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_5_1 = aie.lock(%mem_tile_5_1, 3) {init = 1 : i32}
    %lock_5_1_6 = aie.lock(%mem_tile_5_1, 2) {init = 0 : i32}
    %lock_5_1_7 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_8 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 3) {init = 1 : i32}
    %lock_4_1_9 = aie.lock(%mem_tile_4_1, 2) {init = 0 : i32}
    %lock_4_1_10 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_11 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_3_1 = aie.lock(%mem_tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_12 = aie.lock(%mem_tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_13 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_14 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_15 = aie.lock(%mem_tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_16 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_17 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_18 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_19 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_20 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_21 = aie.lock(%mem_tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_22 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_23 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_24 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_25 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_26 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_27 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_28 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_29 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_30 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_31 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_32 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_33 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_34 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_35 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_36 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_37 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_38 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_39 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_40 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_41 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_42 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_43 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_4_2 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_44 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_45 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_46 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_47 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_48 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_49 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_50 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_51 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_52 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_53 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_54 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_55 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_56 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_57 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_58 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_59 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_60 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_61 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_62 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_63 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %buf43 = aie.buffer(%mem_tile_0_1) {sym_name = "buf43"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf42 = aie.buffer(%mem_tile_1_1) {sym_name = "buf42"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf41 = aie.buffer(%mem_tile_2_1) {sym_name = "buf41"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf40 = aie.buffer(%mem_tile_3_1) {sym_name = "buf40"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf39 = aie.buffer(%mem_tile_4_1) {sym_name = "buf39"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf38 = aie.buffer(%mem_tile_5_1) {sym_name = "buf38"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf37 = aie.buffer(%mem_tile_6_1) {sym_name = "buf37"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf36 = aie.buffer(%mem_tile_7_1) {sym_name = "buf36"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf35 = aie.buffer(%mem_tile_0_1) {sym_name = "buf35"} : memref<1x8xbf16, 1 : i32> 
    %buf34 = aie.buffer(%mem_tile_1_1) {sym_name = "buf34"} : memref<1x8xbf16, 1 : i32> 
    %buf33 = aie.buffer(%mem_tile_2_1) {sym_name = "buf33"} : memref<1x8xbf16, 1 : i32> 
    %buf32 = aie.buffer(%mem_tile_3_1) {sym_name = "buf32"} : memref<1x8xbf16, 1 : i32> 
    %buf31 = aie.buffer(%mem_tile_4_1) {sym_name = "buf31"} : memref<1x8xbf16, 1 : i32> 
    %buf30 = aie.buffer(%mem_tile_5_1) {sym_name = "buf30"} : memref<1x8xbf16, 1 : i32> 
    %buf29 = aie.buffer(%mem_tile_6_1) {sym_name = "buf29"} : memref<1x8xbf16, 1 : i32> 
    %buf28 = aie.buffer(%mem_tile_7_1) {sym_name = "buf28"} : memref<1x8xbf16, 1 : i32> 
    %buf27 = aie.buffer(%tile_7_2) {sym_name = "buf27"} : memref<8xbf16, 2 : i32> 
    %buf26 = aie.buffer(%tile_7_2) {sym_name = "buf26"} : memref<4x2048xbf16, 2 : i32> 
    %buf25 = aie.buffer(%tile_7_2) {sym_name = "buf25"} : memref<2048xbf16, 2 : i32> 
    %buf24 = aie.buffer(%tile_6_2) {sym_name = "buf24"} : memref<8xbf16, 2 : i32> 
    %buf23 = aie.buffer(%tile_6_2) {sym_name = "buf23"} : memref<4x2048xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_6_2) {sym_name = "buf22"} : memref<2048xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_5_2) {sym_name = "buf21"} : memref<8xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_5_2) {sym_name = "buf20"} : memref<4x2048xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_5_2) {sym_name = "buf19"} : memref<2048xbf16, 2 : i32> 
    %buf18 = aie.buffer(%tile_4_2) {sym_name = "buf18"} : memref<8xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_4_2) {sym_name = "buf17"} : memref<4x2048xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_4_2) {sym_name = "buf16"} : memref<2048xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_3_2) {sym_name = "buf15"} : memref<8xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_3_2) {sym_name = "buf14"} : memref<4x2048xbf16, 2 : i32> 
    %buf13 = aie.buffer(%tile_3_2) {sym_name = "buf13"} : memref<2048xbf16, 2 : i32> 
    %buf12 = aie.buffer(%tile_2_2) {sym_name = "buf12"} : memref<8xbf16, 2 : i32> 
    %buf11 = aie.buffer(%tile_2_2) {sym_name = "buf11"} : memref<4x2048xbf16, 2 : i32> 
    %buf10 = aie.buffer(%tile_2_2) {sym_name = "buf10"} : memref<2048xbf16, 2 : i32> 
    %buf9 = aie.buffer(%tile_1_2) {sym_name = "buf9"} : memref<8xbf16, 2 : i32> 
    %buf8 = aie.buffer(%tile_1_2) {sym_name = "buf8"} : memref<4x2048xbf16, 2 : i32> 
    %buf7 = aie.buffer(%tile_1_2) {sym_name = "buf7"} : memref<2048xbf16, 2 : i32> 
    %buf6 = aie.buffer(%tile_0_2) {sym_name = "buf6"} : memref<8xbf16, 2 : i32> 
    %buf5 = aie.buffer(%tile_0_2) {sym_name = "buf5"} : memref<4x2048xbf16, 2 : i32> 
    %buf4 = aie.buffer(%tile_0_2) {sym_name = "buf4"} : memref<2048xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048x2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf27 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf25 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_61, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_59, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_62, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf27) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_7_2_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_59, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf26, %buf25, %buf27) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_60, Release, 1)
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_63, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf24 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_56, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf23 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_54, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_57, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf24) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_6_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_54, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf23, %buf22, %buf24) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_55, Release, 1)
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_58, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf21 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf19 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_51, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf20 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_49, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_52, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf21) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_5_2_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_49, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf20, %buf19, %buf21) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_50, Release, 1)
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_53, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf18 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf16 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_46, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf17 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_44, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_47, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf18) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_4_2_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_44, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf17, %buf16, %buf18) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_45, Release, 1)
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_48, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf15 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_41, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_42, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf15) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_3_2_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf14, %buf13, %buf15) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_40, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_43, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_36, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_37, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf12) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_2_2_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf11, %buf10, %buf12) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_35, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_38, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_32, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf9) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_1_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf8, %buf7, %buf9) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_30, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_33, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_27, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_26, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c2048_i32 = arith.constant 2048 : i32
      %c4_i32 = arith.constant 4 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_27, AcquireGreaterEqual, 1)
      func.call @linalg_fill_bf16(%cst, %buf6) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_0_2_26, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf5, %buf4, %buf6) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_25, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_28, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "q_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    func.func private @linalg_fill_bf16(bf16, memref<8xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
    func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_5_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_6_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_5_2, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_2, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 1)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 1)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf35 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf43 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf43 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_21, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf35 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_23, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf42 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf42 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf33 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf41 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf41 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf33 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf32 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf40 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf40 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf32 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_14, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf31 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf39 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf39 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf31 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf29 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf37 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf37 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf29 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf28 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf36 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf36 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf28 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_27_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_27_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_27_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_27_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_27_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_27_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_27_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_27_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_26_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_26_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_26_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_26_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_26_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_26_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_26_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_26_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_4(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @q_matvec_bf16_0_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048x2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<512x2048xbf16>, %arg6: memref<512xbf16>, %arg7: memref<512x2048xbf16>, %arg8: memref<512xbf16>, %arg9: memref<2048xbf16>, %arg10: memref<512xbf16>, %arg11: memref<2048xbf16>, %arg12: memref<512xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_26_0 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 0, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_26_1 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 16384, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_26_2 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 32768, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_26_3 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 49152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_26_4 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 65536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_26_5 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 81920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_26_6 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 98304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_26_7 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 114688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_4 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_27_0 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 0, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_27_1 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 8, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_27_2 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 16, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_27_3 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 24, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_27_4 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 32, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_27_5 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 40, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_27_6 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 48, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_27_7 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 56, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%9)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%0)
      %17 = aiex.dma_configure_task_for @air_channel_26_0 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2097152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_26_1 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2113536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_26_2 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2129920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_26_3 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2146304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_26_4 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2162688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_26_5 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2179072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_26_6 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2195456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @air_channel_26_7 {
        aie.dma_bd(%arg3 : memref<2048x2048xbf16>, 2211840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @air_channel_4 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @air_channel_27_0 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1024, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @air_channel_27_1 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1032, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @air_channel_27_2 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1040, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @air_channel_27_3 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1048, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @air_channel_27_4 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1056, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @air_channel_27_5 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1064, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @air_channel_27_6 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @air_channel_27_7 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      aiex.dma_await_task(%33)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%31)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%26)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%17)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @r_rms_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<2048xbf16, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<2048xbf16, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<2048xbf16, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<16xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      func.call @rms_norm_2048_bf16(%buf3, %buf1, %buf2, %buf0) : (memref<2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<16xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.use_lock(%lock_0_2_1, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "r_rms_herd", air.herd_size = array<i64: 1, 1>, link_with = "rms_norm_2048_bf16.o"}
    func.func private @rms_norm_2048_bf16(memref<2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<16xbf16, 2 : i32>) attributes {link_with = "rms_norm_2048_bf16.o", llvm.emit_c_interface}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_2(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @r_rms_seg_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048x2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<512x2048xbf16>, %arg6: memref<512xbf16>, %arg7: memref<512x2048xbf16>, %arg8: memref<512xbf16>, %arg9: memref<2048xbf16>, %arg10: memref<512xbf16>, %arg11: memref<2048xbf16>, %arg12: memref<512xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_2 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) {
    aie.runtime_sequence @rms_gemv_rope(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048x2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<512x2048xbf16>, %arg6: memref<512xbf16>, %arg7: memref<512x2048xbf16>, %arg8: memref<512xbf16>, %arg9: memref<2048xbf16>, %arg10: memref<512xbf16>, %arg11: memref<2048xbf16>, %arg12: memref<512xbf16>) {
      aiex.configure @r_rms_seg {
        aiex.run @r_rms_seg_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048x2048xbf16>, memref<2048xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>)
      }
      aiex.configure @q_matvec_bf16_0 {
        aiex.run @q_matvec_bf16_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048x2048xbf16>, memref<2048xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>)
      }
      aiex.configure @k_matvec_bf16_0 {
        aiex.run @k_matvec_bf16_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048x2048xbf16>, memref<2048xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>)
      }
      aiex.configure @v_matvec_bf16_0 {
        aiex.run @v_matvec_bf16_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048x2048xbf16>, memref<2048xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>)
      }
      aiex.configure @rq_rope_seg {
        aiex.run @rq_rope_seg_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048x2048xbf16>, memref<2048xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>)
      }
      aiex.configure @rk_rope_seg {
        aiex.run @rk_rope_seg_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048x2048xbf16>, memref<2048xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<512x2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>, memref<2048xbf16>, memref<512xbf16>)
      }
    }
  }
}
