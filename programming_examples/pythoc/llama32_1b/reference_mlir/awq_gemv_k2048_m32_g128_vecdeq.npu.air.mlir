module {
  aie.device(npu2) @awq_gemv_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 7) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 6) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<2048xbf16, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<32768xui8, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<1024xbf16, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<32xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<32768xui8>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<1024xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<32xbf16>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<32xbf16, 2 : i32>, 0, 32) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<32768xui8, 2 : i32>, 0, 32768) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      func.call @awq_gemv_u4_bf16(%buf3, %buf2, %buf1, %buf0) : (memref<2048xbf16, 2 : i32>, memref<32768xui8, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<32xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_6, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "awq_gemv_herd", air.herd_size = array<i64: 1, 1>, link_with = "awq_gemv_k2048_m32_g128_vecdeq_pythoc.o"}
    func.func private @awq_gemv_u4_bf16(memref<2048xbf16, 2 : i32>, memref<32768xui8, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<32xbf16, 2 : i32>) attributes {link_with = "awq_gemv_k2048_m32_g128_vecdeq_pythoc.o", llvm.emit_c_interface}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_3(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_2(%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence @awq_gemv_seg_sequence(%arg0: memref<2048xbf16>, %arg1: memref<32768xui8>, %arg2: memref<1024xbf16>, %arg3: memref<32xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg1 : memref<32768xui8>, 0, 32768, [<size = 64, stride = 512>, <size = 512, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg2 : memref<1024xbf16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_3 {
        aie.dma_bd(%arg3 : memref<32xbf16>, 0, 32, [<size = 32, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_free_task(%0)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) {
    aie.runtime_sequence @awq_gemv(%arg0: memref<2048xbf16>, %arg1: memref<32768xui8>, %arg2: memref<1024xbf16>, %arg3: memref<32xbf16>) {
      aiex.configure @awq_gemv_seg {
        aiex.run @awq_gemv_seg_sequence(%arg0, %arg1, %arg2, %arg3) : (memref<2048xbf16>, memref<32768xui8>, memref<1024xbf16>, memref<32xbf16>)
      }
    }
  }
}
