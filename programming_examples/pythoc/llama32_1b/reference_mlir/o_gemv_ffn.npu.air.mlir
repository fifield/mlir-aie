#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @a2_eltwise_add_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_5 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_6 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_7 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_8 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_9 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_10 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_11 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_12 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_13 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_14 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_15 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_16 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_17 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_18 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_19 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_4_2 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_20 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_21 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_22 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_23 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_24 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_25 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_26 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_27 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_28 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_29 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_30 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_31 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_32 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_33 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_34 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_35 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_36 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_37 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_38 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_39 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %buf235 = aie.buffer(%tile_7_2) {sym_name = "buf235"} : memref<256xbf16, 2 : i32> 
    %buf234 = aie.buffer(%tile_7_2) {sym_name = "buf234"} : memref<256xbf16, 2 : i32> 
    %buf233 = aie.buffer(%tile_7_2) {sym_name = "buf233"} : memref<256xbf16, 2 : i32> 
    %buf232 = aie.buffer(%tile_6_2) {sym_name = "buf232"} : memref<256xbf16, 2 : i32> 
    %buf231 = aie.buffer(%tile_6_2) {sym_name = "buf231"} : memref<256xbf16, 2 : i32> 
    %buf230 = aie.buffer(%tile_6_2) {sym_name = "buf230"} : memref<256xbf16, 2 : i32> 
    %buf229 = aie.buffer(%tile_5_2) {sym_name = "buf229"} : memref<256xbf16, 2 : i32> 
    %buf228 = aie.buffer(%tile_5_2) {sym_name = "buf228"} : memref<256xbf16, 2 : i32> 
    %buf227 = aie.buffer(%tile_5_2) {sym_name = "buf227"} : memref<256xbf16, 2 : i32> 
    %buf226 = aie.buffer(%tile_4_2) {sym_name = "buf226"} : memref<256xbf16, 2 : i32> 
    %buf225 = aie.buffer(%tile_4_2) {sym_name = "buf225"} : memref<256xbf16, 2 : i32> 
    %buf224 = aie.buffer(%tile_4_2) {sym_name = "buf224"} : memref<256xbf16, 2 : i32> 
    %buf223 = aie.buffer(%tile_3_2) {sym_name = "buf223"} : memref<256xbf16, 2 : i32> 
    %buf222 = aie.buffer(%tile_3_2) {sym_name = "buf222"} : memref<256xbf16, 2 : i32> 
    %buf221 = aie.buffer(%tile_3_2) {sym_name = "buf221"} : memref<256xbf16, 2 : i32> 
    %buf220 = aie.buffer(%tile_2_2) {sym_name = "buf220"} : memref<256xbf16, 2 : i32> 
    %buf219 = aie.buffer(%tile_2_2) {sym_name = "buf219"} : memref<256xbf16, 2 : i32> 
    %buf218 = aie.buffer(%tile_2_2) {sym_name = "buf218"} : memref<256xbf16, 2 : i32> 
    %buf217 = aie.buffer(%tile_1_2) {sym_name = "buf217"} : memref<256xbf16, 2 : i32> 
    %buf216 = aie.buffer(%tile_1_2) {sym_name = "buf216"} : memref<256xbf16, 2 : i32> 
    %buf215 = aie.buffer(%tile_1_2) {sym_name = "buf215"} : memref<256xbf16, 2 : i32> 
    %buf214 = aie.buffer(%tile_0_2) {sym_name = "buf214"} : memref<256xbf16, 2 : i32> 
    %buf213 = aie.buffer(%tile_0_2) {sym_name = "buf213"} : memref<256xbf16, 2 : i32> 
    %buf212 = aie.buffer(%tile_0_2) {sym_name = "buf212"} : memref<256xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf233 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf235 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_38, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_35, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf235[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf234[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf233[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_36, Release, 1)
      aie.use_lock(%lock_7_2, Release, 1)
      aie.use_lock(%lock_7_2_39, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf232 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf231 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_30, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf232[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf231[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf230[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_31, Release, 1)
      aie.use_lock(%lock_6_2, Release, 1)
      aie.use_lock(%lock_6_2_34, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf227 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf229 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf228 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_25, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_28, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_25, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf229[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf228[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf227[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_26, Release, 1)
      aie.use_lock(%lock_5_2, Release, 1)
      aie.use_lock(%lock_5_2_29, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf224 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf226 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf225 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_23, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_20, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf226[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf225[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf224[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_21, Release, 1)
      aie.use_lock(%lock_4_2, Release, 1)
      aie.use_lock(%lock_4_2_24, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf221 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf223 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_17, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf222 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_15, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_18, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_17, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_15, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf223[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf222[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf221[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_16, Release, 1)
      aie.use_lock(%lock_3_2, Release, 1)
      aie.use_lock(%lock_3_2_19, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf218 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf220 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_12, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf219 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_10, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_13, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_12, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_10, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf220[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf219[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf218[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_11, Release, 1)
      aie.use_lock(%lock_2_2, Release, 1)
      aie.use_lock(%lock_2_2_14, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf215 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf217 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_7, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf216 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_5, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_8, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_7, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf217[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf216[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf215[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_6, Release, 1)
      aie.use_lock(%lock_1_2, Release, 1)
      aie.use_lock(%lock_1_2_9, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf212 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf214 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf213 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf214[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf213[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf212[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "a2_herd_0", air.herd_size = array<i64: 8, 1>}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %tile_5_2, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %tile_6_2, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_5_2, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%tile_7_2, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_31_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_31_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_31_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_31_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_31_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_31_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_31_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_31_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_29_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_29_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_29_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_29_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_29_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_29_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_29_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_29_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_30_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_30_1(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_30_2(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_30_3(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_30_4(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_30_5(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_30_6(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_30_7(%shim_noc_tile_7_0, MM2S, 1)
    aie.runtime_sequence @a2_eltwise_add_seg_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_29_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 0, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_29_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 256, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_29_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 512, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_29_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 768, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_29_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1024, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_29_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1280, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_29_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1536, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_29_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1792, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_30_0 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 0, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_30_1 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 256, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_30_2 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 512, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_30_3 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 768, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_30_4 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1024, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_30_5 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1280, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_30_6 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1536, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_30_7 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1792, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_31_0 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 0, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @air_channel_31_1 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 256, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_31_2 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 512, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_31_3 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 768, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_31_4 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 1024, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_31_5 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 1280, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_31_6 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 1536, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_31_7 {
        aie.dma_bd(%arg14 : memref<2048xbf16>, 1792, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%12)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%17)
      aiex.dma_free_task(%15)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%18)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%2)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @dg_matvec_bf16_0 {
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
    %buf211 = aie.buffer(%mem_tile_0_1) {sym_name = "buf211"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf210 = aie.buffer(%mem_tile_1_1) {sym_name = "buf210"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf209 = aie.buffer(%mem_tile_2_1) {sym_name = "buf209"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf208 = aie.buffer(%mem_tile_3_1) {sym_name = "buf208"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf207 = aie.buffer(%mem_tile_4_1) {sym_name = "buf207"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf206 = aie.buffer(%mem_tile_5_1) {sym_name = "buf206"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf205 = aie.buffer(%mem_tile_6_1) {sym_name = "buf205"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf204 = aie.buffer(%mem_tile_7_1) {sym_name = "buf204"} : memref<1x2x8192xbf16, 1 : i32> 
    %buf203 = aie.buffer(%mem_tile_0_1) {sym_name = "buf203"} : memref<1x2xbf16, 1 : i32> 
    %buf202 = aie.buffer(%mem_tile_1_1) {sym_name = "buf202"} : memref<1x2xbf16, 1 : i32> 
    %buf201 = aie.buffer(%mem_tile_2_1) {sym_name = "buf201"} : memref<1x2xbf16, 1 : i32> 
    %buf200 = aie.buffer(%mem_tile_3_1) {sym_name = "buf200"} : memref<1x2xbf16, 1 : i32> 
    %buf199 = aie.buffer(%mem_tile_4_1) {sym_name = "buf199"} : memref<1x2xbf16, 1 : i32> 
    %buf198 = aie.buffer(%mem_tile_5_1) {sym_name = "buf198"} : memref<1x2xbf16, 1 : i32> 
    %buf197 = aie.buffer(%mem_tile_6_1) {sym_name = "buf197"} : memref<1x2xbf16, 1 : i32> 
    %buf196 = aie.buffer(%mem_tile_7_1) {sym_name = "buf196"} : memref<1x2xbf16, 1 : i32> 
    %buf195 = aie.buffer(%tile_7_2) {sym_name = "buf195"} : memref<2xbf16, 2 : i32> 
    %buf194 = aie.buffer(%tile_7_2) {sym_name = "buf194"} : memref<1x8192xbf16, 2 : i32> 
    %buf193 = aie.buffer(%tile_7_2) {sym_name = "buf193"} : memref<8192xbf16, 2 : i32> 
    %buf192 = aie.buffer(%tile_6_2) {sym_name = "buf192"} : memref<2xbf16, 2 : i32> 
    %buf191 = aie.buffer(%tile_6_2) {sym_name = "buf191"} : memref<1x8192xbf16, 2 : i32> 
    %buf190 = aie.buffer(%tile_6_2) {sym_name = "buf190"} : memref<8192xbf16, 2 : i32> 
    %buf189 = aie.buffer(%tile_5_2) {sym_name = "buf189"} : memref<2xbf16, 2 : i32> 
    %buf188 = aie.buffer(%tile_5_2) {sym_name = "buf188"} : memref<1x8192xbf16, 2 : i32> 
    %buf187 = aie.buffer(%tile_5_2) {sym_name = "buf187"} : memref<8192xbf16, 2 : i32> 
    %buf186 = aie.buffer(%tile_4_2) {sym_name = "buf186"} : memref<2xbf16, 2 : i32> 
    %buf185 = aie.buffer(%tile_4_2) {sym_name = "buf185"} : memref<1x8192xbf16, 2 : i32> 
    %buf184 = aie.buffer(%tile_4_2) {sym_name = "buf184"} : memref<8192xbf16, 2 : i32> 
    %buf183 = aie.buffer(%tile_3_2) {sym_name = "buf183"} : memref<2xbf16, 2 : i32> 
    %buf182 = aie.buffer(%tile_3_2) {sym_name = "buf182"} : memref<1x8192xbf16, 2 : i32> 
    %buf181 = aie.buffer(%tile_3_2) {sym_name = "buf181"} : memref<8192xbf16, 2 : i32> 
    %buf180 = aie.buffer(%tile_2_2) {sym_name = "buf180"} : memref<2xbf16, 2 : i32> 
    %buf179 = aie.buffer(%tile_2_2) {sym_name = "buf179"} : memref<1x8192xbf16, 2 : i32> 
    %buf178 = aie.buffer(%tile_2_2) {sym_name = "buf178"} : memref<8192xbf16, 2 : i32> 
    %buf177 = aie.buffer(%tile_1_2) {sym_name = "buf177"} : memref<2xbf16, 2 : i32> 
    %buf176 = aie.buffer(%tile_1_2) {sym_name = "buf176"} : memref<1x8192xbf16, 2 : i32> 
    %buf175 = aie.buffer(%tile_1_2) {sym_name = "buf175"} : memref<8192xbf16, 2 : i32> 
    %buf174 = aie.buffer(%tile_0_2) {sym_name = "buf174"} : memref<2xbf16, 2 : i32> 
    %buf173 = aie.buffer(%tile_0_2) {sym_name = "buf173"} : memref<1x8192xbf16, 2 : i32> 
    %buf172 = aie.buffer(%tile_0_2) {sym_name = "buf172"} : memref<8192xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048x8192xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<8192xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf195 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf193 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_61, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf194 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_59, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_62, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf195) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_7_2_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_59, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf194, %buf193, %buf195) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_60, Release, 1)
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_63, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf192 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf190 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_56, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf191 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_54, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_57, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf192) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_6_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_54, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf191, %buf190, %buf192) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_55, Release, 1)
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_58, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf189 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf187 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_51, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf188 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_49, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_52, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf189) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_5_2_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_49, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf188, %buf187, %buf189) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_50, Release, 1)
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_53, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf186 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf184 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_46, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf185 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_44, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_47, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf186) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_4_2_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_44, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf185, %buf184, %buf186) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_45, Release, 1)
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_48, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf183 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf181 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_41, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf182 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_42, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf183) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_3_2_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf182, %buf181, %buf183) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_40, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_43, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf180 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf178 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_36, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf179 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_37, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf180) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_2_2_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf179, %buf178, %buf180) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_35, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_38, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf177 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf175 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf176 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_32, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf177) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_1_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf176, %buf175, %buf177) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_30, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_33, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf174 : memref<2xbf16, 2 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_27, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf172 : memref<8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_26, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf173 : memref<1x8192xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %cst = arith.constant 0.000000e+00 : bf16
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_27, AcquireGreaterEqual, 1)
      func.call @dg_linalg_fill_bf16(%cst, %buf174) : (bf16, memref<2xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        aie.use_lock(%lock_0_2_26, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
        func.call @dg_matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %arg0, %buf173, %buf172, %buf174) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_25, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_28, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "dg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv_k8192.o"}
    func.func private @dg_linalg_fill_bf16(bf16, memref<2xbf16, 2 : i32>) attributes {link_with = "mv_k8192.o", llvm.emit_c_interface}
    func.func private @dg_matvec_vectorized_bf16_bf16(i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) attributes {link_with = "mv_k8192.o", llvm.emit_c_interface}
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
      aie.dma_bd(%buf203 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf211 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf211 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_21, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf203 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_23, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf202 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf210 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf210 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf202 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf201 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf209 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf209 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf201 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf200 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf208 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf208 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf200 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_14, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf199 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf207 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf207 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf199 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf198 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf206 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf206 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf198 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf197 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf205 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf205 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf197 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf196 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf204 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf204 : memref<1x2x8192xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf196 : memref<1x2xbf16, 1 : i32>, 0, 2) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_37_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_37_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_37_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_37_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_37_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_37_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_37_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_37_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_34_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_34_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_34_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_34_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_34_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_34_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_34_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_34_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_25(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @dg_matvec_bf16_0_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 0, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 16384, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 32768, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 49152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 65536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 81920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 98304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 114688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 0, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 2, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 4, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 6, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 8, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 10, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 12, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 14, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
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
      %17 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2097152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2113536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2129920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2146304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2162688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2179072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2195456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 2211840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 256, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 258, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 260, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 262, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 264, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 266, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 268, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 270, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
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
      %34 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4194304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4210688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4227072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4243456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4259840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4276224, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4292608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 4308992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 512, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 514, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 516, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 518, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 520, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 522, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 524, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 526, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      aiex.dma_await_task(%50)
      aiex.dma_await_task(%49)
      aiex.dma_await_task(%48)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%43)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%34)
      %51 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6291456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6307840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6324224, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6340608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6356992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6373376, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6389760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 6406144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 768, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 770, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 772, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 774, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 776, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 778, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 780, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 782, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      aiex.dma_await_task(%67)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%60)
      aiex.dma_free_task(%59)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%51)
      %68 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8388608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8404992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8421376, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8437760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8454144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8470528, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8486912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 8503296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1024, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1026, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1028, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1030, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1032, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1034, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1036, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1038, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%83)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%77)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%68)
      %85 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10485760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10502144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10518528, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10534912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10551296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10567680, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10584064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 10600448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1280, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1282, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1284, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1286, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1288, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1290, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1292, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1294, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%94)
      aiex.dma_free_task(%93)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%85)
      %102 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12582912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12599296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12615680, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12632064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12648448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12664832, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12681216, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 12697600, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1536, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1538, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1540, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1542, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1544, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1546, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1548, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1550, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%111)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%102)
      %119 = aiex.dma_configure_task_for @air_channel_34_0 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14680064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @air_channel_34_1 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14696448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @air_channel_34_2 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14712832, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @air_channel_34_3 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14729216, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @air_channel_34_4 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14745600, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @air_channel_34_5 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14761984, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @air_channel_34_6 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14778368, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @air_channel_34_7 {
        aie.dma_bd(%arg12 : memref<2048x8192xbf16>, 14794752, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @air_channel_25 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @air_channel_37_0 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1792, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @air_channel_37_1 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1794, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @air_channel_37_2 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1796, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @air_channel_37_3 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1798, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @air_channel_37_4 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1800, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @air_channel_37_5 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1802, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @air_channel_37_6 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1804, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @air_channel_37_7 {
        aie.dma_bd(%arg13 : memref<2048xbf16>, 1806, 32, [<size = 16, stride = 16>, <size = 2, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      aiex.dma_await_task(%135)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%131)
      aiex.dma_await_task(%130)
      aiex.dma_await_task(%129)
      aiex.dma_await_task(%128)
      aiex.dma_free_task(%127)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%119)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @sw_silu_mul_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_5 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_6 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_7 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_8 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_9 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_10 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_11 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_12 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_13 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_14 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_15 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_16 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_17 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_18 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_19 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_4_2 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_20 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_21 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_22 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_23 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_24 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_25 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_26 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_27 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_28 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_29 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_30 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_31 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_32 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_33 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_34 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_35 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_36 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_37 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_38 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_39 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %buf171 = aie.buffer(%tile_7_2) {sym_name = "buf171"} : memref<1024xbf16, 2 : i32> 
    %buf170 = aie.buffer(%tile_7_2) {sym_name = "buf170"} : memref<1024xbf16, 2 : i32> 
    %buf169 = aie.buffer(%tile_7_2) {sym_name = "buf169"} : memref<1024xbf16, 2 : i32> 
    %buf168 = aie.buffer(%tile_6_2) {sym_name = "buf168"} : memref<1024xbf16, 2 : i32> 
    %buf167 = aie.buffer(%tile_6_2) {sym_name = "buf167"} : memref<1024xbf16, 2 : i32> 
    %buf166 = aie.buffer(%tile_6_2) {sym_name = "buf166"} : memref<1024xbf16, 2 : i32> 
    %buf165 = aie.buffer(%tile_5_2) {sym_name = "buf165"} : memref<1024xbf16, 2 : i32> 
    %buf164 = aie.buffer(%tile_5_2) {sym_name = "buf164"} : memref<1024xbf16, 2 : i32> 
    %buf163 = aie.buffer(%tile_5_2) {sym_name = "buf163"} : memref<1024xbf16, 2 : i32> 
    %buf162 = aie.buffer(%tile_4_2) {sym_name = "buf162"} : memref<1024xbf16, 2 : i32> 
    %buf161 = aie.buffer(%tile_4_2) {sym_name = "buf161"} : memref<1024xbf16, 2 : i32> 
    %buf160 = aie.buffer(%tile_4_2) {sym_name = "buf160"} : memref<1024xbf16, 2 : i32> 
    %buf159 = aie.buffer(%tile_3_2) {sym_name = "buf159"} : memref<1024xbf16, 2 : i32> 
    %buf158 = aie.buffer(%tile_3_2) {sym_name = "buf158"} : memref<1024xbf16, 2 : i32> 
    %buf157 = aie.buffer(%tile_3_2) {sym_name = "buf157"} : memref<1024xbf16, 2 : i32> 
    %buf156 = aie.buffer(%tile_2_2) {sym_name = "buf156"} : memref<1024xbf16, 2 : i32> 
    %buf155 = aie.buffer(%tile_2_2) {sym_name = "buf155"} : memref<1024xbf16, 2 : i32> 
    %buf154 = aie.buffer(%tile_2_2) {sym_name = "buf154"} : memref<1024xbf16, 2 : i32> 
    %buf153 = aie.buffer(%tile_1_2) {sym_name = "buf153"} : memref<1024xbf16, 2 : i32> 
    %buf152 = aie.buffer(%tile_1_2) {sym_name = "buf152"} : memref<1024xbf16, 2 : i32> 
    %buf151 = aie.buffer(%tile_1_2) {sym_name = "buf151"} : memref<1024xbf16, 2 : i32> 
    %buf150 = aie.buffer(%tile_0_2) {sym_name = "buf150"} : memref<1024xbf16, 2 : i32> 
    %buf149 = aie.buffer(%tile_0_2) {sym_name = "buf149"} : memref<1024xbf16, 2 : i32> 
    %buf148 = aie.buffer(%tile_0_2) {sym_name = "buf148"} : memref<1024xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<8192xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<8192xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<8192xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf169 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf171 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf170 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_38, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_35, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf171, %buf170, %buf169, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_7_2_36, Release, 1)
      aie.use_lock(%lock_7_2, Release, 1)
      aie.use_lock(%lock_7_2_39, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf166 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf168 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf167 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_30, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf168, %buf167, %buf166, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_6_2_31, Release, 1)
      aie.use_lock(%lock_6_2, Release, 1)
      aie.use_lock(%lock_6_2_34, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf163 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf165 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf164 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_25, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_28, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_25, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf165, %buf164, %buf163, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_5_2_26, Release, 1)
      aie.use_lock(%lock_5_2, Release, 1)
      aie.use_lock(%lock_5_2_29, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf160 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf162 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf161 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_23, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_20, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf162, %buf161, %buf160, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_4_2_21, Release, 1)
      aie.use_lock(%lock_4_2, Release, 1)
      aie.use_lock(%lock_4_2_24, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf157 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf159 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_17, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf158 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_15, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_18, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_17, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_15, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf159, %buf158, %buf157, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_3_2_16, Release, 1)
      aie.use_lock(%lock_3_2, Release, 1)
      aie.use_lock(%lock_3_2_19, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf154 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf156 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_12, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf155 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_10, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_13, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_12, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_10, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf156, %buf155, %buf154, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_2_2_11, Release, 1)
      aie.use_lock(%lock_2_2, Release, 1)
      aie.use_lock(%lock_2_2_14, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf151 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf153 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_7, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf152 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_5, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_8, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_7, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf153, %buf152, %buf151, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_1_2_6, Release, 1)
      aie.use_lock(%lock_1_2, Release, 1)
      aie.use_lock(%lock_1_2_9, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf148 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf150 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf149 : memref<1024xbf16, 2 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      func.call @silu_and_mul_bf16(%buf150, %buf149, %buf148, %c1024_i32) : (memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "sw_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "silu_and_mul.o"}
    func.func private @silu_and_mul_bf16(memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) attributes {link_with = "silu_and_mul.o", llvm.emit_c_interface}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %tile_5_2, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %tile_6_2, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_5_2, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%tile_7_2, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_23_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_23_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_23_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_23_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_23_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_23_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_23_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_23_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_21_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_21_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_21_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_21_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_21_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_21_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_21_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_21_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_22_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_22_1(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_22_2(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_22_3(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_22_4(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_22_5(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_22_6(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_22_7(%shim_noc_tile_7_0, MM2S, 1)
    aie.runtime_sequence @sw_silu_mul_seg_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_21_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_21_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1024, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_21_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2048, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_21_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3072, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_21_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4096, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_21_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5120, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_21_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6144, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_21_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7168, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_22_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_22_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1024, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_22_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2048, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_22_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3072, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_22_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4096, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_22_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5120, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_22_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6144, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_22_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7168, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_23_0 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @air_channel_23_1 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 1024, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_23_2 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 2048, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_23_3 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 3072, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_23_4 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 4096, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_23_5 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 5120, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_23_6 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 6144, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_23_7 {
        aie.dma_bd(%arg11 : memref<8192xbf16>, 7168, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%12)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%17)
      aiex.dma_free_task(%15)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%18)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%2)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @ug_matvec_bf16_0 {
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
    %buf147 = aie.buffer(%mem_tile_0_1) {sym_name = "buf147"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf146 = aie.buffer(%mem_tile_1_1) {sym_name = "buf146"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf145 = aie.buffer(%mem_tile_2_1) {sym_name = "buf145"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf144 = aie.buffer(%mem_tile_3_1) {sym_name = "buf144"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf143 = aie.buffer(%mem_tile_4_1) {sym_name = "buf143"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf142 = aie.buffer(%mem_tile_5_1) {sym_name = "buf142"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf141 = aie.buffer(%mem_tile_6_1) {sym_name = "buf141"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf140 = aie.buffer(%mem_tile_7_1) {sym_name = "buf140"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf139 = aie.buffer(%mem_tile_0_1) {sym_name = "buf139"} : memref<1x8xbf16, 1 : i32> 
    %buf138 = aie.buffer(%mem_tile_1_1) {sym_name = "buf138"} : memref<1x8xbf16, 1 : i32> 
    %buf137 = aie.buffer(%mem_tile_2_1) {sym_name = "buf137"} : memref<1x8xbf16, 1 : i32> 
    %buf136 = aie.buffer(%mem_tile_3_1) {sym_name = "buf136"} : memref<1x8xbf16, 1 : i32> 
    %buf135 = aie.buffer(%mem_tile_4_1) {sym_name = "buf135"} : memref<1x8xbf16, 1 : i32> 
    %buf134 = aie.buffer(%mem_tile_5_1) {sym_name = "buf134"} : memref<1x8xbf16, 1 : i32> 
    %buf133 = aie.buffer(%mem_tile_6_1) {sym_name = "buf133"} : memref<1x8xbf16, 1 : i32> 
    %buf132 = aie.buffer(%mem_tile_7_1) {sym_name = "buf132"} : memref<1x8xbf16, 1 : i32> 
    %buf131 = aie.buffer(%tile_7_2) {sym_name = "buf131"} : memref<8xbf16, 2 : i32> 
    %buf130 = aie.buffer(%tile_7_2) {sym_name = "buf130"} : memref<4x2048xbf16, 2 : i32> 
    %buf129 = aie.buffer(%tile_7_2) {sym_name = "buf129"} : memref<2048xbf16, 2 : i32> 
    %buf128 = aie.buffer(%tile_6_2) {sym_name = "buf128"} : memref<8xbf16, 2 : i32> 
    %buf127 = aie.buffer(%tile_6_2) {sym_name = "buf127"} : memref<4x2048xbf16, 2 : i32> 
    %buf126 = aie.buffer(%tile_6_2) {sym_name = "buf126"} : memref<2048xbf16, 2 : i32> 
    %buf125 = aie.buffer(%tile_5_2) {sym_name = "buf125"} : memref<8xbf16, 2 : i32> 
    %buf124 = aie.buffer(%tile_5_2) {sym_name = "buf124"} : memref<4x2048xbf16, 2 : i32> 
    %buf123 = aie.buffer(%tile_5_2) {sym_name = "buf123"} : memref<2048xbf16, 2 : i32> 
    %buf122 = aie.buffer(%tile_4_2) {sym_name = "buf122"} : memref<8xbf16, 2 : i32> 
    %buf121 = aie.buffer(%tile_4_2) {sym_name = "buf121"} : memref<4x2048xbf16, 2 : i32> 
    %buf120 = aie.buffer(%tile_4_2) {sym_name = "buf120"} : memref<2048xbf16, 2 : i32> 
    %buf119 = aie.buffer(%tile_3_2) {sym_name = "buf119"} : memref<8xbf16, 2 : i32> 
    %buf118 = aie.buffer(%tile_3_2) {sym_name = "buf118"} : memref<4x2048xbf16, 2 : i32> 
    %buf117 = aie.buffer(%tile_3_2) {sym_name = "buf117"} : memref<2048xbf16, 2 : i32> 
    %buf116 = aie.buffer(%tile_2_2) {sym_name = "buf116"} : memref<8xbf16, 2 : i32> 
    %buf115 = aie.buffer(%tile_2_2) {sym_name = "buf115"} : memref<4x2048xbf16, 2 : i32> 
    %buf114 = aie.buffer(%tile_2_2) {sym_name = "buf114"} : memref<2048xbf16, 2 : i32> 
    %buf113 = aie.buffer(%tile_1_2) {sym_name = "buf113"} : memref<8xbf16, 2 : i32> 
    %buf112 = aie.buffer(%tile_1_2) {sym_name = "buf112"} : memref<4x2048xbf16, 2 : i32> 
    %buf111 = aie.buffer(%tile_1_2) {sym_name = "buf111"} : memref<2048xbf16, 2 : i32> 
    %buf110 = aie.buffer(%tile_0_2) {sym_name = "buf110"} : memref<8xbf16, 2 : i32> 
    %buf109 = aie.buffer(%tile_0_2) {sym_name = "buf109"} : memref<4x2048xbf16, 2 : i32> 
    %buf108 = aie.buffer(%tile_0_2) {sym_name = "buf108"} : memref<2048xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<8192x2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<8192xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf131 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf129 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_61, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf130 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf131) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_7_2_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_59, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf130, %buf129, %buf131) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_60, Release, 1)
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_63, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf128 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf126 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_56, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf127 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf128) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_6_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_54, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf127, %buf126, %buf128) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_55, Release, 1)
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_58, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf125 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf123 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_51, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf124 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf125) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_5_2_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_49, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf124, %buf123, %buf125) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_50, Release, 1)
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_53, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf122 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf120 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_46, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf121 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf122) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_4_2_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_44, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf121, %buf120, %buf122) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_45, Release, 1)
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_48, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf119 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf117 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_41, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf118 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf119) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_3_2_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf118, %buf117, %buf119) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_40, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_43, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf116 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf114 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_36, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf115 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf116) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_2_2_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf115, %buf114, %buf116) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_35, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_38, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf113 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf111 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf112 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf113) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_1_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf112, %buf111, %buf113) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_30, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_33, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf110 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_27, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf108 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_26, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf109 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf110) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_0_2_26, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf109, %buf108, %buf110) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_25, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_28, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "ug_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
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
      aie.dma_bd(%buf139 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf147 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf147 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_21, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf139 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_23, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf138 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf146 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf146 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf138 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf137 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf145 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf145 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf137 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf136 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf144 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf144 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf136 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_14, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf135 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf143 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf143 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf135 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf134 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf142 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf142 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf134 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf133 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf141 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf141 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf133 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf132 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf140 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf140 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf132 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_33_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_33_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_33_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_33_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_33_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_33_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_33_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_33_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_35_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_35_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_35_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_35_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_35_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_35_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_35_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_35_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_17(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @ug_matvec_bf16_0_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 0, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 16384, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 32768, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 49152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 65536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 81920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 98304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 114688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 0, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 8, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 16, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 24, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 32, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 40, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 48, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 56, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
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
      %17 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2097152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2113536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2129920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2146304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2162688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2179072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2195456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 2211840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1024, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1032, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1040, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1048, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1056, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1064, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 1080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
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
      %34 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4194304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4210688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4227072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4243456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4259840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4276224, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4292608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 4308992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2048, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2056, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2064, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2088, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2096, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 2104, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      aiex.dma_await_task(%50)
      aiex.dma_await_task(%49)
      aiex.dma_await_task(%48)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%43)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%34)
      %51 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6291456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6307840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6324224, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6340608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6356992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6373376, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6389760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 6406144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3088, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3096, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3104, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3112, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3120, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 3128, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      aiex.dma_await_task(%67)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%60)
      aiex.dma_free_task(%59)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%51)
      %68 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8388608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8404992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8421376, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8437760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8454144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8470528, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8486912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 8503296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4096, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4104, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4112, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4120, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4128, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4136, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4144, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 4152, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%83)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%77)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%68)
      %85 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10485760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10502144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10518528, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10534912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10551296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10567680, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10584064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 10600448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5120, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5128, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5136, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5144, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5152, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5160, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5168, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 5176, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%94)
      aiex.dma_free_task(%93)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%85)
      %102 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12582912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12599296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12615680, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12632064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12648448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12664832, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12681216, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 12697600, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6144, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6152, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6160, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6168, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6176, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6184, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6192, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 6200, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%111)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%102)
      %119 = aiex.dma_configure_task_for @air_channel_35_0 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14680064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @air_channel_35_1 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14696448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @air_channel_35_2 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14712832, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @air_channel_35_3 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14729216, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @air_channel_35_4 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14745600, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @air_channel_35_5 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14761984, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @air_channel_35_6 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14778368, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @air_channel_35_7 {
        aie.dma_bd(%arg9 : memref<8192x2048xbf16>, 14794752, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @air_channel_17 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @air_channel_33_0 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7168, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @air_channel_33_1 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7176, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @air_channel_33_2 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7184, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @air_channel_33_3 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7192, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @air_channel_33_4 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7200, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @air_channel_33_5 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7208, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @air_channel_33_6 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7216, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @air_channel_33_7 {
        aie.dma_bd(%arg10 : memref<8192xbf16>, 7224, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      aiex.dma_await_task(%135)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%131)
      aiex.dma_await_task(%130)
      aiex.dma_await_task(%129)
      aiex.dma_await_task(%128)
      aiex.dma_free_task(%127)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%119)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @gg_matvec_bf16_0 {
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
    %buf107 = aie.buffer(%mem_tile_0_1) {sym_name = "buf107"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf106 = aie.buffer(%mem_tile_1_1) {sym_name = "buf106"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf105 = aie.buffer(%mem_tile_2_1) {sym_name = "buf105"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf104 = aie.buffer(%mem_tile_3_1) {sym_name = "buf104"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf103 = aie.buffer(%mem_tile_4_1) {sym_name = "buf103"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf102 = aie.buffer(%mem_tile_5_1) {sym_name = "buf102"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf101 = aie.buffer(%mem_tile_6_1) {sym_name = "buf101"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf100 = aie.buffer(%mem_tile_7_1) {sym_name = "buf100"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf99 = aie.buffer(%mem_tile_0_1) {sym_name = "buf99"} : memref<1x8xbf16, 1 : i32> 
    %buf98 = aie.buffer(%mem_tile_1_1) {sym_name = "buf98"} : memref<1x8xbf16, 1 : i32> 
    %buf97 = aie.buffer(%mem_tile_2_1) {sym_name = "buf97"} : memref<1x8xbf16, 1 : i32> 
    %buf96 = aie.buffer(%mem_tile_3_1) {sym_name = "buf96"} : memref<1x8xbf16, 1 : i32> 
    %buf95 = aie.buffer(%mem_tile_4_1) {sym_name = "buf95"} : memref<1x8xbf16, 1 : i32> 
    %buf94 = aie.buffer(%mem_tile_5_1) {sym_name = "buf94"} : memref<1x8xbf16, 1 : i32> 
    %buf93 = aie.buffer(%mem_tile_6_1) {sym_name = "buf93"} : memref<1x8xbf16, 1 : i32> 
    %buf92 = aie.buffer(%mem_tile_7_1) {sym_name = "buf92"} : memref<1x8xbf16, 1 : i32> 
    %buf91 = aie.buffer(%tile_7_2) {sym_name = "buf91"} : memref<8xbf16, 2 : i32> 
    %buf90 = aie.buffer(%tile_7_2) {sym_name = "buf90"} : memref<4x2048xbf16, 2 : i32> 
    %buf89 = aie.buffer(%tile_7_2) {sym_name = "buf89"} : memref<2048xbf16, 2 : i32> 
    %buf88 = aie.buffer(%tile_6_2) {sym_name = "buf88"} : memref<8xbf16, 2 : i32> 
    %buf87 = aie.buffer(%tile_6_2) {sym_name = "buf87"} : memref<4x2048xbf16, 2 : i32> 
    %buf86 = aie.buffer(%tile_6_2) {sym_name = "buf86"} : memref<2048xbf16, 2 : i32> 
    %buf85 = aie.buffer(%tile_5_2) {sym_name = "buf85"} : memref<8xbf16, 2 : i32> 
    %buf84 = aie.buffer(%tile_5_2) {sym_name = "buf84"} : memref<4x2048xbf16, 2 : i32> 
    %buf83 = aie.buffer(%tile_5_2) {sym_name = "buf83"} : memref<2048xbf16, 2 : i32> 
    %buf82 = aie.buffer(%tile_4_2) {sym_name = "buf82"} : memref<8xbf16, 2 : i32> 
    %buf81 = aie.buffer(%tile_4_2) {sym_name = "buf81"} : memref<4x2048xbf16, 2 : i32> 
    %buf80 = aie.buffer(%tile_4_2) {sym_name = "buf80"} : memref<2048xbf16, 2 : i32> 
    %buf79 = aie.buffer(%tile_3_2) {sym_name = "buf79"} : memref<8xbf16, 2 : i32> 
    %buf78 = aie.buffer(%tile_3_2) {sym_name = "buf78"} : memref<4x2048xbf16, 2 : i32> 
    %buf77 = aie.buffer(%tile_3_2) {sym_name = "buf77"} : memref<2048xbf16, 2 : i32> 
    %buf76 = aie.buffer(%tile_2_2) {sym_name = "buf76"} : memref<8xbf16, 2 : i32> 
    %buf75 = aie.buffer(%tile_2_2) {sym_name = "buf75"} : memref<4x2048xbf16, 2 : i32> 
    %buf74 = aie.buffer(%tile_2_2) {sym_name = "buf74"} : memref<2048xbf16, 2 : i32> 
    %buf73 = aie.buffer(%tile_1_2) {sym_name = "buf73"} : memref<8xbf16, 2 : i32> 
    %buf72 = aie.buffer(%tile_1_2) {sym_name = "buf72"} : memref<4x2048xbf16, 2 : i32> 
    %buf71 = aie.buffer(%tile_1_2) {sym_name = "buf71"} : memref<2048xbf16, 2 : i32> 
    %buf70 = aie.buffer(%tile_0_2) {sym_name = "buf70"} : memref<8xbf16, 2 : i32> 
    %buf69 = aie.buffer(%tile_0_2) {sym_name = "buf69"} : memref<4x2048xbf16, 2 : i32> 
    %buf68 = aie.buffer(%tile_0_2) {sym_name = "buf68"} : memref<2048xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<8192x2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<8192xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf91 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf89 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_61, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf91) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_7_2_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_59, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf90, %buf89, %buf91) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_60, Release, 1)
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_63, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf88 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf86 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_56, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf87 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf88) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_6_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_54, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf87, %buf86, %buf88) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_55, Release, 1)
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_58, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf85 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf83 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_51, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf84 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf85) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_5_2_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_49, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf84, %buf83, %buf85) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_50, Release, 1)
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_53, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf82 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf80 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_46, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf82) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_4_2_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_44, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf81, %buf80, %buf82) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_45, Release, 1)
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_48, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf79 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf77 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_41, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf78 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf79) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_3_2_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf78, %buf77, %buf79) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_40, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_43, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_36, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf75 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf76) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_2_2_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf75, %buf74, %buf76) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_35, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_38, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf73 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf71 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf72 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf73) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_1_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf72, %buf71, %buf73) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_30, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_33, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf70 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_27, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf68 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_26, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf69 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf70) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_0_2_26, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf69, %buf68, %buf70) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_25, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_28, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "gg_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
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
      aie.dma_bd(%buf99 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf107 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf107 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_21, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf99 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_23, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf98 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf106 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf106 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf98 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf97 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf105 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf105 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf97 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf96 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf104 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf104 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf96 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_14, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf95 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf103 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf103 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf95 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf94 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf102 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf102 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf94 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf93 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf101 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf101 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf93 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf92 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf100 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf100 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf92 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_38_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_38_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_38_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_38_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_38_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_38_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_38_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_38_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_39_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_39_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_39_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_39_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_39_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_39_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_39_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_39_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_12(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @gg_matvec_bf16_0_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 0, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 16384, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 32768, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 49152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 65536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 81920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 98304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 114688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 0, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 8, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 16, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 24, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 32, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 40, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 48, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 56, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
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
      %17 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2097152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2113536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2129920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2146304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2162688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2179072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2195456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 2211840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1024, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1032, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1040, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1048, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1056, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1064, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 1080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
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
      %34 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4194304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4210688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4227072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4243456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4259840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4276224, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4292608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 4308992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2048, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2056, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2064, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2088, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2096, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 2104, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      aiex.dma_await_task(%50)
      aiex.dma_await_task(%49)
      aiex.dma_await_task(%48)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%43)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%34)
      %51 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6291456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6307840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6324224, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6340608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6356992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6373376, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6389760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 6406144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3088, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3096, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3104, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3112, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3120, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 3128, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      aiex.dma_await_task(%67)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%60)
      aiex.dma_free_task(%59)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%51)
      %68 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8388608, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8404992, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8421376, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8437760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8454144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8470528, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8486912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 8503296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4096, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4104, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4112, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4120, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4128, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4136, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4144, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 4152, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%83)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%77)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%68)
      %85 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10485760, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10502144, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10518528, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10534912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10551296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10567680, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10584064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 10600448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5120, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5128, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5136, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5144, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5152, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5160, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5168, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 5176, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%94)
      aiex.dma_free_task(%93)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%85)
      %102 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12582912, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12599296, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12615680, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12632064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12648448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12664832, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12681216, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 12697600, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6144, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6152, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6160, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6168, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6176, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6184, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6192, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 6200, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%111)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%102)
      %119 = aiex.dma_configure_task_for @air_channel_39_0 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14680064, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @air_channel_39_1 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14696448, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @air_channel_39_2 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14712832, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @air_channel_39_3 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14729216, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @air_channel_39_4 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14745600, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @air_channel_39_5 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14761984, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @air_channel_39_6 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14778368, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @air_channel_39_7 {
        aie.dma_bd(%arg7 : memref<8192x2048xbf16>, 14794752, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @air_channel_12 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @air_channel_38_0 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7168, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @air_channel_38_1 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7176, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @air_channel_38_2 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7184, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @air_channel_38_3 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7192, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @air_channel_38_4 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7200, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @air_channel_38_5 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7208, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @air_channel_38_6 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7216, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @air_channel_38_7 {
        aie.dma_bd(%arg8 : memref<8192xbf16>, 7224, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      aiex.dma_await_task(%135)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%131)
      aiex.dma_await_task(%130)
      aiex.dma_await_task(%129)
      aiex.dma_await_task(%128)
      aiex.dma_free_task(%127)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%119)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @rm_rms_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf67 = aie.buffer(%tile_0_2) {sym_name = "buf67"} : memref<2048xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_0_2) {sym_name = "buf66"} : memref<2048xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_0_2) {sym_name = "buf65"} : memref<2048xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_0_2) {sym_name = "buf64"} : memref<16xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf66 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf65 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf67 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %cst = arith.constant 0.000000e+00 : bf16
      %cst_5 = arith.constant 2.048000e+03 : bf16
      %cst_6 = arith.constant 1.001360e-05 : bf16
      %c2048 = arith.constant 2048 : index
      %c16 = arith.constant 16 : index
      %cst_7 = arith.constant dense<0.000000e+00> : vector<16xbf16>
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      vector.transfer_write %cst_7, %buf64[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c2048 step %c16 {
        %subview = memref.subview %buf67[%arg0] [16] [1] : memref<2048xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_8 = memref.subview %buf66[%arg0] [16] [1] : memref<2048xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %8 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %9 = arith.mulf %8, %8 : vector<16xbf16>
        vector.transfer_write %9, %subview_8[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %10 = vector.transfer_read %subview_8[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %11 = vector.transfer_read %buf64[%c0], %cst {in_bounds = [true]} : memref<16xbf16, 2 : i32>, vector<16xbf16>
        %12 = arith.addf %11, %10 : vector<16xbf16>
        vector.transfer_write %12, %buf64[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, 2 : i32>
      } {loop_annotation = #loop_annotation}
      %0 = vector.transfer_read %buf64[%c0], %cst {in_bounds = [true]} : memref<16xbf16, 2 : i32>, vector<16xbf16>
      %1 = vector.reduction <add>, %0 : vector<16xbf16> into bf16
      %2 = arith.divf %1, %cst_5 : bf16
      %3 = arith.addf %2, %cst_6 : bf16
      %4 = arith.extf %3 : bf16 to f32
      %5 = math.rsqrt %4 : f32
      %6 = arith.truncf %5 : f32 to bf16
      %7 = vector.broadcast %6 : bf16 to vector<16xbf16>
      scf.for %arg0 = %c0 to %c2048 step %c16 {
        %subview = memref.subview %buf67[%arg0] [16] [1] : memref<2048xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_8 = memref.subview %buf65[%arg0] [16] [1] : memref<2048xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_9 = memref.subview %buf66[%arg0] [16] [1] : memref<2048xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %8 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %9 = vector.transfer_read %subview_8[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %10 = arith.mulf %8, %7 : vector<16xbf16>
        %11 = arith.mulf %10, %9 : vector<16xbf16>
        vector.transfer_write %11, %subview_9[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.use_lock(%lock_0_2_1, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "rm_herd_0", air.herd_size = array<i64: 1, 1>}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_10(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_8(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_9(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @rm_rms_seg_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_8 {
        aie.dma_bd(%arg5 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_9 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_10 {
        aie.dma_bd(%arg6 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @a1_eltwise_add_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_5 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_6 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_7 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_8 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_9 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_10 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_11 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_12 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_13 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_14 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_15 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_16 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_17 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_18 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_19 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_4_2 = aie.lock(%tile_4_2, 5) {init = 1 : i32}
    %lock_4_2_20 = aie.lock(%tile_4_2, 4) {init = 0 : i32}
    %lock_4_2_21 = aie.lock(%tile_4_2, 3) {init = 1 : i32}
    %lock_4_2_22 = aie.lock(%tile_4_2, 2) {init = 0 : i32}
    %lock_4_2_23 = aie.lock(%tile_4_2, 1) {init = 1 : i32}
    %lock_4_2_24 = aie.lock(%tile_4_2, 0) {init = 0 : i32}
    %lock_5_2 = aie.lock(%tile_5_2, 5) {init = 1 : i32}
    %lock_5_2_25 = aie.lock(%tile_5_2, 4) {init = 0 : i32}
    %lock_5_2_26 = aie.lock(%tile_5_2, 3) {init = 1 : i32}
    %lock_5_2_27 = aie.lock(%tile_5_2, 2) {init = 0 : i32}
    %lock_5_2_28 = aie.lock(%tile_5_2, 1) {init = 1 : i32}
    %lock_5_2_29 = aie.lock(%tile_5_2, 0) {init = 0 : i32}
    %lock_6_2 = aie.lock(%tile_6_2, 5) {init = 1 : i32}
    %lock_6_2_30 = aie.lock(%tile_6_2, 4) {init = 0 : i32}
    %lock_6_2_31 = aie.lock(%tile_6_2, 3) {init = 1 : i32}
    %lock_6_2_32 = aie.lock(%tile_6_2, 2) {init = 0 : i32}
    %lock_6_2_33 = aie.lock(%tile_6_2, 1) {init = 1 : i32}
    %lock_6_2_34 = aie.lock(%tile_6_2, 0) {init = 0 : i32}
    %lock_7_2 = aie.lock(%tile_7_2, 5) {init = 1 : i32}
    %lock_7_2_35 = aie.lock(%tile_7_2, 4) {init = 0 : i32}
    %lock_7_2_36 = aie.lock(%tile_7_2, 3) {init = 1 : i32}
    %lock_7_2_37 = aie.lock(%tile_7_2, 2) {init = 0 : i32}
    %lock_7_2_38 = aie.lock(%tile_7_2, 1) {init = 1 : i32}
    %lock_7_2_39 = aie.lock(%tile_7_2, 0) {init = 0 : i32}
    %buf63 = aie.buffer(%tile_7_2) {sym_name = "buf63"} : memref<256xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_7_2) {sym_name = "buf62"} : memref<256xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_7_2) {sym_name = "buf61"} : memref<256xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_6_2) {sym_name = "buf60"} : memref<256xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_6_2) {sym_name = "buf59"} : memref<256xbf16, 2 : i32> 
    %buf58 = aie.buffer(%tile_6_2) {sym_name = "buf58"} : memref<256xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_5_2) {sym_name = "buf57"} : memref<256xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_5_2) {sym_name = "buf56"} : memref<256xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_5_2) {sym_name = "buf55"} : memref<256xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_4_2) {sym_name = "buf54"} : memref<256xbf16, 2 : i32> 
    %buf53 = aie.buffer(%tile_4_2) {sym_name = "buf53"} : memref<256xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_4_2) {sym_name = "buf52"} : memref<256xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_3_2) {sym_name = "buf51"} : memref<256xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_3_2) {sym_name = "buf50"} : memref<256xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_3_2) {sym_name = "buf49"} : memref<256xbf16, 2 : i32> 
    %buf48 = aie.buffer(%tile_2_2) {sym_name = "buf48"} : memref<256xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_2_2) {sym_name = "buf47"} : memref<256xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_2_2) {sym_name = "buf46"} : memref<256xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_1_2) {sym_name = "buf45"} : memref<256xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_1_2) {sym_name = "buf44"} : memref<256xbf16, 2 : i32> 
    %buf43 = aie.buffer(%tile_1_2) {sym_name = "buf43"} : memref<256xbf16, 2 : i32> 
    %buf42 = aie.buffer(%tile_0_2) {sym_name = "buf42"} : memref<256xbf16, 2 : i32> 
    %buf41 = aie.buffer(%tile_0_2) {sym_name = "buf41"} : memref<256xbf16, 2 : i32> 
    %buf40 = aie.buffer(%tile_0_2) {sym_name = "buf40"} : memref<256xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf61 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf63 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf62 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_35, Release, 1)
      aie.next_bd ^bb6
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_38, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_7_2_35, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf63[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf62[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf61[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_36, Release, 1)
      aie.use_lock(%lock_7_2, Release, 1)
      aie.use_lock(%lock_7_2_39, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf58 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf60 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf59 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_30, Release, 1)
      aie.next_bd ^bb6
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_6_2_30, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf60[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf59[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf58[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_31, Release, 1)
      aie.use_lock(%lock_6_2, Release, 1)
      aie.use_lock(%lock_6_2_34, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf55 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf57 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf56 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_25, Release, 1)
      aie.next_bd ^bb6
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_28, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_5_2_25, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf57[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf56[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf55[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_26, Release, 1)
      aie.use_lock(%lock_5_2, Release, 1)
      aie.use_lock(%lock_5_2_29, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf52 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf53 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_23, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_4_2_20, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf54[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf53[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf52[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_21, Release, 1)
      aie.use_lock(%lock_4_2, Release, 1)
      aie.use_lock(%lock_4_2_24, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf49 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf51 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_17, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf50 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_15, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_18, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_17, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_15, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf51[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf50[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf49[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_16, Release, 1)
      aie.use_lock(%lock_3_2, Release, 1)
      aie.use_lock(%lock_3_2_19, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf46 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf48 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_12, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf47 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_10, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_13, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_12, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_10, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf48[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf47[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf46[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_11, Release, 1)
      aie.use_lock(%lock_2_2, Release, 1)
      aie.use_lock(%lock_2_2_14, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf43 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf45 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_7, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf44 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_5, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_8, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_7, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf45[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf44[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf43[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_6, Release, 1)
      aie.use_lock(%lock_1_2, Release, 1)
      aie.use_lock(%lock_1_2_9, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf40 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf42 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf41 : memref<256xbf16, 2 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c16 {
        %subview = memref.subview %buf42[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_40 = memref.subview %buf41[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %subview_41 = memref.subview %buf40[%arg0] [16] [1] : memref<256xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %1 = vector.transfer_read %subview_40[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
        %2 = arith.addf %0, %1 : vector<16xbf16>
        vector.transfer_write %2, %subview_41[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "a1_herd_0", air.herd_size = array<i64: 8, 1>}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %tile_5_2, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %tile_6_2, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_5_2, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%tile_7_2, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_7_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_7_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_7_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_7_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_7_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_7_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_7_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_7_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_5_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_5_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_5_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_5_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_5_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_5_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_5_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_5_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_6_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_6_1(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_6_2(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_6_3(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_6_4(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_6_5(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_6_6(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_6_7(%shim_noc_tile_7_0, MM2S, 1)
    aie.runtime_sequence @a1_eltwise_add_seg_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_5_0 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_5_1 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 256, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_5_2 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 512, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_5_3 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 768, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_5_4 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1024, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_5_5 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1280, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_5_6 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1536, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_5_7 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1792, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_6_0 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 0, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_6_1 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 256, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_6_2 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 512, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_6_3 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 768, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_6_4 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 1024, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_6_5 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 1280, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_6_6 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 1536, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_6_7 {
        aie.dma_bd(%arg3 : memref<2048xbf16>, 1792, 256, [<size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_7_0 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 0, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @air_channel_7_1 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 256, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_7_2 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 512, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_7_3 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 768, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_7_4 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1024, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_7_5 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1280, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_7_6 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1536, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_7_7 {
        aie.dma_bd(%arg4 : memref<2048xbf16>, 1792, 256, [<size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%12)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%17)
      aiex.dma_free_task(%15)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%18)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%2)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) @og_matvec_bf16_0 {
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
    %buf39 = aie.buffer(%mem_tile_0_1) {sym_name = "buf39"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf38 = aie.buffer(%mem_tile_1_1) {sym_name = "buf38"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf37 = aie.buffer(%mem_tile_2_1) {sym_name = "buf37"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf36 = aie.buffer(%mem_tile_3_1) {sym_name = "buf36"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf35 = aie.buffer(%mem_tile_4_1) {sym_name = "buf35"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf34 = aie.buffer(%mem_tile_5_1) {sym_name = "buf34"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf33 = aie.buffer(%mem_tile_6_1) {sym_name = "buf33"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf32 = aie.buffer(%mem_tile_7_1) {sym_name = "buf32"} : memref<1x8x2048xbf16, 1 : i32> 
    %buf31 = aie.buffer(%mem_tile_0_1) {sym_name = "buf31"} : memref<1x8xbf16, 1 : i32> 
    %buf30 = aie.buffer(%mem_tile_1_1) {sym_name = "buf30"} : memref<1x8xbf16, 1 : i32> 
    %buf29 = aie.buffer(%mem_tile_2_1) {sym_name = "buf29"} : memref<1x8xbf16, 1 : i32> 
    %buf28 = aie.buffer(%mem_tile_3_1) {sym_name = "buf28"} : memref<1x8xbf16, 1 : i32> 
    %buf27 = aie.buffer(%mem_tile_4_1) {sym_name = "buf27"} : memref<1x8xbf16, 1 : i32> 
    %buf26 = aie.buffer(%mem_tile_5_1) {sym_name = "buf26"} : memref<1x8xbf16, 1 : i32> 
    %buf25 = aie.buffer(%mem_tile_6_1) {sym_name = "buf25"} : memref<1x8xbf16, 1 : i32> 
    %buf24 = aie.buffer(%mem_tile_7_1) {sym_name = "buf24"} : memref<1x8xbf16, 1 : i32> 
    %buf23 = aie.buffer(%tile_7_2) {sym_name = "buf23"} : memref<8xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_7_2) {sym_name = "buf22"} : memref<4x2048xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_7_2) {sym_name = "buf21"} : memref<2048xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_6_2) {sym_name = "buf20"} : memref<8xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_6_2) {sym_name = "buf19"} : memref<4x2048xbf16, 2 : i32> 
    %buf18 = aie.buffer(%tile_6_2) {sym_name = "buf18"} : memref<2048xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_5_2) {sym_name = "buf17"} : memref<8xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_5_2) {sym_name = "buf16"} : memref<4x2048xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_5_2) {sym_name = "buf15"} : memref<2048xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_4_2) {sym_name = "buf14"} : memref<8xbf16, 2 : i32> 
    %buf13 = aie.buffer(%tile_4_2) {sym_name = "buf13"} : memref<4x2048xbf16, 2 : i32> 
    %buf12 = aie.buffer(%tile_4_2) {sym_name = "buf12"} : memref<2048xbf16, 2 : i32> 
    %buf11 = aie.buffer(%tile_3_2) {sym_name = "buf11"} : memref<8xbf16, 2 : i32> 
    %buf10 = aie.buffer(%tile_3_2) {sym_name = "buf10"} : memref<4x2048xbf16, 2 : i32> 
    %buf9 = aie.buffer(%tile_3_2) {sym_name = "buf9"} : memref<2048xbf16, 2 : i32> 
    %buf8 = aie.buffer(%tile_2_2) {sym_name = "buf8"} : memref<8xbf16, 2 : i32> 
    %buf7 = aie.buffer(%tile_2_2) {sym_name = "buf7"} : memref<4x2048xbf16, 2 : i32> 
    %buf6 = aie.buffer(%tile_2_2) {sym_name = "buf6"} : memref<2048xbf16, 2 : i32> 
    %buf5 = aie.buffer(%tile_1_2) {sym_name = "buf5"} : memref<8xbf16, 2 : i32> 
    %buf4 = aie.buffer(%tile_1_2) {sym_name = "buf4"} : memref<4x2048xbf16, 2 : i32> 
    %buf3 = aie.buffer(%tile_1_2) {sym_name = "buf3"} : memref<2048xbf16, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<8xbf16, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<4x2048xbf16, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<2048xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2048x2048xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2048xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2048xbf16>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_2_63, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf23 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf21 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_7_2_61, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf23) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_7_2_61, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_7_2_59, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf22, %buf21, %buf23) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_7_2_60, Release, 1)
        aie.use_lock(%lock_7_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_7_2_63, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 7, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_2_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf20 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf18 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_6_2_56, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf19 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf20) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_6_2_56, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_6_2_54, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf19, %buf18, %buf20) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_6_2_55, Release, 1)
        aie.use_lock(%lock_6_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_6_2_58, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 6, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_2_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf17 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_52, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf15 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_5_2_51, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf16 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf17) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_5_2_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_5_2_49, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf16, %buf15, %buf17) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_5_2_50, Release, 1)
        aie.use_lock(%lock_5_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_5_2_53, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 5, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_2_48, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_4_2_46, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf14) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_4_2_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_4_2_44, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf13, %buf12, %buf14) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_4_2_45, Release, 1)
        aie.use_lock(%lock_4_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_4_2_48, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 4, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_41, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf11) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_3_2_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf10, %buf9, %buf11) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_40, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_3_2_43, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 3, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_37, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_36, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf8) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_2_2_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf7, %buf6, %buf8) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_35, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_2_2_38, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 2, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf5) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_1_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf4, %buf3, %buf5) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_30, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_1_2_33, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 1, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<8xbf16, 2 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_27, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<2048xbf16, 2 : i32>, 0, 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_26, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<4x2048xbf16, 2 : i32>, 0, 8192) {task_id = 0 : i32}
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
      func.call @linalg_fill_bf16(%cst, %buf2) : (bf16, memref<8xbf16, 2 : i32>) -> ()
      scf.for %arg0 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
        aie.use_lock(%lock_0_2_26, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
        func.call @matvec_vectorized_bf16_bf16(%c4_i32, %c2048_i32, %arg0, %buf1, %buf0, %buf2) : (i32, i32, i32, memref<4x2048xbf16, 2 : i32>, memref<2048xbf16, 2 : i32>, memref<8xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_25, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_28, Release, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "og_herd_0", air.herd_size = array<i64: 8, 1>, link_with = "mv.o"}
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
      aie.dma_bd(%buf31 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf39 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf39 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_21, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf31 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_23, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf29 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf37 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf37 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf29 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf28 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf36 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf36 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf28 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_14, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf27 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf35 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf35 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf27 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf25 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf33 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf33 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf25 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb8
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf24 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb7
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf32 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf32 : memref<1x8x2048xbf16, 1 : i32>, 0, 16384) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 1, ^bb8, ^bb2)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf24 : memref<1x8xbf16, 1 : i32>, 0, 8) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @air_channel_32_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_32_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_32_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_32_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_32_4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_32_5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_32_6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_32_7(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_36_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_36_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_36_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_36_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_36_4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_36_5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_36_6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_36_7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1(%shim_noc_tile_0_0, MM2S, 1)
    aie.runtime_sequence @og_matvec_bf16_0_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_36_0 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 0, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_36_1 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 16384, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_36_2 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 32768, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_36_3 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 49152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_36_4 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 65536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_36_5 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 81920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_36_6 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 98304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_36_7 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 114688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @air_channel_32_0 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @air_channel_32_1 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 8, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @air_channel_32_2 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 16, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @air_channel_32_3 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 24, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @air_channel_32_4 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 32, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @air_channel_32_5 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 40, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @air_channel_32_6 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 48, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @air_channel_32_7 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 56, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
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
      %17 = aiex.dma_configure_task_for @air_channel_36_0 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2097152, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @air_channel_36_1 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2113536, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @air_channel_36_2 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2129920, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @air_channel_36_3 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2146304, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @air_channel_36_4 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2162688, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @air_channel_36_5 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2179072, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @air_channel_36_6 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2195456, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @air_channel_36_7 {
        aie.dma_bd(%arg0 : memref<2048x2048xbf16>, 2211840, 262144, [<size = 16, stride = 131072>, <size = 32, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 4, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @air_channel_32_0 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1024, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @air_channel_32_1 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1032, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @air_channel_32_2 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1040, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @air_channel_32_3 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1048, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @air_channel_32_4 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1056, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @air_channel_32_5 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1064, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @air_channel_32_6 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1072, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @air_channel_32_7 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1080, 128, [<size = 16, stride = 64>, <size = 8, stride = 1>])
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
  aie.device(npu2) {
    aie.runtime_sequence @o_gemv_ffn(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>, %arg3: memref<2048xbf16>, %arg4: memref<2048xbf16>, %arg5: memref<2048xbf16>, %arg6: memref<2048xbf16>, %arg7: memref<8192x2048xbf16>, %arg8: memref<8192xbf16>, %arg9: memref<8192x2048xbf16>, %arg10: memref<8192xbf16>, %arg11: memref<8192xbf16>, %arg12: memref<2048x8192xbf16>, %arg13: memref<2048xbf16>, %arg14: memref<2048xbf16>) {
      aiex.configure @og_matvec_bf16_0 {
        aiex.run @og_matvec_bf16_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
      aiex.configure @a1_eltwise_add_seg {
        aiex.run @a1_eltwise_add_seg_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
      aiex.configure @rm_rms_seg {
        aiex.run @rm_rms_seg_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
      aiex.configure @gg_matvec_bf16_0 {
        aiex.run @gg_matvec_bf16_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
      aiex.configure @ug_matvec_bf16_0 {
        aiex.run @ug_matvec_bf16_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
      aiex.configure @sw_silu_mul_seg {
        aiex.run @sw_silu_mul_seg_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
      aiex.configure @dg_matvec_bf16_0 {
        aiex.run @dg_matvec_bf16_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
      aiex.configure @a2_eltwise_add_seg {
        aiex.run @a2_eltwise_add_seg_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (memref<2048x2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192x2048xbf16>, memref<8192xbf16>, memref<8192xbf16>, memref<2048x8192xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
    }
  }
}
