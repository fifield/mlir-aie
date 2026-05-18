module {
  aie.device(npu2_1col) {
    func.func private @passThroughLine(memref<1024xi32>, memref<1024xi32>, i32) attributes {link_with = "passThrough.cc.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in_c0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo @out_c0(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @in_c0(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %2 = aie.objectfifo.acquire @out_c0(Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %c4096_i32 = arith.constant 4096 : i32
        func.call @passThroughLine(%1, %3, %c4096_i32) : (memref<1024xi32>, memref<1024xi32>, i32) -> ()
        aie.objectfifo.release @in_c0(Consume, 1)
        aie.objectfifo.release @out_c0(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @seq(%arg0: memref<8192xi32>, %arg1: memref<8192xi32>) {
      aiex.npu.load_pdi {device_ref = @main}
      %0 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 1024, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %3 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 1024, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 2048, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %5 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 2048, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 3072, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %7 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 3072, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 4096, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %9 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 4096, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 5120, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %11 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 5120, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 6144, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %13 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 6144, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @in_c0 {
        aie.dma_bd(%arg0 : memref<8192xi32>, 7168, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %15 = aiex.dma_configure_task_for @out_c0 {
        aie.dma_bd(%arg1 : memref<8192xi32>, 7168, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      aiex.dma_start_task(%15)
      aiex.dma_await_task(%15)
    }
  }
}

