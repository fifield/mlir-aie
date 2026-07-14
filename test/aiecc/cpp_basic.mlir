//===- cpp_basic.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --verbose %s | FileCheck %s
// RUN: aiecc --no-xchesscc --no-xbridge -n --verbose %s | FileCheck %s --check-prefix=DRY
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --verbose %s 2>&1 | FileCheck %s --check-prefix=NPU

// Phase 4 — checkpoint dumps from the split resource-allocation and NPU
// lowering pipelines. Confirm each intermediate file exists, and that the
// chain accumulated by dma_to_npu.mlir contains both the early 'input' and
// late 'dma-to-npu' stage labels. The device here is unnamed, so it takes
// DeviceOp's default sym_name ("main") and the dumps are main_*.
// RUN: rm -rf %t_npu_tmp && mkdir -p %t_npu_tmp
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --keep-loc --dump-intermediates --tmpdir=%t_npu_tmp %s
// RUN: test -f %t_npu_tmp/input.mlir
// RUN: test -f %t_npu_tmp/objectfifo_expanded.mlir
// RUN: test -f %t_npu_tmp/main_bd_chains_materialized.mlir
// RUN: test -f %t_npu_tmp/main_dma_tasks_to_npu.mlir
// RUN: test -f %t_npu_tmp/main_dma_to_npu.mlir
// RUN: test -f %t_npu_tmp/main_set_lock_lowered.mlir
// RUN: test -f %t_npu_tmp/main_npu_lowered.mlir
// RUN: FileCheck %s --check-prefix=DMA_NPU < %t_npu_tmp/main_dma_to_npu.mlir

// CHECK: Successfully parsed input file
// CHECK: Found 1 AIE device
// CHECK: Running resource allocation pipeline in-memory
// CHECK: Resource allocation pipeline completed successfully
// CHECK: Running routing pipeline in-memory
// CHECK: Routing pipeline completed successfully
// CHECK: Compilation completed successfully

// DRY: Dry run - command not executed
// DRY: Compilation completed successfully

// NPU: Generating NPU instructions for device
// NPU: Compilation completed successfully

// By the time we get to dma_to_npu.mlir, the chain has accumulated several
// stage labels. We assert the latest (dma-to-npu) and the earliest (input).
// DMA_NPU-DAG: fused<"checkpoint:dma-to-npu">
// DMA_NPU-DAG: fused<"checkpoint:input">
// DMA_NPU-DAG: fused<"checkpoint:objectfifo-expanded">

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    
    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32
      
      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      
      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      
      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<16xi32>
      }
      
      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }
    
    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
