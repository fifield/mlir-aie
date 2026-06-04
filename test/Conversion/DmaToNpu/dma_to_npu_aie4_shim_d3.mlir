//===- dma_to_npu_aie4_shim_d3.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 48, 50343938, 2097152, 4, 12, 0, 32768]>

module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][1, 2, 2, 3, 4][0, 32, 12, 4, 1]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<128xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

// -----

// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 24, 50343938, 2097152, 2, 12, 0, 24576]>

module {
  aie.device(npu3) {
    aie.runtime_sequence() {
      aiex.npu.writebd {
        bd_id = 0 : i32,
        buffer_length = 24 : i32,
        buffer_offset = 0 : i32,
        enable_packet = 0 : i32,
        out_of_order_id = 0 : i32,
        packet_id = 0 : i32,
        packet_type = 0 : i32,
        column = 0 : i32,
        row = 0 : i32,
        channel = 0 : i32,
        direction = 0 : i32,
        d0_size = 4 : i32,
        d0_stride = 0 : i32,
        d0_zero_after = 0 : i32,
        d0_zero_before = 0 : i32,
        d1_size = 3 : i32,
        d1_stride = 2 : i32,
        d1_zero_after = 0 : i32,
        d1_zero_before = 0 : i32,
        d2_size = 2 : i32,
        d2_stride = 12 : i32,
        d2_zero_after = 0 : i32,
        d2_zero_before = 0 : i32,
        d3_size = 2 : i32,
        d3_stride = 24 : i32,
        iteration_current = 0 : i32,
        iteration_size = 0 : i32,
        iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32,
        lock_acq_id = 0 : i32,
        lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32,
        lock_rel_val = 0 : i32,
        next_bd = 0 : i32,
        use_next_bd = 0 : i32,
        valid_bd = 1 : i32,
        burst_length = 0 : i32
      }
    }
  }
}
