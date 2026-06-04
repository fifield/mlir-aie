//===- verify_aie4_d0_stride.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<64xi32>) {
      // expected-error @below {{AIE4 D0 stride is fixed by hardware and must be 1}}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][1, 1, 1, 2, 4][0, 0, 0, 4, 2]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<64xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

// -----

module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<64xi32>) {
      // expected-error @below {{AIE4 D0 stride is fixed by hardware and must be 1}}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][1, 1, 1, 3, 1][0, 0, 0, 4, 7]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<64xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

// -----

module {
  aie.device(npu3) {
    aie.runtime_sequence() {
      // expected-error @below {{AIE4 D0 stride is implied by hardware and must be 0}}
      aiex.npu.writebd {
        bd_id = 0 : i32,
        buffer_length = 4 : i32,
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
        d0_stride = 1 : i32,
        d0_zero_after = 0 : i32,
        d0_zero_before = 0 : i32,
        d1_size = 1 : i32,
        d1_stride = 0 : i32,
        d1_zero_after = 0 : i32,
        d1_zero_before = 0 : i32,
        d2_size = 1 : i32,
        d2_stride = 0 : i32,
        d2_zero_after = 0 : i32,
        d2_zero_before = 0 : i32,
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
        valid_bd = 1 : i32
      }
    }
  }
}
