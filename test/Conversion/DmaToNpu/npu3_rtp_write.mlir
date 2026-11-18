//===- npu3_rtp_write.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// AIE4 rtp_write resolves like a buffer-relative write32; memtile data memory
// starts 1 MiB into its window, so offset 0 of tile (0, 1) is 0x00200000.

// CHECK-LABEL: aie.runtime_sequence @seq
// CHECK:     %[[MT0:.*]] = arith.constant 2097152 : i32
// CHECK:     aiex.npu.write32(%[[MT0]], %{{.*}}) : i32, i32
// CHECK:     %[[MT4:.*]] = arith.constant 2097168 : i32
// CHECK:     aiex.npu.write32(%[[MT4]], %{{.*}}) : i32, i32
// CHECK:     %[[W0:.*]] = arith.constant 2097152 : i32
// CHECK:     aiex.npu.write32(%[[W0]], %{{.*}}) : i32, i32
// CHECK:     %[[CT:.*]] = arith.constant 6292496 : i32
// CHECK:     aiex.npu.write32(%[[CT]], %{{.*}}) : i32, i32
// CHECK-NOT: column
// CHECK:     }

aie.device(npu3) {
  %mt = aie.tile(0, 1)
  %ct = aie.tile(0, 2)
  %rtp_mt = aie.buffer(%mt) {address = 0 : i32, sym_name = "rtp_mt"} : memref<16xi32>
  %rtp_ct = aie.buffer(%ct) {address = 1024 : i32, sym_name = "rtp_ct"} : memref<16xi32>
  aie.runtime_sequence @seq() {
    %v = arith.constant 7 : i32
    aiex.npu.rtp_write(@rtp_mt, 0, %v) : i32
    aiex.npu.rtp_write(@rtp_mt, 4, %v) : i32
    %c0 = arith.constant 0 : i32
    aiex.npu.write32(%c0, %v) {buffer = @rtp_mt} : i32, i32
    aiex.npu.rtp_write(@rtp_ct, 4, %v) : i32
  }
}
