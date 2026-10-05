//===- specialized_shim_channels_aie4.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// the architecture specification enumerates the shim's channels as
// MM2S0-3, S2MM0-1, Control MM2S0-1 and Trace S2MM: the control and trace
// channels are separate hardware from the data ports and consume no
// WireBundle::DMA connection. They must therefore not be counted against the
// data channel limits (4 MM2S / 2 S2MM on an AIE4b shim).

// CHECK-LABEL: @trace_plus_both_data_s2mm
// CHECK: aie.dma_start(TraceS2MM, 0
// Both data S2MM channels plus trace: 3 "input" directions, only 2 data ports.
module @trace_plus_both_data_s2mm {
 aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %dma = aie.shim_dma(%shim) {
    %s0 = aie.dma_start(S2MM, 0, ^bd0, ^next0)
  ^bd0:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^next0:
    %s1 = aie.dma_start(S2MM, 1, ^bd1, ^next1)
  ^bd1:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd1
  ^next1:
    %t = aie.dma_start(TraceS2MM, 0, ^bd2, ^end)
  ^bd2:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd2
  ^end:
    aie.end
  }
 }
}

// -----

// CHECK-LABEL: @ctrl_plus_all_data_mm2s
// CHECK: aie.dma_start(ControlMM2S, 1
// All four data MM2S channels plus both control-MM2S: 6 "output" directions,
// only 4 data ports.
module @ctrl_plus_all_data_mm2s {
 aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %dma = aie.shim_dma(%shim) {
    %m0 = aie.dma_start(MM2S, 0, ^bd0, ^n0)
  ^bd0:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^n0:
    %m1 = aie.dma_start(MM2S, 1, ^bd1, ^n1)
  ^bd1:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd1
  ^n1:
    %m2 = aie.dma_start(MM2S, 2, ^bd2, ^n2)
  ^bd2:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd2
  ^n2:
    %m3 = aie.dma_start(MM2S, 3, ^bd3, ^n3)
  ^bd3:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd3
  ^n3:
    %c0 = aie.dma_start(ControlMM2S, 0, ^bd4, ^n4)
  ^bd4:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd4
  ^n4:
    %c1 = aie.dma_start(ControlMM2S, 1, ^bd5, ^end)
  ^bd5:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd5
  ^end:
    aie.end
  }
 }
}
