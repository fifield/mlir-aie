//===- bad_trace_s2mm_channel_aie4.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize --split-input-file %s 2>&1 | FileCheck %s
// CHECK: 'aie.dma_start' op TraceS2MM channel 1 is out of range; expected 0 <= channel < 1

// AIE4 has exactly one trace-S2MM channel. In particular, there is no
// application-B trace channel or second trace register bank.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %dma = aie.shim_dma(%shim) {
    %start = aie.dma_start(TraceS2MM, 1, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// CHECK: 'aie.dma' op TraceS2MM channel 1 is out of range; expected 0 <= channel < 1

// Region-form DMA operations must enforce the same channel bounds.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %lock = aie.lock(%shim, 0)
  %shim_dma = aie.shim_dma(%shim) {
    %bad = aie.dma(TraceS2MM, 1) [{
      %zero = arith.constant 0 : i32
      aie.use_lock(%lock, Acquire, %zero)
      aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%lock, Release, %zero)
    }]
    aie.end
  }
}

// -----

// CHECK: 'aie.dma_start' op ControlMM2S channel 2 is out of range; expected 0 <= channel < 2

// AIE4 has exactly two control-MM2S channels.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %dma = aie.shim_dma(%shim) {
    %start = aie.dma_start(ControlMM2S, 2, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// CHECK: 'aie.dma' op ControlMM2S channel 2 is out of range; expected 0 <= channel < 2

// Region-form control DMA operations must enforce the same channel bounds.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %lock = aie.lock(%shim, 0)
  %shim_dma = aie.shim_dma(%shim) {
    %bad = aie.dma(ControlMM2S, 2) [{
      %zero = arith.constant 0 : i32
      aie.use_lock(%lock, Acquire, %zero)
      aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%lock, Release, %zero)
    }]
    aie.end
  }
}
