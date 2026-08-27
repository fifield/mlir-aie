//===- bad_specialized_shim_bd_id_aie4.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The AIE4 shim's control-MM2S and trace-S2MM channels have private BD banks
// of 8 and 4 entries. A bd_id past the end of the owning bank clears the flat
// per-tile count (32 on an AIE4b shim) but resolves into a neighbouring bank:
// control-MM2S bd 8 lands at 0x18300 + 8*0x30 = 0x18480, which is trace-S2MM
// bd 0.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  // expected-error@+1 {{bd_id 8 is not accessible from channel (ControlMM2S, 0), whose BD pool spans ids 0 to 7}}
  %dma = aie.shim_dma(%shim) {
    %c = aie.dma_start(ControlMM2S, 0, ^bd, ^end)
  ^bd:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64) {bd_id = 8 : i32}
    aie.next_bd ^bd
  ^end:
    aie.end
  }
}

// -----

// Trace-S2MM has only four BDs.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  // expected-error@+1 {{bd_id 4 is not accessible from channel (TraceS2MM, 0), whose BD pool spans ids 0 to 3}}
  %dma = aie.shim_dma(%shim) {
    %t = aie.dma_start(TraceS2MM, 0, ^bd, ^end)
  ^bd:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64) {bd_id = 4 : i32}
    aie.next_bd ^bd
  ^end:
    aie.end
  }
}

// -----

// An id inside the bank stays legal.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %dma = aie.shim_dma(%shim) {
    %c = aie.dma_start(ControlMM2S, 0, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64) {bd_id = 7 : i32}
    aie.next_bd ^bd
  ^end:
    aie.end
  }
}
