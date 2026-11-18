//===- bad_control_trace_dma_channel.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ControlMM2S (2) and TraceS2MM (1) exist only on AIE4 shim tiles.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<64xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{ControlMM2S channel 0 is out of range: the tile has 0 ControlMM2S channels}}
    %0 = aie.dma_start(ControlMM2S, 0, ^bd, ^end)
  ^bd:
    aie.dma_bd(%ext : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu1) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{TraceS2MM channel 0 is out of range: the tile has 0 TraceS2MM channels}}
  aie.shim_dma_allocation @trace(%shim, TraceS2MM, 0)
}

// -----

aie.device(npu2) {
  aie.runtime_sequence @seq() {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{ControlMM2S channel 0 is out of range: the tile has 0 ControlMM2S channels}}
    aiex.npu.push_queue (0, 0, ControlMM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu3) {
  aie.runtime_sequence @seq() {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    // expected-error@+1 {{bd_id 8 is not in the BD pool of ControlMM2S channel 1}}
    aiex.npu.push_queue (0, 0, ControlMM2S:1) bd_id %c8 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu3) {
  aie.runtime_sequence @seq() {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    // expected-error@+1 {{bd_id 4 is not in the BD pool of TraceS2MM channel 0}}
    aiex.npu.push_queue (0, 0, TraceS2MM:0) bd_id %c4 repeat %c0 {issue_token = true} : i32, i32
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<64xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{ControlMM2S channel 2 is out of range: the tile has 2 ControlMM2S channels}}
    %0 = aie.dma_start(ControlMM2S, 2, ^bd, ^end)
  ^bd:
    aie.dma_bd(%ext : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @seq(%arg0 : memref<64xi32>) {
    // expected-error@+1 {{TraceS2MM channel 1 is out of range: the tile has 1 TraceS2MM channels}}
    %task = aiex.dma_configure_task(%shim, TraceS2MM, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
  }
}

// -----

// A memtile has no control or trace DMA.
aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) : memref<64xi32>
  %lock0 = aie.lock(%mem, 0) {init = 1 : i32}
  %lock1 = aie.lock(%mem, 1) {init = 0 : i32}
  aie.memtile_dma(%mem) {
    // expected-error@+1 {{ControlMM2S channel 0 is out of range: the tile has 0 ControlMM2S channels}}
    %0 = aie.dma(ControlMM2S, 0) [{
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<64xi32>)
      aie.use_lock(%lock1, Release, %c1)
    }]
    aie.end
  }
}

// -----

// Nor does a core tile.
aie.device(npu3) {
  %core = aie.tile(0, 2)
  aie.core(%core) {
    // expected-error@+1 {{pushes onto TraceS2MM channel 0, but tile (0, 2) has 0}}
    aie.dma_push_queue(TraceS2MM : 0) bd_id 0
    aie.end
  }
}

// -----

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  aie.runtime_sequence @seq() {
    // expected-error@+1 {{channel 0 out of range for this tile and direction (tile has 0 DMA channel(s) in this direction)}}
    aiex.dma_channel_reset(%mem, TraceS2MM, 0)
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %core = aie.tile(0, 2)
  aie.bd_chain @chain(%buf: memref<64xi32>) {
    aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
    aie.end
  }
  aie.runtime_sequence @seq(%arg0 : memref<64xi32>) {
    // expected-error@+1 {{ControlMM2S channel 0 is out of range: the tile has 0 ControlMM2S channels}}
    %task = aiex.dma_start_bd_chain @chain(%arg0) : (memref<64xi32>) on (%core, ControlMM2S, 0)
  }
}

// -----

// In aie.trace.port the direction picks the port's side, not a DMA.
aie.device(npu2) {
  %core = aie.tile(0, 2)
  aie.trace @trace(%core) {
    // expected-error@+1 {{direction must be S2MM (master) or MM2S (slave)}}
    aie.trace.port<0> port=North channel=1 direction=TraceS2MM
  }
}
