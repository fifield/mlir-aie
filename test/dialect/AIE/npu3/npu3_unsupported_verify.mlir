//===- npu3_unsupported_verify.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifiers reject AIE4 features the target model cannot describe yet.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Memtile BD pools are private per channel and direction.
aie.device(npu3) {
  %mem = aie.tile(1, 1)
  %buf = aie.buffer(%mem) : memref<256xi32>
  aie.memtile_dma(%mem) {
    // expected-error@+1 {{AIE4 memtile DMA channels have a private BD pool per channel and direction}}
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// The same through aie.dma.
aie.device(npu3) {
  %mem = aie.tile(1, 1)
  %buf = aie.buffer(%mem) : memref<256xi32>
  aie.memtile_dma(%mem) {
    // expected-error@+1 {{AIE4 memtile DMA channels have a private BD pool per channel and direction}}
    aie.dma(S2MM, 3) [{
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
    }]
    aie.end
  }
}

// -----

// Shim channel 1 takes BDs 0-15 as MM2S and 16-31 as S2MM.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<256xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{AIE4 shim DMA channel 1 uses BDs 0-15 as MM2S and 16-31 as S2MM}}
    %0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// Runtime tasks on a memtile.
aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) : memref<256xi32>
  aie.runtime_sequence @seq(%arg0 : memref<256xi32>) {
    // expected-error@+1 {{AIE4 memtile DMA channels have a private BD pool per channel and direction}}
    %task = aiex.dma_configure_task(%mem, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}

// -----

// Runtime tasks on shim channel 1.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @seq(%arg0 : memref<256xi32>) {
    // expected-error@+1 {{AIE4 shim DMA channel 1 uses BDs 0-15 as MM2S and 16-31 as S2MM}}
    %task = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}

// -----

// AIE4 control packets have their own header and payload size.
aie.device(npu3) {
  aie.runtime_sequence @seq() {
    // expected-error@+1 {{control packets are not yet supported on AIE4}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 0 : i32, stream_id = 0 : i32, data = array<i32: 1>}
  }
}
