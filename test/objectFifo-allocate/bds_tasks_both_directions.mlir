//===- bds_tasks_both_directions.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-packet-ids --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// A core's 16 BDs serve both directions. Channel 1's largest runtime task (7
// BDs) counts once across MM2S/S2MM, so 3 + 7 <= 16 fits without a warning.
module @bds_tasks_both_directions {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %b = aie.buffer(%core) {sym_name = "b"} : memref<48xi32>
    %t = aie.buffer(%core) {sym_name = "t"} : memref<112xi32>
    aie.route_endpoint @out(%core) DMA
    aie.route_endpoint @dst(%shim) DMA {fifoName = "b"}
    aie.route from @out to [@dst]
    aie.mem(%core) {
      aie.dma_start(MM2S, @out, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<48xi32> offset = 0 len = 16)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%b : memref<48xi32> offset = 16 len = 16)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%b : memref<48xi32> offset = 32 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      %out = aiex.dma_configure_task(%core, MM2S, 1) {
        aie.dma_bd(%t : memref<112xi32> offset = 0 len = 16)
        aie.next_bd ^o1
      ^o1:
        aie.dma_bd(%t : memref<112xi32> offset = 16 len = 16)
        aie.next_bd ^o2
      ^o2:
        aie.dma_bd(%t : memref<112xi32> offset = 32 len = 16)
        aie.next_bd ^o3
      ^o3:
        aie.dma_bd(%t : memref<112xi32> offset = 48 len = 16)
        aie.next_bd ^o4
      ^o4:
        aie.dma_bd(%t : memref<112xi32> offset = 64 len = 16)
        aie.next_bd ^o5
      ^o5:
        aie.dma_bd(%t : memref<112xi32> offset = 80 len = 16)
        aie.next_bd ^o6
      ^o6:
        aie.dma_bd(%t : memref<112xi32> offset = 96 len = 16)
        aie.end
      }
      %in = aiex.dma_configure_task(%core, S2MM, 1) {
        aie.dma_bd(%t : memref<112xi32> offset = 0 len = 16)
        aie.next_bd ^i1
      ^i1:
        aie.dma_bd(%t : memref<112xi32> offset = 16 len = 16)
        aie.next_bd ^i2
      ^i2:
        aie.dma_bd(%t : memref<112xi32> offset = 32 len = 16)
        aie.next_bd ^i3
      ^i3:
        aie.dma_bd(%t : memref<112xi32> offset = 48 len = 16)
        aie.next_bd ^i4
      ^i4:
        aie.dma_bd(%t : memref<112xi32> offset = 64 len = 16)
        aie.next_bd ^i5
      ^i5:
        aie.dma_bd(%t : memref<112xi32> offset = 80 len = 16)
        aie.next_bd ^i6
      ^i6:
        aie.dma_bd(%t : memref<112xi32> offset = 96 len = 16)
        aie.end
      }
    }
  }
}
