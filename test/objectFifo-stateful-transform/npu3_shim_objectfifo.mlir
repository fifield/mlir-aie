//===- npu3_shim_objectfifo.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-create-pathfinder-flows %s | FileCheck %s

// AIE4 shim objectfifos route through the shim switchbox, without a shim_mux.

// CHECK-LABEL: aie.device(npu3)
// CHECK-NOT:     aie.shim_mux
// CHECK:         %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK:         aie.shim_dma_allocation @in__shim_alloc(%[[SHIM]], MM2S, 0)
// CHECK:         aie.shim_dma_allocation @out__shim_alloc(%[[SHIM]], S2MM, 0)
// CHECK:         %[[SB:.*]] = aie.switchbox(%[[SHIM]]) {
// CHECK-NEXT:      aie.connect<DMA : 0, North : {{[0-9]+}}>
// CHECK-NEXT:      aie.connect<North : {{[0-9]+}}, DMA : 0>
// CHECK-NEXT:    }
// CHECK:         aie.wire(%[[SHIM]] : DMA, %[[SB]] : DMA)
// CHECK-NOT:     aie.shim_mux

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
      %1 = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
  }
}
