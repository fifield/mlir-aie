//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    // Shim tiles
    %t00 = aie.tile(0, 0)
    %t10 = aie.tile(1, 0)
    %t20 = aie.tile(2, 0)

    // Mem tiles
    %t01 = aie.tile(0, 1)
    %t11 = aie.tile(1, 1)
    %t21 = aie.tile(2, 1)

    // Compute tiles - 3x3 grid (rows 2, 3, 4)
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %t22 = aie.tile(2, 2)

    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)

    %t04 = aie.tile(0, 4)
    %t14 = aie.tile(1, 4)
    %t24 = aie.tile(2, 4)

    // Locks for synchronization
    %lock02 = aie.lock(%t02, 10) { init = 1 : i32 }
    %lock12 = aie.lock(%t12, 10) { init = 1 : i32 }
    %lock22 = aie.lock(%t22, 10) { init = 1 : i32 }

    %lock03 = aie.lock(%t03, 10) { init = 1 : i32 }
    %lock13 = aie.lock(%t13, 10) { init = 1 : i32 }
    %lock23 = aie.lock(%t23, 10) { init = 1 : i32 }

    %lock04 = aie.lock(%t04, 10) { init = 1 : i32 }
    %lock14 = aie.lock(%t14, 10) { init = 1 : i32 }
    %lock24 = aie.lock(%t24, 10) { init = 1 : i32 }

    // Cascade flows (north to south) for each column
    // Column 0: row 4 -> 3 -> 2
    aie.cascade_flow(%t04, %t03)
    aie.cascade_flow(%t03, %t02)

    // Column 1: row 4 -> 3 -> 2
    aie.cascade_flow(%t14, %t13)
    aie.cascade_flow(%t13, %t12)

    // Column 2: row 4 -> 3 -> 2
    aie.cascade_flow(%t24, %t23)
    aie.cascade_flow(%t23, %t22)

    // Output objectFifo: collect results from bottom row (row 2) to host
    aie.objectfifo @objFifo_out0(%t02, {%t01}, 1 : i32) : !aie.objectfifo<memref<9xi32>>
    aie.objectfifo @objFifo_out1(%t12, {%t11}, 1 : i32) : !aie.objectfifo<memref<9xi32>>
    aie.objectfifo @objFifo_out2(%t22, {%t21}, 1 : i32) : !aie.objectfifo<memref<9xi32>>

    aie.objectfifo @objFifo_out_shim0(%t01, {%t00}, 1 : i32) : !aie.objectfifo<memref<9xi32>>
    aie.objectfifo @objFifo_out_shim1(%t11, {%t10}, 1 : i32) : !aie.objectfifo<memref<9xi32>>
    aie.objectfifo @objFifo_out_shim2(%t21, {%t20}, 1 : i32) : !aie.objectfifo<memref<9xi32>>

    aie.objectfifo.link [@objFifo_out0] -> [@objFifo_out_shim0] ([] [])
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out_shim1] ([] [])
    aie.objectfifo.link [@objFifo_out2] -> [@objFifo_out_shim2] ([] [])

    func.func private @systolic_kernel(%buf: memref<9xi32>, %size: i32, %id: i32, %is_top: i32, %is_bottom: i32, %phase: i32) -> ()

    // Row 4 (top row) - cores 0, 1, 2
    %buf04 = aie.buffer(%t04) : memref<9xi32>
    %core04 = aie.core(%t04) {
        %size = arith.constant 9 : i32
        %id = arith.constant 0 : i32
        %is_top = arith.constant 1 : i32
        %is_bottom = arith.constant 0 : i32
        %is_left = arith.constant 1 : i32
        %is_right = arith.constant 0 : i32
        aie.use_lock(%lock04, AcquireGreaterEqual, 1)
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf04, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%lock04, AcquireGreaterEqual, 1)
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf04, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.end
    } { link_with="kernel.o" }

    %buf14 = aie.buffer(%t14) : memref<9xi32>
    %core14 = aie.core(%t14) {
        %size = arith.constant 9 : i32
        %id = arith.constant 1 : i32
        %is_top = arith.constant 1 : i32
        %is_bottom = arith.constant 0 : i32
        %is_left = arith.constant 0 : i32
        %is_right = arith.constant 0 : i32
        aie.use_lock(%lock14, AcquireGreaterEqual, 1)
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf14, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%lock14, AcquireGreaterEqual, 1)
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf14, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.end
    } { link_with="kernel.o" }

    %buf24 = aie.buffer(%t14) : memref<9xi32>
    %core24 = aie.core(%t24) {
        %size = arith.constant 9 : i32
        %id = arith.constant 2 : i32
        %is_top = arith.constant 1 : i32
        %is_bottom = arith.constant 0 : i32
        %is_left = arith.constant 0 : i32
        %is_right = arith.constant 1 : i32
        aie.use_lock(%lock24, AcquireGreaterEqual, 1)
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf24, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%lock24, AcquireGreaterEqual, 1)
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf24, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.end
    } { link_with="kernel.o" }

    // Row 3 (middle row) - cores 3, 4, 5
    %buf03 = aie.buffer(%t03) : memref<9xi32>
    %core03 = aie.core(%t03) {
        %size = arith.constant 9 : i32
        %id = arith.constant 3 : i32
        %is_top = arith.constant 0 : i32
        %is_bottom = arith.constant 0 : i32
        %is_left = arith.constant 1 : i32
        %is_right = arith.constant 0 : i32
        aie.use_lock(%lock03, AcquireGreaterEqual, 1)
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf03, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%lock03, AcquireGreaterEqual, 1)
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf03, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.end
    } { link_with="kernel.o" }

    %buf13 = aie.buffer(%t13) : memref<9xi32>
    %core13 = aie.core(%t13) {
        %size = arith.constant 9 : i32
        %id = arith.constant 4 : i32
        %is_top = arith.constant 0 : i32
        %is_bottom = arith.constant 0 : i32
        %is_left = arith.constant 0 : i32
        %is_right = arith.constant 0 : i32
        aie.use_lock(%lock13, AcquireGreaterEqual, 1)
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf13, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%lock13, AcquireGreaterEqual, 1)
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf13, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.end
    } { link_with="kernel.o" }

    %buf23 = aie.buffer(%t23) : memref<9xi32>
    %core23 = aie.core(%t23) {
        %size = arith.constant 9 : i32
        %id = arith.constant 5 : i32
        %is_top = arith.constant 0 : i32
        %is_bottom = arith.constant 0 : i32
        %is_left = arith.constant 0 : i32
        %is_right = arith.constant 1 : i32
        aie.use_lock(%lock23, AcquireGreaterEqual, 1)
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf23, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%lock23, AcquireGreaterEqual, 1)
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf23, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.end
    } { link_with="kernel.o" }

    // Row 2 (bottom row) - cores 6, 7, 8
    %core02 = aie.core(%t02) {
        aie.use_lock(%lock02, AcquireGreaterEqual, 1)
        %subview = aie.objectfifo.acquire @objFifo_out0(Produce, 1) : !aie.objectfifosubview<memref<9xi32>>
        %buf = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<9xi32>> -> memref<9xi32>
        %size = arith.constant 9 : i32
        %id = arith.constant 6 : i32
        %is_top = arith.constant 0 : i32
        %is_bottom = arith.constant 1 : i32
        %is_left = arith.constant 1 : i32
        %is_right = arith.constant 0 : i32
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @objFifo_out0(Produce, 1)

        aie.use_lock(%lock02, AcquireGreaterEqual, 1)
        %subview1 = aie.objectfifo.acquire @objFifo_out0(Produce, 1) : !aie.objectfifosubview<memref<9xi32>>
        %buf1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<9xi32>> -> memref<9xi32>
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf1, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @objFifo_out0(Produce, 1)
        aie.end
    } { link_with="kernel.o" }

    %core12 = aie.core(%t12) {
        aie.use_lock(%lock12, AcquireGreaterEqual, 1)
        %subview = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<9xi32>>
        %buf = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<9xi32>> -> memref<9xi32>
        %size = arith.constant 9 : i32
        %id = arith.constant 7 : i32
        %is_top = arith.constant 0 : i32
        %is_bottom = arith.constant 1 : i32
        %is_left = arith.constant 0 : i32
        %is_right = arith.constant 0 : i32
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @objFifo_out1(Produce, 1)

        %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<9xi32>>
        %buf1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<9xi32>> -> memref<9xi32>
        aie.use_lock(%lock12, AcquireGreaterEqual, 1)
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf1, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @objFifo_out1(Produce, 1)
        aie.end
    } { link_with="kernel.o" }

    %core22 = aie.core(%t22) {
        aie.use_lock(%lock22, AcquireGreaterEqual, 1)
        %subview = aie.objectfifo.acquire @objFifo_out2(Produce, 1) : !aie.objectfifosubview<memref<9xi32>>
        %buf = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<9xi32>> -> memref<9xi32>
        %size = arith.constant 9 : i32
        %id = arith.constant 8 : i32
        %is_top = arith.constant 0 : i32
        %is_bottom = arith.constant 1 : i32
        %is_left = arith.constant 0 : i32
        %is_right = arith.constant 1 : i32
        %phase0 = arith.constant 0 : i32
        func.call @systolic_kernel(%buf, %size, %id, %is_top, %is_bottom, %phase0) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @objFifo_out2(Produce, 1)

        aie.use_lock(%lock22, AcquireGreaterEqual, 1)
        %subview1 = aie.objectfifo.acquire @objFifo_out2(Produce, 1) : !aie.objectfifosubview<memref<9xi32>>
        %buf1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<9xi32>> -> memref<9xi32>
        %phase1 = arith.constant 1 : i32
        func.call @systolic_kernel(%buf1, %size, %id, %is_left, %is_right, %phase1) : (memref<9xi32>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @objFifo_out2(Produce, 1)
        aie.end
    } { link_with="kernel.o" }

    aiex.runtime_sequence(%out0 : memref<18xi32>, %out1 : memref<18xi32>, %out2 : memref<18xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c9 = arith.constant 9 : i64

      // Setup output DMAs for all three columns
      aiex.npu.dma_memcpy_nd (%out0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c9][%c0,%c0,%c0,%c1]) { metadata = @objFifo_out_shim0, id = 0 : i64, issue_token = true } : memref<18xi32>
      aiex.npu.dma_memcpy_nd (%out1[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c9][%c0,%c0,%c0,%c1]) { metadata = @objFifo_out_shim1, id = 1 : i64, issue_token = true } : memref<18xi32>
      aiex.npu.dma_memcpy_nd (%out2[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c9][%c0,%c0,%c0,%c1]) { metadata = @objFifo_out_shim2, id = 2 : i64, issue_token = true } : memref<18xi32>

      // Wait for all outputs to complete
      aiex.npu.dma_wait { symbol = @objFifo_out_shim0 }
      aiex.npu.dma_wait { symbol = @objFifo_out_shim1 }
      aiex.npu.dma_wait { symbol = @objFifo_out_shim2 }

      // reconfigure cascade
      aiex.npu.write32 {address = 2318432 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 35872864 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 69427296 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 3367008 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 36921440 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 70475872 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 4415584 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 37970016 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 71524448 : ui32, value = 3 : ui32}

      // Release locks with value=1
      aiex.npu.write32 {address = 2224288 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35778720 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 69333152 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 3272864 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 36827296 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 70381728 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 4321440 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 37875872 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 71430304 : ui32, value = 1 : ui32}

      // Setup output DMAs for all three columns
      aiex.npu.dma_memcpy_nd (%out0[%c0,%c0,%c0,%c9][%c1,%c1,%c1,%c9][%c0,%c0,%c0,%c1]) { metadata = @objFifo_out_shim0, id = 0 : i64, issue_token = true } : memref<18xi32>
      aiex.npu.dma_memcpy_nd (%out1[%c0,%c0,%c0,%c9][%c1,%c1,%c1,%c9][%c0,%c0,%c0,%c1]) { metadata = @objFifo_out_shim1, id = 1 : i64, issue_token = true } : memref<18xi32>
      aiex.npu.dma_memcpy_nd (%out2[%c0,%c0,%c0,%c9][%c1,%c1,%c1,%c9][%c0,%c0,%c0,%c1]) { metadata = @objFifo_out_shim2, id = 2 : i64, issue_token = true } : memref<18xi32>

      // Wait for all outputs to complete
      aiex.npu.dma_wait { symbol = @objFifo_out_shim0 }
      aiex.npu.dma_wait { symbol = @objFifo_out_shim1 }
      aiex.npu.dma_wait { symbol = @objFifo_out_shim2 }
    }
  }
}
