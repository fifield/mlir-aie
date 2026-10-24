// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate --aie-place-tiles --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=ROUTE

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Channel selection before placement must allow four direct shim inputs.
module {
  aie.device(npu3) {
    %shim = aie.logical_tile<ShimNOCTile>(0, ?)
    %mem = aie.logical_tile<MemTile>(0, ?)
    aie.objectfifo.pool @p0(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @p1(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @p2(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @p3(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.route_endpoint @src0(%shim) DMA {fifoName = "f0"}
    aie.route_endpoint @src1(%shim) DMA {fifoName = "f1"}
    aie.route_endpoint @src2(%shim) DMA {fifoName = "f2"}
    aie.route_endpoint @src3(%shim) DMA {fifoName = "f3"}
    aie.objectfifo.dma_endpoint @dst0(%mem) fills @p0
    aie.objectfifo.dma_endpoint @dst1(%mem) fills @p1
    aie.objectfifo.dma_endpoint @dst2(%mem) fills @p2
    aie.objectfifo.dma_endpoint @dst3(%mem) fills @p3
    aie.route from @src0 to [@dst0]
    aie.route from @src1 to [@dst1]
    aie.route from @src2 to [@dst2]
    aie.route from @src3 to [@dst3]

    aie.objectfifo.dma_endpoint @out0(%mem) drains @p0
    aie.objectfifo.dma_endpoint @out1(%mem) drains @p1
    aie.objectfifo.dma_endpoint @out2(%mem) drains @p2
    aie.objectfifo.dma_endpoint @out3(%mem) drains @p3
    aie.objectfifo.dma_endpoint @out4(%mem) drains @p0
  }
}

// CHECK: aie.objectfifo.dma_endpoint @dst0({{.*}}) fills @p0 {channelIndex = 0 : i32}
// CHECK: aie.objectfifo.dma_endpoint @dst1({{.*}}) fills @p1 {channelIndex = 1 : i32}
// CHECK: aie.objectfifo.dma_endpoint @dst2({{.*}}) fills @p2 {channelIndex = 4 : i32}
// CHECK: aie.objectfifo.dma_endpoint @dst3({{.*}}) fills @p3 {channelIndex = 5 : i32}
// Drains skip MM2S 4, which reaches only North0; the order among the rest is
// the allocator's.
// CHECK-NOT: drains @p{{[0-9]}} {channelIndex = 4 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out4({{.*}}) drains @p0
// ROUTE: aie.connect<South : 0, DMA : 0>
// ROUTE: aie.connect<South : 1, DMA : 1>
// ROUTE: aie.connect<South : 2, DMA : 4>
// ROUTE: aie.connect<South : 3, DMA : 5>
