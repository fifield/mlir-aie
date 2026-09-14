// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocate every physical channel, using constrained channels as fallbacks.
module @full_capacity {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @in0(%mem) fills @p
    aie.objectfifo.dma_endpoint @in1(%mem) fills @p
    aie.objectfifo.dma_endpoint @in2(%mem) fills @p
    aie.objectfifo.dma_endpoint @in3(%mem) fills @p
    aie.objectfifo.dma_endpoint @in4(%mem) fills @p
    aie.objectfifo.dma_endpoint @in5(%mem) fills @p
    aie.objectfifo.dma_endpoint @in6(%mem) fills @p
    aie.objectfifo.dma_endpoint @in7(%mem) fills @p
    aie.objectfifo.dma_endpoint @out0(%mem) drains @p
    aie.objectfifo.dma_endpoint @out1(%mem) drains @p
    aie.objectfifo.dma_endpoint @out2(%mem) drains @p
    aie.objectfifo.dma_endpoint @out3(%mem) drains @p
    aie.objectfifo.dma_endpoint @out4(%mem) drains @p
    aie.objectfifo.dma_endpoint @out5(%mem) drains @p
    aie.objectfifo.dma_endpoint @out6(%mem) drains @p
    aie.objectfifo.dma_endpoint @out7(%mem) drains @p
    aie.objectfifo.dma_endpoint @out8(%mem) drains @p
    aie.objectfifo.dma_endpoint @out9(%mem) drains @p
  }
}

// CHECK-LABEL: module @full_capacity
// CHECK: aie.objectfifo.dma_endpoint @in0({{.*}}) fills @p {channelIndex = 0 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in1({{.*}}) fills @p {channelIndex = 1 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in2({{.*}}) fills @p {channelIndex = 4 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in3({{.*}}) fills @p {channelIndex = 5 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in4({{.*}}) fills @p {channelIndex = 2 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in5({{.*}}) fills @p {channelIndex = 3 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in6({{.*}}) fills @p {channelIndex = 6 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in7({{.*}}) fills @p {channelIndex = 7 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out0({{.*}}) drains @p {channelIndex = 0 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out1({{.*}}) drains @p {channelIndex = 1 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out2({{.*}}) drains @p {channelIndex = 2 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out3({{.*}}) drains @p {channelIndex = 3 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out4({{.*}}) drains @p {channelIndex = 5 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out5({{.*}}) drains @p {channelIndex = 6 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out6({{.*}}) drains @p {channelIndex = 7 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out7({{.*}}) drains @p {channelIndex = 8 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out8({{.*}}) drains @p {channelIndex = 4 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out9({{.*}}) drains @p {channelIndex = 9 : i32}

// -----

// Pins take precedence even when declared after unpinned logical endpoints.
module @pinned_channels {
  aie.device(npu3) {
    %mem = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @in0(%mem) fills @p
    aie.objectfifo.dma_endpoint @in1(%mem) fills @p
    aie.objectfifo.dma_endpoint @in2(%mem) fills @p
    aie.objectfifo.dma_endpoint @in3(%mem) fills @p
    aie.objectfifo.dma_endpoint @in4(%mem) fills @p
    aie.objectfifo.dma_endpoint @in5(%mem) fills @p
    aie.objectfifo.dma_endpoint @in_preferred(%mem) fills @p {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @in_fallback(%mem) fills @p {channelIndex = 2 : i32}
    aie.objectfifo.dma_endpoint @out0(%mem) drains @p
    aie.objectfifo.dma_endpoint @out1(%mem) drains @p
    aie.objectfifo.dma_endpoint @out2(%mem) drains @p
    aie.objectfifo.dma_endpoint @out3(%mem) drains @p
    aie.objectfifo.dma_endpoint @out4(%mem) drains @p
    aie.objectfifo.dma_endpoint @out5(%mem) drains @p
    aie.objectfifo.dma_endpoint @out6(%mem) drains @p
    aie.objectfifo.dma_endpoint @out7(%mem) drains @p
    aie.objectfifo.dma_endpoint @out_preferred(%mem) drains @p {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @out_fallback(%mem) drains @p {channelIndex = 4 : i32}
  }
}

// CHECK-LABEL: module @pinned_channels
// CHECK: aie.objectfifo.dma_endpoint @in0({{.*}}) fills @p {channelIndex = 1 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in1({{.*}}) fills @p {channelIndex = 4 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in2({{.*}}) fills @p {channelIndex = 5 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in3({{.*}}) fills @p {channelIndex = 3 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in4({{.*}}) fills @p {channelIndex = 6 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in5({{.*}}) fills @p {channelIndex = 7 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in_preferred({{.*}}) fills @p {channelIndex = 0 : i32}
// CHECK: aie.objectfifo.dma_endpoint @in_fallback({{.*}}) fills @p {channelIndex = 2 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out0({{.*}}) drains @p {channelIndex = 1 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out1({{.*}}) drains @p {channelIndex = 2 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out2({{.*}}) drains @p {channelIndex = 3 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out3({{.*}}) drains @p {channelIndex = 5 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out4({{.*}}) drains @p {channelIndex = 6 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out5({{.*}}) drains @p {channelIndex = 7 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out6({{.*}}) drains @p {channelIndex = 8 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out7({{.*}}) drains @p {channelIndex = 9 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out_preferred({{.*}}) drains @p {channelIndex = 0 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out_fallback({{.*}}) drains @p {channelIndex = 4 : i32}

// -----

module @s2mm_exhausted {
  aie.device(npu3) {
    // expected-error@+1 {{number of input DMA channel exceeded!}}
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @dma0(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma1(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma2(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma3(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma4(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma5(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma6(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma7(%mem) fills @p
    aie.objectfifo.dma_endpoint @dma8(%mem) fills @p
  }
}

// -----

module @mm2s_exhausted {
  aie.device(npu3) {
    // expected-error@+1 {{number of output DMA channel exceeded!}}
    %mem = aie.tile(0, 1)
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @dma0(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma1(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma2(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma3(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma4(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma5(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma6(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma7(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma8(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma9(%mem) drains @p
    aie.objectfifo.dma_endpoint @dma10(%mem) drains @p
  }
}
