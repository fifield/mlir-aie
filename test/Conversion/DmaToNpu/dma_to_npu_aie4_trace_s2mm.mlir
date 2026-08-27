//===- dma_to_npu_aie4_trace_s2mm.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -convert-aie-to-transaction %s | FileCheck %s --check-prefix=TXN

// Trace S2MM uses a dedicated shim DMA with its own four-BD private pool and
// channel-control register. It must not alias the regular S2MM resources.

// BD 0 starts at the trace pool base, 0x18480.
// TXN: aiex.npu.blockwrite({{.*}}) {address = 99456 : ui32}
// The trace channel task queue is control address 0x18548 + 4 = 0x1854c.
// Enable_Token_Issue is bit 31; BD 0 with repeat count 1 sets no other bits.
// TXN-DAG: %[[QUEUE:.*]] = arith.constant 99660 : i32
// TXN-DAG: %[[TOKEN:.*]] = arith.constant {{-?}}2147483648 : i32
// TXN: aiex.npu.write32(%[[QUEUE]], %[[TOKEN]]) : i32, i32

module {
  aie.device(npu3) {
    %buffer = aie.external_buffer {address = 2147483648 : i64} : memref<64xi32>
    %shim = aie.tile(0, 0)
    %dma = aie.shim_dma(%shim) {
      %start = aie.dma_start(TraceS2MM, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64 sizes = [64] strides = [1]) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
