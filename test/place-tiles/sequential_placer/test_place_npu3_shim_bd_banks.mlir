//===- test_place_npu3_shim_bd_banks.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// npu3 shim BD banks: MM2S 0-1 and S2MM 0 use BDs 0-15, MM2S 2-3 and S2MM 1
// use 16-31. Two LTOs needing 12 and 8 BDs on MM2S 0/1 overflow the lower bank,
// so the placer puts them on separate shims.

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-place-tiles,aie.device(aie-objectFifo-stateful-transform,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %s | FileCheck %s --check-prefix=LOWER

// CHECK-LABEL: @split_shims
// CHECK-DAG:   %[[S0:.*]] = aie.tile(0, 0)
// CHECK-DAG:   %[[S1:.*]] = aie.tile(1, 0)
// CHECK:       aie.objectfifo @in_a(%[[S0]],
// CHECK:       aie.objectfifo @in_b(%[[S1]],

// LOWER-LABEL: @split_shims
// LOWER-COUNT-20: aie.dma_bd({{.*}}bd_id =
module @split_shims {
  aie.device(npu3) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in_a(%shim_a, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @in_b(%shim_b, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @in_a(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_a(Consume, 1)
      %1 = aie.objectfifo.acquire @in_b(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<67108864xbf16>, %b: memref<67108864xbf16>) {
      %ta0 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<67108864xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [16777216, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta0)
      %ta1 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<67108864xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [16777216, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta1)
      %ta2 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<67108864xbf16> offset = 128 len = 524288 sizes = [4, 128, 64, 64] strides = [16777216, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta2)
      %tb0 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<67108864xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [16777216, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb0)
      %tb1 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<67108864xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [16777216, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb1)
      aiex.dma_await_task(%ta0)
      aiex.dma_await_task(%ta1)
      aiex.dma_await_task(%ta2)
      aiex.dma_await_task(%tb0)
      aiex.dma_await_task(%tb1)
    }
  }
}
