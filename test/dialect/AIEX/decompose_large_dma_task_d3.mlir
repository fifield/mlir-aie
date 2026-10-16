//===- decompose_large_dma_task_d3.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MemTiles and AIE4 shim NOC tiles have a D3 address dimension besides the
// iteration one, so a five-entry [iteration, d3, d2, d1, d0] descriptor that
// the hardware takes is lowered as it is by aie-dma-tasks-to-npu.
// aie-decompose-large-dma-bd must leave such a BD alone: reducing it as if it
// had only [iteration, d2, d1, d0] drops the unit (or stride-0) iteration
// entry, slides d3 into the iteration slot and shrinks len to one d3 step, so
// the pushed task moves only 1/d3_size of the data. A five-entry descriptor is
// taken as having a d3 when its length covers d3 x d2 x d1 x d0.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,canonicalize,aie-normalize-dma-bd-dims,aie-decompose-large-dma-bd,aie-split-long-repeats,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWERED
//
// No "iteration dimension ... pushed with repeat_count 0" warning: d3 is not
// mistaken for an iteration dimension.
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,canonicalize,aie-normalize-dma-bd-dims,aie-decompose-large-dma-bd,aie-split-long-repeats,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file --verify-diagnostics %s -o /dev/null

// An AIE4 shim NOC tile BD with a size-1 iteration entry and a d3 of 8.
// CHECK-LABEL: @shim_d3
// CHECK:         aiex.dma_configure_task_for @kv
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 262144 sizes = [1, 8, 2, 32, 512] strides = [0, 16384, 393216, 512, 1])
// CHECK-NEXT:      aie.end
// CHECK-NOT:     aiex.dma_configure_task_for
// LOWERED-LABEL: @shim_d3
// LOWERED:         aiex.npu.writebd
// LOWERED-SAME:      buffer_length = 131072 : i32
// LOWERED-SAME:      d3_size = 8 : i32, d3_stride = 8192 : i32
// LOWERED-SAME:      iteration_size = 0 : i32, iteration_stride = 0 : i32
// LOWERED-NOT:     aiex.npu.writebd
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @kv (%t, MM2S, 0)
    aie.runtime_sequence @shim_d3(%in: memref<786432xbf16>) {
      %0 = aiex.dma_configure_task_for @kv {
        aie.dma_bd(%in : memref<786432xbf16> offset = 0 len = 262144 sizes = [1, 8, 2, 32, 512] strides = [0, 16384, 393216, 512, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A stride-0 iteration entry of size 2 is no real iteration either; the BD is
// lowered the same way.
// CHECK-LABEL: @shim_d3_stride0_iteration
// CHECK:         aiex.dma_configure_task_for @kv
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 262144 sizes = [2, 8, 2, 32, 512] strides = [0, 16384, 393216, 512, 1])
// CHECK-NEXT:      aie.end
// CHECK-NOT:     aiex.dma_configure_task_for
// LOWERED-LABEL: @shim_d3_stride0_iteration
// LOWERED:         aiex.npu.writebd
// LOWERED-SAME:      buffer_length = 131072 : i32
// LOWERED-SAME:      d3_size = 8 : i32, d3_stride = 8192 : i32
// LOWERED-SAME:      iteration_size = 0 : i32, iteration_stride = 0 : i32
// LOWERED-NOT:     aiex.npu.writebd
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @kv (%t, MM2S, 0)
    aie.runtime_sequence @shim_d3_stride0_iteration(%in: memref<786432xbf16>) {
      %0 = aiex.dma_configure_task_for @kv {
        aie.dma_bd(%in : memref<786432xbf16> offset = 0 len = 262144 sizes = [2, 8, 2, 32, 512] strides = [0, 16384, 393216, 512, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A contiguous shim D3 transfer can exceed the ND wrap widths because it
// lowers through buffer_length. A stride-0 iteration uses the queue repeat.
// CHECK-LABEL: @shim_contiguous_d3_repeat
// CHECK:         aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 262144 sizes = [2, 8, 2, 2, 8192] strides = [0, 32768, 16384, 8192, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 1 : i32}
// CHECK-NOT:     aiex.dma_configure_task_for
// LOWERED-LABEL: @shim_contiguous_d3_repeat
// LOWERED:         aiex.npu.writebd
// LOWERED-SAME:      buffer_length = 131072 : i32
// LOWERED-SAME:      iteration_size = 0 : i32, iteration_stride = 0 : i32
// LOWERED-NOT:     aiex.npu.writebd
// LOWERED-NOT:     aiex.npu.push_queue
// LOWERED:         [[REPEAT:%[a-zA-Z0-9_]+]] = arith.constant 1 : i32
// LOWERED-NEXT:    aiex.npu.push_queue(0, 0, MM2S : 0) bd_id {{%[a-zA-Z0-9_]+}} repeat [[REPEAT]]
// LOWERED-NOT:     aiex.npu.writebd
// LOWERED-NOT:     aiex.npu.push_queue
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @shim_contiguous_d3_repeat(%in: memref<262144xbf16>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<262144xbf16> offset = 0 len = 262144 sizes = [2, 8, 2, 2, 8192] strides = [0, 32768, 16384, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A positive-stride iteration also stays outside the linear D0..D3 scan.
// CHECK-LABEL: @shim_contiguous_d3_iteration
// CHECK:         aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 262144 sizes = [2, 8, 2, 2, 8192] strides = [262144, 32768, 16384, 8192, 1])
// CHECK-NEXT:      aie.end
// CHECK-NEXT:    } {issue_token = true, repeat_count = 1 : i32}
// CHECK-NOT:     aiex.dma_configure_task_for
// LOWERED-LABEL: @shim_contiguous_d3_iteration
// LOWERED:         aiex.npu.writebd
// LOWERED-SAME:      buffer_length = 131072 : i32
// LOWERED-SAME:      iteration_size = 1 : i32, iteration_stride = 131072 : i32
// LOWERED-NOT:     aiex.npu.writebd
// LOWERED-NOT:     aiex.npu.push_queue
// LOWERED:         [[REPEAT:%[a-zA-Z0-9_]+]] = arith.constant 1 : i32
// LOWERED-NEXT:    aiex.npu.push_queue(0, 0, MM2S : 0) bd_id {{%[a-zA-Z0-9_]+}} repeat [[REPEAT]]
// LOWERED-NOT:     aiex.npu.writebd
// LOWERED-NOT:     aiex.npu.push_queue
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @shim_contiguous_d3_iteration(%in: memref<524288xbf16>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<524288xbf16> offset = 0 len = 262144 sizes = [2, 8, 2, 2, 8192] strides = [262144, 32768, 16384, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A MemTile BD has a d3 on every architecture.
// CHECK-LABEL: @memtile_d3
// CHECK:         aiex.dma_configure_task
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 4096 sizes = [1, 4, 2, 16, 32] strides = [0, 32, 2048, 128, 1])
// CHECK-NEXT:      aie.end
// CHECK-NOT:     aiex.dma_configure_task
// LOWERED-LABEL: @memtile_d3
// LOWERED:         aiex.npu.writebd
// LOWERED-SAME:      buffer_length = 2048 : i32
// LOWERED-SAME:      d3_size = 4 : i32, d3_stride = 16 : i32
// LOWERED-SAME:      iteration_size = 0 : i32, iteration_stride = 0 : i32
// LOWERED-NOT:     aiex.npu.writebd
module {
  aie.device(npu3) {
    %m = aie.tile(0, 1)
    %buf = aie.buffer(%m) {address = 0 : i32} : memref<4096xbf16>
    aie.runtime_sequence @memtile_d3() {
      %0 = aiex.dma_configure_task(%m, MM2S, 0) {
        aie.dma_bd(%buf : memref<4096xbf16> offset = 0 len = 4096 sizes = [1, 4, 2, 16, 32] strides = [0, 32, 2048, 128, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A lock-taking descriptor is not reduced, but one that fits with d3 is not
// reported either.
// CHECK-LABEL: @memtile_d3_locks
// CHECK:         aie.use_lock
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 4096 sizes = [1, 4, 2, 16, 32] strides = [0, 32, 2048, 128, 1])
// LOWERED-LABEL: @memtile_d3_locks
// LOWERED:         aiex.npu.writebd
// LOWERED-SAME:      buffer_length = 2048 : i32
// LOWERED-SAME:      d3_size = 4 : i32, d3_stride = 16 : i32
// LOWERED-SAME:      iteration_size = 0 : i32, iteration_stride = 0 : i32
module {
  aie.device(npu3) {
    %m = aie.tile(0, 1)
    %buf = aie.buffer(%m) {address = 0 : i32} : memref<4096xbf16>
    %lk = aie.lock(%m) {init = 0 : i32}
    aie.runtime_sequence @memtile_d3_locks() {
      %c1 = arith.constant 1 : i32
      %0 = aiex.dma_configure_task(%m, MM2S, 0) {
        aie.use_lock(%lk, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<4096xbf16> offset = 0 len = 4096 sizes = [1, 4, 2, 16, 32] strides = [0, 32, 2048, 128, 1])
        aie.use_lock(%lk, Release, %c1)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A five-entry descriptor whose length leaves d3 out (here 2 x 32 x 32) gives
// two iteration dimensions, not a d3, and is reduced as on a tile without one:
// as in decompose_large_dma_task_nd.mlir, it splits into 3 pieces.
// CHECK-LABEL: @shim_two_iterations
// CHECK:         aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 0 len = 2048 sizes = [4, 2, 32, 32] strides = [32, 4096, 1024, 1])
// CHECK:         aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 262144 len = 2048 sizes = [4, 2, 32, 32] strides = [32, 4096, 1024, 1])
// CHECK:         aiex.dma_configure_task_for @a
// CHECK-NEXT:      aie.dma_bd({{.*}} offset = 524288 len = 2048 sizes = [4, 2, 32, 32] strides = [32, 4096, 1024, 1])
// LOWERED-LABEL: @shim_two_iterations
// LOWERED-COUNT-3: aiex.npu.writebd
// LOWERED-NOT:     aiex.npu.writebd
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @shim_two_iterations(%in: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<1048576xbf16> offset = 0 len = 2048 sizes = [3, 4, 2, 32, 32] strides = [262144, 32, 4096, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 11 : i32}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}
