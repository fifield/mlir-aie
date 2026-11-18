//===- npu3_bd_pools.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids %s | FileCheck %s

// Each AIE4 memtile channel has a private 16-BD pool with channel-local ids.
// Shim banks: MM2S 0-1 and S2MM 0 use BDs 0-15; MM2S 2-3 and S2MM 1 use 16-31.

// CHECK-LABEL: aie.device(npu3)
// CHECK:       aie.memtile_dma
// CHECK:         aie.dma_start(MM2S, 0, ^[[M0:.*]], ^{{.*}})
// CHECK:       ^[[M0]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 1 : i32, next_bd_id = 0 : i32}
// CHECK:         aie.dma_start(S2MM, 0, ^[[S0:.*]], ^{{.*}})
// CHECK:       ^[[S0]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:         aie.dma_start(MM2S, 9, ^[[M9:.*]], ^{{.*}})
// CHECK:       ^[[M9]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:         aie.dma_start(S2MM, 7, ^[[S7:.*]], ^{{.*}})
// CHECK:       ^[[S7]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 5 : i32, next_bd_id = 0 : i32}
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 5 : i32}
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_start(S2MM, 1, ^[[SS1:.*]], ^{{.*}})
// CHECK:       ^[[SS1]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 16 : i32
// CHECK:         aie.dma_start(MM2S, 1, ^[[SM1:.*]], ^{{.*}})
// CHECK:       ^[[SM1]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK:         aie.dma_start(MM2S, 2, ^[[SM2:.*]], ^{{.*}})
// CHECK:       ^[[SM2]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 17 : i32
// CHECK:         aie.dma_start(S2MM, 0, ^[[SS0:.*]], ^{{.*}})
// CHECK:       ^[[SS0]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 1 : i32
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %a = aie.buffer(%mem) : memref<256xi32>
    %b = aie.buffer(%mem) : memref<256xi32>
    %ext = aie.external_buffer : memref<256xi32>
    aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 0, ^m0a, ^s0)
    ^m0a:
      aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^m0b
    ^m0b:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^m0a
    ^s0:
      %1 = aie.dma_start(S2MM, 0, ^s0a, ^m9)
    ^s0a:
      aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^s0a
    ^m9:
      %2 = aie.dma_start(MM2S, 9, ^m9a, ^s7)
    ^m9a:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^m9a
    ^s7:
      %3 = aie.dma_start(S2MM, 7, ^s7a, ^end)
    ^s7a:
      aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256) {bd_id = 5 : i32}
      aie.next_bd ^s7b
    ^s7b:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^s7a
    ^end:
      aie.end
    }
    aie.shim_dma(%shim) {
      %0 = aie.dma_start(S2MM, 1, ^ss1, ^sm1)
    ^ss1:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^end
    ^sm1:
      %1 = aie.dma_start(MM2S, 1, ^sm1a, ^sm2)
    ^sm1a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^end
    ^sm2:
      %2 = aie.dma_start(MM2S, 2, ^sm2a, ^ss0)
    ^sm2a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^end
    ^ss0:
      %3 = aie.dma_start(S2MM, 0, ^ss0a, ^end)
    ^ss0a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
