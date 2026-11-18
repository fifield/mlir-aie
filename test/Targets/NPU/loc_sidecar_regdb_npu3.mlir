// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- loc_sidecar_regdb_npu3.mlir ------------------------*- MLIR -*-===//
//
//===----------------------------------------------------------------------===//

// The locmap sidecar names npu3 registers from the AIE4 regdb; memtiles span
// five 1MB address rows, so core row r is address row r + 4.

// RUN: mkdir -p loc_sidecar_regdb_npu3
// RUN: aie-translate --aie-npu-to-binary --aie-npu-emit-locmap=loc_sidecar_regdb_npu3/locmap.json %s -o loc_sidecar_regdb_npu3/insts.bin
// RUN: FileCheck %s < loc_sidecar_regdb_npu3/locmap.json

module {
  aie.device(npu3) {
    aie.runtime_sequence(%arg0: memref<16xf32>) {
      // col 0, row 2 (core tile, address row 6), offset 0xA3000 -> CORE_CONTROL.
      %cst_npu_0 = arith.constant 0x6A3000 : i32
      %cst_npu_1 = arith.constant 0x1 : i32
      aiex.npu.write32(%cst_npu_0, %cst_npu_1) : i32, i32
      // col 1, row 2 (core tile), memory module offset 0xACE00 -> DMA_S2MM_0_CTRL.
      %cst_npu_2 = arith.constant 0x26ACE00 : i32
      %cst_npu_3 = arith.constant 0x2 : i32
      aiex.npu.write32(%cst_npu_2, %cst_npu_3) : i32, i32
      // col 0, row 1 (mem tile), offset 0x9E00 -> DMA_S2MM_0_CTRL.
      %cst_npu_4 = arith.constant 0x109E00 : i32
      %cst_npu_5 = arith.constant 0x3 : i32
      aiex.npu.write32(%cst_npu_4, %cst_npu_5) : i32, i32
      // col 0, row 0 (shim tile), offset 0x18544 -> DMA_S2MM_0_TASK_QUEUE.
      %cst_npu_6 = arith.constant 0x18544 : i32
      %cst_npu_7 = arith.constant 0x4 : i32
      aiex.npu.write32(%cst_npu_6, %cst_npu_7) : i32, i32
    }
  }
}

// CHECK: "address": "0x6A3000"
// CHECK: "register": "CORE_CONTROL"
// CHECK-NEXT: "register_module": "core"

// CHECK: "address": "0x26ACE00"
// CHECK: "register": "DMA_S2MM_0_CTRL"
// CHECK-NEXT: "register_module": "memory"

// CHECK: "address": "0x109E00"
// CHECK: "register": "DMA_S2MM_0_CTRL"
// CHECK-NEXT: "register_module": "memory_tile"

// CHECK: "address": "0x18544"
// CHECK: "register": "DMA_S2MM_0_TASK_QUEUE"
// CHECK-NEXT: "register_module": "shim"
