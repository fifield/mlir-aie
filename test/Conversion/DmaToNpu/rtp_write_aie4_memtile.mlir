//===- rtp_write_aie4_memtile.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An rtp_write lowers to a tile-relative write32, which cannot reach an AIE4
// mem tile's data memory: it starts 1 MiB into the tile's address window. So
// a mem tile RTP is written by buffer instead, which aie-dma-to-npu resolves
// to an absolute address. Word 2 of a buffer at 64 in mem
// tile (0, 1) is 0x200000 + 0x48 = 2097224. A core tile RTP keeps the
// tile-relative form: word 1 of a buffer at 1024 in core tile (0, 2) is 0x404
// = 1028, at 0x600404 = 6292484.

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s
// RUN: aie-opt --aie-dma-to-npu --aie-npu-to-cert %s | FileCheck %s --check-prefix=CERT

// CHECK-DAG: %[[A0:.*]] = arith.constant 2097224 : i32
// CHECK-DAG: %[[V0:.*]] = arith.constant 50 : i32
// CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
// CHECK-DAG: %[[A1:.*]] = arith.constant 1028 : i32
// CHECK-DAG: %[[V1:.*]] = arith.constant 99 : i32
// CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) {column = 0 : i32, row = 2 : i32} : i32, i32
// CERT: aiex.cert.write32(2097224, 50)
// CERT: aiex.cert.write32(6292484, 99)
module {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %m = aie.buffer(%mem) {address = 64 : i32, sym_name = "mem_rtp"} : memref<16xi32>
    %c = aie.buffer(%core) {address = 1024 : i32, sym_name = "core_rtp"} : memref<16xi32>
    aie.runtime_sequence() {
      %v0 = arith.constant 50 : i32
      aiex.npu.rtp_write(@mem_rtp, 2, %v0) : i32
      %v1 = arith.constant 99 : i32
      aiex.npu.rtp_write(@core_rtp, 1, %v1) : i32
    }
  }
}
