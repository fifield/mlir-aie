//===- aie4.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: mesimulator
//
// RUN: rm -rf project_base aie_overlay.mlir ctrlpkt_overlay.mlir transaction.bin ctrlpkt.bin
// RUN: aie-opt -aie-assign-tile-controller-ids -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" %s -o aie_overlay.mlir
// RUN: aiecc --tmpdir=project_base --alloc-scheme=basic-sequential --xchesscc --xbridge --aie-generate-txn --txn-name=transaction.bin --aiesim aie_overlay.mlir %test_utils_flags %S/test.cpp
// RUN: FileCheck %s --check-prefix=TXN < project_base/main_txn.mlir
// RUN: aie-opt -aie-assign-tile-controller-ids -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" %S/ctrlpkt.mlir -o ctrlpkt_overlay.mlir
// RUN: aie-translate --aie-ctrlpkt-to-bin ctrlpkt_overlay.mlir -o ctrlpkt.bin
// RUN: project_base/mesim.sh | FileCheck %s
//
// CHECK: Control packet lock value: 0x2
// CHECK: PASS!
// TXN: aiex.npu.write32 {address = 6959200 : ui32, value = 1 : ui32}
// TXN: aiex.npu.blockwrite(%{{.*}}) {address = 99072 : ui32}
// TXN: aiex.npu.write32 {address = 99684 : ui32
// TXN: aie.masterset(TileControl : 0, %{{.*}}) {keep_pkt_header = false}
// TXN-NEXT: aie.packet_rules(SouthControl : 0)
// TXN: aie.packet_source<%{{.*}}, TileControl : 0>
// TXN: aie.packet_dest<%{{.*}}, TileControl : 0>
// TXN: aie.connect<TileControl : 0, NorthControl : 0>
// TXN: aie.masterset(NorthControl : 0, %{{.*}})
// TXN-NEXT: aie.packet_rules(SouthControl : 0)

module {
  aie.device(npu3) {
    %shim_0_0 = aie.tile(0, 0)
    %core_0_2 = aie.tile(0, 2)

    %ctrlpkt_ddr = aie.external_buffer {address = 0x80000000 : i64, sym_name = "ctrlpkt_ddr"} : memref<3xi32>
    %shim_lock_start = aie.lock(%shim_0_0, 0) {init = 1 : i32}
    %shim_lock_done = aie.lock(%shim_0_0, 1)
    %shim_dma_0_0 = aie.shim_dma(%shim_0_0) {
      %0 = aie.dma_start(TileControl, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%shim_lock_start, AcquireGreaterEqual, 1)
      aie.dma_bd(%ctrlpkt_ddr : memref<3xi32>, 0, 3) {bd_id = 0 : i32}
      aie.use_lock(%shim_lock_done, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    %lock_0_2 = aie.lock(%core_0_2, 0) {init = 0 : i32}

  }
}
