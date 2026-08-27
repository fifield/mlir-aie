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
// RUN: aiecc --tmpdir=project_base --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile --aie-generate-txn --txn-name=transaction.bin --aiesim aie_overlay.mlir %test_utils_flags %S/test.cpp
// RUN: aie-opt -aie-assign-tile-controller-ids -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" %S/ctrlpkt.mlir -o ctrlpkt_overlay.mlir
// RUN: aie-translate --aie-ctrlpkt-to-bin ctrlpkt_overlay.mlir -o ctrlpkt.bin
// RUN: project_base/mesim.sh | FileCheck %s
//
// CHECK: Control packet lock value: 0x2
// CHECK: PASS!

module {
  aie.device(npu3) {
    %shim_0_0 = aie.tile(0, 0)
    %core_0_2 = aie.tile(0, 2)

    // The base image installs the control-packet route and leaves this lock at
    // zero. The host validates the generated packet and mirrors its payload
    // into this simulated register.
    %lock_0_2 = aie.lock(%core_0_2, 0) {init = 0 : i32}
  }
}
