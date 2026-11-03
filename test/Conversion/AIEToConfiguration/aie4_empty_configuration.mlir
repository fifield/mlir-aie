//===- aie4_empty_configuration.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --convert-aie-to-transaction %s | FileCheck %s

// An AIE4 device that configures nothing of its own (here a pure dispatcher)
// gets an empty configuration: it does not disable the other devices'
// connections either (tile (0, 2) master DMA:0 at 0x6b4008 = 7028744), so it
// has no section and the load of the device it dispatches leads its program.
// CHECK-LABEL: aie.device(npu3_1col) {
// CHECK: aie.runtime_sequence @configure() attributes {aie.generated_configure} {
// CHECK-NEXT: }
module {
  aie.device(npu3_1col) @main {
    aie.runtime_sequence @seq(%x: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @other}
    }
  }
  aie.device(npu3_1col) @other {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      aie.connect<South : 0, DMA : 0>
    }
    aie.runtime_sequence @run(%x: memref<16xi32>) {
    }
  }
}

// -----

// Routing and shim DMA allocations of its own (materialization copies those
// of the runtime sequences it calls) do not count: such a dispatcher gets an
// empty configuration too.
// CHECK-LABEL: aie.device(npu3_1col) {
// CHECK: aie.runtime_sequence @configure() attributes {aie.generated_configure} {
// CHECK-NEXT: }
module {
  aie.device(npu3_1col) @main {
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      aie.connect<South : 0, DMA : 0>
    }
    aie.shim_dma_allocation @in (%s, MM2S, 0)
    aie.runtime_sequence @seq(%x: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @other}
    }
  }
  aie.device(npu3_1col) @other {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      aie.connect<South : 0, DMA : 0>
    }
    aie.runtime_sequence @run(%x: memref<16xi32>) {
    }
  }
}

// -----

// A device with configuration of its own (a memtile lock) still disables the
// other devices' connections.
// CHECK-LABEL: aie.device(npu3_1col) {
// CHECK: aie.runtime_sequence @configure
// CHECK: aiex.npu.write32(%c7028744_i32, %c0_i32)
module {
  aie.device(npu3_1col) @main {
    %m = aie.tile(0, 1)
    %l = aie.lock(%m, 0) {init = 1 : i32}
    aie.runtime_sequence @seq(%x: memref<16xi32>) {
    }
  }
  aie.device(npu3_1col) @other {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      aie.connect<South : 0, DMA : 0>
    }
    aie.runtime_sequence @run(%x: memref<16xi32>) {
    }
  }
}
