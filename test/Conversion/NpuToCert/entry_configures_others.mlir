//===- entry_configures_others.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-npu-to-cert -split-input-file -verify-diagnostics %s

// An AIE4 entry that configures resources of its own may not also configure
// other devices: their configuration starts with a reset of the partition,
// which would undo the entry's.

module {
  aie.device(npu3) @cfg {
    aie.runtime_sequence @configure() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x22222222 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
  // expected-error @+1 {{the entry device configures resources of its own and also configures other devices}}
  aie.device(npu3) @main {
    aie.runtime_sequence @configure() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x11111111 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// The same for configuration written in line (--expand-load-pdis).

module {
  aie.device(npu3) @cfg {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 1 : i32}
  }
  // expected-error @+1 {{the entry device configures resources of its own and also configures other devices}}
  aie.device(npu3) @main {
    aie.runtime_sequence @configure() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x11111111 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @seq() {
      aiex.npu.write_reset
      aiex.npu.write_config @cfg
    }
  }
}
