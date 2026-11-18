//===- npu3_refused.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A PDI load does not reset an AIE4 array, so load_pdi expansion is refused.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-expand-load-pdi %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-expand-load-pdi="ctrl-pkt=true" %s | FileCheck %s

module {
  aie.device(npu3) @config {
    %t02 = aie.tile(0, 2)
  }
  aie.device(npu3) @main {
    aie.runtime_sequence @seq() {
      // expected-error@+1 {{'aiex.npu.load_pdi' op cannot be expanded on AIE4: a PDI load does not reset the array}}
      aiex.npu.load_pdi {device_ref = @config}
    }
  }
}

// -----

// A load_pdi that is not expanded is left alone.
// CHECK-NOT: @empty
// CHECK: aiex.npu.load_pdi {device_ref = @config, expand_mode = 0 : i32}
// CHECK-NOT: @empty
module {
  aie.device(npu3) @config {
    %t02 = aie.tile(0, 2)
  }
  aie.device(npu3) @main {
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @config, expand_mode = 0 : i32}
    }
  }
}
