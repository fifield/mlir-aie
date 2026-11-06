//===- write_reset.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// On AIE4 a write_config follows a write_reset, or a write_teardown standing
// in for it.
module {
  aie.device(npu3_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu3_1col) @main {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @seq() {
      aiex.npu.write_reset
      aiex.npu.write_config @a
      aiex.npu.write_teardown @a
      aiex.npu.write_config @a
    }
  }
}

// -----

// A load_pdi does not reset an AIE4 array.
module {
  aie.device(npu3_1col) @empty {
  }
  aie.device(npu3_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu3_1col) @main {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @empty}
      // expected-error @+1 {{must follow the write_reset that resets the array}}
      aiex.npu.write_config @a
    }
  }
}

// -----

// On other targets a write_reset may stand in for the load_pdi of an empty
// device.
module {
  aie.device(npu2) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @seq() {
      aiex.npu.write_reset
      aiex.npu.write_config @a
    }
  }
}
