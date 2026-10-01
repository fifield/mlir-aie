//===- npu3_other_device_switches_unchanged.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// No connection is disabled for an npu3 device that is alone in its module, or
// for a device that is not AIE4.

// RUN: aie-translate --split-input-file --aie-generate-cdo --cdo-debug=true --aie-device-name=b %s 2>&1 | FileCheck %s

// CHECK-NOT: Data:  0x00000000
// CHECK:     Address:  0x00000000006B4024 Data:  0x80000001
// CHECK-NOT: Data:  0x00000000
// CHECK:     Address:  0x000000000023F01C Data:  0x80000001
// CHECK-NOT: Data:  0x00000000

module {
  aie.device(npu3) @b {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, South : 2>
    }
  }
}

// -----

module {
  aie.device(npu2) @a {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, South : 1>
    }
  }
  aie.device(npu2) @b {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, South : 2>
    }
  }
}
