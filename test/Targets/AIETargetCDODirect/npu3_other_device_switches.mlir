//===- npu3_other_device_switches.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A device's configuration first disables the stream switch connections of
// the other devices with the same target, then enables its own.

// RUN: aie-translate --aie-generate-cdo --cdo-debug=true --aie-device-name=b %s 2>&1 | FileCheck %s --check-prefix=B
// RUN: aie-translate --aie-generate-cdo --cdo-debug=true --aie-device-name=a %s 2>&1 | FileCheck %s --check-prefix=A
// RUN: aie-translate --aie-generate-cdo --cdo-debug=true --aie-device-name=c %s 2>&1 | FileCheck %s --check-prefix=C
// RUN: aie-opt --convert-aie-to-transaction=device-name=b --canonicalize %s | FileCheck %s --check-prefix=TXN

// Tile (0, 2): master South:1 and slave DMA:0 of @a's circuit connection, then
// shim (1, 0): master West32:0 and slave Control32:1 of @a's packet route.
// @a's control packet overlay connection (master NorthControl:0, slave
// SouthControl:1) and @c's connection (a different target) are left alone.
// B:      Address:  0x00000000006B401C Data:  0x00000000
// B-NEXT: Address:  0x00000000006B4104 Data:  0x00000000
// B-NEXT: Address:  0x0000000002008194 Data:  0x00000000
// B-NEXT: Address:  0x00000000020481B0 Data:  0x00000000
// B-NEXT: Address:  0x00000000006B4024 Data:  0x80000001
// B-NEXT: Address:  0x00000000006B4104 Data:  0x80000000
// B-NOT:  Address:

// @a disables @b's connection, master South:2 and slave DMA:0, before its own.
// A:      Address:  0x00000000006B4024 Data:  0x00000000
// A-NEXT: Address:  0x00000000006B4104 Data:  0x00000000
// A-NEXT: Address:  0x00000000006B401C Data:  0x80000001
// A-NOT:  Data:  0x00000000

// The transaction path writes the same registers in the same order.
// TXN-LABEL: aie.device(npu3) @b
// TXN:      aiex.npu.write32(%c7028764_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c7028996_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c33587604_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c33849776_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c7028772_i32, %c-2147483647_i32)
// TXN-NEXT: aiex.npu.write32(%c7028996_i32, %c-2147483648_i32)

// @c has no other device with its target.
// C:      Address:  0x00000000006B4018 Data:  0x80000011
// C-NEXT: Address:  0x00000000006B4144 Data:  0x80000000
// C-NOT:  Address:

module {
  aie.device(npu3) @a {
    %t02 = aie.tile(0, 2)
    %t10 = aie.tile(1, 0)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, South : 1>
      aie.connect<SouthControl : 1, NorthControl : 0> {is_ctrl_pkt_overlay}
    }
    %sb10 = aie.switchbox(%t10) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West32 : 0, %0)
      aie.packet_rules(Control32 : 1) {
        aie.rule(31, 1, %0)
      }
    }
  }
  aie.device(npu3) @b {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, South : 2>
    }
  }
  aie.device(npu3_1col) @c {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<North : 0, South : 0>
    }
  }
}
