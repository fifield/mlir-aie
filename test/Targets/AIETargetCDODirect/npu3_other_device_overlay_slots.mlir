// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An installed overlay and application share North:0. Retiring the previous
// application clears slots 1 and 3, retaining overlay slots 0 and 2 and the
// enabled slave. North:1 carries overlay rules only; North:2 is an ordinary
// application port and must still be disabled in full.
// RUN: aie-translate --aie-generate-cdo --cdo-debug=true --aie-device-name=b %s 2>&1 | FileCheck %s --check-prefix=CDO \
// RUN:   --implicit-check-not=0x00000000006B4144 --implicit-check-not=0x00000000006B4310 --implicit-check-not=0x00000000006B4318 \
// RUN:   --implicit-check-not=0x00000000006B4148 --implicit-check-not=0x00000000006B4320 --implicit-check-not=0x00000000006B4018
// RUN: aie-opt --convert-aie-to-transaction=device-name=b --canonicalize %s | FileCheck %s --check-prefix=TXN
// The pre-rebase spelling must retain the same slot-level protection.
// RUN: sed 's/{aie.is_ctrl_pkt_overlay}/{is_ctrl_pkt_overlay}/g' %s > %t.legacy.mlir
// RUN: aie-translate --aie-generate-cdo --cdo-debug=true --aie-device-name=b %t.legacy.mlir 2>&1 | FileCheck %s --check-prefix=CDO \
// RUN:   --implicit-check-not=0x00000000006B4144 --implicit-check-not=0x00000000006B4310 --implicit-check-not=0x00000000006B4318 \
// RUN:   --implicit-check-not=0x00000000006B4148 --implicit-check-not=0x00000000006B4320 --implicit-check-not=0x00000000006B4018
// RUN: aie-opt --convert-aie-to-transaction=device-name=b --canonicalize %t.legacy.mlir | FileCheck %s --check-prefix=TXN

// Disable @a's ordinary master, its two application slots, and its ordinary
// slave. Then install @b's route before its DMA is started. The interleaved
// overlay rules must not compress application slot indices during cleanup.
// CDO:      Address:  0x00000000006B401C Data:  0x00000000
// CDO-NEXT: Address:  0x00000000006B4314 Data:  0x00000000
// CDO-NEXT: Address:  0x00000000006B431C Data:  0x00000000
// CDO-NEXT: Address:  0x00000000006B4150 Data:  0x00000000
// CDO-NEXT: Address:  0x00000000006B4024 Data:  0x80000001
// CDO-NEXT: Address:  0x00000000006B4104 Data:  0x80000000
// CDO:      Address:  0x00000000006ACE14 Data:  0x00000000

// Transaction lowering must preserve the same retirement/configuration/start
// order. The zero-valued writes to slots 1 and 3 remain actual register writes.
// TXN-LABEL: aie.device(npu3) @b
// TXN:      aiex.npu.write32(%c7028764_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c7029524_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c7029532_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c7029072_i32, %c0_i32)
// TXN-NEXT: aiex.npu.write32(%c7028772_i32, %c-2147483647_i32)
// TXN-NEXT: aiex.npu.write32(%c7028996_i32, %c-2147483648_i32)
// TXN:      aiex.npu.write32(%c6999572_i32, %c0_i32)

module {
  aie.device(npu3) @a {
    %tile = aie.tile(0, 2)
    %sw = aie.switchbox(%tile) {
      %overlay = aie.amsel<0> (0)
      %application = aie.amsel<1> (0)
      aie.masterset(South : 0, %overlay) {aie.is_ctrl_pkt_overlay}
      aie.masterset(South : 1, %application)
      aie.packet_rules(North : 0) {
        aie.rule(31, 1, %overlay) {aie.is_ctrl_pkt_overlay}
        aie.rule(31, 2, %application)
        aie.rule(31, 3, %overlay) {aie.is_ctrl_pkt_overlay}
        aie.rule(31, 4, %application)
      }
      aie.packet_rules(North : 1) {
        aie.rule(31, 5, %overlay) {aie.is_ctrl_pkt_overlay}
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 6, %application)
      }
    }
  }
  aie.device(npu3) @b {
    %tile = aie.tile(0, 2)
    %buffer = aie.buffer(%tile) {address = 1024 : i32} : memref<16xi32>
    %sw = aie.switchbox(%tile) {
      aie.connect<DMA : 0, South : 2>
    }
    %dma = aie.mem(%tile) {
      %start = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
