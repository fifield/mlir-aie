//===- aie4_configure_switch_clear.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Phase S of the AIE4 configure-step reset: between the core reset readback
// (QB) and the DMA channel resets (Q1), every stream switch register of every
// tile in the partition is written to 0, one uC-DMA block set per run of
// consecutive registers. The registers are both switches' read-write
// Stream_Switch_* registers, without parity injection/status and event port
// selection; runs never cover an address-map hole. Packet slots are cleared
// only on the tiles where the configuration that follows has packet rules
// (c3): S takes every port out of packet mode, so elsewhere a stale slot is
// inert.

// RUN: rm -rf aie4_configure_switch_clear_prj && split-file %s aie4_configure_switch_clear_prj
// RUN: mkdir -p aie4_configure_switch_clear_prj/v1_c1
// RUN: aie-opt --convert-aie-to-transaction --aie-npu-to-cert --cert-legalize-pages aie4_configure_switch_clear_prj/c1.mlir | aie-translate --aie-cert-to-asm --cert-section-dir=aie4_configure_switch_clear_prj/v1_c1 -o aie4_configure_switch_clear_prj/v1_c1.asm
// RUN: sed -e '/^\.include main\.asm$/r aie4_configure_switch_clear_prj/v1_c1/main.asm' -e '/^\.include main\.asm$/d' aie4_configure_switch_clear_prj/v1_c1.asm > aie4_configure_switch_clear_prj/v1_c1.inl.asm
// RUN: mkdir -p aie4_configure_switch_clear_prj/v1_c2
// RUN: aie-opt --convert-aie-to-transaction --aie-npu-to-cert --cert-legalize-pages aie4_configure_switch_clear_prj/c2.mlir | aie-translate --aie-cert-to-asm --cert-section-dir=aie4_configure_switch_clear_prj/v1_c2 -o aie4_configure_switch_clear_prj/v1_c2.asm
// RUN: sed -e '/^\.include main\.asm$/r aie4_configure_switch_clear_prj/v1_c2/main.asm' -e '/^\.include main\.asm$/d' aie4_configure_switch_clear_prj/v1_c2.asm > aie4_configure_switch_clear_prj/v1_c2.inl.asm
// RUN: cat aie4_configure_switch_clear_prj/v1_c1.inl.asm aie4_configure_switch_clear_prj/v1_c2.inl.asm | FileCheck %s --implicit-check-not=0x006b4f0 --implicit-check-not=0x006b4f10 --implicit-check-not=0x007b4f0 --implicit-check-not=0x007b4f10 --implicit-check-not=0x006e002 --implicit-check-not=0x006a051c --implicit-check-not=0x0012025 --implicit-check-not=0x00120260 --implicit-check-not=0x0000825 --implicit-check-not=0x00008260 --implicit-check-not=0x001e00 --implicit-check-not=0x000e005 --implicit-check-not=0x000e0074 --implicit-check-not=0x006b4010 --implicit-check-not=0x006b4020 --implicit-check-not=0x006b414c --implicit-check-not=0x00120010 --implicit-check-not=0x00008008
// RUN: mkdir -p aie4_configure_switch_clear_prj/v1_c3
// RUN: aie-opt --aie-create-pathfinder-flows --convert-aie-to-transaction --aie-npu-to-cert --cert-legalize-pages aie4_configure_switch_clear_prj/c3.mlir | aie-translate --aie-cert-to-asm --cert-section-dir=aie4_configure_switch_clear_prj/v1_c3 -o aie4_configure_switch_clear_prj/v1_c3.asm
// RUN: sed -e '/^\.include main\.asm$/r aie4_configure_switch_clear_prj/v1_c3/main.asm' -e '/^\.include main\.asm$/d' aie4_configure_switch_clear_prj/v1_c3.asm > aie4_configure_switch_clear_prj/v1_c3.inl.asm
// RUN: FileCheck %s --check-prefix=PKT --input-file=aie4_configure_switch_clear_prj/v1_c3.inl.asm
// Control packets never get the reset, so never the clear.
// RUN: aie-opt --split-input-file --convert-aie-to-control-packets %s | FileCheck %s --check-prefix=CTRL

// npu3_1col. The chain sits between the last core's readback and the first
// Q1 write. The design's routes, control-packet overlay included, are
// installed again afterwards.
// CHECK-LABEL: .partition 1column
// CHECK:      START_JOB 1
// CHECK:      MASK_POLL_32 {{ *}}0x009a3000, 0x00000003, 0x00000002
// CHECK-NEXT: uC_DMA_WRITE_DES_SYNC {{ *}}@[[S:chain_[0-9]+]]
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 {{ *}}0x00109e00, 0x00000002, 0x00000002
// CHECK-NOT:  uC_DMA_WRITE_DES_SYNC
// CHECK:      {{^ *}}MASK_WRITE_32 {{ *}}0x009a3000, 0x00000002, 0x00000000
// CHECK-NOT:  uC_DMA_WRITE_DES_SYNC
// Tile (0,2): DMA:0 -> South:2, then the overlay's SouthControl:1 ->
// NorthControl:0.
// CHECK:      {{^ *}}WRITE_32 {{ *}}0x006b4024, 0x80000001
// CHECK-NEXT: {{^ *}}WRITE_32 {{ *}}0x006b4104, 0x80000000
// CHECK-NEXT: {{^ *}}WRITE_32 {{ *}}0x006b4070, 0x8000001c
// CHECK-NEXT: {{^ *}}WRITE_32 {{ *}}0x006b4170, 0x80000000
// CHECK-NEXT: END_JOB
// The chain: column 0, tiles in row order, each run's start and word count.
// No tile of this design has packet rules, so no packet slot is cleared;
// holes in the address map and the skipped slot registers split the runs.
// Nothing in column 1 or beyond.
// CHECK:      [[S]]:
// Shim (0,0): 21 runs, 63 words.
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00008000, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00008018, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x0000802c, @reset_{{[0-9]+}}, 7, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x0000805c, @reset_{{[0-9]+}}, 8, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00008170, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00008190, @reset_{{[0-9]+}}, 5, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x000081b0, @reset_{{[0-9]+}}, 7, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048000, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048020, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048034, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x0004803c, @reset_{{[0-9]+}}, 3, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048064, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048074, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048170, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048190, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00048198, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x000481a0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x000481b0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x000481b8, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x000481c0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x000481c8, @reset_{{[0-9]+}}, 1, 0, 1
// Memory tile (0,1) at 0x100000: 30 runs, 74 words.
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00120000, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00120014, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00120020, @reset_{{[0-9]+}}, 7, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00120040, @reset_{{[0-9]+}}, 5, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x0012005c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x0012006c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00120078, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00120180, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001201a0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001201a8, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001201b0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001201c0, @reset_{{[0-9]+}}, 3, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001201d0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001201d8, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00160000, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00160014, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00160020, @reset_{{[0-9]+}}, 5, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00160038, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00160040, @reset_{{[0-9]+}}, 5, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x0016005c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x0016006c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00160078, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00160180, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001601a0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001601a8, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001601b0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001601c0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001601c8, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001601d0, @reset_{{[0-9]+}}, 1, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x001601d8, @reset_{{[0-9]+}}, 1, 0, 1
// Core (0,2) at 0x600000: 15 runs, 68 words.
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4000, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4018, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4024, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4030, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4040, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4070, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4100, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4114, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4144, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4150, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b415c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b416c, @reset_{{[0-9]+}}, 3, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4800, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4900, @reset_{{[0-9]+}}, 7, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x006b4a00, @reset_{{[0-9]+}}, 10, 0, 1
// Core (0,3) at 0x700000: 15 runs, 68 words.
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4000, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4018, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4024, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4030, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4040, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4070, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4100, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4114, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4144, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4150, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b415c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b416c, @reset_{{[0-9]+}}, 3, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4800, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4900, @reset_{{[0-9]+}}, 7, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x007b4a00, @reset_{{[0-9]+}}, 10, 0, 1
// Core (0,4) at 0x800000: 15 runs, 68 words.
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4000, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4018, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4024, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4030, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4040, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4070, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4100, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4114, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4144, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4150, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b415c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b416c, @reset_{{[0-9]+}}, 3, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4800, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4900, @reset_{{[0-9]+}}, 7, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x008b4a00, @reset_{{[0-9]+}}, 10, 0, 1
// Core (0,5) at 0x900000: 15 runs, 68 words.
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4000, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4018, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4024, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4030, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4040, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4070, @reset_{{[0-9]+}}, 4, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4100, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4114, @reset_{{[0-9]+}}, 10, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4144, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4150, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b415c, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b416c, @reset_{{[0-9]+}}, 3, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4800, @reset_{{[0-9]+}}, 6, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4900, @reset_{{[0-9]+}}, 7, 0, 1
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x009b4a00, @reset_{{[0-9]+}}, 10, 0, 0
// CHECK-NOT:  UC_DMA_BD
// Every payload is zero: 409 words over the 9 distinct run lengths.
// CHECK-NOT:  .long {{ *}}0x{{0*[1-9a-f]}}

// CTRL-LABEL: aie.device(npu3_1col)
// CTRL-NOT:   aiex.control_packet
// CTRL:       aiex.control_packet {address = 7028772 : ui32, data = array<i32: -2147483647>
// CTRL-NEXT:  aiex.control_packet {address = 7028996 : ui32
// CTRL-NEXT:  aiex.control_packet {address = 7028848 : ui32
// CTRL-NEXT:  aiex.control_packet {address = 7029104 : ui32
// CTRL-NOT:   aiex.control_packet
//--- c1.mlir
module {
  aie.device(npu3_1col) @main {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, South : 2>
      aie.connect<SouthControl : 1, NorthControl : 0> {is_ctrl_pkt_overlay}
    }
  }
}

// -----
//--- c2.mlir

// npu3_2col: both columns are cleared although the device declares only a
// column 1 tile. Only chains and job boundaries come between the last readback
// and Q1 (page legalization may split the clear over several jobs). Together
// the chains clear both columns, from shim (0, 0) to core (1, 5).
// CHECK-LABEL: .partition 2column
// CHECK:      MASK_POLL_32 {{ *}}0x029a3000, 0x00000003, 0x00000002
// CHECK-NEXT: uC_DMA_WRITE_DES_SYNC {{ *}}@[[S1:chain_[0-9]+]]
// CHECK-NOT:  {{WRITE_32|POLL_32}}
// CHECK:      {{^ *}}MASK_WRITE_32 {{ *}}0x00109e00, 0x00000002, 0x00000002
// CHECK:      END_JOB
// CHECK:      [[S1]]:
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00008000, @reset_{{[0-9]+}}, 2, 0, 1
// CHECK:      UC_DMA_BD {{ *}}0, 0x02008000, @reset_{{[0-9]+}}, 2, 0, {{[01]}}
// CHECK:      UC_DMA_BD {{ *}}0, 0x02120000, @reset_{{[0-9]+}}, 4, 0, {{[01]}}
// CHECK:      UC_DMA_BD {{ *}}0, 0x026b4000, @reset_{{[0-9]+}}, 4, 0, {{[01]}}
// CHECK:      UC_DMA_BD {{ *}}0, 0x029b4a00, @reset_{{[0-9]+}}, 10, 0, 0
// CHECK-NOT:  UC_DMA_BD
// CTRL-LABEL: aie.device(npu3_2col)
// CTRL-NOT:   aiex.control_packet
module {
  aie.device(npu3_2col) @main {
    %t10 = aie.tile(1, 0)
  }
}

// -----
//--- c3.mlir

// A packet flow from shim (0, 0) to core (0, 2): the configuration has packet
// rules on the tiles of its route, so the clear covers their packet slots too,
// and no other tile's.
// PKT-LABEL: .partition 1column
// PKT:      MASK_POLL_32 {{ *}}0x009a3000, 0x00000003, 0x00000002
// PKT-NEXT: uC_DMA_WRITE_DES_SYNC {{ *}}@[[S:chain_[0-9]+]]
// PKT:      [[S]]:
// Shim (0,0), both switches' slots.
// PKT:      UC_DMA_BD {{ *}}0, 0x00008080, @reset_{{[0-9]+}}, 8, 0, 1
// PKT:      UC_DMA_BD {{ *}}0, 0x00048080, @reset_{{[0-9]+}}, 8, 0, 1
// Memtile (0,1).
// PKT:      UC_DMA_BD {{ *}}0, 0x00120078, @reset_{{[0-9]+}}, 22, 0, 1
// PKT:      UC_DMA_BD {{ *}}0, 0x00160078, @reset_{{[0-9]+}}, 22, 0, 1
// Core (0,2).
// PKT:      UC_DMA_BD {{ *}}0, 0x006b4b00, @reset_{{[0-9]+}}, 40, 0, 1
// Cores (0,3) and up are off the route: no slots.
// PKT-NOT:  0x00{{[7-9]}}b4b00
// PKT:      UC_DMA_BD {{.*}}, 0, 0{{$}}
module {
  aie.device(npu3_1col) @main {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(3) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    }
  }
}
