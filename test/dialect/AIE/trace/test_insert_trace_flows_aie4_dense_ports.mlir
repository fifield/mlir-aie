// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-insert-trace-flows | FileCheck %s --check-prefix=INSERT
// RUN: aie-opt %s --aie-insert-trace-flows --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUTE
// RUN: aie-opt %s --aie-insert-trace-flows --aie-create-pathfinder-flows | aie-translate --aie-generate-xaie | FileCheck %s --check-prefix=XAIE
// RUN: aie-opt %s --aie-insert-trace-flows --aie-create-pathfinder-flows --convert-aie-to-transaction --canonicalize | FileCheck %s --check-prefix=TXN

// Both ordinary S2MM destinations are busy. A one-column device also rules
// out accidentally passing by redirecting trace to an unused shim.
// INSERT: aiex.npu.writebd {bd_id = 3 : i32,
// INSERT-SAME: channel = 0 : i32,
// INSERT-SAME: direction = 3 : i32,
// INSERT: aie.packet_source<%{{.*}}, Trace32 : 0>
// INSERT: aie.packet_dest<%{{.*}}, DMATrace : 0>

// ROUTE: aie.switchbox(%shim_noc_tile_0_0)
// ROUTE-DAG: aie.connect<North : {{[0-9]+}}, DMA : 0>
// ROUTE-DAG: aie.connect<North : {{[0-9]+}}, DMA : 1>
// ROUTE-DAG: aie.masterset(DMATrace : 0, {{.*}}) {keep_pkt_header = true}

// The generated API uses the dedicated hardware manager, preserving headers.
// XAIE: XAie_StrmPktSwMstrPortEnable({{.*}}, DMA_Trace, 0, /* drop_header */ XAIE_SS_PKT_DONOT_DROP_HEADER
// The transaction backend programs Stream_Switch_Manager_Config_DMA_Trace,
// offset 0x8004, with manager and packet enable set and Drop_Header clear.
// TXN-DAG: %[[PORT:.*]] = arith.constant 32772 : i32
// TXN-DAG: %[[CONFIG:.*]] = arith.constant -1073741816 : i32
// TXN: aiex.npu.write32(%[[PORT]], %[[CONFIG]])

module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.flow(%mem, DMA : 0, %shim, DMA : 0)
    aie.flow(%mem, DMA : 1, %shim, DMA : 1)
    aie.shim_dma_allocation @data0(%shim, S2MM, 0)
    aie.shim_dma_allocation @data1(%shim, S2MM, 1)
    aie.trace @trace(%core) {
      aie.trace.packet id = 1 type = core
      aie.trace.event <"INSTR_EVENT_0">
      aie.trace.start broadcast = 3
      aie.trace.stop broadcast = 4
    }
    aie.runtime_sequence(%out: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace
    }
  }
}
