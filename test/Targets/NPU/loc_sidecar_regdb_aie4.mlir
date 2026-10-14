// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-translate --aie-npu-to-binary --aie-npu-emit-locmap=loc_sidecar_regdb_aie4.json %s -o loc_sidecar_regdb_aie4.bin
// RUN: FileCheck %s < loc_sidecar_regdb_aie4.json

// AIE4 has distinct core and memory registers sharing names. Decode the
// physical address using the AIE4 database, including nonzero columns,
// compute row expansion, and both memory and core modules.
module {
  aie.device(npu3) {
    aie.runtime_sequence() {
      %value = arith.constant 1 : i32
      %addr0 = arith.constant 0x26f0030 : i32
      aiex.npu.write32(%addr0, %value) : i32, i32
      %addr1 = arith.constant 0x26e0080 : i32
      aiex.npu.write32(%addr1, %value) : i32, i32
      %addr2 = arith.constant 0x26f0010 : i32
      aiex.npu.write32(%addr2, %value) : i32, i32
      %addr3 = arith.constant 0x26f0014 : i32
      aiex.npu.write32(%addr3, %value) : i32, i32
      %addr4 = arith.constant 0x26a7120 : i32
      aiex.npu.write32(%addr4, %value) : i32, i32
      %addr5 = arith.constant 0x26a00f0 : i32
      aiex.npu.write32(%addr5, %value) : i32, i32
      %addr6 = arith.constant 0x2109e60 : i32
      aiex.npu.write32(%addr6, %value) : i32, i32
    }
  }
}

// CHECK: "address": "0x26F0030"
// CHECK: "register": "Memory_Control"
// CHECK-NEXT: "register_module": "memory"

// CHECK: "address": "0x26E0080"
// CHECK: "register": "Memory_Control"
// CHECK-NEXT: "register_module": "core"

// CHECK: "address": "0x26F0010"
// CHECK: "register": "ECC_Scrubbing_Period"
// CHECK-NEXT: "register_module": "memory"

// CHECK: "address": "0x26F0014"
// CHECK: "register": "ECC_Failing_Address"
// CHECK-NEXT: "register_module": "memory"

// CHECK: "address": "0x26A7120"
// CHECK: "register": "Spare_Reg"
// CHECK-NEXT: "register_module": "memory"

// CHECK: "address": "0x26A00F0"
// CHECK: "register": "Trace1_Control0"
// CHECK-NEXT: "register_module": "core"

// CHECK: "address": "0x2109E60"
// CHECK: "register": "DMA_MM2S_0_Ctrl"
// CHECK-NEXT: "register_module": "memory_tile"
