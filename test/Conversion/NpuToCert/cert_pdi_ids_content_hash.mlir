//===- cert_pdi_ids_content_hash.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie-cert-assign-pdi-ids derives each cert.load_pdi id from the content of
// the section it loads. The CERT firmware skips a LOAD_PDI whose id matches
// the last one it loaded -- across every ELF on a hardware context -- so equal
// ids have to mean the same configuration, independent of how the compilation
// named or numbered its sections and helper symbols (chains, data globals).

// RUN: rm -rf cert_pdi_ids_content_hash && mkdir -p cert_pdi_ids_content_hash
// RUN: aie-opt --split-input-file --pass-pipeline="builtin.module(aie-npu-to-cert,aie.device(cert-legalize-pages,aie-cert-assign-pdi-ids))" %s -o cert_pdi_ids_content_hash/1.mlir
// RUN: FileCheck %s --input-file=cert_pdi_ids_content_hash/1.mlir

// Determinism: the same input gives byte-identical ids.
// RUN: aie-opt --split-input-file --pass-pipeline="builtin.module(aie-npu-to-cert,aie.device(cert-legalize-pages,aie-cert-assign-pdi-ids))" %s -o cert_pdi_ids_content_hash/2.mlir
// RUN: cmp cert_pdi_ids_content_hash/1.mlir cert_pdi_ids_content_hash/2.mlir

// (a) Different configurations get different ids. (b) The same writes under
// another device symbol, loaded third, get job id 3 instead of 1: job ids are
// content, so a different id too ((e) shows the name itself is not hashed).
// aie-npu-to-cert alone would number these 1, 2, 3.
// (CHECK-NOT between CHECK and CHECK-SAME rejects [[IDA]] on that very line.)
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi([[IDA:[0-9]+]], @cfg_a)
// CHECK: aiex.cert.load_pdi
// CHECK-NOT: ([[IDA]],
// CHECK-SAME: @cfg_b)
// CHECK: aiex.cert.load_pdi
// CHECK-NOT: ([[IDA]],
// CHECK-SAME: @cfg_a_copy)
// CHECK-DAG: aiex.cert.section @cfg_a {
// CHECK-DAG: aiex.cert.section @cfg_b {
// CHECK-DAG: aiex.cert.section @cfg_a_copy {

module {
  aie.device(npu3) @main {
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @cfg_a}
      aiex.npu.load_pdi {device_ref = @cfg_b}
      aiex.npu.load_pdi {device_ref = @cfg_a_copy}
    }
  }
  aie.device(npu3) @cfg_a {
    memref.global "private" constant @data : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      %0 = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%0) {address = 0x2000 : ui32} : memref<4xi32>
    }
  }
  aie.device(npu3) @cfg_b {
    memref.global "private" constant @data : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 2 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      %0 = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%0) {address = 0x2000 : ui32} : memref<4xi32>
    }
  }
  aie.device(npu3) @cfg_a_copy {
    memref.global "private" constant @data : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      %0 = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%0) {address = 0x2000 : ui32} : memref<4xi32>
    }
  }
}

// -----

// (c) Another "compilation" of @cfg_a: same content, but its
// payload global is named differently and @main does a blockwrite of its own
// first, which shifts the chain numbering (the section's raw job id, 1, is the
// same as in the first module). The id is still [[IDA]].
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi([[IDA]], @cfg_a)

module {
  aie.device(npu3) @main {
    memref.global "private" constant @main_data : memref<2xi32> = dense<[9, 9]>
    aie.runtime_sequence @seq() {
      %0 = memref.get_global @main_data : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 0x3000 : ui32} : memref<2xi32>
      aiex.npu.load_pdi {device_ref = @cfg_a}
    }
  }
  aie.device(npu3) @cfg_a {
    memref.global "private" constant @payload : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      %0 = memref.get_global @payload : memref<4xi32>
      aiex.npu.blockwrite(%0) {address = 0x2000 : ui32} : memref<4xi32>
    }
  }
}

// -----

// (d) @cfg_a with one blockwrite data word changed: a different id.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi
// CHECK-NOT: ([[IDA]],
// CHECK-SAME: @cfg_a)

module {
  aie.device(npu3) @main {
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @cfg_a}
    }
  }
  aie.device(npu3) @cfg_a {
    memref.global "private" constant @data : memref<4xi32> = dense<[1, 2, 3, 5]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      %0 = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%0) {address = 0x2000 : ui32} : memref<4xi32>
    }
  }
}

// -----

// (e) The same content as @cfg_a in the first module -- same job id, chain and
// data -- from a device with another name: the same id. Only the name
// differs, and it is not hashed.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi([[IDA]], @cfg_renamed)

module {
  aie.device(npu3) @main {
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @cfg_renamed}
    }
  }
  aie.device(npu3) @cfg_renamed {
    memref.global "private" constant @data : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      %0 = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%0) {address = 0x2000 : ui32} : memref<4xi32>
    }
  }
}
