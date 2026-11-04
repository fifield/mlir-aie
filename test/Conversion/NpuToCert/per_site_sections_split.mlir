//===- per_site_sections_split.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline="builtin.module(aie-npu-to-cert,aie.device(cert-legalize-pages,aie-cert-per-site-sections,aie-cert-assign-pdi-ids))" %s | FileCheck %s

// A -> work -> A, where @cfg's configuration (a 12 KB blockwrite) is too big
// for one page. Both sites get copies of one legalized template, so both
// copies have the same pages and the same job ids, and so the same PDI id.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi([[ID:[0-9]+]], @cfg__site0)
// CHECK: aiex.cert.write32(4096, 7)
// CHECK: aiex.cert.load_pdi([[ID]], @cfg__site1)
// CHECK: aiex.cert.section @cfg__site0 {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(8192, 1)
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@[[C0:chain_[0-9]+]])
// CHECK: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@[[C1:chain_[0-9]+]])
// CHECK: aiex.cert.section @cfg__site1 {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(8192, 1)
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@[[C0]])
// CHECK: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@[[C1]])

module {
  aie.device(npu3) @main {
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @cfg}
      %a = arith.constant 4096 : i32
      %v = arith.constant 7 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
  aie.device(npu3) @cfg {
    memref.global "private" constant @data : memref<3072xi32> = dense<5>
    aie.runtime_sequence @configure() {
      %a = arith.constant 8192 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      %0 = memref.get_global @data : memref<3072xi32>
      aiex.npu.blockwrite(%0) {address = 0x100000 : ui32} : memref<3072xi32>
    }
  }
}
