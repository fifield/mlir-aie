//===- cert_absorbed_device_globals.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// --aie-npu-to-cert absorbs every referenced device into the main device and
// then erases the source. The cert ops it clones address their data through
// memref.global symbols, which are per-device and which the clone does not
// carry: once the source device is gone, a uc_dma_bd whose remote_address is
// @config_blockwrite_data_0 resolves that symbol against its new parent and
// fails downstream with "Global symbol not found".
//
// The globals must therefore be hoisted into the main device before absorption
// -- and renamed while hoisting, because every device numbers its blockwrite
// data from zero, so the names collide by construction.

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s
// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s --check-prefix=CLASH

// Every global ends up in the one surviving device, and the colliding pair does
// not collapse into one: three distinct symbols survive for the three inputs,
// carrying three distinct initialisers.
//
// CHECK: aie.device(npu3) {
// CHECK-DAG: memref.global "private" constant @config_blockwrite_data_0 : memref<2xi32> = dense<[1, 2]>
// CHECK-DAG: memref.global "private" constant @config_blockwrite_data_0_{{[0-9]+}} : memref<2xi32> = dense<[3, 4]>
// CHECK-DAG: memref.global "private" constant @config_blockwrite_data_1 : memref<2xi32> = dense<[5, 6]>
//
// A THIRD device collides with the same name again, so the rename counter has
// to keep going rather than reusing the first free suffix. Four inputs named
// @config_blockwrite_data_0 or _1 must leave four distinct symbols, each with
// its own initialiser -- if any two collapsed, one device's config data would
// silently become another's.
// CHECK-DAG: memref.global "private" constant @config_blockwrite_data_0_{{[0-9]+}} : memref<2xi32> = dense<[7, 8]>

// Both referenced devices were absorbed into sections and erased, so no global
// can have been left behind in one.
// CHECK-DAG: aiex.cert.section @seg_a
// CHECK-DAG: aiex.cert.section @seg_b
// CHECK-DAG: aiex.cert.section @seg_c
// CHECK-DAG: aiex.cert.section @seg_no_globals
// CHECK-NOT: aie.device(npu3) @seg_a
// CHECK-NOT: aie.device(npu3) @seg_b
// CHECK-NOT: aie.device(npu3) @seg_c
// CHECK-NOT: aie.device(npu3) @seg_no_globals

module {
  aie.device(npu3) @main {
    memref.global "private" constant @config_blockwrite_data_0 : memref<2xi32> = dense<[1, 2]>
    memref.global "private" constant @clash : memref<2xi32> = dense<[9, 10]>
    aie.runtime_sequence @main_seq() {
      aiex.npu.load_pdi {device_ref = @seg_a}
      aiex.npu.load_pdi {device_ref = @seg_b}
      aiex.npu.load_pdi {device_ref = @seg_c}
      aiex.npu.load_pdi {device_ref = @seg_no_globals}
      aiex.npu.load_pdi {device_ref = @seg_clash}
    }
  }

  // @seg_a's global collides with the main device's own @config_blockwrite_data_0
  // and must be renamed; @seg_b's does not and must keep its name. Renaming
  // unconditionally, or not at all, fails one of those two.
  aie.device(npu3) @seg_a {
    memref.global "private" constant @config_blockwrite_data_0 : memref<2xi32> = dense<[3, 4]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }

  aie.device(npu3) @seg_b {
    memref.global "private" constant @config_blockwrite_data_1 : memref<2xi32> = dense<[5, 6]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x2000 : i32
      %v = arith.constant 2 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }

  // Third collision on the same name: the rename counter must advance past the
  // suffix already taken by @seg_a's copy.
  aie.device(npu3) @seg_c {
    memref.global "private" constant @config_blockwrite_data_0 : memref<2xi32> = dense<[7, 8]>
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x3000 : i32
      %v = arith.constant 3 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }

  // A referenced device with no globals at all: nothing to hoist, and the
  // device must still be absorbed into a section and erased.
  aie.device(npu3) @seg_no_globals {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x4000 : i32
      %v = arith.constant 4 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }

  // A rename must be free in the SOURCE device too, not just in main. Here the
  // source holds both @clash (colliding with main's) and @clash_0 -- the very
  // name a main-only check would pick for @clash. Getting this wrong leaves
  // two symbols of that name in the source, and the later rename of the
  // original @clash_0 then walks BOTH sets of uses onto one global: the build
  // still assembles, it just addresses the wrong payload.
  //
  // The references are what make it observable; checking the definitions alone
  // does not, since three distinct globals survive either way.
  // CLASH-DAG: memref.global "private" constant @[[A:clash_[0-9]+]] : memref<2xi32> = dense<[11, 12]>
  // CLASH-DAG: memref.global "private" constant @[[B:clash_[0-9]+]] : memref<2xi32> = dense<[13, 14]>
  // CLASH-DAG: uc_dma_bd @[[A]], 4096
  // CLASH-DAG: uc_dma_bd @[[B]], 8192
  aie.device(npu3) @seg_clash {
    memref.global "private" constant @clash : memref<2xi32> = dense<[11, 12]>
    memref.global "private" constant @clash_0 : memref<2xi32> = dense<[13, 14]>
    aie.runtime_sequence @configure() {
      %a = memref.get_global @clash : memref<2xi32>
      aiex.npu.blockwrite(%a) {address = 4096 : ui32} : memref<2xi32>
      %b = memref.get_global @clash_0 : memref<2xi32>
      aiex.npu.blockwrite(%b) {address = 8192 : ui32} : memref<2xi32>
    }
  }
}
