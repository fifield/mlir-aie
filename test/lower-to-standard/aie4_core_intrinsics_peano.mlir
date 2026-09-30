//===- aie4_core_intrinsics_peano.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check AIE4 core intrinsic signatures and instruction selection with Peano.
// Calls are required to verify signatures; declarations alone are insufficient.
// set.ctrl.reg is omitted because this lowering only calls it on AIE2/AIE2P.
//
// REQUIRES: peano
// RUN: aie-opt --aie-localize-locks --aie-normalize-address-spaces \
// RUN:   --aie-standard-lowering --aiex-standard-lowering \
// RUN:   --convert-aievec-to-llvm="aie-target=aie4" --canonicalize \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm="use-bare-ptr-memref-call-conv" \
// RUN:   --convert-to-llvm=dynamic=true --convert-vector-to-llvm \
// RUN:   --reconcile-unrealized-casts %s -o %t.llvm.mlir
// RUN: aie-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: %PEANO_INSTALL_DIR/bin/opt -S -mtriple=aie4 %t.ll -o %t.opt.ll
// RUN: FileCheck %s --check-prefix=IR < %t.opt.ll
// RUN: %PEANO_INSTALL_DIR/bin/llc -O2 -mtriple=aie4 -filetype=obj %t.opt.ll -o %t.o
// RUN: %PEANO_INSTALL_DIR/bin/llvm-objdump -d --no-show-raw-insn %t.o \
// RUN:   | FileCheck %s --check-prefix=ASM

// IR-DAG: call void @llvm.aie4.acquire(i32 {{[0-9]+}}, i32 -1)
// IR-DAG: call void @llvm.aie4.event(i32 0)
// IR-DAG: call <32 x i32> @llvm.aie4.get.ss(i32 0)
// IR-DAG: call <16 x i32> @llvm.aie4.getl.wss(i32 1)
// IR-DAG: call void @llvm.aie4.put.ms(i32 %{{.*}}, i32 0)
// IR-DAG: call void @llvm.aie4.put.ms.v8(<8 x i32> %{{.*}}, i32 0)
// IR-DAG: call void @llvm.aie4.put.ms.v16(<16 x i32> %{{.*}}, i32 0)
// IR-DAG: call void @llvm.aie4.mcd.write.vec(<16 x i32> %{{.*}}, i32 1)
// IR-DAG: call <16 x i32> @llvm.aie4.scd.read.vec(i32 1)
// IR-DAG: call void @llvm.aie4.event(i32 1)
// IR-DAG: call void @llvm.aie4.release(i32 {{[0-9]+}}, i32 1)

// ASM-LABEL: <core_0_2>:
// ASM-DAG: acq
// ASM-DAG: event #0
// ASM-DAG: vmov{{[abx]?}} y{{[0-9]+}}, ss[p{{[0-9]+}}]
// ASM-DAG: vmov{{[abx]?}} x{{[0-9]+}}, ss[p{{[0-9]+}}]
// ASM-DAG: mov ms, r{{[0-9]+}}
// ASM-DAG: vmov ms, w{{[lh][0-9]+}}
// ASM-DAG: vmov ms, x{{[0-9]+}}
// ASM-DAG: vmov mcd, x{{[0-9]+}}
// ASM-DAG: vmov x{{[0-9]+}}, scd
// ASM-DAG: event #1
// ASM-DAG: rel
module {
  aie.device(npu3) {
    %t02 = aie.tile(0, 2)
    %l0 = aie.lock(%t02, 0) {init = 1 : i32}
    %l1 = aie.lock(%t02, 1) {init = 0 : i32}
    %in = aie.buffer(%t02) {sym_name = "in"} : memref<16xi32>
    %out1k = aie.buffer(%t02) {sym_name = "out1k"} : memref<32xi32>
    %out512 = aie.buffer(%t02) {sym_name = "out512"} : memref<16xi32>
    %outcas = aie.buffer(%t02) {sym_name = "outcas"} : memref<16xi32>
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : i32
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      aie.use_lock(%l0, AcquireGreaterEqual, %c1)
      aie.event(0)
      %w = aie.get_stream(%s0 : i32) : vector<32xi32>
      vector.store %w, %out1k[%c0] : memref<32xi32>, vector<32xi32>
      %n = aie.get_stream(%s1 : i32) : vector<16xi32>
      vector.store %n, %out512[%c0] : memref<16xi32>, vector<16xi32>
      %x = memref.load %in[%c0] : memref<16xi32>
      aie.put_stream(%s0 : i32, %x : i32)
      %v8 = vector.load %in[%c0] : memref<16xi32>, vector<8xi32>
      aie.put_stream(%s0 : i32, %v8 : vector<8xi32>)
      %v = vector.load %in[%c0] : memref<16xi32>, vector<16xi32>
      aie.put_stream(%s0 : i32, %v : vector<16xi32>)
      aie.put_cascade(%v : vector<16xi32>)
      %g = aie.get_cascade() : vector<16xi32>
      vector.store %g, %outcas[%c0] : memref<16xi32>, vector<16xi32>
      aie.event(1)
      aie.use_lock(%l1, Release, %c1)
      aie.end
    }
  }
}
