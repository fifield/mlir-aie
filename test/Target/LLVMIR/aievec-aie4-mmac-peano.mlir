// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile the dense MMAC, staging and commit XLLVM ops with Peano and check
// the selected instructions. llc accepts unknown llvm.aie4.* names as
// external calls, so the checks also reject any residual call.
//
// REQUIRES: peano
//
// RUN: aie-translate %s -mlir-to-llvmir -o %t.ll
// RUN: llc -O2 --mtriple=aie4 --mcpu=aie4b -filetype=obj %t.ll -o %t.o
// RUN: llvm-objdump -dr %t.o | FileCheck %s --implicit-check-not=llvm.aie4 --implicit-check-not=jl

// Instructions within a function are matched in any order. The post-RA
// scheduler uses exposed-pipeline latencies (the commit reads za in a later
// stage than the insert writes it), so issue order need not follow IR order.

// bf16: the insert writes crIntMMode = 2, the commit is the plain vmov
// zb, za, and the MMAC reads zb with conf 0 (no zero_acc).
// CHECK-LABEL: <bf16>:
// CHECK-DAG: movx{{[[:blank:]]+}}crintmmode, #0x2{{([[:blank:]]|;|$)}}
// CHECK-DAG: vmov{{[[:blank:]]+}}za[i{{[0-9]+}}], y{{[0-9]+}}
// CHECK-DAG: vmov{{[[:blank:]]+}}zb, za{{([[:blank:]]|;|$)}}
// CHECK-DAG: mova{{[[:blank:]]+}}[[BFCONF:r[0-9]+]], #0x0{{([[:blank:]]|;|$)}}
// CHECK-DAG: mmac.f{{[[:blank:]]+}}dm{{[a-z]*[0-9]+}}, dm{{[a-z]*[0-9]+}}, x{{[0-9]+}}, zb, [[BFCONF]]{{([[:blank:]]|;|$)}}
llvm.func @bf16(%b: vector<64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %i = llvm.mlir.constant(5 : i32) : i32
  "xllvm.intr.aie4.insert.staging.bf16.I1024"(%b, %i) : (vector<64xbf16>, i32) -> ()
  "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
  %r = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%c, %a) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
  llvm.return %r : vector<64xf32>
}

// int8, signed x signed: crIntMMode = 3 and conf 0x308 (bmode bit 3 set).
// CHECK-LABEL: <i8>:
// CHECK-DAG: movx{{[[:blank:]]+}}crintmmode, #0x3{{([[:blank:]]|;|$)}}
// CHECK-DAG: vmov{{[[:blank:]]+}}za[i{{[0-9]+}}], y{{[0-9]+}}
// CHECK-DAG: vmov{{[[:blank:]]+}}zb, za{{([[:blank:]]|;|$)}}
// CHECK-DAG: mova{{[[:blank:]]+}}[[I8CONF:r[0-9]+]], #0x308{{([[:blank:]]|;|$)}}
// CHECK-DAG: mmac.i{{[[:blank:]]+}}dm{{[a-z]*[0-9]+}}, dm{{[a-z]*[0-9]+}}, x{{[0-9]+}}, zb, [[I8CONF]]{{([[:blank:]]|;|$)}}
llvm.func @i8(%b: vector<128xi8>, %a: vector<64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %i = llvm.mlir.constant(7 : i32) : i32
  "xllvm.intr.aie4.insert.staging.int.I1024"(%b, %i) {mode = 3 : i32} : (vector<128xi8>, i32) -> ()
  "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
  %r = "xllvm.intr.aie4.I512.ACC2048.mmac"(%c, %a) {bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  llvm.return %r : vector<64xi32>
}

// int4, signed x signed: crIntMMode = 1, the I4 commit, and conf 0x300
// (bmode = 0). The MMAC still names zb; the I4 commit fills it.
// CHECK-LABEL: <i4>:
// CHECK-DAG: movx{{[[:blank:]]+}}crintmmode, #0x1{{([[:blank:]]|;|$)}}
// CHECK-DAG: vmov{{[[:blank:]]+}}za[i{{[0-9]+}}], y{{[0-9]+}}
// CHECK-DAG: vmov{{[[:blank:]]+}}i4_zb, i4_za{{([[:blank:]]|;|$)}}
// CHECK-DAG: mova{{[[:blank:]]+}}[[I4CONF:r[0-9]+]], #0x300{{([[:blank:]]|;|$)}}
// CHECK-DAG: mmac.i{{[[:blank:]]+}}dm{{[a-z]*[0-9]+}}, dm{{[a-z]*[0-9]+}}, x{{[0-9]+}}, zb, [[I4CONF]]{{([[:blank:]]|;|$)}}
llvm.func @i4(%b: vector<128xi8>, %a: vector<64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %i = llvm.mlir.constant(9 : i32) : i32
  "xllvm.intr.aie4.insert.staging.int.I1024"(%b, %i) {mode = 1 : i32} : (vector<128xi8>, i32) -> ()
  "xllvm.intr.aie4.staging.to.matrix.I4.I32768"() : () -> ()
  %r = "xllvm.intr.aie4.I512.ACC2048.mmac"(%c, %a) {bmode = 0 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  llvm.return %r : vector<64xi32>
}

// fp8: no crIntMMode write, the f_za/f_zb register forms, and conf 0x1
// (zero_acc, emode = all).
// CHECK-LABEL: <fp8>:
// CHECK-NOT: crintmmode
// CHECK-DAG: vmov{{[[:blank:]]+}}f_za[i{{[0-9]+}}], y{{[0-9]+}}
// CHECK-DAG: vmov{{[[:blank:]]+}}f_zb, f_za{{([[:blank:]]|;|$)}}
// CHECK-DAG: mova{{[[:blank:]]+}}[[F8CONF:r[0-9]+]], #0x1{{([[:blank:]]|;|$)}}
// CHECK-DAG: mmac.f{{[[:blank:]]+}}dm{{[a-z]*[0-9]+}}, dm{{[a-z]*[0-9]+}}, x{{[0-9]+}}, f_zb, [[F8CONF]]{{([[:blank:]]|;|$)}}
// CHECK-NOT: crintmmode
llvm.func @fp8(%b: vector<128xi8>, %a: vector<64xi8>, %c: vector<64xf32>) -> vector<64xf32> {
  %i = llvm.mlir.constant(11 : i32) : i32
  "xllvm.intr.aie4.insert.staging.fp8.I1024"(%b, %i) : (vector<128xi8>, i32) -> ()
  "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
  %r = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%c, %a) {zeroAcc, emode = 0 : i32} : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
  llvm.return %r : vector<64xf32>
}
