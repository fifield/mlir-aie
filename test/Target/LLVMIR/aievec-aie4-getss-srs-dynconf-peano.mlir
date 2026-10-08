// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Check Peano instruction selection and reject residual intrinsic calls.
//
// REQUIRES: peano
//
// RUN: aie-translate %s -mlir-to-llvmir -o %t.ll
// RUN: llc -O2 --mtriple=aie4 --mcpu=aie4b -filetype=obj %t.ll -o %t.o
// RUN: llvm-objdump -dr %t.o | FileCheck %s --implicit-check-not=llvm.aie4 --implicit-check-not=jl

// Forward the stream port and select a native 1024-bit stream read.
// CHECK-LABEL: <get_ss>:
// CHECK-DAG: mova{{[[:blank:]]+}}[[P:p[0-9]+]], #0x1{{([[:blank:]]|;|$)}}
// CHECK-DAG: vmovb{{[[:blank:]]+}}y{{[0-9]+}}, ss[[[P]]]{{([[:blank:]]|;|$)}}
llvm.func @get_ss(%out: !llvm.ptr) {
  %p = llvm.mlir.constant(1 : i32) : i32
  %0 = "xllvm.intr.aie4.get.ss"(%p) : (i32) -> vector<32xi32>
  llvm.store %0, %out : vector<32xi32>, !llvm.ptr
  llvm.return
}

// Check the maximum shift, signed form and crSRSMode initialization.
// CHECK-LABEL: <srs_signed_59>:
// CHECK-DAG: movx{{[[:blank:]]+}}crsrsmode, #0x0{{([[:blank:]]|;|$)}}
// CHECK-DAG: mov{{[[:blank:]]+}}[[S59:s[0-9]+]], #0x3b{{([[:blank:]]|;|$)}}
// CHECK-DAG: vsrs.4x{{[[:blank:]]+}}x{{[0-9]+}}, dm{{[a-z]*[0-9]+}}, [[S59]], srssign1{{([[:blank:]]|;|$)}}
llvm.func @srs_signed_59(%acc: vector<64xi32>) -> vector<64xi8> {
  %0 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 59 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  llvm.return %0 : vector<64xi8>
}

// CHECK-LABEL: <srs_unsigned_0>:
// CHECK: vsrs.4x{{[[:blank:]]+}}x{{[0-9]+}}, dm{{[a-z]*[0-9]+}}, s{{[0-9]+}}, srssign0{{([[:blank:]]|;|$)}}
llvm.func @srs_unsigned_0(%acc: vector<64xi32>) -> vector<64xi8> {
  %0 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 0 : i32, sign = 0 : i32} : (vector<64xi32>) -> vector<64xi8>
  llvm.return %0 : vector<64xi8>
}

// Forward the runtime conf without replacing it with an immediate.
// CHECK-LABEL: <mmac_conf_arg>:
// CHECK-NOT: {{mova|mov[[:blank:]]+r[0-9]+, #}}
// CHECK: mmac.i{{[[:blank:]]+}}dm{{[a-z]*[0-9]+}}, dm{{[a-z]*[0-9]+}}, x{{[0-9]+}}, zb, r{{[0-9]+}}{{([[:blank:]]|;|$)}}
// CHECK-NOT: {{mova|mov[[:blank:]]+r[0-9]+, #}}
llvm.func @mmac_conf_arg(%acc: vector<64xi32>, %a: vector<64xi8>, %conf: i32) -> vector<64xi32> {
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
  llvm.return %0 : vector<64xi32>
}

// Combine the runtime zero_acc bit with the signed int8 configuration.
// CHECK-LABEL: <mmac_zero_if>:
// CHECK-DAG: mova{{[[:blank:]]+}}r{{[0-9]+}}, #0x308{{([[:blank:]]|;|$)}}
// CHECK-DAG: or{{[[:blank:]]+}}[[CONF:r[0-9]+]], r{{[0-9]+}}, r{{[0-9]+}}{{([[:blank:]]|;|$)}}
// CHECK-DAG: mmac.i{{[[:blank:]]+}}dm{{[a-z]*[0-9]+}}, dm{{[a-z]*[0-9]+}}, x{{[0-9]+}}, zb, [[CONF]]{{([[:blank:]]|;|$)}}
llvm.func @mmac_zero_if(%acc: vector<64xi32>, %a: vector<64xi8>, %first: i1) -> vector<64xi32> {
  %c = llvm.mlir.constant(776 : i32) : i32
  %z = llvm.zext %first : i1 to i32
  %conf = llvm.or %c, %z : i32
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
  llvm.return %0 : vector<64xi32>
}
