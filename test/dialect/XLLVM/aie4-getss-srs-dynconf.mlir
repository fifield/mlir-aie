// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Round-trip the new ops and preserve their unknown effects through CSE/DCE.
//
// RUN: aie-opt %s -split-input-file | aie-opt -split-input-file | FileCheck %s
// RUN: aie-opt %s -split-input-file -cse | FileCheck %s --check-prefix=CSE
// RUN: aie-opt %s -split-input-file -canonicalize | FileCheck %s --check-prefix=DCE

// CHECK-LABEL: llvm.func @roundtrip
// CHECK: "xllvm.intr.aie4.get.ss"(%{{.*}}) <{kind = 0 : i32}> : (i32) -> vector<32xi32>
// CHECK: "xllvm.intr.aie4.get.ss"(%{{.*}}) <{kind = 1 : i32}> : (i32) -> vector<32xi32>
// CHECK: "xllvm.intr.aie4.get.ss"(%{{.*}}) <{kind = 2 : i32}> : (i32) -> vector<32xi32>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"(%{{.*}}) <{shift = 0 : i32, sign = 0 : i32}> : (vector<64xi32>) -> vector<64xi8>
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"(%{{.*}}) <{shift = 59 : i32, sign = 1 : i32}> : (vector<64xi32>) -> vector<64xi8>
llvm.func @roundtrip(%port: i32, %acc: vector<64xi32>, %a: vector<64xi8>, %conf: i32) {
  %0 = "xllvm.intr.aie4.get.ss"(%port) {kind = 0 : i32} : (i32) -> vector<32xi32>
  %1 = "xllvm.intr.aie4.get.ss"(%port) {kind = 1 : i32} : (i32) -> vector<32xi32>
  %2 = "xllvm.intr.aie4.get.ss"(%port) {kind = 2 : i32} : (i32) -> vector<32xi32>
  %3 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
  %4 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%3) {shift = 0 : i32, sign = 0 : i32} : (vector<64xi32>) -> vector<64xi8>
  %5 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%3) {shift = 59 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  llvm.return
}

// -----

// CSE-LABEL: llvm.func @pairs_not_merged
// CSE: "xllvm.intr.aie4.get.ss"
// CSE-NEXT: "xllvm.intr.aie4.get.ss"
// CSE: %[[M0:.*]] = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"
// CSE: %[[M1:.*]] = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"
// CSE: llvm.add %[[M0]], %[[M1]]
// CSE: %[[R0:.*]] = "xllvm.intr.aie4.I512.v64.acc32.srs"
// CSE: %[[R1:.*]] = "xllvm.intr.aie4.I512.v64.acc32.srs"
// CSE: llvm.add %[[R0]], %[[R1]]
llvm.func @pairs_not_merged(%port: i32, %acc: vector<64xi32>, %a: vector<64xi8>, %conf: i32) -> vector<64xi8> {
  %0 = "xllvm.intr.aie4.get.ss"(%port) {kind = 1 : i32} : (i32) -> vector<32xi32>
  %1 = "xllvm.intr.aie4.get.ss"(%port) {kind = 1 : i32} : (i32) -> vector<32xi32>
  %3 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
  %4 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
  %5 = llvm.add %3, %4 : vector<64xi32>
  %6 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%5) {shift = 12 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  %7 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%5) {shift = 12 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  %8 = llvm.add %6, %7 : vector<64xi8>
  llvm.return %8 : vector<64xi8>
}

// -----

// DCE-LABEL: llvm.func @unused_kept
// DCE: "xllvm.intr.aie4.get.ss"
// DCE: "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"
// DCE: "xllvm.intr.aie4.I512.v64.acc32.srs"
llvm.func @unused_kept(%port: i32, %acc: vector<64xi32>, %a: vector<64xi8>, %conf: i32) {
  %0 = "xllvm.intr.aie4.get.ss"(%port) {kind = 0 : i32} : (i32) -> vector<32xi32>
  %1 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
  %2 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 12 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  llvm.return
}
