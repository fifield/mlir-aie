// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile the lowered flat matrix-state ops with Peano and check the
// selected instructions. llc accepts unknown llvm.aie4.* names as external
// calls, so the checks also reject any residual call.
//
// REQUIRES: peano
//
// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b" -convert-scf-to-cf -convert-to-llvm | aie-translate --mlir-to-llvmir -o %t.ll
// RUN: llc -O2 --mtriple=aie4 --mcpu=aie4b -filetype=obj %t.ll -o %t.o
// RUN: llvm-objdump -dr %t.o | FileCheck %s --implicit-check-not=llvm.aie4 --implicit-check-not=jl

// A stream pop goes straight into a y register and from there to staging.
// CHECK-LABEL: <stage_stream>:
// CHECK-DAG: vmovb{{[[:blank:]]+}}y{{[0-9]+}}, ss[p{{[0-9]+}}]
// CHECK-DAG: vmov{{[[:blank:]]+}}za[i{{[0-9]+}}], y{{[0-9]+}}
// CHECK-DAG: vmov{{[[:blank:]]+}}zb, za{{([[:blank:]]|;|$)}}
func.func @stage_stream(%port: i32, %slot: i32) {
  aievec.aie4.stage_stream int8 %port, %slot {rhs_signed = true} : i32, i32
  aievec.aie4.commit int8 {rhs_signed = true}
  return
}

// A first-K-tile predicate inside the loop: the MMAC conf register is the
// result of an or with the static conf 0x308, not a single immediate.
// CHECK-LABEL: <mmac_zero_if_loop>:
// CHECK-DAG: mova{{[[:blank:]]+}}[[BASE:r[0-9]+]], #0x308{{([[:blank:]]|;|$)}}
// CHECK-DAG: mmac.i{{[[:blank:]]+}}dm{{[a-z]*[0-9]+}}, dm{{[a-z]*[0-9]+}}, x{{[0-9]+}}, zb, [[CONF:r[0-9]+]]{{([[:blank:]]|;|$)}}
// CHECK-DAG: or{{[[:blank:]]+}}[[CONF]], {{(r[0-9]+, )?}}[[BASE]]{{(, r[0-9]+)?([[:blank:]]|;|$)}}
func.func @mmac_zero_if_loop(%a: vector<64xi8>, %c: vector<64xi32>, %n: index) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %k = %c0 to %n step %c1 iter_args(%acc = %c) -> (vector<64xi32>) {
    %z = arith.cmpi eq, %k, %c0 : index
    %y = aievec.aie4.mmac_active int8 %a, %acc zero_if %z {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    scf.yield %y : vector<64xi32>
  }
  return %r : vector<64xi32>
}
