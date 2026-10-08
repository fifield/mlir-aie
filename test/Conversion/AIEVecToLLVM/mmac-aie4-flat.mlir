// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" -canonicalize | FileCheck %s --check-prefix=CANON

// Flat matrix-state ops lower in place to their XLLVM counterparts.

// stage: one insert per op, the slot passed through, crIntMMode from
// rhs_signed. Logical int4/fp8 chunks are bitcast to bytes; byte images are
// used as they are.
// CHECK-LABEL: func.func @stage
// CHECK-SAME: (%[[BF:.*]]: vector<64xbf16>, %[[I8:.*]]: vector<128xi8>, %[[I4:.*]]: vector<256xi4>, %[[F8:.*]]: vector<128xf8E4M3FN>, %[[S:.*]]: i32)
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[BF]], %[[S]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[I8]], %[[S]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[I8]], %[[S]]) <{mode = 2 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: %[[I4B:.*]] = vector.bitcast %[[I4]] : vector<256xi4> to vector<128xi8>
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[I4B]], %[[S]]) <{mode = 1 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[I8]], %[[S]]) <{mode = 0 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: %[[F8B:.*]] = vector.bitcast %[[F8]] : vector<128xf8E4M3FN> to vector<128xi8>
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[F8B]], %[[S]]) : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[I8]], %[[S]]) : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: return
func.func @stage(%bf: vector<64xbf16>, %i8: vector<128xi8>, %i4: vector<256xi4>, %f8: vector<128xf8E4M3FN>, %slot: i32) {
  aievec.aie4.stage bf16 %bf, %slot : vector<64xbf16>, i32
  aievec.aie4.stage int8 %i8, %slot {rhs_signed = true} : vector<128xi8>, i32
  aievec.aie4.stage int8 %i8, %slot {rhs_signed = false} : vector<128xi8>, i32
  aievec.aie4.stage int4 %i4, %slot {rhs_signed = true} : vector<256xi4>, i32
  aievec.aie4.stage int4 %i8, %slot {rhs_signed = false} : vector<128xi8>, i32
  aievec.aie4.stage fp8 %f8, %slot : vector<128xf8E4M3FN>, i32
  aievec.aie4.stage fp8 %i8, %slot : vector<128xi8>, i32
  return
}

// -----

// stage_stream: a generic 1024-bit stream read is bitcast to the
// format-specific insert's operand type.
// CHECK-LABEL: func.func @stage_stream
// CHECK-SAME: (%[[P:.*]]: i32, %[[S:.*]]: i32)
// CHECK-NEXT: %[[R0:.*]] = "xllvm.intr.aie4.get.ss"(%[[P]]) : (i32) -> vector<32xi32>
// CHECK-NEXT: %[[B0:.*]] = vector.bitcast %[[R0]] : vector<32xi32> to vector<64xbf16>
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[B0]], %[[S]])
// CHECK-NEXT: %[[R1:.*]] = "xllvm.intr.aie4.get.ss"(%[[P]]) : (i32) -> vector<32xi32>
// CHECK-NEXT: %[[B1:.*]] = vector.bitcast %[[R1]] : vector<32xi32> to vector<128xi8>
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[B1]], %[[S]]) <{mode = 3 : i32}>
// CHECK-NEXT: %[[R2:.*]] = "xllvm.intr.aie4.get.ss"(%[[P]]) : (i32) -> vector<32xi32>
// CHECK-NEXT: %[[B2:.*]] = vector.bitcast %[[R2]] : vector<32xi32> to vector<128xi8>
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[B2]], %[[S]]) <{mode = 0 : i32}>
// CHECK-NEXT: %[[R3:.*]] = "xllvm.intr.aie4.get.ss"(%[[P]]) : (i32) -> vector<32xi32>
// CHECK-NEXT: %[[B3:.*]] = vector.bitcast %[[R3]] : vector<32xi32> to vector<128xi8>
// CHECK-NEXT: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[B3]], %[[S]])
// CHECK-NEXT: return
func.func @stage_stream(%port: i32, %slot: i32) {
  aievec.aie4.stage_stream bf16 %port, %slot : i32, i32
  aievec.aie4.stage_stream int8 %port, %slot {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int4 %port, %slot {rhs_signed = false} : i32, i32
  aievec.aie4.stage_stream fp8 %port, %slot : i32, i32
  return
}

// -----

// commit: the per-format staging.to.matrix; rhs_signed is ignored.
// CHECK-LABEL: func.func @commit
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I4.I32768"() : () -> ()
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I4.I32768"() : () -> ()
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
// CHECK-NEXT: return
func.func @commit() {
  aievec.aie4.commit bf16
  aievec.aie4.commit int8 {rhs_signed = true}
  aievec.aie4.commit int4 {rhs_signed = true}
  aievec.aie4.commit int4 {rhs_signed = false}
  aievec.aie4.commit fp8
  return
}

// -----

// Without zero_if, use static MMAC configuration; absent acc selects zero_acc
// with a poison accumulator operand.
// CHECK-LABEL: func.func @mmac_active_static
// CHECK-SAME: (%[[ABF:.*]]: vector<32xbf16>, %[[CF:.*]]: vector<64xf32>, %[[A8:.*]]: vector<64xi8>, %[[A4:.*]]: vector<128xi4>, %[[AF8:.*]]: vector<64xf8E4M3FN>, %[[CI:.*]]: vector<64xi32>)
// CHECK: "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[CF]], %[[ABF]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: %[[PF:.*]] = llvm.mlir.poison : vector<64xf32>
// CHECK: "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[PF]], %[[ABF]]) <{zeroAcc}>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[CI]], %[[A8]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 0 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK: %[[PI:.*]] = llvm.mlir.poison : vector<64xi32>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[PI]], %[[A8]]) <{bmode = 1 : i32, xSign = 0 : i32, zSign = 1 : i32, zeroAcc}>
// CHECK: %[[A4B:.*]] = vector.bitcast %[[A4]] : vector<128xi4> to vector<64xi8>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[CI]], %[[A4B]]) <{bmode = 0 : i32, xSign = 1 : i32, zSign = 1 : i32}>
// CHECK: %[[AF8B:.*]] = vector.bitcast %[[AF8]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[CF]], %[[AF8B]]) <{emode = 0 : i32}>
// CHECK: %[[AF8B2:.*]] = vector.bitcast %[[AF8]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[PF2:.*]] = llvm.mlir.poison : vector<64xf32>
// CHECK: "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[PF2]], %[[AF8B2]]) <{emode = 0 : i32, zeroAcc}>
// CHECK-NOT: aievec.
func.func @mmac_active_static(%abf: vector<32xbf16>, %cf: vector<64xf32>, %a8: vector<64xi8>, %a4: vector<128xi4>, %af8: vector<64xf8E4M3FN>, %ci: vector<64xi32>) -> (vector<64xf32>, vector<64xf32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xf32>, vector<64xf32>) {
  %0 = aievec.aie4.mmac_active bf16 %abf, %cf : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  %1 = aievec.aie4.mmac_active bf16 %abf : vector<32xbf16> -> vector<64xf32>
  %2 = aievec.aie4.mmac_active int8 %a8, %ci {lhs_signed = true, rhs_signed = false} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  %3 = aievec.aie4.mmac_active int8 %a8 {lhs_signed = false, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
  %4 = aievec.aie4.mmac_active int4 %a4, %ci {lhs_signed = true, rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
  %5 = aievec.aie4.mmac_active fp8 %af8, %cf : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
  %6 = aievec.aie4.mmac_active fp8 %af8 : vector<64xf8E4M3FN> -> vector<64xf32>
  return %0, %1, %2, %3, %4, %5, %6 : vector<64xf32>, vector<64xf32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xf32>, vector<64xf32>
}

// -----

// A constant zero_if selects the static form: true is the no-acc form
// (zero_acc, poison accumulator), false the with-acc form.
// CHECK-LABEL: func.func @mmac_active_zero_if_const
// CHECK-SAME: (%[[A:.*]]: vector<64xi8>, %[[C:.*]]: vector<64xi32>)
// CHECK: %[[P:.*]] = llvm.mlir.poison : vector<64xi32>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[P]], %[[A]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32, zeroAcc}>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[C]], %[[A]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32}>
// CHECK-NOT: dyn_conf
// CHECK-NOT: arith.ori
func.func @mmac_active_zero_if_const(%a: vector<64xi8>, %c: vector<64xi32>) -> (vector<64xi32>, vector<64xi32>) {
  %t = arith.constant true
  %f = arith.constant false
  %0 = aievec.aie4.mmac_active int8 %a, %c zero_if %t {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  %1 = aievec.aie4.mmac_active int8 %a, %c zero_if %f {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  return %0, %1 : vector<64xi32>, vector<64xi32>
}

// -----

// A loop-derived predicate (first K tile) sets conf bit 0 at run time.
// int8 signed x signed is 776 (int8x8 | zSign | xSign).
// CHECK-LABEL: func.func @mmac_active_zero_if_loop
// CHECK: scf.for %[[K:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %{{.*}})
// CHECK: %[[Z:.*]] = arith.cmpi eq, %[[K]], %{{.*}} : index
// CHECK: %[[BASE:.*]] = arith.constant 776 : i32
// CHECK: %[[ZB:.*]] = arith.extui %[[Z]] : i1 to i32
// CHECK: %[[CONF:.*]] = arith.ori %[[BASE]], %[[ZB]] : i32
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%[[ACC]], %{{.*}}, %[[CONF]]) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
// int4 unsigned x signed: int4x4 (bit 3 clear) | zSign = 256.
// CHECK: %[[BASE4:.*]] = arith.constant 256 : i32
// CHECK: arith.extui %[[Z]] : i1 to i32
// CHECK: arith.ori %[[BASE4]], %{{.*}} : i32
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"
// CHECK-NOT: aievec.
// The or is not folded away.
// CANON-LABEL: func.func @mmac_active_zero_if_loop
// CANON-DAG: %[[BASE:.*]] = arith.constant 776 : i32
// CANON: scf.for
// CANON: %[[ZB:.*]] = arith.extui %{{.*}} : i1 to i32
// CANON: %[[CONF:.*]] = arith.ori %[[ZB]], %[[BASE]] : i32
// CANON: "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%{{.*}}, %{{.*}}, %[[CONF]])
func.func @mmac_active_zero_if_loop(%as: vector<4x64xi8>, %a4: vector<128xi4>, %c: vector<64xi32>) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<64xi32>) {
    %z = arith.cmpi eq, %k, %c0 : index
    %a = vector.extract %as[%k] : vector<64xi8> from vector<4x64xi8>
    %n = aievec.aie4.mmac_active int8 %a, %acc zero_if %z {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    %m = aievec.aie4.mmac_active int4 %a4, %n zero_if %z {lhs_signed = false, rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
    scf.yield %m : vector<64xi32>
  }
  return %r : vector<64xi32>
}
