// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Check matrix-region and matmul lowering across formats, signedness,
// accumulator forms, layouts, row loops, and canonicalization/CSE.
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" -canonicalize -cse 2>/dev/null | FileCheck %s --enable-var-scope --implicit-check-not=aievec.aie4.

// CHECK-LABEL: func.func @bf16(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<32x64xbf16>, %[[V2:[a-zA-Z0-9_]+]]: vector<64xf32>) -> vector<64xf32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V8]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V11]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK-NEXT: return %[[V12]] : vector<64xf32>
func.func @bf16(%a: vector<32xbf16>, %b: vector<32x64xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    %x = aievec.aie4.mmac bf16 %a, %acc : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// CHECK-LABEL: func.func @bf16_column_major(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x32xbf16>, %[[V2:[a-zA-Z0-9_]+]]: vector<64xf32>) -> vector<64xf32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.shuffle %[[V5]], %[[V6]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V7]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V8]], %[[V9]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V10]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK-NEXT: return %[[V11]] : vector<64xf32>
func.func @bf16_column_major(%a: vector<32xbf16>, %bt: vector<64x32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %b = vector.transpose %bt, [1, 0] : vector<64x32xbf16> to vector<32x64xbf16>
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    %x = aievec.aie4.mmac bf16 %a, %acc : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_zero_acc(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<32x64xbf16>) -> vector<64xf32> {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = llvm.mlir.poison : vector<64xf32>
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V8]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V11]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{zeroAcc}> : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK-NEXT: return %[[V12]] : vector<64xf32>
func.func @bf16_zero_acc(%a: vector<32xbf16>, %b: vector<32x64xbf16>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @int8_signed_b(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<64xi8>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x64xi8>, %[[V2:[a-zA-Z0-9_]+]]: vector<64xi32>) -> vector<64xi32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<64x64xi8> to vector<64x64xi8>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V8]], %[[V4]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V11]], %[[V3]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{bmode = 1 : i32, xSign = 0 : i32, zSign = 1 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK-NEXT: return %[[V12]] : vector<64xi32>
func.func @int8_signed_b(%a: vector<64xi8>, %b: vector<64x64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = true} {
  ^bb0(%acc: vector<64xi32>):
    %x = aievec.aie4.mmac int8 %a, %acc {lhs_signed = false, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// CHECK-LABEL: func.func @int8_unsigned_b(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<64xi8>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x64xi8>, %[[V2:[a-zA-Z0-9_]+]]: vector<64xi32>) -> vector<64xi32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<64x64xi8> to vector<64x64xi8>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V8]], %[[V4]]) <{mode = 2 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 2 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V11]], %[[V3]]) <{mode = 2 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 0 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK-NEXT: return %[[V12]] : vector<64xi32>
func.func @int8_unsigned_b(%a: vector<64xi8>, %b: vector<64x64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = false} {
  ^bb0(%acc: vector<64xi32>):
    %x = aievec.aie4.mmac int8 %a, %acc {lhs_signed = true, rhs_signed = false} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @int4(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<128xi4>, %[[V1:[a-zA-Z0-9_]+]]: vector<128x64xi4>, %[[V2:[a-zA-Z0-9_]+]]: vector<64xi32>) -> vector<64xi32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<128x64xi4> to vector<64x128xi4>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.bitcast %[[V6]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.bitcast %[[V8]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V7]], %[[V9]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V10]], %[[V4]]) <{mode = 1 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 1 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.bitcast %[[V11]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = vector.bitcast %[[V13]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.shuffle %[[V12]], %[[V14]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V15]], %[[V3]]) <{mode = 1 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I4.I32768"() : () -> ()
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.bitcast %[[V0]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V2]], %[[V16]]) <{bmode = 0 : i32, xSign = 1 : i32, zSign = 1 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK-NEXT: return %[[V17]] : vector<64xi32>
func.func @int4(%a: vector<128xi4>, %b: vector<128x64xi4>, %c: vector<64xi32>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int4 %b : vector<128x64xi4>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = true} {
  ^bb0(%acc: vector<64xi32>):
    %x = aievec.aie4.mmac int4 %a, %acc {lhs_signed = true, rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// CHECK-LABEL: func.func @int4_unsigned(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<128xi4>, %[[V1:[a-zA-Z0-9_]+]]: vector<128x64xi4>) -> vector<64xi32> {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = llvm.mlir.poison : vector<64xi32>
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<128x64xi4> to vector<64x128xi4>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.bitcast %[[V6]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.bitcast %[[V8]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V7]], %[[V9]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V10]], %[[V4]]) <{mode = 0 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 0 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.bitcast %[[V11]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = vector.bitcast %[[V13]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.shuffle %[[V12]], %[[V14]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V15]], %[[V3]]) <{mode = 0 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I4.I32768"() : () -> ()
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.bitcast %[[V0]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V2]], %[[V16]]) <{bmode = 0 : i32, xSign = 0 : i32, zSign = 0 : i32, zeroAcc}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK-NEXT: return %[[V17]] : vector<64xi32>
func.func @int4_unsigned(%a: vector<128xi4>, %b: vector<128x64xi4>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int4 %b : vector<128x64xi4> -> vector<64xi32> attributes {rhs_signed = false} {
    %x = aievec.aie4.mmac int4 %a {lhs_signed = false, rhs_signed = false} : vector<128xi4> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @fp8(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<64xf8E4M3FN>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x64xf8E4M3FN>, %[[V2:[a-zA-Z0-9_]+]]: vector<64xf32>) -> vector<64xf32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<64x64xf8E4M3FN> to vector<64x64xf8E4M3FN>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.bitcast %[[V6]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.bitcast %[[V8]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V7]], %[[V9]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V10]], %[[V4]]) : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%{{.*}}, %{{.*}}) : (vector<128xi8>, i32) -> ()
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.bitcast %[[V11]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = vector.bitcast %[[V13]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.shuffle %[[V12]], %[[V14]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V15]], %[[V3]]) : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.bitcast %[[V0]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[V2]], %[[V16]]) <{emode = 0 : i32}> : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
// CHECK-NEXT: return %[[V17]] : vector<64xf32>
func.func @fp8(%a: vector<64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c: vector<64xf32>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    %x = aievec.aie4.mmac fp8 %a, %acc : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<2x32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<32x64xbf16>, %[[V2:[a-zA-Z0-9_]+]]: vector<2x64xf32>) -> vector<2x64xf32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V8]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V11]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][0] : vector<32xbf16> from vector<2x32xbf16>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V2]][0] : vector<64xf32> from vector<2x64xf32>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V13]], %[[V12]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][1] : vector<32xbf16> from vector<2x32xbf16>
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.extract %[[V2]][1] : vector<64xf32> from vector<2x64xf32>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V16]], %[[V15]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: %[[V18:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V14]] : vector<64xf32>
// CHECK: %[[V19:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V17]] : vector<64xf32>
// CHECK: %[[V20:[a-zA-Z0-9_]+]] = vector.from_elements %[[V18]]#0, %[[V18]]#1, %[[V18]]#2, %[[V18]]#3, %[[V18]]#4, %[[V18]]#5, %[[V18]]#6, %[[V18]]#7, %[[V18]]#8, %[[V18]]#9, %[[V18]]#10, %[[V18]]#11, %[[V18]]#12, %[[V18]]#13, %[[V18]]#14, %[[V18]]#15, %[[V18]]#16, %[[V18]]#17, %[[V18]]#18, %[[V18]]#19, %[[V18]]#20, %[[V18]]#21, %[[V18]]#22, %[[V18]]#23, %[[V18]]#24, %[[V18]]#25, %[[V18]]#26, %[[V18]]#27, %[[V18]]#28, %[[V18]]#29, %[[V18]]#30, %[[V18]]#31, %[[V18]]#32, %[[V18]]#33, %[[V18]]#34, %[[V18]]#35, %[[V18]]#36, %[[V18]]#37, %[[V18]]#38, %[[V18]]#39, %[[V18]]#40, %[[V18]]#41, %[[V18]]#42, %[[V18]]#43, %[[V18]]#44, %[[V18]]#45, %[[V18]]#46, %[[V18]]#47, %[[V18]]#48, %[[V18]]#49, %[[V18]]#50, %[[V18]]#51, %[[V18]]#52, %[[V18]]#53, %[[V18]]#54, %[[V18]]#55, %[[V18]]#56, %[[V18]]#57, %[[V18]]#58, %[[V18]]#59, %[[V18]]#60, %[[V18]]#61, %[[V18]]#62, %[[V18]]#63, %[[V19]]#0, %[[V19]]#1, %[[V19]]#2, %[[V19]]#3, %[[V19]]#4, %[[V19]]#5, %[[V19]]#6, %[[V19]]#7, %[[V19]]#8, %[[V19]]#9, %[[V19]]#10, %[[V19]]#11, %[[V19]]#12, %[[V19]]#13, %[[V19]]#14, %[[V19]]#15, %[[V19]]#16, %[[V19]]#17, %[[V19]]#18, %[[V19]]#19, %[[V19]]#20, %[[V19]]#21, %[[V19]]#22, %[[V19]]#23, %[[V19]]#24, %[[V19]]#25, %[[V19]]#26, %[[V19]]#27, %[[V19]]#28, %[[V19]]#29, %[[V19]]#30, %[[V19]]#31, %[[V19]]#32, %[[V19]]#33, %[[V19]]#34, %[[V19]]#35, %[[V19]]#36, %[[V19]]#37, %[[V19]]#38, %[[V19]]#39, %[[V19]]#40, %[[V19]]#41, %[[V19]]#42, %[[V19]]#43, %[[V19]]#44, %[[V19]]#45, %[[V19]]#46, %[[V19]]#47, %[[V19]]#48, %[[V19]]#49, %[[V19]]#50, %[[V19]]#51, %[[V19]]#52, %[[V19]]#53, %[[V19]]#54, %[[V19]]#55, %[[V19]]#56, %[[V19]]#57, %[[V19]]#58, %[[V19]]#59, %[[V19]]#60, %[[V19]]#61, %[[V19]]#62, %[[V19]]#63 : vector<2x64xf32>
// CHECK-NEXT: return %[[V20]] : vector<2x64xf32>
func.func @matmul(%a: vector<2x32xbf16>, %b: vector<32x64xbf16>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %0 = aievec.aie4.matmul %a, %b, %c : vector<2x32xbf16>, vector<32x64xbf16> into vector<2x64xf32>
  return %0 : vector<2x64xf32>
}

// -----

// CHECK-LABEL: func.func @yield_block_arg(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<32x64xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<64xf32>) -> vector<64xf32> {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V4:[a-zA-Z0-9_]+]] = vector.transpose %[[V0]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.extract %[[V4]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V4]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.shuffle %[[V5]], %[[V6]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V7]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V4]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V4]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V8]], %[[V9]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V10]], %[[V2]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK-NEXT: return %[[V1]] : vector<64xf32>
func.func @yield_block_arg(%b: vector<32x64xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    aievec.aie4.matrix_yield %acc : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @k_loop_row_loop(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<4x32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<32x64xbf16>, %[[V2:[a-zA-Z0-9_]+]]: vector<4x64xf32>) -> vector<4x64xf32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = scf.for %[[V6:[a-zA-Z0-9_]+]] = %[[V7:[a-zA-Z0-9_]+]] to %[[V8:[a-zA-Z0-9_]+]] step %[[V9:[a-zA-Z0-9_]+]] iter_args(%[[V10:[a-zA-Z0-9_]+]] = %[[V2]]) -> (vector<4x64xf32>) {
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.extract %[[V11]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V11]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = vector.shuffle %[[V12]], %[[V13]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V14]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.extract %[[V11]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.extract %[[V11]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = vector.shuffle %[[V15]], %[[V16]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V17]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V18:[a-zA-Z0-9_]+]] = scf.for %[[V19:[a-zA-Z0-9_]+]] = %[[V7]] to %[[V8]] step %[[V9]] iter_args(%[[V20:[a-zA-Z0-9_]+]] = %[[V10]]) -> (vector<4x64xf32>) {
// CHECK: %[[V21:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][%[[V19]]] : vector<32xbf16> from vector<4x32xbf16>
// CHECK: %[[V22:[a-zA-Z0-9_]+]] = vector.extract %[[V20]][%[[V19]]] : vector<64xf32> from vector<4x64xf32>
// CHECK: %[[V23:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V22]], %[[V21]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: %[[V24:[a-zA-Z0-9_]+]] = vector.insert %[[V23]], %[[V20]] [%[[V19]]] : vector<64xf32> into vector<4x64xf32>
// CHECK: scf.yield %[[V24]] : vector<4x64xf32>
// CHECK: scf.yield %[[V18]] : vector<4x64xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V5]] : vector<4x64xf32>
func.func @k_loop_row_loop(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc_k = %c) -> (vector<4x64xf32>) {
    %out = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %acc_k : vector<4x64xf32> -> vector<4x64xf32> {
    ^bb0(%acc0: vector<4x64xf32>):
      %rows = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %acc0) -> (vector<4x64xf32>) {
        %a_i = vector.extract %a[%i] : vector<32xbf16> from vector<4x32xbf16>
        %c_i = vector.extract %acc[%i] : vector<64xf32> from vector<4x64xf32>
        %r_i = aievec.aie4.mmac bf16 %a_i, %c_i : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
        %next = vector.insert %r_i, %acc [%i] : vector<64xf32> into vector<4x64xf32>
        scf.yield %next : vector<4x64xf32>
      }
      aievec.aie4.matrix_yield %rows : vector<4x64xf32>
    }
    scf.yield %out : vector<4x64xf32>
  }
  return %res : vector<4x64xf32>
}

// -----

// CHECK-LABEL: func.func @no_cse(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<64xi8>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x64xi8>, %[[V2:[a-zA-Z0-9_]+]]: vector<64xi32>) -> (vector<64xi32>, vector<64xi32>) {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<64x64xi8> to vector<64x64xi8>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V8]], %[[V4]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V11]], %[[V3]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V8]], %[[V4]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V11]], %[[V3]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK-NEXT: return %[[V12]], %[[V13]] : vector<64xi32>, vector<64xi32>
func.func @no_cse(%a: vector<64xi8>, %b: vector<64x64xi8>, %c: vector<64xi32>) -> (vector<64xi32>, vector<64xi32>) {
  %r0 = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = aievec.aie4.mmac int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  %r1 = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = aievec.aie4.mmac int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r0, %r1 : vector<64xi32>, vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @no_dce(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<64xf8E4M3FN>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x64xf8E4M3FN>) {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = llvm.mlir.poison : vector<64xf32>
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<64x64xf8E4M3FN> to vector<64x64xf8E4M3FN>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.bitcast %[[V6]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.bitcast %[[V8]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V7]], %[[V9]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V10]], %[[V4]]) : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%{{.*}}, %{{.*}}) : (vector<128xi8>, i32) -> ()
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.bitcast %[[V11]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = vector.bitcast %[[V13]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.shuffle %[[V12]], %[[V14]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V15]], %[[V3]]) : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.bitcast %[[V0]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[V2]], %[[V16]]) <{emode = 0 : i32, zeroAcc}> : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
// CHECK-NEXT: return
func.func @no_dce(%a: vector<64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>) {
  %r = aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> -> vector<64xf32> {
    %x = aievec.aie4.mmac fp8 %a : vector<64xf8E4M3FN> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @bf16_rows(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<3x32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<32x64xbf16>, %[[V2:[a-zA-Z0-9_]+]]: vector<3x64xf32>) -> vector<3x64xf32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V8]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V11]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][0] : vector<32xbf16> from vector<3x32xbf16>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V2]][0] : vector<64xf32> from vector<3x64xf32>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V13]], %[[V12]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][1] : vector<32xbf16> from vector<3x32xbf16>
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.extract %[[V2]][1] : vector<64xf32> from vector<3x64xf32>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V16]], %[[V15]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: %[[V18:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][2] : vector<32xbf16> from vector<3x32xbf16>
// CHECK: %[[V19:[a-zA-Z0-9_]+]] = vector.extract %[[V2]][2] : vector<64xf32> from vector<3x64xf32>
// CHECK: %[[V20:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V19]], %[[V18]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: %[[V21:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V14]] : vector<64xf32>
// CHECK: %[[V22:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V17]] : vector<64xf32>
// CHECK: %[[V23:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V20]] : vector<64xf32>
// CHECK: %[[V24:[a-zA-Z0-9_]+]] = vector.from_elements %[[V21]]#0, %[[V21]]#1, %[[V21]]#2, %[[V21]]#3, %[[V21]]#4, %[[V21]]#5, %[[V21]]#6, %[[V21]]#7, %[[V21]]#8, %[[V21]]#9, %[[V21]]#10, %[[V21]]#11, %[[V21]]#12, %[[V21]]#13, %[[V21]]#14, %[[V21]]#15, %[[V21]]#16, %[[V21]]#17, %[[V21]]#18, %[[V21]]#19, %[[V21]]#20, %[[V21]]#21, %[[V21]]#22, %[[V21]]#23, %[[V21]]#24, %[[V21]]#25, %[[V21]]#26, %[[V21]]#27, %[[V21]]#28, %[[V21]]#29, %[[V21]]#30, %[[V21]]#31, %[[V21]]#32, %[[V21]]#33, %[[V21]]#34, %[[V21]]#35, %[[V21]]#36, %[[V21]]#37, %[[V21]]#38, %[[V21]]#39, %[[V21]]#40, %[[V21]]#41, %[[V21]]#42, %[[V21]]#43, %[[V21]]#44, %[[V21]]#45, %[[V21]]#46, %[[V21]]#47, %[[V21]]#48, %[[V21]]#49, %[[V21]]#50, %[[V21]]#51, %[[V21]]#52, %[[V21]]#53, %[[V21]]#54, %[[V21]]#55, %[[V21]]#56, %[[V21]]#57, %[[V21]]#58, %[[V21]]#59, %[[V21]]#60, %[[V21]]#61, %[[V21]]#62, %[[V21]]#63, %[[V22]]#0, %[[V22]]#1, %[[V22]]#2, %[[V22]]#3, %[[V22]]#4, %[[V22]]#5, %[[V22]]#6, %[[V22]]#7, %[[V22]]#8, %[[V22]]#9, %[[V22]]#10, %[[V22]]#11, %[[V22]]#12, %[[V22]]#13, %[[V22]]#14, %[[V22]]#15, %[[V22]]#16, %[[V22]]#17, %[[V22]]#18, %[[V22]]#19, %[[V22]]#20, %[[V22]]#21, %[[V22]]#22, %[[V22]]#23, %[[V22]]#24, %[[V22]]#25, %[[V22]]#26, %[[V22]]#27, %[[V22]]#28, %[[V22]]#29, %[[V22]]#30, %[[V22]]#31, %[[V22]]#32, %[[V22]]#33, %[[V22]]#34, %[[V22]]#35, %[[V22]]#36, %[[V22]]#37, %[[V22]]#38, %[[V22]]#39, %[[V22]]#40, %[[V22]]#41, %[[V22]]#42, %[[V22]]#43, %[[V22]]#44, %[[V22]]#45, %[[V22]]#46, %[[V22]]#47, %[[V22]]#48, %[[V22]]#49, %[[V22]]#50, %[[V22]]#51, %[[V22]]#52, %[[V22]]#53, %[[V22]]#54, %[[V22]]#55, %[[V22]]#56, %[[V22]]#57, %[[V22]]#58, %[[V22]]#59, %[[V22]]#60, %[[V22]]#61, %[[V22]]#62, %[[V22]]#63, %[[V23]]#0, %[[V23]]#1, %[[V23]]#2, %[[V23]]#3, %[[V23]]#4, %[[V23]]#5, %[[V23]]#6, %[[V23]]#7, %[[V23]]#8, %[[V23]]#9, %[[V23]]#10, %[[V23]]#11, %[[V23]]#12, %[[V23]]#13, %[[V23]]#14, %[[V23]]#15, %[[V23]]#16, %[[V23]]#17, %[[V23]]#18, %[[V23]]#19, %[[V23]]#20, %[[V23]]#21, %[[V23]]#22, %[[V23]]#23, %[[V23]]#24, %[[V23]]#25, %[[V23]]#26, %[[V23]]#27, %[[V23]]#28, %[[V23]]#29, %[[V23]]#30, %[[V23]]#31, %[[V23]]#32, %[[V23]]#33, %[[V23]]#34, %[[V23]]#35, %[[V23]]#36, %[[V23]]#37, %[[V23]]#38, %[[V23]]#39, %[[V23]]#40, %[[V23]]#41, %[[V23]]#42, %[[V23]]#43, %[[V23]]#44, %[[V23]]#45, %[[V23]]#46, %[[V23]]#47, %[[V23]]#48, %[[V23]]#49, %[[V23]]#50, %[[V23]]#51, %[[V23]]#52, %[[V23]]#53, %[[V23]]#54, %[[V23]]#55, %[[V23]]#56, %[[V23]]#57, %[[V23]]#58, %[[V23]]#59, %[[V23]]#60, %[[V23]]#61, %[[V23]]#62, %[[V23]]#63 : vector<3x64xf32>
// CHECK-NEXT: return %[[V24]] : vector<3x64xf32>
func.func @bf16_rows(%a: vector<3x32xbf16>, %b: vector<32x64xbf16>, %c: vector<3x64xf32>) -> vector<3x64xf32> {
  %0 = aievec.aie4.matmul %a, %b, %c : vector<3x32xbf16>, vector<32x64xbf16> into vector<3x64xf32>
  return %0 : vector<3x64xf32>
}

// -----

// CHECK-LABEL: func.func @int4_signs(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<1x128xi4>, %[[V1:[a-zA-Z0-9_]+]]: vector<128x64xi4>, %[[V2:[a-zA-Z0-9_]+]]: vector<1x64xi32>) -> vector<1x64xi32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<128x64xi4> to vector<64x128xi4>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.bitcast %[[V6]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.bitcast %[[V8]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V7]], %[[V9]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V10]], %[[V4]]) <{mode = 0 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 0 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.bitcast %[[V11]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = vector.bitcast %[[V13]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.shuffle %[[V12]], %[[V14]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V15]], %[[V3]]) <{mode = 0 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I4.I32768"() : () -> ()
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.shape_cast %[[V0]] : vector<1x128xi4> to vector<128xi4>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = vector.shape_cast %[[V2]] : vector<1x64xi32> to vector<64xi32>
// CHECK: %[[V18:[a-zA-Z0-9_]+]] = vector.bitcast %[[V16]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[V19:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V17]], %[[V18]]) <{bmode = 0 : i32, xSign = 1 : i32, zSign = 0 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK: %[[V20:[a-zA-Z0-9_]+]] = vector.shape_cast %[[V19]] : vector<64xi32> to vector<1x64xi32>
// CHECK-NEXT: return %[[V20]] : vector<1x64xi32>
func.func @int4_signs(%a: vector<1x128xi4>, %b: vector<128x64xi4>, %c: vector<1x64xi32>) -> vector<1x64xi32> {
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = false} : vector<1x128xi4>, vector<128x64xi4> into vector<1x64xi32>
  return %0 : vector<1x64xi32>
}

// -----

// CHECK-LABEL: func.func @two_k_tiles(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<2x64xf8E4M3FN>, %[[V1:[a-zA-Z0-9_]+]]: vector<2x64xf8E4M3FN>, %[[V2:[a-zA-Z0-9_]+]]: vector<64x64xf8E4M3FN>, %[[V3:[a-zA-Z0-9_]+]]: vector<64x64xf8E4M3FN>, %[[V4:[a-zA-Z0-9_]+]]: vector<2x64xf32>) -> vector<2x64xf32> {
// CHECK-DAG: %[[V5:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V6:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.transpose %[[V2]], [1, 0] : vector<64x64xf8E4M3FN> to vector<64x64xf8E4M3FN>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V7]][0] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.bitcast %[[V8]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V7]][1] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.bitcast %[[V10]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V11]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V12]], %[[V6]]) : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%{{.*}}, %{{.*}}) : (vector<128xi8>, i32) -> ()
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V7]][62] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = vector.bitcast %[[V13]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.extract %[[V7]][63] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.bitcast %[[V15]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = vector.shuffle %[[V14]], %[[V16]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V17]], %[[V5]]) : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
// CHECK: %[[V18:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][0] : vector<64xf8E4M3FN> from vector<2x64xf8E4M3FN>
// CHECK: %[[V19:[a-zA-Z0-9_]+]] = vector.extract %[[V4]][0] : vector<64xf32> from vector<2x64xf32>
// CHECK: %[[V20:[a-zA-Z0-9_]+]] = vector.bitcast %[[V18]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V21:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[V19]], %[[V20]]) <{emode = 0 : i32}> : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
// CHECK: %[[V22:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][1] : vector<64xf8E4M3FN> from vector<2x64xf8E4M3FN>
// CHECK: %[[V23:[a-zA-Z0-9_]+]] = vector.extract %[[V4]][1] : vector<64xf32> from vector<2x64xf32>
// CHECK: %[[V24:[a-zA-Z0-9_]+]] = vector.bitcast %[[V22]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V25:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[V23]], %[[V24]]) <{emode = 0 : i32}> : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
// CHECK: %[[V26:[a-zA-Z0-9_]+]] = vector.transpose %[[V3]], [1, 0] : vector<64x64xf8E4M3FN> to vector<64x64xf8E4M3FN>
// CHECK: %[[V27:[a-zA-Z0-9_]+]] = vector.extract %[[V26]][0] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V28:[a-zA-Z0-9_]+]] = vector.bitcast %[[V27]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V29:[a-zA-Z0-9_]+]] = vector.extract %[[V26]][1] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V30:[a-zA-Z0-9_]+]] = vector.bitcast %[[V29]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V31:[a-zA-Z0-9_]+]] = vector.shuffle %[[V28]], %[[V30]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V31]], %[[V6]]) : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%{{.*}}, %{{.*}}) : (vector<128xi8>, i32) -> ()
// CHECK: %[[V32:[a-zA-Z0-9_]+]] = vector.extract %[[V26]][62] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V33:[a-zA-Z0-9_]+]] = vector.bitcast %[[V32]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V34:[a-zA-Z0-9_]+]] = vector.extract %[[V26]][63] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[V35:[a-zA-Z0-9_]+]] = vector.bitcast %[[V34]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V36:[a-zA-Z0-9_]+]] = vector.shuffle %[[V33]], %[[V35]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[V36]], %[[V5]]) : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
// CHECK: %[[V37:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][0] : vector<64xf8E4M3FN> from vector<2x64xf8E4M3FN>
// CHECK: %[[V38:[a-zA-Z0-9_]+]] = vector.bitcast %[[V37]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V39:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[V21]], %[[V38]]) <{emode = 0 : i32}> : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
// CHECK: %[[V40:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][1] : vector<64xf8E4M3FN> from vector<2x64xf8E4M3FN>
// CHECK: %[[V41:[a-zA-Z0-9_]+]] = vector.bitcast %[[V40]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[V42:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%[[V25]], %[[V41]]) <{emode = 0 : i32}> : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
// CHECK: %[[V43:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V39]] : vector<64xf32>
// CHECK: %[[V44:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V42]] : vector<64xf32>
// CHECK: %[[V45:[a-zA-Z0-9_]+]] = vector.from_elements %[[V43]]#0, %[[V43]]#1, %[[V43]]#2, %[[V43]]#3, %[[V43]]#4, %[[V43]]#5, %[[V43]]#6, %[[V43]]#7, %[[V43]]#8, %[[V43]]#9, %[[V43]]#10, %[[V43]]#11, %[[V43]]#12, %[[V43]]#13, %[[V43]]#14, %[[V43]]#15, %[[V43]]#16, %[[V43]]#17, %[[V43]]#18, %[[V43]]#19, %[[V43]]#20, %[[V43]]#21, %[[V43]]#22, %[[V43]]#23, %[[V43]]#24, %[[V43]]#25, %[[V43]]#26, %[[V43]]#27, %[[V43]]#28, %[[V43]]#29, %[[V43]]#30, %[[V43]]#31, %[[V43]]#32, %[[V43]]#33, %[[V43]]#34, %[[V43]]#35, %[[V43]]#36, %[[V43]]#37, %[[V43]]#38, %[[V43]]#39, %[[V43]]#40, %[[V43]]#41, %[[V43]]#42, %[[V43]]#43, %[[V43]]#44, %[[V43]]#45, %[[V43]]#46, %[[V43]]#47, %[[V43]]#48, %[[V43]]#49, %[[V43]]#50, %[[V43]]#51, %[[V43]]#52, %[[V43]]#53, %[[V43]]#54, %[[V43]]#55, %[[V43]]#56, %[[V43]]#57, %[[V43]]#58, %[[V43]]#59, %[[V43]]#60, %[[V43]]#61, %[[V43]]#62, %[[V43]]#63, %[[V44]]#0, %[[V44]]#1, %[[V44]]#2, %[[V44]]#3, %[[V44]]#4, %[[V44]]#5, %[[V44]]#6, %[[V44]]#7, %[[V44]]#8, %[[V44]]#9, %[[V44]]#10, %[[V44]]#11, %[[V44]]#12, %[[V44]]#13, %[[V44]]#14, %[[V44]]#15, %[[V44]]#16, %[[V44]]#17, %[[V44]]#18, %[[V44]]#19, %[[V44]]#20, %[[V44]]#21, %[[V44]]#22, %[[V44]]#23, %[[V44]]#24, %[[V44]]#25, %[[V44]]#26, %[[V44]]#27, %[[V44]]#28, %[[V44]]#29, %[[V44]]#30, %[[V44]]#31, %[[V44]]#32, %[[V44]]#33, %[[V44]]#34, %[[V44]]#35, %[[V44]]#36, %[[V44]]#37, %[[V44]]#38, %[[V44]]#39, %[[V44]]#40, %[[V44]]#41, %[[V44]]#42, %[[V44]]#43, %[[V44]]#44, %[[V44]]#45, %[[V44]]#46, %[[V44]]#47, %[[V44]]#48, %[[V44]]#49, %[[V44]]#50, %[[V44]]#51, %[[V44]]#52, %[[V44]]#53, %[[V44]]#54, %[[V44]]#55, %[[V44]]#56, %[[V44]]#57, %[[V44]]#58, %[[V44]]#59, %[[V44]]#60, %[[V44]]#61, %[[V44]]#62, %[[V44]]#63 : vector<2x64xf32>
// CHECK-NEXT: return %[[V45]] : vector<2x64xf32>
func.func @two_k_tiles(%a0: vector<2x64xf8E4M3FN>, %a1: vector<2x64xf8E4M3FN>, %b0: vector<64x64xf8E4M3FN>, %b1: vector<64x64xf8E4M3FN>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %0 = aievec.aie4.matmul %a0, %b0, %c : vector<2x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<2x64xf32>
  %1 = aievec.aie4.matmul %a1, %b1, %0 : vector<2x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<2x64xf32>
  return %1 : vector<2x64xf32>
}

// -----

// CHECK-LABEL: func.func @row_major(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<32x64xbf16>) -> vector<64xf32> {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = llvm.mlir.poison : vector<64xf32>
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V8]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V11]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{zeroAcc}> : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK-NEXT: return %[[V12]] : vector<64xf32>
func.func @row_major(%a: vector<32xbf16>, %b: vector<32x64xbf16>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_row_major(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<2x64xi8>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x64xi8>, %[[V2:[a-zA-Z0-9_]+]]: vector<2x64xi32>) -> vector<2x64xi32> {
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<64x64xi8> to vector<64x64xi8>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V8]], %[[V4]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V11]], %[[V3]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][0] : vector<64xi8> from vector<2x64xi8>
// CHECK: %[[V13:[a-zA-Z0-9_]+]] = vector.extract %[[V2]][0] : vector<64xi32> from vector<2x64xi32>
// CHECK: %[[V14:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V13]], %[[V12]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK: %[[V15:[a-zA-Z0-9_]+]] = vector.extract %[[V0]][1] : vector<64xi8> from vector<2x64xi8>
// CHECK: %[[V16:[a-zA-Z0-9_]+]] = vector.extract %[[V2]][1] : vector<64xi32> from vector<2x64xi32>
// CHECK: %[[V17:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V16]], %[[V15]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK: %[[V18:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V14]] : vector<64xi32>
// CHECK: %[[V19:[a-zA-Z0-9_]+]]:64 = vector.to_elements %[[V17]] : vector<64xi32>
// CHECK: %[[V20:[a-zA-Z0-9_]+]] = vector.from_elements %[[V18]]#0, %[[V18]]#1, %[[V18]]#2, %[[V18]]#3, %[[V18]]#4, %[[V18]]#5, %[[V18]]#6, %[[V18]]#7, %[[V18]]#8, %[[V18]]#9, %[[V18]]#10, %[[V18]]#11, %[[V18]]#12, %[[V18]]#13, %[[V18]]#14, %[[V18]]#15, %[[V18]]#16, %[[V18]]#17, %[[V18]]#18, %[[V18]]#19, %[[V18]]#20, %[[V18]]#21, %[[V18]]#22, %[[V18]]#23, %[[V18]]#24, %[[V18]]#25, %[[V18]]#26, %[[V18]]#27, %[[V18]]#28, %[[V18]]#29, %[[V18]]#30, %[[V18]]#31, %[[V18]]#32, %[[V18]]#33, %[[V18]]#34, %[[V18]]#35, %[[V18]]#36, %[[V18]]#37, %[[V18]]#38, %[[V18]]#39, %[[V18]]#40, %[[V18]]#41, %[[V18]]#42, %[[V18]]#43, %[[V18]]#44, %[[V18]]#45, %[[V18]]#46, %[[V18]]#47, %[[V18]]#48, %[[V18]]#49, %[[V18]]#50, %[[V18]]#51, %[[V18]]#52, %[[V18]]#53, %[[V18]]#54, %[[V18]]#55, %[[V18]]#56, %[[V18]]#57, %[[V18]]#58, %[[V18]]#59, %[[V18]]#60, %[[V18]]#61, %[[V18]]#62, %[[V18]]#63, %[[V19]]#0, %[[V19]]#1, %[[V19]]#2, %[[V19]]#3, %[[V19]]#4, %[[V19]]#5, %[[V19]]#6, %[[V19]]#7, %[[V19]]#8, %[[V19]]#9, %[[V19]]#10, %[[V19]]#11, %[[V19]]#12, %[[V19]]#13, %[[V19]]#14, %[[V19]]#15, %[[V19]]#16, %[[V19]]#17, %[[V19]]#18, %[[V19]]#19, %[[V19]]#20, %[[V19]]#21, %[[V19]]#22, %[[V19]]#23, %[[V19]]#24, %[[V19]]#25, %[[V19]]#26, %[[V19]]#27, %[[V19]]#28, %[[V19]]#29, %[[V19]]#30, %[[V19]]#31, %[[V19]]#32, %[[V19]]#33, %[[V19]]#34, %[[V19]]#35, %[[V19]]#36, %[[V19]]#37, %[[V19]]#38, %[[V19]]#39, %[[V19]]#40, %[[V19]]#41, %[[V19]]#42, %[[V19]]#43, %[[V19]]#44, %[[V19]]#45, %[[V19]]#46, %[[V19]]#47, %[[V19]]#48, %[[V19]]#49, %[[V19]]#50, %[[V19]]#51, %[[V19]]#52, %[[V19]]#53, %[[V19]]#54, %[[V19]]#55, %[[V19]]#56, %[[V19]]#57, %[[V19]]#58, %[[V19]]#59, %[[V19]]#60, %[[V19]]#61, %[[V19]]#62, %[[V19]]#63 : vector<2x64xi32>
// CHECK-NEXT: return %[[V20]] : vector<2x64xi32>
func.func @matmul_row_major(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<2x64xi8>, vector<64x64xi8> into vector<2x64xi32>
  return %0 : vector<2x64xi32>
}

// -----

// CHECK-LABEL: func.func @column_major(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<64x32xbf16>) -> vector<64xf32> {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = llvm.mlir.poison : vector<64xf32>
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.shuffle %[[V5]], %[[V6]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V7]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V1]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.shuffle %[[V8]], %[[V9]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V10]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{zeroAcc}> : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK-NEXT: return %[[V11]] : vector<64xf32>
func.func @column_major(%a: vector<32xbf16>, %bt: vector<64x32xbf16>) -> vector<64xf32> {
  %b = vector.transpose %bt, [1, 0] : vector<64x32xbf16> to vector<32x64xbf16>
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @splat(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<64xi8>, %[[V1:[a-zA-Z0-9_]+]]: i8) -> vector<64xi32> {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = llvm.mlir.poison : vector<64xi32>
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.broadcast %[[V1]] : i8 to vector<128xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V5]], %[[V4]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[V5]], %[[V3]]) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32, zeroAcc}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
// CHECK-NEXT: return %[[V6]] : vector<64xi32>
func.func @splat(%a: vector<64xi8>, %s: i8) -> vector<64xi32> {
  %b = vector.broadcast %s : i8 to vector<64x64xi8>
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = aievec.aie4.mmac int8 %a {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @identity_transpose(
// CHECK-SAME: %[[V0:[a-zA-Z0-9_]+]]: vector<32xbf16>, %[[V1:[a-zA-Z0-9_]+]]: vector<32x64xbf16>) -> vector<64xf32> {
// CHECK-DAG: %[[V2:[a-zA-Z0-9_]+]] = llvm.mlir.poison : vector<64xf32>
// CHECK-DAG: %[[V3:[a-zA-Z0-9_]+]] = arith.constant 31 : i32
// CHECK-DAG: %[[V4:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[V5:[a-zA-Z0-9_]+]] = vector.transpose %[[V1]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[V6:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V7:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V8:[a-zA-Z0-9_]+]] = vector.shuffle %[[V6]], %[[V7]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V8]], %[[V4]]) : (vector<64xbf16>, i32) -> ()
// CHECK-COUNT-30: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %{{.*}}) : (vector<64xbf16>, i32) -> ()
// CHECK: %[[V9:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][62] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V10:[a-zA-Z0-9_]+]] = vector.extract %[[V5]][63] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[V11:[a-zA-Z0-9_]+]] = vector.shuffle %[[V9]], %[[V10]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[V11]], %[[V3]]) : (vector<64xbf16>, i32) -> ()
// CHECK-NOT: "xllvm.intr.aie4.
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK: %[[V12:[a-zA-Z0-9_]+]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[V2]], %[[V0]]) <{zeroAcc}> : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK-NEXT: return %[[V12]] : vector<64xf32>
func.func @identity_transpose(%a: vector<32xbf16>, %b0: vector<32x64xbf16>) -> vector<64xf32> {
  %b = vector.transpose %b0, [0, 1] : vector<32x64xbf16> to vector<32x64xbf16>
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}
