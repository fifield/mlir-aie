// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: %python %S/Inputs/check_aie4_mmac_staging.py %t.mlir
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" -canonicalize | FileCheck %s --check-prefix=CANON

// Region lowering: 32 staging inserts of two B columns each, then the commit,
// then the body in place. Row MMACs carry their configuration fields.

// CHECK-LABEL: func.func @bf16
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32x64xbf16>, %[[C:.*]]: vector<64xf32>)
// B^T rows are the B columns (column-major staging layout).
// CHECK: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>
// CHECK: %[[L0:.*]] = vector.extract %[[BT]][0] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[H0:.*]] = vector.extract %[[BT]][1] : vector<32xbf16> from vector<64x32xbf16>
// CHECK: %[[K0:.*]] = vector.shuffle %[[L0]], %[[H0]] [0, 1, 2, {{.*}}, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[I0:.*]] = arith.constant 0 : i32
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[K0]], %[[I0]]) : (vector<64xbf16>, i32) -> ()
// CHECK: vector.extract %[[BT]][2]
// CHECK: vector.extract %[[BT]][3]
// CHECK: %[[I1:.*]] = arith.constant 1 : i32
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %[[I1]])
// CHECK-COUNT-29: "xllvm.intr.aie4.insert.staging.bf16.I1024"
// CHECK: vector.extract %[[BT]][62]
// CHECK: vector.extract %[[BT]][63]
// CHECK: %[[I31:.*]] = arith.constant 31 : i32
// CHECK: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%{{.*}}, %[[I31]])
// CHECK-NOT: insert.staging
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
// CHECK-NOT: insert.staging
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[C]], %[[A]]) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
// CHECK: return %[[R]]
// CHECK-NOT: aievec.

// A row-major B keeps the transpose.
// CANON-LABEL: func.func @bf16
// CANON: vector.transpose %arg1, [1, 0] : vector<32x64xbf16> to vector<64x32xbf16>

// A transposed producer (B stored column-major) cancels the transpose, so
// each insert concatenates two stored columns directly.
func.func @bf16(%a: vector<32xbf16>, %b: vector<32x64xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    %x = aievec.aie4.mmac bf16 %a, %acc : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// CANON-LABEL: func.func @bf16_column_major
// CANON-SAME: (%{{.*}}: vector<32xbf16>, %[[BT:.*]]: vector<64x32xbf16>,
// CANON-NOT: vector.transpose
// CANON: %[[L0:.*]] = vector.extract %[[BT]][0] : vector<32xbf16> from vector<64x32xbf16>
// CANON: %[[H0:.*]] = vector.extract %[[BT]][1] : vector<32xbf16> from vector<64x32xbf16>
// CANON: %[[K0:.*]] = vector.shuffle %[[L0]], %[[H0]]
// CANON: "xllvm.intr.aie4.insert.staging.bf16.I1024"(%[[K0]], %{{.*}})
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

// Without an accumulator the MMAC sets zero_acc and passes a poison acc.
// CHECK-LABEL: func.func @bf16_zero_acc
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"
// CHECK: %[[P:.*]] = llvm.mlir.poison : vector<64xf32>
// CHECK: "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%[[P]], %{{.*}}) <{zeroAcc}>
func.func @bf16_zero_acc(%a: vector<32xbf16>, %b: vector<32x64xbf16>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// int8: crIntMMode int8 (3) for signed B, uint8 (2) for unsigned B; bmode
// int8x8 (1); X and Z signedness from the op.
// CHECK-LABEL: func.func @int8_signed_b
// CHECK: %[[L0:.*]] = vector.extract %{{.*}}[0] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[H0:.*]] = vector.extract %{{.*}}[1] : vector<64xi8> from vector<64x64xi8>
// CHECK: %[[K0:.*]] = vector.shuffle %[[L0]], %[[H0]] [0, 1, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[K0]], %{{.*}}) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-31: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 3 : i32}> : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I32768"
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%{{.*}}, %{{.*}}) <{bmode = 1 : i32, xSign = 0 : i32, zSign = 1 : i32}> : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
func.func @int8_signed_b(%a: vector<64xi8>, %b: vector<64x64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = true} {
  ^bb0(%acc: vector<64xi32>):
    %x = aievec.aie4.mmac int8 %a, %acc {lhs_signed = false, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// CHECK-LABEL: func.func @int8_unsigned_b
// CHECK-COUNT-32: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 2 : i32}>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%{{.*}}, %{{.*}}) <{bmode = 1 : i32, xSign = 1 : i32, zSign = 0 : i32}>
func.func @int8_unsigned_b(%a: vector<64xi8>, %b: vector<64x64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = false} {
  ^bb0(%acc: vector<64xi32>):
    %x = aievec.aie4.mmac int8 %a, %acc {lhs_signed = true, rhs_signed = false} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// int4: each 128-element column and the A row pack to bytes, element 2j in
// the low nibble of byte j (vector.bitcast); crIntMMode int4 (1) / uint4
// (0); the I4 commit; bmode int4x4 (0). The staging mode and the MMAC Z
// sign (conf bit 8) must agree:
// both follow rhs_signed, signed here (mode 1, zSign 1) and unsigned in
// @int4_unsigned (mode 0, zSign 0).
// CHECK-LABEL: func.func @int4
// CHECK-SAME: (%[[A:.*]]: vector<128xi4>,
// CHECK: %[[L0:.*]] = vector.extract %{{.*}}[0] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[L0B:.*]] = vector.bitcast %[[L0]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[H0:.*]] = vector.extract %{{.*}}[1] : vector<128xi4> from vector<64x128xi4>
// CHECK: %[[H0B:.*]] = vector.bitcast %[[H0]] : vector<128xi4> to vector<64xi8>
// CHECK: %[[K0:.*]] = vector.shuffle %[[L0B]], %[[H0B]] [0, 1, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.int.I1024"(%[[K0]], %{{.*}}) <{mode = 1 : i32}>
// CHECK-COUNT-31: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 1 : i32}>
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I4.I32768"
// CHECK: %[[AB:.*]] = vector.bitcast %[[A]] : vector<128xi4> to vector<64xi8>
// CHECK: "xllvm.intr.aie4.I512.ACC2048.mmac"(%{{.*}}, %[[AB]]) <{bmode = 0 : i32, xSign = 1 : i32, zSign = 1 : i32}>
func.func @int4(%a: vector<128xi4>, %b: vector<128x64xi4>, %c: vector<64xi32>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int4 %b : vector<128x64xi4>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = true} {
  ^bb0(%acc: vector<64xi32>):
    %x = aievec.aie4.mmac int4 %a, %acc {lhs_signed = true, rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// CHECK-LABEL: func.func @int4_unsigned
// CHECK-COUNT-32: "xllvm.intr.aie4.insert.staging.int.I1024"(%{{.*}}, %{{.*}}) <{mode = 0 : i32}>
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I4.I32768"
// CHECK: <{bmode = 0 : i32, xSign = 0 : i32, zSign = 0 : i32, zeroAcc}>
func.func @int4_unsigned(%a: vector<128xi4>, %b: vector<128x64xi4>) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int4 %b : vector<128x64xi4> -> vector<64xi32> attributes {rhs_signed = false} {
    %x = aievec.aie4.mmac int4 %a {lhs_signed = false, rhs_signed = false} : vector<128xi4> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// fp8: E4M3 values are bitcast to bytes; no staging mode; the FP commit;
// emode 0 (dense).
// CHECK-LABEL: func.func @fp8
// CHECK-SAME: (%[[A:.*]]: vector<64xf8E4M3FN>,
// CHECK: %[[L0:.*]] = vector.extract %{{.*}}[0] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[L0B:.*]] = vector.bitcast %[[L0]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[H0:.*]] = vector.extract %{{.*}}[1] : vector<64xf8E4M3FN> from vector<64x64xf8E4M3FN>
// CHECK: %[[H0B:.*]] = vector.bitcast %[[H0]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: %[[K0:.*]] = vector.shuffle %[[L0B]], %[[H0B]] [0, 1, {{.*}}, 126, 127] : vector<64xi8>, vector<64xi8>
// CHECK: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%[[K0]], %{{.*}}) : (vector<128xi8>, i32) -> ()
// CHECK-COUNT-31: "xllvm.intr.aie4.insert.staging.fp8.I1024"(%{{.*}}, %{{.*}}) : (vector<128xi8>, i32) -> ()
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.FP32768"
// CHECK: %[[AB:.*]] = vector.bitcast %[[A]] : vector<64xf8E4M3FN> to vector<64xi8>
// CHECK: "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%{{.*}}, %[[AB]]) <{emode = 0 : i32}>
func.func @fp8(%a: vector<64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c: vector<64xf32>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    %x = aievec.aie4.mmac fp8 %a, %acc : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// The semantic op is expanded and lowered by the same pass.
// CHECK-LABEL: func.func @matmul
// CHECK-COUNT-32: insert.staging.bf16.I1024
// CHECK-NEXT: "xllvm.intr.aie4.staging.to.matrix.I32768"
// CHECK-COUNT-2: "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"
// CHECK-NOT: aievec.
func.func @matmul(%a: vector<2x32xbf16>, %b: vector<32x64xbf16>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %0 = aievec.aie4.matmul %a, %b, %c : vector<2x32xbf16>, vector<32x64xbf16> into vector<2x64xf32>
  return %0 : vector<2x64xf32>
}

// -----

// A yielded block argument is replaced by its init.
// CHECK-LABEL: func.func @yield_block_arg
// CHECK-SAME: (%{{.*}}: vector<32x64xbf16>, %[[C:.*]]: vector<64xf32>)
// CHECK: "xllvm.intr.aie4.staging.to.matrix.I32768"
// CHECK-NEXT: return %[[C]]
func.func @yield_block_arg(%b: vector<32x64xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    aievec.aie4.matrix_yield %acc : vector<64xf32>
  }
  return %r : vector<64xf32>
}
