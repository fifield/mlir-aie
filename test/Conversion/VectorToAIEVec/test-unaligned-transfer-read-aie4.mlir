//===- test-unaligned-transfer-read-aie4.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Canonicalization widens the unaligned read and extracts the requested lanes.
// AIE4 conversion must keep the resulting slice legal for generic lowering.

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" \
// RUN:   --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// CHECK-LABEL: func.func @unaligned_read(
// CHECK: vector.transfer_read %{{.*}} : memref<64xi32>, vector<32xi32>
// CHECK: vector.extract_strided_slice %{{.*}} offsets = [3], sizes = [16], strides = [1] : vector<32xi32> to vector<16xi32>
func.func @unaligned_read(%mem: memref<64xi32>) -> vector<16xi32> {
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  %v = vector.transfer_read %mem[%c3], %c0 {in_bounds = [true]} : memref<64xi32>, vector<16xi32>
  return %v : vector<16xi32>
}
