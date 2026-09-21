// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// Lower MAC to XLLVM with integer mode, 16x16 block mode, and signed inputs
// (config 0x308). The zeroAcc UnitAttr must be absent to retain the accumulator.
func.func @mac_ewise_i16_i32(%a : vector<32xi16>, %b : vector<32xi16>,
                             %acc : vector<32xi32>) -> vector<32xi32> {
  %0 = aievec.aie4.mac_ewise_i16_i32 %a, %b, %acc : vector<32xi16>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @mac_ewise_i16_i32
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>, %[[ACC:.*]]: vector<32xi32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"(
// CHECK-SAME: %[[A]], %[[B]], %[[ACC]]
// CHECK-SAME: amode = 0 : i32
// CHECK-SAME: bmode = 1 : i32
// CHECK-SAME: lhsSign = 1 : i32
// CHECK-SAME: rhsSign = 1 : i32
// CHECK-NOT: zeroAcc
// CHECK-SAME: (vector<32xi16>, vector<32xi16>, vector<32xi32>) -> vector<32xi32>
// CHECK: return %[[R]] : vector<32xi32>

// -----

// Preserve the SRS shift and signedness as verified XLLVM attributes.
func.func @srs_i32_i16(%acc : vector<32xi32>) -> vector<32xi16> {
  %0 = aievec.aie4.srs_i32_i16 %acc {shift = 4 : i32} : vector<32xi32>, vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: @srs_i32_i16
// CHECK-SAME: %[[ACC:.*]]: vector<32xi32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.I512.v32.acc32.srs"(
// CHECK-SAME: %[[ACC]]
// CHECK-SAME: shift = 4 : i32
// CHECK-SAME: sign = 1 : i32
// CHECK-SAME: (vector<32xi32>) -> vector<32xi16>
// CHECK: return %[[R]] : vector<32xi16>

// -----

// The unsigned instruction form (`srssign0`) is reachable and distinct.
func.func @srs_unsigned(%acc : vector<32xi32>) -> vector<32xi16> {
  %0 = aievec.aie4.srs_i32_i16 %acc {shift = 0 : i32, isSigned = false} : vector<32xi32>, vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: @srs_unsigned
// CHECK: "xllvm.intr.aie4.I512.v32.acc32.srs"(
// CHECK-SAME: shift = 0 : i32
// CHECK-SAME: sign = 0 : i32
