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

// -----

// AIE4 UPS -> one XLLVM intrinsic op. Like SRS, shift/sign are trailing
// attribute-resident arguments of `llvm.aie4.acc32.v32.I512.ups`, not opaque
// operands.
func.func @ups_i16_i32(%src : vector<32xi16>) -> vector<32xi32> {
  %0 = aievec.aie4.ups_i16_i32 %src {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @ups_i16_i32
// CHECK-SAME: %[[SRC:.*]]: vector<32xi16>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.acc32.v32.I512.ups"(
// CHECK-SAME: %[[SRC]]
// CHECK-SAME: shift = 0 : i32
// CHECK-SAME: sign = 1 : i32
// CHECK-SAME: (vector<32xi16>) -> vector<32xi32>
// CHECK: return %[[R]] : vector<32xi32>

// -----

// The unsigned instruction form (`upssign0`) is reachable and distinct.
func.func @ups_unsigned(%src : vector<32xi16>) -> vector<32xi32> {
  %0 = aievec.aie4.ups_i16_i32 %src {shift = 0 : i32, isSigned = false} : vector<32xi16>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @ups_unsigned
// CHECK: "xllvm.intr.aie4.acc32.v32.I512.ups"(
// CHECK-SAME: shift = 0 : i32
// CHECK-SAME: sign = 0 : i32

// -----

// AIE4 bf16 -> f32 accumulator widening conv -> one XLLVM intrinsic op.
//
// Unlike SRS/UPS, this instruction takes no config word at all -- no shift,
// no sign, no packed attributes -- so the conversion pattern forwards the
// single operand directly with no attribute plumbing. It reads no control
// register (`AIE4GenInstrInfo.td`'s `VCONV_FP32_BF16` variant has no `Uses`
// list), which is exactly why `AIEVec_AIE4ConvBF16F32Op` is marked `Pure`
// while its narrowing counterpart below is not.
func.func @conv_bf16_f32(%src : vector<32xbf16>) -> vector<32xf32> {
  %0 = aievec.aie4.conv_bf16_f32 %src : vector<32xbf16>, vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @conv_bf16_f32
// CHECK-SAME: %[[SRC:.*]]: vector<32xbf16>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.v32bf16.to.v32accfloat"(%[[SRC]])
// CHECK-SAME: (vector<32xbf16>) -> vector<32xf32>
// CHECK: return %[[R]] : vector<32xf32>

// -----

// AIE4 f32 accumulator -> bf16 narrowing conv -> one XLLVM intrinsic op.
//
// Explicit narrowing uses ambient rounding and saturation modes.
func.func @conv_f32_bf16(%src : vector<32xf32>) -> vector<32xbf16> {
  %0 = aievec.aie4.conv_f32_bf16 %src : vector<32xf32>, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @conv_f32_bf16
// CHECK-SAME: %[[SRC:.*]]: vector<32xf32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.v32accfloat.to.v32bfloat16"(%[[SRC]])
// CHECK-SAME: (vector<32xf32>) -> vector<32xbf16>
// CHECK: return %[[R]] : vector<32xbf16>
