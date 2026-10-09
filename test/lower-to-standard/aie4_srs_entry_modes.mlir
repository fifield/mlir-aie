//===- aie4_srs_entry_modes.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Explicit SRS modes are written once at core entry; omitted attributes
// preserve ambient modes. Functions containing explicit-mode SRS receive
// `aie4.entry_modes` to carry the guarantee through outlining.
//
// aie.use_lock needs aie-localize-locks first, as in aiecc's core pipeline.
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering --split-input-file %s | FileCheck %s
// Entry writes produced by standard lowering must remain accepted when the
// assumption option is also present. The entry attribute stays authoritative.
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering --split-input-file %s -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering --split-input-file %s -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:9,sat:1" | FileCheck %s --check-prefix=LLVM
// LLVM-LABEL: module @posinf_sat
// LLVM: call @llvm.aie4.set.ctrl.reg
// LLVM: "xllvm.intr.aie4.I512.v64.acc32.srs"
// LLVM-NOT: aievec.aie4.srs
// The per-core flow is in aie4_srs_entry_modes_percore.mlir.

// Both modes explicit: crVecRnd, crRnd = 9, crSat = 1, in that order.
// CHECK-LABEL: module @posinf_sat
// CHECK:       func.func @core_0_2() attributes {aie4.entry_modes = {rnd = 9 : i32, sat = 1 : i32}} {
// CHECK-NEXT:    %[[R0:.*]] = arith.constant 24 : i32
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg(%[[R0]], %[[V0]]) : (i32, i32) -> ()
// CHECK-NEXT:    %[[R1:.*]] = arith.constant 46 : i32
// CHECK-NEXT:    %[[V1:.*]] = arith.constant 9 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg(%[[R1]], %[[V1]]) : (i32, i32) -> ()
// CHECK-NEXT:    %[[R2:.*]] = arith.constant 45 : i32
// CHECK-NEXT:    %[[V2:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg(%[[R2]], %[[V2]]) : (i32, i32) -> ()
// CHECK-NOT:     call @llvm.aie4.set.ctrl.reg
// CHECK:         aievec.aie4.srs
module @posinf_sat {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<64xi8>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      vector.store %y, %b[%c0] : memref<64xi8>, vector<64xi8>
      aie.end
    }
  }
}

// -----

// Rounding only: the op reads the ambient crSat, so crSat is not written.
// CHECK-LABEL: module @rnd_only
// CHECK:       func.func @core_0_2() attributes {aie4.entry_modes = {rnd = 9 : i32}} {
// CHECK:         arith.constant 46 : i32
// CHECK-NEXT:    arith.constant 9 : i32
// CHECK-NOT:     arith.constant 45 : i32
// CHECK:         aievec.aie4.srs
module @rnd_only {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<64xi8>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      vector.store %y, %b[%c0] : memref<64xi8>, vector<64xi8>
      aie.end
    }
  }
}

// -----

// Saturation only: the op reads the ambient crRnd, so crRnd is not written.
// CHECK-LABEL: module @sat_only
// CHECK:       func.func @core_0_2() attributes {aie4.entry_modes = {sat = 1 : i32}} {
// CHECK-NEXT:    arith.constant 24 : i32
// CHECK-NEXT:    arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg
// CHECK-NEXT:    arith.constant 45 : i32
// CHECK-NEXT:    arith.constant 1 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg
// CHECK-NOT:     call @llvm.aie4.set.ctrl.reg
// CHECK:         aievec.aie4.srs
module @sat_only {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<32xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi16>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<32xi32>, vector<32xi32>
      %y = aievec.aie4.srs %v {shift = 3 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<32xi32> -> vector<32xi16>
      vector.store %y, %b[%c0] : memref<32xi16>, vector<32xi16>
      aie.end
    }
  }
}

// -----

// An explicit conv_even agrees with the generic bf16 rounding (crRnd = 12).
// CHECK-LABEL: module @conv_even_truncf
// CHECK:       func.func @core_0_2() attributes {aie4.entry_modes = {rnd = 12 : i32}} {
// CHECK:         arith.constant 46 : i32
// CHECK-NEXT:    arith.constant 12 : i32
// CHECK-NOT:     arith.constant 45 : i32
// CHECK:         arith.truncf
module @conv_even_truncf {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<64xi8>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<16xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<16xbf16>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<conv_even>} : vector<64xi32> -> vector<64xi8>
      vector.store %y, %b[%c0] : memref<64xi8>, vector<64xi8>
      %x = memref.load %f[%c0] : memref<16xf32>
      %z = arith.truncf %x : f32 to bf16
      memref.store %z, %h[%c0] : memref<16xbf16>
      aie.end
    }
  }
}

// -----

// An explicit sym_floor (2) agrees with a float -> int conversion.
// CHECK-LABEL: module @sym_floor_fptosi
// CHECK:       func.func @core_0_2() attributes {aie4.entry_modes = {rnd = 2 : i32}} {
// CHECK:         arith.constant 46 : i32
// CHECK-NEXT:    arith.constant 2 : i32
// CHECK:         aievec.aie4.conv_f32_i32
module @sym_floor_fptosi {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<64xi8>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %i = aie.buffer(%t) {sym_name = "i"} : memref<32xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<sym_floor>} : vector<64xi32> -> vector<64xi8>
      vector.store %y, %b[%c0] : memref<64xi8>, vector<64xi8>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      %n = aievec.aie4.conv_f32_i32 %x : vector<32xf32>, vector<32xi32>
      vector.store %n, %i[%c0] : memref<32xi32>, vector<32xi32>
      aie.end
    }
  }
}

// -----

// A shift-0 UPS cannot overflow, so it does not read crSat; an aie.use_lock
// becomes llvm.aie4.acquire/release, which do not write the modes.
// CHECK-LABEL: module @sat_with_ups0_and_lock
// CHECK:       func.func @core_0_2() attributes {aie4.entry_modes = {rnd = 9 : i32, sat = 1 : i32}} {
// CHECK:         arith.constant 45 : i32
// CHECK-NEXT:    arith.constant 1 : i32
// CHECK:         call @llvm.aie4.acquire
// CHECK:         aievec.aie4.ups_i16_i32
module @sat_with_ups0_and_lock {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 1 : i32}
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<64xi8>
    %s = aie.buffer(%t) {sym_name = "s"} : memref<32xi16>
    %w = aie.buffer(%t) {sym_name = "w"} : memref<32xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %c1)
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      vector.store %y, %b[%c0] : memref<64xi8>, vector<64xi8>
      %x = vector.load %s[%c0] : memref<32xi16>, vector<32xi16>
      %u = aievec.aie4.ups_i16_i32 %x {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
      vector.store %u, %w[%c0] : memref<32xi32>, vector<32xi32>
      aie.use_lock(%l, Release, %c1)
      aie.end
    }
  }
}

// -----

// An explicit op in a device-level callee: the callee carries the attribute,
// the core function does not, and the writes are in the core.
// CHECK-LABEL: module @in_callee
// CHECK:       func.func @narrow(%{{.*}}: vector<64xi32>) -> vector<64xi8> attributes {aie4.entry_modes = {rnd = 9 : i32, sat = 1 : i32}} {
// CHECK:       func.func @core_0_2() {
// CHECK:         arith.constant 46 : i32
// CHECK-NEXT:    arith.constant 9 : i32
// CHECK:         arith.constant 45 : i32
// CHECK-NEXT:    arith.constant 1 : i32
// CHECK:         call @narrow
module @in_callee {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<64xi8>
    func.func @narrow(%v: vector<64xi32>) -> vector<64xi8> {
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      return %y : vector<64xi8>
    }
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %y = func.call @narrow(%v) : (vector<64xi32>) -> vector<64xi8>
      vector.store %y, %b[%c0] : memref<64xi8>, vector<64xi8>
      aie.end
    }
  }
}

// -----

// No readers at all: crRnd = 12 as before, no crSat write, no attribute.
// CHECK-LABEL: module @no_readers
// CHECK-NOT:   aie4.entry_modes
// CHECK:       func.func @core_0_2() {
// CHECK-NEXT:    arith.constant 24 : i32
// CHECK-NEXT:    arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg
// CHECK-NEXT:    arith.constant 46 : i32
// CHECK-NEXT:    arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg
// CHECK-NOT:     arith.constant 45 : i32
// CHECK-NOT:     aie4.entry_modes
// CHECK:         return
module @no_readers {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<16xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      memref.store %v, %b[%c0] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Ambient crSat readers alone never cause a crSat write.
// CHECK-LABEL: module @ambient_sat_readers
// CHECK:       func.func @core_0_2() {
// CHECK-NOT:     arith.constant 45 : i32
// CHECK:         aievec.aie4.srs_i32_i16
// CHECK:         aievec.aie4.ups_i16_i32
// CHECK:         aievec.aie4.srs {{.*}}isSigned = true, shift = 4 : i32
module @ambient_sat_readers {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<32xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi16>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<32xi32>, vector<32xi32>
      %y = aievec.aie4.srs_i32_i16 %v {shift = 4 : i32} : vector<32xi32>, vector<32xi16>
      %u = aievec.aie4.ups_i16_i32 %y {shift = 4 : i32} : vector<32xi16>, vector<32xi32>
      %z = aievec.aie4.srs %u {shift = 4 : i32, isSigned = true} : vector<32xi32> -> vector<32xi16>
      vector.store %z, %b[%c0] : memref<32xi16>, vector<32xi16>
      aie.end
    }
  }
}

