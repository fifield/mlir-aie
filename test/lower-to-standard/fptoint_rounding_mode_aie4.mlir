//===- fptoint_rounding_mode_aie4.mlir - AIE4 float->int rounding ---------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// AIE4's float->int conversion rounds by crRnd, and arith.fptosi/fptoui mean
// truncation toward zero, so a core containing aievec.aie4.conv_f32_i32 sets
// crRnd = 2 at entry instead of the default 12. A core that also needs 12, or
// reads the ambient crRnd, is rejected.
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering --split-input-file --verify-diagnostics %s | FileCheck %s

// Only the crRnd value changes; crVecRnd stays 12 and nothing else is written.
// CHECK-LABEL: module @fptoint_only
// CHECK:       func.func @core_0_2() {
// CHECK-NEXT:    %[[R0:.*]] = arith.constant 24 : i32
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg(%[[R0]], %[[V0]]) : (i32, i32) -> ()
// CHECK-NEXT:    %[[R1:.*]] = arith.constant 46 : i32
// CHECK-NEXT:    %[[V1:.*]] = arith.constant 2 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg(%[[R1]], %[[V1]]) : (i32, i32) -> ()
// CHECK-NOT:     call @llvm.aie4.set.ctrl.reg
// CHECK:         aievec.aie4.conv_f32_i32
module @fptoint_only {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %src = arith.constant dense<1.5> : vector<32xf32>
      %res = aievec.aie4.conv_f32_i32 %src : vector<32xf32>, vector<32xi32>
      vector.store %res, %b[%c0] : memref<32xi32>, vector<32xi32>
      aie.end
    }
  }
}

// -----

// f32 arithmetic reads neither mode, so it does not conflict.
// CHECK-LABEL: module @fptoint_with_f32
// CHECK:       arith.constant 46 : i32
// CHECK-NEXT:  arith.constant 2 : i32
module @fptoint_with_f32 {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %x = arith.constant dense<1.5> : vector<32xf32>
      %y = arith.addf %x, %x : vector<32xf32>
      %res = aievec.aie4.conv_f32_i32 %y : vector<32xf32>, vector<32xi32>
      vector.store %res, %b[%c0] : memref<32xi32>, vector<32xi32>
      aie.end
    }
  }
}

// -----

// A conversion in a called function counts as the core's.
// CHECK-LABEL: module @fptoint_in_callee
// CHECK:       func.func @core_0_2() {
// CHECK:         arith.constant 46 : i32
// CHECK-NEXT:    arith.constant 2 : i32
module @fptoint_in_callee {
  aie.device(npu3) {
    func.func @convert(%x: vector<32xf32>) -> vector<32xi32> {
      %r = aievec.aie4.conv_f32_i32 %x : vector<32xf32>, vector<32xi32>
      return %r : vector<32xi32>
    }
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %x = arith.constant dense<1.5> : vector<32xf32>
      %r = func.call @convert(%x) : (vector<32xf32>) -> vector<32xi32>
      vector.store %r, %b[%c0] : memref<32xi32>, vector<32xi32>
      aie.end
    }
  }
}

// -----

// A core without the conversion keeps the default.
// CHECK-LABEL: module @no_fptoint
// CHECK:       arith.constant 46 : i32
// CHECK-NEXT:  arith.constant 12 : i32
module @no_fptoint {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xf32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %x = arith.constant dense<1.5> : vector<32xf32>
      %y = arith.mulf %x, %x : vector<32xf32>
      vector.store %y, %b[%c0] : memref<32xf32>, vector<32xf32>
      aie.end
    }
  }
}

// -----

module @fptoint_with_bf16 {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xbf16>
    // expected-error@+3 {{'aie.core' op mixes a float -> int conversion, which needs crRnd = 2 (toward zero), with an op that needs round-to-nearest-even (crRnd = 12)}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %x = arith.constant dense<1.5> : vector<32xf32>
      // expected-note@+1 {{needs crRnd = 2}}
      %r = aievec.aie4.conv_f32_i32 %x : vector<32xf32>, vector<32xi32>
      vector.store %r, %b[%c0] : memref<32xi32>, vector<32xi32>
      %a = memref.load %h[%c0] : memref<32xbf16>
      // expected-note@+1 {{needs crRnd = 12}}
      %s = arith.addf %a, %a : bf16
      memref.store %s, %h[%c0] : memref<32xbf16>
      aie.end
    }
  }
}

// -----

module @fptoint_with_ambient {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xbf16>
    // expected-error@+3 {{'aie.core' op mixes a float -> int conversion, which needs crRnd = 2 (toward zero), with an op that reads the ambient crRnd}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %x = arith.constant dense<1.5> : vector<32xf32>
      // expected-note@+1 {{needs crRnd = 2}}
      %r = aievec.aie4.conv_f32_i32 %x : vector<32xf32>, vector<32xi32>
      vector.store %r, %b[%c0] : memref<32xi32>, vector<32xi32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %n = aievec.aie4.conv_f32_bf16 %x : vector<32xf32>, vector<32xbf16>
      vector.store %n, %h[%c0] : memref<32xbf16>, vector<32xbf16>
      aie.end
    }
  }
}
