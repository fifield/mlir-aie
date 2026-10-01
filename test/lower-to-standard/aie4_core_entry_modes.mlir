//===- aie4_core_entry_modes.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// D19: AIE4 cores write their rounding controls at entry instead of relying on
// the loader. The ISA resets crVecRnd (id 24) and crRnd (id 46) to 0 (floor);
// on NPU3 the loader sets crVecRnd = 12 but leaves crRnd at 0.
//
// - crVecRnd = 12 (RNE) in every AIE4 core.
// - crRnd = 12 in every AIE4 core, except one that contains an explicit
//   aievec.aie4.srs_i32_i16 / conv_f32_bf16 op, which reads the ambient
//   crRnd. Generic bf16/f16 rounding (arith arithmetic, int -> float,
//   f32 truncation, math), in the core or in a function it calls, needs 12;
//   combining it with an explicit ambient op is rejected.
// - With crRnd = 12 guaranteed, arith.truncf's to_nearest_even attribute is
//   dropped (its constrained-intrinsic form does not compile on Peano).
//
// RUN: aie-opt --aie-standard-lowering --split-input-file --verify-diagnostics %s | FileCheck %s

// Both writes open every ordinary AIE4 core, even one without float code.
// CHECK-LABEL: module @aie4_plain
// CHECK:       func.func @core_0_2() {
// CHECK-NEXT:    %[[R0:.*]] = arith.constant 24 : i32
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg(%[[R0]], %[[V0]]) : (i32, i32) -> ()
// CHECK-NEXT:    %[[R1:.*]] = arith.constant 46 : i32
// CHECK-NEXT:    %[[V1:.*]] = arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg(%[[R1]], %[[V1]]) : (i32, i32) -> ()
// CHECK-NOT:     call @llvm.aie4.set.ctrl.reg
// CHECK:         return
module @aie4_plain {
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

// A to_nearest_even truncation keeps its meaning but loses the attribute.
// CHECK-LABEL: module @aie4_truncf_rne
// CHECK:       func.func @core_0_2() {
// CHECK:         arith.constant 46 : i32
// CHECK:         arith.truncf %{{[0-9]+}} : f32 to bf16
// CHECK-NOT:     to_nearest_even
module @aie4_truncf_rne {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %f = aie.buffer(%t) {sym_name = "f"} : memref<16xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<16xbf16>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %x = memref.load %f[%c0] : memref<16xf32>
      %y = arith.truncf %x to_nearest_even : f32 to bf16
      memref.store %y, %h[%c0] : memref<16xbf16>
      aie.end
    }
  }
}

// -----

// Explicit ambient-mode ops keep the ambient crRnd: only crVecRnd is written.
// CHECK-LABEL: module @aie4_explicit_srs
// CHECK:       func.func @core_0_2() {
// CHECK-NEXT:    arith.constant 24 : i32
// CHECK-NEXT:    arith.constant 12 : i32
// CHECK-NEXT:    call @llvm.aie4.set.ctrl.reg
// CHECK-NOT:     arith.constant 46 : i32
// CHECK:         aievec.aie4.srs_i32_i16
module @aie4_explicit_srs {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xi16>
    aie.core(%t) {
      %acc = arith.constant dense<100> : vector<32xi32>
      %n = aievec.aie4.srs_i32_i16 %acc {shift = 2 : i32} : vector<32xi32>, vector<32xi16>
      %c0 = arith.constant 0 : index
      vector.store %n, %b[%c0] : memref<32xi16>, vector<32xi16>
      aie.end
    }
  }
}

// -----

// Generic rounding readers mixed with an explicit ambient op are rejected.
// Each core pairs aievec.aie4.conv_f32_bf16 with one generic reader.

// Scalar bf16 add (Peano: f32 add, then vconv.bf16.fp32).
module @conflict_bf16_add {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xbf16>
    // expected-error@+3 {{'aie.core' op mixes ops that need round-to-nearest-even (crRnd = 12) with an explicit op that reads the ambient crRnd}}
    // expected-note@+2 {{split them into separate cores, or use aievec.aie4.conv_f32_bf16 if floor rounding is acceptable; explicit ops will get a rounding attribute (bead aie4-r2-explicit-rnd)}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %p = arith.constant 0.0 : f32
      %v = vector.transfer_read %f[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %e = aievec.aie4.conv_f32_bf16 %v : vector<32xf32>, vector<32xbf16>
      vector.store %e, %h[%c0] : memref<32xbf16>, vector<32xbf16>
      %a = memref.load %h[%c0] : memref<32xbf16>
      // expected-note@+1 {{needs crRnd = 12}}
      %s = arith.addf %a, %a : bf16
      memref.store %s, %h[%c0] : memref<32xbf16>
      aie.end
    }
  }
}

// -----

// int -> bf16.
module @conflict_sitofp {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xbf16>
    %i = aie.buffer(%t) {sym_name = "i"} : memref<16xi32>
    // expected-error@+3 {{mixes ops that need round-to-nearest-even}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %p = arith.constant 0.0 : f32
      %v = vector.transfer_read %f[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %e = aievec.aie4.conv_f32_bf16 %v : vector<32xf32>, vector<32xbf16>
      vector.store %e, %h[%c0] : memref<32xbf16>, vector<32xbf16>
      %n = memref.load %i[%c0] : memref<16xi32>
      // expected-note@+1 {{needs crRnd = 12}}
      %s = arith.sitofp %n : i32 to bf16
      memref.store %s, %h[%c0] : memref<32xbf16>
      aie.end
    }
  }
}

// -----

// A truncation in a device-level function the core calls.
module @conflict_callee {
  aie.device(npu3) {
    func.func @cvt(%x: f32) -> bf16 {
      // expected-note@+1 {{needs crRnd = 12}}
      %y = arith.truncf %x : f32 to bf16
      return %y : bf16
    }
    %t = aie.tile(0, 2)
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xbf16>
    // expected-error@+3 {{mixes ops that need round-to-nearest-even}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %p = arith.constant 0.0 : f32
      %v = vector.transfer_read %f[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %e = aievec.aie4.conv_f32_bf16 %v : vector<32xf32>, vector<32xbf16>
      vector.store %e, %h[%c0] : memref<32xbf16>, vector<32xbf16>
      %x = memref.load %f[%c0] : memref<32xf32>
      %y = func.call @cvt(%x) : (f32) -> bf16
      memref.store %y, %h[%c0] : memref<32xbf16>
      aie.end
    }
  }
}

// -----

// Scalar f16 multiply.
module @conflict_f16_mul {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xbf16>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xf16>
    // expected-error@+3 {{mixes ops that need round-to-nearest-even}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %p = arith.constant 0.0 : f32
      %v = vector.transfer_read %f[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %e = aievec.aie4.conv_f32_bf16 %v : vector<32xf32>, vector<32xbf16>
      vector.store %e, %h[%c0] : memref<32xbf16>, vector<32xbf16>
      %a = memref.load %g[%c0] : memref<16xf16>
      // expected-note@+1 {{needs crRnd = 12}}
      %s = arith.mulf %a, %a : f16
      memref.store %s, %g[%c0] : memref<16xf16>
      aie.end
    }
  }
}

// -----

// A math op with a bf16 result.
module @conflict_math {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xbf16>
    // expected-error@+3 {{mixes ops that need round-to-nearest-even}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %p = arith.constant 0.0 : f32
      %v = vector.transfer_read %f[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %e = aievec.aie4.conv_f32_bf16 %v : vector<32xf32>, vector<32xbf16>
      vector.store %e, %h[%c0] : memref<32xbf16>, vector<32xbf16>
      %a = memref.load %h[%c0] : memref<32xbf16>
      // expected-note@+1 {{needs crRnd = 12}}
      %s = math.sqrt %a : bf16
      memref.store %s, %h[%c0] : memref<32xbf16>
      aie.end
    }
  }
}

// -----

// A truncation that asks for a non-RNE rounding mode cannot be honored.
module @aie4_other_rounding {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<16xbf16>
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %x = arith.constant 1.5 : f32
      // expected-error@+1 {{requests a rounding mode other than to_nearest_even}}
      %y = arith.truncf %x downward : f32 to bf16
      %c0 = arith.constant 0 : index
      memref.store %y, %b[%c0] : memref<16xbf16>
      aie.end
    }
  }
}

// -----

// AIE2P is unchanged: no AIE4 control writes.
// CHECK-LABEL: module @aie2p_unchanged
// CHECK:       func.func @core_0_2() {
// CHECK-NOT:   llvm.aie4.set.ctrl.reg
// CHECK-NOT:   set.ctrl.reg
// CHECK:       return
module @aie2p_unchanged {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %f = aie.buffer(%t) {sym_name = "f"} : memref<16xf32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<16xbf16>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %x = memref.load %f[%c0] : memref<16xf32>
      %y = arith.truncf %x : f32 to bf16
      memref.store %y, %h[%c0] : memref<16xbf16>
      aie.end
    }
  }
}
