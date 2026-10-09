// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Low-level SRS/UPS/conversion readers retain their ambient mode contracts,
// including when reached through an internal helper.
// RUN: aie-opt --aie-standard-lowering --convert-aievec-to-llvm="aie-target=aie4b" --split-input-file --verify-diagnostics %s

module @xllvm_srs_helper {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @raw_helper(%v: vector<64xi32>) -> vector<64xi8> {
      // expected-note@+1 {{reads the ambient crRnd}}
      %raw = "xllvm.intr.aie4.I512.v64.acc32.srs"(%v) {shift = 1 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
      return %raw : vector<64xi8>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with an op that reads the ambient crRnd}}
    // expected-note@+2 {{give every crRnd reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %v = arith.constant dense<0> : vector<64xi32>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v) : (vector<64xi32>) -> vector<64xi8>
      aie.end
    }
  }
}

// -----

module @func_srs {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @llvm.aie4.I512.v64.acc32.srs(vector<64xi32>, i32, i32) -> vector<64xi8>
    func.func private @raw_helper(%v: vector<64xi32>) -> vector<64xi8> {
      %one = arith.constant 1 : i32
      // expected-note@+1 {{reads the ambient crRnd}}
      %raw = func.call @llvm.aie4.I512.v64.acc32.srs(%v, %one, %one) : (vector<64xi32>, i32, i32) -> vector<64xi8>
      return %raw : vector<64xi8>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with an op that reads the ambient crRnd}}
    // expected-note@+2 {{give every crRnd reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %v = arith.constant dense<0> : vector<64xi32>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v) : (vector<64xi32>) -> vector<64xi8>
      aie.end
    }
  }
}

// -----

module @llvm_intrinsic_srs {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @raw_helper(%v: vector<32xi32>) -> vector<32xi16> {
      %one = arith.constant 1 : i32
      // expected-note@+1 {{reads the ambient crSat}}
      %raw = llvm.call_intrinsic "llvm.aie4.I512.v32.acc32.srs"(%v, %one, %one) : (vector<32xi32>, i32, i32) -> vector<32xi16>
      return %raw : vector<32xi16>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crSat = 1 (saturate), with an op that reads the ambient crSat}}
    // expected-note@+2 {{give every crSat reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %v = arith.constant dense<0> : vector<32xi32>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crSat = 1 (saturate)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v) : (vector<32xi32>) -> vector<32xi16>
      aie.end
    }
  }
}

// -----

module @xllvm_ups_nonzero {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @raw_helper(%v: vector<32xi16>) -> vector<32xi32> {
      // expected-note@+1 {{reads the ambient crSat}}
      %raw = "xllvm.intr.aie4.acc32.v32.I512.ups"(%v) {shift = 1 : i32, sign = 1 : i32} : (vector<32xi16>) -> vector<32xi32>
      return %raw : vector<32xi32>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crSat = 1 (saturate), with an op that reads the ambient crSat}}
    // expected-note@+2 {{give every crSat reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %v = arith.constant dense<0> : vector<32xi16>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crSat = 1 (saturate)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v) : (vector<32xi16>) -> vector<32xi32>
      aie.end
    }
  }
}

// -----

module @func_ups_dynamic_helper {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @llvm.aie4.acc32.v32.I512.ups(vector<32xi16>, i32, i32) -> vector<32xi32>
    func.func private @raw_helper(%v: vector<32xi16>, %shift: i32) -> vector<32xi32> {
      %one = arith.constant 1 : i32
      // expected-note@+1 {{reads the ambient crSat}}
      %raw = func.call @llvm.aie4.acc32.v32.I512.ups(%v, %shift, %one) : (vector<32xi16>, i32, i32) -> vector<32xi32>
      return %raw : vector<32xi32>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crSat = 1 (saturate), with an op that reads the ambient crSat}}
    // expected-note@+2 {{give every crSat reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %one = arith.constant 1 : i32
      %v = arith.constant dense<0> : vector<32xi16>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crSat = 1 (saturate)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v, %one) : (vector<32xi16>, i32) -> vector<32xi32>
      aie.end
    }
  }
}

// -----

module @xllvm_bf16 {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @raw_helper(%v: vector<32xf32>) -> vector<32xbf16> {
      // expected-note@+1 {{reads the ambient crRnd}}
      %raw = "xllvm.intr.aie4.v32accfloat.to.v32bfloat16"(%v) : (vector<32xf32>) -> vector<32xbf16>
      return %raw : vector<32xbf16>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with an op that reads the ambient crRnd}}
    // expected-note@+2 {{give every crRnd reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %v = arith.constant dense<0.0> : vector<32xf32>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v) : (vector<32xf32>) -> vector<32xbf16>
      aie.end
    }
  }
}

// -----

module @xllvm_float_int_rounding {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @raw_helper(%v: vector<32xf32>) -> vector<32xi32> {
      // expected-note@+1 {{reads the ambient crRnd}}
      %raw = "xllvm.intr.aie4.v32float.to.v32int32"(%v) {scale = 0 : i32, sign = 1 : i32} : (vector<32xf32>) -> vector<32xi32>
      return %raw : vector<32xi32>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with an op that reads the ambient crRnd}}
    // expected-note@+2 {{give every crRnd reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %v = arith.constant dense<0.0> : vector<32xf32>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v) : (vector<32xf32>) -> vector<32xi32>
      aie.end
    }
  }
}

// -----

module @func_float_int_saturation {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @llvm.aie4.v32float.to.v32int32(vector<32xf32>, i32, i32) -> vector<32xi32>
    func.func private @raw_helper(%v: vector<32xf32>) -> vector<32xi32> {
      %one = arith.constant 1 : i32
      // expected-note@+1 {{reads the ambient crSat}}
      %raw = func.call @llvm.aie4.v32float.to.v32int32(%v, %one, %one) : (vector<32xf32>, i32, i32) -> vector<32xi32>
      return %raw : vector<32xi32>
    }
    // expected-error@+3 {{mixes an explicit-mode SRS, which needs crSat = 1 (saturate), with an op that reads the ambient crSat}}
    // expected-note@+2 {{give every crSat reader in the core an explicit}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %v = arith.constant dense<0.0> : vector<32xf32>
      %acc = arith.constant dense<0> : vector<64xi32>
      // expected-note@+1 {{needs crSat = 1 (saturate)}}
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      %raw = func.call @raw_helper(%v) : (vector<32xf32>) -> vector<32xi32>
      aie.end
    }
  }
}
