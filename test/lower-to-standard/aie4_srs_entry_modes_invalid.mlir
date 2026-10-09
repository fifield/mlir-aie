//===- aie4_srs_entry_modes_invalid.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reject conflicting rounding/saturation requirements and mode clobbers;
// suggest explicit rounding attributes where they resolve ambient conflicts.
//
// RUN: aie-opt --aie-standard-lowering --split-input-file --verify-diagnostics %s

// Two explicit roundings differ.
module @r1_two_roundings {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op has explicit-mode SRS ops that need different rounding modes (crRnd = 9 and crRnd = 12); a core has one crRnd}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{needs crRnd = 12 (conv_even)}}
      %y2 = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<conv_even>} : vector<64xi32> -> vector<64xi8>
      aie.end
    }
  }
}

// -----

// Explicit rounding with srs_i32_i16 (ambient crRnd).
module @r2_srs_i32_i16 {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with an op that reads the ambient crRnd; setting crRnd for one would change the other's numerics}}
    // expected-note@+2 {{give every crRnd reader in the core an explicit rounding attribute, or split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{reads the ambient crRnd}}
      %n = aievec.aie4.srs_i32_i16 %w {shift = 2 : i32} : vector<32xi32>, vector<32xi16>
      aie.end
    }
  }
}

// -----

// Explicit rounding with an aievec.aie4.srs that omits it.
module @r2_modeless_srs {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{op mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with an op that reads the ambient crRnd}}
    // expected-note@+2 {{give every crRnd reader in the core an explicit rounding attribute}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{reads the ambient crRnd}}
      %y2 = aievec.aie4.srs %w {shift = 12 : i32, isSigned = true} : vector<32xi32> -> vector<32xi16>
      aie.end
    }
  }
}

// -----

// Explicit pos_inf with generic bf16 rounding.
module @r3_truncf {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with ops that need round-to-nearest-even (crRnd = 12)}}
    // expected-note@+2 {{use rounding = conv_even on the SRS, or split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      %s = memref.load %f[%c0] : memref<32xf32>
      // expected-note@+1 {{needs crRnd = 12}}
      %b = arith.truncf %s : f32 to bf16
      memref.store %b, %g[%c0] : memref<16xbf16>
      aie.end
    }
  }
}

// -----

// Explicit pos_inf with a float -> int conversion.
module @r4_fptosi {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op mixes an explicit-mode SRS, which needs crRnd = 9 (pos_inf), with a float -> int conversion, which needs crRnd = 2 (toward zero)}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{needs crRnd = 2}}
      %i = aievec.aie4.conv_f32_i32 %x : vector<32xf32>, vector<32xi32>
      aie.end
    }
  }
}

// -----

// Two explicit saturations differ.
module @s1_two_saturations {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op has explicit-mode SRS ops that need different saturation modes (crSat = 1 and crSat = 3); a core has one crSat}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crSat = 1 (saturate)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{needs crSat = 3 (symmetric)}}
      %y2 = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<symmetric>} : vector<64xi32> -> vector<64xi8>
      aie.end
    }
  }
}

// -----

// Explicit saturation conflicts with srs_i32_i16's ambient crSat.
// Omit rounding so both ops read ambient crRnd.
module @s2_srs_i32_i16 {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op mixes an explicit-mode SRS, which needs crSat = 1 (saturate), with an op that reads the ambient crSat; setting crSat for one would change the other's numerics}}
    // expected-note@+2 {{give every crSat reader in the core an explicit saturation attribute, or split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crSat = 1 (saturate)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{reads the ambient crSat}}
      %n = aievec.aie4.srs_i32_i16 %w {shift = 2 : i32} : vector<32xi32>, vector<32xi16>
      aie.end
    }
  }
}

// -----

// Explicit saturation with a shifting UPS (ambient crSat).
module @s2_ups_shift {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{op mixes an explicit-mode SRS, which needs crSat = 1 (saturate), with an op that reads the ambient crSat}}
    // expected-note@+2 {{give every crSat reader in the core an explicit saturation attribute}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crSat = 1 (saturate)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      %n = vector.extract_strided_slice %y offsets = [0], sizes = [32], strides = [1] : vector<64xi8> to vector<32xi8>
      %m = aievec.aie4.unpack_i8_i16 %n : vector<32xi8>, vector<32xi16>
      // expected-note@+1 {{reads the ambient crSat}}
      %u = aievec.aie4.ups_i16_i32 %m {shift = 4 : i32} : vector<32xi16>, vector<32xi32>
      aie.end
    }
  }
}

// -----

// Symmetric saturation with a float -> int conversion.
module @s3_symmetric_fptosi {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op mixes an explicit-mode SRS, which needs crSat = 3 (symmetric), with a float -> int conversion, which needs asymmetric saturation (crSat bit 1 clear) to keep INT32_MIN exact}}
    // expected-note@+2 {{split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crSat = 3 (symmetric)}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<sym_floor>, saturation = #aievec.aie4_sat<symmetric>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{needs crSat bit 1 clear}}
      %i = aievec.aie4.conv_f32_i32 %x : vector<32xf32>, vector<32xi32>
      aie.end
    }
  }
}

// -----

// An external call.
module @c1_external_call {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    func.func private @ext()
    // expected-error@+3 {{'aie.core' op has an explicit-mode SRS that relies on modes written at core entry, and a call that may change them}}
    // expected-note@+2 {{external calls, indirect calls and inline assembly are mode clobbers (D19); remove the call from this core}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf) and crSat = 1 (saturate) from core entry}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{may write crRnd and crSat}}
      func.call @ext() : () -> ()
      aie.end
    }
  }
}

// -----

// A direct llvm.aie4.set.ctrl.reg call.
module @c1_set_ctrl_reg {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    func.func private @llvm.aie4.set.ctrl.reg(i32, i32)
    // expected-error@+3 {{'aie.core' op has an explicit-mode SRS that relies on modes written at core entry, and a call that may change them}}
    // expected-note@+2 {{external calls, indirect calls and inline assembly are mode clobbers (D19); remove the call from this core}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf) from core entry}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      %r = arith.constant 46 : i32
      %m = arith.constant 0 : i32
      // expected-note@+1 {{may write crRnd and crSat}}
      func.call @llvm.aie4.set.ctrl.reg(%r, %m) : (i32, i32) -> ()
      aie.end
    }
  }
}

// -----

// The clobber is inside a callee.
module @c1_in_callee {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    func.func private @ext()
    func.func @helper() {
      // expected-note@+1 {{may write crRnd and crSat}}
      func.call @ext() : () -> ()
      return
    }
    // expected-error@+3 {{'aie.core' op has an explicit-mode SRS that relies on modes written at core entry, and a call that may change them}}
    // expected-note@+2 {{external calls, indirect calls and inline assembly are mode clobbers (D19); remove the call from this core}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 12 (conv_even) and crSat = 1 (saturate) from core entry}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<conv_even>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      func.call @helper() : () -> ()
      aie.end
    }
  }
}

// -----

// An indirect call.
module @c1_call_indirect {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    func.func private @ext()
    // expected-error@+3 {{'aie.core' op has an explicit-mode SRS that relies on modes written at core entry, and a call that may change them}}
    // expected-note@+2 {{external calls, indirect calls and inline assembly are mode clobbers (D19); remove the call from this core}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf) from core entry}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
      %f = func.constant @ext : () -> ()
      // expected-note@+1 {{may write crRnd and crSat}}
      func.call_indirect %f() : () -> ()
      aie.end
    }
  }
}

// -----

// Inline assembly.
module @c1_inline_asm {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    // expected-error@+3 {{'aie.core' op has an explicit-mode SRS that relies on modes written at core entry, and a call that may change them}}
    // expected-note@+2 {{external calls, indirect calls and inline assembly are mode clobbers (D19); remove the call from this core}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      // expected-note@+1 {{needs crSat = 1 (saturate) from core entry}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{may write crRnd and crSat}}
      llvm.inline_asm has_side_effects "nop", "" : () -> ()
      aie.end
    }
  }
}

// -----

// aie.debug lowers to a call to the external debug_i32.
module @c1_debug {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op has an explicit-mode SRS that relies on modes written at core entry, and a call that may change them}}
    // expected-note@+2 {{external calls, indirect calls and inline assembly are mode clobbers (D19); remove the call from this core}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{needs crRnd = 9 (pos_inf) and crSat = 1 (saturate) from core entry}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      %d = arith.constant 1 : i32
      // expected-note@+1 {{may write crRnd and crSat}}
      aie.debug(%d : i32)
      aie.end
    }
  }
}

// -----

// Suggest conv_even when ambient SRS conflicts with generic bf16 rounding.
module @check2_modeless_srs {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+3 {{'aie.core' op mixes ops that need round-to-nearest-even (crRnd = 12) with an explicit op that reads the ambient crRnd; setting crRnd for one would change the other's numerics}}
    // expected-note@+2 {{add 'rounding = #aievec.aie4_rnd<conv_even>' to this aievec.aie4.srs so it shares crRnd = 12 with the other ops, or split them into separate cores}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      %s = memref.load %f[%c0] : memref<32xf32>
      // expected-note@+1 {{needs crRnd = 12}}
      %b = arith.truncf %s : f32 to bf16
      memref.store %b, %g[%c0] : memref<16xbf16>
      aie.end
    }
  }
}

// -----

// Suggest sym_floor when ambient SRS conflicts with float-to-int conversion.
module @check3_modeless_srs {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<64xi32>
    %h = aie.buffer(%t) {sym_name = "h"} : memref<32xi32>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xf32>
    %g = aie.buffer(%t) {sym_name = "g"} : memref<16xbf16>
    // expected-error@+4 {{'aie.core' op mixes a float -> int conversion, which needs crRnd = 2 (toward zero), with an op that reads the ambient crRnd; setting crRnd for one would change the other's numerics}}
    // expected-note@+3 {{split them into separate cores}}
    // expected-note@+2 {{or add 'rounding = #aievec.aie4_rnd<sym_floor>' to this aievec.aie4.srs}}
    // expected-error@+1 {{failed to legalize operation 'aie.core'}}
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = vector.load %a[%c0] : memref<64xi32>, vector<64xi32>
      %w = vector.load %h[%c0] : memref<32xi32>, vector<32xi32>
      %x = vector.load %f[%c0] : memref<32xf32>, vector<32xf32>
      // expected-note@+1 {{reads the ambient crRnd}}
      %y = aievec.aie4.srs %v {shift = 12 : i32, isSigned = true} : vector<64xi32> -> vector<64xi8>
      // expected-note@+1 {{needs crRnd = 2}}
      %i = aievec.aie4.conv_f32_i32 %x : vector<32xf32>, vector<32xi32>
      aie.end
    }
  }
}
