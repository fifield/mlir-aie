//===- lower_stream_aie4_invalid.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reject unsupported stream result types and constant indices during
// verification, before standard lowering.
//
// RUN: aie-opt --split-input-file --verify-diagnostics %s

module @aie4_scalar_i32 {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{AIE4 cores read streams in 512-bit or 1024-bit units; use a vector<16xi32> or vector<32xi32> result instead of 'i32'}}
    %0 = aie.get_stream(%c0 : i32) : i32
    aie.end
  }
 }
}

// -----

module @aie4_scalar_f32 {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{instead of 'f32'}}
    %0 = aie.get_stream(%c0 : i32) : f32
    aie.end
  }
 }
}

// -----

module @aie4_wide_i128 {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{instead of 'i128'}}
    %0 = aie.get_stream(%c0 : i32) : i128
    aie.end
  }
 }
}

// -----

// Only input streams 0 and 1 are valid.
module @aie4_bad_stream_index {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c2 = arith.constant 2 : i32
    // expected-error@+1 {{AIE4 cores have input streams 0 and 1 only; got 2}}
    %0 = aie.get_stream(%c2 : i32) : vector<32xi32>
    aie.end
  }
 }
}

// -----

// Diagnose out-of-range constants without narrowing to 64 bits.
module @aie4_bad_stream_index_wide {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %big = arith.constant 18446744073709551616 : i128
    // expected-error@+1 {{AIE4 cores have input streams 0 and 1 only; got 18446744073709551616}}
    %0 = aie.get_stream(%big : i128) : vector<32xi32>
    aie.end
  }
 }
}

// -----

// Vector reads have no AIE2P counterpart in this lowering.
module @aie2p_vector_read {
 aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{vector stream reads are only supported on AIE4}}
    %0 = aie.get_stream(%c0 : i32) : vector<16xi32>
    aie.end
  }
 }
}

// -----

module @aie4_rank2_vector_read {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{AIE4 stream reads require a fixed rank-1 vector<16xi32> or vector<32xi32> result; got 'vector<2x16xi32>'}}
    %0 = aie.get_stream(%c0 : i32) : vector<2x16xi32>
    aie.end
  }
 }
}

// -----

module @aie4_scalable_vector_read {
 aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{AIE4 stream reads require a fixed rank-1 vector<16xi32> or vector<32xi32> result; got 'vector<[16]xi32>'}}
    %0 = aie.get_stream(%c0 : i32) : vector<[16]xi32>
    aie.end
  }
 }
}
