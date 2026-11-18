//===- lower_put_stream_aie4_invalid.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: aie-opt --split-input-file --verify-diagnostics %s

module @float {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %value = arith.constant 0.0 : f32
    // expected-error@+1 {{AIE4 stream writes require i32 or a fixed rank-1 vector<8xi32> or vector<16xi32> value}}
    aie.put_stream(%channel : i32, %value : f32)
    aie.end
  }
 }
}

// -----

module @i128 {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %value = arith.constant 0 : i128
    // expected-error@+1 {{AIE4 stream writes require i32 or a fixed rank-1 vector<8xi32> or vector<16xi32> value}}
    aie.put_stream(%channel : i32, %value : i128)
    aie.end
  }
 }
}

// -----

module @rank2 {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %value = arith.constant dense<0> : vector<2x8xi32>
    // expected-error@+1 {{AIE4 stream writes require i32 or a fixed rank-1 vector<8xi32> or vector<16xi32> value}}
    aie.put_stream(%channel : i32, %value : vector<2x8xi32>)
    aie.end
  }
 }
}

// -----

module @scalable {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %value = arith.constant dense<0> : vector<[8]xi32>
    // expected-error@+1 {{AIE4 stream writes require i32 or a fixed rank-1 vector<8xi32> or vector<16xi32> value}}
    aie.put_stream(%channel : i32, %value : vector<[8]xi32>)
    aie.end
  }
 }
}

// -----

module @v4 {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %value = arith.constant dense<0> : vector<4xi32>
    // expected-error@+1 {{operand #1 must be}}
    aie.put_stream(%channel : i32, %value : vector<4xi32>)
    aie.end
  }
 }
}

// -----

module @v32 {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %value = arith.constant dense<0> : vector<32xi32>
    // expected-error@+1 {{operand #1 must be}}
    aie.put_stream(%channel : i32, %value : vector<32xi32>)
    aie.end
  }
 }
}

// -----

module @channel1 {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 1 : i32
    %value = arith.constant 0 : i32
    // expected-error@+1 {{AIE4 cores have output stream 0 only}}
    aie.put_stream(%channel : i32, %value : i32)
    aie.end
  }
 }
}

// -----

module @negative_channel {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant -1 : i32
    %value = arith.constant dense<0> : vector<8xi32>
    // expected-error@+1 {{AIE4 cores have output stream 0 only}}
    aie.put_stream(%channel : i32, %value : vector<8xi32>)
    aie.end
  }
 }
}

// -----

module @wide_channel {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 18446744073709551616 : i128
    %value = arith.constant dense<0> : vector<16xi32>
    // expected-error@+1 {{AIE4 cores have output stream 0 only}}
    aie.put_stream(%channel : i128, %value : vector<16xi32>)
    aie.end
  }
 }
}

// -----

module @aie2p_vector {
 aie.device(npu2) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %value = arith.constant dense<0> : vector<16xi32>
    // expected-error@+1 {{vector stream writes are only supported on AIE4}}
    aie.put_stream(%channel : i32, %value : vector<16xi32>)
    aie.end
  }
 }
}

// -----

// 32-bit, 256-bit and 512-bit writes to stream 0 verify.
module @valid_writes {
 aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    %channel = arith.constant 0 : i32
    %scalar = arith.constant 7 : i32
    %narrow = arith.constant dense<1> : vector<8xi32>
    %wide = arith.constant dense<2> : vector<16xi32>
    aie.put_stream(%channel : i32, %scalar : i32)
    aie.put_stream(%channel : i32, %narrow : vector<8xi32>)
    aie.put_stream(%channel : i32, %wide : vector<16xi32>)
    aie.end
  }
 }
}
