//===- bad_control_packet.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The payload word count is encoded as (words - 1) in a 4-bit field of the
// AIE4 control word (the architecture specification), so counts outside [1, 16] must
// be rejected rather than masked into a different length.

aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error @+1 {{read control packet length 17 exceeds the maximum of 16 32-bit words}}
    aiex.control_packet {address = 7340032 : ui32, length = 17 : i32, opcode = 1 : i32, stream_id = 0 : i32}
  }
}

// -----

// Earlier targets carry a 2-bit beats field, so the limit there is 4 words.

aie.device(npu1) {
  aie.runtime_sequence() {
    // expected-error @+1 {{read control packet length 5 exceeds the maximum of 4 32-bit words}}
    aiex.control_packet {address = 1024 : ui32, length = 5 : i32, opcode = 1 : i32, stream_id = 0 : i32}
  }
}

// -----

// A read with no length would encode beats = -1, which masks to a 16-word
// request.

aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error @+1 {{read control packet requires a length attribute}}
    aiex.control_packet {address = 7340032 : ui32, opcode = 1 : i32, stream_id = 0 : i32}
  }
}

// -----

// A zero-word payload is not representable either.

aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error @+1 {{control packet payload must be at least 1 word}}
    aiex.control_packet {address = 7340032 : ui32, length = 0 : i32, opcode = 1 : i32, stream_id = 0 : i32}
  }
}

// -----

// Opcode "11" is reserved (the architecture specification).

aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error @+1 {{opcode 3 is reserved}}
    aiex.control_packet {address = 7340032 : ui32, data = array<i32: 1>, opcode = 3 : i32, stream_id = 0 : i32}
  }
}

// -----

// A read returns its data on the response port; it must not carry a payload.

aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error @+1 {{read control packet must not carry a data payload}}
    aiex.control_packet {address = 7340032 : ui32, data = array<i32: 1>, opcode = 1 : i32, stream_id = 0 : i32}
  }
}

// -----

// A write needs one.

aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error @+1 {{control packet with a write opcode requires a data payload}}
    aiex.control_packet {address = 7340032 : ui32, length = 2 : i32, opcode = 0 : i32, stream_id = 0 : i32}
  }
}
