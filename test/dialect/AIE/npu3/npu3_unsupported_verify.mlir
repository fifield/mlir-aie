//===- npu3_unsupported_verify.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifiers reject AIE4 features the target model cannot describe yet.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// AIE4 control packets have their own header and payload size.
aie.device(npu3) {
  aie.runtime_sequence @seq() {
    // expected-error@+1 {{control packets are not yet supported on AIE4}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 0 : i32, stream_id = 0 : i32, data = array<i32: 1>}
  }
}
