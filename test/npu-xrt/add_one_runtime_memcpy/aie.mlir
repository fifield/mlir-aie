//===- aie.mlir ------------------------------------------------*- MLIR -*-===
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

module {
  // Simple "add one" design that uses token-gated memcpy ops inside
  // an aie.runtime_sequence to move data between host buffers and the
  // tile-local buffers that the core operates on.
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    // Tile-local storage for the core.
    %core_in = aie.buffer(%core) {address = 0x400 : i32} : memref<64xi32>
    %core_out = aie.buffer(%core) {address = 0x1000 : i32} : memref<64xi32>

    // Tokens gate the pipeline: input memcpy → core compute → output memcpy.
    aiex.token(0) { sym_name = "token_in" }
    aiex.token(0) { sym_name = "token_out" }

    // Core adds 1 to every element in the tile-local buffer.
    %core_func = aie.core(%core) {
      // Wait for the inbound memcpy to finish.
      aiex.useToken @token_in(Acquire, 1)

      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      scf.for %i = %c0 to %c64 step %c1 {
        %val = memref.load %core_in[%i] : memref<64xi32>
        %inc = arith.addi %val, %c1_i32 : i32
        memref.store %inc, %core_out[%i] : memref<64xi32>
      }

      // Signal that output is ready for the outbound memcpy.
      aiex.useToken @token_out(Release, 1)
      aie.end
    }

    // Move host → core buffer, then core buffer → host, using token-gated memcpy
    // inside the runtime sequence.
    aie.runtime_sequence @run(%input: memref<64xi32>, %output: memref<64xi32>) {
      // Kick off inbound DMA immediately and bump token_in to 1 when done.
      aiex.memcpy @token_in(0, 1) (%shim : <%input, 0, 64>, %core : <%core_in, 0, 64>) : (memref<64xi32>, memref<64xi32>)

      // Wait for the core to signal completion (token_out reaches 1), then
      // copy results back to host.
      aiex.memcpy @token_out(1, 0) (%core : <%core_out, 0, 64>, %shim : <%output, 0, 64>) : (memref<64xi32>, memref<64xi32>)
    }
  }
}
