//===- npu3_lock_limits.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 lock ids are bounded per tile type: 16 locks in a core tile, 64 in a
// memtile and 32 in a shim tile.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  %l0 = aie.lock(%shim, 31)
  %l1 = aie.lock(%mem, 63)
  %l2 = aie.lock(%core, 15)
}

// -----

aie.device(npu3) {
  %core = aie.tile(0, 2)
  // expected-error@+1 {{lock assigned invalid id (maximum is 15)}}
  %l = aie.lock(%core, 16)
}

// -----

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  // expected-error@+1 {{lock assigned invalid id (maximum is 63)}}
  %l = aie.lock(%mem, 64)
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{lock assigned invalid id (maximum is 31)}}
  %l = aie.lock(%shim, 32)
}
