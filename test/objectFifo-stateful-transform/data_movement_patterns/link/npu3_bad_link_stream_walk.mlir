//===- npu3_bad_link_stream_walk.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A stream walk needs one BD per output on one AIE4 MemTile channel (16 BDs).

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @all(%mem dimensionsToStream [<size = 16, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo.link [@rows] -> [@all] ([] [])
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @all(%mem dimensionsToStream [<size = 17, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  // expected-error@+1 {{lowers to one descriptor per output, 17, but a MemTile channel has 16}}
  aie.objectfifo.link [@rows] -> [@all] ([] [])
}
