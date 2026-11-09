// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: split-file %s %t
// RUN: not aie-opt --aie-npu-to-cert %t/missing-small.mlir 2>&1 | FileCheck %s --check-prefix=MISSING-SMALL
// RUN: not aie-opt --aie-npu-to-cert %t/missing-large.mlir 2>&1 | FileCheck %s --check-prefix=MISSING-LARGE
// RUN: not aie-opt --aie-npu-to-cert %t/unassigned-small.mlir 2>&1 | FileCheck %s --check-prefix=UNASSIGNED
// RUN: not aie-opt --aie-npu-to-cert %t/unassigned-large.mlir 2>&1 | FileCheck %s --check-prefix=UNASSIGNED

// Both direct lowering and splitting must reject unresolved buffer addresses.
// MISSING-SMALL: error: buffer 'missing_small' not found in device
// MISSING-LARGE: error: buffer 'missing_large' not found in device
// UNASSIGNED: error: referenced buffer must have address assigned
// UNASSIGNED: note: This buffer must have an address.

//--- missing-small.mlir
aie.device(npu3) {
  memref.global "private" constant @data : memref<4xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @missing_small} : memref<4xi32>
  }
}

//--- missing-large.mlir
aie.device(npu3) {
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @missing_large} : memref<4096xi32>
  }
}

//--- unassigned-small.mlir
aie.device(npu3) {
  %tile = aie.tile(1, 2)
  %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<4097xi32>
  memref.global "private" constant @data : memref<4xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4xi32>
  }
}

//--- unassigned-large.mlir
aie.device(npu3) {
  %tile = aie.tile(1, 2)
  %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<4097xi32>
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4096xi32>
  }
}
