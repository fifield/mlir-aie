//===- npu3_unsupported_lowering.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Lowerings that would produce AIE2-shaped output for AIE4 refuse it.

// RUN: not aie-opt --convert-aie-to-transaction %s 2>&1 | FileCheck %s --check-prefix=TXN
// TXN: error: 'aie.device' op aie-codegen configuration is not yet supported for the AIE4 architecture

// RUN: not aie-translate --aie-generate-cdo %s 2>&1 | FileCheck %s --check-prefix=CDO
// CDO: error: 'aie.device' op aie-codegen configuration is not yet supported for the AIE4 architecture

// RUN: not aie-translate --aie-generate-xaie %s 2>&1 | FileCheck %s --check-prefix=XAIE
// XAIE: error: 'aie.device' op --aie-generate-xaie is not supported for the AIE4 architecture

// RUN: not aie-translate --aie-mlir-to-scsim-config %s 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: not aie-translate --aie-mlir-to-xpe %s 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: not aie-translate --aie-mlir-to-shim-solution %s 2>&1 | FileCheck %s --check-prefix=SIM
// SIM: error: 'aie.device' op simulation files are not supported for the AIE4 architecture

// RUN: not aie-opt --aie-standard-lowering %s 2>&1 | FileCheck %s --check-prefix=CORE
// CORE: error: 'aie.use_lock' op has no core lowering for AIE4

// RUN: not aie-translate --aie-npu-to-binary --aie-sequence-name=seq %s 2>&1 | FileCheck %s --check-prefix=ADDR
// ADDR: error: 'aiex.npu.write32' op cannot encode the address of tile (0, 2) on AIE4

// RUN: not aie-opt --aie-generate-column-control-overlay="configuration=true" %s 2>&1 | FileCheck %s --check-prefix=OVERLAY
// OVERLAY: error: 'aie.device' op the column control overlay is not yet supported on AIE4

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %lock = aie.lock(%core, 0) {init = 1 : i32}
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.core(%core) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.use_lock(%lock, Release, %c1)
      aie.end
    }
    aie.runtime_sequence @seq() {
      %addr = arith.constant 0x1D000 : i32
      %value = arith.constant 1 : i32
      aiex.npu.write32(%addr, %value) {column = 0 : i32, row = 2 : i32} : i32, i32
    }
  }
}
