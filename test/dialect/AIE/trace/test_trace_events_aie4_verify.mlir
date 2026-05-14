//===- test_trace_events_aie4_verify.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// AIE2 core event on AIE4 device should fail
module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    aie.trace @bad_arch(%tile_0_2) {
      // expected-error @+1 {{event 'CoreEventAIE2::PERF_CNT_0' is not valid for core tile}}
      aie.trace.event<CoreEventAIE2::PERF_CNT_0>
    }
  }
}

// -----

// ShimTileEvent on a core tile should fail
module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    aie.trace @bad_tile_type(%tile_0_2) {
      // expected-error @+1 {{event 'ShimTileEventAIE4::DMA_S2MM_0_START_TASK' is not valid for core tile}}
      aie.trace.event<ShimTileEventAIE4::DMA_S2MM_0_START_TASK>
    }
  }
}

// -----

// CoreEvent on a shim tile should fail
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0)
    aie.trace @bad_core_on_shim(%tile_0_0) {
      // expected-error @+1 {{event 'CoreEventAIE4::PERF_CNT_0' is not valid for shim tile}}
      aie.trace.event<CoreEventAIE4::PERF_CNT_0>
    }
  }
}

// -----

// CoreEvent on a mem tile should fail
module {
  aie.device(npu3) {
    %tile_0_1 = aie.tile(0, 1)
    aie.trace @bad_core_on_memtile(%tile_0_1) {
      // expected-error @+1 {{event 'CoreEventAIE4::PERF_CNT_0' is not valid for mem tile}}
      aie.trace.event<CoreEventAIE4::PERF_CNT_0>
    }
  }
}
