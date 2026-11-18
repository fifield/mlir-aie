//===- AIEEnums.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_ENUMS_H
#define MLIR_AIE_ENUMS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "aie/Dialect/AIE/IR/AIEEnums.h.inc"

namespace xilinx::AIE {

/// True if `dir` moves data out of memory onto a stream (MM2S, ControlMM2S).
inline bool isMM2S(DMAChannelDir dir) {
  return dir == DMAChannelDir::MM2S || dir == DMAChannelDir::ControlMM2S;
}

/// True if `dir` moves data from a stream into memory (S2MM, TraceS2MM).
inline bool isS2MM(DMAChannelDir dir) {
  return dir == DMAChannelDir::S2MM || dir == DMAChannelDir::TraceS2MM;
}

/// The stream-switch bundle the channels of `dir` use.
inline WireBundle getDMAChannelBundle(DMAChannelDir dir) {
  switch (dir) {
  case DMAChannelDir::S2MM:
  case DMAChannelDir::MM2S:
    break;
  case DMAChannelDir::ControlMM2S:
    return WireBundle::DMAControl;
  case DMAChannelDir::TraceS2MM:
    return WireBundle::DMATrace;
  }
  return WireBundle::DMA;
}

} // namespace xilinx::AIE

#endif