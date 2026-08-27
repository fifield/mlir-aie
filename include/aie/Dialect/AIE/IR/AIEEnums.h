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

inline bool isMM2S(DMAChannelDir dir) {
  return dir == DMAChannelDir::MM2S || dir == DMAChannelDir::ControlMM2S;
}

inline bool isS2MM(DMAChannelDir dir) {
  return dir == DMAChannelDir::S2MM || dir == DMAChannelDir::TraceS2MM;
}

/// True for the two specialized DMA channel directions. These exist only on
/// AIE4 shim NOC tiles -- this predicate tests the direction alone, so callers
/// that care must check the target and tile type themselves
inline bool isSpecializedShimDMAChannel(DMAChannelDir dir) {
  return dir == DMAChannelDir::ControlMM2S || dir == DMAChannelDir::TraceS2MM;
}

} // namespace xilinx::AIE

#endif
