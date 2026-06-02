//===- AIELocCheckpoint.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Util/AIELocCheckpoint.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace xilinx::AIE {

Location fuseStageLabel(Location loc, llvm::StringRef stage) {
  MLIRContext *ctx = loc.getContext();
  std::string label = (getCheckpointMetadataPrefix().str() + stage.str());
  StringAttr metadata = StringAttr::get(ctx, label);

  // Idempotent: if the top-level metadata already matches, no-op.
  if (auto fused = dyn_cast<FusedLoc>(loc))
    if (auto existing = dyn_cast_or_null<StringAttr>(fused.getMetadata()))
      if (existing == metadata)
        return loc;

  return FusedLoc::get(ctx, {loc}, metadata);
}

void applyStageLabel(Operation *op, llvm::StringRef stage) {
  if (!op)
    return;
  op->walk([&](Operation *child) {
    child->setLoc(fuseStageLabel(child->getLoc(), stage));
  });
}

} // namespace xilinx::AIE
