//===- AIE4Staging.h - AIE4 flat matrix-state scope helpers -----*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Shared scope classification for staging checks and native contraction
// lowering.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_AIE4STAGING_H
#define AIE_DIALECT_AIEVEC_TRANSFORMS_AIE4STAGING_H

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseSet.h"

namespace xilinx::aievec {

/// aievec.aie4.stage, stage_stream, commit or mmac_active.
bool isAIE4FlatMatrixOp(mlir::Operation *op);

/// The first proper ancestor of `op` that is an aie.core, a
/// FunctionOpInterface or IsIsolatedFromAbove. aie.device and builtin.module
/// are skipped. Returns nullptr if there is no such ancestor.
mlir::Operation *getAIE4ExecutionScope(mlir::Operation *op);

/// The execution scopes under `root` that contain at least one flat op.
llvm::DenseSet<mlir::Operation *> collectAIE4FlatScopes(mlir::Operation *root);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_AIE4STAGING_H
