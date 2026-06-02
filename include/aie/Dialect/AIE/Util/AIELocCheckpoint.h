//===- AIELocCheckpoint.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Stage-checkpoint location helper. Used by aiecc to fuse a
// "checkpoint:<stage>" StringAttr into every op's mlir::Location at the
// boundary of each intermediate `.mlir` dump file. Subsequent passes
// propagate getLoc() naturally, so a final aiex.npu.* op carries the chain
// of all stages it survived through.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_UTIL_AIELOCCHECKPOINT_H
#define AIE_DIALECT_AIE_UTIL_AIELOCCHECKPOINT_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx {
namespace AIE {

/// The metadata-attribute prefix this helper uses on the FusedLoc it
/// produces. JSON serializers and tooling can detect a checkpoint metadata
/// by looking for a StringAttr starting with this prefix.
inline llvm::StringRef getCheckpointMetadataPrefix() { return "checkpoint:"; }

/// Fuse a "checkpoint:<stage>" StringAttr into `loc` and return a new
/// FusedLoc wrapping it. UnknownLoc is wrapped (not passed through) so
/// every op carries stage-provenance regardless of whether it has
/// source-provenance. Idempotent: if `loc` is already a FusedLoc whose
/// top-level metadata is the same "checkpoint:<stage>", returns `loc`
/// unchanged.
mlir::Location fuseStageLabel(mlir::Location loc, llvm::StringRef stage);

/// Walk `op` and replace every nested op's location with
/// fuseStageLabel(child->getLoc(), stage). Cheap attribute rewrite, no IR
/// restructuring. Walking from a ModuleOp covers the whole module.
void applyStageLabel(mlir::Operation *op, llvm::StringRef stage);

} // namespace AIE
} // namespace xilinx

#endif // AIE_DIALECT_AIE_UTIL_AIELOCCHECKPOINT_H
