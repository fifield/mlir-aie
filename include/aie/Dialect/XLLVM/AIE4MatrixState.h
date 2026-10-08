/* Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

//===- AIE4MatrixState.h - AIE4 MMAC matrix-state op traits --------C++ -*-===//
//
// Classify access to staging (za) and the active matrix (zb) in both dialects.
// Header-only traits let AIEVec recognize XLLVM writers without linking XLLVM.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_AIE4MATRIXSTATE_H
#define AIE_DIALECT_XLLVM_AIE4MATRIXSTATE_H

#include "mlir/IR/OpDefinition.h"

namespace xilinx::xllvm {

/// Marks an operation that writes one or more za staging chunks.
template <typename ConcreteType>
class AIE4StagingWriter
    : public mlir::OpTrait::TraitBase<ConcreteType, AIE4StagingWriter> {};

/// Marks an operation that copies staging to the active matrix.
template <typename ConcreteType>
class AIE4MatrixCommit
    : public mlir::OpTrait::TraitBase<ConcreteType, AIE4MatrixCommit> {};

/// Marks an operation that reads the active matrix.
template <typename ConcreteType>
class AIE4MatrixReader
    : public mlir::OpTrait::TraitBase<ConcreteType, AIE4MatrixReader> {};

/// True for a staging or active-matrix writer, in XLLVM or AIEVec.
inline bool isAIE4MatrixStateWriter(mlir::Operation *op) {
  return op->hasTrait<AIE4StagingWriter>() || op->hasTrait<AIE4MatrixCommit>();
}

} // namespace xilinx::xllvm

#endif // AIE_DIALECT_XLLVM_AIE4MATRIXSTATE_H
