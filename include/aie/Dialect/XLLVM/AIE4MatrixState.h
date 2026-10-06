/* Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

//===- AIE4MatrixState.h - AIE4 MMAC matrix-state op traits --------C++ -*-===//
//
// Header-only traits let AIEVec classify matrix-state operations without
// linking XLLVM. Staging writers and commits also carry AIE4MatrixStateWriter
// so matrix_region can reject operations that clobber its resident tile.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_AIE4MATRIXSTATE_H
#define AIE_DIALECT_XLLVM_AIE4MATRIXSTATE_H

#include "mlir/IR/OpDefinition.h"

namespace xilinx::xllvm {

/// Marks an operation that writes AIE4 matrix staging or active-matrix state.
template <typename ConcreteType>
class AIE4MatrixStateWriter
    : public mlir::OpTrait::TraitBase<ConcreteType, AIE4MatrixStateWriter> {};

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

} // namespace xilinx::xllvm

#endif // AIE_DIALECT_XLLVM_AIE4MATRIXSTATE_H
