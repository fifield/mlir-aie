/* Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

//===- AIE4MatrixState.h - AIE4 MMAC matrix-state op trait ---------C++ -*-===//
//
// The AIE4 matrix MAC reads B from hidden matrix registers (za staging, zb
// active matrix). Operations that write that state carry the
// `AIE4MatrixStateWriter` trait. AIEVec uses it to reject staging and commit
// operations inside `aievec.aie4.matrix_region` without linking XLLVM.
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

} // namespace xilinx::xllvm

#endif // AIE_DIALECT_XLLVM_AIE4MATRIXSTATE_H
