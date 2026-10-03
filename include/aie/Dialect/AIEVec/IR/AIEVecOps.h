//===- AIEVecOps.h - AIE Vector Dialect and Operations ----------*- C++ -*-===//
//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines the AIE vector dialect and the operations.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "aie/Dialect/AIEVec/IR/AIEVecEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecAttributes.h.inc"

#include "AIEVecDialect.h"

// Shared control-register resources for AIE4 operations.
#include "aie/Dialect/XLLVM/AIE4CtrlRegResources.h"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h.inc"

namespace xilinx::aievec {
/// Row limit for the unrolled matmul lowering, which keeps M accumulator
/// rows live. Bounds register pressure against 64 2048-bit accumulator
/// registers; surrounding live values can still cause spills. Larger
/// contractions remain generic, and larger explicit matmuls are rejected.
inline constexpr int64_t kAIE4MatMulMaxRows = 64;
} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H
