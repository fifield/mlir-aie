//===- AIE4MaskLowering.h - AIE4 tail predication ---------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_AIE4MASKLOWERING_H
#define AIE_DIALECT_AIEVEC_TRANSFORMS_AIE4MASKLOWERING_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace xilinx::aievec {

// Lower AIE4 masked memory operations to active-lane accesses, using vector
// accesses only when all lanes are active and alignment is proven. Fill
// inactive reduction lanes with the identity. Diagnose unsupported forms.
mlir::LogicalResult lowerAIE4Masks(mlir::Operation *root);

// Create the pass that runs before canonicalization in convert-vector-to-aievec.
// Only AIE4 targets are affected.
std::unique_ptr<mlir::Pass>
createAIE4LowerVectorMasksPass(llvm::StringRef aieTarget);
void registerAIE4LowerVectorMasksPass();

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_AIE4MASKLOWERING_H
