//===- AIEVecToLLVM.h - AIEVec to LLVM dialect conversion -------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H
#define AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H

#include "aie/Dialect/AIEVec/Utils/TargetInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
class ModuleOp;
} // namespace mlir

namespace xilinx {
namespace aievec {

enum class Aie2Fp32Emulation : uint32_t;

void populateAIEVecToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    Aie2Fp32Emulation aie2Fp32EmulationOption, AIEVecTargetInfo aieTarget);

void populateAIEVecToLLVMCommonConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void populateAIEVecToLLVMAIE2ConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void populateAIEVecToLLVMAIE2pConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void populateAIEVecToLLVMAIE4ConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void configureAIEVecToLLVMAIE4Legalizations(mlir::ConversionTarget &target);

/// Lower flat matrix operations to XLLVM after matmul and region expansion.
void populateAIEVecToLLVMAIE4MmacPatterns(mlir::LLVMTypeConverter &converter,
                                          mlir::RewritePatternSet &patterns);

/// Rewrite every aievec.aie4.matmul under `root` into an
/// aievec.aie4.matrix_region of row MMACs.
mlir::LogicalResult expandAIE4MatMulOps(mlir::Operation *root);

std::unique_ptr<mlir::Pass> createAIEVecAIE4ExpandMatMulPass();

/// Rewrite every aievec.aie4.matrix_region under `root` into 32 stage ops,
/// a commit and the inlined body with mmac -> mmac_active.
mlir::LogicalResult expandAIE4MatrixRegionOps(mlir::Operation *root);

std::unique_ptr<mlir::Pass> createAIEVecAIE4ExpandMatrixRegionPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass();
} // namespace aievec

// Forward declare options struct from generated code (in xilinx:: namespace)
struct ConvertAIEVecToLLVMOptions;

namespace aievec {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass(
    const xilinx::ConvertAIEVecToLLVMOptions &options);
} // namespace aievec
} // namespace xilinx

#endif // AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H
