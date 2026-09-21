//===---- XLLVMOps.cpp - XLLVM Dialect Operations ---------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// External LLVM (XLLVM) Dialect implementation.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/XLLVM/XLLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::xllvm;

#include "aie/Dialect/XLLVM/IR/XLLVMDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// XLLVMDialect
//===----------------------------------------------------------------------===//

void xllvm::XLLVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/XLLVM/IR/XLLVMOps.cpp.inc"
      >();
}

namespace xilinx::xllvm {

static llvm::Function *
getNamedIntrinsicDeclaration(llvm::Module *M, llvm::StringRef fullName,
                             llvm::Type *resTy, ArrayRef<llvm::Type *> argsTy) {
  auto *FT = llvm::FunctionType::get(resTy, argsTy, /*isVarArg=*/false);
  return cast<llvm::Function>(M->getOrInsertFunction(fullName, FT).getCallee());
}

llvm::CallInst *createExternalLLVMIntrinsicCall(
    llvm::IRBuilderBase &builder, LLVM::ModuleTranslation &moduleTranslation,
    Operation *intrOp, llvm::StringRef intrinsicName) {
  llvm::Type *resTy = nullptr;
  unsigned numResults = intrOp->getNumResults();
  if (numResults == 1)
    resTy = moduleTranslation.convertType(*(intrOp->getResultTypes().begin()));
  else if (numResults > 1) {
    SmallVector<llvm::Type *> resTys;
    for (auto ty : intrOp->getResultTypes())
      resTys.push_back(moduleTranslation.convertType(ty));
    resTy = llvm::StructType::get(builder.getContext(), resTys);
  }
  auto operands = moduleTranslation.lookupValues(intrOp->getOperands());
  SmallVector<llvm::Type *> types;
  for (auto *op : operands)
    types.push_back(op->getType());
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *llvmIntr =
      getNamedIntrinsicDeclaration(module, intrinsicName, resTy, types);
  return builder.CreateCall(llvmIntr, operands);
}

uint32_t encodeAIE4MulConf(bool zeroAcc, AIE4AMode amode, AIE4BMode bmode) {
  return (zeroAcc ? 1u : 0u) | (static_cast<uint32_t>(amode) << 1) |
         (static_cast<uint32_t>(bmode) << 3);
}

uint32_t encodeAIE4MacConf(bool zeroAcc, AIE4AMode amode, AIE4BMode bmode,
                           AIE4OperandSign lhsSign, AIE4OperandSign rhsSign) {
  return encodeAIE4MulConf(zeroAcc, amode, bmode) |
         (static_cast<uint32_t>(rhsSign) << 8) |
         (static_cast<uint32_t>(lhsSign) << 9);
}

llvm::CallInst *createAIE4ConfIntrinsicCall(
    llvm::IRBuilderBase &builder, LLVM::ModuleTranslation &moduleTranslation,
    Operation *intrOp, llvm::StringRef intrinsicName, uint32_t conf) {
  return createAIE4ConfIntrinsicCall(builder, moduleTranslation, intrOp,
                                     intrinsicName, ArrayRef<uint32_t>{conf});
}

llvm::CallInst *
createAIE4ConfIntrinsicCall(llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation,
                            Operation *intrOp, llvm::StringRef intrinsicName,
                            ArrayRef<uint32_t> trailingArgs) {
  assert(intrOp->getNumResults() == 1 &&
         "AIE4 conf intrinsics have exactly one result");
  llvm::Type *resTy =
      moduleTranslation.convertType(*(intrOp->getResultTypes().begin()));

  auto operands = moduleTranslation.lookupValues(intrOp->getOperands());
  // Materialize attributes as trailing i32 arguments in intrinsic order.
  for (uint32_t arg : trailingArgs)
    operands.push_back(builder.getInt32(arg));

  SmallVector<llvm::Type *> types;
  for (auto *op : operands)
    types.push_back(op->getType());

  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *llvmIntr =
      getNamedIntrinsicDeclaration(module, intrinsicName, resTy, types);
  return builder.CreateCall(llvmIntr, operands);
}

} // namespace xilinx::xllvm

#include "aie/Dialect/XLLVM/IR/XLLVMEnums.cpp.inc"

#define GET_OP_CLASSES
#include "aie/Dialect/XLLVM/IR/XLLVMOps.cpp.inc"
