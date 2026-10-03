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
#include "llvm/IR/Intrinsics.h"

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
  llvm::Type *resTy = builder.getVoidTy();
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

uint32_t encodeAIE4Bf16FpConf(bool zeroAcc, AIE4AMode amode, AIE4BMode bmode,
                              bool xIsBf16, bool yIsBf16) {
  return encodeAIE4MulConf(zeroAcc, amode, bmode) |
         (xIsBf16 ? 1u << 5 : 0u) | (yIsBf16 ? 1u << 6 : 0u);
}

uint32_t encodeAIE4MmacIntConf(bool zeroAcc, AIE4MmacIntPrecision bmode,
                               AIE4OperandSign xSign, AIE4OperandSign zSign) {
  return (zeroAcc ? 1u : 0u) | (static_cast<uint32_t>(bmode) << 3) |
         (static_cast<uint32_t>(zSign) << 8) |
         (static_cast<uint32_t>(xSign) << 9);
}

uint32_t encodeAIE4MmacFp8Conf(bool zeroAcc, AIE4MmacFp8EMode emode) {
  return (zeroAcc ? 1u : 0u) | (static_cast<uint32_t>(emode) << 10);
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
  assert(intrOp->getNumResults() <= 1 &&
         "AIE4 conf intrinsics have at most one result");
  llvm::Type *resTy =
      intrOp->getNumResults() == 0
          ? builder.getVoidTy()
          : moduleTranslation.convertType(intrOp->getResultTypes().front());

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

llvm::Value *createAIE4FPOrderIntrinsicCall(
    llvm::IRBuilderBase &builder, LLVM::ModuleTranslation &moduleTranslation,
    Operation *intrOp, llvm::StringRef intrinsicName, uint32_t opsel) {
  auto *resTy = moduleTranslation.convertType(intrOp->getResult(0).getType());
  // Compare selectors leave the vector result undefined. Keep that result out
  // of the helper ABI: Peano cannot assign a register class to it at O0.
  auto selector = static_cast<AIE4VFCmpOpSel>(opsel);
  bool isCompare = selector == AIE4VFCmpOpSel::Bf16Lt ||
                   selector == AIE4VFCmpOpSel::Bf16Ge ||
                   selector == AIE4VFCmpOpSel::Fp16Lt ||
                   selector == AIE4VFCmpOpSel::Fp16Ge ||
                   selector == AIE4VFCmpOpSel::Fp32Lt ||
                   selector == AIE4VFCmpOpSel::Fp32Ge;
  auto *helperResTy = isCompare
                          ? llvm::cast<llvm::StructType>(resTy)->getElementType(1)
                          : resTy;
  auto operands = moduleTranslation.lookupValues(intrOp->getOperands());
  SmallVector<llvm::Type *> types;
  for (auto *operand : operands)
    types.push_back(operand->getType());

  auto *module = builder.GetInsertBlock()->getModule();
  const std::string baseName =
      ("__mlir_aie4_fporder_" + intrinsicName + "_" + llvm::Twine(opsel)).str();
  std::string helperName = baseName;
  auto *helperTy = llvm::FunctionType::get(helperResTy, types, false);
  auto *helper = module->getFunction(helperName);
  // Do not capture a user symbol that happens to use the generated name.
  // The marker also lets subsequent operations reuse a renamed helper.
  for (unsigned suffix = 1; module->getNamedValue(helperName); ++suffix) {
    if (helper && helper->hasInternalLinkage() &&
        helper->getFunctionType() == helperTy &&
        helper->hasFnAttribute("aie4-fporder-helper") &&
        helper->getFnAttribute("aie4-fporder-helper").getValueAsString() ==
            baseName)
      break;
    helperName = (baseName + "." + llvm::Twine(suffix)).str();
    helper = module->getFunction(helperName);
  }
  if (!helper) {
    helper = llvm::Function::Create(
        helperTy, llvm::GlobalValue::InternalLinkage, helperName, module);
    // Keep the native instruction inside a call boundary during scheduling.
    // llvm.sideeffect gives that call inaccessible-memory effects even after
    // interprocedural attribute inference, so LLVM cannot CSE or move it across
    // mode changes despite the incorrectly pure native intrinsic. This costs a
    // call until Peano models crFPOrder on these intrinsics.
    helper->addFnAttr(llvm::Attribute::NoInline);
    helper->addFnAttr("aie4-fporder-helper", baseName);
    llvm::IRBuilder<> helperBuilder(
        llvm::BasicBlock::Create(builder.getContext(), "entry", helper));
    helperBuilder.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
        module, llvm::Intrinsic::sideeffect));
    SmallVector<llvm::Value *> args;
    for (auto &arg : helper->args())
      args.push_back(&arg);
    for (uint32_t arg : {0u, opsel, 0u}) {
      args.push_back(helperBuilder.getInt32(arg));
      types.push_back(helperBuilder.getInt32Ty());
    }
    auto *intrinsic =
        getNamedIntrinsicDeclaration(module, intrinsicName, resTy, types);
    llvm::Value *result = helperBuilder.CreateCall(intrinsic, args);
    if (isCompare)
      result = helperBuilder.CreateExtractValue(result, 1);
    helperBuilder.CreateRet(result);
  }
  llvm::Value *result = builder.CreateCall(helper, operands);
  if (isCompare)
    result = builder.CreateInsertValue(llvm::PoisonValue::get(resTy), result, 1);
  return result;
}

llvm::CallInst *
createAIE4SelIntrinsicCall(llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation,
                           Operation *intrOp, llvm::StringRef intrinsicName,
                           uint32_t sign) {
  assert(intrOp->getNumOperands() == 3 &&
         "AIE4 select intrinsic takes exactly (lhs, rhs, sel)");
  assert(intrOp->getNumResults() == 1 &&
         "AIE4 select intrinsic has exactly one result");
  llvm::Type *resTy =
      moduleTranslation.convertType(*(intrOp->getResultTypes().begin()));

  auto operandVals = moduleTranslation.lookupValues(intrOp->getOperands());
  // The mask precedes fporder in the intrinsic signature.
  SmallVector<llvm::Value *> operands = {
      operandVals[0],           operandVals[1],
      builder.getInt32(sign),   builder.getInt32(2), // opsel = VSEL
      operandVals[2],           builder.getInt32(0)  // fporder = 0
  };

  SmallVector<llvm::Type *> types;
  for (auto *operand : operands)
    types.push_back(operand->getType());

  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *llvmIntr =
      getNamedIntrinsicDeclaration(module, intrinsicName, resTy, types);
  return builder.CreateCall(llvmIntr, operands);
}

} // namespace xilinx::xllvm

#include "aie/Dialect/XLLVM/IR/XLLVMEnums.cpp.inc"

#define GET_OP_CLASSES
#include "aie/Dialect/XLLVM/IR/XLLVMOps.cpp.inc"
