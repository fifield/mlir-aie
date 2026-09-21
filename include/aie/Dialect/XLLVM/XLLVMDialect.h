/* Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

//===- XLLVMDialect.h - External LLVM (xllvm) dialect --------------C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file defines the XLLVM dialect, containing LLVM intrinsic operations
// for an external LLVM compiler.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_XLLVMDIALECT_H
#define AIE_DIALECT_XLLVM_XLLVMDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/ThreadLocalCache.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

// AIE4 config-word enums must precede the op classes: the generated op
// accessors return these enum types by value.
#include "aie/Dialect/XLLVM/IR/XLLVMEnums.h.inc"

#include "aie/Dialect/XLLVM/AIE4CtrlRegResources.h"

#define GET_OP_CLASSES
#include "aie/Dialect/XLLVM/IR/XLLVMOps.h.inc"

#include "aie/Dialect/XLLVM/IR/XLLVMDialect.h.inc"

namespace llvm {

class CallInst;
class IRBuilderBase;
class StringRef;

} // namespace llvm

namespace mlir {

class Operation;

namespace LLVM {
class ModuleTranslation;
} // namespace LLVM

} // namespace mlir

namespace xilinx::xllvm {

llvm::CallInst *createExternalLLVMIntrinsicCall(
    llvm::IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &moduleTranslation, mlir::Operation *intrOp,
    llvm::StringRef intrinsicName);

/// Pack an AIE4 vector-multiplier configuration word from its typed fields.
///
/// Layout defined by the ISA, restricted to the fields AIE4 elementwise
/// multiply uses:
///   bit    0   zero_acc
///   bits 2:1   amode
///   bits 4:3   bmode
///
/// e.g. zeroAcc=true, amode=Float(1), bmode=Mode32x32(3) -> 0b11011 == 27.
uint32_t encodeAIE4MulConf(bool zeroAcc, AIE4AMode amode, AIE4BMode bmode);

/// Pack an integer MAC configuration using the encodeAIE4MulConf fields,
/// plus signedness bits 8 (RHS) and 9 (LHS).
/// zeroAcc=false, amode=Int, bmode=Mode16x16, both signed encodes 0x308.
uint32_t encodeAIE4MacConf(bool zeroAcc, AIE4AMode amode, AIE4BMode bmode,
                           AIE4OperandSign lhsSign, AIE4OperandSign rhsSign);

/// Like `createExternalLLVMIntrinsicCall`, but appends `conf` as a trailing
/// i32 argument. AIE4 config words live in MLIR *attributes* (so their fields
/// can be range-verified) rather than in operands, so they are not picked up
/// by the operand-forwarding path.
llvm::CallInst *
createAIE4ConfIntrinsicCall(llvm::IRBuilderBase &builder,
                            mlir::LLVM::ModuleTranslation &moduleTranslation,
                            mlir::Operation *intrOp,
                            llvm::StringRef intrinsicName, uint32_t conf);

/// Append attributes as trailing i32 arguments, e.g. SRS shift then sign.
llvm::CallInst *createAIE4ConfIntrinsicCall(
    llvm::IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &moduleTranslation, mlir::Operation *intrOp,
    llvm::StringRef intrinsicName, llvm::ArrayRef<uint32_t> trailingArgs);

/// Builds a lane-select call with operands (lhs, rhs, sign, 2, sel, 0).
/// `intrOp` supplies (lhs, rhs, sel); the literals encode VSEL and fporder.
llvm::CallInst *
createAIE4SelIntrinsicCall(llvm::IRBuilderBase &builder,
                           mlir::LLVM::ModuleTranslation &moduleTranslation,
                           mlir::Operation *intrOp,
                           llvm::StringRef intrinsicName, uint32_t sign);

} // namespace xilinx::xllvm

#endif // AIE_DIALECT_XLLVM_XLLVMDIALECT_H
