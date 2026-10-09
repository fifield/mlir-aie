//===- AIE4EntryModes.h - AIE4 SRS entry-mode helpers -----------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry-mode contracts shared by core outlining and AIEVec-to-LLVM lowering.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIE4ENTRYMODES_H
#define AIE_DIALECT_AIEVEC_IR_AIE4ENTRYMODES_H

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace xilinx::aievec {

/// Function attribute: the crRnd ("rnd") / crSat ("sat") values guaranteed at
/// entry on every path from an AIE4 core.
inline constexpr llvm::StringLiteral kAIE4EntryModesAttr = "aie4.entry_modes";

inline std::string describeAIE4Rnd(int32_t mode) {
  return ("crRnd = " + llvm::Twine(mode) + " (" +
          stringifyAIE4RoundingMode(static_cast<AIE4RoundingMode>(mode)) + ")")
      .str();
}

inline std::string describeAIE4Sat(int32_t mode) {
  return ("crSat = " + llvm::Twine(mode) + " (" +
          stringifyAIE4SaturationMode(static_cast<AIE4SaturationMode>(mode)) +
          ")")
      .str();
}

// Internal func.call bodies are inspected separately. AIE4 intrinsics preserve
// crRnd/crSat except set.ctrl.reg; unknown calls and inline assembly may
// clobber.
inline bool mayWriteAIE4Modes(mlir::Operation *op) {
  // Standalone AIEVec conversion can run before AIE standard lowering turns
  // aie.debug into a call to the external debug_i32.
  if (op->getName().getStringRef() == "aie.debug")
    return true;
  if (llvm::isa<mlir::func::CallIndirectOp, mlir::LLVM::CallOp,
                mlir::LLVM::InlineAsmOp>(op))
    return true;
  if (auto ci = llvm::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op))
    return ci.getIntrin() == "llvm.aie4.set.ctrl.reg";
  auto call = llvm::dyn_cast<mlir::func::CallOp>(op);
  if (!call)
    return false;
  llvm::StringRef name = call.getCallee();
  if (name == "llvm.aie4.set.ctrl.reg")
    return true;
  if (name.starts_with("llvm.aie4."))
    return false;
  auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
      call, call.getCalleeAttr());
  return !callee || callee.isExternal();
}

// Visit `root` and each transitively called func.func or llvm.func body once.
inline void walkAIE4ModesAndCallees(
    mlir::Operation *root,
    llvm::function_ref<mlir::WalkResult(mlir::Operation *)> fn) {
  llvm::SmallVector<mlir::Operation *> worklist{root};
  llvm::SmallPtrSet<mlir::Operation *, 8> visited{root};
  while (!worklist.empty()) {
    mlir::Operation *region = worklist.pop_back_val();
    mlir::WalkResult r =
        region->walk([&](mlir::Operation *op) -> mlir::WalkResult {
          if (auto call = llvm::dyn_cast<mlir::func::CallOp>(op)) {
            auto callee =
                mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                    call, call.getCalleeAttr());
            if (callee && !callee.isExternal() && visited.insert(callee).second)
              worklist.push_back(callee);
          } else if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(op)) {
            // LLVM calls count as clobbers; also visit their bodies to find
            // SRS ops that depend on the caller's modes.
            if (auto symbol = call.getCalleeAttr()) {
              auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<
                  mlir::LLVM::LLVMFuncOp>(call, symbol);
              if (callee && !callee.isExternal() &&
                  visited.insert(callee).second)
                worklist.push_back(callee);
            }
          }
          return fn(op);
        });
    if (r.wasInterrupted())
      return;
  }
}

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_IR_AIE4ENTRYMODES_H
