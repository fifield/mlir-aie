//===- TargetInfo.h - AIEVec target parsing --------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file centralizes parsing of the `aie-target` string used by the
// AIEVec pipelines/passes into a single, well-defined representation. It is
// the single source of truth for accepted/rejected spellings so the four
// AIEVec passes (canonicalize-vector-for-aievec, lower-vector-to-aievec,
// optimize-aievec, convert-vector-to-aievec) stay in sync.
//
//===----------------------------------------------------------------------===//
#ifndef AIE_DIALECT_AIEVEC_UTILS_TARGETINFO_H
#define AIE_DIALECT_AIEVEC_UTILS_TARGETINFO_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace mlir {
class Operation;
} // namespace mlir

namespace xilinx::aievec {

// Deliberately NOT named AIEArch: xilinx::AIE::AIEArch already exists
// (AIEAttrs.td:) and several AIEVec translation units see both namespaces.
enum class AIEVecArch { AIE1, AIE2, AIE2P, AIE2PS, AIE4 };

struct AIEVecTargetInfo {
  AIEVecArch arch = AIEVecArch::AIE1;

  bool isAIE1() const;
  bool isAIE2() const;
  bool isAIE2P() const;
  bool isAIE2PS() const;
  bool isAIE4() const;
};

// Diagnostic-free core: fills `error` with a complete actionable sentence,
// returns nullopt. Callers own emission.
std::optional<AIEVecTargetInfo> parseAIEVecTarget(llvm::StringRef spelling,
                                                  std::string &error);

// Adapter for the four in-pass call sites, which all have an Operation*.
mlir::FailureOr<AIEVecTargetInfo> parseAIEVecTarget(llvm::StringRef spelling,
                                                    mlir::Operation *op);

llvm::StringRef getCanonicalAIEVecTargetName(AIEVecTargetInfo info);

// Pipeline-readiness gates
bool isVectorLoweringEnabled(AIEVecTargetInfo info);
bool isAIEVecToLLVMLoweringEnabled(AIEVecTargetInfo info);
bool isConvolutionFoldingEnabled(AIEVecTargetInfo info); // AIE2 family only

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_UTILS_TARGETINFO_H
