//===- TargetInfo.cpp - AIEVec target parsing ------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/Utils/TargetInfo.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace llvm;

namespace xilinx::aievec {

bool AIEVecTargetInfo::isAIE1() const { return arch == AIEVecArch::AIE1; }
bool AIEVecTargetInfo::isAIE2() const { return arch == AIEVecArch::AIE2; }
bool AIEVecTargetInfo::isAIE2P() const { return arch == AIEVecArch::AIE2P; }
bool AIEVecTargetInfo::isAIE2PS() const { return arch == AIEVecArch::AIE2PS; }
bool AIEVecTargetInfo::isAIE4() const { return arch == AIEVecArch::AIE4; }

// Single source of truth for accepted/rejected `aie-target` spellings.
std::optional<AIEVecTargetInfo> parseAIEVecTarget(StringRef spelling,
                                                  std::string &error) {
  if (spelling.empty() || spelling == "aie")
    return AIEVecTargetInfo{AIEVecArch::AIE1};
  if (spelling == "aieml" || spelling == "aie2")
    return AIEVecTargetInfo{AIEVecArch::AIE2};
  if (spelling == "aie2p")
    return AIEVecTargetInfo{AIEVecArch::AIE2P};
  if (spelling == "aie4" || spelling == "aie4b")
    return AIEVecTargetInfo{AIEVecArch::AIE4};
  if (spelling == "aie2ps")
    return AIEVecTargetInfo{AIEVecArch::AIE2PS};
  error = ("unknown AIE target '" + spelling + "'").str();
  return std::nullopt;
}

FailureOr<AIEVecTargetInfo> parseAIEVecTarget(StringRef spelling,
                                              Operation *op) {
  std::string error;
  std::optional<AIEVecTargetInfo> info = parseAIEVecTarget(spelling, error);
  if (!info) {
    op->emitError() << error;
    return failure();
  }
  return *info;
}

StringRef getCanonicalAIEVecTargetName(AIEVecTargetInfo info) {
  switch (info.arch) {
  case AIEVecArch::AIE1:
    return "aie";
  case AIEVecArch::AIE2:
    return "aie2";
  case AIEVecArch::AIE2P:
    return "aie2p";
  case AIEVecArch::AIE2PS:
    return "aie2ps";
  case AIEVecArch::AIE4:
    // canonicalize to the more explicit "aie4b" spelling.
    return "aie4b";
  }
  llvm_unreachable("unhandled AIEVecArch");
}

bool isVectorLoweringEnabled(AIEVecTargetInfo info) {
  return !info.isAIE2PS();
}

bool isAIEVecToLLVMLoweringEnabled(AIEVecTargetInfo info) {
  return !info.isAIE2PS();
}

bool isConvolutionFoldingEnabled(AIEVecTargetInfo info) {
  return info.isAIE2() || info.isAIE2P();
}

} // namespace xilinx::aievec
