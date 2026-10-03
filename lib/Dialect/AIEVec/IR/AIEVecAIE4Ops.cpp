//===- AIEVecAIE4Ops.cpp - AIE4 matrix MAC op verifiers ---------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Verifiers for AIE4 dense MMAC operations:
// aievec.aie4.matmul, aievec.aie4.matrix_region and aievec.aie4.mmac.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/XLLVM/AIE4MatrixState.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

namespace {

/// Operand shapes and element types of one dense MMAC format.
struct AIE4MatrixFormatInfo {
  Type elementType;    // A and B element type
  int64_t k;           // A row length and B rows
  Type accElementType; // accumulator element type
  bool isInteger;
};

constexpr int64_t kAIE4MmacColumns = 64;

AIE4MatrixFormatInfo getFormatInfo(MLIRContext *ctx, AIE4MatrixFormat format) {
  Builder b(ctx);
  switch (format) {
  case AIE4MatrixFormat::Bf16:
    return {b.getBF16Type(), 32, b.getF32Type(), false};
  case AIE4MatrixFormat::Int8:
    return {b.getIntegerType(8), 64, b.getI32Type(), true};
  case AIE4MatrixFormat::Int4:
    return {b.getIntegerType(4), 128, b.getI32Type(), true};
  case AIE4MatrixFormat::Fp8:
    return {Float8E4M3FNType::get(ctx), 64, b.getF32Type(), false};
  }
  llvm_unreachable("unknown AIE4 matrix format");
}

std::optional<AIE4MatrixFormat> getFormatForElementType(Type elementType) {
  if (elementType.isBF16())
    return AIE4MatrixFormat::Bf16;
  if (elementType.isSignlessInteger(8))
    return AIE4MatrixFormat::Int8;
  if (elementType.isSignlessInteger(4))
    return AIE4MatrixFormat::Int4;
  if (isa<Float8E4M3FNType>(elementType))
    return AIE4MatrixFormat::Fp8;
  return std::nullopt;
}

/// The single native B tile of `format`: vector<K x 64 x elem>.
VectorType getTileType(MLIRContext *ctx, AIE4MatrixFormat format) {
  AIE4MatrixFormatInfo info = getFormatInfo(ctx, format);
  return VectorType::get({info.k, kAIE4MmacColumns}, info.elementType);
}

/// Integer formats need an explicit signedness; floating formats reject it.
LogicalResult verifySignedness(Operation *op, bool isInteger, StringRef name,
                               BoolAttr attr) {
  if (isInteger && !attr)
    return op->emitOpError("integer format requires '") << name << "'";
  if (!isInteger && attr)
    return op->emitOpError("'") << name << "' applies only to integer formats";
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// AIE4MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult AIE4MatMulOp::verify() {
  auto lhsType = cast<VectorType>(getLhs().getType());
  auto rhsType = cast<VectorType>(getRhs().getType());
  auto accType = cast<VectorType>(getAcc().getType());

  if (lhsType.isScalable() || rhsType.isScalable() || accType.isScalable())
    return emitOpError("operands must be fixed-length vectors");
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      accType.getRank() != 2)
    return emitOpError("lhs, rhs and acc must be rank-2 vectors");

  Type elementType = lhsType.getElementType();
  std::optional<AIE4MatrixFormat> format = getFormatForElementType(elementType);
  if (!format)
    return emitOpError("unsupported lhs element type ")
           << elementType << "; expected bf16, i8, i4 or f8E4M3FN";
  if (rhsType.getElementType() != elementType)
    return emitOpError("rhs element type ")
           << rhsType.getElementType() << " must match lhs element type "
           << elementType;

  AIE4MatrixFormatInfo info = getFormatInfo(getContext(), *format);
  VectorType tileType = getTileType(getContext(), *format);
  if (rhsType != tileType)
    return emitOpError("rhs must be exactly one native B tile ")
           << tileType << ", got " << rhsType;

  int64_t m = lhsType.getDimSize(0);
  if (lhsType.getDimSize(1) != info.k)
    return emitOpError("lhs must have ")
           << info.k << " columns for " << elementType << ", got " << lhsType;

  auto expectedAcc =
      VectorType::get({m, kAIE4MmacColumns}, info.accElementType);
  if (accType != expectedAcc)
    return emitOpError("acc must be ")
           << expectedAcc << " for lhs " << lhsType << ", got " << accType;

  if (failed(verifySignedness(*this, info.isInteger, "lhs_signed",
                              getLhsSignedAttr())) ||
      failed(verifySignedness(*this, info.isInteger, "rhs_signed",
                              getRhsSignedAttr())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// AIE4MatrixRegionOp
//===----------------------------------------------------------------------===//

LogicalResult AIE4MatrixRegionOp::verify() {
  AIE4MatrixFormat format = getFormat();
  AIE4MatrixFormatInfo info = getFormatInfo(getContext(), format);
  VectorType tileType = getTileType(getContext(), format);
  if (getRhs().getType() != tileType)
    return emitOpError("rhs must be the ")
           << stringifyAIE4MatrixFormat(format) << " B tile " << tileType
           << ", got " << getRhs().getType();

  if (failed(verifySignedness(*this, info.isInteger, "rhs_signed",
                              getRhsSignedAttr())))
    return failure();

  // A nested region would replace the outer region's active matrix.
  if ((*this)->getParentOfType<AIE4MatrixRegionOp>())
    return emitOpError("must not be nested in another "
                       "'aievec.aie4.matrix_region'; there is one active "
                       "matrix");
  return success();
}

LogicalResult AIE4MatrixRegionOp::verifyRegions() {
  Block &body = getBody().front();
  if (body.getArgumentTypes() != getInits().getTypes())
    return emitOpError("entry block argument types must match the types of "
                       "'inits'");

  auto yield = dyn_cast<AIE4MatrixYieldOp>(body.getTerminator());
  if (!yield)
    return emitOpError("body must end with 'aievec.aie4.matrix_yield'");
  if (yield.getValues().getTypes() != getResults().getTypes())
    return emitOpError("yielded types must match the result types");

  // Reject nested operations that may clobber the matrix or execute outside
  // this region. Match LLVM ops by name and XLLVM writers by trait to avoid
  // linking those dialects into AIEVec.
  WalkResult walk = body.walk([&](Operation *op) {
    auto reject = [&](StringRef what, StringRef note) {
      InFlightDiagnostic diag = emitOpError("body must not contain ");
      diag << what << "; it may replace or clobber the active matrix";
      diag.attachNote(op->getLoc()) << note;
      return WalkResult::interrupt();
    };
    if (!op->isRegistered())
      return reject("unregistered operations", "unregistered operation here");
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return reject("isolated-from-above operations",
                    "isolated-from-above operation here");
    StringRef name = op->getName().getStringRef();
    if (isa<CallOpInterface>(op) || name == "llvm.call_intrinsic")
      return reject("calls", "call here");
    if (name == "llvm.inline_asm")
      return reject("inline assembly", "inline assembly here");
    if (op->hasTrait<xllvm::AIE4MatrixStateWriter>())
      return reject("matrix staging or commit operations",
                    "matrix-state write here");
    if (isa<AIE4MatMulOp>(op))
      return reject("'aievec.aie4.matmul', which stages its own matrix",
                    "matmul here");
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// AIE4MmacOp
//===----------------------------------------------------------------------===//

LogicalResult AIE4MmacOp::verify() {
  AIE4MatrixFormat format = getFormat();
  AIE4MatrixFormatInfo info = getFormatInfo(getContext(), format);
  StringRef formatName = stringifyAIE4MatrixFormat(format);

  auto rowType = VectorType::get({info.k}, info.elementType);
  if (getLhs().getType() != rowType)
    return emitOpError("lhs must be ")
           << rowType << " for format " << formatName << ", got "
           << getLhs().getType();

  auto accType = VectorType::get({kAIE4MmacColumns}, info.accElementType);
  if (Value acc = getAcc(); acc && acc.getType() != accType)
    return emitOpError("acc must be ")
           << accType << " for format " << formatName << ", got "
           << acc.getType();
  if (getResult().getType() != accType)
    return emitOpError("result must be ")
           << accType << " for format " << formatName << ", got "
           << getResult().getType();

  if (failed(verifySignedness(*this, info.isInteger, "lhs_signed",
                              getLhsSignedAttr())) ||
      failed(verifySignedness(*this, info.isInteger, "rhs_signed",
                              getRhsSignedAttr())))
    return failure();

  // Stop at isolation boundaries: a nested function can execute outside
  // the enclosing region and cannot rely on its resident matrix.
  AIE4MatrixRegionOp region;
  for (Operation *parent = (*this)->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if ((region = dyn_cast<AIE4MatrixRegionOp>(parent)))
      break;
    if (parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;
  }
  if (!region)
    return emitOpError("must be nested in an 'aievec.aie4.matrix_region' "
                       "that provides the active matrix");
  if (region.getFormat() != format)
    return emitOpError("format ")
           << formatName << " does not match the enclosing region's format "
           << stringifyAIE4MatrixFormat(region.getFormat());
  if (region.getRhsSignedAttr() != getRhsSignedAttr())
    return emitOpError(
        "'rhs_signed' must match the enclosing region's 'rhs_signed'");
  return success();
}
