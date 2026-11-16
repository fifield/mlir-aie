//===-VectorToAIEVecConversions.cpp - Vector to AIEVec convs. ---*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains conversions from the Vector dialect into the AIEVec
// dialect. Conversions assume that the Vector dialect has been rectricted
// to ops that can be translated to a sequence of valid AIEVec ops.
//===----------------------------------------------------------------------===//

#include "AIE2PShufflePlanner.h"

#include "AIE4MaskLowering.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecMatMul.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/Transforms/AIE4Staging.h"
#include "aie/Dialect/AIEVec/Utils/TargetInfo.h"

#include "aie/Dialect/AIEVec/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"
#include <bitset>
#include <optional>
#include <tuple>

#define DEBUG_TYPE "lower-vector-to-aievec"

using namespace llvm;
using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// A scalar i32 -> i8/i16 trunci's [clamp] chain as one SRS: the value
// before the clamp, the sign of the saturation (1 signed, 0 unsigned) and the
// clamp's ops, if any. nullopt if the clamp is not the destination type's
// full range.
struct ScalarSRSChain {
  Value source;
  int32_t sign;
  arith::MinSIOp minOp;
  arith::MaxSIOp maxOp;
};
static std::optional<ScalarSRSChain>
matchScalarSRSChain(arith::TruncIOp truncOp) {
  auto dstIntType = dyn_cast<IntegerType>(truncOp.getOut().getType());
  if (!dstIntType)
    return std::nullopt;
  unsigned dstBits = dstIntType.getWidth();
  if (dstBits != 8 && dstBits != 16)
    return std::nullopt;
  auto srcIntType = dyn_cast<IntegerType>(truncOp.getIn().getType());
  if (!srcIntType || srcIntType.getWidth() != 32)
    return std::nullopt;

  // Walk backward through optional clamp chain
  Value source = truncOp.getIn();
  arith::MinSIOp minOp = nullptr;
  arith::MaxSIOp maxOp = nullptr;
  if (auto minsiOp = source.getDefiningOp<arith::MinSIOp>()) {
    if (auto maxsiOp = minsiOp.getLhs().getDefiningOp<arith::MaxSIOp>()) {
      minOp = minsiOp;
      maxOp = maxsiOp;
      source = maxOp.getLhs();
    }
  } else if (auto maxsiOp = source.getDefiningOp<arith::MaxSIOp>()) {
    if (auto minsiOp = maxsiOp.getLhs().getDefiningOp<arith::MinSIOp>()) {
      maxOp = maxsiOp;
      minOp = minsiOp;
      source = minOp.getLhs();
    }
  }
  if (!minOp || !maxOp)
    return ScalarSRSChain{source, 1, nullptr, nullptr};

  // A clamp must be a valid saturation range
  std::optional<int64_t> loVal = getConstantIntValue(maxOp.getRhs());
  std::optional<int64_t> hiVal = getConstantIntValue(minOp.getRhs());
  if (!loVal || !hiVal)
    return std::nullopt;
  uint64_t one = 1ULL;
  int64_t unsignedHi = static_cast<int64_t>((one << dstBits) - 1);
  int64_t signedLo = -static_cast<int64_t>(one << (dstBits - 1));
  int64_t signedHi = static_cast<int64_t>((one << (dstBits - 1)) - 1);
  if (*loVal == 0 && *hiVal == unsignedHi)
    return ScalarSRSChain{source, 0, minOp, maxOp};
  if (*loVal == signedLo && *hiVal == signedHi)
    return ScalarSRSChain{source, 1, minOp, maxOp};
  return std::nullopt;
}

// Check if a TruncIOp is part of a shrsi+[clamp]+trunc chain that can be
// lowered to a compound SRS pattern. Only returns true for chains that the
// ShiftClampTruncToSRSPattern rewrite will actually match:
// - shrsi → trunci (no clamp)
// - shrsi → maxsi → minsi → trunci (full clamp pair)
// - shrsi → minsi → maxsi → trunci (reversed clamp pair)
// Does NOT return true for a single min or max without a matching pair,
// to avoid marking trunci illegal when no rewrite can handle it.
static bool isSRSCompoundCandidate(arith::TruncIOp trunciOp) {
  // LowerScalarShiftClampTruncToSRS converts fewer scalar chains than match
  // the shapes below.
  if (!isa<VectorType>(trunciOp.getType()) && !matchScalarSRSChain(trunciOp))
    return false;

  Value source = trunciOp.getIn();

  // Case 1: direct shrsi → trunci
  if (source.getDefiningOp<arith::ShRSIOp>())
    return true;

  // Case 2: minsi(maxsi(shrsi(...))) → trunci
  if (auto minsiOp = source.getDefiningOp<arith::MinSIOp>()) {
    if (auto maxsiOp = minsiOp.getLhs().getDefiningOp<arith::MaxSIOp>()) {
      if (maxsiOp.getLhs().getDefiningOp<arith::ShRSIOp>())
        return true;
    }
  }

  // Case 3: maxsi(minsi(shrsi(...))) → trunci (reversed order)
  if (auto maxsiOp = source.getDefiningOp<arith::MaxSIOp>()) {
    if (auto minsiOp = maxsiOp.getLhs().getDefiningOp<arith::MinSIOp>()) {
      if (minsiOp.getLhs().getDefiningOp<arith::ShRSIOp>())
        return true;
    }
  }

  // Case 4: minsi(maxsi(x)) → trunci (clamp-only, no shrsi)
  // Used for saturating clamp after skip-add (e.g., skip_scale=0).
  if (auto minsiOp = source.getDefiningOp<arith::MinSIOp>()) {
    if (minsiOp.getLhs().getDefiningOp<arith::MaxSIOp>())
      return true;
  }

  // Case 5: maxsi(minsi(x)) → trunci (reversed clamp-only, no shrsi)
  if (auto maxsiOp = source.getDefiningOp<arith::MaxSIOp>()) {
    if (maxsiOp.getLhs().getDefiningOp<arith::MinSIOp>())
      return true;
  }

  return false;
}

// Check if a ShRSIOp's result feeds into a trunci (possibly through
// clamp ops), meaning the compound pattern will consume it.
static bool shrsiUsedByCompoundSRS(arith::ShRSIOp rsOp) {
  for (Operation *user : rsOp->getUsers()) {
    // Direct: shrsi → trunci (validate full chain via isSRSCompoundCandidate)
    if (auto truncOp = dyn_cast<arith::TruncIOp>(user))
      if (isSRSCompoundCandidate(truncOp))
        return true;
    // Through clamp: shrsi → maxsi → minsi → trunci
    //            or: shrsi → minsi → maxsi → trunci
    if (isa<arith::MaxSIOp, arith::MinSIOp>(user)) {
      for (Operation *user2 : user->getUsers()) {
        if (auto truncOp2 = dyn_cast<arith::TruncIOp>(user2))
          if (isSRSCompoundCandidate(truncOp2))
            return true;
        if (isa<arith::MaxSIOp, arith::MinSIOp>(user2)) {
          for (Operation *user3 : user2->getUsers()) {
            if (auto truncOp3 = dyn_cast<arith::TruncIOp>(user3))
              if (isSRSCompoundCandidate(truncOp3))
                return true;
          }
        }
      }
    }
  }
  return false;
}

// Check if a scalar maxsi/minsi is sandwiched in a compound SRS chain
// (between shrsi and trunci). Used to keep the op legal so the scalar
// compound SRS pattern can consume the entire chain.
static bool scalarClampInCompoundSRS(Operation *op) {
  if (!isa<arith::MaxSIOp, arith::MinSIOp>(op))
    return false;
  // Only apply to scalar types
  if (isa<VectorType>(op->getResult(0).getType()))
    return false;
  for (Operation *user : op->getUsers()) {
    if (auto truncOp = dyn_cast<arith::TruncIOp>(user)) {
      if (isSRSCompoundCandidate(truncOp))
        return true;
    }
    if (isa<arith::MaxSIOp, arith::MinSIOp>(user)) {
      for (Operation *user2 : user->getUsers()) {
        if (auto truncOp2 = dyn_cast<arith::TruncIOp>(user2)) {
          if (isSRSCompoundCandidate(truncOp2))
            return true;
        }
      }
    }
  }
  return false;
}

// The value at the bottom of a chain of vector.shape_casts that ends at the
// type it started from, or `v` itself. Flattening an op to rank 1 leaves such
// round trips between it and the ops that use it.
static Value skipShapeCastRoundTrip(Value v) {
  Value src = v;
  while (auto castOp = src.getDefiningOp<vector::ShapeCastOp>())
    src = castOp.getSource();
  return src.getType() == v.getType() ? src : v;
}

// Given a Value, if it is defined by a widening op (arith:ExtSIOp,
// arith::ExtUIOp, arith::ExtFOp, aievec::UPSOp + aievec::SRSOp,
// aievec::UPSOp + aievec::CastOp), return the source of the widening op.
static std::optional<Value> getSourceOfWideningOp(Value src) {
  src = skipShapeCastRoundTrip(src);
  if (auto extSIOp = src.getDefiningOp<arith::ExtSIOp>())
    return extSIOp.getIn();
  if (auto extUIOp = src.getDefiningOp<arith::ExtUIOp>())
    return extUIOp.getIn();
  if (auto extFOp = src.getDefiningOp<arith::ExtFOp>())
    return extFOp.getIn();
  if (auto srsOp = src.getDefiningOp<aievec::SRSOp>()) {
    // Conversion through AIE intrinsics takes two steps:
    //     1) Load to accumulator: aievec.ups
    //     2) Move from accumulator: aievec.srs
    auto srsSource = srsOp.getSource();
    if (srsSource)
      if (auto upsOp = srsSource.getDefiningOp<aievec::UPSOp>())
        return upsOp.getSource();
  }
  if (auto castOp = src.getDefiningOp<aievec::CastOp>()) {
    // Conversion through AIE intrinsics can also take the following two steps:
    //     1) Load to accumulator: aievec.ups
    //     2) Move from accumulator: aievec.cast
    auto castSource = castOp.getSource();
    if (castSource)
      if (auto upsOp = castSource.getDefiningOp<aievec::UPSOp>())
        return upsOp.getSource();
  }
  return std::optional<Value>();
}

// The bf16 scalar that the f32 vector `src` splats, where `src` is a splat
// of a scalar widened from bf16, possibly already lowered to aievec
// broadcasts; else null.
static Value getSplatOfWidenedBF16(Value src) {
  src = skipShapeCastRoundTrip(src);
  if (auto concatOp = src.getDefiningOp<aievec::ConcatOp>()) {
    auto parts = concatOp.getSources();
    if (parts.empty() || !llvm::all_equal(parts))
      return {};
    src = parts.front();
  }
  Value scalar;
  if (auto bcastOp = src.getDefiningOp<vector::BroadcastOp>())
    scalar = bcastOp.getSource();
  else if (auto bcastOp = src.getDefiningOp<aievec::BroadcastScalarOp>())
    scalar = bcastOp.getSource();
  if (!scalar || isa<VectorType>(scalar.getType()))
    return {};
  if (auto extFOp = scalar.getDefiningOp<arith::ExtFOp>())
    if (extFOp.getIn().getType().isBF16())
      return extFOp.getIn();
  return {};
}

// Whether the f32 vector `src` holds bf16 values: it is widened from bf16, a
// splat of a scalar widened from bf16, or a constant that is exact in bf16.
static bool holdsBF16(Value src) {
  if (getSplatOfWidenedBF16(src))
    return true;
  if (auto wideSrc = getSourceOfWideningOp(src))
    return getElementTypeOrSelf(wideSrc->getType()).isBF16();
  DenseFPElementsAttr dense;
  if (!matchPattern(src, m_Constant(&dense)) || !dense.getElementType().isF32())
    return false;
  return llvm::all_of(dense.getValues<APFloat>(), [](APFloat v) {
    bool losesInfo = false;
    v.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  });
}

// The bf16 values held by `src`, which must satisfy holdsBF16. A constant is
// narrowed where it is used rather than folded to a bf16 constant, so it does
// not occupy a vector register outside the code that uses it.
static Value getBF16Source(OpBuilder &builder, Location loc, Value src) {
  auto bf16Type = VectorType::get(cast<VectorType>(src.getType()).getShape(),
                                  builder.getBF16Type());
  if (Value scalar = getSplatOfWidenedBF16(src))
    return vector::BroadcastOp::create(builder, loc, bf16Type, scalar);
  if (auto wideSrc = getSourceOfWideningOp(src))
    return *wideSrc;
  auto shift =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
  return aievec::SRSOp::create(builder, loc, bf16Type, src, shift);
}

// Given a Value, if it is defined by an integer widening op, return whether
// that widening is signed: arith.extsi -> signed, arith.extui -> unsigned, and
// an aievec.srs (from an already-rewritten extsi/extui) -> its own `sign`
// attribute. Returns std::nullopt when the defining op is not a recognized
// integer widening op, or carries no signedness: a floating-point arith.extf,
// or an aievec.ups/aievec.cast (which do not encode signedness). Callers record
// this on the created matmul op, since peeling the widening op would otherwise
// discard it (MLIR integers are signless).
static std::optional<bool> getSignednessOfWideningOp(Value src) {
  // Look through shape casts, which may sit between the extension op and the
  // consumer.
  while (auto castOp = src.getDefiningOp<vector::ShapeCastOp>())
    src = castOp.getSource();
  if (src.getDefiningOp<arith::ExtSIOp>())
    return true;
  if (src.getDefiningOp<arith::ExtUIOp>())
    return false;
  // An arith.extsi/extui may already have been rewritten into an
  // aievec.ups + aievec.srs pair by another pattern in this same conversion.
  // Only aievec.srs records a signedness; aievec.ups and aievec.cast do not.
  if (auto srsOp = src.getDefiningOp<aievec::SRSOp>())
    return srsOp.getSign() != 0;
  return std::nullopt;
}

// Given a Value, if it is defined by a narrowing op (arith::TruncFOp,
// arith::TruncIOp), return the source of the narrowing op.
static std::optional<Value> getSourceOfNarrowingOp(Value src) {
  if (auto truncFOp = src.getDefiningOp<arith::TruncFOp>())
    return truncFOp.getIn();
  if (auto truncIOp = src.getDefiningOp<arith::TruncIOp>())
    return truncIOp.getIn();
  return std::optional<Value>();
}

//===----------------------------------------------------------------------===//
// Type conversion utilities with narrowing/widening optimization awareness
//===----------------------------------------------------------------------===//

// Smart widen a value to target type. If the value was previously narrowed
// from the target type, reuse the original source to avoid truncf->extf chains.
static Value widenValueWithNarrowingCheck(Value val, Type targetType,
                                          Location loc,
                                          ConversionPatternRewriter &rewriter) {
  // Check if this value was narrowed from the target type
  if (auto narrowedSrc = getSourceOfNarrowingOp(val)) {
    if (narrowedSrc->getType() == targetType)
      return *narrowedSrc; // Reuse the original value (skip truncf->extf)
  }

  // Otherwise, create the widening op if needed
  if (val.getType() == targetType)
    return val;

  return arith::ExtFOp::create(rewriter, loc, targetType, val,
                               /*fastmath=*/nullptr);
}

// Result structure for smart narrowing operation
struct NarrowingResult {
  Value narrowedValue;     // The narrowed value (or original if optimized)
  bool skipNarrowing;      // True if we should skip creating truncf
  Operation *wideningUser; // The widening op to replace (if skipNarrowing)
};

// Smart narrow a value to target type. If the result will be immediately
// widened back, skip both truncf and extf operations.
static NarrowingResult
narrowValueWithWideningCheck(Operation *srcOp, Value val, Type targetType,
                             Location loc,
                             ConversionPatternRewriter &rewriter) {
  NarrowingResult result;
  result.narrowedValue = val;
  result.skipNarrowing = false;
  result.wideningUser = nullptr;

  // Check if srcOp will be immediately widened back
  if (srcOp->hasOneUse()) {
    Operation *user = *srcOp->getUsers().begin();
    if (auto extfOp = dyn_cast<arith::ExtFOp>(user)) {
      // The result will be widened - skip both truncf and extf
      result.skipNarrowing = true;
      result.wideningUser = extfOp;
      return result;
    }
  }

  // Normal case: create the narrowing op
  result.narrowedValue =
      arith::TruncFOp::create(rewriter, loc, targetType, val);
  return result;
}

// High-level helper to perform a binary operation on bf16 values in f32
// precision. This function handles:
// 1. Smart widening of operands (reuses f32 source if narrowed from f32)
// 2. Executing the operation in f32
// 3. Smart narrowing back to bf16 (skips truncf->extf if result is widened)
static void
performBF16BinaryOpInF32(Value lhs, Value rhs, Operation *srcOp, Location loc,
                         ConversionPatternRewriter &rewriter,
                         const std::function<Value(Value, Value)> &opBuilder) {
  Type f32Type = rewriter.getF32Type();

  // Smart widen both operands (reuse f32 source if narrowed from f32)
  Value lhsF32 = widenValueWithNarrowingCheck(lhs, f32Type, loc, rewriter);
  Value rhsF32 = widenValueWithNarrowingCheck(rhs, f32Type, loc, rewriter);

  // Perform operation in f32
  Value resultF32 = opBuilder(lhsF32, rhsF32);

  // Smart narrow back to bf16 (skip if result will be widened)
  auto narrowResult = narrowValueWithWideningCheck(
      srcOp, resultF32, lhs.getType(), loc, rewriter);

  if (narrowResult.skipNarrowing) {
    // Replace the widening user directly with f32 result
    rewriter.replaceOp(narrowResult.wideningUser, resultF32);
    rewriter.eraseOp(srcOp);
  } else {
    rewriter.replaceOp(srcOp, narrowResult.narrowedValue);
  }
}

// Given the LHS and RHS of an `arith::AddIOp`, if one of them is defined by an
// `arith::MulIOp`, return a tuple with the `lhs`, `rhs`, and `acc` of the MAC
// operation that can replace them.
static std::optional<std::tuple<Value, Value, Value>>
extractMACOperandsFromAddOperands(Value addLhs, Value addRhs) {
  auto *lhsDefOp = addLhs.getDefiningOp();
  auto *rhsDefOp = addRhs.getDefiningOp();
  arith::MulIOp mulOp = nullptr;
  Value acc;
  if (lhsDefOp) {
    mulOp = dyn_cast<arith::MulIOp>(lhsDefOp);
    acc = addRhs;
  }
  if (!mulOp && rhsDefOp) {
    mulOp = dyn_cast<arith::MulIOp>(rhsDefOp);
    acc = addLhs;
  }
  if (mulOp)
    return std::make_tuple(mulOp.getLhs(), mulOp.getRhs(), acc);

  // If the MulIOp has been already translated to aievec::aie1::MulOp:
  auto lhsSrsOp = addLhs.getDefiningOp<aievec::SRSOp>();
  auto rhsSrsOp = addRhs.getDefiningOp<aievec::SRSOp>();
  aievec::aie1::MulOp aieMulOp = nullptr;
  if (lhsSrsOp) {
    aieMulOp = lhsSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>();
    acc = addRhs;
  }
  if (!aieMulOp && rhsSrsOp) {
    aieMulOp = rhsSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>();
    acc = addLhs;
  }
  if (aieMulOp)
    return std::make_tuple(aieMulOp.getLhs(), aieMulOp.getRhs(), acc);
  return {};
}

// Given the LHS and RHS of an `arith::AddFOp`, if one of them is defined by an
// `arith::MulFOp`, return a tuple with the `lhs`, `rhs`, and `acc` of the FMA
// operation that can replace them.
static std::optional<std::tuple<Value, Value, Value>>
extractFMACOperandsFromAddOperands(Value addLhs, Value addRhs) {
  auto *lhsDefOp = addLhs.getDefiningOp();
  auto *rhsDefOp = addRhs.getDefiningOp();
  arith::MulFOp mulOp = nullptr;
  Value acc;
  if (lhsDefOp) {
    mulOp = dyn_cast<arith::MulFOp>(lhsDefOp);
    acc = addRhs;
  }
  if (!mulOp && rhsDefOp) {
    mulOp = dyn_cast<arith::MulFOp>(rhsDefOp);
    acc = addLhs;
  }
  if (mulOp)
    return std::make_tuple(mulOp.getLhs(), mulOp.getRhs(), acc);
  return {};
}

// Convert a input value to a target vector type. This function can insert
// multiple aievec ops depending on the combination of input and output vector
// types.
static std::optional<Value>
convertValueToTargetTypeAIE2(ConversionPatternRewriter &rewriter, Location loc,
                             Value inputVal, VectorType tgtType) {
  auto srcType = cast<VectorType>(inputVal.getType());
  auto srcElemType = srcType.getElementType();
  unsigned srcBitWidth = srcElemType.getIntOrFloatBitWidth();
  unsigned srcLaneSize = getVectorLaneSize(srcType);

  auto tgtElemType = tgtType.getElementType();
  unsigned tgtBitWidth = tgtElemType.getIntOrFloatBitWidth();
  unsigned tgtLaneSize = getVectorLaneSize(tgtType);

  if (srcType == tgtType)
    return inputVal;

  if ((srcElemType == tgtElemType) && (srcLaneSize != tgtLaneSize)) {
    // TODO: relax the condition below?
    if ((srcLaneSize == 16 && tgtLaneSize == 32 &&
         isa<FloatType>(srcElemType)) ||
        (srcLaneSize == 32 && tgtLaneSize == 64 &&
         isa<IntegerType>(srcElemType))) {
      auto zeroConstOp = arith::ConstantOp::create(
          rewriter, loc, srcType.getElementType(),
          rewriter.getZeroAttr(srcType.getElementType()));
      auto broadcastZeroOp = aievec::BroadcastScalarOp::create(
          rewriter, loc, tgtType, zeroConstOp->getResult(0));
      auto extOp = aievec::ExtOp::create(rewriter, loc, srcType,
                                         broadcastZeroOp->getResult(0), 0);

      SmallVector<Value> inputSources = {inputVal, extOp->getResult(0)};
      auto concatOp =
          aievec::ConcatOp::create(rewriter, loc, tgtType, inputSources);

      return concatOp.getResult();
    }
  } else if ((srcElemType != tgtElemType) && (srcLaneSize == tgtLaneSize) &&
             isa<IntegerType>(srcElemType) && isa<IntegerType>(tgtElemType)) {
    if (srcBitWidth == 16 && tgtBitWidth == 32 && srcLaneSize == 16) {
      // Case 1: vector<16xi16> to vector<16xi32> conversion by aievec.ups +
      // aievec.cast
      auto accType = getVectorOpDestType(srcType, /*AIE2 =*/true);
      auto upsOp = aievec::UPSOp::create(rewriter, loc, accType, inputVal);
      auto castOp = aievec::CastOp::create(
          rewriter, loc, tgtType, upsOp.getResult(), /*isResAcc*/ false);
      return castOp.getResult();
    }

    if (srcBitWidth == 8 && tgtBitWidth == 32 && srcLaneSize == 16) {
      // Case 2: vector<16xi8> to vector<16xi32> conversion by aievec.concat +
      // aievec.ups + aievec.cast + aievec.ext
      auto concatOutType = createVectorType(32, srcElemType);
      auto concatOp =
          aievec::ConcatOp::create(rewriter, loc, concatOutType,
                                   SmallVector<Value>({inputVal, inputVal}));
      auto accType = getVectorOpDestType(concatOutType, /*AIE2 =*/true);
      auto upsOp =
          aievec::UPSOp::create(rewriter, loc, accType, concatOp.getResult());
      auto castType = createVectorType(32, tgtElemType);
      auto castOp = aievec::CastOp::create(
          rewriter, loc, castType, upsOp.getResult(), /*isResAcc*/ false);
      auto extOp =
          aievec::ExtOp::create(rewriter, loc, tgtType, castOp.getResult(), 0);
      return extOp.getResult();
    }

    if (srcBitWidth == 8 && tgtBitWidth == 16 && srcLaneSize == 32) {
      // Case 3: vector<32xi8> to vector<32xi16> conversion by aievec.unpack
      auto unpackOp =
          aievec::UnpackOp::create(rewriter, loc, tgtType, inputVal);
      return unpackOp.getResult();
    }
  }

  return std::nullopt;
}

// Return the list of attributes that configure an `aievec.select` op to
// perform a rotation of the input vector by `rotation` number of elements.
// The attribute values depend on the vector type of the select operation.
static SmallVector<NamedAttribute>
buildAttributeListForRotationSelectOp(PatternRewriter &rewriter, VectorType vTy,
                                      int64_t rotation) {
  unsigned width = 0;
  auto elemTy = vTy.getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    width = intTy.getWidth();
  StringAttr attr0 = rewriter.getStringAttr("0");
  StringAttr attr0x06040200 = rewriter.getStringAttr("0x06040200");
  StringAttr attr0x0e0c0a08 = rewriter.getStringAttr("0x0e0c0a08");
  StringAttr attr0x2103 = rewriter.getStringAttr("0x2103");
  StringAttr attr0x3210 = rewriter.getStringAttr("0x3210");
  StringAttr selectAttrName = rewriter.getStringAttr("select");
  StringAttr xoffsetsAttrName = rewriter.getStringAttr("xoffsets");
  StringAttr xoffsetsHiAttrName = rewriter.getStringAttr("xoffsets_hi");
  StringAttr xsquareAttrName = rewriter.getStringAttr("xsquare");
  StringAttr xstartAttrName = rewriter.getStringAttr("xstart");
  StringAttr yoffsetsAttrName = rewriter.getStringAttr("yoffsets");
  StringAttr yoffsetsHiAttrName = rewriter.getStringAttr("yoffsets_hi");
  StringAttr ysquareAttrName = rewriter.getStringAttr("ysquare");
  StringAttr ystartAttrName = rewriter.getStringAttr("ystart");

  switch (width) {
  case 16: {
    if (rotation % 2) {
      int64_t xstart = rotation + 1;
      int64_t ystart = rotation - 1;
      return SmallVector<NamedAttribute, 9>(
          {{selectAttrName, rewriter.getStringAttr("0x11111111")},
           {xoffsetsAttrName, attr0x06040200},
           {xoffsetsHiAttrName, attr0x0e0c0a08},
           {xsquareAttrName, attr0x2103},
           {xstartAttrName, rewriter.getStringAttr(std::to_string(xstart))},
           {yoffsetsAttrName, rewriter.getStringAttr("0x0503010f")},
           {yoffsetsHiAttrName, rewriter.getStringAttr("0x0d0b0907")},
           {ysquareAttrName, attr0x2103},
           {ystartAttrName, rewriter.getStringAttr(std::to_string(ystart))}});
    }
    return SmallVector<NamedAttribute, 9>(
        {{selectAttrName, attr0},
         {xoffsetsAttrName, attr0x06040200},
         {xoffsetsHiAttrName, attr0x0e0c0a08},
         {xsquareAttrName, attr0x3210},
         {xstartAttrName, rewriter.getStringAttr(std::to_string(rotation))},
         {yoffsetsAttrName, attr0},
         {yoffsetsHiAttrName, attr0},
         {ysquareAttrName, attr0},
         {ystartAttrName, attr0}});
  }
  case 32:
    return SmallVector<NamedAttribute, 7>(
        {{selectAttrName, attr0},
         {xoffsetsAttrName, rewriter.getStringAttr("0x76543210")},
         {xsquareAttrName, attr0x3210},
         {xstartAttrName, rewriter.getStringAttr(std::to_string(rotation))},
         {yoffsetsAttrName, attr0},
         {ysquareAttrName, attr0},
         {ystartAttrName, attr0}});
  default:
    llvm::report_fatal_error("Unexpected width!");
  }

  return {};
}

namespace xilinx::aievec {

// Also called from AIEVecOptimizations.cpp via a forward declaration there;
// not static.
SmallVector<NamedAttribute>
// NOLINTNEXTLINE(misc-use-internal-linkage)
buildFMAOpSplatAttrForElemTy(aievec::aie1::FMAOp fmaOp, int64_t bcastPos,
                             int64_t step = 1) {
  unsigned width = 0;
  auto elemTy = fmaOp.getLhs().getType().getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    width = intTy.getWidth();
  auto *ctx = fmaOp.getContext();
  switch (width) {
  case 16:
    // NOTE: The pattern is:
    //       acc[0]  = x[0]  * z[bcastPos] + x[16] * z[bcastPos+step]
    //       acc[1]  = x[1]  * z[bcastPos] + x[17] * z[bcastPos+step]
    //       acc[2]  = x[2]  * z[bcastPos] + x[18] * z[bcastPos+step]
    //       acc[3]  = x[3]  * z[bcastPos] + x[19] * z[bcastPos+step]
    //       acc[4]  = x[4]  * z[bcastPos] + x[20] * z[bcastPos+step]
    //       acc[5]  = x[5]  * z[bcastPos] + x[21] * z[bcastPos+step]
    //       acc[6]  = x[6]  * z[bcastPos] + x[22] * z[bcastPos+step]
    //       acc[7]  = x[7]  * z[bcastPos] + x[23] * z[bcastPos+step]
    //       acc[8]  = x[8]  * z[bcastPos] + x[24] * z[bcastPos+step]
    //       acc[9]  = x[9]  * z[bcastPos] + x[25] * z[bcastPos+step]
    //       acc[10] = x[10] * z[bcastPos] + x[26] * z[bcastPos+step]
    //       acc[11] = x[11] * z[bcastPos] + x[27] * z[bcastPos+step]
    //       acc[12] = x[12] * z[bcastPos] + x[28] * z[bcastPos+step]
    //       acc[13] = x[13] * z[bcastPos] + x[29] * z[bcastPos+step]
    //       acc[14] = x[14] * z[bcastPos] + x[30] * z[bcastPos+step]
    //       acc[15] = x[15] * z[bcastPos] + x[31] * z[bcastPos+step]
    return SmallVector<NamedAttribute, 11>(
        {{fmaOp.getXstartAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getXoffsetsAttrName(), StringAttr::get(ctx, "0x73727170")},
         {fmaOp.getXoffsetsHiAttrName(), StringAttr::get(ctx, "0x77767574")},
         {fmaOp.getXstepAttrName(), fmaOp.getXstepAttr()},
         {fmaOp.getXsquareAttrName(), StringAttr::get(ctx, "0x3120")},
         {fmaOp.getZstartAttrName(),
          StringAttr::get(ctx, std::to_string(bcastPos))},
         {fmaOp.getZoffsetsAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getZoffsetsHiAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getZstepAttrName(), StringAttr::get(ctx, std::to_string(step))},
         {fmaOp.getZsquareAttrName(), fmaOp.getZsquareAttr()},
         {fmaOp.getFmsubAttrName(), fmaOp.getFmsubAttr()}});
  case 32:
    return SmallVector<NamedAttribute, 11>(
        {{fmaOp.getXstartAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getXoffsetsAttrName(), StringAttr::get(ctx, "0x76543210")},
         {fmaOp.getXoffsetsHiAttrName(), fmaOp.getXoffsetsHiAttr()},
         {fmaOp.getXstepAttrName(), fmaOp.getXstepAttr()},
         {fmaOp.getXsquareAttrName(), fmaOp.getXsquareAttr()},
         {fmaOp.getZstartAttrName(),
          StringAttr::get(ctx, std::to_string(bcastPos))},
         {fmaOp.getZoffsetsAttrName(), StringAttr::get(ctx, "0x00000000")},
         {fmaOp.getZoffsetsHiAttrName(), fmaOp.getZoffsetsHiAttr()},
         {fmaOp.getZstepAttrName(), fmaOp.getZstepAttr()},
         {fmaOp.getZsquareAttrName(), fmaOp.getZsquareAttr()},
         {fmaOp.getFmsubAttrName(), fmaOp.getFmsubAttr()}});
  default:
    llvm::report_fatal_error("Unexpected width!");
  }

  return {};
}

} // namespace xilinx::aievec

template <typename SrcOpTy, typename AIEv2ElemOp>
static LogicalResult genAddElemAIE2(ConversionPatternRewriter &rewriter,
                                    Value lval, Value rval, VectorType srcType,
                                    SrcOpTy srcOp) {
  auto lCastOp = aievec::CastOp::create(rewriter, srcOp.getLoc(), srcType, lval,
                                        /*isResAcc*/ true);
  auto rCastOp = aievec::CastOp::create(rewriter, srcOp.getLoc(), srcType, rval,
                                        /*isResAcc*/ true);
  auto elemOp = AIEv2ElemOp::create(
      rewriter, srcOp.getLoc(), lCastOp->getResult(0).getType(),
      lCastOp->getResult(0), rCastOp->getResult(0));
  rewriter.replaceOpWithNewOp<aievec::CastOp>(
      srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
  return success();
}

static arith::CmpIPredicate
convertToIntegerPredicate(arith::CmpFPredicate pred) {
  switch (pred) {
  case CmpFPredicate::UEQ:
  case CmpFPredicate::OEQ:
    return CmpIPredicate::eq;
  case CmpFPredicate::UGT:
    return CmpIPredicate::ugt;
  case CmpFPredicate::OGT:
    return CmpIPredicate::sgt;
  case CmpFPredicate::UGE:
    return CmpIPredicate::uge;
  case CmpFPredicate::OGE:
    return CmpIPredicate::sge;
  case CmpFPredicate::ULT:
    return CmpIPredicate::ult;
  case CmpFPredicate::OLT:
    return CmpIPredicate::slt;
  case CmpFPredicate::ULE:
    return CmpIPredicate::ule;
  case CmpFPredicate::OLE:
    return CmpIPredicate::sle;
  case CmpFPredicate::UNE:
  case CmpFPredicate::ONE:
    return CmpIPredicate::ne;
  default:
    llvm::report_fatal_error("Unexpected predicate!");
  }
}

static arith::CmpIPredicate
convertToIntegerPredicate(arith::CmpIPredicate pred) {
  return pred;
}

static aievec::CmpOp createCmpOpAIE2(ConversionPatternRewriter &rewriter,
                                     CmpIPredicate pred, Location loc,
                                     Type type, Value lhs, Value rhs) {
  switch (pred) {
  case CmpIPredicate::eq:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "eq");
  case CmpIPredicate::ne:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ne");
  case CmpIPredicate::slt:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "slt");
  case CmpIPredicate::ult:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ult");
  case CmpIPredicate::sle:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "sle");
  case CmpIPredicate::ule:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ule");
  case CmpIPredicate::sgt:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "sgt");
  case CmpIPredicate::ugt:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ugt");
  case CmpIPredicate::sge:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "sge");
  case CmpIPredicate::uge:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "uge");
  }
  return nullptr;
}

template <typename DstOpTy>
static aievec::ExtElemOp
generateAIEVecOpsForReductionOp(ConversionPatternRewriter &rewriter,
                                vector::ReductionOp srcOp, int shiftIndex,
                                Value curValue) {
  assert(shiftIndex > 0 && (shiftIndex & (shiftIndex - 1)) == 0 &&
         "shiftIndex must be power of 2");

  Location loc = srcOp.getLoc();
  auto vType = cast<VectorType>(curValue.getType());
  Type scalarType = vType.getElementType();
  Type vecType = curValue.getType();
  DstOpTy curOp = nullptr;
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();

  for (int id = shiftIndex; id > 0; id /= 2) {
    auto constOp = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(id * elWidth / 8));

    auto shiftBytesOp = aievec::ShiftOp::create(
        rewriter, loc, vecType, curValue, curValue, constOp.getResult());

    curOp = DstOpTy::create(rewriter, loc, vecType, curValue,
                            shiftBytesOp.getResult());

    curValue = curOp.getResult();
  }

  auto zeroConstOp =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  return aievec::ExtElemOp::create(rewriter, loc, scalarType, curOp,
                                   zeroConstOp.getResult());
}

// Helper to pad a v16bf16 vector to v32bf16 by concatenating with a splat of
// the given infinity value. Used by min/max reduction patterns.
// Returns {paddedVector, newLaneSize}.
static std::pair<Value, unsigned>
padV16ToV32WithInfinity(ConversionPatternRewriter &rewriter, Location loc,
                        Value inputVec, Type scalarType, bool negativeInf) {
  VectorType v32bf16Type = createVectorType(32, scalarType);
  VectorType v16bf16Type = createVectorType(16, scalarType);

  // Create a scalar infinity constant
  auto infAttr = rewriter.getFloatAttr(
      scalarType,
      APFloat::getInf(cast<FloatType>(scalarType).getFloatSemantics(),
                      negativeInf));
  auto splatInf = arith::ConstantOp::create(rewriter, loc, infAttr).getResult();

  // Broadcast to v32bf16, then extract upper half (which is also infinity)
  auto infVec =
      aievec::BroadcastScalarOp::create(rewriter, loc, v32bf16Type, splatInf);
  auto infUpperHalf =
      aievec::ExtOp::create(rewriter, loc, v16bf16Type, infVec, 1);

  // Concatenate input with infinity padding
  Value paddedVec =
      aievec::ConcatOp::create(rewriter, loc, v32bf16Type,
                               ValueRange{inputVec, infUpperHalf.getResult()});

  return {paddedVec, 32};
}

// Helper to pad a v16bf16 vector to v32bf16 by concatenating with zeros.
// Used by elementwise min/max/cmp/sel patterns for 256-bit bf16 support.
static Value padV16ToV32WithZeros(ConversionPatternRewriter &rewriter,
                                  Location loc, Value inputVec,
                                  Type scalarType) {
  VectorType v32Type = createVectorType(32, scalarType);
  VectorType v16Type = createVectorType(16, scalarType);
  auto zeroAttr = rewriter.getZeroAttr(v16Type);
  auto zeroVec = arith::ConstantOp::create(rewriter, loc, zeroAttr);
  return aievec::ConcatOp::create(rewriter, loc, v32Type,
                                  ValueRange{inputVec, zeroVec.getResult()});
}

static func::FuncOp getOrInsertFuncDecl(ConversionPatternRewriter &rewriter,
                                        Operation *parentSymbolTableOp,
                                        StringRef funcName, TypeRange inTypes,
                                        TypeRange outTypes) {

  mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(
      &parentSymbolTableOp->getRegions().front().getBlocks().front());
  SymbolTable st = SymbolTable(parentSymbolTableOp);
  func::FuncOp fnOpLookup = st.lookup<func::FuncOp>(funcName);
  func::FuncOp fnOp;
  // if the function is already declared, use the existing function, don't
  // declare multiple times
  if (fnOpLookup != nullptr) {
    fnOp = fnOpLookup;
  } else {
    StringAttr t1 = rewriter.getStringAttr("sym_visibility");
    StringAttr t2 = rewriter.getStringAttr("private");
    NamedAttribute funcAccess = NamedAttribute(t1, t2);
    FunctionType fnType =
        mlir::FunctionType::get(rewriter.getContext(), inTypes, outTypes);
    fnOp = func::FuncOp::create(rewriter, parentSymbolTableOp->getLoc(),
                                funcName, fnType, funcAccess);
  }
  return fnOp;
}

//===----------------------------------------------------------------------===//
// Wide vector splitting utility
//===----------------------------------------------------------------------===//

// Utility function to split a wide vector operation (e.g., v32bf16) into two
// half-width operations (e.g., v16bf16) and concatenate the results.
// This pattern is common for AIE2 when hardware only supports 16-lane
// operations but we need to process 32-lane vectors.
//
// Template parameters:
//   SrcOpTy - The source operation type (e.g., arith::MulFOp, math::ExpOp)
//
// Parameters:
//   srcOp - The source operation to replace
//   wideInputs - The wide input values (e.g., lhs and rhs for binary ops)
//   halfType - The half-width vector type (e.g., vector<16xbf16>)
//   wideType - The wide vector type (e.g., vector<32xbf16>)
//   rewriter - The pattern rewriter
//   processHalves - Callback that processes the lower and upper halves
//                   and returns a pair of half-width results to be concatenated
//
// The callback signature is:
//   std::pair<Value, Value>(ArrayRef<std::pair<Value, Value>> halfInputs,
//                           Location loc, ConversionPatternRewriter &rewriter)
// where halfInputs[i] is {lowerHalf, upperHalf} for each wideInput
template <typename SrcOpTy, typename Func>
static void splitWideVectorOp(SrcOpTy srcOp, ArrayRef<Value> wideInputs,
                              VectorType halfType, VectorType wideType,
                              ConversionPatternRewriter &rewriter,
                              Func &&processHalves) {

  Location loc = srcOp.getLoc();

  // Extract lower and upper halves for each wide input
  SmallVector<std::pair<Value, Value>> halfInputs;
  halfInputs.reserve(wideInputs.size());
  for (Value wideInput : wideInputs) {
    auto lowerHalf =
        aievec::ExtOp::create(rewriter, loc, halfType, wideInput, 0);
    auto upperHalf =
        aievec::ExtOp::create(rewriter, loc, halfType, wideInput, 1);
    halfInputs.emplace_back(lowerHalf.getResult(), upperHalf.getResult());
  }

  // Process halves using the callback
  auto [lowResult, highResult] = processHalves(halfInputs, loc, rewriter);

  // Concatenate results
  SmallVector<Value> concatSources = {lowResult, highResult};
  rewriter.replaceOpWithNewOp<aievec::ConcatOp>(srcOp, wideType, concatSources);
}

// Simplified version for unary operations
template <typename SrcOpTy>
static void splitWideUnaryVectorOp(
    SrcOpTy srcOp, Value wideInput, VectorType halfType, VectorType wideType,
    ConversionPatternRewriter &rewriter,
    std::function<Value(Value, Location, ConversionPatternRewriter &)>
        processHalf) {

  splitWideVectorOp<SrcOpTy>(
      srcOp, {wideInput}, halfType, wideType, rewriter,
      [&processHalf](ArrayRef<std::pair<Value, Value>> halfInputs, Location loc,
                     ConversionPatternRewriter &rewriter) {
        auto [lowerHalf, upperHalf] = halfInputs[0];
        Value lowResult = processHalf(lowerHalf, loc, rewriter);
        Value highResult = processHalf(upperHalf, loc, rewriter);
        return std::make_pair(lowResult, highResult);
      });
}

// Simplified version for binary operations
template <typename SrcOpTy>
static void splitWideBinaryVectorOp(
    SrcOpTy srcOp, Value lhs, Value rhs, VectorType halfType,
    VectorType wideType, ConversionPatternRewriter &rewriter,
    std::function<Value(Value, Value, Location, ConversionPatternRewriter &)>
        processHalf) {

  splitWideVectorOp<SrcOpTy>(
      srcOp, {lhs, rhs}, halfType, wideType, rewriter,
      [&processHalf](ArrayRef<std::pair<Value, Value>> halfInputs, Location loc,
                     ConversionPatternRewriter &rewriter) {
        auto [lhsLow, lhsHigh] = halfInputs[0];
        auto [rhsLow, rhsHigh] = halfInputs[1];
        Value lowResult = processHalf(lhsLow, rhsLow, loc, rewriter);
        Value highResult = processHalf(lhsHigh, rhsHigh, loc, rewriter);
        return std::make_pair(lowResult, highResult);
      });
}

//===----------------------------------------------------------------------===//
// Math operation matching utilities
//===----------------------------------------------------------------------===//

// Check if math.exp op matches AIE2 LUT-based exp constraints
static bool matchExpOpForAIE2LUT(math::ExpOp::Adaptor adaptor) {
  auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());

  if (!srcType)
    return false;

  Type scalarType = srcType.getElementType();
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();
  unsigned laneSize = getVectorLaneSize(srcType);
  // AIE2 LUT-based exp: supports v16bf16 and v32bf16
  return isa<FloatType>(scalarType) && (laneSize == 16 || laneSize == 32) &&
         elWidth == 16;
}

// Check if math.exp op matches AIE2P exp constraints
static bool matchExpOpForAIE2P(math::ExpOp::Adaptor adaptor) {
  auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());

  if (!srcType)
    return false;

  Type scalarType = srcType.getElementType();
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();
  unsigned laneSize = getVectorLaneSize(srcType);
  // AIE2P exp: supports v16bf16 and v32bf16
  return scalarType.isBF16() && (laneSize == 16 || laneSize == 32) &&
         elWidth == 16;
}

// Check if a vector type matches the AIE2P tanh and exp2 constraints. These
// intrinsics take an accfloat operand, so f32 vectors map onto them as well as
// bf16.
static bool isAIE2PAccFloatUnaryType(VectorType type) {
  if (type.isScalable())
    return false;
  Type elemType = type.getElementType();
  unsigned laneSize = getVectorLaneSize(type);
  return (elemType.isBF16() || elemType.isF32()) &&
         (laneSize == 16 || laneSize == 32);
}

// Check if math.tanh, math.erf or math.exp2 of `type` is lowered for AIE2P: a
// bf16 or f32 scalar, one-lane vector, or vector matching
// isAIE2PAccFloatUnaryType.
static bool isAIE2PAccFloatUnaryOrScalarType(Type type) {
  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType)
    return type.isBF16() || type.isF32();
  if (!vecType.isScalable() && vecType.getRank() == 1 &&
      vecType.getNumElements() == 1) {
    Type elemType = vecType.getElementType();
    return elemType.isBF16() || elemType.isF32();
  }
  return isAIE2PAccFloatUnaryType(vecType);
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

// This pattern fold `vector.extract` and `vector.broadcast` into
// `aievec.broadcast` for AIE2
struct FoldVectorExtractAndSplatToAIEBroadcast
    : OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto extOp = adaptor.getSource().getDefiningOp<vector::ExtractOp>();

    if (!extOp)
      return failure();

    auto src = extOp.getSource();
    auto pos = extOp.getStaticPosition();
    int64_t posVal = pos[0];
    auto srcVecType = cast<VectorType>(src.getType());
    auto resultType = cast<VectorType>(bcastOp.getResult().getType());
    if (srcVecType != resultType) {
      if (srcVecType.getNumElements() != 2 * resultType.getNumElements())
        return failure();
      auto half = static_cast<int8_t>(posVal / resultType.getNumElements());
      posVal -= half * resultType.getNumElements();
      src = aievec::ExtOp::create(rewriter, extOp.getLoc(), resultType, src,
                                  rewriter.getI8IntegerAttr(half))
                .getResult();
    }

    unsigned elWidth = resultType.getElementType().getIntOrFloatBitWidth();

    if (unsigned laneSize = getVectorLaneSize(resultType);
        laneSize * elWidth == 512) {
      // Common use case for the broadcast_elem intrinsic
      rewriter.replaceOpWithNewOp<aievec::BroadcastOp>(bcastOp, resultType, src,
                                                       posVal);
    } else if (laneSize * elWidth == 256) {
      // e.g. need v16bf16 due to the subsequent v16accfloat operation
      VectorType aievecBcastType =
          createVectorType(512 / elWidth, resultType.getElementType());
      auto concatOp =
          aievec::ConcatOp::create(rewriter, bcastOp.getLoc(), aievecBcastType,
                                   SmallVector<Value>({src, src}));
      auto aieBcastOp = aievec::BroadcastOp::create(
          rewriter, bcastOp.getLoc(), aievecBcastType, concatOp.getResult(),
          posVal);
      rewriter.replaceOpWithNewOp<aievec::ExtOp>(bcastOp, resultType,
                                                 aieBcastOp.getResult(), 0);
    } else if (laneSize * elWidth == 1024) {
      // e.g. need v32int32 due to the subsequent v32acc32 operation
      VectorType aievecBcastType =
          createVectorType(512 / elWidth, resultType.getElementType());
      auto half = static_cast<int8_t>(posVal / resultType.getNumElements());
      posVal -= half * resultType.getNumElements();
      auto extOp =
          aievec::ExtOp::create(rewriter, bcastOp.getLoc(), aievecBcastType,
                                src, rewriter.getI8IntegerAttr(half));
      auto aieBcastOp = aievec::BroadcastOp::create(rewriter, bcastOp.getLoc(),
                                                    aievecBcastType,
                                                    extOp.getResult(), posVal);
      rewriter.replaceOpWithNewOp<aievec::ConcatOp>(
          bcastOp, resultType,
          SmallVector<Value>({aieBcastOp.getResult(), aieBcastOp.getResult()}));
    } else {
      return failure();
    }

    return success();
  }
};

struct ConvertSplatToAIEBroadcast : OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (adaptor.getSource().getDefiningOp<vector::ExtractOp>())
      return failure();

    auto resultType = cast<VectorType>(bcastOp.getResult().getType());
    auto flatResultType = getFlattenedVectorType(resultType);
    Type scalarType = resultType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    auto src = bcastOp.getSource();

    if (laneSize * elWidth == 512) {
      Value newOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), flatResultType, src);
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    if (laneSize * elWidth == 256) {
      VectorType vecType = createVectorType(512 / elWidth, scalarType);
      auto aieBcastOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), vecType, src);
      Value newOp =
          aievec::ExtOp::create(rewriter, bcastOp.getLoc(), flatResultType,
                                aieBcastOp.getResult(), 0);
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    if (laneSize * elWidth == 1024) {
      VectorType vecType = createVectorType(512 / elWidth, scalarType);
      auto aieBcastOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), vecType, src);
      Value newOp = aievec::ConcatOp::create(
          rewriter, bcastOp.getLoc(), flatResultType,
          SmallVector<Value>({aieBcastOp.getResult(), aieBcastOp.getResult()}));
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    return failure();
  }
};

// This pattern replaces `arith.muli`+`arith.addi` on vectors with
// `aievec.mac_elem`. This pattern works for AIE2.
struct ConvertMulAddToAIEVecFMAElemOpPattern
    : OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertMulAddToAIEVecFMAElemOpPattern(MLIRContext *context,
                                        unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    auto resultType = dyn_cast<VectorType>(addOp.getType());
    if (!resultType)
      return failure();

    // Verify it can be replaced by a MAC
    auto res =
        extractMACOperandsFromAddOperands(adaptor.getLhs(), adaptor.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    // Verify the vector type is supported by AIE2
    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || resultElWidth != 16) &&
        (laneSize != 16 || resultElWidth != 32))
      return failure();

    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIE2 =*/true);
    auto upsOp = aievec::UPSOp::create(rewriter, addOp.getLoc(), accType, acc,
                                       shiftParam);
    auto fmaElemOp = aievec::FMAElemOp::create(
        rewriter, addOp.getLoc(), accType, lhs, rhs, upsOp.getResult(),
        /*fmsub=*/false);

    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, addOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        addOp, resultType, fmaElemOp.getResult(), shiftParamOp.getResult());

    return success();
  }

  unsigned shiftParam;
};

// Lower a single 16-lane bf16 FMA half: UPS -> FMAElem -> SRS.
static Value lowerBF16FMAHalf(Value lhs, Value rhs, Value acc,
                              unsigned shiftParam, Location loc,
                              ConversionPatternRewriter &rewriter) {
  auto f32AccType = VectorType::get({16}, rewriter.getF32Type());
  auto upsOp =
      aievec::UPSOp::create(rewriter, loc, f32AccType, acc, shiftParam);
  auto fmaElemOp = aievec::FMAElemOp::create(rewriter, loc, f32AccType, lhs,
                                             rhs, upsOp.getResult(),
                                             /*fmsub=*/false);
  auto shiftParamOp = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(shiftParam));
  auto srsOp =
      aievec::SRSOp::create(rewriter, loc, cast<VectorType>(lhs.getType()),
                            fmaElemOp.getResult(), shiftParamOp.getResult());
  return srsOp.getResult();
}

// Convert `vector.fma` to `aievec.mac_elem`. Supported operand types:
// `vector<16xf32>`, `vector<16xbf16>`, and `vector<32xbf16>` (split into two
// 16-lane FMAs). In the case of vectors with
// `f32` elemental type, this pattern will try to match `bf16` to `f32`
// widening ops in the `lhs` and `rhs` operands, or fail otherwise.
// TODO: When sign extensions are not found, a conversion from `f32` to `bf16`
// TODO: can be inserted to emulate `f32` fma with `bf16` logic.
struct ConvertVectorFMAOpToAIEVecFMAElemOpPattern
    : OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertVectorFMAOpToAIEVecFMAElemOpPattern(MLIRContext *context,
                                             unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify the vector type is supported by AIE2
    auto resVecTy = cast<VectorType>(fmaOp.getType());
    auto resElemTy = resVecTy.getElementType();
    unsigned numElems = getVectorLaneSize(resVecTy);

    // Only support f32 and bf16 with 16 or 32 lanes.
    if ((!resElemTy.isF32() && !resElemTy.isBF16()) ||
        (numElems != 16 && numElems != 32))
      return rewriter.notifyMatchFailure(
          fmaOp, "Unsupported operand types in vector.fma lowering.");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value acc = adaptor.getAcc();

    if (resElemTy.isF32()) {
      if (!holdsBF16(lhs) || !holdsBF16(rhs))
        return rewriter.notifyMatchFailure(
            fmaOp, "vector.fma operands are f32 and do not hold bf16 values; "
                   "can't lower to aievec.");
      Location loc = fmaOp.getLoc();
      lhs = getBF16Source(rewriter, loc, lhs);
      rhs = getBF16Source(rewriter, loc, rhs);
      if (numElems == 16) {
        rewriter.replaceOpWithNewOp<aievec::FMAElemOp>(
            fmaOp, acc.getType(), lhs, rhs, acc, /*fmsub=*/false);
        return success();
      }
      // 32 lanes: two 16-lane halves.
      auto bf16HalfTy = createVectorType(16, rewriter.getBF16Type());
      auto f32HalfTy = createVectorType(16, rewriter.getF32Type());
      SmallVector<Value> halves;
      for (int idx : {0, 1}) {
        Value lhsHalf =
            aievec::ExtOp::create(rewriter, loc, bf16HalfTy, lhs, idx);
        Value rhsHalf =
            aievec::ExtOp::create(rewriter, loc, bf16HalfTy, rhs, idx);
        Value accHalf =
            aievec::ExtOp::create(rewriter, loc, f32HalfTy, acc, idx);
        halves.push_back(aievec::FMAElemOp::create(rewriter, loc, f32HalfTy,
                                                   lhsHalf, rhsHalf, accHalf,
                                                   /*fmsub=*/false));
      }
      rewriter.replaceOpWithNewOp<aievec::ConcatOp>(fmaOp, resVecTy, halves);
      return success();
    }

    // Handle vector<32xbf16> by splitting into two vector<16xbf16> FMAs
    if (numElems == 32 && resElemTy.isBF16()) {
      VectorType halfType = createVectorType(16, resElemTy);
      unsigned localShiftParam = shiftParam;

      splitWideVectorOp<vector::FMAOp>(
          fmaOp, {lhs, rhs, acc}, halfType, resVecTy, rewriter,
          [localShiftParam](ArrayRef<std::pair<Value, Value>> halfInputs,
                            Location loc, ConversionPatternRewriter &rewriter) {
            auto [lhsLow, lhsHigh] = halfInputs[0];
            auto [rhsLow, rhsHigh] = halfInputs[1];
            auto [accLow, accHigh] = halfInputs[2];

            Value lowResult = lowerBF16FMAHalf(lhsLow, rhsLow, accLow,
                                               localShiftParam, loc, rewriter);
            Value highResult = lowerBF16FMAHalf(lhsHigh, rhsHigh, accHigh,
                                                localShiftParam, loc, rewriter);
            return std::make_pair(lowResult, highResult);
          });
      return success();
    }

    acc = aievec::UPSOp::create(rewriter, fmaOp.getLoc(),
                                VectorType::get({16}, rewriter.getF32Type()),
                                acc, shiftParam);
    Value newOp =
        aievec::FMAElemOp::create(rewriter, fmaOp.getLoc(), acc.getType(), lhs,
                                  rhs, acc, /*fmsub=*/false);
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, fmaOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
    newOp = aievec::SRSOp::create(rewriter, fmaOp.getLoc(), resVecTy, newOp,
                                  shiftParamOp);

    rewriter.replaceOp(fmaOp, newOp);

    return success();
  }

  unsigned shiftParam;
};

// AIE2P multiplies 32 bf16 lanes into an f32 accumulator in one op. Lower a
// 32-lane f32 `vector.fma` with bf16 inputs to it.
struct ConvertVectorFMAOpToAIEVecFMAElemOpAIE2PPattern
    : OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resVecTy = cast<VectorType>(fmaOp.getType());
    if (!resVecTy.getElementType().isF32() || resVecTy.getRank() != 1 ||
        resVecTy.getNumElements() != 32 || !holdsBF16(adaptor.getLhs()) ||
        !holdsBF16(adaptor.getRhs()))
      return failure();
    Location loc = fmaOp.getLoc();
    Value lhs = getBF16Source(rewriter, loc, adaptor.getLhs());
    Value rhs = getBF16Source(rewriter, loc, adaptor.getRhs());
    rewriter.replaceOpWithNewOp<aievec::FMAElemOp>(
        fmaOp, resVecTy, lhs, rhs, adaptor.getAcc(), /*fmsub=*/false);
    return success();
  }
};

// The multiply-only form of the pattern above, for 32 or 64 lanes.
struct ConvertMulFToAIEVecMulElemOpAIE2PPattern
    : OpConversionPattern<arith::MulFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulFOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resVecTy = dyn_cast<VectorType>(mulOp.getType());
    if (!resVecTy || !resVecTy.getElementType().isF32() ||
        resVecTy.getRank() != 1 ||
        (resVecTy.getNumElements() != 32 && resVecTy.getNumElements() != 64) ||
        !holdsBF16(adaptor.getLhs()) || !holdsBF16(adaptor.getRhs()))
      return failure();
    Location loc = mulOp.getLoc();
    Value lhs = getBF16Source(rewriter, loc, adaptor.getLhs());
    Value rhs = getBF16Source(rewriter, loc, adaptor.getRhs());
    auto mulElemOp =
        aievec::MulElemOp::create(rewriter, loc, resVecTy, lhs, rhs);
    rewriter.replaceOpWithNewOp<aievec::CastOp>(mulOp, resVecTy, mulElemOp,
                                                /*isResAcc=*/false);
    return success();
  }
};

// This pattern fuses `arith.mulf` + `arith.addf` on bf16 vectors into
// `aievec.mac_elem` (float FMA). This pattern works for AIE2.
struct ConvertMulAddFToAIEVecFMAElemOpPattern
    : OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertMulAddFToAIEVecFMAElemOpPattern(MLIRContext *context,
                                         unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::AddFOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    auto resultType = dyn_cast<VectorType>(addOp.getType());
    if (!resultType)
      return failure();

    // Only handle bf16 element type
    auto elemType = resultType.getElementType();
    if (!elemType.isBF16())
      return failure();

    // Verify it can be replaced by an FMA
    auto res =
        extractFMACOperandsFromAddOperands(adaptor.getLhs(), adaptor.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    unsigned laneSize = getVectorLaneSize(resultType);

    // Handle vector<32xbf16> by splitting into two vector<16xbf16> FMAs
    if (laneSize == 32) {
      VectorType halfType = createVectorType(16, elemType);
      unsigned localShiftParam = shiftParam;

      splitWideVectorOp<arith::AddFOp>(
          addOp, {lhs, rhs, acc}, halfType, resultType, rewriter,
          [localShiftParam](ArrayRef<std::pair<Value, Value>> halfInputs,
                            Location loc, ConversionPatternRewriter &rewriter) {
            auto [lhsLow, lhsHigh] = halfInputs[0];
            auto [rhsLow, rhsHigh] = halfInputs[1];
            auto [accLow, accHigh] = halfInputs[2];

            Value lowResult = lowerBF16FMAHalf(lhsLow, rhsLow, accLow,
                                               localShiftParam, loc, rewriter);
            Value highResult = lowerBF16FMAHalf(lhsHigh, rhsHigh, accHigh,
                                                localShiftParam, loc, rewriter);
            return std::make_pair(lowResult, highResult);
          });
      return success();
    }

    // Handle vector<16xbf16>
    if (laneSize != 16)
      return failure();

    auto f32AccType = VectorType::get({16}, rewriter.getF32Type());
    auto upsOp = aievec::UPSOp::create(rewriter, addOp.getLoc(), f32AccType,
                                       acc, shiftParam);
    auto fmaElemOp = aievec::FMAElemOp::create(
        rewriter, addOp.getLoc(), f32AccType, lhs, rhs, upsOp.getResult(),
        /*fmsub=*/false);

    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, addOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        addOp, resultType, fmaElemOp.getResult(), shiftParamOp.getResult());

    return success();
  }

  unsigned shiftParam;
};

// This pattern replaces `arith.mulf` on vectors with
// `aievec.mul_elem`. This pattern works for AIE2.
struct ConvertMulFToAIEVecMulElemOpPattern
    : OpConversionPattern<arith::MulFOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertMulFToAIEVecMulElemOpPattern(MLIRContext *context,
                                      unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::MulFOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    auto resultType = dyn_cast<VectorType>(mulOp.getType());
    if (!resultType)
      return failure();

    // Skip standalone mul conversion when this MulFOp feeds into an AddFOp as
    // its sole user. ConvertMulAddFToAIEVecFMAElemOpPattern will fuse the
    // multiply and add into a single aievec.mac_elem.
    // Only defer to FMA for bf16 where the FMA pattern is registered.
    auto isAddOp = [&](Operation *op) { return isa<arith::AddFOp>(op); };
    if (resultType.getElementType().isBF16() && mulOp->hasOneUse() &&
        llvm::any_of(mulOp->getUsers(), isAddOp))
      return failure();

    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();

    unsigned laneSize = getVectorLaneSize(resultType);

    // f32 product of bf16 inputs, in 16-lane pieces.
    if (resultType.getElementType().isF32() &&
        (laneSize == 16 || laneSize == 32) && holdsBF16(adaptor.getLhs()) &&
        holdsBF16(adaptor.getRhs())) {
      Location loc = mulOp.getLoc();
      Value lhs = getBF16Source(rewriter, loc, adaptor.getLhs());
      Value rhs = getBF16Source(rewriter, loc, adaptor.getRhs());
      auto bf16HalfTy = createVectorType(16, rewriter.getBF16Type());
      auto f32HalfTy = createVectorType(16, rewriter.getF32Type());
      auto mul16 = [&](Value l, Value r) -> Value {
        auto mulElemOp =
            aievec::MulElemOp::create(rewriter, loc, f32HalfTy, l, r);
        return aievec::CastOp::create(rewriter, loc, f32HalfTy, mulElemOp,
                                      /*isResAcc=*/false);
      };
      if (laneSize == 16) {
        rewriter.replaceOp(mulOp, mul16(lhs, rhs));
        return success();
      }
      SmallVector<Value> halves;
      for (int idx : {0, 1})
        halves.push_back(
            mul16(aievec::ExtOp::create(rewriter, loc, bf16HalfTy, lhs, idx),
                  aievec::ExtOp::create(rewriter, loc, bf16HalfTy, rhs, idx)));
      rewriter.replaceOpWithNewOp<aievec::ConcatOp>(mulOp, resultType, halves);
      return success();
    }

    // Handle vector<32xbf16> by splitting into two vector<16xbf16> operations
    if (laneSize == 32 && resultElWidth == 16) {
      VectorType halfType = createVectorType(16, resultType.getElementType());
      unsigned localShiftParam = shiftParam;

      splitWideBinaryVectorOp<arith::MulFOp>(
          mulOp, adaptor.getLhs(), adaptor.getRhs(), halfType, resultType,
          rewriter,
          [localShiftParam](Value lhsHalf, Value rhsHalf, Location loc,
                            ConversionPatternRewriter &rewriter) -> Value {
            Type accType = getVectorOpDestType(
                cast<VectorType>(lhsHalf.getType()), /*AIE2 =*/true);
            auto mulElemOp = aievec::MulElemOp::create(rewriter, loc, accType,
                                                       lhsHalf, rhsHalf);
            auto shiftParamOp = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(localShiftParam));
            auto srsOp = aievec::SRSOp::create(
                rewriter, loc, cast<VectorType>(lhsHalf.getType()),
                mulElemOp.getResult(), shiftParamOp.getResult());
            return srsOp.getResult();
          });
      return success();
    }

    // bfloat16 and float type (laneSize == 16)
    if (laneSize != 16 || (resultElWidth != 16 && resultElWidth != 32))
      return failure();

    // Decide the accType for aievec.mul_elem based on mulOp's lhs & rhs
    auto lval = adaptor.getLhs();
    auto rval = adaptor.getRhs();
    lval = getSourceOfWideningOp(lval).value_or(lval);
    rval = getSourceOfWideningOp(rval).value_or(rval);
    auto lSrcType = cast<VectorType>(lval.getType());
    auto rSrcType = cast<VectorType>(rval.getType());
    unsigned lBitWidth = lSrcType.getElementType().getIntOrFloatBitWidth();
    unsigned rBitWidth = rSrcType.getElementType().getIntOrFloatBitWidth();
    Type accType = getVectorOpDestType(lSrcType, /*AIE2 =*/true);
    if (rBitWidth > lBitWidth) {
      accType = getVectorOpDestType(rSrcType, /*AIE2 =*/true);
    }
    // Only support the same lhs/rhs type at the moment
    if (lSrcType != rSrcType) {
      return failure();
    }

    // Prepare lhr/rhs for the aievec.mul_elem op
    unsigned bitWidth = (rBitWidth > lBitWidth) ? rBitWidth : lBitWidth;
    Type srcElemType = (rBitWidth > lBitWidth) ? rSrcType.getElementType()
                                               : lSrcType.getElementType();
    unsigned numLanes = 0;
    // float16/32 and int32 both land on 16 lanes -- genuinely distinct
    // source types happen to share a lane count, not a copy-paste clone.
    if (isa<FloatType>(srcElemType) &&
        (bitWidth == 16 || bitWidth == 32)) { // NOLINT(bugprone-branch-clone)
      numLanes = 16;
    } else if (isa<IntegerType>(srcElemType) &&
               (bitWidth == 8 || bitWidth == 16)) {
      numLanes = 32;
    } else if (isa<IntegerType>(srcElemType) && (bitWidth == 32)) {
      numLanes = 16;
    } else {
      return failure();
    }
    VectorType targetInputType = createVectorType(numLanes, srcElemType);
    if (targetInputType != lSrcType) {
      auto converted = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(),
                                                    lval, targetInputType);
      if (!converted)
        return failure();
      lval = *converted;
    }
    if (targetInputType != rSrcType) {
      auto converted = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(),
                                                    rval, targetInputType);
      if (!converted)
        return failure();
      rval = *converted;
    }
    if (!lval || !rval)
      return failure();

    // Create an aievec.mul_elem op
    auto mulElemOp = aievec::MulElemOp::create(rewriter, mulOp.getLoc(),
                                               accType, lval, rval);

    // Create an aievec.cast or an aievec.srs op
    auto mulElemResultType = mulElemOp.getType();
    auto mulElemResultElWidth =
        mulElemResultType.getElementType().getIntOrFloatBitWidth();

    if (mulElemResultElWidth == resultElWidth) {
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          mulOp, resultType, mulElemOp.getResult(), /*isResAcc*/ false);
    } else if (mulElemResultElWidth > resultElWidth) {
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, mulOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          mulOp, resultType, mulElemOp.getResult(), shiftParamOp.getResult());
    } else {
      return failure();
    }

    return success();
  }

  unsigned shiftParam;
};

// This pattern replaces `arith.muli` on vectors with
// `aievec.mul_elem`. This pattern works for AIE2.
struct ConvertMulIToAIEVecMulElemOpPattern
    : OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertMulIToAIEVecMulElemOpPattern(MLIRContext *context,
                                      unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::MulIOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    auto resultType = dyn_cast<VectorType>(mulOp.getType());
    if (!resultType)
      return failure();

    // FIXME: Verify it is not a part of MAC
    auto isAddOp = [&](Operation *op) { return isa<arith::AddIOp>(op); };
    if (mulOp->hasOneUse() && llvm::any_of(mulOp->getUsers(), isAddOp))
      return failure();

    // Verify the vector type is supported by AIE2
    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || (resultElWidth != 16 && resultElWidth != 8)) &&
        ((laneSize != 16 && laneSize != 32) || resultElWidth != 32))
      return failure();

    // Decide the accType for aievec.mul_elem based on mulOp's lhs & rhs
    auto lval = adaptor.getLhs();
    auto rval = adaptor.getRhs();

    lval = getSourceOfWideningOp(lval).value_or(lval);
    rval = getSourceOfWideningOp(rval).value_or(rval);

    auto lSrcType = cast<VectorType>(lval.getType());
    auto rSrcType = cast<VectorType>(rval.getType());
    unsigned lBitWidth = lSrcType.getElementType().getIntOrFloatBitWidth();
    unsigned rBitWidth = rSrcType.getElementType().getIntOrFloatBitWidth();
    Type accType = getVectorOpDestType(lSrcType, /*AIE2 =*/true);
    if (rBitWidth > lBitWidth) {
      accType = getVectorOpDestType(rSrcType, /*AIE2 =*/true);
    }

    // Prepare lhr/rhs for the aievec.mul_elem op
    unsigned bitWidth = (rBitWidth > lBitWidth) ? rBitWidth : lBitWidth;
    Type srcElemType = (rBitWidth > lBitWidth) ? rSrcType.getElementType()
                                               : lSrcType.getElementType();
    unsigned numLanes = 0;
    // float16/32 and int32 both land on 16 lanes -- genuinely distinct
    // source types happen to share a lane count, not a copy-paste clone.
    if (isa<FloatType>(srcElemType) &&
        (bitWidth == 16 || bitWidth == 32)) { // NOLINT(bugprone-branch-clone)
      numLanes = 16;
    } else if (isa<IntegerType>(srcElemType) &&
               (bitWidth == 8 || bitWidth == 16)) {
      numLanes = 32;
    } else if (isa<IntegerType>(srcElemType) && (bitWidth == 32)) {
      numLanes = 16;
    } else {
      return failure();
    }
    VectorType targetInputType = createVectorType(numLanes, srcElemType);
    if (targetInputType != lSrcType) {
      auto converted = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(),
                                                    lval, targetInputType);
      if (!converted)
        return failure();
      lval = *converted;
    }
    if (targetInputType != rSrcType) {
      auto converted = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(),
                                                    rval, targetInputType);
      if (!converted)
        return failure();
      rval = *converted;
    }
    if (!lval || !rval)
      return failure();

    // Create an aievec.mul_elem op
    auto mulElemOp = aievec::MulElemOp::create(rewriter, mulOp.getLoc(),
                                               accType, lval, rval);

    // Create an aievec.cast or an aievec.srs op
    auto mulElemResultType = mulElemOp.getType();
    auto mulElemResultElWidth =
        mulElemResultType.getElementType().getIntOrFloatBitWidth();

    if (mulElemResultElWidth == resultElWidth) {
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          mulOp, resultType, mulElemOp.getResult(), /*isResAcc*/ false);
    } else if (mulElemResultElWidth > resultElWidth) {
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, mulOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          mulOp, resultType, mulElemOp.getResult(), shiftParamOp.getResult());
    } else {
      return failure();
    }

    return success();
  }

  unsigned shiftParam;
};

// This pattern folds an extract + broadcast feeding into an
// `aievec::aie1::FMAOp` into the op, using the shuffle attributes.
struct FoldSplatToFMAOp : OpConversionPattern<aievec::aie1::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto concatOp =
        dyn_cast<aievec::ConcatOp>(adaptor.getLhs().getDefiningOp());
    if (!concatOp)
      return failure();
    vector::BroadcastOp bcastOp = nullptr;
    auto *concatDefOp = concatOp.getSources()[0].getDefiningOp();
    if (concatDefOp)
      bcastOp = dyn_cast<vector::BroadcastOp>(concatDefOp);
    Value lhs = adaptor.getRhs();
    if (!bcastOp) {
      bcastOp = dyn_cast<vector::BroadcastOp>(adaptor.getRhs().getDefiningOp());
      if (!bcastOp)
        return failure();
      lhs = concatOp.getSources()[0];
    }
    auto extOp =
        dyn_cast<vector::ExtractOp>(bcastOp.getSource().getDefiningOp());
    if (!extOp)
      return failure();

    auto rhs = extOp.getSource();
    auto concatVecType = cast<VectorType>(concatOp.getResult().getType());
    auto zvec =
        arith::ConstantOp::create(rewriter, concatOp.getLoc(), lhs.getType(),
                                  rewriter.getZeroAttr(lhs.getType()));
    auto lhsX2 =
        aievec::ConcatOp::create(rewriter, concatOp.getLoc(), concatVecType,
                                 SmallVector<Value, 2>({lhs, zvec}))
            .getResult();
    // XXX: We assume a 1D vector
    auto pos = extOp.getStaticPosition();
    int64_t zstart = pos[0];
    auto fmaOpAttr = buildFMAOpSplatAttrForElemTy(fmaOp, zstart);
    aievec::aie1::FMAOp::Properties fmaProps;
    if (failed(aievec::aie1::FMAOp::setPropertiesFromAttr(
            fmaProps, rewriter.getDictionaryAttr(fmaOpAttr),
            [&]() { return fmaOp.emitError(); })))
      return failure();
    rewriter.replaceOpWithNewOp<aievec::aie1::FMAOp>(
        fmaOp, TypeRange({fmaOp.getResult().getType()}),
        ValueRange({lhsX2, rhs, adaptor.getAcc()}), fmaProps);

    return success();
  }
};

struct ConvertMulAddToAIEVecFMAOpPattern
    : OpConversionPattern<aievec::aie1::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::AddOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecType = cast<VectorType>(addOp.getType());

    auto res =
        extractMACOperandsFromAddOperands(adaptor.getLhs(), adaptor.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    SmallVector<int64_t, 4> concatVecShape(vecType.getShape().begin(),
                                           vecType.getShape().end());
    concatVecShape[vecType.getRank() - 1] *= 2;
    auto concatVecType =
        VectorType::get(concatVecShape, vecType.getElementType());
    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIE2 =*/false);
    auto lhsX2 =
        aievec::ConcatOp::create(rewriter, addOp.getLoc(), concatVecType,
                                 SmallVector<Value, 2>(2, lhs))
            .getResult();
    auto upsOp = aievec::UPSOp::create(rewriter, addOp.getLoc(), accType, acc);
    auto fmaOp = aievec::aie1::FMAOp::create(
        rewriter, addOp.getLoc(), accType, lhsX2, rhs, upsOp.getResult(),
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xstep=*/"",
        /*xsquare=*/"", /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"",
        /*zstep=*/"", /*zsquare=*/"", /*fmsub=*/false);
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, addOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        addOp, vecType, fmaOp.getResult(), shiftParamOp.getResult());
    return success();
  }
};

// This pattern replaces `vector.transfer_read` with `aievec.upd`. Right now,
// it performs a naïve direct translation. This needs to be expanded to
// support more complex scenarios.
struct LowerVectorTransferReadToAIEUPD
    : OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LowerVectorTransferReadToAIEUPD(MLIRContext *context, int64_t minVectorSize,
                                  int64_t maxVectorSize, int64_t alignment,
                                  int64_t maxLoadSize)
      : OpConversionPattern(context), minVectorSize(minVectorSize),
        maxVectorSize(maxVectorSize), vectorAlignment(alignment),
        maxLoadSize(maxLoadSize) {}

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Masked loads
    if (readOp.getMask())
      return readOp.emitError() << "AIE doesn't support masked loads.";

    // Non-contiguous loads
    AffineMap map = readOp.getPermutationMap();
    if (!map.isMinorIdentity())
      return failure();

    // Splats
    if (map.isConstant())
      return failure();

    // Misaligned accesses
    auto vType = readOp.getVectorType();
    if (getTransferReadAlignmentOffset(adaptor, vType, vectorAlignment)
            .value_or(0) != 0)
      return failure();

    // Invalid vector size.
    // We can handle cases where the vector size is:
    //   1) the minimum vector size
    //   2) a square multiple of the alignment size and up to the maximum
    //      vector size.
    int64_t vSize = vType.getNumElements() * vType.getElementTypeBitWidth();
    if (vSize > maxVectorSize ||
        (vSize % vectorAlignment && vSize != minVectorSize))
      return failure();
    // We can deal with linked update instructions when the vector size is
    // exactly twice the load size. This could change in future architectures
    if (vSize > maxLoadSize && vSize != maxLoadSize * 2)
      return failure();
    int64_t multiplicity = vSize / vectorAlignment;
    if ((vSize > minVectorSize) && std::bitset<8>(multiplicity).count() != 1)
      return failure();

    auto updOp = xilinx::aievec::UPDOp::create(
        rewriter, readOp.getLoc(), vType, adaptor.getBase(),
        adaptor.getIndices(), 0, 0, TypedValue<VectorType>(nullptr));
    if (vSize > maxLoadSize) {
      updOp = xilinx::aievec::UPDOp::create(
          rewriter, readOp.getLoc(), vType, adaptor.getBase(),
          adaptor.getIndices(), maxLoadSize, 1, updOp.getResult());
    }
    rewriter.replaceOp(readOp, updOp.getResult());

    return success();
  }

  int64_t minVectorSize, maxVectorSize, vectorAlignment, maxLoadSize;
};

// XXX: Notice that this template doesn't verify that the vector element type
// XXX: is supported by the target architecture.
template <typename SrcOpTy, typename DstOpTy>
struct OneToOneVectorOpToAIEVecOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOpTy>(
        srcOp, srcOp.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xsquare=*/"",
        /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"", /*zsquare=*/"");
    return success();
  }
};

struct LowerVectorAddIOpToAIEVecAddOp : OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = addOp.getType();
    if (!isa<VectorType>(resType))
      return failure();

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto *lhsDefOp = lhs.getDefiningOp();
    auto *rhsDefOp = rhs.getDefiningOp();
    if ((isa_and_nonnull<arith::MulIOp>(lhsDefOp)) ||
        (isa_and_nonnull<arith::MulIOp>(rhsDefOp)))
      return failure();

    rewriter.replaceOpWithNewOp<aievec::aie1::AddOp>(
        addOp, resType, lhs, rhs,
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xsquare=*/"",
        /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"", /*zsquare=*/"");
    return success();
  }
};

using LowerVectorAddFOpToAIEVecAddOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::AddFOp, aievec::aie1::AddOp>;
using LowerVectorMulFOpToAIEVecMulOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::MulFOp, aievec::aie1::MulOp>;
using LowerVectorSubIOpToAIEVecSubOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::SubIOp, aievec::aie1::SubOp>;
using LowerVectorSubFOpToAIEVecSubOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::SubFOp, aievec::aie1::SubOp>;

struct LowerVectorMulIOpToAIEVecMulOp : OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::MulIOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = dyn_cast<VectorType>(mulOp.getType());
    if (!resTy)
      return failure();
    auto accTy = getVectorOpDestType(resTy, /*AIE2 =*/false);
    auto newMulOp = aievec::aie1::MulOp::create(
        rewriter, mulOp.getLoc(), accTy, adaptor.getLhs(), adaptor.getRhs());
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, mulOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        mulOp, resTy, newMulOp.getResult(), shiftParamOp.getResult());
    return success();
  }
};

template <typename SrcOpTy, typename DstOpTy>
struct LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp
    : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    // A set recording the vector lane size and element width we are supporting
    // for AIE2.
    llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
    laneSizeElWidthPairSet.insert({64, 8});
    laneSizeElWidthPairSet.insert({32, 16});
    laneSizeElWidthPairSet.insert({16, 32});
    laneSizeElWidthPairSet.insert({32, 32});

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsDefOp = lhs.getDefiningOp();
    auto rhsDefOp = rhs.getDefiningOp();
    // Check if this is part of a MAC/FMA pattern (mul + add).
    // We only skip conversion if BOTH operands could potentially be part of an
    // FMA pattern (i.e., neither is a constant). Constants can never be the
    // multiply result in an FMA, so we should allow conversion in those cases.
    bool lhsIsMul = lhsDefOp && (isa<arith::MulIOp>(lhsDefOp) ||
                                 isa<arith::MulFOp>(lhsDefOp));
    bool rhsIsMul = rhsDefOp && (isa<arith::MulIOp>(rhsDefOp) ||
                                 isa<arith::MulFOp>(rhsDefOp));
    bool lhsIsConst = isa_and_nonnull<arith::ConstantOp>(lhsDefOp);
    bool rhsIsConst = isa_and_nonnull<arith::ConstantOp>(rhsDefOp);

    // Defer to FMA/MAC patterns when a multiply feeds into add, UNLESS the
    // element type is f32 (where no FMA pattern exists). For bf16 and integer
    // types, FMA/MAC patterns handle the fusion.
    if (!resultType.getElementType().isF32() &&
        ((lhsIsMul && !rhsIsConst) || (rhsIsMul && !lhsIsConst)))
      return failure();

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    // Integer cases
    if (isa<IntegerType>(scalarType)) {
      if (!laneSizeElWidthPairSet.count(
              std::make_pair(laneSize, resultElWidth)))
        return failure();

      // If the ops are defined without extension ops and with supported data
      // type, the arith::AddI or arith::SubI can be directly replaced with
      // aievec::AddElem or aievec::SubElem.
      if (!lhsDefOp && !rhsDefOp) {
        if (laneSize * resultElWidth == 512) {
          rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(), lhs,
                                               rhs);
          return success();
        }
        return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs, resultType,
                                                srcOp);
      }

      // If element width is 32, we need to consider sign extension cases
      if (resultElWidth == 32) {
        auto lhsExt = getSourceOfWideningOp(lhs).value_or(nullptr);
        auto rhsExt = getSourceOfWideningOp(rhs).value_or(nullptr);

        if (!lhsExt && !rhsExt) {
          if (laneSize * resultElWidth == 512) {
            rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(), lhs,
                                                 rhs);
            return success();
          }
          return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                  resultType, srcOp);
        }

        if (lhsExt && rhsExt) {
          auto lval = lhsExt;
          auto rval = rhsExt;
          VectorType lSrcType = cast<VectorType>(lval.getType());

          Type accType = getVectorOpDestType(lSrcType, /*AIE2 =*/true);
          auto lUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
          auto rUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);
          auto elemOp = DstOpTy::create(
              rewriter, srcOp.getLoc(), lUpsOp->getResult(0).getType(),
              lUpsOp->getResult(0), rUpsOp->getResult(0));
          rewriter.replaceOpWithNewOp<aievec::CastOp>(
              srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
          return success();
        }

        if (!lhsExt || !rhsExt) {
          auto lval = lhsExt ? lhsExt : lhs;
          auto rval = rhsExt ? rhsExt : rhs;
          auto extVal = lhsExt ? lval : rval;
          VectorType vType = cast<VectorType>(extVal.getType());
          unsigned bitWidth = vType.getElementType().getIntOrFloatBitWidth();

          if (bitWidth != 8 && bitWidth != 16) {
            return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                    resultType, srcOp);
          }

          if (bitWidth * laneSize != 256) {
            return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                    resultType, srcOp);
          }

          Type accType = nullptr;

          if (bitWidth == 8) {
            accType = getVectorOpDestType(vType, /*AIE2 =*/true);
            Value valToUps = lhsExt ? lval : rval;
            Value valToCast = lhsExt ? rval : lval;
            auto upsOp = aievec::UPSOp::create(rewriter, srcOp.getLoc(),
                                               accType, valToUps);
            auto castOp =
                aievec::CastOp::create(rewriter, srcOp.getLoc(), resultType,
                                       valToCast, /*isResAcc*/ true);
            Value lhsToElemOp =
                lhsExt ? upsOp->getResult(0) : castOp->getResult(0);
            Value rhsToElemOp =
                lhsExt ? castOp->getResult(0) : upsOp->getResult(0);
            auto elemOp = DstOpTy::create(rewriter, srcOp.getLoc(),
                                          upsOp->getResult(0).getType(),
                                          lhsToElemOp, rhsToElemOp);
            rewriter.replaceOpWithNewOp<aievec::CastOp>(
                srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
            return success();
          }

          if (bitWidth == 16) {
            accType = getVectorOpDestType(resultType, /*AIE2 =*/true);
            auto lUpsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
            auto rUpsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);

            auto elemOp = DstOpTy::create(
                rewriter, srcOp.getLoc(), lUpsOp->getResult(0).getType(),
                lUpsOp->getResult(0), rUpsOp->getResult(0));

            auto shiftParamOp = arith::ConstantOp::create(
                rewriter, srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
            rewriter.replaceOpWithNewOp<aievec::SRSOp>(
                srcOp, srcOp.getType(), elemOp.getResult(),
                shiftParamOp.getResult());
            return success();
          }
        }
      } else {
        rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(), lhs, rhs);
        return success();
      }
    }
    // Float types
    else {
      // f32 at 64 lanes, the width of the accumulator, as one op.
      if (laneSize == 64 && scalarType.isF32())
        return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs, resultType,
                                                srcOp);
      if (laneSize != 16 && laneSize != 32)
        return failure();

      // v32f32: split into two v16f32 ops
      if (laneSize == 32 && resultElWidth == 32) {
        VectorType halfType = createVectorType(16, scalarType);

        splitWideBinaryVectorOp<SrcOpTy>(
            srcOp, lhs, rhs, halfType, resultType, rewriter,
            [](Value lhsHalf, Value rhsHalf, Location loc,
               ConversionPatternRewriter &rewriter) -> Value {
              VectorType halfVecType = cast<VectorType>(lhsHalf.getType());
              // For f32, use cast to acc, add_elem/sub_elem, cast back
              auto lCastOp = aievec::CastOp::create(rewriter, loc, halfVecType,
                                                    lhsHalf, /*isResAcc*/ true);
              auto rCastOp = aievec::CastOp::create(rewriter, loc, halfVecType,
                                                    rhsHalf, /*isResAcc*/ true);
              auto elemOp = DstOpTy::create(
                  rewriter, loc, lCastOp->getResult(0).getType(),
                  lCastOp->getResult(0), rCastOp->getResult(0));
              auto resCastOp = aievec::CastOp::create(
                  rewriter, loc, halfVecType, elemOp.getResult(),
                  /*isResAcc*/ false);
              return resCastOp.getResult();
            });
        return success();
      }

      // v32bf16: split into two v16bf16 ops
      if (laneSize == 32 && resultElWidth == 16) {
        VectorType halfType = createVectorType(16, scalarType);

        splitWideBinaryVectorOp<SrcOpTy>(
            srcOp, lhs, rhs, halfType, resultType, rewriter,
            [](Value lhsHalf, Value rhsHalf, Location loc,
               ConversionPatternRewriter &rewriter) -> Value {
              VectorType halfVecType = cast<VectorType>(lhsHalf.getType());
              Type accType = getVectorOpDestType(halfVecType, /*AIE2 =*/true);
              auto lUpsOp =
                  aievec::UPSOp::create(rewriter, loc, accType, lhsHalf);
              auto rUpsOp =
                  aievec::UPSOp::create(rewriter, loc, accType, rhsHalf);
              auto elemOp =
                  DstOpTy::create(rewriter, loc, lUpsOp->getResult(0).getType(),
                                  lUpsOp->getResult(0), rUpsOp->getResult(0));
              auto shiftParamOp = arith::ConstantOp::create(
                  rewriter, loc, rewriter.getI32IntegerAttr(0));
              auto srsOp = aievec::SRSOp::create(rewriter, loc, halfVecType,
                                                 elemOp.getResult(),
                                                 shiftParamOp.getResult());
              return srsOp.getResult();
            });
        return success();
      }

      // Now we know laneSize == 16 for remaining float cases
      // v16float or v16bf16 with extension op case
      if (resultElWidth == 32) {
        if (!lhsDefOp && !rhsDefOp) {
          return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                  resultType, srcOp);
        }

        auto lhsExt = getSourceOfWideningOp(lhs).value_or(nullptr);
        auto rhsExt = getSourceOfWideningOp(rhs).value_or(nullptr);
        // v16float
        if (!lhsExt && !rhsExt) {
          return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                  resultType, srcOp);
        }

        // v16bf16 with two extension ops
        if (lhsExt && rhsExt) {
          auto lval = lhsExt;
          auto rval = rhsExt;
          VectorType vType = cast<VectorType>(lval.getType());

          Type accType = getVectorOpDestType(vType, /*AIE2 =*/true);
          auto lUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
          auto rUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);
          auto elemOp = DstOpTy::create(
              rewriter, srcOp.getLoc(), lUpsOp->getResult(0).getType(),
              lUpsOp->getResult(0), rUpsOp->getResult(0));
          rewriter.replaceOpWithNewOp<aievec::CastOp>(srcOp, srcOp.getType(),
                                                      elemOp.getResult());
          return success();
        }

        // v16bf16 with one extension op
        if (!lhsExt || !rhsExt) {
          auto lval = lhsExt ? lhsExt : lhs;
          auto rval = rhsExt ? rhsExt : rhs;
          auto extVal = lhsExt ? lval : rval;
          VectorType vType = cast<VectorType>(extVal.getType());
          Type accType = getVectorOpDestType(vType, /*AIE2 =*/true);

          aievec::UPSOp upsOp;
          aievec::CastOp castOp;
          if (lhsExt) {
            upsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
            castOp = aievec::CastOp::create(rewriter, srcOp.getLoc(),
                                            resultType, rval,
                                            /*isResAcc*/ true);
          } else {
            upsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);
            castOp = aievec::CastOp::create(rewriter, srcOp.getLoc(),
                                            resultType, lval,
                                            /*isResAcc*/ true);
          }

          auto elemOp = DstOpTy::create(
              rewriter, srcOp.getLoc(), upsOp->getResult(0).getType(),
              upsOp->getResult(0), castOp->getResult(0));

          rewriter.replaceOpWithNewOp<aievec::CastOp>(
              srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);

          return success();
        }
      }

      // v16bfloat16
      Type accType = getVectorOpDestType(resultType, /*AIE2 =*/true);
      auto lUpsOp =
          aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lhs);
      auto rUpsOp =
          aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rhs);
      auto elemOp = DstOpTy::create(rewriter, srcOp.getLoc(),
                                    lUpsOp->getResult(0).getType(),
                                    lUpsOp->getResult(0), rUpsOp->getResult(0));
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          srcOp, srcOp.getType(), elemOp.getResult(), shiftParamOp.getResult());

      return success();
    }

    return failure();
  }
};

using LowerVectorAddIOpToAIEVecAddElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::AddIOp,
                                                    aievec::AddElemOp>;
using LowerVectorSubIOpToAIEVecSubElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::SubIOp,
                                                    aievec::SubElemOp>;
using LowerVectorAddFOpToAIEVecAddElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::AddFOp,
                                                    aievec::AddElemOp>;
using LowerVectorSubFOpToAIEVecSubElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::SubFOp,
                                                    aievec::SubElemOp>;

template <typename SrcOpTy, typename DstOpTy>
struct LowerVectorMinMaxOpToAIEVecMinMaxOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    // A set recording the element width we are supporting for AIE2.
    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    unsigned totalBits = laneSize * resultElWidth;
    if (!elWidthSet.count(resultElWidth) ||
        (totalBits != 512 && (totalBits != 256 || resultElWidth != 16)))
      return failure();

    if (totalBits == 256 && resultElWidth == 16) {
      // Pad v16bf16 to v32bf16, apply max/min, extract lower half
      Location loc = srcOp.getLoc();
      VectorType wideType = createVectorType(32, scalarType);
      Value lhsPad =
          padV16ToV32WithZeros(rewriter, loc, adaptor.getLhs(), scalarType);
      Value rhsPad =
          padV16ToV32WithZeros(rewriter, loc, adaptor.getRhs(), scalarType);
      auto wideOp = DstOpTy::create(rewriter, loc, wideType, lhsPad, rhsPad);
      rewriter.replaceOpWithNewOp<aievec::ExtOp>(srcOp, resultType,
                                                 wideOp.getResult(), 0);
      return success();
    }

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(),
                                         adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

using LowerVectorMinSIOpToAIEVecMinOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MinSIOp, aievec::MinOp>;
using LowerVectorMaxSIOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaxSIOp, aievec::MaxOp>;

// aievec.min/aievec.max are signed-only, so unsigned integer min/max have no
// direct AIE intrinsic. Re-expand them into cmp+select, which the aievec
// cmp/sel lowerings turn into aievec.cmp/aievec.sel.
template <typename SrcOpTy, arith::CmpIPredicate Pred>
struct LowerVectorUMinMaxOpToCmpSelOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<VectorType>(srcOp.getType()))
      return failure();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto cmpOp =
        arith::CmpIOp::create(rewriter, srcOp.getLoc(), Pred, lhs, rhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(srcOp, cmpOp, lhs, rhs);
    return success();
  }
};

using LowerVectorMinUIOpToCmpSelOp =
    LowerVectorUMinMaxOpToCmpSelOp<arith::MinUIOp, arith::CmpIPredicate::ult>;
using LowerVectorMaxUIOpToCmpSelOp =
    LowerVectorUMinMaxOpToCmpSelOp<arith::MaxUIOp, arith::CmpIPredicate::ugt>;

// Promote scalar arith.maxsi/arith.minsi to vector aievec.max/aievec.min
// to avoid the AIE2 G_SELECT legalizer crash on scalar i32 select.
template <typename SrcOpTy, typename DstOpTy>
struct LowerScalarMinMaxToAIEVecMinMaxOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match scalar integer types (reject vectors)
    Type resultType = srcOp.getType();
    if (isa<VectorType>(resultType))
      return failure();

    auto intType = dyn_cast<IntegerType>(resultType);
    if (!intType)
      return failure();

    unsigned elWidth = intType.getWidth();
    if (elWidth != 8 && elWidth != 16 && elWidth != 32)
      return failure();

    unsigned numLanes = 512 / elWidth;
    VectorType vecType = createVectorType(numLanes, intType);
    Location loc = srcOp.getLoc();

    // Broadcast both scalars to 512-bit vectors
    auto lhsBcast = aievec::BroadcastScalarOp::create(rewriter, loc, vecType,
                                                      adaptor.getLhs());
    auto rhsBcast = aievec::BroadcastScalarOp::create(rewriter, loc, vecType,
                                                      adaptor.getRhs());

    // Apply vector min/max
    auto vecOp = DstOpTy::create(rewriter, loc, vecType, lhsBcast.getResult(),
                                 rhsBcast.getResult());

    // Extract element 0 back to scalar
    auto zeroIdx =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(
        srcOp, intType, vecOp.getResult(), zeroIdx.getResult());
    return success();
  }
};

using LowerScalarMinSIOpToAIEVecMinOp =
    LowerScalarMinMaxToAIEVecMinMaxOp<arith::MinSIOp, aievec::MinOp>;
using LowerScalarMaxSIOpToAIEVecMaxOp =
    LowerScalarMinMaxToAIEVecMinMaxOp<arith::MaxSIOp, aievec::MaxOp>;

using LowerVectorMinimumFOpToAIEVecMinOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MinimumFOp, aievec::MinOp>;
using LowerVectorMaximumFOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaximumFOp, aievec::MaxOp>;
using LowerVectorMaxNumFFOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaxNumFOp, aievec::MaxOp>;

template <typename SrcOpTy, typename CmpTy>
struct LowerVectorCmpOpToAIEVecCmpOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType lhsType = dyn_cast<VectorType>(srcOp.getLhs().getType());
    if (!lhsType)
      return failure();

    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = lhsType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);

    unsigned totalBits = laneSize * elWidth;
    if (!elWidthSet.count(elWidth) ||
        (totalBits != 512 && (totalBits != 256 || elWidth != 16)))
      return failure();

    Location loc = srcOp.getLoc();
    Value lhs = srcOp.getLhs();
    Value rhs = srcOp.getRhs();
    unsigned effectiveLaneSize = laneSize;

    if (totalBits == 256 && elWidth == 16) {
      lhs = padV16ToV32WithZeros(rewriter, loc, lhs, scalarType);
      rhs = padV16ToV32WithZeros(rewriter, loc, rhs, scalarType);
      effectiveLaneSize = 32;
    }

    // Unsigned int and unsigned long long are acceptable type.
    Type type = mlir::IntegerType::get(srcOp.getContext(),
                                       effectiveLaneSize <= 32 ? 32 : 64,
                                       mlir::IntegerType::Unsigned);

    CmpTy pred = srcOp.getPredicate();

    arith::CmpIPredicate ipred = convertToIntegerPredicate(pred);

    aievec::CmpOp aieCmpOp =
        createCmpOpAIE2(rewriter, ipred, loc, type, lhs, rhs);

    if (!aieCmpOp)
      return failure();

    VectorType resultType = cast<VectorType>(srcOp.getResult().getType());
    // Convert vector i1 type to unsigned interger type by built-in unrealized
    // conversion cast op.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        srcOp, resultType, aieCmpOp.getResult());

    return success();
  }
};

using LowerVectorCmpIOpToAIEVecCmpOp =
    LowerVectorCmpOpToAIEVecCmpOp<arith::CmpIOp, CmpIPredicate>;
using LowerVectorCmpFOpToAIEVecCmpOp =
    LowerVectorCmpOpToAIEVecCmpOp<arith::CmpFOp, CmpFPredicate>;

struct LowerVectorSelectOpToAIEVecSelOp : OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    unsigned totalBits = laneSize * resultElWidth;
    if (!elWidthSet.count(resultElWidth) ||
        (totalBits != 512 && (totalBits != 256 || resultElWidth != 16)))
      return failure();

    if (totalBits == 256 && resultElWidth == 16) {
      // Pad trueValue and falseValue to v32, do sel, extract lower half
      Location loc = srcOp.getLoc();
      VectorType wideType = createVectorType(32, scalarType);

      // aievec.sel: bit=0 selects lhs, bit=1 selects rhs.
      // aievec.cmp: bit=1 where predicate is true.
      // arith.select: condition=1 returns true_value.
      // So false_value goes to lhs (bit=0) and true_value goes to rhs (bit=1).
      Value falsePad = padV16ToV32WithZeros(rewriter, loc,
                                            srcOp.getFalseValue(), scalarType);
      Value truePad =
          padV16ToV32WithZeros(rewriter, loc, srcOp.getTrueValue(), scalarType);

      // The condition bitmask from cmp was produced at 32 lanes (since cmp
      // was also padded). Use 32-lane integer type for the cast.
      Type type = mlir::IntegerType::get(srcOp.getContext(), 32,
                                         mlir::IntegerType::Unsigned);
      auto convertOp = UnrealizedConversionCastOp::create(
          rewriter, loc, type, adaptor.getCondition());

      auto wideSelOp = aievec::SelOp::create(rewriter, loc, wideType, falsePad,
                                             truePad, convertOp.getResult(0));

      rewriter.replaceOpWithNewOp<aievec::ExtOp>(srcOp, resultType,
                                                 wideSelOp.getResult(), 0);
      return success();
    }

    Type type =
        mlir::IntegerType::get(srcOp.getContext(), laneSize <= 32 ? 32 : 64,
                               mlir::IntegerType::Unsigned);

    auto convertOp = UnrealizedConversionCastOp::create(
        rewriter, srcOp.getLoc(), type, adaptor.getCondition());

    // aievec.sel: bit=0 selects lhs, bit=1 selects rhs.
    // aievec.cmp: bit=1 where predicate is true.
    // arith.select: condition=1 returns true_value.
    // So false_value goes to lhs (bit=0) and true_value goes to rhs (bit=1).
    rewriter.replaceOpWithNewOp<aievec::SelOp>(
        srcOp, srcOp.getResult().getType(), srcOp.getFalseValue(),
        srcOp.getTrueValue(), convertOp.getResult(0));

    return success();
  }
};

struct LowerVectorReductionMinOp : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::MINSI &&
                                     kind != vector::CombiningKind::MINUI &&
                                     kind != vector::CombiningKind::MINIMUMF &&
                                     kind != vector::CombiningKind::MINNUMF)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);
    unsigned vectorSize = laneSize * elWidth;

    // Support 512-bit vectors directly, and 256-bit bf16 vectors by padding
    if (vectorSize != 512 && (vectorSize != 256 || !scalarType.isBF16()))
      return failure();

    Location loc = srcOp.getLoc();
    Value inputVec = srcOp.getVector();

    // For 256-bit bf16 (v16bf16), pad to 512-bit (v32bf16) with +inf
    if (vectorSize == 256) {
      std::tie(inputVec, laneSize) = padV16ToV32WithInfinity(
          rewriter, loc, srcOp.getVector(), scalarType, /*negativeInf=*/false);
    }

    int shiftIndex = laneSize / 2;
    auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::MinOp>(
        rewriter, srcOp, shiftIndex, inputVec);

    if (srcOp.getAcc()) {
      Value reduceResult = reduceResultOp.getResult();
      Value acc = srcOp.getAcc();

      // If accumulator is bf16, use the high-level helper for bf16->f32->bf16
      if (acc.getType().isBF16()) {
        // Define the min operation to be performed in f32
        auto minOpBuilder = [&](Value lhs, Value rhs) -> Value {
          auto cmpOp = arith::CmpFOp::create(
              rewriter, srcOp.getLoc(), arith::CmpFPredicate::OLT, lhs, rhs);
          return arith::SelectOp::create(rewriter, srcOp.getLoc(), cmpOp, lhs,
                                         rhs);
        };

        // Use helper to handle bf16->f32 conversion, perform min, and convert
        // back
        performBF16BinaryOpInF32(reduceResult, acc, srcOp, srcOp.getLoc(),
                                 rewriter, minOpBuilder);
      } else {
        // Non-bf16 path: perform min using cmpf and select
        auto cmpOp =
            arith::CmpFOp::create(rewriter, srcOp.getLoc(),
                                  arith::CmpFPredicate::OLT, reduceResult, acc);
        rewriter.replaceOpWithNewOp<arith::SelectOp>(srcOp, cmpOp, reduceResult,
                                                     acc);
      }
    } else {
      rewriter.replaceOp(srcOp, reduceResultOp);
    }
    return success();
  }
};

struct LowerVectorReductionMaxOp : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::MAXSI &&
                                     kind != vector::CombiningKind::MAXUI &&
                                     kind != vector::CombiningKind::MAXIMUMF &&
                                     kind != vector::CombiningKind::MAXNUMF)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);
    unsigned vectorSize = laneSize * elWidth;

    // Support 512-bit vectors directly, and 256-bit bf16 vectors by padding
    // Only bf16 is supported for the 256-bit padding path (not f16)
    if (vectorSize != 512 && (vectorSize != 256 || !scalarType.isBF16()))
      return failure();

    Location loc = srcOp.getLoc();
    Value inputVec = srcOp.getVector();

    // For 256-bit bf16 (v16bf16), pad to 512-bit (v32bf16) with -inf
    if (vectorSize == 256) {
      std::tie(inputVec, laneSize) = padV16ToV32WithInfinity(
          rewriter, loc, srcOp.getVector(), scalarType, /*negativeInf=*/true);
    }

    int shiftIndex = laneSize / 2;
    auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::MaxOp>(
        rewriter, srcOp, shiftIndex, inputVec);

    if (srcOp.getAcc()) {
      Value reduceResult = reduceResultOp.getResult();
      Value acc = srcOp.getAcc();

      // If accumulator is bf16, use the high-level helper for bf16->f32->bf16
      if (acc.getType().isBF16()) {
        // Define the max operation to be performed in f32
        auto maxOpBuilder = [&](Value lhs, Value rhs) -> Value {
          auto cmpOp = arith::CmpFOp::create(
              rewriter, srcOp.getLoc(), arith::CmpFPredicate::OGT, lhs, rhs);
          return arith::SelectOp::create(rewriter, srcOp.getLoc(), cmpOp, lhs,
                                         rhs);
        };

        // Use helper to handle bf16->f32 conversion, perform max, and convert
        // back
        performBF16BinaryOpInF32(reduceResult, acc, srcOp, srcOp.getLoc(),
                                 rewriter, maxOpBuilder);
      } else {
        // Non-bf16 path: perform max directly
        auto cmpOp =
            arith::CmpFOp::create(rewriter, srcOp.getLoc(),
                                  arith::CmpFPredicate::OGT, reduceResult, acc);
        rewriter.replaceOpWithNewOp<arith::SelectOp>(srcOp, cmpOp, reduceResult,
                                                     acc);
      }
    } else {
      rewriter.replaceOp(srcOp, reduceResultOp);
    }
    return success();
  }
};

struct LowerVectorReductionAddIntOp : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);
    llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
    laneSizeElWidthPairSet.insert({64, 8});
    laneSizeElWidthPairSet.insert({32, 16});
    laneSizeElWidthPairSet.insert({32, 32});
    laneSizeElWidthPairSet.insert({16, 32});

    if (!isa<IntegerType>(scalarType) ||
        !laneSizeElWidthPairSet.count(std::make_pair(laneSize, elWidth)))
      return failure();

    int shiftIndex = laneSize / 2;
    if (laneSize == 32 && elWidth == 32) {
      Location loc = srcOp.getLoc();
      VectorType vecType = createVectorType(laneSize / 2, scalarType);

      auto lExtOp =
          aievec::ExtOp::create(rewriter, loc, vecType, srcOp.getVector(), 0);
      auto rExtOp =
          aievec::ExtOp::create(rewriter, loc, vecType, srcOp.getVector(), 1);
      auto addElemOp =
          aievec::AddElemOp::create(rewriter, loc, lExtOp.getResult().getType(),
                                    lExtOp.getResult(), rExtOp.getResult());
      shiftIndex /= 2;
      auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::AddElemOp>(
          rewriter, srcOp, shiftIndex, addElemOp.getResult());
      if (srcOp.getAcc())
        rewriter.replaceOpWithNewOp<arith::AddIOp>(
            srcOp, reduceResultOp.getResult(), srcOp.getAcc());
      else
        rewriter.replaceOp(srcOp, reduceResultOp);
    } else {
      auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::AddElemOp>(
          rewriter, srcOp, shiftIndex, srcOp.getVector());
      if (srcOp.getAcc())
        rewriter.replaceOpWithNewOp<arith::AddIOp>(
            srcOp, reduceResultOp.getResult(), srcOp.getAcc());
      else
        rewriter.replaceOp(srcOp, reduceResultOp);
    }

    return success();
  }
};

struct LowerVectorReductionAddFloatOp
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);

    if (!isa<FloatType>(scalarType) || laneSize != 16 || elWidth != 32)
      return failure();

    int shiftIndex = laneSize / 2;
    assert(shiftIndex > 0 && (shiftIndex & (shiftIndex - 1)) == 0 &&
           "shiftIndex must be power of 2");

    Location loc = srcOp.getLoc();
    Value curValue = srcOp.getVector();
    aievec::CastOp curOp = nullptr;

    for (int id = shiftIndex; id > 0; id /= 2) {
      auto constOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(id * elWidth / 8));

      auto shiftBytesOp = aievec::ShiftOp::create(
          rewriter, loc, vType, curValue, curValue, constOp.getResult());

      auto lCastOp = aievec::CastOp::create(rewriter, loc, vType, curValue,
                                            /*isResAcc*/ true);
      auto rCastOp =
          aievec::CastOp::create(rewriter, loc, vType, shiftBytesOp.getResult(),
                                 /*isResAcc*/ true);
      auto elemOp = aievec::AddElemOp::create(
          rewriter, loc, lCastOp.getResult().getType(), lCastOp.getResult(),
          rCastOp.getResult());
      curOp = aievec::CastOp::create(rewriter, loc, vType, elemOp.getResult(),
                                     /*isResAcc*/ false);
      curValue = curOp.getResult();
    }

    auto zeroConstOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto reduceResultOp = aievec::ExtElemOp::create(
        rewriter, srcOp.getLoc(), scalarType, curOp, zeroConstOp.getResult());

    if (srcOp.getAcc())
      rewriter.replaceOpWithNewOp<arith::AddFOp>(
          srcOp, reduceResultOp.getResult(), srcOp.getAcc());
    else
      rewriter.replaceOp(srcOp, reduceResultOp);
    return success();
  }
};

// AIE2-specific bf16 ADD reduction - requires concat to v32bf16 before ext_elem
// due to aie2 ext_elem limitation
struct LowerVectorReductionAddBfloat16OpAIE2
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD) {
      return failure();
    }

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);

    // Support both lane=16 and lane=32 for bf16
    if (!isa<FloatType>(scalarType) || (laneSize != 16 && laneSize != 32) ||
        elWidth != 16) {
      return failure();
    }

    Location loc = srcOp.getLoc();
    Value curValue = srcOp.getVector();
    VectorType currentVType = vType; // Track current working vector type

    // For lane=32, split into two v16bf16 halves and add them
    if (laneSize == 32) {
      VectorType halfType = createVectorType(laneSize / 2, scalarType);
      auto lowerHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 0);
      auto upperHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 1);

      Type accType = getVectorOpDestType(halfType, /*AIE2 =*/true);
      auto lUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, lowerHalf.getResult());
      auto rUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, upperHalf.getResult());
      auto addElemOp = aievec::AddElemOp::create(
          rewriter, loc, accType, lUpsOp.getResult(), rUpsOp.getResult());
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(0));
      auto srsOp =
          aievec::SRSOp::create(rewriter, loc, halfType, addElemOp.getResult(),
                                shiftParamOp.getResult());
      curValue = srsOp.getResult();
      currentVType = halfType; // Update to v16bf16 after split
    }

    int shiftIndex = 8; // Always 8 since we work with v16bf16
    Type accType = getVectorOpDestType(cast<VectorType>(curValue.getType()),
                                       /*AIE2 =*/true);
    unsigned accWidth =
        cast<VectorType>(accType).getElementType().getIntOrFloatBitWidth();

    auto upsOp = aievec::UPSOp::create(rewriter, loc, accType, curValue);
    curValue = upsOp.getResult();

    aievec::AddElemOp curOp = nullptr;

    for (int id = shiftIndex; id > 0; id /= 2) {
      auto constOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(id * accWidth / 8));
      auto shiftBytesOp = aievec::ShiftOp::create(
          rewriter, loc, accType, curValue, curValue, constOp, true);
      curOp = aievec::AddElemOp::create(rewriter, loc, accType, curValue,
                                        shiftBytesOp.getResult());
      curValue = curOp.getResult();
    }

    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
    // Use currentVType instead of vType to ensure lane count matches
    auto srsOp =
        aievec::SRSOp::create(rewriter, loc, currentVType, curOp.getResult(),
                              shiftParamOp.getResult());

    // AIE2 ext_elem requires v32bf16, so concat v16bf16 to v32bf16
    VectorType vecType = createVectorType(32, scalarType);
    SmallVector<Value> concatSources = {srsOp.getResult(), srsOp.getResult()};
    auto concatOp =
        aievec::ConcatOp::create(rewriter, loc, vecType, concatSources);

    auto zeroConstOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto reduceResultOp =
        aievec::ExtElemOp::create(rewriter, srcOp.getLoc(), scalarType,
                                  concatOp, zeroConstOp.getResult());

    if (srcOp.getAcc())
      rewriter.replaceOpWithNewOp<arith::AddFOp>(
          srcOp, reduceResultOp.getResult(), srcOp.getAcc());
    else
      rewriter.replaceOp(srcOp, reduceResultOp);
    return success();
  }
};

// AIE2P-specific bf16 ADD reduction - can extract directly from v16bf16
struct LowerVectorReductionAddBfloat16OpAIE2P
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);

    // Support both lane=16 and lane=32 for bf16
    if (!isa<FloatType>(scalarType) || (laneSize != 16 && laneSize != 32) ||
        elWidth != 16)
      return failure();

    Location loc = srcOp.getLoc();
    int shiftIndex = laneSize / 2;
    Value inputToReduce = srcOp.getVector();

    // For lane=32, split into two v16bf16 halves, add them, then reduce
    if (laneSize == 32) {
      VectorType halfType = createVectorType(laneSize / 2, scalarType);

      // Extract lower and upper halves
      auto lowerHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 0);
      auto upperHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 1);

      // Add the two halves together
      Type accType = getVectorOpDestType(halfType, /*AIE2 =*/true);
      auto lUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, lowerHalf.getResult());
      auto rUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, upperHalf.getResult());
      auto addElemOp = aievec::AddElemOp::create(
          rewriter, loc, accType, lUpsOp.getResult(), rUpsOp.getResult());
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(0));
      auto srsOp =
          aievec::SRSOp::create(rewriter, loc, halfType, addElemOp.getResult(),
                                shiftParamOp.getResult());

      inputToReduce = srsOp.getResult();
      shiftIndex = 8;
    }

    // Perform reduction using utility
    Type accType = getVectorOpDestType(
        cast<VectorType>(inputToReduce.getType()), /*AIE2 =*/true);
    unsigned accWidth =
        cast<VectorType>(accType).getElementType().getIntOrFloatBitWidth();

    auto upsOp = aievec::UPSOp::create(rewriter, loc, accType, inputToReduce);
    Value curValue = upsOp.getResult();

    aievec::AddElemOp curOp = nullptr;
    for (int id = shiftIndex; id > 0; id /= 2) {
      auto constOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(id * accWidth / 8));
      auto shiftBytesOp = aievec::ShiftOp::create(
          rewriter, loc, accType, curValue, curValue, constOp, true);
      curOp = aievec::AddElemOp::create(rewriter, loc, accType, curValue,
                                        shiftBytesOp.getResult());
      curValue = curOp.getResult();
    }

    // Extract element 0 from the f32 accumulator
    // The loop has already fully reduced the vector to a single value in
    // element 0
    auto zeroConstOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto extractedF32 = aievec::ExtElemOp::create(
        rewriter, srcOp.getLoc(), rewriter.getF32Type(), curOp.getResult(),
        zeroConstOp.getResult());

    // Convert extracted f32 to bf16
    auto reduceResultBF16 = arith::TruncFOp::create(
        rewriter, srcOp.getLoc(), scalarType, extractedF32.getResult());

    if (srcOp.getAcc())
      rewriter.replaceOpWithNewOp<arith::AddFOp>(srcOp, reduceResultBF16,
                                                 srcOp.getAcc());
    else
      rewriter.replaceOp(srcOp, reduceResultBF16);
    return success();
  }
};

// Convert a `vector.extract_strided_slice` op on 1D vectors into an
// `aievec.select` + `aievec.ext` op.
struct LowerVectorExtractStridedSliceOpAIEv1Pattern
    : OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vType = extractOp.getSourceVectorType();
    if (vType.getRank() != 1)
      return failure();

    int64_t stride = cast<IntegerAttr>(adaptor.getStrides()[0]).getInt();
    if (stride != 1)
      return failure();

    // AIE doesn't support select operations on i8
    if (getElementSizeInBits(vType) == 8)
      return extractOp.emitError()
             << "AIEv1 doesn't support select ops on int8 types";

    // We only accept the case where we are extracting a slice half the size of
    // the input vector.
    int64_t size = cast<IntegerAttr>(adaptor.getSizes()[0]).getInt();
    if (vType.getNumElements() != 2 * size)
      return failure();

    int64_t offset = cast<IntegerAttr>(adaptor.getOffsets()[0]).getInt();
    aievec::aie1::SelectOp::Properties selectProps;
    if (failed(aievec::aie1::SelectOp::setPropertiesFromAttr(
            selectProps,
            rewriter.getDictionaryAttr(
                buildAttributeListForRotationSelectOp(rewriter, vType, offset)),
            [&]() { return extractOp.emitError(); })))
      return failure();
    auto selectOp = aievec::aie1::SelectOp::create(
        rewriter, extractOp.getLoc(), TypeRange({vType}),
        ValueRange({adaptor.getSource()}), selectProps);
    rewriter.replaceOpWithNewOp<aievec::aie1::ExtOp>(
        extractOp, extractOp.getType(), selectOp.getResult(),
        rewriter.getI8IntegerAttr(0));
    return success();
  }
};

// Convert a `vector.extract_strided_slice` op on 1D vectors into an
// `aievec.shift` op (an `aievec.ext` alone for the bottom or top half).
struct LowerVectorExtractStridedSliceOpAIE2Pattern
    : OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vType = cast<VectorType>(adaptor.getSource().getType());
    if (vType.getRank() != 1)
      return failure();

    int64_t stride = cast<IntegerAttr>(adaptor.getStrides()[0]).getInt();
    if (stride != 1)
      return failure();

    // We only accept the case where we are extracting a slice half the size of
    // the input vector.
    int64_t size = cast<IntegerAttr>(adaptor.getSizes()[0]).getInt();
    if (vType.getNumElements() != 2 * size)
      return failure();

    auto shortVecType = cast<VectorType>(extractOp.getResult().getType());
    int64_t offset = cast<IntegerAttr>(adaptor.getOffsets()[0]).getInt();
    // The bottom or the top half itself: no shift.
    if (offset == 0 || offset == size) {
      rewriter.replaceOpWithNewOp<aievec::ExtOp>(
          extractOp, shortVecType, adaptor.getSource(),
          rewriter.getI8IntegerAttr(offset == 0 ? 0 : 1));
      return success();
    }
    auto bottomHalf =
        aievec::ExtOp::create(rewriter, extractOp.getLoc(), shortVecType,
                              adaptor.getSource(), rewriter.getI8IntegerAttr(0))
            .getResult();
    auto topHalf =
        aievec::ExtOp::create(rewriter, extractOp.getLoc(), shortVecType,
                              adaptor.getSource(), rewriter.getI8IntegerAttr(1))
            .getResult();
    int32_t shiftBytes = offset * getElementSizeInBits(vType) / 8;
    auto shiftBytesConstOp = arith::ConstantOp::create(
        rewriter, extractOp.getLoc(), rewriter.getIntegerType(32),
        rewriter.getI32IntegerAttr(shiftBytes));
    rewriter.replaceOpWithNewOp<aievec::ShiftOp>(
        extractOp, shortVecType, bottomHalf, topHalf, shiftBytesConstOp);

    return success();
  }
};

// Replaces a short UPD op with a wide one followed by an ext op of the bottom
// half.
struct ExpandUPDToUPDAndExtPattern : OpConversionPattern<aievec::UPDOp> {
  using OpConversionPattern::OpConversionPattern;

  ExpandUPDToUPDAndExtPattern(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(aievec::UPDOp updOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify that we haven't already expanded this one
    if (updOp->hasOneUse() && isa<aievec::ExtOp>(*updOp->getUsers().begin()))
      return failure();

    auto vecType = cast<VectorType>(updOp.getType());
    SmallVector<int64_t, 4> vecShape(vecType.getShape().begin(),
                                     vecType.getShape().end());
    vecShape[vecType.getRank() - 1] *= 2;
    auto longVecType = VectorType::get(vecShape, vecType.getElementType());
    auto newUpdOp = aievec::UPDOp::create(
        rewriter, updOp.getLoc(), longVecType, adaptor.getSource(),
        adaptor.getIndices(), adaptor.getOffset(), adaptor.getIndex(),
        adaptor.getVector());
    rewriter.replaceOpWithNewOp<aievec::ExtOp>(
        updOp, vecType, newUpdOp.getResult(), rewriter.getI8IntegerAttr(0));

    return success();
  }
};

// Replaces a wide UPD op followed by an ext op of the bottom half with a short
// UPD op.
struct FuseExtIntoUPDPattern : OpConversionPattern<aievec::ExtOp> {
  using OpConversionPattern::OpConversionPattern;

  FuseExtIntoUPDPattern(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(aievec::ExtOp extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify we are extracting the lower half...
    if (extOp.getIndex() != 0)
      return failure();
    // ...of a UPDOp
    auto updOp = dyn_cast<aievec::UPDOp>(extOp.getSource().getDefiningOp());
    if (!updOp)
      return failure();

    // Verify that this is a direct upd -> ext pattern
    if (!updOp->hasOneUse())
      return failure();

    rewriter.replaceOpWithNewOp<aievec::UPDOp>(
        extOp, extOp.getType(), updOp.getSource(), updOp.getIndices(),
        updOp.getOffset(), updOp.getIndex(), updOp.getVector());

    return success();
  }
};

// Convert math.exp to aievec.exp for AIE2P (will be further lowered to exp2
// intrinsic)
struct ConvertMathExpToAIEVecExpOpPattern : OpConversionPattern<math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpOp expOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (matchExpOpForAIE2P(adaptor)) {
      auto srcType = cast<VectorType>(adaptor.getOperand().getType());
      rewriter.replaceOpWithNewOp<aievec::ExpOp>(expOp, srcType,
                                                 adaptor.getOperand());
      return success();
    }

    // aievec.exp takes bf16, so narrow an f32 operand and widen the result.
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    if (!srcType || srcType.isScalable() || srcType.getRank() != 1 ||
        !srcType.getElementType().isF32() ||
        (srcType.getNumElements() != 16 && srcType.getNumElements() != 32))
      return failure();

    Location loc = expOp.getLoc();
    auto bf16Type = VectorType::get(srcType.getShape(), rewriter.getBF16Type());
    auto shift =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value input = aievec::SRSOp::create(rewriter, loc, bf16Type,
                                        adaptor.getOperand(), shift);
    Value exp = aievec::ExpOp::create(rewriter, loc, bf16Type, input);
    rewriter.replaceOpWithNewOp<aievec::UPSOp>(expOp, srcType, exp,
                                               /*shift=*/0);
    return success();
  }
};

// Convert math.tanh to aievec.tanh, or math.exp2 to aievec.exp2, for AIE2P.
// Both are further lowered to the hardware intrinsic of the same name.
template <typename MathOpTy, typename AIEVecOpTy>
struct ConvertMathToAIEVecAccFloatUnaryOpPattern
    : OpConversionPattern<MathOpTy> {
  using OpConversionPattern<MathOpTy>::OpConversionPattern;
  using OpAdaptor = typename MathOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(MathOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    if (!srcType || !isAIE2PAccFloatUnaryType(srcType))
      return failure();

    rewriter.replaceOpWithNewOp<AIEVecOpTy>(op, srcType, adaptor.getOperand());
    return success();
  }
};

// Convert math.erf to aievec.tanh for AIE2P, using
//
//   erf(x) ~= tanh(a * x + b * x^3),  a = 2/sqrt(pi),  b = 2 * a * 0.044715
//
// (the constants of the tanh form of GELU). The tanh argument is accumulated
// in f32 from bf16 multiplies. `a` is split into bf16 high and low parts so
// the linear term keeps f32 precision; x^2, x^3 and `b` are rounded to bf16.
struct ComputeErfOpAIE2PPattern : OpConversionPattern<math::ErfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ErfOp erfOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    if (!srcType || srcType.getRank() != 1 ||
        !isAIE2PAccFloatUnaryType(srcType))
      return failure();

    constexpr double a = 1.1283791670955126;  // 2/sqrt(pi)
    constexpr double b = 0.10091092485409293; // 2 * a * 0.044715

    Location loc = erfOp.getLoc();
    auto bf16Type = VectorType::get(srcType.getShape(), rewriter.getBF16Type());
    auto f32Type = VectorType::get(srcType.getShape(), rewriter.getF32Type());
    auto shift =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto toBF16 = [&](Value v) -> Value {
      return aievec::SRSOp::create(rewriter, loc, bf16Type, v, shift);
    };
    // A bf16 splat of `value`, and the value it was rounded to.
    auto splatBF16 = [&](double value) -> std::pair<Value, double> {
      auto attr = rewriter.getFloatAttr(rewriter.getBF16Type(), value);
      Value splat = arith::ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(bf16Type, attr));
      return {splat, attr.getValueAsDouble()};
    };

    Value x = adaptor.getOperand();
    if (srcType.getElementType().isF32())
      x = toBF16(x);

    auto [aHi, aHiValue] = splatBF16(a);
    Value aLo = splatBF16(a - aHiValue).first;
    Value bSplat = splatBF16(b).first;

    Value x2 = toBF16(aievec::MulElemOp::create(rewriter, loc, f32Type, x, x));
    Value x3 = toBF16(aievec::MulElemOp::create(rewriter, loc, f32Type, x2, x));
    Value arg = aievec::MulElemOp::create(rewriter, loc, f32Type, x, aHi);
    arg = aievec::FMAElemOp::create(rewriter, loc, f32Type, x, aLo, arg,
                                    /*fmsub=*/false);
    arg = aievec::FMAElemOp::create(rewriter, loc, f32Type, x3, bSplat, arg,
                                    /*fmsub=*/false);
    Value result = aievec::TanhOp::create(rewriter, loc, f32Type, arg);
    if (srcType.getElementType().isBF16())
      result = toBF16(result);

    rewriter.replaceOp(erfOp, result);
    return success();
  }
};

// Convert a scalar math.tanh, math.erf or math.exp2 for AIE2P by broadcasting
// it to a 16-lane vector, applying the vector op and extracting lane 0.
template <typename OpTy>
struct ComputeScalarMathOpByVectorAIE2PPattern : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = adaptor.getOperand().getType();
    if (!type.isBF16() && !type.isF32())
      return failure();

    Location loc = op.getLoc();
    auto vecType = VectorType::get({16}, type);
    Value splat = aievec::BroadcastScalarOp::create(rewriter, loc, vecType,
                                                    adaptor.getOperand());
    Value result = OpTy::create(rewriter, loc, splat);
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, result, 0);
    return success();
  }
};

// Convert a one-lane math.tanh, math.erf or math.exp2 for AIE2P to the scalar
// op, which ComputeScalarMathOpByVectorAIE2PPattern lowers.
template <typename OpTy>
struct ComputeOneLaneMathOpAIE2PPattern : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    if (!srcType || srcType.isScalable() || srcType.getRank() != 1 ||
        srcType.getNumElements() != 1)
      return failure();

    Location loc = op.getLoc();
    Value scalar =
        vector::ExtractOp::create(rewriter, loc, adaptor.getOperand(), 0);
    Value result = OpTy::create(rewriter, loc, scalar);
    Value poison = ub::PoisonOp::create(rewriter, loc, srcType);
    rewriter.replaceOpWithNewOp<vector::InsertOp>(op, result, poison, 0);
    return success();
  }
};

struct ComputeExpOpByLUTLLVMPattern : OpConversionPattern<math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpOp expOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!matchExpOpForAIE2LUT(adaptor))
      return failure();

    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    unsigned laneSize = getVectorLaneSize(srcType);
    Location loc = expOp.getLoc();
    StringRef funcName = "getExpBf16";

    VectorType v16bf16Ty = mlir::VectorType::get({16}, rewriter.getBF16Type());
    VectorType v8i64Ty = mlir::VectorType::get({8}, rewriter.getI64Type());
    func::FuncOp fnOp = getOrInsertFuncDecl(
        rewriter, expOp->getParentWithTrait<OpTrait::SymbolTable>(), funcName,
        TypeRange{v16bf16Ty}, TypeRange{v8i64Ty});

    // Handle v32bf16 by splitting into two v16bf16 operations
    if (laneSize == 32) {
      splitWideUnaryVectorOp<math::ExpOp>(
          expOp, adaptor.getOperand(), v16bf16Ty, srcType, rewriter,
          [&fnOp](Value halfInput, Location loc,
                  ConversionPatternRewriter &rewriter) -> Value {
            VectorType v16bf16Ty =
                mlir::VectorType::get({16}, rewriter.getBF16Type());
            auto callOp = func::CallOp::create(rewriter, loc, fnOp,
                                               SmallVector<Value>{halfInput});
            Type accType = getVectorOpDestType(v16bf16Ty, /*AIE2 =*/true);
            auto resCastOp = vector::BitCastOp::create(rewriter, loc, accType,
                                                       callOp.getResults());
            auto shiftParamOp = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(0));
            auto srsOp = aievec::SRSOp::create(rewriter, loc, v16bf16Ty,
                                               resCastOp.getResult(),
                                               shiftParamOp.getResult());
            return srsOp.getResult();
          });
      return success();
    }

    // Handle v16bf16 directly
    SmallVector<Value> expOperands = {adaptor.getOperand()};

    Type accTypeNative = getVectorOpDestType(srcType, /*AIE2 =*/true);
    auto callOp = func::CallOp::create(rewriter, loc, fnOp, expOperands);
    auto resCastOp = vector::BitCastOp::create(rewriter, loc, accTypeNative,
                                               callOp.getResults());
    auto shiftParamOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        expOp, srcType, resCastOp.getResult(), shiftParamOp.getResult());

    return success();
  }
};
// Lower ExpOp to function call
struct ComputeExpOpByLUTPattern : OpConversionPattern<math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpOp expOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchExpOpForAIE2LUT(adaptor))
      return failure();
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    StringRef includeName = "lut_based_ops.h";
    auto moduleOp = expOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(expOp);

    auto v16bf16OpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, expOp.getLoc(), v16bf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> expOperands = {opaquedOperand};

    Type accTypeNative = getVectorOpDestType(srcType, /*AIE2 =*/true);
    Type v16accf32OpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), "v16accfloat");
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, expOp.getLoc(), TypeRange{v16accf32OpaqueTy}, "getExpBf16",
        nullptr, nullptr, expOperands);
    auto resCastOp = UnrealizedConversionCastOp::create(
        rewriter, expOp.getLoc(), accTypeNative, callOp.getResults());
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, expOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        expOp, srcType, resCastOp.getResult(0), shiftParamOp.getResult());

    return success();
  }
};

// Lower the inverse of a float to aievec.inv (LLVMIR backend for AIE2P)
// Supports both scalar f32 and vector<Nxf32> types.
// Convert the pattern-
//  %cst = arith.constant 1.000000e+00 : f32
//  %0 = arith.divf %cst, %arg1 : f32
// to -
//  %0 = aievec.inv %arg1 : f32
// Also supports:
//  %cst = arith.constant dense<1.0> : vector<16xf32>
//  %0 = arith.divf %cst, %arg1 : vector<16xf32>
// to -
//  %0 = aievec.inv %arg1 : vector<16xf32>
struct ConvertDivFToAIEVecInvOpPattern : OpConversionPattern<arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivFOp divOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getLhs().getType();

    // Check if LHS is defined by an operation
    auto *defOp = divOp.getLhs().getDefiningOp();
    if (!defOp)
      return failure();

    auto constOp = dyn_cast<arith::ConstantOp>(defOp);
    if (!constOp)
      return failure();

    // Handle scalar f32 case
    if (auto fType = dyn_cast<FloatType>(srcType)) {
      if (fType.getWidth() != 32)
        return failure();

      auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
      if (!floatAttr || !floatAttr.getValue().isExactlyValue(1.0))
        return failure();

      rewriter.replaceOpWithNewOp<aievec::InvOp>(divOp, srcType,
                                                 adaptor.getRhs());
      return success();
    }

    // Handle vector f32 case
    if (auto vecType = dyn_cast<VectorType>(srcType)) {
      auto elemType = vecType.getElementType();
      if (!elemType.isF32())
        return failure();

      // Check for supported vector sizes (16 or 32 lanes)
      unsigned laneSize = getVectorLaneSize(vecType);
      if (laneSize != 16 && laneSize != 32)
        return failure();

      // Check if it's a splat of 1.0
      auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
      if (!denseAttr || !denseAttr.isSplat())
        return failure();

      if (!denseAttr.getSplatValue<APFloat>().isExactlyValue(1.0))
        return failure();

      rewriter.replaceOpWithNewOp<aievec::InvOp>(divOp, vecType,
                                                 adaptor.getRhs());
      return success();
    }

    return failure();
  }
};

// Convert math.tanh to a function call to compute tanh(x) by look up tables
struct ComputeTanhOpByLUTPattern : OpConversionPattern<math::TanhOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::TanhOp tanhOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(tanhOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || laneSize != 16)
      return failure();

    StringRef includeName = "lut_based_ops.h";
    auto moduleOp = tanhOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(tanhOp);
    Type v16bf16OpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, tanhOp.getLoc(), v16bf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> tanhOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(rewriter, tanhOp.getLoc(),
                                              v16bf16OpaqueTy, "getTanhBf16",
                                              nullptr, nullptr, tanhOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        tanhOp, TypeRange{tanhOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.sqrt to a function call to compute sqrt(x) for v16bfloat16 and
// v32bfloat16 types
struct ComputeSqrtOpAIE2Pattern : OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::SqrtOp sqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(sqrtOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = sqrtOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(sqrtOp);
    Type vLNbf16OpaqueTy;
    if (laneSize == 16)
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, sqrtOp.getLoc(), vLNbf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> sqrtOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, sqrtOp.getLoc(), TypeRange{vLNbf16OpaqueTy}, "getSqrtBf16",
        nullptr, nullptr, sqrtOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        sqrtOp, TypeRange{sqrtOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

struct ComputeRsqrtOpLLVMAIE2Pattern : OpConversionPattern<math::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp rsqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(srcType);

    // Only support v16bf16 for LLVM backend
    if (!isa<FloatType>(scalarType) || laneSize != 16 || elWidth != 16)
      return failure();

    StringRef funcName = "getRsqrtBf16";

    VectorType v16bf16Ty = mlir::VectorType::get({16}, rewriter.getBF16Type());
    VectorType v8i64Ty = mlir::VectorType::get({8}, rewriter.getI64Type());
    func::FuncOp fnOp = getOrInsertFuncDecl(
        rewriter, rsqrtOp->getParentWithTrait<OpTrait::SymbolTable>(), funcName,
        TypeRange{v16bf16Ty}, TypeRange{v8i64Ty});

    SmallVector<Value> rsqrtOperands = {adaptor.getOperand()};

    Type accTypeNative = getVectorOpDestType(srcType, /*AIE2 =*/true);
    auto callOp =
        func::CallOp::create(rewriter, rsqrtOp.getLoc(), fnOp, rsqrtOperands);
    auto resCastOp = vector::BitCastOp::create(
        rewriter, rsqrtOp.getLoc(), accTypeNative, callOp.getResults());
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, rsqrtOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        rsqrtOp, srcType, resCastOp.getResult(), shiftParamOp.getResult());

    return success();
  }
};

// Convert math.rsqrt to a function call to compute 1.0f / sqrt(x) for
// v16bfloat16 and v32bfloat16 types
struct ComputeRsqrtOpPattern : OpConversionPattern<math::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp rsqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(rsqrtOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = rsqrtOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(rsqrtOp);
    Type vLNbf16OpaqueTy;
    if (laneSize == 16)
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, rsqrtOp.getLoc(), vLNbf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> rsqrtOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, rsqrtOp.getLoc(), TypeRange{vLNbf16OpaqueTy}, "getRsqrtBf16",
        nullptr, nullptr, rsqrtOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        rsqrtOp, TypeRange{rsqrtOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.erf to a function call to compute erf(x) for v16bfloat16 and
// v32bfloat16 types
struct ComputeErfOpPattern : OpConversionPattern<math::ErfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ErfOp erfOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(erfOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = erfOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(erfOp);
    Type vLNbf16OpaqueTy;
    if (laneSize == 16)
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, erfOp.getLoc(), vLNbf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> erfOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, erfOp.getLoc(), TypeRange{vLNbf16OpaqueTy}, "getErfBf16",
        nullptr, nullptr, erfOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        erfOp, TypeRange{erfOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.absf and math.absi to a function call to compute abs(x) for
// v16bfloat16, v32bfloat16, v16float, v16int32, v32int16 and v64int8 types
template <typename SrcOpTy>
struct ComputeAbsOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy absOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(absOp.getOperand().getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();

    unsigned laneSize = getVectorLaneSize(vecTy);
    unsigned elWidth = elemTy.getIntOrFloatBitWidth();

    StringRef includeName = "vec_math.h";
    auto moduleOp = absOp->template getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(absOp);
    std::ostringstream typeName;
    typeName << "v" << laneSize;
    if (isa<FloatType>(elemTy)) {
      if (elWidth == 16)
        typeName << "bfloat16";
      else
        typeName << "float";
    } else
      typeName << "int" << elWidth;
    Type vecOpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), typeName.str());
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, absOp.getLoc(),
                                           vecOpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> absOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(rewriter, absOp.getLoc(),
                                              TypeRange{vecOpaqueTy}, "getAbs",
                                              nullptr, nullptr, absOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        absOp, TypeRange{absOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

using ComputeAbsFOpPattern = ComputeAbsOpPattern<math::AbsFOp>;
using ComputeAbsIOpPattern = ComputeAbsOpPattern<math::AbsIOp>;

template <typename SrcOpTy>
struct LowerExtOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = cast<VectorType>(extOp.getIn().getType());
    VectorType dstType = cast<VectorType>(extOp.getOut().getType());

    Type scalarType = dstType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    auto accType =
        isa<IntegerType>(scalarType) && (elWidth == 32 || elWidth == 64)
            ? dstType
            : getVectorOpDestType(srcType, /*AIE2 =*/true);
    auto upsOp =
        aievec::UPSOp::create(rewriter, extOp.getLoc(), accType, extOp.getIn());

    if (dstType.getElementType().getIntOrFloatBitWidth() == 16) {
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, extOp.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          extOp, dstType, upsOp.getResult(), shiftParamOp.getResult());
    } else
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          extOp, dstType, upsOp.getResult(), /*isResAcc*/ false);

    return success();
  }
};

using LowerExtFOpPattern = LowerExtOpPattern<arith::ExtFOp>;
using LowerExtSIOpPattern = LowerExtOpPattern<arith::ExtSIOp>;

// Vector arith.sitofp i16 -> bf16 on AIE2P via the classic "magic number"
// trick that aie_api uses in detail/aie2p/elementary.hpp:
//
//   1. UPS the int16 input to acc32 (32-bit integer accumulator lanes).
//   2. Add a per-lane "magic_l" constant to the acc32 vector. The constant
//      is the bit pattern of bf16(0x4b01) (= 2^23 + 128) broadcast over the
//      lanes and reinterpreted as i32 -- chosen so that the resulting acc32
//      vector, when bit-reinterpreted as f32 and subtracted from the same
//      magic constant in accfloat space, yields the float representation of
//      the original signed int16 input.
//   3. aievec.cast acc32 -> accfloat is a register bit-alias on AIE2P
//      (the same physical accumulator register).
//   4. accfloat subtract of the magic constant.
//   5. SRS accfloat -> bf16.
//
// Lane width 16 or 32 maps directly to the native AIE2P 256/1024-bit
// accumulator. 64 splits into two 32-wide halves and concats the results.
//
// arith.uitofp takes u ^ 0x8000, which as a signed value is u - 32768,
// through the same sequence, and subtracts a magic constant 32768 smaller,
// 2^23 + 32768, as an f32 vector (bf16 has no such value).
template <typename SrcOpTy>
struct LowerVectorIntToFPI16BF16AIE2pPattern : OpConversionPattern<SrcOpTy> {
  using OpAdaptor = typename SrcOpTy::Adaptor;
  static constexpr bool isSigned = std::is_same_v<SrcOpTy, arith::SIToFPOp>;

  // `accumulatorAdd`: add the magic constant on the accumulator
  // (aievec.add_elem, AIE2P only) rather than as an arith.addi.
  LowerVectorIntToFPI16BF16AIE2pPattern(MLIRContext *context,
                                        bool accumulatorAdd,
                                        PatternBenefit benefit)
      : OpConversionPattern<SrcOpTy>(context, benefit),
        accumulatorAdd(accumulatorAdd) {}

  bool accumulatorAdd;

  Value emitMagicSiToFp(ConversionPatternRewriter &rewriter, Location loc,
                        Value srcI16, unsigned lanes) const {
    Type i16Ty = rewriter.getI16Type();
    Type i32Ty = rewriter.getI32Type();
    Type bf16Ty = rewriter.getBF16Type();
    Type f32Ty = rewriter.getF32Type();
    VectorType vecI16 = VectorType::get({lanes}, i16Ty);
    VectorType vecI32 = VectorType::get({lanes}, i32Ty);
    VectorType vecBF16 = VectorType::get({lanes}, bf16Ty);
    VectorType vecF32 = VectorType::get({lanes}, f32Ty);

    DenseElementsAttr magicI16Attr = DenseElementsAttr::get(
        vecI16, rewriter.getI16IntegerAttr(static_cast<int16_t>(0x4b01)));
    Value magicI16Vec =
        arith::ConstantOp::create(rewriter, loc, vecI16, magicI16Attr);
    Value magicBF16Vec =
        vector::BitCastOp::create(rewriter, loc, vecBF16, magicI16Vec);
    Value magicAccF =
        aievec::UPSOp::create(rewriter, loc, vecF32, magicBF16Vec);
    Value magicAccI = aievec::CastOp::create(rewriter, loc, vecI32, magicAccF,
                                             /*isResAcc=*/true);
    if (!isSigned) {
      Value flip = arith::ConstantOp::create(
          rewriter, loc, vecI16,
          DenseElementsAttr::get(vecI16, rewriter.getI16IntegerAttr(
                                             static_cast<int16_t>(0x8000))));
      srcI16 = arith::XOrIOp::create(rewriter, loc, srcI16, flip);
    }
    Value vAccI = aievec::UPSOp::create(rewriter, loc, vecI32, srcI16);
    Value vSumI =
        accumulatorAdd
            ? aievec::AddElemOp::create(rewriter, loc, vecI32, vAccI, magicAccI)
                  .getResult()
            : arith::AddIOp::create(rewriter, loc, vAccI, magicAccI)
                  .getResult();
    Value vSumAccF = aievec::CastOp::create(rewriter, loc, vecF32, vSumI,
                                            /*isResAcc=*/true);
    Value subtrahend = magicAccF;
    if (!isSigned)
      subtrahend = arith::ConstantOp::create(
          rewriter, loc, vecF32,
          DenseElementsAttr::get(vecF32, rewriter.getF32FloatAttr(8421376.0f)));
    Value vDiffAccF =
        aievec::SubElemOp::create(rewriter, loc, vecF32, vSumAccF, subtrahend);
    Value shiftCst =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    return aievec::SRSOp::create(rewriter, loc, vecBF16, vDiffAccF, shiftCst)
        .getResult();
  }

  LogicalResult
  matchAndRewrite(SrcOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<VectorType>(adaptor.getIn().getType());
    auto dstTy = dyn_cast<VectorType>(op.getType());
    if (!srcTy || !dstTy || srcTy.getRank() != 1 || dstTy.getRank() != 1)
      return failure();
    if (!srcTy.getElementType().isInteger(16))
      return failure();
    if (!dstTy.getElementType().isBF16())
      return failure();
    unsigned lanes = srcTy.getNumElements();
    if (lanes != 16 && lanes != 32 && lanes != 64)
      return failure();

    Location loc = op.getLoc();
    if (lanes == 16 || lanes == 32) {
      Value out = emitMagicSiToFp(rewriter, loc, adaptor.getIn(), lanes);
      rewriter.replaceOp(op, out);
      return success();
    }

    // 64-wide: split into two 32-wide halves via vector.shuffle
    // (vector.extract_strided_slice is illegal in this conversion target).
    SmallVector<int64_t> loMask, hiMask, catMask;
    for (int64_t i = 0; i < 32; ++i)
      loMask.push_back(i);
    for (int64_t i = 32; i < 64; ++i)
      hiMask.push_back(i);
    for (int64_t i = 0; i < 64; ++i)
      catMask.push_back(i);
    Value lo = vector::ShuffleOp::create(rewriter, loc, adaptor.getIn(),
                                         adaptor.getIn(), loMask)
                   .getResult();
    Value hi = vector::ShuffleOp::create(rewriter, loc, adaptor.getIn(),
                                         adaptor.getIn(), hiMask)
                   .getResult();
    Value loBF16 = emitMagicSiToFp(rewriter, loc, lo, 32);
    Value hiBF16 = emitMagicSiToFp(rewriter, loc, hi, 32);
    Value cat =
        vector::ShuffleOp::create(rewriter, loc, loBF16, hiBF16, catMask)
            .getResult();
    rewriter.replaceOp(op, cat);
    return success();
  }
};
using LowerVectorSIToFPI16BF16AIE2pPattern =
    LowerVectorIntToFPI16BF16AIE2pPattern<arith::SIToFPOp>;
using LowerVectorUIToFPI16BF16AIE2pPattern =
    LowerVectorIntToFPI16BF16AIE2pPattern<arith::UIToFPOp>;

// Match `arith.extui (vector.bitcast %x : vector<Mxi8> to vector<2Mxi4>)
// to vector<2Mxi8>` and replace with `aievec.unpack %x` directly on the
// byte-packed source. The vector<...xi4> intermediate never reaches Peano
// llc, which cannot legalize llvm.bitcast on i4 vectors.
//
// This recognises the canonical standard-MLIR encoding of an AWQ int4
// unpack: byte load of packed nibbles -> bitcast to i4 vector -> extui to
// i8 vector. The aievec.unpack op then lowers to the AIE2P intrinsic
// llvm.aie2p.unpack.I512.I8.I4 (or the I1024 variant) directly.
struct LowerExtUIOfBitcastI4ToUnpackPattern
    : OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern<arith::ExtUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtUIOp extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<VectorType>(adaptor.getIn().getType());
    auto dstTy = dyn_cast<VectorType>(extOp.getType());
    if (!srcTy || !dstTy)
      return failure();
    if (srcTy.getRank() != 1 || dstTy.getRank() != 1)
      return failure();
    if (!srcTy.getElementType().isInteger(4) ||
        !dstTy.getElementType().isInteger(8))
      return failure();
    unsigned lanes = srcTy.getNumElements();
    if (lanes != dstTy.getNumElements())
      return failure();
    if (lanes != 64 && lanes != 128)
      return failure();

    auto bitcastOp = adaptor.getIn().getDefiningOp<vector::BitCastOp>();
    if (!bitcastOp)
      return failure();
    auto bitcastSrcTy = dyn_cast<VectorType>(bitcastOp.getSource().getType());
    if (!bitcastSrcTy || bitcastSrcTy.getRank() != 1 ||
        !bitcastSrcTy.getElementType().isInteger(8))
      return failure();
    if (bitcastSrcTy.getNumElements() * 2 != lanes)
      return failure();

    rewriter.replaceOpWithNewOp<aievec::UnpackOp>(extOp, dstTy,
                                                  bitcastOp.getSource());
    return success();
  }
};

// Scalarize a vector arith int<->fp conversion (sitofp / uitofp / fptosi /
// fptoui). The AIE2/AIE2P GISel legalizers only handle the scalar form of
// these conversions (libcalls); there is no vector int<->fp instruction.
// Vector forms reach the legalizer as G_SITOFP/G_FPTOSI on a vector type
// and abort with "unable to legalize". Lower the vector form here into
// per-lane extract -> scalar conversion -> insert so each lane goes through
// the supported scalar path.
template <typename SrcOpTy>
struct ScalarizeVectorIntFpConversionOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getIn().getType());
    auto dstType = dyn_cast<VectorType>(op.getType());
    if (!srcType || !dstType)
      return failure(); // already scalar — let the standard path handle it
    if (srcType.getRank() != 1 || dstType.getRank() != 1)
      return failure(); // conservative: 1-D vectors only
    if (srcType.getNumElements() != dstType.getNumElements())
      return failure();

    Location loc = op.getLoc();
    Type scalarDstTy = dstType.getElementType();
    int64_t lanes = srcType.getNumElements();

    // Seed result with a zero vector of the destination type.
    Value result = arith::ConstantOp::create(rewriter, loc, dstType,
                                             rewriter.getZeroAttr(dstType));

    for (int64_t i = 0; i < lanes; ++i) {
      Value scalar = vector::ExtractOp::create(rewriter, loc, adaptor.getIn(),
                                               ArrayRef<int64_t>{i});
      Value cvt = SrcOpTy::create(rewriter, loc, scalarDstTy, scalar);
      result = vector::InsertOp::create(rewriter, loc, cvt, result,
                                        ArrayRef<int64_t>{i});
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

using ScalarizeVectorSIToFPOpPattern =
    ScalarizeVectorIntFpConversionOpPattern<arith::SIToFPOp>;
using ScalarizeVectorUIToFPOpPattern =
    ScalarizeVectorIntFpConversionOpPattern<arith::UIToFPOp>;
using ScalarizeVectorFPToSIOpPattern =
    ScalarizeVectorIntFpConversionOpPattern<arith::FPToSIOp>;
using ScalarizeVectorFPToUIOpPattern =
    ScalarizeVectorIntFpConversionOpPattern<arith::FPToUIOp>;

template <typename SrcOpTy>
struct LowerTruncOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy truncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = cast<VectorType>(truncOp.getIn().getType());
    VectorType dstType = cast<VectorType>(truncOp.getOut().getType());
    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    auto accType =
        isa<IntegerType>(scalarType) && (elWidth == 32 || elWidth == 64)
            ? srcType
            : getVectorOpDestType(srcType, /*AIE2 =*/true);

    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, truncOp.getLoc(), rewriter.getI32IntegerAttr(0));
    if (elWidth == 16) {
      auto upsOp = aievec::UPSOp::create(rewriter, truncOp.getLoc(), accType,
                                         truncOp.getIn());
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          truncOp, dstType, upsOp.getResult(), shiftParamOp.getResult());
    } else {
      auto castOp = aievec::CastOp::create(rewriter, truncOp.getLoc(), accType,
                                           truncOp.getIn(), true);
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          truncOp, dstType, castOp.getResult(), shiftParamOp.getResult());
    }

    return success();
  }
};

using LowerTruncFOpPattern = LowerTruncOpPattern<arith::TruncFOp>;
using LowerTruncIOpPattern = LowerTruncOpPattern<arith::TruncIOp>;

// If `op` is the last operation in the sequence:
//     %0 = unrealized_conversion_cast <%IN> : <native type>, !emitc.opaque_type
//     %1 = emitc.call_opaque <funcName>, %0...
//     %2 = unrealized_conversion_cast %1 : !emitc.opaque_type, <native type>
// return the value <%IN>.
static std::optional<Value>
getUnOpaquedOperandOfEmitCOpaqueCallOp(Operation *op, StringRef funcName) {
  auto uccOp = dyn_cast<UnrealizedConversionCastOp>(op);
  if (!uccOp)
    return {};

  auto inVal = uccOp.getInputs()[0];
  if (!isa<emitc::OpaqueType>(inVal.getType()))
    return {};

  auto callOp = inVal.getDefiningOp<emitc::CallOpaqueOp>();
  if (callOp.getCallee() != funcName)
    return {};

  auto callOperandsUccOp =
      callOp.getOperands()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!callOperandsUccOp)
    return {};

  return callOperandsUccOp.getInputs()[0];
}

// Check there is an operation chain like-
//
//      %cst_0 = arith.constant dense<1.000000e+00> : vector<16xbf16>
//      %cst_1 = arith.constant 0.000000e+00 : bf16
//      %0 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>,
//      vector<16xbf16>
//      %1 = arith.negf %0 : vector<16xbf16>
//      %2 = math.exp %1 : vector<16xbf16>
//      %3 = arith.addf %2, %cst_0 : vector<16xbf16>
//      %4 = arith.divf %cst_0, %3 : vector<16xbf16>
//
// so that this operation chain can be converted to a function call to compute
// sigmoid value for v16bfloat16 and v32bfloat16 types
template <typename DivFOpTy>
static bool hasSigmoidComputationChain(DivFOpTy divfOp, arith::NegFOp &negOp) {
  auto *lhsDefOp = divfOp.getLhs().getDefiningOp();
  if (!lhsDefOp)
    return false;
  auto constOp = dyn_cast<arith::ConstantOp>(lhsDefOp);
  if (!constOp)
    return false;

  auto cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;

  if (cstDense.template getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  Operation *addLvalOp;
  Operation *addRvalOp;
  // divfOp's rval could be an arith::AddFOp or the pattern like-
  // %1 = aievec.ups %a
  // %2 = aievec.ups %b;
  // %3 = aievec.add_elem %1, %2
  // %4 = aievec.srs %3;
  auto *rhsDefOp = divfOp.getRhs().getDefiningOp();
  if (!rhsDefOp)
    return false;
  auto addOp = dyn_cast<arith::AddFOp>(rhsDefOp);
  if (!addOp) {
    auto srsOp = dyn_cast<aievec::SRSOp>(rhsDefOp);
    if (!srsOp)
      return false;

    auto addElemOp =
        dyn_cast<aievec::AddElemOp>(srsOp.getSource().getDefiningOp());
    if (!addElemOp)
      return false;

    auto lUpsOp = dyn_cast<aievec::UPSOp>(addElemOp.getLhs().getDefiningOp());
    auto rUpsOp = dyn_cast<aievec::UPSOp>(addElemOp.getRhs().getDefiningOp());
    if (!lUpsOp || !rUpsOp)
      return false;

    addLvalOp = lUpsOp.getSource().getDefiningOp();
    addRvalOp = rUpsOp.getSource().getDefiningOp();
    // One of add operation's operand is a constant op and another operand could
    // be arith::ExpOp or the combination of emitc.call and aievec.srs
    auto addDefOp = isa<arith::ConstantOp>(addLvalOp)
                        ? dyn_cast<aievec::SRSOp>(addRvalOp)
                        : dyn_cast<aievec::SRSOp>(addLvalOp);
    if (!addDefOp)
      addLvalOp = isa<arith::ConstantOp>(addLvalOp)
                      ? dyn_cast<math::ExpOp>(addRvalOp)
                      : dyn_cast<math::ExpOp>(addLvalOp);
    else
      addLvalOp = addDefOp.getSource().getDefiningOp();

    addRvalOp = isa<arith::ConstantOp>(addLvalOp)
                    ? lUpsOp.getSource().getDefiningOp()
                    : rUpsOp.getSource().getDefiningOp();
  } else {
    addLvalOp = addOp.getLhs().getDefiningOp();
    addRvalOp = addOp.getRhs().getDefiningOp();
  }

  if (!addLvalOp || !addRvalOp)
    return false;

  auto addLvalExpOp = dyn_cast<math::ExpOp>(addLvalOp);
  auto addRvalExpOp = dyn_cast<math::ExpOp>(addRvalOp);
  auto addLvalExpOpIn =
      getUnOpaquedOperandOfEmitCOpaqueCallOp(addLvalOp, "getExpBf16")
          .value_or(nullptr);
  auto addRvalExpOpIn =
      getUnOpaquedOperandOfEmitCOpaqueCallOp(addRvalOp, "getExpBf16")
          .value_or(nullptr);
  if (!addLvalExpOpIn && addLvalExpOp)
    addLvalExpOpIn = addLvalExpOp.getOperand();
  if (!addRvalExpOpIn && addRvalExpOp)
    addRvalExpOpIn = addRvalExpOp.getOperand();

  if (!((addLvalExpOpIn && isa<arith::ConstantOp>(addRvalOp)) ||
        (addRvalExpOpIn && isa<arith::ConstantOp>(addLvalOp))))
    return false;

  constOp = isa<arith::ConstantOp>(addLvalOp)
                ? cast<arith::ConstantOp>(addLvalOp)
                : cast<arith::ConstantOp>(addRvalOp);

  cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;
  if (cstDense.template getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  auto expOperand = addLvalExpOpIn ? addLvalExpOpIn : addRvalExpOpIn;

  negOp = expOperand.getDefiningOp<arith::NegFOp>();

  return negOp != nullptr;
}

// Convert the operation chain like-
//
//      %cst_0 = arith.constant dense<1.000000e+00> : vector<16xbf16>
//      %cst_1 = arith.constant 0.000000e+00 : bf16
//      %0 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>,
//      vector<16xbf16>
//      %1 = arith.negf %0 : vector<16xbf16>
//      %2 = math.exp %1 :vector<16xbf16>
//      %3 = arith.addf %2, %cst_0 : vector<16xbf16>
//      %4 = arith.divf %cst_0, %3 : vector<16xbf16>
//
// to a function call to compute sigmoid value for v16bfloat16 and
// v32bfloat16 types
struct ComputeSigmoidOpPattern : OpConversionPattern<arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivFOp divfOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    arith::NegFOp negOp = nullptr;
    if (!hasSigmoidComputationChain(adaptor, negOp))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = divfOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(divfOp);
    Type vecOpaqueTy;
    if (laneSize == 16)
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, divfOp.getLoc(),
                                           vecOpaqueTy, negOp.getOperand())
            .getResult(0);
    SmallVector<Value> sigmoidOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, divfOp.getLoc(), TypeRange{vecOpaqueTy}, "getSigmoidBf16",
        nullptr, nullptr, sigmoidOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        divfOp, TypeRange{adaptor.getLhs().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.ceil to a function call to compute ceil(x) for v16bfloat16
struct ComputeCeilOpPattern : OpConversionPattern<math::CeilOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CeilOp ceilOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(ceilOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = ceilOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(ceilOp);
    Type vecOpaqueTy;
    if (laneSize == 16)
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, ceilOp.getLoc(),
                                           vecOpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> ceilOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, ceilOp.getLoc(), TypeRange{vecOpaqueTy}, "getCeilBf16",
        nullptr, nullptr, ceilOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        ceilOp, TypeRange{ceilOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.floor to a function call to compute floor(x) for v16bfloat16
struct ComputeFloorOpPattern : OpConversionPattern<math::FloorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::FloorOp floorOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(floorOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = floorOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(floorOp);
    Type vecOpaqueTy;
    if (laneSize == 16)
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, floorOp.getLoc(),
                                           vecOpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> floorOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, floorOp.getLoc(), TypeRange{vecOpaqueTy}, "getFloorBf16",
        nullptr, nullptr, floorOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        floorOp, TypeRange{floorOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert arith.negf to aievec.neg to negate the vector for v16bfloat16 and
// v16float types, and for v32 of either on AIE2P.
struct ComputeNegOpPattern : OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::NegFOp negOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(negOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    if (laneSize != 16 && laneSize != 32)
      return failure();

    Location loc = negOp.getLoc();
    auto accType = getVectorOpDestType(srcType, /*AIE2 =*/true);

    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth == 16) {
      auto upsOp =
          aievec::UPSOp::create(rewriter, loc, accType, adaptor.getOperand());
      auto aieNegOp =
          aievec::NegOp::create(rewriter, loc, accType, upsOp.getResult());
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, negOp.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          negOp, srcType, aieNegOp.getResult(), shiftParamOp.getResult());
    } else {
      auto castOp = aievec::CastOp::create(
          rewriter, loc, accType, adaptor.getOperand(), /*isResAcc*/ true);
      auto aieNegOp =
          aievec::NegOp::create(rewriter, loc, accType, castOp.getResult());
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          negOp, srcType, aieNegOp.getResult(), /*isResAcc*/ false);
    }

    return success();
  }
};

// Check whether the value of constant operation is int type and the dense value
// is -1.
static bool hasConstNegOneValue(arith::ConstantOp constOp, unsigned elWidth) {
  if (!constOp)
    return false;

  auto cstDense = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;

  if (elWidth == 32)
    return cstDense.getSplatValue<int32_t>() == -1;
  if (elWidth == 16)
    return cstDense.getSplatValue<int16_t>() == -1;
  if (elWidth == 8)
    return cstDense.getSplatValue<int8_t>() == -1;
  return false;
}

// Convert arith.xori to aievec.bxor to compute bitwise xor of two vectors for
// integer types
struct ComputeBxorAndBnegOpPattern : OpConversionPattern<arith::XOrIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp xorOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(xorOp.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (laneSize * elWidth != 512)
      return failure();

    auto lhsConstOp = xorOp.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhsConstOp = xorOp.getRhs().getDefiningOp<arith::ConstantOp>();

    // If one of operands in xorOp is a constant -1, xorOp will be replaced with
    // aievec::BnegOp.
    if ((lhsConstOp && hasConstNegOneValue(lhsConstOp, elWidth)) ||
        (rhsConstOp && hasConstNegOneValue(rhsConstOp, elWidth))) {
      Value val = hasConstNegOneValue(lhsConstOp, elWidth) ? adaptor.getRhs()
                                                           : adaptor.getLhs();
      rewriter.replaceOpWithNewOp<aievec::BnegOp>(xorOp, srcType, val);
    } else
      rewriter.replaceOpWithNewOp<aievec::BxorOp>(
          xorOp, srcType, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

template <typename SrcOpTy, typename DstOpTy>
struct ComputeBandAndBorOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = dyn_cast<VectorType>(srcOp.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (laneSize * elWidth != 512)
      return failure();

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getResult().getType(),
                                         adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

using ComputeBorOpPattern =
    ComputeBandAndBorOpPattern<arith::OrIOp, aievec::BorOp>;
using ComputeBandOpPattern =
    ComputeBandAndBorOpPattern<arith::AndIOp, aievec::BandOp>;

// Convert arith.shrsi to a combination of aievec.ups and aievec.srs to compute
// arithmetic right shift for integer types. Currently, only support the shift
// value with a broadcast vector.
struct ComputeSignedIntRightShiftOpPattern
    : OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRSIOp rsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (laneSize * elWidth != 512)
      return failure();

    auto bcastOp =
        dyn_cast<aievec::BroadcastOp>(adaptor.getRhs().getDefiningOp());
    if (!bcastOp)
      return failure();

    auto constOp =
        arith::ConstantOp::create(rewriter, bcastOp.getLoc(),
                                  rewriter.getI32IntegerAttr(bcastOp.getIdx()));
    auto extElemOp = aievec::ExtElemOp::create(
        rewriter, bcastOp.getLoc(), scalarType, bcastOp, constOp.getResult());
    Location loc = rsOp.getLoc();

    // The vector with v64int8 type can be divided into two v32int8 vectors and
    // be processed individually and be concatenated at the end.
    if (elWidth == 8) {
      VectorType halfSrcType = createVectorType(laneSize / 2, scalarType);
      auto rsOpLow = aievec::ExtOp::create(rewriter, loc, halfSrcType,
                                           adaptor.getLhs(), 0);
      auto rsOpHigh = aievec::ExtOp::create(rewriter, loc, halfSrcType,
                                            adaptor.getLhs(), 1);
      Type accType = getVectorOpDestType(halfSrcType, /*AIE2 =*/true);
      auto upsOpLow =
          aievec::UPSOp::create(rewriter, loc, accType, rsOpLow.getResult());
      auto srsOpLow =
          aievec::SRSOp::create(rewriter, loc, halfSrcType,
                                upsOpLow.getResult(), extElemOp.getResult());
      auto upsOpHigh =
          aievec::UPSOp::create(rewriter, loc, accType, rsOpHigh.getResult());
      auto srsOpHigh =
          aievec::SRSOp::create(rewriter, loc, halfSrcType,
                                upsOpHigh.getResult(), extElemOp.getResult());
      SmallVector<Value> inputSources = {srsOpLow.getResult(),
                                         srsOpHigh.getResult()};
      rewriter.replaceOpWithNewOp<aievec::ConcatOp>(rsOp, srcType,
                                                    inputSources);
    } else {
      Type accType = getVectorOpDestType(srcType, /*AIE2 =*/true);
      auto upsOp =
          aievec::UPSOp::create(rewriter, loc, accType, adaptor.getLhs());
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          rsOp, srcType, upsOp.getResult(), extElemOp.getResult());
    }

    return success();
  }
};

// Recognize the compound shift+clamp+truncate pattern and lower it to
// aievec.ups + aievec.srs. This maps the standard quantized neural network
// "shift-round-saturate + clamp" idiom to the AIE2 SRS hardware unit.
//
// Pattern A (with clamp):
//   %shifted  = arith.shrsi %wide, %shift_splat
//   %clamped0 = arith.maxsi %shifted, %c_lo
//   %clamped  = arith.minsi %clamped0, %c_hi
//   %result   = arith.trunci %clamped
//
// Pattern B (without clamp):
//   %shifted  = arith.shrsi %wide, %shift_splat
//   %result   = arith.trunci %shifted
//
// Both lower to: aievec.ups + aievec.srs
struct ShiftClampTruncToSRSPattern : OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  ShiftClampTruncToSRSPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpConversionPattern(context, benefit) {}

  // Try to extract a scalar integer splat value from a Value.
  // Returns std::nullopt if the value is not a constant splat.
  static std::optional<int64_t> getConstantSplatValue(Value val) {
    auto defOp = val.getDefiningOp<arith::ConstantOp>();
    if (!defOp)
      return std::nullopt;
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(defOp.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return std::nullopt;
    return denseAttr.getSplatValue<APInt>().getSExtValue();
  }

  // Try to extract the shift amount from the right operand of shrsi.
  // Accepts either a constant splat vector or an aievec.broadcast of a
  // constant.
  static std::optional<Value>
  getShiftValue(Value rhs, ConversionPatternRewriter &rewriter, Location loc) {
    // Case 1: constant splat vector
    if (auto constOp = rhs.getDefiningOp<arith::ConstantOp>()) {
      auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
      if (denseAttr && denseAttr.isSplat()) {
        int64_t shiftVal = denseAttr.getSplatValue<APInt>().getSExtValue();
        return arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getI32IntegerAttr(shiftVal))
            .getResult();
      }
    }
    // Case 2: aievec.broadcast
    if (auto bcastOp = dyn_cast<aievec::BroadcastOp>(rhs.getDefiningOp())) {
      auto constOp = arith::ConstantOp::create(
          rewriter, bcastOp.getLoc(),
          rewriter.getI32IntegerAttr(bcastOp.getIdx()));
      return aievec::ExtElemOp::create(rewriter, bcastOp.getLoc(),
                                       rewriter.getI32Type(), bcastOp,
                                       constOp.getResult())
          .getResult();
    }
    return std::nullopt;
  }

  LogicalResult
  matchAndRewrite(arith::TruncIOp truncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = dyn_cast<VectorType>(truncOp.getOut().getType());
    if (!dstType)
      return failure();

    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(dstScalarType))
      return failure();

    // Walk backward through optional clamp chain
    Value source = adaptor.getIn();
    int32_t sign = 1; // default: signed

    // Check for minsi(maxsi(...), hi) or maxsi(minsi(...), lo) clamp pattern
    arith::MinSIOp minOp = nullptr;
    arith::MaxSIOp maxOp = nullptr;

    if (auto minsiOp = source.getDefiningOp<arith::MinSIOp>()) {
      if (auto maxsiOp = minsiOp.getLhs().getDefiningOp<arith::MaxSIOp>()) {
        minOp = minsiOp;
        maxOp = maxsiOp;
        source = maxOp.getLhs();
      }
    } else if (auto maxsiOp = source.getDefiningOp<arith::MaxSIOp>()) {
      if (auto minsiOp = maxsiOp.getLhs().getDefiningOp<arith::MinSIOp>()) {
        maxOp = maxsiOp;
        minOp = minsiOp;
        source = minOp.getLhs();
      }
    }

    // If we found a clamp, verify it's a valid saturation range
    if (minOp && maxOp) {
      auto loVal = getConstantSplatValue(maxOp.getRhs());
      auto hiVal = getConstantSplatValue(minOp.getRhs());
      if (!loVal || !hiVal)
        return failure();

      unsigned dstBits = dstScalarType.getIntOrFloatBitWidth();
      // Guard against UB from shifting into or past the sign bit.
      if (dstBits == 0 || dstBits > 63)
        return failure();
      uint64_t one = 1ULL;
      int64_t unsignedLo = 0;
      int64_t unsignedHi = static_cast<int64_t>((one << dstBits) - 1);
      int64_t signedLo = -static_cast<int64_t>(one << (dstBits - 1));
      int64_t signedHi = static_cast<int64_t>((one << (dstBits - 1)) - 1);

      if (*loVal == unsignedLo && *hiVal == unsignedHi) {
        sign = 0; // unsigned saturation
      } else if (*loVal == signedLo && *hiVal == signedHi) {
        sign = 1; // signed saturation
      } else {
        // Clamp range doesn't match standard saturation — don't match
        return failure();
      }
    }

    // Now source should be the shrsi result
    auto shrsiOp = source.getDefiningOp<arith::ShRSIOp>();
    if (!shrsiOp)
      return failure();

    auto srcType = dyn_cast<VectorType>(shrsiOp.getLhs().getType());
    if (!srcType)
      return failure();

    Type srcScalarType = srcType.getElementType();
    if (!isa<IntegerType>(srcScalarType))
      return failure();

    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    if (dstElWidth >= srcElWidth)
      return failure();

    Location loc = truncOp.getLoc();

    // Extract the shift amount
    auto shiftVal = getShiftValue(shrsiOp.getRhs(), rewriter, loc);
    if (!shiftVal)
      return failure();

    // Get the wide input (pre-shift)
    Value wideInput = shrsiOp.getLhs();

    unsigned laneSize = getVectorLaneSize(srcType);
    bool needsPadding = (laneSize % 16 != 0);

    VectorType paddedSrcType = srcType;
    VectorType paddedDstType = dstType;

    if (needsPadding) {
      // Round up to nearest multiple of 16
      unsigned paddedLanes = ((laneSize + 15) / 16) * 16;
      paddedSrcType = createVectorType(paddedLanes, srcScalarType);
      paddedDstType = createVectorType(paddedLanes, dstScalarType);

      // Zero-pad the input using insert_strided_slice
      auto zeroAttr = rewriter.getZeroAttr(paddedSrcType);
      auto zeroPad =
          arith::ConstantOp::create(rewriter, loc, zeroAttr).getResult();
      SmallVector<int64_t> offsets(1, 0);
      SmallVector<int64_t> strides(1, 1);
      wideInput = vector::InsertStridedSliceOp::create(
                      rewriter, loc, wideInput, zeroPad, offsets, strides)
                      .getResult();
    }

    // Determine accumulator type and create the accumulator value.
    // For i16 source: use UPS to widen to accumulator type.
    // For i32/i64 source: use CastOp (marks as accumulator without widening),
    // matching the approach in LowerTruncOpPattern.
    Type accScalarType = paddedSrcType.getElementType();
    unsigned accElWidth = accScalarType.getIntOrFloatBitWidth();
    Value accValue;
    if (accElWidth == 16) {
      Type accType = getVectorOpDestType(paddedSrcType, /*AIE2=*/true);
      accValue =
          aievec::UPSOp::create(rewriter, loc, accType, wideInput).getResult();
    } else {
      // For i32/i64: CastOp with isResAcc=true marks as accumulator
      accValue = aievec::CastOp::create(rewriter, loc, paddedSrcType, wideInput,
                                        /*isResAcc=*/true)
                     .getResult();
    }
    auto srsOp = aievec::SRSOp::create(rewriter, loc, paddedDstType, accValue,
                                       *shiftVal, sign);

    Value result = srsOp.getResult();

    if (needsPadding) {
      // Extract original lanes from the padded result
      SmallVector<int64_t> offsets(1, 0);
      SmallVector<int64_t> sizes = {static_cast<int64_t>(laneSize)};
      SmallVector<int64_t> strides(1, 1);
      result = vector::ExtractStridedSliceOp::create(rewriter, loc, result,
                                                     offsets, sizes, strides)
                   .getResult();
    }

    rewriter.replaceOp(truncOp, result);

    // Erase the intermediate clamp/shift ops if they have no other uses.
    // These must be cleaned up because shrsi on 512-bit vectors is marked
    // illegal and would cause conversion failure if left as dead ops.
    SmallVector<Operation *, 3> opsToErase;
    if (minOp && minOp->use_empty())
      opsToErase.push_back(minOp);
    if (maxOp && maxOp->use_empty())
      opsToErase.push_back(maxOp);
    if (shrsiOp->use_empty())
      opsToErase.push_back(shrsiOp);
    for (Operation *op : opsToErase)
      rewriter.eraseOp(op);

    return success();
  }
};

// Promote scalar shrsi + [clamp] + trunci chain to a vectorized SRS sequence.
// Anchored on scalar arith::TruncIOp. Fuses the entire chain into:
//   broadcast_scalar -> cast(isResAcc) -> srs(narrowed, shift, sign) ->
//   ext_elem
// This prevents scalar arith.trunci i32->i8 from reaching the AIE2 backend
// where it crashes (the backend only supports 32->16/20/32 truncations).
struct LowerScalarShiftClampTruncToSRS : OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncIOp truncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match scalar types (vector compound pattern handles vectors)
    if (isa<VectorType>(truncOp.getOut().getType()))
      return failure();
    std::optional<ScalarSRSChain> chain = matchScalarSRSChain(truncOp);
    if (!chain)
      return failure();
    Value source = chain->source;
    int32_t sign = chain->sign;
    arith::MinSIOp minOp = chain->minOp;
    arith::MaxSIOp maxOp = chain->maxOp;
    auto srcIntType = cast<IntegerType>(truncOp.getIn().getType());
    auto dstIntType = cast<IntegerType>(truncOp.getOut().getType());
    unsigned dstBits = dstIntType.getWidth();

    // Now source should be the shrsi result, or any i32 value for clamp-only
    Location loc = truncOp.getLoc();
    Value preShiftVal;
    Value shiftVal;
    arith::ShRSIOp shrsiOp = source.getDefiningOp<arith::ShRSIOp>();
    if (shrsiOp) {
      // Verify shrsi operand types
      if (!isa<IntegerType>(shrsiOp.getLhs().getType()))
        return failure();
      preShiftVal = shrsiOp.getLhs();
      shiftVal = shrsiOp.getRhs();
    } else {
      // No shrsi found: treat as identity shift (shift=0).
      // This handles clamp+trunci patterns without a preceding shift,
      // e.g., after skip-add with skip_scale=0.
      preShiftVal = source;
      shiftVal = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getI32IntegerAttr(0));
    }

    // Create 512-bit vector type: vector<16xi32>
    unsigned srcLanes = 512 / srcIntType.getWidth(); // 16 for i32
    VectorType bcastVecType = createVectorType(srcLanes, srcIntType);

    // Broadcast pre-shift value to 512-bit vector
    auto bcast = aievec::BroadcastScalarOp::create(rewriter, loc, bcastVecType,
                                                   preShiftVal);

    Value accValue;
    VectorType srsOutType;

    if (dstBits == 8) {
      // i32→i8: The SRS intrinsic I256V32Acc32Srs needs 1024-bit source
      // (vector<32xi32>, cast to vector<16xi64> internally) and produces
      // vector<32xi8> (256-bit). Concat two broadcast copies to get 1024 bits.
      unsigned accLanes = srcLanes * 2; // 32
      VectorType accVecType =
          createVectorType(accLanes, srcIntType); // vector<32xi32>
      auto concatSrc = aievec::ConcatOp::create(
          rewriter, loc, accVecType,
          SmallVector<Value>({bcast.getResult(), bcast.getResult()}));
      accValue =
          aievec::CastOp::create(rewriter, loc, accVecType,
                                 concatSrc.getResult(), /*isResAcc=*/true)
              .getResult();
      srsOutType = createVectorType(accLanes, dstIntType); // vector<32xi8>
    } else {
      // i32→i16: The SRS intrinsic I256V16Acc32Srs needs 512-bit source
      // (vector<16xi32>, cast to vector<8xi64> internally) and produces
      // vector<16xi16> (256-bit). 512-bit broadcast works directly.
      accValue = aievec::CastOp::create(rewriter, loc, bcastVecType,
                                        bcast.getResult(), /*isResAcc=*/true)
                     .getResult();
      srsOutType = createVectorType(srcLanes, dstIntType); // vector<16xi16>
    }

    // SRS: accumulator → narrowed output with shift and sign
    auto srsOp = aievec::SRSOp::create(rewriter, loc, srsOutType, accValue,
                                       shiftVal, sign);

    // ExtElem needs 512-bit source. SRS output is 256-bit, so concat to 512.
    unsigned extLanes = 512 / dstBits; // 64 for i8, 32 for i16
    VectorType extVecType = createVectorType(extLanes, dstIntType);
    auto concatForExt = aievec::ConcatOp::create(
        rewriter, loc, extVecType,
        SmallVector<Value>({srsOp.getResult(), srsOp.getResult()}));

    // Extract element 0 back to scalar
    auto zeroIdx =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(
        truncOp, dstIntType, concatForExt.getResult(), zeroIdx.getResult());

    // Erase dead intermediate ops
    SmallVector<Operation *, 3> opsToErase;
    if (minOp && minOp->use_empty())
      opsToErase.push_back(minOp);
    if (maxOp && maxOp->use_empty())
      opsToErase.push_back(maxOp);
    if (shrsiOp && shrsiOp->use_empty())
      opsToErase.push_back(shrsiOp);
    for (Operation *op : opsToErase)
      rewriter.eraseOp(op);

    return success();
  }
};

// Promote scalar arith.shrsi to vector aievec.ups + aievec.srs to prevent
// LLVM's SLP vectorizer from creating sub-512-bit vector shifts that the
// AIE2 backend cannot legalize (G_LSHR on <4 x s32>).
struct LowerScalarShRSIToAIEVecUPSSRS : OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRSIOp rsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match scalar i32
    Type resultType = rsOp.getType();
    if (isa<VectorType>(resultType))
      return failure();

    auto intType = dyn_cast<IntegerType>(resultType);
    if (!intType || intType.getWidth() != 32)
      return failure();

    Location loc = rsOp.getLoc();
    VectorType vecType = createVectorType(16, intType); // vector<16xi32>

    // Broadcast scalar value to 512-bit vector
    auto lhsBcast = aievec::BroadcastScalarOp::create(rewriter, loc, vecType,
                                                      adaptor.getLhs());

    // UPS: vector<16xi32> -> accumulator type (vector<16xi64>)
    Type accType = getVectorOpDestType(vecType, /*AIE2=*/true);
    auto upsOp =
        aievec::UPSOp::create(rewriter, loc, accType, lhsBcast.getResult());

    // SRS: accumulator + i32 shift -> vector<16xi32>
    auto srsOp = aievec::SRSOp::create(rewriter, loc, vecType,
                                       upsOp.getResult(), adaptor.getRhs());

    // Extract element 0 back to scalar
    auto zeroIdx =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(
        rsOp, intType, srsOp.getResult(), zeroIdx.getResult());
    return success();
  }
};

// A `vector.contract` read as one matmul tile: the sizes of its m, n and k
// dims, and the order each operand stores them in once its unit dims are
// dropped. An m (n) dim indexes lhs (rhs) and acc, a k dim indexes lhs and
// rhs and is reduced. Unit dims may sit anywhere and play any role.
struct ContractionTile {
  int64_t m = 1, n = 1, k = 1;
  // lhs stored [k, m], rhs stored [n, k], acc stored [n, m]. Only set when
  // both dims of the operand are larger than 1.
  bool lhsTransposed = false, rhsTransposed = false, accTransposed = false;
};

static LogicalResult analyzeContraction(vector::ContractionOp op,
                                        ContractionTile &tile,
                                        std::string &why) {
  llvm::raw_string_ostream os(why);
  if (op.getKind() != vector::CombiningKind::ADD) {
    os << "the combining kind is not add";
    return failure();
  }
  auto accTy = dyn_cast<VectorType>(op.getAccType());
  if (!accTy) {
    os << "the accumulator is a scalar";
    return failure();
  }
  SmallVector<AffineMap, 3> maps = op.getIndexingMapsArray();
  if (!llvm::all_of(maps, [](AffineMap map) {
        return map.isProjectedPermutation(/*allowZeroInResults=*/false);
      })) {
    os << "an indexing map is not a projected permutation";
    return failure();
  }
  enum Role { M, N, K, Other };
  unsigned numDims = maps[0].getNumDims();
  SmallVector<int64_t> dimSize(numDims, 1);
  SmallVector<std::array<bool, 3>> indexes(numDims, {false, false, false});
  ArrayRef<int64_t> shapes[3] = {op.getLhsType().getShape(),
                                 op.getRhsType().getShape(), accTy.getShape()};
  for (unsigned operand = 0; operand < 3; ++operand)
    for (auto [i, expr] : llvm::enumerate(maps[operand].getResults())) {
      unsigned d = cast<AffineDimExpr>(expr).getPosition();
      dimSize[d] = shapes[operand][i];
      indexes[d][operand] = true;
    }
  SmallVector<vector::IteratorType> iterators = op.getIteratorTypesArray();
  std::optional<unsigned> dimOf[3];
  for (unsigned d = 0; d < numDims; ++d) {
    if (dimSize[d] == 1)
      continue;
    auto [inLhs, inRhs, inAcc] = indexes[d];
    bool reduced = iterators[d] == vector::IteratorType::reduction;
    Role r = Other;
    if (inLhs && !inRhs && inAcc && !reduced)
      r = M;
    else if (!inLhs && inRhs && inAcc && !reduced)
      r = N;
    else if (inLhs && inRhs && !inAcc && reduced)
      r = K;
    if (r == Other || dimOf[r]) {
      os << "d" << d << " of size " << dimSize[d]
         << " is not the only m, n or k dim larger than 1";
      return failure();
    }
    dimOf[r] = d;
  }
  auto sizeOf = [&](Role r) -> int64_t {
    std::optional<unsigned> d = dimOf[r];
    return d ? dimSize[*d] : 1;
  };
  tile.m = sizeOf(M);
  tile.n = sizeOf(N);
  tile.k = sizeOf(K);
  // Whether `second` comes before `first` among the results of `map`.
  auto swapped = [&](AffineMap map, Role first, Role second) {
    if (!dimOf[first] || !dimOf[second])
      return false;
    return *map.getResultPosition(
               getAffineDimExpr(*dimOf[second], map.getContext())) <
           *map.getResultPosition(
               getAffineDimExpr(*dimOf[first], map.getContext()));
  };
  tile.lhsTransposed = swapped(maps[0], M, K);
  tile.rhsTransposed = swapped(maps[1], K, N);
  tile.accTransposed = swapped(maps[2], M, N);
  return success();
}

// Convert a `vector.contract` op to an `aievec.matmul` op for AIE2 or
// `aievec.matmul_aie2p` for AIE2P. The operands keep the layout the
// contraction stores them in, recorded by the lhs_transposed and
// rhs_transposed flags; the lowering of the matmul op moves data only where
// the multiply reads an operand the other way. An accumulator stored [n, m]
// is handled by multiplying rhs^T by lhs^T, which swaps the operands and
// flips both of their flags.
template <typename MatMulOpTy>
struct LowerVectorContractionOpToAIEVecMatMulPattern
    : OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern::OpConversionPattern;

  static constexpr bool isAIE2P =
      std::is_same_v<MatMulOpTy, aievec::MatMulOp_AIE2P>;

  static Value reshape(OpBuilder &b, Value v, ArrayRef<int64_t> shape) {
    auto vecTy = cast<VectorType>(v.getType());
    auto newTy = VectorType::get(shape, vecTy.getElementType());
    if (newTy == vecTy)
      return v;
    return vector::ShapeCastOp::create(b, v.getLoc(), newTy, v).getResult();
  }

  LogicalResult
  matchAndRewrite(vector::ContractionOp contractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef target = isAIE2P ? "AIE2P" : "AIE2";
    std::string why;
    ContractionTile tile;
    if (failed(analyzeContraction(contractOp, tile, why))) {
      contractOp.emitError()
          << "cannot lower to an " << target << " matrix multiply: " << why;
      return failure();
    }

    Value lhsIn = adaptor.getLhs(), rhsIn = adaptor.getRhs();
    Value lhsOrig = contractOp.getLhs(), rhsOrig = contractOp.getRhs();
    int64_t m = tile.m, n = tile.n, k = tile.k;
    bool lhsT = tile.lhsTransposed, rhsT = tile.rhsTransposed;
    if (tile.accTransposed) {
      std::swap(lhsIn, rhsIn);
      std::swap(lhsOrig, rhsOrig);
      std::swap(m, n);
      // A flag only matters, and is only set, when both dims are larger
      // than 1.
      std::tie(lhsT, rhsT) =
          std::make_pair(!rhsT && m > 1 && k > 1, !lhsT && n > 1 && k > 1);
    }
    SmallVector<int64_t, 2> lhsShape =
        lhsT ? SmallVector<int64_t, 2>{k, m} : SmallVector<int64_t, 2>{m, k};
    SmallVector<int64_t, 2> rhsShape =
        rhsT ? SmallVector<int64_t, 2>{n, k} : SmallVector<int64_t, 2>{k, n};
    auto accTy =
        VectorType::get({m, n}, getElementTypeOrSelf(adaptor.getAcc()));

    // Recover operand signedness from the widening op before it is peeled.
    // Consult both the (possibly already-rewritten) adaptor operands and the
    // original pre-conversion contraction operands.
    auto lhsSigned = getSignednessOfWideningOp(lhsIn);
    if (!lhsSigned)
      lhsSigned = getSignednessOfWideningOp(lhsOrig);
    auto rhsSigned = getSignednessOfWideningOp(rhsIn);
    if (!rhsSigned)
      rhsSigned = getSignednessOfWideningOp(rhsOrig);

    // Carry the signedness in the matmul operand element type (si8/ui8). MLIR
    // has no signless->signed/unsigned cast, so materialize an
    // unrealized_conversion_cast; it folds to a no-op once both sides lower to
    // signless LLVM integers. bf16/unknown-signedness operands are unchanged.
    auto retype = [&](Value v, std::optional<bool> isSigned) -> Value {
      if (!isSigned)
        return v;
      auto vecTy = cast<VectorType>(v.getType());
      auto intTy = dyn_cast<IntegerType>(vecTy.getElementType());
      if (!intTy || !intTy.isSignless())
        return v;
      auto signednessTy = IntegerType::get(intTy.getContext(), intTy.getWidth(),
                                           *isSigned ? IntegerType::Signed
                                                     : IntegerType::Unsigned);
      return UnrealizedConversionCastOp::create(rewriter, v.getLoc(),
                                                vecTy.clone(signednessTy), v)
          .getResult(0);
    };

    auto tileFits = [&](Value lhs, Value rhs, bool report) {
      auto emitError = [&]() {
        return contractOp.emitError()
               << "cannot lower to an " << target << " matrix multiply: ";
      };
      return checkMatMulTile(
                 isAIE2P, VectorType::get(lhsShape, getElementTypeOrSelf(lhs)),
                 VectorType::get(rhsShape, getElementTypeOrSelf(rhs)), accTy,
                 lhsT, rhsT,
                 report ? llvm::function_ref<InFlightDiagnostic()>(emitError)
                        : nullptr)
          .has_value();
    };
    // Multiply the operands as they are if they form a tile. Otherwise they
    // may be widened from a type a multiply takes, as when a linalg op is
    // converted to contractions with its operands cast to the accumulator
    // type outside the contraction.
    if (!tileFits(lhsIn, rhsIn, /*report=*/false)) {
      lhsIn = getSourceOfWideningOp(lhsIn).value_or(lhsIn);
      rhsIn = getSourceOfWideningOp(rhsIn).value_or(rhsIn);
      if (!tileFits(lhsIn, rhsIn, /*report=*/true))
        return failure();
    }
    Value acc = reshape(rewriter, adaptor.getAcc(), accTy.getShape());
    Value lhs = reshape(rewriter, lhsIn, lhsShape);
    Value rhs = reshape(rewriter, rhsIn, rhsShape);

    auto matmulOp =
        MatMulOpTy::create(rewriter, contractOp.getLoc(), acc.getType(),
                           retype(lhs, lhsSigned), retype(rhs, rhsSigned), acc);
    matmulOp.setLhsTransposed(lhsT);
    matmulOp.setRhsTransposed(rhsT);
    rewriter.replaceOp(
        contractOp,
        reshape(rewriter, matmulOp.getResult(),
                cast<VectorType>(adaptor.getAcc().getType()).getShape()));
    return success();
  }
};

using LowerVectorContractionOpToAIEVecMatMulOpAIE2 =
    LowerVectorContractionOpToAIEVecMatMulPattern<aievec::MatMulOp>;
using LowerVectorContractionOpToAIEVecMatMulOpAIE2P =
    LowerVectorContractionOpToAIEVecMatMulPattern<aievec::MatMulOp_AIE2P>;

// Convert a `vector.transpose` op to an `aievec.shuffle` op for AIE2.
struct LowerVectorTransposeOpToAIEVecShuffleOpPattern
    : OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::TransposeOp transpOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = transpOp.getResultVectorType();
    auto resShape = resTy.getShape();
    auto elemTyBitWidth = resTy.getElementTypeBitWidth();
    auto vBitWidth = std::accumulate(resShape.begin(), resShape.end(),
                                     elemTyBitWidth, std::multiplies<>());
    if (vBitWidth != 512)
      return failure();

    if (elemTyBitWidth != 8 && elemTyBitWidth != 16 && elemTyBitWidth != 32)
      return failure();

    // Verify leading dimensions are all 1.
    for (int64_t i = 0; i < static_cast<int64_t>(resShape.size() - 2); ++i)
      if (resShape[i] != 1)
        return failure();

    // Only permutation of the 2 innermost dimensions are supported.
    ArrayRef<int64_t> perm = transpOp.getPermutation();
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size() - 2); ++i)
      if (perm[i] != i)
        return failure();
    if (perm.back() != static_cast<int64_t>(perm.size() - 2))
      return failure();

    auto shuffleMode = aievec::ShuffleMode::T32_4X4;
    if (elemTyBitWidth == 8) {
      switch (resShape.back()) {
      case 4:
        shuffleMode = aievec::ShuffleMode::T8_4X16;
        break;
      case 8:
        shuffleMode = aievec::ShuffleMode::T8_8X8;
        break;
      case 16:
        shuffleMode = aievec::ShuffleMode::T8_16X4;
        break;
      default:
        return failure();
      }
    } else if (elemTyBitWidth == 16) {
      switch (resShape.back()) {
      case 2:
        shuffleMode = aievec::ShuffleMode::T16_2X16;
        break;
      case 4:
        shuffleMode = aievec::ShuffleMode::T16_4X8;
        break;
      case 8:
        shuffleMode = aievec::ShuffleMode::T16_8X4;
        break;
      case 16:
        shuffleMode = aievec::ShuffleMode::T16_16X2;
        break;
      default:
        return failure();
      }
    } else if (resShape.back() != 4)
      return failure();

    auto flatVecTy =
        VectorType::get({512 / elemTyBitWidth}, resTy.getElementType());
    auto loc = transpOp.getLoc();
    auto flatInput = vector::ShapeCastOp::create(rewriter, loc, flatVecTy,
                                                 adaptor.getVector());
    auto shuffOp = aievec::ShuffleOp::create(rewriter, loc, flatVecTy,
                                             flatInput, nullptr, shuffleMode);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(transpOp, resTy, shuffOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern collection
//===----------------------------------------------------------------------===//

static void
populateAIEVecCommonConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<LowerExtFOpPattern,
               LowerExtSIOpPattern,
               LowerTruncFOpPattern,
               LowerTruncIOpPattern>(patterns.getContext());
  // Match canonical int4 unpack (vector.bitcast + arith.extui) ahead of the
  // generic ExtUI/ExtSI handling; runs on both AIE2 and AIE2P call sites
  // but only triggers for the i4-packed shape, so it is benign for AIE2.
  patterns.add<LowerExtUIOfBitcastI4ToUnpackPattern>(patterns.getContext(),
                                                     /*benefit=*/2);
  // AIE2P magic-number lowering for arith.sitofp v16i16/v32i16 -> v16/v32
  // bf16. Higher benefit than the generic per-lane scalarise pattern; runs
  // on AIE2 too but only fires when the legality predicate marks the op
  // illegal (currently AIE2P only).
  patterns.add<LowerVectorSIToFPI16BF16AIE2pPattern>(
      patterns.getContext(), /*accumulatorAdd=*/false, /*benefit=*/3);
  // clang-format on
}

static void populateAIEVecV1ConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerVectorTransferReadToAIEUPD>(patterns.getContext(), 128, 512,
                                                128, 256);
  // clang-format off
  patterns.add<LowerVectorAddIOpToAIEVecAddOp,
               LowerVectorSubIOpToAIEVecSubOp,
               LowerVectorMulIOpToAIEVecMulOp,
               LowerVectorAddFOpToAIEVecAddOp,
               LowerVectorSubFOpToAIEVecSubOp,
               LowerVectorMulFOpToAIEVecMulOp,
               ConvertMulAddToAIEVecFMAOpPattern,
               FoldSplatToFMAOp,
               LowerVectorExtractStridedSliceOpAIEv1Pattern>(patterns.getContext());
  // clang-format on
}

// Populate common conversion patterns for AIE2 and AIE2P
static void
populateAIEVecV2CommonConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  // TODO: Reorder these alphabetically
  patterns.add<
      LowerVectorAddFOpToAIEVecAddElemOp,
      LowerVectorSubFOpToAIEVecSubElemOp,
      // AIE2/AIE2P have no vector int<->fp instruction; scalarize so each
      // lane reaches the supported scalar libcall path.
      ScalarizeVectorSIToFPOpPattern,
      ScalarizeVectorUIToFPOpPattern,
      ScalarizeVectorFPToSIOpPattern,
      ScalarizeVectorFPToUIOpPattern
    >(patterns.getContext());

  // Add the compound shift+clamp+trunc→SRS pattern with higher benefit
  // so it takes priority over the individual shrsi and trunci patterns.
  patterns.add<ShiftClampTruncToSRSPattern>(patterns.getContext(),
                                            /*benefit=*/2);
  // Scalar version of compound SRS with even higher benefit.
  patterns.add<LowerScalarShiftClampTruncToSRS>(patterns.getContext(),
                                                /*benefit=*/3);
  patterns.add<
      ComputeTanhOpByLUTPattern,
      ComputeSqrtOpAIE2Pattern,
      ComputeErfOpPattern,
      ComputeAbsFOpPattern,
      ComputeAbsIOpPattern,
      ComputeSigmoidOpPattern,
      ComputeCeilOpPattern,
      ComputeFloorOpPattern,
      ComputeNegOpPattern,
      ComputeSignedIntRightShiftOpPattern,
      LowerScalarShRSIToAIEVecUPSSRS,
      ConvertMulIToAIEVecMulElemOpPattern,
      ConvertMulFToAIEVecMulElemOpPattern,
      LowerVectorMinSIOpToAIEVecMinOp,
      LowerScalarMinSIOpToAIEVecMinOp,
      LowerVectorMinimumFOpToAIEVecMinOp,
      LowerVectorMaxSIOpToAIEVecMaxOp,
      LowerScalarMaxSIOpToAIEVecMaxOp,
      LowerVectorMaximumFOpToAIEVecMaxOp,
      LowerVectorMaxNumFFOpToAIEVecMaxOp,
      LowerVectorMinUIOpToCmpSelOp,
      LowerVectorMaxUIOpToCmpSelOp,
      LowerVectorCmpIOpToAIEVecCmpOp,
      LowerVectorCmpFOpToAIEVecCmpOp,
      LowerVectorSelectOpToAIEVecSelOp,
      LowerVectorReductionMinOp,
      LowerVectorReductionMaxOp,
      LowerVectorReductionAddIntOp,
      LowerVectorReductionAddFloatOp,
      FoldVectorExtractAndSplatToAIEBroadcast,
      ConvertSplatToAIEBroadcast,
      ConvertMulAddToAIEVecFMAElemOpPattern,
      ConvertMulAddFToAIEVecFMAElemOpPattern,
      ConvertVectorFMAOpToAIEVecFMAElemOpPattern,
      LowerVectorExtractStridedSliceOpAIE2Pattern,
      LowerVectorTransposeOpToAIEVecShuffleOpPattern
      >(patterns.getContext());
  // clang-format on
}

static void populateAIEVecV2ConversionPatterns(RewritePatternSet &patterns) {
  populateAIEVecV2CommonConversionPatterns(patterns);
  patterns.add<LowerVectorContractionOpToAIEVecMatMulOpAIE2>(
      patterns.getContext());
  patterns.add<LowerVectorReductionAddBfloat16OpAIE2>(patterns.getContext());
  // For AIE2 with LLVMIR backend, use LUT-based exp and rsqrt
  patterns.add<ComputeExpOpByLUTLLVMPattern, ComputeRsqrtOpLLVMAIE2Pattern>(
      patterns.getContext());
}

// AIE2p-specific version of ConvertSplatToAIEBroadcast that supports direct
// 256-bit broadcasts without extract
struct ConvertSplatToAIEBroadcastAIE2p
    : OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (adaptor.getSource().getDefiningOp<vector::ExtractOp>())
      return failure();

    auto resultType = cast<VectorType>(bcastOp.getResult().getType());
    auto flatResultType = getFlattenedVectorType(resultType);
    Type scalarType = resultType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    auto src = bcastOp.getSource();

    // AIE2p supports both 256-bit and 512-bit broadcast directly
    if (laneSize * elWidth == 512 || laneSize * elWidth == 256) {
      Value newOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), flatResultType, src);
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    if (laneSize * elWidth == 1024) {
      VectorType vecType = createVectorType(512 / elWidth, scalarType);
      auto aieBcastOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), vecType, src);
      Value newOp = aievec::ConcatOp::create(
          rewriter, bcastOp.getLoc(), flatResultType,
          SmallVector<Value>({aieBcastOp.getResult(), aieBcastOp.getResult()}));
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    return failure();
  }
};

// Whether FlattenElementwiseVectorToRank1Pattern rewrites `op`: an n-D
// elementwise op whose vector operands all have the shape of its result.
static bool isFlattenableElementwise(Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!resultType || resultType.getRank() == 1 || resultType.isScalable())
    return false;

  // The contraction lowering folds an extf on its operand into the matmul.
  // It looks for the extf directly, so a flattened extf hidden behind a
  // shape_cast would leave the contraction with no lowering.
  if (isa<arith::ExtFOp>(op) &&
      llvm::any_of(op->getUsers(), [](Operation *user) {
        return isa<vector::ContractionOp>(user);
      }))
    return false;

  // Only shapes that agree with the result are elementwise; anything else (a
  // broadcast of a differently shaped vector, say) is left to the patterns
  // that understand it.
  return llvm::all_of(op->getOperandTypes(), [&](Type type) {
    auto operandType = dyn_cast<VectorType>(type);
    return !operandType || operandType.getShape() == resultType.getShape();
  });
}

// Rewrite an elementwise op on an n-D vector into the same op on the rank-1
// vector with the same lane count, between a pair of vector.shape_casts.
//
// The legality predicates count lanes with getVectorLaneSize, the product of
// all dimensions, while the patterns they select match rank-1 operands.
// Flattening first makes the two agree; the widths are left as they were.
// Rank 1 is excluded so this cannot re-match its own output.
template <typename OpTy>
struct FlattenElementwiseVectorToRank1Pattern : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isFlattenableElementwise(op))
      return failure();

    auto resultType = cast<VectorType>(op.getType());
    auto flatResultType = getFlattenedVectorType(resultType);
    Location loc = op.getLoc();

    SmallVector<Value> flatOperands;
    flatOperands.reserve(adaptor.getOperands().size());
    for (Value operand : adaptor.getOperands()) {
      auto operandType = dyn_cast<VectorType>(operand.getType());
      if (!operandType) {
        flatOperands.push_back(operand);
        continue;
      }
      auto flatOperandType = getFlattenedVectorType(operandType);
      // Rematerialise a splat at the flat type instead of shape_casting it:
      // patterns that match on a constant operand (the reciprocal lowering
      // looks for a dividend of 1.0) cannot see through a shape_cast.
      if (auto cstOp = operand.template getDefiningOp<arith::ConstantOp>()) {
        if (auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue())) {
          if (dense.isSplat()) {
            flatOperands.push_back(arith::ConstantOp::create(
                rewriter, loc,
                DenseElementsAttr::get(flatOperandType,
                                       dense.getSplatValue<Attribute>())));
            continue;
          }
        }
      }
      flatOperands.push_back(
          vector::ShapeCastOp::create(rewriter, loc, flatOperandType, operand));
    }

    auto flatOp = OpTy::create(rewriter, loc, TypeRange{flatResultType},
                               flatOperands, op.getProperties(),
                               op->getDiscardableAttrDictionary().getValue());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resultType,
                                                     flatOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4
//===----------------------------------------------------------------------===//

// AIE4 f32 and bf16 MUL/MAC matching. Generic LLVM lowering cannot legalize
// fmul on <32 x float>, so surviving products require native lowering.
static bool isVecF32x32(Type ty) {
  auto vecTy = dyn_cast<VectorType>(ty);
  return vecTy && !vecTy.isScalable() && vecTy.getRank() == 1 &&
         vecTy.getShape()[0] == 32 && vecTy.getElementType().isF32();
}
static bool isVecBf16x32(Type ty) {
  auto vecTy = dyn_cast<VectorType>(ty);
  return vecTy && !vecTy.isScalable() && vecTy.getRank() == 1 &&
         vecTy.getShape()[0] == 32 && vecTy.getElementType().isBF16();
}

static bool hasContract(arith::FastMathFlagsAttr fmf) {
  return fmf && arith::bitEnumContainsAll(fmf.getValue(),
                                          arith::FastMathFlags::contract);
}

/// Return the bf16 source of a fixed vector<32xbf16> -> vector<32xf32> extf.
static Value matchExtFFromBf16(Value v) {
  auto extOp = v.getDefiningOp<arith::ExtFOp>();
  if (!extOp || !isVecF32x32(extOp.getType()) ||
      !isVecBf16x32(extOp.getIn().getType()))
    return nullptr;
  return extOp.getIn();
}

/// Match a widened bf16 product and return its bf16 operands. Widening is
/// exact and native multiplication retains f32 product rounding, so this
/// replacement needs no contraction permission.
static bool matchBf16Product(arith::MulFOp mulOp, Value &a, Value &b) {
  if (!isVecF32x32(mulOp.getType()))
    return false;
  a = matchExtFFromBf16(mulOp.getLhs());
  b = matchExtFFromBf16(mulOp.getRhs());
  return a && b;
}

struct FloatMacMatch {
  arith::MulFOp product;
  Value acc;
  Value lhs;
  Value rhs;
  bool isBf16;
};

/// Select the product of a contract-enabled f32 or widened-bf16 MAC.
/// Both the multiply and add must permit contraction, since fusion removes
/// the intermediate rounding of the product.
/// Legality and both rewrites must agree on this choice when both add operands
/// are products. Prefer the RHS, irrespective of the product's element type.
static std::optional<FloatMacMatch> matchFloatMacChain(arith::AddFOp addOp) {
  if (!isVecF32x32(addOp.getType()) || !hasContract(addOp.getFastmathAttr()))
    return std::nullopt;
  auto tryOperand = [&](Value mulSide,
                        Value accSide) -> std::optional<FloatMacMatch> {
    auto mulOp = mulSide.getDefiningOp<arith::MulFOp>();
    if (!mulOp || !isVecF32x32(mulOp.getType()) ||
        !hasContract(mulOp.getFastmathAttr()))
      return std::nullopt;
    Value a, b;
    if (matchBf16Product(mulOp, a, b))
      return FloatMacMatch{mulOp, accSide, a, b, true};
    return FloatMacMatch{mulOp, accSide, mulOp.getLhs(), mulOp.getRhs(), false};
  };
  if (auto match = tryOperand(addOp.getRhs(), addOp.getLhs()))
    return match;
  return tryOperand(addOp.getLhs(), addOp.getRhs());
}

/// Select the product of a contract-enabled f32 or widened-bf16 multiply-
/// subtract. Subtraction does not commute, so only the right-hand operand is
/// a candidate: `%acc - %a * %b` is a multiply-subtract, `%a * %b - %acc` is
/// not.
static std::optional<FloatMacMatch> matchFloatMscChain(arith::SubFOp subOp) {
  if (!isVecF32x32(subOp.getType()) || !hasContract(subOp.getFastmathAttr()))
    return std::nullopt;
  auto mulOp = subOp.getRhs().getDefiningOp<arith::MulFOp>();
  if (!mulOp || !isVecF32x32(mulOp.getType()) ||
      !hasContract(mulOp.getFastmathAttr()))
    return std::nullopt;
  Value a, b;
  if (matchBf16Product(mulOp, a, b))
    return FloatMacMatch{mulOp, subOp.getLhs(), a, b, true};
  return FloatMacMatch{mulOp, subOp.getLhs(), mulOp.getLhs(), mulOp.getRhs(),
                       false};
}

/// Defer a multiply only if every use disappears through MAC fusion. Native
/// float multiplies are not Pure, so eagerly lowering a product that fusion
/// eliminates would leave an unnecessary, non-eliminable instruction.
/// Conversely, a surviving product must be lowered now: dialect conversion does
/// not revisit operations it has already accepted as legal.
static bool feedsFusableFloatMacChain(arith::MulFOp mulOp) {
  if (mulOp.getResult().use_empty())
    return false;
  return llvm::all_of(mulOp.getResult().getUsers(), [&](Operation *user) {
    std::optional<FloatMacMatch> match;
    if (auto addOp = dyn_cast<arith::AddFOp>(user))
      match = matchFloatMacChain(addOp);
    else if (auto subOp = dyn_cast<arith::SubFOp>(user))
      match = matchFloatMscChain(subOp);
    else
      return false;
    // For addf %m, %m, fusion still needs %m as its accumulator.
    return match && match->product == mulOp && match->acc != mulOp.getResult();
  });
}

struct ConvertMulFToAIE4MulEwiseF32Pattern
    : OpConversionPattern<arith::MulFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulFOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(mulOp.getType());
    if (!vecTy || vecTy.isScalable() || vecTy.getRank() != 1 ||
        !vecTy.getElementType().isF32() || vecTy.getShape()[0] != 32)
      return failure();

    // Let the bf16 pattern use the operands before widening.
    Value a, b;
    if (matchBf16Product(mulOp, a, b))
      return failure();

    rewriter.replaceOpWithNewOp<aievec::AIE4MulEwiseF32Op>(
        mulOp, vecTy, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// arith.mulf(extf(a: bf16), extf(b: bf16)) : vector<32xf32>
//     -> aievec.aie4.mul_ewise_bf16_f32
struct ConvertMulFToAIE4MulEwiseBf16Pattern
    : OpConversionPattern<arith::MulFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulFOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value a, b;
    if (!matchBf16Product(mulOp, a, b))
      return failure();

    rewriter.replaceOpWithNewOp<aievec::AIE4MulEwiseBf16F32Op>(
        mulOp, mulOp.getType(), a, b);
    return success();
  }
};

// arith.addf(acc, arith.mulf(a, b)) : vector<32xf32>, contract/fast on both
//     -> aievec.aie4.mac_ewise_f32
struct ConvertMacFToAIE4MacEwiseF32Pattern
    : OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto match = matchFloatMacChain(addOp);
    if (!match || match->isBf16)
      return failure();

    rewriter.replaceOpWithNewOp<aievec::AIE4MacEwiseF32Op>(
        addOp, addOp.getType(), match->lhs, match->rhs, match->acc);
    return success();
  }
};

// arith.addf(acc, arith.mulf(extf(a: bf16), extf(b: bf16))) : vector<32xf32>,
// contract/fast on both -> aievec.aie4.mac_ewise_bf16_f32
struct ConvertMacFToAIE4MacEwiseBf16Pattern
    : OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto match = matchFloatMacChain(addOp);
    if (!match || !match->isBf16)
      return failure();

    rewriter.replaceOpWithNewOp<aievec::AIE4MacEwiseBf16F32Op>(
        addOp, addOp.getType(), match->lhs, match->rhs, match->acc);
    return success();
  }
};

// arith.subf(acc, arith.mulf(a, b)) : vector<32xf32>, contract/fast on both
//     -> aievec.aie4.msc_ewise_f32
struct ConvertMscFToAIE4MscEwiseF32Pattern
    : OpConversionPattern<arith::SubFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SubFOp subOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto match = matchFloatMscChain(subOp);
    if (!match || match->isBf16)
      return failure();

    rewriter.replaceOpWithNewOp<aievec::AIE4MscEwiseF32Op>(
        subOp, subOp.getType(), match->lhs, match->rhs, match->acc);
    return success();
  }
};

// arith.subf(acc, arith.mulf(extf(a: bf16), extf(b: bf16))) : vector<32xf32>,
// contract/fast on both -> aievec.aie4.msc_ewise_bf16_f32
struct ConvertMscFToAIE4MscEwiseBf16Pattern
    : OpConversionPattern<arith::SubFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SubFOp subOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto match = matchFloatMscChain(subOp);
    if (!match || !match->isBf16)
      return failure();

    rewriter.replaceOpWithNewOp<aievec::AIE4MscEwiseBf16F32Op>(
        subOp, subOp.getType(), match->lhs, match->rhs, match->acc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4 integer MAC matching helpers
//===----------------------------------------------------------------------===//

/// True iff `ty` is exactly `vector<N x iBits>`.
static bool isVecOfIntN(Type ty, int64_t n, unsigned bits) {
  auto vecTy = dyn_cast<VectorType>(ty);
  return vecTy && !vecTy.isScalable() && vecTy.getRank() == 1 &&
         vecTy.getShape()[0] == n && vecTy.getElementType().isInteger(bits);
}

/// Return the narrow source of an extension to vector<32xi32> supported by
/// the signed i16 VMAC, or nullptr. Signed i16 is consumed directly; signed
/// and unsigned i8 fit in signed i16 after unpacking with their own signedness.
/// Unsigned i16 is rejected because it does not fit in signed i16.
static Value matchMacOperandSource(Value v) {
  Operation *extOp = v.getDefiningOp();
  if (!extOp || !isa<arith::ExtSIOp, arith::ExtUIOp>(extOp) ||
      !isVecOfIntN(v.getType(), 32, 32))
    return nullptr;
  Value src = extOp->getOperand(0);
  if (isVecOfIntN(src.getType(), 32, 8))
    return src;
  if (isa<arith::ExtSIOp>(extOp) && isVecOfIntN(src.getType(), 32, 16))
    return src;
  return nullptr;
}

/// Convert an extension accepted by matchMacOperandSource to a VMAC operand:
/// pass through i16, or unpack i8 to i16 with the extension's signedness.
static Value widenMacOperandToI16(OpBuilder &b, Location loc, Value ext) {
  Value src = matchMacOperandSource(ext);
  assert(src && "not a MAC operand extension");
  if (isVecOfIntN(src.getType(), 32, 16))
    return src;
  bool isSigned = isa<arith::ExtSIOp>(ext.getDefiningOp());
  auto i16Ty = VectorType::get({32}, b.getI16Type());
  return aievec::AIE4UnpackI8I16Op::create(b, loc, i16Ty, src,
                                           b.getBoolAttr(isSigned));
}

/// Match a vector<32xi32> product of supported extensions, binding `a` and
/// `b` to the extensions rather than their narrow sources.
static bool matchWidenedI16Product(Value v, Value &a, Value &b) {
  auto mulOp = v.getDefiningOp<arith::MulIOp>();
  if (!mulOp || !isVecOfIntN(mulOp.getType(), 32, 32))
    return false;
  if (!matchMacOperandSource(mulOp.getLhs()) ||
      !matchMacOperandSource(mulOp.getRhs()))
    return false;
  a = mulOp.getLhs();
  b = mulOp.getRhs();
  return true;
}

struct IntMacMatch {
  arith::MulIOp product;
  Value acc;
  Value lhs;
  Value rhs;
};

/// Select the widened product fused into an integer MAC. When both operands
/// are products, prefer the RHS; the LHS must still be materialized as the
/// accumulator. Legality and the rewrites must agree on this choice.
static std::optional<IntMacMatch> matchIntMacChain(arith::AddIOp addOp) {
  if (!isVecOfIntN(addOp.getType(), 32, 32))
    return std::nullopt;
  auto tryOperand = [&](Value product,
                        Value acc) -> std::optional<IntMacMatch> {
    Value a, b;
    if (!matchWidenedI16Product(product, a, b))
      return std::nullopt;
    return IntMacMatch{product.getDefiningOp<arith::MulIOp>(), acc, a, b};
  };
  if (auto match = tryOperand(addOp.getRhs(), addOp.getLhs()))
    return match;
  return tryOperand(addOp.getLhs(), addOp.getRhs());
}

/// Defer a product only when every use disappears through MAC fusion.
/// A product used as the accumulator (including addi %p, %p) or outside
/// fusible MACs needs native multiply lowering. Dialect conversion does
/// not revisit operations it has already accepted as legal.
static bool feedsFusableIntMacChain(arith::MulIOp mulOp) {
  if (mulOp.getResult().use_empty())
    return false;
  return llvm::all_of(mulOp.getResult().getUsers(), [&](Operation *user) {
    auto addOp = dyn_cast<arith::AddIOp>(user);
    if (!addOp)
      return false;
    auto match = matchIntMacChain(addOp);
    return match && match->product == mulOp && match->acc != mulOp.getResult();
  });
}

// Materialize a surviving widened product with a zero-accumulator VMAC.
// A signed i16 product fits in i32, so saturation cannot change the result.
// This also handles products whose initial add-to-zero was canonicalized away.
struct ConvertIntMulToAIE4MacPattern : OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value a, b;
    if (!matchWidenedI16Product(mulOp.getResult(), a, b))
      return failure();
    Location loc = mulOp.getLoc();
    a = widenMacOperandToI16(rewriter, loc, a);
    b = widenMacOperandToI16(rewriter, loc, b);
    Value zero = arith::ConstantOp::create(
        rewriter, loc, mulOp.getType(), rewriter.getZeroAttr(mulOp.getType()));
    rewriter.replaceOpWithNewOp<aievec::AIE4MacEwiseI16I32Op>(
        mulOp, mulOp.getType(), a, b, zero);
    return success();
  }
};

// arith.addi(acc, arith.muli(arith.ext{si,ui} a, arith.ext{si,ui} b))
//     -> aievec.aie4.mac_ewise_i16_i32 a, b, acc
//
// Fuse the widening, multiply, and add into the native MAC. i8 operands
// (signed or unsigned) are unpacked to i16 first.
struct ConvertIntMacToAIE4MacPattern : OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto match = matchIntMacChain(addOp);
    if (!match)
      return failure();

    Location loc = addOp.getLoc();
    Value a = widenMacOperandToI16(rewriter, loc, match->lhs);
    Value b = widenMacOperandToI16(rewriter, loc, match->rhs);
    auto macOp = aievec::AIE4MacEwiseI16I32Op::create(
        rewriter, loc, addOp.getType(), a, b, match->acc);
    rewriter.replaceOp(addOp, macOp.getResult());
    return success();
  }
};

// Preserve extensions used exclusively by supported widened products so
// native multiply lowering or MAC fusion can consume them. Independent uses
// need standalone widening.
static bool feedsAIE4WidenedProduct(Operation *extOp) {
  Value res = extOp->getResult(0);
  return !res.use_empty() && llvm::all_of(res.getUsers(), [](Operation *user) {
    auto mulOp = dyn_cast<arith::MulIOp>(user);
    Value a, b;
    return mulOp && matchWidenedI16Product(mulOp.getResult(), a, b);
  });
}

// Lower vector<32xi16> -> vector<32xi32> extensions to exact, shift-zero UPS.
template <typename SrcOpTy>
struct ConvertExtToAIE4UPSPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  static bool matches(SrcOpTy extOp) {
    auto srcTy = dyn_cast<VectorType>(extOp.getIn().getType());
    auto dstTy = dyn_cast<VectorType>(extOp.getOut().getType());
    if (!srcTy || !dstTy)
      return false;
    if (!isVecOfIntN(srcTy, 32, 16) || !isVecOfIntN(dstTy, 32, 32))
      return false;
    if constexpr (std::is_same_v<SrcOpTy, arith::ExtSIOp>) {
      if (feedsAIE4WidenedProduct(extOp))
        return false;
    }
    return true;
  }

  LogicalResult
  matchAndRewrite(SrcOpTy extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(extOp))
      return failure();

    constexpr bool isSigned = std::is_same_v<SrcOpTy, arith::ExtSIOp>;
    auto upsOp = aievec::AIE4UPSI16I32Op::create(
        rewriter, extOp.getLoc(), extOp.getType(), adaptor.getIn(),
        rewriter.getI32IntegerAttr(0), rewriter.getBoolAttr(isSigned));
    rewriter.replaceOp(extOp, upsOp.getResult());
    return success();
  }
};

using ConvertExtSIToAIE4UPSPattern = ConvertExtToAIE4UPSPattern<arith::ExtSIOp>;
using ConvertExtUIToAIE4UPSPattern = ConvertExtToAIE4UPSPattern<arith::ExtUIOp>;

// `arith.extsi`/`arith.extui` vector<32xi8> -> vector<32xi16>
//     -> aievec.aie4.unpack_i8_i16
// `arith.extsi`/`arith.extui` vector<32xi8> -> vector<32xi32>
//     -> aievec.aie4.unpack_i8_i16 + aievec.aie4.ups_i16_i32 {shift = 0}
//
// Both are exact. Preserve i8 -> i32 extensions used exclusively by supported
// widened products for native multiply lowering or MAC fusion.
template <typename SrcOpTy>
struct ConvertExtI8ToAIE4UnpackPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  static bool matches(SrcOpTy extOp) {
    if (!isVecOfIntN(extOp.getIn().getType(), 32, 8))
      return false;
    Type dstTy = extOp.getOut().getType();
    if (isVecOfIntN(dstTy, 32, 16))
      return true;
    if (!isVecOfIntN(dstTy, 32, 32))
      return false;
    return !feedsAIE4WidenedProduct(extOp);
  }

  LogicalResult
  matchAndRewrite(SrcOpTy extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(extOp))
      return failure();

    constexpr bool isSigned = std::is_same_v<SrcOpTy, arith::ExtSIOp>;
    Location loc = extOp.getLoc();
    auto i16Ty = VectorType::get({32}, rewriter.getI16Type());
    Value wide = aievec::AIE4UnpackI8I16Op::create(
        rewriter, loc, i16Ty, adaptor.getIn(), rewriter.getBoolAttr(isSigned));
    if (isVecOfIntN(extOp.getType(), 32, 32))
      wide = aievec::AIE4UPSI16I32Op::create(
          rewriter, loc, extOp.getType(), wide, rewriter.getI32IntegerAttr(0),
          rewriter.getBoolAttr(isSigned));
    rewriter.replaceOp(extOp, wide);
    return success();
  }
};

using ConvertExtSIToAIE4UnpackPattern =
    ConvertExtI8ToAIE4UnpackPattern<arith::ExtSIOp>;
using ConvertExtUIToAIE4UnpackPattern =
    ConvertExtI8ToAIE4UnpackPattern<arith::ExtUIOp>;

//===----------------------------------------------------------------------===//
// AIE4 bf16 widening
//===----------------------------------------------------------------------===//

struct ConvertExtFToAIE4ConvPattern : OpConversionPattern<arith::ExtFOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool matches(arith::ExtFOp extOp) {
    auto srcTy = dyn_cast<VectorType>(extOp.getIn().getType());
    auto dstTy = dyn_cast<VectorType>(extOp.getOut().getType());
    if (!srcTy || !dstTy || srcTy.isScalable() || dstTy.isScalable())
      return false;
    return srcTy.getRank() == 1 && dstTy.getRank() == 1 &&
           srcTy.getShape()[0] == 32 && dstTy.getShape()[0] == 32 &&
           srcTy.getElementType().isBF16() && dstTy.getElementType().isF32();
  }

  LogicalResult
  matchAndRewrite(arith::ExtFOp extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(extOp))
      return failure();
    rewriter.replaceOpWithNewOp<aievec::AIE4ConvBF16F32Op>(
        extOp, extOp.getType(), adaptor.getIn());
    return success();
  }
};

// arith.fptosi / arith.fptoui on <32xf32> -> <32xi32> becomes the native AIE4
// conversion. The generic alternative is the scalarize pattern, which needs 32
// extract/convert/insert triples for what the part does in one instruction.
//
// Both arith ops truncate toward zero. The conversion rounds by the ambient
// crRnd, which AIECoreToStandard establishes; see the note there for why the
// mode is set at core entry.
template <typename FPToIntOp>
struct ConvertFPToIntToAIE4ConvPattern : OpConversionPattern<FPToIntOp> {
  using OpConversionPattern<FPToIntOp>::OpConversionPattern;
  using OpAdaptor = typename FPToIntOp::Adaptor;

  static bool matches(FPToIntOp op) {
    auto srcTy = dyn_cast<VectorType>(op.getIn().getType());
    auto dstTy = dyn_cast<VectorType>(op.getOut().getType());
    if (!srcTy || !dstTy || srcTy.isScalable() || dstTy.isScalable())
      return false;
    return srcTy.getRank() == 1 && dstTy.getRank() == 1 &&
           srcTy.getShape()[0] == 32 && dstTy.getShape()[0] == 32 &&
           srcTy.getElementType().isF32() &&
           dstTy.getElementType().isSignlessInteger(32);
  }

  LogicalResult
  matchAndRewrite(FPToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(op))
      return failure();
    rewriter.replaceOpWithNewOp<aievec::AIE4ConvF32I32Op>(
        op, op.getType(), adaptor.getIn(),
        /*isSigned=*/
        rewriter.getBoolAttr(std::is_same_v<FPToIntOp, arith::FPToSIOp>));
    return success();
  }
};

// AIE4 has no bf16 add or subtract, so these run in the f32 unit: widen both
// operands, operate in f32, truncate. f32 carries more than twice bf16's
// precision, so rounding the f32 result to nearest even gives the correctly
// rounded bf16 result. The truncf stays generic here, and AIEVecToLLVM lowers
// it to the native narrowing conversion.
template <typename SrcOpTy>
struct ConvertBF16ArithToAIE4F32Pattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  static bool matches(SrcOpTy op) {
    auto vecTy = dyn_cast<VectorType>(op.getType());
    return vecTy && !vecTy.isScalable() && vecTy.getRank() == 1 &&
           vecTy.getShape()[0] == 32 && vecTy.getElementType().isBF16();
  }

  LogicalResult
  matchAndRewrite(SrcOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(op))
      return failure();
    Location loc = op.getLoc();
    auto f32Ty = VectorType::get({32}, rewriter.getF32Type());
    Value lhs = arith::ExtFOp::create(rewriter, loc, f32Ty, adaptor.getLhs());
    Value rhs = arith::ExtFOp::create(rewriter, loc, f32Ty, adaptor.getRhs());
    Value wide = SrcOpTy::create(rewriter, loc, lhs, rhs, op.getFastmathAttr());
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, op.getType(), wide);
    return success();
  }
};

using ConvertFPToSIToAIE4ConvPattern =
    ConvertFPToIntToAIE4ConvPattern<arith::FPToSIOp>;
using ConvertFPToUIToAIE4ConvPattern =
    ConvertFPToIntToAIE4ConvPattern<arith::FPToUIOp>;

using ConvertBF16AddFToAIE4F32Pattern =
    ConvertBF16ArithToAIE4F32Pattern<arith::AddFOp>;
using ConvertBF16SubFToAIE4F32Pattern =
    ConvertBF16ArithToAIE4F32Pattern<arith::SubFOp>;

// A bf16 vector.fma rounds once. Computing a * b + c in f32 and then
// narrowing rounds twice, and the f32 rounding can land exactly on a bf16
// midpoint that the exact result is not on. Instead, round the f32 sum to
// odd: an inexact result is moved to the f32 value with an odd last bit on
// the exact result's side. f32 has at least two more bits than bf16, so
// narrowing a round-to-odd f32 to nearest even rounds correctly.
//
// a * b is exact in f32. The sum and its exact error come from TwoSum, which
// needs the f32 adds to round to nearest even.
struct ConvertBF16FMAToAIE4F32Pattern : OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool matches(vector::FMAOp op) {
    VectorType vecTy = op.getVectorType();
    return !vecTy.isScalable() && vecTy.getRank() == 1 &&
           vecTy.getShape()[0] == 32 && vecTy.getElementType().isBF16();
  }

  LogicalResult
  matchAndRewrite(vector::FMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(op))
      return failure();
    Location loc = op.getLoc();
    auto f32Ty = VectorType::get({32}, rewriter.getF32Type());
    auto i32Ty = VectorType::get({32}, rewriter.getI32Type());
    auto widen = [&](Value v) {
      return arith::ExtFOp::create(rewriter, loc, f32Ty, v).getResult();
    };
    auto i32Splat = [&](int32_t v) {
      return arith::ConstantOp::create(
                 rewriter, loc, i32Ty,
                 DenseElementsAttr::get(i32Ty, rewriter.getI32IntegerAttr(v)))
          .getResult();
    };

    Value p = arith::MulFOp::create(rewriter, loc, widen(adaptor.getLhs()),
                                    widen(adaptor.getRhs()));
    Value c = widen(adaptor.getAcc());
    Value s = arith::AddFOp::create(rewriter, loc, p, c);
    // TwoSum: e = (p + c) - s exactly.
    Value bv = arith::SubFOp::create(rewriter, loc, s, p);
    Value av = arith::SubFOp::create(rewriter, loc, s, bv);
    Value br = arith::SubFOp::create(rewriter, loc, c, bv);
    Value ar = arith::SubFOp::create(rewriter, loc, p, av);
    Value e = arith::AddFOp::create(rewriter, loc, ar, br);

    // The sum is inexact when e is finite and nonzero. When s overflows, e is
    // NaN and s is kept. When s is further from zero than the exact result,
    // step its magnitude down one f32 ulp before setting the last bit.
    Value sBits = arith::BitcastOp::create(rewriter, loc, i32Ty, s);
    Value eBits = arith::BitcastOp::create(rewriter, loc, i32Ty, e);
    Value eMag =
        arith::AndIOp::create(rewriter, loc, eBits, i32Splat(0x7fffffff));
    Value inexact = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult,
        arith::SubIOp::create(rewriter, loc, eMag, i32Splat(1)),
        i32Splat(0x7f7fffff));
    Value signsDiffer = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt,
        arith::XOrIOp::create(rewriter, loc, sBits, eBits), i32Splat(0));
    Value down = arith::SelectOp::create(rewriter, loc, signsDiffer,
                                         i32Splat(1), i32Splat(0));
    Value odd = arith::OrIOp::create(
        rewriter, loc, arith::SubIOp::create(rewriter, loc, sBits, down),
        i32Splat(1));
    Value rBits = arith::SelectOp::create(rewriter, loc, inexact, odd, sBits);
    Value r = arith::BitcastOp::create(rewriter, loc, f32Ty, rBits);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, op.getType(), r);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4 integer min/max/compare/select
//===----------------------------------------------------------------------===//
//
// Match the fixed signless integer shapes supported by the native operations.
// Other types retain generic lowering.
static bool isAIE4VAddShape(VectorType vecTy) {
  if (vecTy.isScalable() || vecTy.getRank() != 1 ||
      !vecTy.getElementType().isSignlessInteger())
    return false;
  unsigned elWidth = vecTy.getElementType().getIntOrFloatBitWidth();
  int64_t lanes = vecTy.getShape()[0];
  return (elWidth == 8 && lanes == 64) || (elWidth == 16 && lanes == 32) ||
         (elWidth == 16 && lanes == 64) || (elWidth == 32 && lanes == 16) ||
         (elWidth == 32 && lanes == 32);
}

// Match the mask widths enforced by AIE4CmpOp and AIE4SelOp verifiers.
static Type getAIE4MaskCarrierType(MLIRContext *ctx, VectorType vecTy) {
  unsigned elWidth = vecTy.getElementType().getIntOrFloatBitWidth();
  int64_t lanes = vecTy.getShape()[0];
  unsigned bits =
      (elWidth == 8 && lanes == 64) || (elWidth == 16 && lanes == 64) ? 64 : 32;
  return IntegerType::get(ctx, bits, IntegerType::Unsigned);
}

// Reinterpret the packed mask bits as Boolean lanes. The i32 carrier for
// vector<16xi32> uses only its low 16 bits.
static Value unpackAIE4Mask(Location loc, Value mask, VectorType resultType,
                            ConversionPatternRewriter &rewriter) {
  unsigned bits = cast<IntegerType>(mask.getType()).getWidth();
  auto scalarType = rewriter.getIntegerType(bits);
  Value scalar =
      UnrealizedConversionCastOp::create(rewriter, loc, scalarType, mask)
          .getResult(0);
  auto laneBitsType = rewriter.getIntegerType(resultType.getNumElements());
  if (laneBitsType != scalarType)
    scalar = arith::TruncIOp::create(rewriter, loc, laneBitsType, scalar);
  Value singleton = vector::BroadcastOp::create(
      rewriter, loc, VectorType::get({1}, laneBitsType), scalar);
  return vector::BitCastOp::create(rewriter, loc, resultType, singleton);
}

// Pack lane i into bit i. Sign-extending a scalar condition selects the
// same operand in every lane; vector masks zero their unused carrier bits.
static Value packAIE4Mask(Location loc, Value condition, Type carrierType,
                          int64_t lanes, ConversionPatternRewriter &rewriter) {
  auto scalarType =
      rewriter.getIntegerType(cast<IntegerType>(carrierType).getWidth());
  Value scalar;
  if (condition.getType().isInteger(1)) {
    scalar = arith::ExtSIOp::create(rewriter, loc, scalarType, condition);
  } else {
    auto laneBitsType = rewriter.getIntegerType(lanes);
    Value singleton = vector::BitCastOp::create(
        rewriter, loc, VectorType::get({1}, laneBitsType), condition);
    scalar = vector::ExtractOp::create(rewriter, loc, singleton,
                                       ArrayRef<int64_t>{0});
    if (laneBitsType != scalarType)
      scalar = arith::ExtUIOp::create(rewriter, loc, scalarType, scalar);
  }
  if (scalar.getType() == carrierType)
    return scalar;
  // Signedness changes no bits and disappears during LLVM type conversion.
  return UnrealizedConversionCastOp::create(rewriter, loc, carrierType, scalar)
      .getResult(0);
}

static bool isAIE4MaskExtract(vector::ExtractOp op) {
  VectorType sourceType = op.getSourceVectorType();
  if (sourceType.isScalable() || sourceType.getRank() != 1 ||
      !sourceType.getElementType().isInteger(1) || !op.getType().isInteger(1))
    return false;
  int64_t lanes = sourceType.getNumElements();
  return lanes == 16 || lanes == 32 || lanes == 64;
}

// Read mask bits through a scalar carrier regardless of their producer.
// Peano cannot extract Boolean lanes from a bitcast vector<2xi32> mask.
struct LowerAIE4MaskExtractOp : OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  // Recursive extraction moves toward the Boolean expression leaves.
  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult
  matchAndRewrite(vector::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isAIE4MaskExtract(op))
      return failure();

    Location loc = op.getLoc();
    // Apply Boolean logic to the extracted bits. Peano cannot lower bitwise
    // operations on the vector<2xi32> carrier used by 64-bit masks.
    Operation *logic = adaptor.getSource().getDefiningOp();
    if (isa_and_nonnull<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(logic)) {
      Value lhs = vector::ExtractOp::create(rewriter, loc, logic->getOperand(0),
                                            adaptor.getDynamicPosition(),
                                            op.getStaticPosition());
      Value rhs = vector::ExtractOp::create(rewriter, loc, logic->getOperand(1),
                                            adaptor.getDynamicPosition(),
                                            op.getStaticPosition());
      if (isa<arith::AndIOp>(logic))
        rewriter.replaceOpWithNewOp<arith::AndIOp>(op, lhs, rhs);
      else if (isa<arith::OrIOp>(logic))
        rewriter.replaceOpWithNewOp<arith::OrIOp>(op, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, lhs, rhs);
      return success();
    }

    int64_t lanes = op.getSourceVectorType().getNumElements();
    auto scalarType = rewriter.getI32Type();
    Value index;
    if (adaptor.getDynamicPosition().empty())
      index = arith::ConstantIntOp::create(rewriter, loc, scalarType,
                                           op.getStaticPosition()[0]);
    else
      index = arith::IndexCastUIOp::create(rewriter, loc, scalarType,
                                           adaptor.getDynamicPosition()[0]);

    Value mask;
    if (lanes == 64) {
      // Extract a scalar word before bitwise logic can recombine into an
      // unsupported operation on the two-word vector carrier.
      Value words = vector::BitCastOp::create(
          rewriter, loc, VectorType::get({2}, scalarType), adaptor.getSource());
      Value low =
          vector::ExtractOp::create(rewriter, loc, words, ArrayRef<int64_t>{0});
      Value high =
          vector::ExtractOp::create(rewriter, loc, words, ArrayRef<int64_t>{1});
      Value wordBits = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
      Value useHigh = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::uge, index, wordBits);
      mask = arith::SelectOp::create(rewriter, loc, useHigh, high, low);
      Value bitMask = arith::ConstantIntOp::create(rewriter, loc, 31, 32);
      index = arith::AndIOp::create(rewriter, loc, index, bitMask);
    } else {
      mask =
          packAIE4Mask(loc, adaptor.getSource(), scalarType, lanes, rewriter);
    }
    Value shifted = arith::ShRUIOp::create(rewriter, loc, mask, index);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, rewriter.getI1Type(),
                                                 shifted);
    return success();
  }
};

// Supported float shapes: fixed 512- or 1024-bit BF16, FP16, and FP32 vectors.
// Their integer bitcasts use the same compare/select mask carriers.
static bool isAIE4VFCmpShape(VectorType vecTy) {
  if (vecTy.isScalable() || vecTy.getRank() != 1)
    return false;
  Type elTy = vecTy.getElementType();
  int64_t lanes = vecTy.getShape()[0];
  return (elTy.isBF16() && (lanes == 32 || lanes == 64)) ||
         (elTy.isF16() && (lanes == 32 || lanes == 64)) ||
         (elTy.isF32() && (lanes == 16 || lanes == 32));
}

// Preserve each lane's bits in a signless integer vector for native integer
// comparisons and selects. The caller must check isAIE4VFCmpShape first.
static Value bitcastToAIE4SelIntType(ConversionPatternRewriter &rewriter,
                                     Location loc, Value v) {
  auto vecTy = cast<VectorType>(v.getType());
  Type intElemTy =
      rewriter.getIntegerType(vecTy.getElementType().getIntOrFloatBitWidth());
  Type intVecTy = VectorType::get(vecTy.getShape(), intElemTy);
  return vector::BitCastOp::create(rewriter, loc, intVecTy, v);
}

template <typename SrcOpTy>
struct LowerAIE4MinMaxOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType || !isAIE4VAddShape(resultType))
      return failure();

    constexpr bool isSigned = std::is_same_v<SrcOpTy, arith::MinSIOp> ||
                              std::is_same_v<SrcOpTy, arith::MaxSIOp>;
    constexpr bool isMin = std::is_same_v<SrcOpTy, arith::MinSIOp> ||
                           std::is_same_v<SrcOpTy, arith::MinUIOp>;
    auto kind =
        isMin ? aievec::AIE4MinMaxKind::Min : aievec::AIE4MinMaxKind::Max;

    rewriter.replaceOpWithNewOp<aievec::AIE4MinMaxOp>(
        srcOp, resultType, adaptor.getLhs(), adaptor.getRhs(), kind, isSigned);
    return success();
  }
};

using LowerAIE4MinSIOp = LowerAIE4MinMaxOp<arith::MinSIOp>;
using LowerAIE4MaxSIOp = LowerAIE4MinMaxOp<arith::MaxSIOp>;
using LowerAIE4MinUIOp = LowerAIE4MinMaxOp<arith::MinUIOp>;
using LowerAIE4MaxUIOp = LowerAIE4MinMaxOp<arith::MaxUIOp>;

// Ordered comparisons use native lt/ge predicates, swapping operands for
// gt/le. Equality predicates retain generic lowering.
struct LowerAIE4CmpIOp : OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<VectorType>(srcOp.getLhs().getType());
    if (!lhsType || !isAIE4VAddShape(lhsType))
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    bool isSigned;
    aievec::AIE4CmpPredicate predicate;
    switch (srcOp.getPredicate()) {
    case arith::CmpIPredicate::slt:
      isSigned = true;
      predicate = aievec::AIE4CmpPredicate::Lt;
      break;
    case arith::CmpIPredicate::sge:
      isSigned = true;
      predicate = aievec::AIE4CmpPredicate::Ge;
      break;
    case arith::CmpIPredicate::sgt:
      isSigned = true;
      predicate = aievec::AIE4CmpPredicate::Lt;
      std::swap(lhs, rhs);
      break;
    case arith::CmpIPredicate::sle:
      isSigned = true;
      predicate = aievec::AIE4CmpPredicate::Ge;
      std::swap(lhs, rhs);
      break;
    case arith::CmpIPredicate::ult:
      isSigned = false;
      predicate = aievec::AIE4CmpPredicate::Lt;
      break;
    case arith::CmpIPredicate::uge:
      isSigned = false;
      predicate = aievec::AIE4CmpPredicate::Ge;
      break;
    case arith::CmpIPredicate::ugt:
      isSigned = false;
      predicate = aievec::AIE4CmpPredicate::Lt;
      std::swap(lhs, rhs);
      break;
    case arith::CmpIPredicate::ule:
      isSigned = false;
      predicate = aievec::AIE4CmpPredicate::Ge;
      std::swap(lhs, rhs);
      break;
    default:
      // Equality predicates retain generic lowering.
      return failure();
    }

    Location loc = srcOp.getLoc();
    Type maskTy = getAIE4MaskCarrierType(srcOp.getContext(), lhsType);
    auto cmpOp = aievec::AIE4CmpOp::create(rewriter, loc, maskTy, lhs, rhs,
                                           predicate, isSigned);

    VectorType resultType = cast<VectorType>(srcOp.getResult().getType());
    rewriter.replaceOp(
        srcOp, unpackAIE4Mask(loc, cmpOp.getResult(), resultType, rewriter));
    return success();
  }
};

// Native select chooses rhs for a set mask bit, so the false value is lhs.
struct LowerAIE4SelectOp : OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();
    bool isFloatShape = isAIE4VFCmpShape(resultType);
    if (!isAIE4VAddShape(resultType) && !isFloatShape)
      return failure();

    Location loc = srcOp.getLoc();
    Type maskTy = getAIE4MaskCarrierType(srcOp.getContext(), resultType);
    Value mask = packAIE4Mask(loc, adaptor.getCondition(), maskTy,
                              resultType.getNumElements(), rewriter);

    if (!isFloatShape) {
      rewriter.replaceOpWithNewOp<aievec::AIE4SelOp>(
          srcOp, resultType, adaptor.getFalseValue(), adaptor.getTrueValue(),
          mask);
      return success();
    }

    // AIE4SelOp accepts integers; bitcast float lanes before and after
    // selection.
    Value falseInt =
        bitcastToAIE4SelIntType(rewriter, loc, adaptor.getFalseValue());
    Value trueInt =
        bitcastToAIE4SelIntType(rewriter, loc, adaptor.getTrueValue());
    Value selInt = aievec::AIE4SelOp::create(rewriter, loc, falseInt.getType(),
                                             falseInt, trueInt, mask);
    rewriter.replaceOpWithNewOp<vector::BitCastOp>(srcOp, resultType, selInt);
    return success();
  }
};

// Compare IEEE encodings with integer instructions so ambient crFPOrder cannot
// change source arithmetic. Masks use signless scalars for Boolean composition.
class AIE4FloatOrder {
public:
  AIE4FloatOrder(ConversionPatternRewriter &rewriter, Location loc, Value a,
                 Value b)
      : rewriter(rewriter), loc(loc), floatType(cast<VectorType>(a.getType())),
        maskType(getAIE4MaskCarrierType(rewriter.getContext(), floatType)),
        scalarType(rewriter.getIntegerType(maskType.getIntOrFloatBitWidth())) {
    lhs = bitcastToAIE4SelIntType(rewriter, loc, a);
    rhs = bitcastToAIE4SelIntType(rewriter, loc, b);
    unsigned width = floatType.getElementTypeBitWidth();
    Value signBit = splat(uint64_t(1) << (width - 1));
    Value magnitudeMask = splat((uint64_t(1) << (width - 1)) - 1);
    lhsMagnitude = arith::AndIOp::create(rewriter, loc, lhs, magnitudeMask);
    rhsMagnitude = arith::AndIOp::create(rewriter, loc, rhs, magnitudeMask);
    // Complement negative encodings; flip the sign bit on positive encodings.
    // Unsigned key order then places -0 immediately before +0.
    auto key = [&](Value bits) -> Value {
      Value negative = cmp(bits, splat(0), true);
      Value flipped = arith::XOrIOp::create(rewriter, loc, bits, signBit);
      Value inverted = arith::XOrIOp::create(rewriter, loc, bits, splat(-1));
      return select(flipped, inverted, negative);
    };
    lhsKey = key(lhs);
    rhsKey = key(rhs);
    uint64_t infinity = floatType.getElementType().isF32()   ? 0x7f800000
                        : floatType.getElementType().isF16() ? 0x7c00
                                                             : 0x7f80;
    Value firstNaN = splat(infinity + 1);
    // Clearing the sign puts every NaN above infinity in unsigned order.
    lhsNumber = cmp(lhsMagnitude, firstNaN);
    rhsNumber = cmp(rhsMagnitude, firstNaN);
  }

  Value constant(int64_t value) {
    return arith::ConstantOp::create(
        rewriter, loc, scalarType, rewriter.getIntegerAttr(scalarType, value));
  }
  Value bitNot(Value value) {
    return arith::XOrIOp::create(rewriter, loc, value, constant(-1));
  }
  Value bitAnd(Value a, Value b) {
    return arith::AndIOp::create(rewriter, loc, a, b);
  }
  Value bitOr(Value a, Value b) {
    return arith::OrIOp::create(rewriter, loc, a, b);
  }
  Value ordered() { return bitAnd(lhsNumber, rhsNumber); }
  Value less(bool swap = false) {
    Value lt = swap ? cmp(rhsKey, lhsKey) : cmp(lhsKey, rhsKey);
    // IEEE comparisons treat -0 and +0 as equal, unlike the sortable keys.
    Value magnitudes =
        arith::OrIOp::create(rewriter, loc, lhsMagnitude, rhsMagnitude);
    Value nonzero = bitNot(cmp(magnitudes, splat(1)));
    return bitAnd(bitAnd(lt, nonzero), ordered());
  }
  Value minimumMaximum(bool isMin, bool numberSemantics) {
    Value lt = cmp(lhsKey, rhsKey);
    Value result = isMin ? select(rhs, lhs, lt) : select(lhs, rhs, lt);
    // minnum/maxnum prefer the numeric operand; minimum/maximum propagate NaN.
    if (numberSemantics) {
      result = select(lhs, result, rhsNumber);
      result = select(rhs, result, lhsNumber);
    } else {
      result = select(rhs, result, rhsNumber);
      result = select(lhs, result, lhsNumber);
    }
    return vector::BitCastOp::create(rewriter, loc, floatType, result);
  }

private:
  Value splat(int64_t value) {
    auto type = cast<VectorType>(lhs.getType());
    return arith::ConstantOp::create(
        rewriter, loc, type,
        DenseElementsAttr::get(
            type, rewriter.getIntegerAttr(type.getElementType(), value)));
  }
  Value cmp(Value a, Value b, bool isSigned = false) {
    Value mask = aievec::AIE4CmpOp::create(
        rewriter, loc, maskType, a, b, aievec::AIE4CmpPredicate::Lt, isSigned);
    return UnrealizedConversionCastOp::create(rewriter, loc, scalarType, mask)
        .getResult(0);
  }
  Value select(Value a, Value b, Value mask) {
    // A set mask bit selects the second operand.
    Value carrier =
        UnrealizedConversionCastOp::create(rewriter, loc, maskType, mask)
            .getResult(0);
    return aievec::AIE4SelOp::create(rewriter, loc, a.getType(), a, b, carrier);
  }

  ConversionPatternRewriter &rewriter;
  Location loc;
  VectorType floatType;
  Type maskType, scalarType;
  Value lhs, rhs, lhsMagnitude, rhsMagnitude, lhsKey, rhsKey;
  Value lhsNumber, rhsNumber;
};

template <typename SrcOpTy>
struct LowerAIE4FloatMinMaxOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType || !isAIE4VFCmpShape(resultType))
      return failure();
    constexpr bool isMin = std::is_same_v<SrcOpTy, arith::MinimumFOp> ||
                           std::is_same_v<SrcOpTy, arith::MinNumFOp>;
    constexpr bool numberSemantics =
        std::is_same_v<SrcOpTy, arith::MinNumFOp> ||
        std::is_same_v<SrcOpTy, arith::MaxNumFOp>;
    AIE4FloatOrder order(rewriter, srcOp.getLoc(), adaptor.getLhs(),
                         adaptor.getRhs());
    rewriter.replaceOp(srcOp, order.minimumMaximum(isMin, numberSemantics));
    return success();
  }
};

using LowerAIE4MinimumFOp = LowerAIE4FloatMinMaxOp<arith::MinimumFOp>;
using LowerAIE4MaximumFOp = LowerAIE4FloatMinMaxOp<arith::MaximumFOp>;
using LowerAIE4MinNumFOp = LowerAIE4FloatMinMaxOp<arith::MinNumFOp>;
using LowerAIE4MaxNumFOp = LowerAIE4FloatMinMaxOp<arith::MaxNumFOp>;

struct LowerAIE4CmpFOp : OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<VectorType>(srcOp.getLhs().getType());
    if (!lhsType || !isAIE4VFCmpShape(lhsType))
      return failure();
    Location loc = srcOp.getLoc();
    AIE4FloatOrder order(rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    Value result;
    switch (srcOp.getPredicate()) {
    case arith::CmpFPredicate::OLT:
      result = order.less();
      break;
    case arith::CmpFPredicate::OGT:
      result = order.less(true);
      break;
    case arith::CmpFPredicate::OLE:
      result = order.bitAnd(order.bitNot(order.less(true)), order.ordered());
      break;
    case arith::CmpFPredicate::OGE:
      result = order.bitAnd(order.bitNot(order.less()), order.ordered());
      break;
    case arith::CmpFPredicate::OEQ:
      result = order.bitAnd(
          order.bitNot(order.bitOr(order.less(), order.less(true))),
          order.ordered());
      break;
    case arith::CmpFPredicate::ONE:
      result = order.bitOr(order.less(), order.less(true));
      break;
    case arith::CmpFPredicate::ORD:
      result = order.ordered();
      break;
    case arith::CmpFPredicate::UNO:
      result = order.bitNot(order.ordered());
      break;
    case arith::CmpFPredicate::ULT:
      result = order.bitOr(order.less(), order.bitNot(order.ordered()));
      break;
    case arith::CmpFPredicate::UGE:
      result = order.bitNot(order.less());
      break;
    case arith::CmpFPredicate::UGT:
      result = order.bitOr(order.less(true), order.bitNot(order.ordered()));
      break;
    case arith::CmpFPredicate::ULE:
      result = order.bitNot(order.less(true));
      break;
    case arith::CmpFPredicate::UEQ:
      result = order.bitNot(order.bitOr(order.less(), order.less(true)));
      break;
    case arith::CmpFPredicate::UNE:
      result = order.bitOr(order.bitOr(order.less(), order.less(true)),
                           order.bitNot(order.ordered()));
      break;
    case arith::CmpFPredicate::AlwaysFalse:
      result = order.constant(0);
      break;
    case arith::CmpFPredicate::AlwaysTrue:
      result = order.constant(-1);
      break;
    }
    auto resultType = cast<VectorType>(srcOp.getResult().getType());
    rewriter.replaceOp(srcOp,
                       unpackAIE4Mask(loc, result, resultType, rewriter));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4 vector reductions
//===----------------------------------------------------------------------===//
//
// Only fixed 32-lane signed min/max, floating minimum/maximum, and
// reassociable f32 add map to native reductions. Integer add wraps in MLIR
// but saturates on AIE4; ordered f32 add may round differently on AIE4.
//
// AIE4 has no bf16 reduction. bf16 minimum/maximum widen to f32 and reduce
// there: the result is one of the inputs, so narrowing it back is exact.
struct LowerAIE4VectorReductionOp : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Dialect conversion hides pattern-emitted errors; warnings retain the
    // specific refusal reason alongside its legalization error.
    auto vecTy = dyn_cast<VectorType>(srcOp.getVector().getType());
    if (!vecTy || vecTy.isScalable() || vecTy.getRank() != 1) {
      srcOp.emitWarning("AIE4: only fixed 1-D vector.reduction is supported");
      return failure();
    }

    Type elemTy = vecTy.getElementType();
    bool isBF16 = elemTy.isBF16();
    if (!elemTy.isInteger(32) && !elemTy.isF32() && !isBF16) {
      srcOp.emitWarning()
          << "AIE4: vector.reduction only supports i32/f32 elements, and bf16 "
             "minimum/maximum, not "
          << elemTy;
      return failure();
    }

    if (int64_t lanes = vecTy.getShape()[0]; lanes != 32) {
      srcOp.emitWarning()
          << "AIE4: vector.reduction requires a 1024-bit (32-lane) "
             "vector; no reduce instruction exists at "
          << (lanes * (isBF16 ? 32 : elemTy.getIntOrFloatBitWidth()))
          << "-bit width (in particular, there is no 512-bit reduce "
             "instruction at all)";
      return failure();
    }

    aievec::AIE4ReduceKind kind;
    switch (srcOp.getKind()) {
    case vector::CombiningKind::ADD:
      kind = aievec::AIE4ReduceKind::Add;
      break;
    case vector::CombiningKind::MINSI:
      kind = aievec::AIE4ReduceKind::Min;
      break;
    case vector::CombiningKind::MAXSI:
      kind = aievec::AIE4ReduceKind::Max;
      break;
    case vector::CombiningKind::MINIMUMF:
      kind = aievec::AIE4ReduceKind::Min;
      break;
    case vector::CombiningKind::MAXIMUMF:
      kind = aievec::AIE4ReduceKind::Max;
      break;
    case vector::CombiningKind::MINUI:
    case vector::CombiningKind::MAXUI:
      srcOp.emitWarning()
          << "AIE4 has no unsigned vector reduce instruction (the ISA "
             "defines the int32 reduce variant as signed-only); rewrite "
             "as a signed reduction or scalarize";
      return failure();
    case vector::CombiningKind::MINNUMF:
    case vector::CombiningKind::MAXNUMF:
      srcOp.emitWarning()
          << "AIE4's VMINR.f/VMAXR.f hard-wire IEEE minimum/maximum "
             "(NaN-propagating) semantics with no control-register "
             "escape, so minnumf/maxnumf cannot be lowered; use "
             "minimumf/maximumf if that NaN-propagation is acceptable";
      return failure();
    case vector::CombiningKind::MUL:
    case vector::CombiningKind::AND:
    case vector::CombiningKind::OR:
    case vector::CombiningKind::XOR:
      srcOp.emitWarning()
          << "AIE4 has no vector reduce instruction for this combining "
             "kind (only add/min/max are implemented in hardware)";
      return failure();
    }

    if (isBF16) {
      assert(kind != aievec::AIE4ReduceKind::Add &&
             "bf16 add reductions round and are left legal");
      Location loc = srcOp.getLoc();
      Type f32Ty = rewriter.getF32Type();
      Value src = arith::ExtFOp::create(
          rewriter, loc, VectorType::get({32}, f32Ty), adaptor.getVector(),
          arith::FastMathFlagsAttr{});
      Value acc = adaptor.getAcc();
      if (acc)
        acc = arith::ExtFOp::create(rewriter, loc, f32Ty, acc,
                                    arith::FastMathFlagsAttr{});
      Value reduced =
          aievec::AIE4ReduceOp::create(rewriter, loc, f32Ty, src, acc, kind);
      rewriter.replaceOpWithNewOp<arith::TruncFOp>(srcOp, elemTy, reduced);
      return success();
    }

    rewriter.replaceOpWithNewOp<aievec::AIE4ReduceOp>(
        srcOp, elemTy, adaptor.getVector(), adaptor.getAcc(), kind);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Shuffle / transpose
//===----------------------------------------------------------------------===//
//
// Lower supported transposes and interleaves to AIE4 shuffles. Other forms
// remain legal for generic lowering.

// Flatten square transposes to one 512-bit vector, then restore the shape.
struct LowerVectorTransposeOpToAIE4ShuffleOpPattern
    : OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  // Share matching with the legality predicate so every illegal op has a
  // matching rewrite.
  static std::optional<aievec::AIE4ShuffleMode>
  matchMode(vector::TransposeOp transpOp) {
    auto resTy = transpOp.getResultVectorType();
    auto resShape = resTy.getShape();
    if (resTy.isScalable() || resShape.size() != 2)
      return std::nullopt;

    Type elemTy = resTy.getElementType();
    if (!elemTy.isSignlessInteger(8) && !elemTy.isSignlessInteger(32) &&
        !elemTy.isF32())
      return std::nullopt;

    auto elemTyBitWidth = resTy.getElementTypeBitWidth();
    auto vBitWidth = resShape[0] * resShape[1] * elemTyBitWidth;
    if (vBitWidth != 512)
      return std::nullopt;

    // These modes implement square transposes with permutation [1, 0].
    ArrayRef<int64_t> perm = transpOp.getPermutation();
    if (perm.size() != 2 || perm[0] != 1 || perm[1] != 0)
      return std::nullopt;
    if (resShape[0] != resShape[1])
      return std::nullopt;

    if (elemTyBitWidth == 32 && resShape[0] == 4)
      return aievec::AIE4ShuffleMode::T32_4X4;
    if (elemTyBitWidth == 8 && resShape[0] == 8)
      return aievec::AIE4ShuffleMode::T8_8X8;
    return std::nullopt;
  }

  LogicalResult
  matchAndRewrite(vector::TransposeOp transpOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto modeOr = matchMode(transpOp);
    if (!modeOr)
      return failure();

    auto resTy = transpOp.getResultVectorType();
    auto elemTyBitWidth = resTy.getElementTypeBitWidth();
    auto loc = transpOp.getLoc();
    auto flatVecTy =
        VectorType::get({512 / elemTyBitWidth}, resTy.getElementType());
    auto flatInput = vector::ShapeCastOp::create(rewriter, loc, flatVecTy,
                                                 adaptor.getVector());
    auto shuffOp = aievec::AIE4ShuffleOp::create(
        rewriter, loc, flatVecTy, flatInput, /*rhs=*/nullptr, *modeOr);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(transpOp, resTy, shuffOp);
    return success();
  }
};

// Match interleaves of lanes 0-7 (lo) or 8-15 (hi) of two 16-lane i32
// vectors. The mask alternates v1[base+i] and v2[base+i] for i in [0, 8).
struct LowerVectorShuffleOpToAIE4ShuffleOpPattern
    : OpConversionPattern<vector::ShuffleOp> {
  using OpConversionPattern::OpConversionPattern;

  static SmallVector<int64_t, 16> interleaveMask(int64_t base) {
    SmallVector<int64_t, 16> m;
    for (int64_t i = 0; i < 8; ++i) {
      m.push_back(base + i);
      m.push_back(base + i + 16);
    }
    return m;
  }

  // Use the same match for rewriting and dynamic legality.
  static std::optional<aievec::AIE4ShuffleMode>
  matchMode(vector::ShuffleOp shuffleOp) {
    auto v1Ty = dyn_cast<VectorType>(shuffleOp.getV1().getType());
    auto v2Ty = dyn_cast<VectorType>(shuffleOp.getV2().getType());
    if (!v1Ty || !v2Ty || v1Ty.isScalable() || v2Ty.isScalable() ||
        v1Ty.getRank() != 1 || v2Ty.getRank() != 1)
      return std::nullopt;
    if (v1Ty.getShape()[0] != 16 || v2Ty.getShape()[0] != 16 ||
        !v1Ty.getElementType().isSignlessInteger(32) ||
        !v2Ty.getElementType().isSignlessInteger(32))
      return std::nullopt;

    ArrayRef<int64_t> mask = shuffleOp.getMask();
    if (mask.size() != 16)
      return std::nullopt;

    if (mask == ArrayRef<int64_t>(interleaveMask(0)))
      return aievec::AIE4ShuffleMode::T32_2X16_LO;
    if (mask == ArrayRef<int64_t>(interleaveMask(8)))
      return aievec::AIE4ShuffleMode::T32_2X16_HI;
    return std::nullopt;
  }

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto modeOr = matchMode(shuffleOp);
    if (!modeOr)
      return failure();
    rewriter.replaceOpWithNewOp<aievec::AIE4ShuffleOp>(
        shuffleOp, cast<VectorType>(shuffleOp.getV1().getType()),
        adaptor.getV1(), adaptor.getV2(), *modeOr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4 approximate exp2 / tanh
//===----------------------------------------------------------------------===//
//
// Approximation requires afn (also included in fast); nnan/ninf are not
// assumed. Peano cannot legalize strict exp2/tanh, so diagnose them here.
// Supported fixed 1-D f32/bf16 vectors split into 32-lane chunks and retain
// their type.

static constexpr int64_t kAIE4NlfLanes = 32;
// VEXP2 floors (x - 895) to a 2^-11 grid. For f32 results, the exact
// 895 + 2^-12 bias compensates that truncation; see the NPU3 accuracy test
// for measured error bounds under RNE vector addition.
static constexpr float kAIE4Exp2BiasF32 = 895.000244140625f;
// The bf16 form uses the vendor aie4_nlf.h bias (895 + 2^-10 in f32).
static constexpr float kAIE4Exp2BiasBF16 = 895.001f;

template <typename MathOpTy>
struct LowerAIE4ApproxMathOp : OpConversionPattern<MathOpTy> {
  using OpConversionPattern<MathOpTy>::OpConversionPattern;
  using OpAdaptor = typename MathOpTy::Adaptor;
  static constexpr bool isExp2 = std::is_same_v<MathOpTy, math::Exp2Op>;
  // exp has no instruction of its own: exp(x) = exp2(x * log2(e)), with the
  // product formed in f32 so that a bf16 input does not also round log2(e).
  static constexpr bool isExp = std::is_same_v<MathOpTy, math::ExpOp>;

  LogicalResult
  matchAndRewrite(MathOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Dialect conversion hides pattern-emitted errors; warnings retain the
    // specific refusal reason alongside its legalization error.
    StringRef opName = MathOpTy::getOperationName();
    if (!arith::bitEnumContainsAll(op.getFastmath(),
                                   arith::FastMathFlags::afn)) {
      op.emitWarning() << "AIE4: " << opName
                       << " requires fastmath<afn>; AIE4 has only the "
                          "approximate "
                       << (isExp2 || isExp ? "VEXP2" : "VTANH")
                       << " lowering and no strict lowering";
      return failure();
    }

    auto vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.isScalable() || vecTy.getRank() != 1) {
      op.emitWarning() << "AIE4: afn " << opName
                       << " is only supported on fixed 1-D vectors";
      return failure();
    }
    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isBF16()) {
      op.emitWarning() << "AIE4: afn " << opName
                       << " only supports f32/bf16 elements, not " << elemTy;
      return failure();
    }
    int64_t lanes = vecTy.getShape()[0];
    if (lanes % kAIE4NlfLanes != 0) {
      op.emitWarning() << "AIE4: afn " << opName
                       << " requires a multiple of 32 lanes, not " << lanes;
      return failure();
    }

    Location loc = op.getLoc();
    auto f32ChunkTy = VectorType::get({kAIE4NlfLanes}, rewriter.getF32Type());
    auto bf16ChunkTy = VectorType::get({kAIE4NlfLanes}, rewriter.getBF16Type());
    auto chunkTy = VectorType::get({kAIE4NlfLanes}, elemTy);

    Value bias;
    if constexpr (isExp2 || isExp) {
      float biasValue = elemTy.isF32() ? kAIE4Exp2BiasF32 : kAIE4Exp2BiasBF16;
      bias = arith::ConstantOp::create(
          rewriter, loc, f32ChunkTy,
          DenseElementsAttr::get(f32ChunkTy,
                                 rewriter.getF32FloatAttr(biasValue)));
    }

    auto lowerChunk = [&](Value chunk) -> Value {
      Value x = chunk;
      if (elemTy.isBF16())
        x = arith::ExtFOp::create(rewriter, loc, f32ChunkTy, x,
                                  arith::FastMathFlagsAttr{});
      if constexpr (isExp) {
        Value log2e = arith::ConstantOp::create(
            rewriter, loc, f32ChunkTy,
            DenseElementsAttr::get(
                f32ChunkTy, rewriter.getF32FloatAttr(llvm::numbers::log2ef)));
        x = arith::MulFOp::create(rewriter, loc, x, log2e);
      }
      if constexpr (isExp2 || isExp) {
        // Strict add: no fastmath flags, so it cannot be reassociated away.
        Value biased = arith::AddFOp::create(rewriter, loc, x, bias);
        return aievec::AIE4BiasedExp2Op::create(rewriter, loc, chunkTy, biased);
      } else {
        Value r = aievec::AIE4TanhOp::create(rewriter, loc, bf16ChunkTy, x);
        if (elemTy.isF32())
          r = arith::ExtFOp::create(rewriter, loc, f32ChunkTy, r,
                                    arith::FastMathFlagsAttr{});
        return r;
      }
    };

    // Chunking supports any multiple of 32. Peano may still reject wide
    // loads such as vector<128xf32> or vector<96xbf16> in the source.
    Value src = adaptor.getOperand();
    if (lanes == kAIE4NlfLanes) {
      rewriter.replaceOp(op, lowerChunk(src));
      return success();
    }

    Value result = ub::PoisonOp::create(rewriter, loc, vecTy);
    for (int64_t off = 0; off < lanes; off += kAIE4NlfLanes) {
      Value chunk = vector::ExtractStridedSliceOp::create(
          rewriter, loc, src, ArrayRef<int64_t>{off},
          ArrayRef<int64_t>{kAIE4NlfLanes}, ArrayRef<int64_t>{1});
      result = vector::InsertStridedSliceOp::create(
          rewriter, loc, lowerChunk(chunk), result, ArrayRef<int64_t>{off},
          ArrayRef<int64_t>{1});
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4 approximate scalar reciprocal
//===----------------------------------------------------------------------===//
//
// INV approximates 1/x within 1 ULP for normal inputs and results, and
// flushes denormals. Like exp2/tanh, it requires `fastmath<afn>`. Strict divf
// keeps its generic lowering, which Peano compiles to a __divsf3 call.
struct LowerAIE4ReciprocalOp : OpConversionPattern<arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool matches(arith::DivFOp op) {
    if (!op.getType().isF32() ||
        !arith::bitEnumContainsAll(op.getFastmath(), arith::FastMathFlags::afn))
      return false;
    FloatAttr one;
    return matchPattern(op.getLhs(), m_Constant(&one)) &&
           one.getValue().isExactlyValue(1.0);
  }

  LogicalResult
  matchAndRewrite(arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(op))
      return failure();
    rewriter.replaceOpWithNewOp<aievec::AIE4InvOp>(op, op.getType(),
                                                   adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIE4 dense matrix multiplication
//===----------------------------------------------------------------------===//

/// A vector.contract that computes exactly one aievec.aie4.matmul.
struct AIE4MatMulContractMatch {
  Value lhs; // narrow A, vector<M x K x elem>
  Value rhs; // narrow B tile, vector<K x 64 x elem>
  std::optional<bool> lhsSigned, rhsSigned; // set for integer formats only
};

/// Native dense MMAC formats: the narrow element type,
/// K, and the accumulator element type.
static std::optional<std::pair<int64_t, Type>>
getAIE4MatMulFormat(Type narrowElemTy) {
  MLIRContext *ctx = narrowElemTy.getContext();
  if (narrowElemTy.isBF16())
    return std::make_pair(int64_t(32), Type(Float32Type::get(ctx)));
  if (isa<Float8E4M3FNType>(narrowElemTy))
    return std::make_pair(int64_t(64), Type(Float32Type::get(ctx)));
  if (narrowElemTy.isSignlessInteger(8))
    return std::make_pair(int64_t(64), Type(IntegerType::get(ctx, 32)));
  if (narrowElemTy.isSignlessInteger(4))
    return std::make_pair(int64_t(128), Type(IntegerType::get(ctx, 32)));
  return std::nullopt;
}

/// Peel the widening op that produced a contraction operand. Returns the
/// narrow value and, for integer extensions, its signedness. Only exact
/// widenings qualify: extf (bf16 and f8E4M3FN to f32 are exact), extsi and
/// extui.
static std::optional<std::pair<Value, std::optional<bool>>>
peelAIE4MatMulWidening(Value wide) {
  if (auto ext = wide.getDefiningOp<arith::ExtFOp>())
    return std::make_pair(ext.getIn(), std::optional<bool>());
  if (auto ext = wide.getDefiningOp<arith::ExtSIOp>())
    return std::make_pair(ext.getIn(), std::optional<bool>(true));
  if (auto ext = wide.getDefiningOp<arith::ExtUIOp>())
    return std::make_pair(ext.getIn(), std::optional<bool>(false));
  return std::nullopt;
}

/// Match an unmasked MxK * Kx64 contraction over directly widened native
/// operands. Transposed maps must first be canonicalized to explicit
/// transposes; larger shapes must be tiled by the producer.
///
/// Floating formats require the contraction's `afn` flag for hardware
/// rounding and subnormal behavior. Integer widening supplies signedness,
/// and exact products with i32 wraparound match contraction semantics.
/// Contractions inside a matrix_region stay generic because a matmul would
/// replace its active B tile.
static std::optional<AIE4MatMulContractMatch>
matchAIE4MatMulContract(vector::ContractionOp op) {
  if (op->getParentOfType<aievec::AIE4MatrixRegionOp>())
    return std::nullopt;
  if (op.getKind() != vector::CombiningKind::ADD)
    return std::nullopt;
  if (cast<vector::MaskableOpInterface>(op.getOperation()).isMasked())
    return std::nullopt;

  auto accTy = dyn_cast<VectorType>(op.getAccType());
  VectorType lhsTy = op.getLhsType(), rhsTy = op.getRhsType();
  if (!accTy || lhsTy.isScalable() || rhsTy.isScalable() ||
      accTy.isScalable() || lhsTy.getRank() != 2 || rhsTy.getRank() != 2 ||
      accTy.getRank() != 2)
    return std::nullopt;

  // Plain row-major matmul: A(m,k) * B(k,n) -> C(m,n). With these maps the
  // contraction verifier already makes k the only reduction dimension and
  // ties the shared M, K and N sizes of lhs, rhs and acc together.
  MLIRContext *ctx = op.getContext();
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  SmallVector<AffineMap> matmulMaps =
      AffineMap::inferFromExprList(MapList{{m, k}, {k, n}, {m, n}}, ctx);
  if (op.getIndexingMapsArray() != ArrayRef<AffineMap>(matmulMaps))
    return std::nullopt;

  auto lhsPeel = peelAIE4MatMulWidening(op.getLhs());
  auto rhsPeel = peelAIE4MatMulWidening(op.getRhs());
  if (!lhsPeel || !rhsPeel)
    return std::nullopt;
  auto [lhs, lhsSigned] = *lhsPeel;
  auto [rhs, rhsSigned] = *rhsPeel;
  auto narrowLhsTy = cast<VectorType>(lhs.getType());
  auto narrowRhsTy = cast<VectorType>(rhs.getType());
  Type narrowTy = narrowLhsTy.getElementType();
  if (narrowRhsTy.getElementType() != narrowTy)
    return std::nullopt;

  auto format = getAIE4MatMulFormat(narrowTy);
  if (!format)
    return std::nullopt;
  auto [kDim, wideTy] = *format;
  // The contraction must compute in the native accumulator type.
  if (lhsTy.getElementType() != wideTy || accTy.getElementType() != wideTy)
    return std::nullopt;
  bool isInteger = isa<IntegerType>(narrowTy);

  if (lhsTy.getDimSize(1) != kDim || accTy.getDimSize(1) != 64 ||
      lhsTy.getDimSize(0) > aievec::kAIE4MatMulMaxRows)
    return std::nullopt;

  if (!isInteger &&
      !arith::bitEnumContainsAll(op.getFastmath(), arith::FastMathFlags::afn))
    return std::nullopt;

  return AIE4MatMulContractMatch{lhs, rhs, lhsSigned, rhsSigned};
}

// vector.contract (one native B tile, see matchAIE4MatMulContract)
//     -> aievec.aie4.matmul
//
// Refuse native lowering in an execution scope that holds flat staging ops.
// Both matmul and matrix_region stage and commit their own B tile, replacing
// any primed staging contents or committed matrix used by the flat code.
struct LowerAIE4ContractToMatMulPattern
    : OpConversionPattern<vector::ContractionOp> {
  LowerAIE4ContractToMatMulPattern(
      MLIRContext *ctx, const llvm::DenseSet<Operation *> &flatScopes)
      : OpConversionPattern(ctx), flatScopes(&flatScopes) {}

  static bool matches(vector::ContractionOp op) {
    return matchAIE4MatMulContract(op).has_value();
  }

  LogicalResult
  matchAndRewrite(vector::ContractionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto match = matchAIE4MatMulContract(op);
    if (!match)
      return rewriter.notifyMatchFailure(op, "not one native AIE4 matmul");
    auto boolAttr = [&](std::optional<bool> v) -> BoolAttr {
      return v ? rewriter.getBoolAttr(*v) : BoolAttr();
    };
    if (flatScopes->contains(aievec::getAIE4ExecutionScope(op))) {
      // Dialect conversion hides pattern-emitted errors; retain the refusal
      // reason as a warning alongside the legalization error.
      op.emitWarning()
          << "AIE4: native vector.contract lowering must not share an "
             "execution scope with flat staging operations; staging and "
             "committing its B tile could replace a primed or committed tile. "
             "Use explicit matrix operations where replacing the tile is safe";
      return failure();
    }
    rewriter.replaceOpWithNewOp<aievec::AIE4MatMulOp>(
        op, adaptor.getAcc().getType(), rewriter.getRemappedValue(match->lhs),
        rewriter.getRemappedValue(match->rhs), adaptor.getAcc(),
        boolAttr(match->lhsSigned), boolAttr(match->rhsSigned));
    return success();
  }

private:
  const llvm::DenseSet<Operation *> *flatScopes;
};

static void populateAIEVecV4ConversionPatterns(
    RewritePatternSet &patterns,
    const llvm::DenseSet<Operation *> &flatScopes) {
  patterns.add<ConvertIntMulToAIE4MacPattern>(patterns.getContext());
  patterns.add<LowerAIE4ApproxMathOp<math::ExpOp>>(patterns.getContext());
  patterns.add<ConvertBF16AddFToAIE4F32Pattern, ConvertBF16SubFToAIE4F32Pattern,
               ConvertBF16FMAToAIE4F32Pattern>(patterns.getContext());
  patterns.add<
      ConvertMulFToAIE4MulEwiseF32Pattern, ConvertMulFToAIE4MulEwiseBf16Pattern,
      ConvertMacFToAIE4MacEwiseF32Pattern, ConvertMacFToAIE4MacEwiseBf16Pattern,
      ConvertMscFToAIE4MscEwiseF32Pattern, ConvertMscFToAIE4MscEwiseBf16Pattern,
      ConvertIntMacToAIE4MacPattern, ConvertExtSIToAIE4UPSPattern,
      ConvertExtUIToAIE4UPSPattern, ConvertExtSIToAIE4UnpackPattern,
      ConvertExtUIToAIE4UnpackPattern, ConvertExtFToAIE4ConvPattern,
      // Shapes the native float->int conversion does not cover, and both
      // int->float directions, which AIE4 has no instruction for. The
      // prohibition on these lives in configureAIEVecCommonLegalizations
      // and applies to every target, so without a pattern here they would
      // have nothing to satisfy it.
      ScalarizeVectorSIToFPOpPattern, ScalarizeVectorUIToFPOpPattern,
      ScalarizeVectorFPToSIOpPattern, ScalarizeVectorFPToUIOpPattern,
      LowerAIE4MinSIOp, LowerAIE4MaxSIOp, LowerAIE4MinUIOp, LowerAIE4MaxUIOp,
      LowerAIE4CmpIOp, LowerAIE4SelectOp, LowerAIE4MaskExtractOp,
      LowerAIE4VectorReductionOp, LowerAIE4MinimumFOp, LowerAIE4MaximumFOp,
      LowerAIE4CmpFOp, LowerAIE4MinNumFOp, LowerAIE4MaxNumFOp,
      LowerVectorTransposeOpToAIE4ShuffleOpPattern,
      LowerVectorShuffleOpToAIE4ShuffleOpPattern,
      LowerAIE4ApproxMathOp<math::Exp2Op>, LowerAIE4ApproxMathOp<math::TanhOp>,
      LowerAIE4ReciprocalOp>(patterns.getContext());
  patterns.add<LowerAIE4ContractToMatMulPattern>(patterns.getContext(),
                                                 flatScopes);
  // Higher benefit than the scalarize patterns above, which also match
  // fptosi/fptoui and would otherwise take the native shape.
  patterns.add<ConvertFPToSIToAIE4ConvPattern, ConvertFPToUIToAIE4ConvPattern>(
      patterns.getContext(), /*benefit=*/2);
}

static void configureAIEVecV4Legalizations(ConversionTarget &target) {
  target.addLegalDialect<aievec::AIEVecDialect>();

  // Mask materialization uses generic vector operations. LLVM conversion
  // erases signedness from the unsigned mask carriers.
  target.addLegalOp<UnrealizedConversionCastOp, vector::BroadcastOp>();
  target.addDynamicallyLegalOp<vector::ExtractOp>(
      [](vector::ExtractOp op) { return !isAIE4MaskExtract(op); });

  // Transpose lowering uses shape casts around the flat AIE4 shuffle.
  target.addLegalOp<vector::ShapeCastOp>();

  // Unaligned reads introduce slices, but AIE4 has no AIEVec slice pattern.
  // Keep them legal for generic lowering, which may scalarize unaligned slices.
  target.addLegalOp<vector::ExtractStridedSliceOp>();

  // Only the exact <32xf32> shape is claimed by an AIE4 intrinsic; every
  // other arith.mulf stays legal and falls through to the generic path.
  target.addDynamicallyLegalOp<arith::MulFOp>([](arith::MulFOp mulOp) {
    auto vecTy = dyn_cast<VectorType>(mulOp.getType());
    if (!vecTy || vecTy.isScalable() || vecTy.getRank() != 1 ||
        !vecTy.getElementType().isF32() || vecTy.getShape()[0] != 32)
      return true;
    // Defer only products whose uses all disappear through MAC fusion.
    return feedsFusableFloatMacChain(mulOp);
  });

  // Materialize surviving widened products; defer only those eliminated by
  // fusion. Unsupported integer multiplies retain their generic lowering.
  target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp mulOp) {
    Value a, b;
    return !matchWidenedI16Product(mulOp.getResult(), a, b) ||
           feedsFusableIntMacChain(mulOp);
  });

  // Use the same predicate as the rewrite so only supported MAC roots are
  // illegal.
  target.addDynamicallyLegalOp<arith::AddIOp>(
      [](arith::AddIOp addOp) { return !matchIntMacChain(addOp); });

  // Keep shift/truncate arithmetic legal: SRS reads mutable crRnd/crSat state.
  // Fusion requires lowering that establishes floor rounding and
  // wrapping, then restores the caller's modes. Explicit SRS ops remain legal.
  target.addLegalOp<arith::ShRSIOp, arith::TruncIOp>();

  // Use native conversion only for shapes claimed by the AIE4 patterns.
  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp op) {
    return !ConvertExtSIToAIE4UPSPattern::matches(op) &&
           !ConvertExtSIToAIE4UnpackPattern::matches(op);
  });
  target.addDynamicallyLegalOp<arith::ExtUIOp>([](arith::ExtUIOp op) {
    return !ConvertExtUIToAIE4UPSPattern::matches(op) &&
           !ConvertExtUIToAIE4UnpackPattern::matches(op);
  });
  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp op) {
    return !ConvertExtFToAIE4ConvPattern::matches(op);
  });
  // Native narrowing reads ambient rounding and saturation modes. Keep truncf
  // generic until lowering can establish its requested modes and restore them.
  target.addLegalOp<arith::TruncFOp>();

  // Fuse matching MSC roots. matchFloatMscChain applies its own shape and
  // fastmath conditions; every other subtract stays legal except bf16, which
  // is computed in f32.
  target.addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) {
    return !matchFloatMscChain(op) &&
           !ConvertBF16SubFToAIE4F32Pattern::matches(op);
  });
  // Fuse matching MAC roots. matchFloatMacChain applies its own shape and
  // fastmath conditions; every other add stays legal except bf16, which is
  // computed in f32.
  target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    return !matchFloatMacChain(op) &&
           !ConvertBF16AddFToAIE4F32Pattern::matches(op);
  });
  target.addDynamicallyLegalOp<vector::FMAOp>([](vector::FMAOp op) {
    return !ConvertBF16FMAToAIE4F32Pattern::matches(op);
  });

  // Require native lowering only for supported integer vector shapes.
  target.addDynamicallyLegalOp<arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp,
                               arith::MaxUIOp>([](Operation *op) {
    auto vecTy = dyn_cast<VectorType>(op->getResultTypes()[0]);
    return !vecTy || !isAIE4VAddShape(vecTy);
  });
  target.addDynamicallyLegalOp<arith::SelectOp>([](arith::SelectOp selOp) {
    auto vecTy = dyn_cast<VectorType>(selOp.getType());
    return !vecTy || (!isAIE4VAddShape(vecTy) && !isAIE4VFCmpShape(vecTy));
  });

  // Integer encoding comparisons preserve IEEE semantics in every FPOrder mode.
  target.addDynamicallyLegalOp<arith::MinimumFOp, arith::MaximumFOp,
                               arith::MinNumFOp, arith::MaxNumFOp>(
      [](Operation *op) {
        auto vecTy = dyn_cast<VectorType>(op->getResultTypes()[0]);
        return !vecTy || !isAIE4VFCmpShape(vecTy);
      });

  target.addDynamicallyLegalOp<arith::CmpFOp>([](arith::CmpFOp cmpOp) {
    auto vecTy = dyn_cast<VectorType>(cmpOp.getLhs().getType());
    return !vecTy || !isAIE4VFCmpShape(vecTy);
  });

  // Equality predicates and unsupported shapes retain generic lowering.
  target.addDynamicallyLegalOp<arith::CmpIOp>([](arith::CmpIOp cmpOp) {
    auto vecTy = dyn_cast<VectorType>(cmpOp.getLhs().getType());
    if (!vecTy || !isAIE4VAddShape(vecTy))
      return true;
    return cmpOp.getPredicate() == arith::CmpIPredicate::eq ||
           cmpOp.getPredicate() == arith::CmpIPredicate::ne;
  });

  // Only contractions that are exactly one native MMAC matmul, with the
  // required numerical permission, are converted; the rest stay generic.
  target.addDynamicallyLegalOp<vector::ContractionOp>(
      [](vector::ContractionOp op) {
        return !LowerAIE4ContractToMatMulPattern::matches(op);
      });

  // Keep add reductions generic unless native f32 reassociation is allowed.
  target.addDynamicallyLegalOp<vector::ReductionOp>([](vector::ReductionOp op) {
    if (op.getKind() != vector::CombiningKind::ADD)
      return false;
    auto vecTy = cast<VectorType>(op.getVector().getType());
    return !isVecF32x32(vecTy) ||
           !arith::bitEnumContainsAll(op.getFastmath(),
                                      arith::FastMathFlags::reassoc);
  });

  // Lower supported transpose/shuffle modes; leave other forms generic.
  target.addDynamicallyLegalOp<vector::TransposeOp>(
      [](vector::TransposeOp transpOp) {
        return !LowerVectorTransposeOpToAIE4ShuffleOpPattern::matchMode(
                    transpOp)
                    .has_value();
      });
  target.addDynamicallyLegalOp<vector::ShuffleOp>(
      [](vector::ShuffleOp shuffleOp) {
        return !LowerVectorShuffleOpToAIE4ShuffleOpPattern::matchMode(shuffleOp)
                    .has_value();
      });

  // exp2/tanh have only approximate afn lowerings; the pattern diagnoses
  // strict ops and unsupported shapes. Wide vectors reassemble their 32-lane
  // results with generic insert_strided_slice.
  target.addIllegalOp<math::Exp2Op, math::TanhOp>();
  // The same holds for vector exp. Scalar exp was never claimed and is left to
  // generic lowering.
  target.addDynamicallyLegalOp<math::ExpOp>(
      [](math::ExpOp op) { return !isa<VectorType>(op.getType()); });
  target.addLegalOp<vector::InsertStridedSliceOp>();

  // Only afn scalar 1.0 / x uses INV; other divisions stay generic.
  target.addDynamicallyLegalOp<arith::DivFOp>(
      [](arith::DivFOp op) { return !LowerAIE4ReciprocalOp::matches(op); });
}

// The lane counts at which the AIE2P lowering of `op` takes one op, smallest
// first, or none if it has no such lowering. For a compare they are those of
// its operands.
static SmallVector<int64_t, 5> getAIE2PNativeLanes(Operation *op) {
  Value v = isa<arith::CmpIOp, arith::CmpFOp>(op) ? op->getOperand(0)
                                                  : op->getResult(0);
  auto type = dyn_cast<VectorType>(v.getType());
  if (!type || !type.getElementType().isIntOrFloat())
    return {};
  Type elemType = type.getElementType();
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  bool isInt = isa<IntegerType>(elemType) &&
               (bitWidth == 8 || bitWidth == 16 || bitWidth == 32);

  // f32 adds and subtracts run on the 2048-bit accumulator, which holds 64
  // lanes; the LLVM lowering pads 16 or 32 lanes to it. bf16 ones are moved
  // to the accumulator and back at 16 lanes, and 32 lanes as two halves.
  if (isa<arith::AddFOp, arith::SubFOp>(op)) {
    if (elemType.isF32())
      return {16, 32, 64};
    if (elemType.isBF16())
      return {16, 32};
    return {};
  }
  // Integer adds and subtracts are left to LLVM, which takes one 512-bit
  // register or less. Splitting wider ones here rather than in LLVM lets the
  // parts feed the parts of a split compare directly.
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    if (!isInt)
      return {};
    SmallVector<int64_t, 5> lanes;
    for (int64_t n = 2; n <= 512 / bitWidth; n *= 2)
      lanes.push_back(n);
    return lanes;
  }
  // The tanh and exp2 intrinsics take 16 lanes of accfloat, which f32 maps
  // onto as well as bf16; 32 lanes are taken as two halves. erf is computed
  // with tanh.
  if (isa<math::TanhOp, math::ErfOp, math::Exp2Op>(op)) {
    if (elemType.isF32() || elemType.isBF16())
      return {16, 32};
    return {};
  }
  // Float max and min take one 512-bit register of bf16.
  if (isa<arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp>(op)) {
    if (elemType.isBF16())
      return {32};
    return {};
  }
  // Compare and select take 32 lanes of bf16 or i16, 16 of which are padded
  // to 32, and 16 lanes of i32. The LLVM lowering compares bf16 as ordered
  // (false if either is NaN); a compare it does not honour is not given a
  // native width, so it is not split or repeated.
  if (isa<arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(op)) {
    if (auto cmpOp = dyn_cast<arith::CmpFOp>(op);
        cmpOp && !llvm::is_contained({CmpFPredicate::OEQ, CmpFPredicate::ONE,
                                      CmpFPredicate::OLT, CmpFPredicate::OLE,
                                      CmpFPredicate::OGT, CmpFPredicate::OGE},
                                     cmpOp.getPredicate()))
      return {};
    if (elemType.isBF16() || elemType.isInteger(16))
      return {16, 32};
    if (elemType.isInteger(32))
      return {16};
    return {};
  }
  return {};
}

// Whether the AIE2P lowering of `op` takes it at its lane count.
static bool isAIE2PNativeWidth(Operation *op) {
  auto type = dyn_cast<VectorType>(op->getResult(0).getType());
  return type && !type.isScalable() &&
         llvm::is_contained(getAIE2PNativeLanes(op), type.getNumElements());
}

// The compare that `op`, a select, selects on.
static Operation *getSelectCompare(arith::SelectOp op) {
  Operation *cmp = skipShapeCastRoundTrip(op.getCondition()).getDefiningOp();
  return isa_and_nonnull<arith::CmpIOp, arith::CmpFOp>(cmp) ? cmp : nullptr;
}

// aievec.ext numbers the parts of its source with an i8.
constexpr int64_t kAIE2PMaxParts = 128;

// The lanes of each part an elementwise op is rewritten into when the AIE2P
// lowering does not take its lane count, or none. Lanes are counted over all
// dimensions. A vector is split into parts of the widest native width that
// divides it or, if none does, repeated to the narrowest native width that
// holds a whole number of copies.
//
// A compare is split together with the select that uses it, as the parts of
// its mask cannot be put back together; the parts are as wide as both the
// compare and the select take.
static std::optional<int64_t> getAIE2PPartLanes(Operation *op) {
  if (op->getNumResults() != 1 || isa<arith::CmpIOp, arith::CmpFOp>(op))
    return std::nullopt;
  auto type = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!type || type.isScalable())
    return std::nullopt;
  int64_t lanes = type.getNumElements();
  if (lanes < 2 || llvm::any_of(
                       op->getOperandTypes(),
                       [&](Type t) {
                         auto vt = dyn_cast<VectorType>(t);
                         return !vt || vt.getShape() != type.getShape();
                       }))
    return std::nullopt;

  SmallVector<int64_t, 5> native = getAIE2PNativeLanes(op);
  if (auto selOp = dyn_cast<arith::SelectOp>(op)) {
    Operation *cmp = getSelectCompare(selOp);
    if (!cmp)
      return std::nullopt;
    SmallVector<int64_t, 5> cmpNative = getAIE2PNativeLanes(cmp);
    llvm::erase_if(
        native, [&](int64_t n) { return !llvm::is_contained(cmpNative, n); });
  }
  if (native.empty() || llvm::is_contained(native, lanes))
    return std::nullopt;
  // LLVM takes an integer add of one register whatever its lane count.
  if (isa<arith::AddIOp, arith::SubIOp>(op) && lanes < native.back())
    return std::nullopt;

  for (int64_t part : llvm::reverse(native))
    if (part < lanes && lanes % part == 0 && lanes / part <= kAIE2PMaxParts)
      return part;
  if (isa<arith::SelectOp>(op))
    return std::nullopt;
  for (int64_t part : native)
    if (part > lanes && part % lanes == 0)
      return part;
  return std::nullopt;
}

// Part `index` of `partLanes` lanes of `v`, or `v` repeated to `partLanes`
// lanes when it has fewer. A splat is built again at the width of the part,
// which keeps the parts of a splat equal, and the part of a concatenation of
// such parts is taken from it directly.
static Value getAIE2PPart(ConversionPatternRewriter &rewriter, Location loc,
                          Value v, int64_t partLanes, int64_t index) {
  auto type = cast<VectorType>(v.getType());
  auto partType = VectorType::get({partLanes}, type.getElementType());
  int64_t lanes = type.getNumElements();
  if (auto cstOp = v.getDefiningOp<arith::ConstantOp>())
    if (auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue());
        dense && dense.isSplat())
      return arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(partType, dense.getSplatValue<Attribute>()));
  if (auto bcastOp = v.getDefiningOp<vector::BroadcastOp>())
    if (!isa<VectorType>(bcastOp.getSourceType()))
      return vector::BroadcastOp::create(rewriter, loc, partType,
                                         bcastOp.getSource());
  if (partLanes < lanes) {
    if (auto concatOp = v.getDefiningOp<aievec::ConcatOp>())
      if (concatOp.getSources().front().getType() == partType)
        return concatOp.getSources()[index];
    return aievec::ExtOp::create(rewriter, loc, partType, v, index);
  }
  SmallVector<Value> copies(partLanes / lanes, v);
  return aievec::ConcatOp::create(rewriter, loc, partType, copies);
}

// Part `index` of the mask of `selOp`. The mask is not cut apart: its compare
// is built again on the parts of the compare's operands. The compare is taken
// from `selOp` itself, as the mask it is converted to may not be a vector.
static Value getAIE2PMaskPart(ConversionPatternRewriter &rewriter, Location loc,
                              arith::SelectOp selOp, int64_t partLanes,
                              int64_t index) {
  Operation *cmp = getSelectCompare(selOp);
  OperationState state(loc, cmp->getName());
  for (Value operand : cmp->getOperands())
    state.addOperands(getAIE2PPart(
        rewriter, loc, rewriter.getRemappedValue(operand), partLanes, index));
  state.addTypes(VectorType::get({partLanes}, rewriter.getI1Type()));
  state.addAttributes(cmp->getAttrs());
  return rewriter.create(state)->getResult(0);
}

// Rewrite a rank-1 elementwise op whose lane count the AIE2P lowering does
// not take into the same op on parts of a lane count it does take (see
// getAIE2PPartLanes): ops on the parts joined by aievec.concat, or one op on
// the repeated operands followed by aievec.ext of the lanes wanted. A select
// takes its compare along.
template <typename OpTy>
struct NormalizeLanesAIE2PPattern : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!type || type.getRank() != 1)
      return failure();
    std::optional<int64_t> partLanes = getAIE2PPartLanes(op);
    if (!partLanes)
      return failure();
    int64_t lanes = type.getNumElements();
    auto partType = VectorType::get({*partLanes}, type.getElementType());
    Location loc = op.getLoc();
    auto createPart = [&](int64_t index) -> Value {
      SmallVector<Value> operands;
      for (Value operand : adaptor.getOperands()) {
        if constexpr (std::is_same_v<OpTy, arith::SelectOp>) {
          if (operands.empty()) {
            operands.push_back(
                getAIE2PMaskPart(rewriter, loc, op, *partLanes, index));
            continue;
          }
        }
        operands.push_back(
            getAIE2PPart(rewriter, loc, operand, *partLanes, index));
      }
      return OpTy::create(rewriter, loc, TypeRange{partType}, operands,
                          op.getProperties(),
                          op->getDiscardableAttrDictionary().getValue())
          ->getResult(0);
    };

    if (*partLanes > lanes) {
      rewriter.replaceOpWithNewOp<aievec::ExtOp>(op, type, createPart(0), 0);
      return success();
    }
    SmallVector<Value> parts;
    for (int64_t i = 0; i < lanes / *partLanes; ++i)
      parts.push_back(createPart(i));
    rewriter.replaceOpWithNewOp<aievec::ConcatOp>(op, type, parts);
    return success();
  }
};

// The elementwise ops AIE2P flattens to rank 1 and, for those with an entry
// in getAIE2PNativeLanes, brings to a lane count its lowering takes. An n-D
// op the rank-1 patterns would leave alone is flattened all the same: the
// LLVM lowering cannot take an n-D vector either.
template <typename... OpTys>
struct AIE2PElementwiseOps {
  static void addPatterns(RewritePatternSet &patterns, PatternBenefit benefit) {
    patterns.add<FlattenElementwiseVectorToRank1Pattern<OpTys>...,
                 NormalizeLanesAIE2PPattern<OpTys>...>(patterns.getContext(),
                                                       benefit);
  }
  static void markIllegal(ConversionTarget &target, MLIRContext *ctx) {
    (markIllegal<OpTys>(target, ctx), ...);
  }
  // An n-D op to flatten, or one to bring to another lane count, is illegal.
  // An op that is already illegal stays so; a dynamically legal one keeps its
  // predicate for the rest.
  template <typename OpTy>
  static void markIllegal(ConversionTarget &target, MLIRContext *ctx) {
    std::optional<ConversionTarget::LegalizationAction> action =
        target.getOpAction(OperationName(OpTy::getOperationName(), ctx));
    if (action == ConversionTarget::LegalizationAction::Legal)
      target.addDynamicallyLegalOp<OpTy>([](Operation *op) {
        return !isFlattenableElementwise(op) && !getAIE2PPartLanes(op);
      });
    else if (action == ConversionTarget::LegalizationAction::Dynamic)
      target.addDynamicallyLegalOp<OpTy>(
          [](Operation *op) -> std::optional<bool> {
            if (isFlattenableElementwise(op) || getAIE2PPartLanes(op))
              return false;
            return std::nullopt;
          });
  }
};
using AIE2PElementwise = AIE2PElementwiseOps<
    arith::MulFOp, arith::DivFOp, arith::AddFOp, arith::SubFOp, arith::NegFOp,
    arith::TruncFOp, arith::ExtFOp, math::ExpOp, math::TanhOp, math::ErfOp,
    math::Exp2Op, arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
    arith::CmpFOp, arith::CmpIOp, arith::SelectOp, arith::AddIOp, arith::SubIOp,
    arith::MulIOp, arith::IndexCastOp>;

static void populateAIEVecV2PConversionPatterns(
    RewritePatternSet &patterns, std::shared_ptr<AIE2PShufflePlanner> planner) {
  populateAIEVecV2CommonConversionPatterns(patterns);
  patterns.add<LowerVectorContractionOpToAIEVecMatMulOpAIE2P>(
      patterns.getContext());
  patterns.add<ConvertVectorFMAOpToAIEVecFMAElemOpAIE2PPattern,
               ConvertMulFToAIEVecMulElemOpAIE2PPattern>(patterns.getContext(),
                                                         /*benefit=*/2);
  patterns.add<LowerVectorSIToFPI16BF16AIE2pPattern,
               LowerVectorUIToFPI16BF16AIE2pPattern>(
      patterns.getContext(), /*accumulatorAdd=*/true, /*benefit=*/4);
  // AIE2p-specific broadcast pattern that handles 256-bit directly
  patterns.add<ConvertSplatToAIEBroadcastAIE2p>(patterns.getContext());
  populateAIE2PShufflePlannerPatterns(patterns, std::move(planner));
  patterns.add<LowerVectorReductionAddBfloat16OpAIE2P>(patterns.getContext());
  // For AIE2P with LLVMIR backend, use aievec.exp and aievec.inv
  // math.rsqrt is kept legal and will be lowered in AIEVecToLLVM pass
  patterns
      .add<ConvertMathExpToAIEVecExpOpPattern, ConvertDivFToAIEVecInvOpPattern>(
          patterns.getContext());
  // Higher benefit than the AIE2P-specific patterns below, which would
  // otherwise take an n-D op before it has been flattened.
  AIE2PElementwise::addPatterns(patterns, /*benefit=*/3);
  // Higher benefit to take priority over the AIE2 LUT-based tanh and the
  // emitc-based erf patterns registered in the common patterns.
  patterns.add<
      ConvertMathToAIEVecAccFloatUnaryOpPattern<math::TanhOp, aievec::TanhOp>,
      ConvertMathToAIEVecAccFloatUnaryOpPattern<math::Exp2Op, aievec::Exp2Op>,
      ComputeErfOpAIE2PPattern,
      ComputeScalarMathOpByVectorAIE2PPattern<math::TanhOp>,
      ComputeScalarMathOpByVectorAIE2PPattern<math::ErfOp>,
      ComputeScalarMathOpByVectorAIE2PPattern<math::Exp2Op>,
      ComputeOneLaneMathOpAIE2PPattern<math::TanhOp>,
      ComputeOneLaneMathOpAIE2PPattern<math::ErfOp>,
      ComputeOneLaneMathOpAIE2PPattern<math::Exp2Op>>(patterns.getContext(),
                                                      /*benefit=*/2);
}

//===----------------------------------------------------------------------===//
// Legalizations
//===----------------------------------------------------------------------===//

// TODO: Review the validity of these legalizations beyond basic cases.

static bool isInSigmoidOperationChain(math::ExpOp expOp) {
  if (!expOp.getOperand().getDefiningOp<arith::NegFOp>())
    return false;

  arith::AddFOp addOp = nullptr;
  for (Operation *user : expOp->getUsers()) {
    addOp = dyn_cast<arith::AddFOp>(user);
    if (addOp)
      break;
  }

  if (!addOp)
    return false;

  auto *addLvalOp = addOp.getLhs().getDefiningOp();
  auto *addRvalOp = addOp.getRhs().getDefiningOp();
  if (!((isa<math::ExpOp>(addLvalOp) && isa<arith::ConstantOp>(addRvalOp)) ||
        (isa<math::ExpOp>(addRvalOp) && isa<arith::ConstantOp>(addLvalOp))))
    return false;

  auto constOp = isa<arith::ConstantOp>(addLvalOp)
                     ? cast<arith::ConstantOp>(addLvalOp)
                     : cast<arith::ConstantOp>(addRvalOp);

  auto cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;

  if (cstDense.getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  arith::DivFOp divOp = nullptr;
  for (Operation *user : addOp->getUsers()) {
    divOp = dyn_cast<arith::DivFOp>(user);
    if (divOp)
      break;
  }

  if (!divOp)
    return false;

  constOp = dyn_cast<arith::ConstantOp>(divOp.getLhs().getDefiningOp());
  if (!constOp)
    return false;
  cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;
  if (cstDense.getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  return true;
}

static void configureAIEVecCommonLegalizations(ConversionTarget &target) {
  target
      .addLegalDialect<xilinx::aievec::aie1::AIEVecAIE1Dialect,
                       xilinx::aievec::AIEVecDialect, arith::ArithDialect,
                       ub::UBDialect, emitc::EmitCDialect, func::FuncDialect>();
  target.addIllegalOp<vector::ExtractStridedSliceOp>();
  target.addLegalOp<vector::BitCastOp>();

  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp extfOp) {
    auto srcType = dyn_cast<VectorType>(extfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    return srcElWidth != 16 || srcLaneSize != 16 || dstElWidth != 32 ||
           dstLaneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp extsiOp) {
    auto srcType = dyn_cast<VectorType>(extsiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extsiOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    return srcLaneSize != 32 || (dstElWidth <= srcElWidth) ||
           (dstLaneSize != srcLaneSize);
  });

  // Mark `arith.extui vector<Nxi4> to vector<Nxi8>` illegal precisely when
  // the producer is a `vector.bitcast vector<N/2xi8> to vector<Nxi4>` of a
  // supported lane count (64 or 128 nibbles). That is the only shape
  // LowerExtUIOfBitcastI4ToUnpackPattern can rewrite; stray ExtUI ops
  // (including i4 ones not fed by a bitcast) stay legal and pass through
  // to the standard arith-to-llvm path.
  target.addDynamicallyLegalOp<arith::ExtUIOp>([](arith::ExtUIOp extuiOp) {
    auto srcType = dyn_cast<VectorType>(extuiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extuiOp.getOut().getType());
    if (!srcType || !dstType)
      return true;
    if (srcType.getRank() != 1 || dstType.getRank() != 1)
      return true;
    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!srcScalarType.isInteger(4) || !dstScalarType.isInteger(8))
      return true;
    unsigned lanes = srcType.getNumElements();
    if (lanes != 64 && lanes != 128)
      return true;
    auto bitcastOp = extuiOp.getIn().getDefiningOp<vector::BitCastOp>();
    if (!bitcastOp)
      return true;
    auto bitcastSrcType = dyn_cast<VectorType>(bitcastOp.getSource().getType());
    return !bitcastSrcType || bitcastSrcType.getRank() != 1 ||
           !bitcastSrcType.getElementType().isInteger(8) ||
           bitcastSrcType.getNumElements() * 2 != lanes;
  });

  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp truncfOp) {
    auto srcType = dyn_cast<VectorType>(truncfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(truncfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    return srcElWidth != 32 || srcLaneSize != 16 || dstElWidth != 16 ||
           dstLaneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::TruncIOp>([](arith::TruncIOp trunciOp) {
    auto srcType = dyn_cast<VectorType>(trunciOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(trunciOp.getOut().getType());
    if (!srcType || !dstType) {
      // Scalar trunci: mark illegal if part of compound SRS chain
      // so the LowerScalarShiftClampTruncToSRS pattern can convert it.
      return !(!srcType && !dstType && isSRSCompoundCandidate(trunciOp));
    }

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    // Also mark vector trunci as illegal if it's part of a compound SRS chain
    if (isSRSCompoundCandidate(trunciOp))
      return false;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();

    return srcLaneSize != 32 || (dstElWidth >= srcElWidth) ||
           (dstLaneSize != srcLaneSize);
  });

  target.addDynamicallyLegalOp<math::TanhOp>([](math::TanhOp tanhOp) {
    auto srcType = dyn_cast<VectorType>(tanhOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || laneSize != 16;
  });

  target.addDynamicallyLegalOp<math::SqrtOp>([](math::SqrtOp sqrtOp) {
    auto srcType = dyn_cast<VectorType>(sqrtOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<math::ErfOp>([](math::ErfOp erfOp) {
    auto srcType = dyn_cast<VectorType>(erfOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<math::AbsFOp>([](math::AbsFOp absfOp) {
    auto srcType = dyn_cast<VectorType>(absfOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth * laneSize != 512 && elWidth * laneSize != 256;
  });

  target.addDynamicallyLegalOp<math::AbsIOp>([](math::AbsIOp absiOp) {
    auto srcType = dyn_cast<VectorType>(absiOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth * laneSize != 512 && elWidth * laneSize != 256;
  });

  target.addDynamicallyLegalOp<math::CeilOp>([](math::CeilOp ceilOp) {
    auto srcType = dyn_cast<VectorType>(ceilOp.getOperand().getType());
    if (!srcType)
      return true;
    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<math::FloorOp>([](math::FloorOp floorOp) {
    auto srcType = dyn_cast<VectorType>(floorOp.getOperand().getType());
    if (!srcType)
      return true;
    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<arith::NegFOp>([](arith::NegFOp negOp) {
    auto srcType = dyn_cast<VectorType>(negOp.getOperand().getType());
    if (!srcType)
      return true;
    if (Type scalarType = srcType.getElementType(); !isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    return laneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::ShRSIOp>([](arith::ShRSIOp rsOp) {
    auto srcType = dyn_cast<VectorType>(rsOp.getLhs().getType());
    if (!srcType) {
      // Scalar i32 shrsi: illegal unless it feeds into a compound SRS chain
      // (the compound pattern consumes it via the trunci anchor)
      if (auto intType = dyn_cast<IntegerType>(rsOp.getLhs().getType()))
        if (intType.getWidth() == 32) {
          // legal (compound pattern will handle) iff feeding a compound SRS;
          // otherwise illegal so the individual pattern promotes it
          return shrsiUsedByCompoundSRS(rsOp);
        }
      return true;
    }

    // If the shrsi feeds into a compound SRS pattern (shrsi+clamp+trunc),
    // keep it legal — the compound pattern will consume it via the trunci.
    if (shrsiUsedByCompoundSRS(rsOp))
      return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::AddFOp>(
      [](arith::AddFOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::SubFOp>(
      [](arith::SubFOp op) { return !isa<VectorType>(op.getType()); });

  // AIE2/AIE2P GISel legalize only the scalar form of int<->fp conversions
  // (libcalls). Vector forms have no legalizer rule and abort with
  // "unable to legalize G_SITOFP <N x sX>". Mark the vector forms that
  // ScalarizeVectorIntFpConversionOpPattern can rewrite (1-D vectors) as
  // illegal so the pattern fires. Scalar and >1-D vector forms stay legal:
  // scalar reaches the supported libcall path directly, and >1-D forms
  // pass through to surface a clearer error from a later stage instead of
  // aborting this conversion. The scalarize pattern emits vector.extract /
  // vector.insert / scalar conv sequences, so those ops must be legal too.
  auto isRank1Vector = [](Type t) {
    auto vt = dyn_cast<VectorType>(t);
    return vt && vt.getRank() == 1;
  };
  target.addDynamicallyLegalOp<arith::SIToFPOp>(
      [=](arith::SIToFPOp op) { return !isRank1Vector(op.getType()); });
  target.addDynamicallyLegalOp<arith::UIToFPOp>(
      [=](arith::UIToFPOp op) { return !isRank1Vector(op.getType()); });
  target.addDynamicallyLegalOp<arith::FPToSIOp>(
      [=](arith::FPToSIOp op) { return !isRank1Vector(op.getType()); });
  target.addDynamicallyLegalOp<arith::FPToUIOp>(
      [=](arith::FPToUIOp op) { return !isRank1Vector(op.getType()); });
  target.addLegalOp<vector::ExtractOp, vector::InsertOp>();
}

static void configureAIEVecV1Legalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<arith::MulIOp>(
      [](arith::MulIOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::MulFOp>(
      [](arith::MulFOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<aievec::aie1::FMAOp>(
      [](xilinx::aievec::aie1::FMAOp op) {
        auto *lhsDefOp = op.getLhs().getDefiningOp();
        aievec::ConcatOp concatOp = nullptr;
        if (lhsDefOp)
          concatOp = dyn_cast<aievec::ConcatOp>(op.getLhs().getDefiningOp());
        if (!concatOp)
          return true;

        vector::BroadcastOp srcBcast = nullptr;
        if (auto *lhsOp = concatOp.getSources()[0].getDefiningOp())
          srcBcast = dyn_cast<vector::BroadcastOp>(lhsOp);
        if (!srcBcast) {
          auto *rhsOp = op.getRhs().getDefiningOp();
          if (!rhsOp)
            return true;
          srcBcast = dyn_cast<vector::BroadcastOp>(rhsOp);
        }

        if (srcBcast)
          if (auto *srcOp = srcBcast.getSource().getDefiningOp())
            return !isa<vector::ExtractOp>(srcOp);

        return true;
      });

  target.addDynamicallyLegalOp<aievec::aie1::AddOp>([](aievec::aie1::AddOp op) {
    auto lSrsOp = op.getLhs().getDefiningOp<aievec::SRSOp>();
    auto rSrsOp = op.getRhs().getDefiningOp<aievec::SRSOp>();
    return (!lSrsOp ||
            !lSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>()) &&
           (!rSrsOp ||
            !rSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>());
  });
  target.addLegalDialect<memref::MemRefDialect>();
}

static void
configureAIEVecV2PLegalizations(ConversionTarget &target,
                                std::shared_ptr<AIE2PShufflePlanner> planner) {
  // AIE2P-specific legalization for rsqrt with LLVMIR backend
  // Vector bf16 rsqrt is illegal (no hardware support)
  // Scalar f32 and vector f32 rsqrt are legal (lowered in AIEVecToLLVM pass)
  target.addDynamicallyLegalOp<math::RsqrtOp>([](math::RsqrtOp rsqrtOp) {
    auto vecType = dyn_cast<VectorType>(rsqrtOp.getOperand().getType());
    // Vector bf16 rsqrt is illegal; everything else is legal (scalar f32,
    // vector f32)
    return !vecType || !vecType.getElementType().isBF16();
  });

  // AIE2P-specific legalization for exp with LLVMIR backend
  // v16bf16 and v32bf16 exp are illegal (uses hardware intrinsic)
  target.addDynamicallyLegalOp<math::ExpOp>([](math::ExpOp expOp) {
    auto srcType = dyn_cast<VectorType>(expOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(srcType);
    // AIE2P LLVMIR: v16f32 and v32f32 are illegal (uses aievec.exp on bf16)
    if (scalarType.isF32())
      return srcType.isScalable() || (laneSize != 16 && laneSize != 32);
    // AIE2P LLVMIR: v16bf16 and v32bf16 are illegal (uses aievec.exp)
    if (!scalarType.isBF16() || (laneSize != 16 && laneSize != 32) ||
        elWidth != 16)
      return true;
    if (expOp->hasOneUse() && isInSigmoidOperationChain(expOp))
      return true;

    return false;
  });

  // AIE2P-specific legalization for tanh, erf and exp2 with LLVMIR backend
  // bf16 and f32 scalars, and one-, 16- and 32-lane vectors are illegal (uses
  // aievec.tanh or aievec.exp2)
  target.addDynamicallyLegalOp<math::TanhOp>([](math::TanhOp tanhOp) {
    return !isAIE2PAccFloatUnaryOrScalarType(tanhOp.getOperand().getType());
  });
  target.addDynamicallyLegalOp<math::ErfOp>([](math::ErfOp erfOp) {
    return !isAIE2PAccFloatUnaryOrScalarType(erfOp.getOperand().getType());
  });
  target.addDynamicallyLegalOp<math::Exp2Op>([](math::Exp2Op exp2Op) {
    return !isAIE2PAccFloatUnaryOrScalarType(exp2Op.getOperand().getType());
  });

  // AIE2P-specific legalization for divf 1.0/x pattern with LLVMIR backend
  // Scalar f32 or vector<Nxf32> divf with constant 1.0 LHS is illegal
  target.addDynamicallyLegalOp<arith::DivFOp>([](arith::DivFOp divfOp) {
    Type srcType = divfOp.getLhs().getType();

    // Check if LHS is defined by a constant operation
    auto constOp =
        dyn_cast_or_null<arith::ConstantOp>(divfOp.getLhs().getDefiningOp());
    if (!constOp)
      return true;

    // Scalar f32 case - check for exactly 1.0
    if (srcType.isF32()) {
      auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
      // illegal (will be converted to aievec.inv) iff exactly 1.0
      return !floatAttr || !floatAttr.getValue().isExactlyValue(1.0);
    }

    // Vector f32 case - check for splat of exactly 1.0
    if (auto vecType = dyn_cast<VectorType>(srcType)) {
      if (vecType.getElementType().isF32()) {
        unsigned laneSize = getVectorLaneSize(vecType);
        if (laneSize == 16 || laneSize == 32) {
          auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
          if (denseAttr && denseAttr.isSplat() &&
              denseAttr.getSplatValue<APFloat>().isExactlyValue(1.0))
            return false; // illegal - will be converted to aievec.inv
        }
      }
    }

    return true;
  });

  // AIE2P-specific legalization: ExtFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp extfOp) {
    auto srcType = dyn_cast<VectorType>(extfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  // AIE2P-specific legalization: TruncFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp truncfOp) {
    auto srcType = dyn_cast<VectorType>(truncfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(truncfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  // AIE2P-specific legalization: ExtSIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp extsiOp) {
    auto srcType = dyn_cast<VectorType>(extsiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extsiOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  // AIE2P-specific legalization: TruncIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncIOp>([](arith::TruncIOp trunciOp) {
    auto srcType = dyn_cast<VectorType>(trunciOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(trunciOp.getOut().getType());
    if (!srcType || !dstType) {
      // Scalar trunci: mark illegal if part of compound SRS chain
      return !(!srcType && !dstType && isSRSCompoundCandidate(trunciOp));
    }
    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    // Also mark as illegal if it's part of a shrsi+clamp+trunc SRS pattern,
    // even for sub-AIE-width vectors
    if (isSRSCompoundCandidate(trunciOp))
      return false;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  // AIE2P-specific legalization: Override AddFOp to support laneSize==32 for
  // float types, and bf16 and f32 at the widths getAIE2PNativeLanes gives.
  target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (scalarType.isBF16() || scalarType.isF32())
      return !isAIE2PNativeWidth(op);

    // For other float types, support both laneSize==16 and laneSize==32
    if (isa<FloatType>(scalarType))
      return laneSize != 16 && laneSize != 32;

    // For other types, only laneSize==16 (same as AIE2)
    return laneSize != 16;
  });

  // AIE2P-specific legalization: Override SubFOp to support laneSize==32 for
  // float types, and bf16 and f32 at the widths getAIE2PNativeLanes gives.
  target.addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (scalarType.isBF16() || scalarType.isF32())
      return !isAIE2PNativeWidth(op);

    // For other float types, support both laneSize==16 and laneSize==32
    if (isa<FloatType>(scalarType))
      return laneSize != 16 && laneSize != 32;

    // For other types, only laneSize==16 (same as AIE2)
    return laneSize != 16;
  });

  // AIE2P-specific legalization: Override NegFOp to support laneSize==32 for
  // bf16 and f32.
  target.addDynamicallyLegalOp<arith::NegFOp>([](arith::NegFOp negOp) {
    auto srcType = dyn_cast<VectorType>(negOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);

    // bf16 and f32 only: the aievec lowering needs an f32 accumulator, and
    // other 16-bit floats (e.g. f16) would take the bf16 UPS path and be
    // reinterpreted.
    if (!scalarType.isBF16() && !scalarType.isF32())
      return true;

    // 16 lanes as before, at whatever rank the common config already took.
    if (laneSize == 16)
      return false;

    // 32 only at rank 1. `getVectorLaneSize` is the product of every
    // dimension, so vector<2x16xf32> counts 32 as well; the widening is
    // scoped to the rank the aievec patterns match, and leaves n-D negates
    // as they were before this branch.
    return laneSize != 32 || srcType.getRank() != 1;
  });

  // AIE2P-specific legalization: Override MulFOp to narrow the mul+add FMA
  // exemption to rank 1. The pattern that fuses the pair matches a rank-1
  // multiply, so above rank 1 the multiply has to stay illegal and be
  // flattened first.
  target.addDynamicallyLegalOp<arith::MulFOp>([](arith::MulFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto isAddOp = [](Operation *user) { return isa<arith::AddFOp>(user); };
    if (resultType.getRank() == 1 && op->hasOneUse() &&
        llvm::any_of(op->getUsers(), isAddOp) &&
        !(holdsBF16(op.getLhs()) && holdsBF16(op.getRhs())))
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (laneSize == 16 && (resultElWidth == 16 || resultElWidth == 32))
      return false; // illegal - will be converted
    if (laneSize == 32 && resultElWidth == 16)
      return false; // illegal - will be split into two v16bf16 ops
    if ((laneSize == 32 || laneSize == 64) && resultElWidth == 32 &&
        resultType.getRank() == 1 && holdsBF16(op.getLhs()) &&
        holdsBF16(op.getRhs()))
      return false; // illegal - bf16 inputs, f32 product

    return true; // legal - not supported
  });

  // A vector.shuffle stays legal for LLVM unless the shuffle planner has a
  // cheaper lowering. Moves of whole register halves, such as the splitting
  // and joining shuffles LowerVectorSIToFPI16BF16AIE2pPattern emits, always
  // stay.
  configureAIE2PShufflePlannerLegality(target, std::move(planner));
}

static void configureAIEVecV2Legalizations(ConversionTarget &target) {
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<vector::ShapeCastOp>();

  // A set recording the vector lane size and element width supported
  llvm::SmallSet<std::pair<unsigned, unsigned>, 16> laneSizeElWidthPairSet;
  laneSizeElWidthPairSet.insert({64, 8});
  laneSizeElWidthPairSet.insert({32, 16});
  laneSizeElWidthPairSet.insert({16, 32});
  laneSizeElWidthPairSet.insert({32, 32});

  // A set recording the element width supported
  llvm::SmallSet<unsigned, 16> elWidthSet;
  elWidthSet.insert(8);
  elWidthSet.insert(16);
  elWidthSet.insert(32);

  target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();

    // Support laneSize == 16 for f32/bf16
    if (laneSize == 16)
      return false; // illegal - will be converted
    // Support laneSize == 32 for bf16 (split into two v16bf16 ops)
    if (laneSize == 32 && resultElWidth == 16)
      return false; // illegal - will be split
    // Support laneSize == 32 for f32 (split into two v16f32 ops)
    if (laneSize == 32 && resultElWidth == 32)
      return false; // illegal - will be split into two v16f32 ops

    return true; // legal - not supported
  });

  target.addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();

    // Support laneSize == 16 for f32/bf16
    if (laneSize == 16)
      return false; // illegal - will be converted
    // Support laneSize == 32 for bf16 (split into two v16bf16 ops)
    if (laneSize == 32 && resultElWidth == 16)
      return false; // illegal - will be split
    // Support laneSize == 32 for f32 (split into two v16f32 ops)
    if (laneSize == 32 && resultElWidth == 32)
      return false; // illegal - will be split into two v16f32 ops

    return true; // legal - not supported
  });

  target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;
    auto isAddOp = [&](Operation *op) { return isa<arith::AddIOp>(op); };
    // Verify it is not a part of MAC
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp))
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return (laneSize != 32 || (resultElWidth != 16 && resultElWidth != 8)) &&
           ((laneSize != 16 && laneSize != 32) || resultElWidth != 32);
  });

  target.addDynamicallyLegalOp<arith::MulFOp>([](arith::MulFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto isAddOp = [&](Operation *op) { return isa<arith::AddFOp>(op); };
    // Verify it is not a part of FMA
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp) &&
        !(holdsBF16(op.getLhs()) && holdsBF16(op.getRhs())))
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    // Support laneSize == 16 for bf16/f32, and laneSize == 32 for bf16 (split)
    if (laneSize == 16 && (resultElWidth == 16 || resultElWidth == 32))
      return false; // illegal - will be converted
    if (laneSize == 32 && resultElWidth == 16)
      return false; // illegal - will be split into two v16bf16 ops
    if (laneSize == 32 && resultElWidth == 32 && holdsBF16(op.getLhs()) &&
        holdsBF16(op.getRhs()))
      return false; // illegal - bf16 inputs, f32 product

    return true; // legal - not supported
  });

  target.addDynamicallyLegalOp<arith::MinSIOp>([=](arith::MinSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      // Scalar i8/i16/i32 minsi: illegal unless in compound SRS chain
      if (auto intType = dyn_cast<IntegerType>(op.getType())) {
        unsigned w = intType.getWidth();
        if (w == 8 || w == 16 || w == 32) {
          // legal (compound pattern consumes) iff feeding a compound SRS;
          // otherwise illegal so the individual pattern promotes it
          return scalarClampInCompoundSRS(op);
        }
      }
      return true;
    }

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::MaxSIOp>([=](arith::MaxSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      // Scalar i8/i16/i32 maxsi: illegal unless in compound SRS chain
      if (auto intType = dyn_cast<IntegerType>(op.getType())) {
        unsigned w = intType.getWidth();
        if (w == 8 || w == 16 || w == 32) {
          // legal (compound pattern consumes) iff feeding a compound SRS;
          // otherwise illegal so the individual pattern promotes it
          return scalarClampInCompoundSRS(op);
        }
      }
      return true;
    }

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::MinUIOp>([=](arith::MinUIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned totalBits = laneSize * resultElWidth;

    return !elWidthSet.count(resultElWidth) ||
           (totalBits != 512 && (totalBits != 256 || resultElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::MaxUIOp>([=](arith::MaxUIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned totalBits = laneSize * resultElWidth;

    return !elWidthSet.count(resultElWidth) ||
           (totalBits != 512 && (totalBits != 256 || resultElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::MinimumFOp>([=](arith::MinimumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned totalBits = laneSize * resultElWidth;

    return !elWidthSet.count(resultElWidth) ||
           (totalBits != 512 && (totalBits != 256 || resultElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::MaximumFOp>([=](arith::MaximumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned totalBits = laneSize * resultElWidth;

    return !elWidthSet.count(resultElWidth) ||
           (totalBits != 512 && (totalBits != 256 || resultElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::MaxNumFOp>([=](arith::MaxNumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned totalBits = laneSize * resultElWidth;

    return !elWidthSet.count(resultElWidth) ||
           (totalBits != 512 && (totalBits != 256 || resultElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::MinNumFOp>([=](arith::MinNumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned totalBits = laneSize * resultElWidth;

    return !elWidthSet.count(resultElWidth) ||
           (totalBits != 512 && (totalBits != 256 || resultElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::CmpIOp>([=](arith::CmpIOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType)
      return true;

    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);
    unsigned totalBits = laneSize * lhsElWidth;

    return !elWidthSet.count(lhsElWidth) ||
           (totalBits != 512 && (totalBits != 256 || lhsElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::CmpFOp>([=](arith::CmpFOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType)
      return true;

    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);
    unsigned totalBits = laneSize * lhsElWidth;

    return !elWidthSet.count(lhsElWidth) ||
           (totalBits != 512 && (totalBits != 256 || lhsElWidth != 16));
  });

  target.addDynamicallyLegalOp<arith::SelectOp>([=](arith::SelectOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned totalBits = laneSize * resultElWidth;

    return !elWidthSet.count(resultElWidth) ||
           (totalBits != 512 && (totalBits != 256 || resultElWidth != 16));
  });

  target.addDynamicallyLegalOp<vector::ReductionOp>(
      [=](vector::ReductionOp op) {
        if (auto kind = op.getKind(); kind != vector::CombiningKind::ADD &&
                                      kind != vector::CombiningKind::MINSI &&
                                      kind != vector::CombiningKind::MINUI &&
                                      kind != vector::CombiningKind::MINIMUMF &&
                                      kind != vector::CombiningKind::MINNUMF &&
                                      kind != vector::CombiningKind::MAXSI &&
                                      kind != vector::CombiningKind::MAXUI &&
                                      kind != vector::CombiningKind::MAXIMUMF &&
                                      kind != vector::CombiningKind::MAXNUMF)
          return true;

        auto vType = dyn_cast<VectorType>(op.getVector().getType());
        if (!vType)
          return true;

        llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
        laneSizeElWidthPairSet.insert({64, 8});
        laneSizeElWidthPairSet.insert({32, 16});
        laneSizeElWidthPairSet.insert({32, 32});
        laneSizeElWidthPairSet.insert({16, 32});

        Type scalarType = vType.getElementType();
        unsigned elWidth = scalarType.getIntOrFloatBitWidth();
        unsigned laneSize = getVectorLaneSize(vType);

        if (isa<IntegerType>(scalarType) &&
            !laneSizeElWidthPairSet.count(std::make_pair(laneSize, elWidth)))
          return true;

        if (isa<FloatType>(scalarType) && laneSize != 16 && laneSize != 32)
          return true;

        return false;
      });

  // AIE2-specific legalization: ExtFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp extfOp) {
    auto srcType = dyn_cast<VectorType>(extfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  // AIE2-specific legalization: TruncFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp truncfOp) {
    auto srcType = dyn_cast<VectorType>(truncfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(truncfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  // AIE2-specific legalization: ExtSIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp extsiOp) {
    auto srcType = dyn_cast<VectorType>(extsiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extsiOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  // AIE2-specific legalization: TruncIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncIOp>([](arith::TruncIOp trunciOp) {
    auto srcType = dyn_cast<VectorType>(trunciOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(trunciOp.getOut().getType());
    if (!srcType || !dstType) {
      // Scalar trunci: mark illegal if part of compound SRS chain
      return !(!srcType && !dstType && isSRSCompoundCandidate(trunciOp));
    }
    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    // Also mark as illegal if it's part of a shrsi+clamp+trunc SRS pattern,
    // even for sub-AIE-width vectors
    if (isSRSCompoundCandidate(trunciOp))
      return false;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    return (srcLaneSize % 16 != 0) || (dstLaneSize % 16 != 0);
  });

  target.addIllegalOp<vector::ContractionOp, vector::TransposeOp,
                      vector::FMAOp>();

  // AIE2-specific legalization: math.exp for v16bf16 and v32bf16 is illegal
  // (uses LUT)
  target.addDynamicallyLegalOp<math::ExpOp>([](math::ExpOp expOp) {
    auto srcType = dyn_cast<VectorType>(expOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(srcType);
    // AIE2: v16bf16 and v32bf16 are illegal (uses LUT-based lowering)
    if (!isa<FloatType>(scalarType) || (laneSize != 16 && laneSize != 32) ||
        elWidth != 16)
      return true;
    if (expOp->hasOneUse() && isInSigmoidOperationChain(expOp))
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::RsqrtOp>([](math::RsqrtOp rsqrtOp) {
    auto srcType = dyn_cast<VectorType>(rsqrtOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });
}

//===----------------------------------------------------------------------===//
// Lowering passes
//===----------------------------------------------------------------------===//

/// Lower incoming vector operations into their corresponding AIE vector
/// intrinsics.
struct LowerVectorToAIEVec : PassWrapper<LowerVectorToAIEVec, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorToAIEVec)

  LowerVectorToAIEVec() = default;
  LowerVectorToAIEVec(const LowerVectorToAIEVec &pass) : PassWrapper(pass) {}

  LowerVectorToAIEVec(const LowerVectorToAIEVecOptions &options)
      : LowerVectorToAIEVec() {
    aieTarget = options.aieTarget;
  }

  // In case we want to register this pass as a standalone pass for test
  // purposes.
  StringRef getArgument() const final { return "test-lower-vector-to-aievec"; }
  StringRef getDescription() const final {
    return "Lower vector operations to AIE vector intrinsics";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, xilinx::aievec::aie1::AIEVecAIE1Dialect,
        xilinx::aievec::AIEVecDialect, arith::ArithDialect,
        memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
        vector::VectorDialect, emitc::EmitCDialect, xilinx::AIE::AIEDialect>();
  }

  Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc(
          "Select AIE version: \"aie\", \"aieml\"/\"aie2\", \"aie2p\", or "
          "\"aie4b\". This will determine the vector size and available "
          "operations."),
      llvm::cl::init("aie")};

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    // `aie.runtime_sequence` bodies are host-side control code (either NPU
    // instruction streams or C++ TXN builders), not AIE-core compute. Vector
    // and AIEVec ops are meaningless there, so leave anything nested inside a
    // runtime sequence untouched regardless of the legalizations configured
    // below.
    target.addLegalOp<xilinx::AIE::RuntimeSequenceOp>();
    target.markOpRecursivelyLegal<xilinx::AIE::RuntimeSequenceOp>();

    // AIE4: refuse native contraction lowering in execution scopes with flat
    // staging ops. Must outlive applyPartialConversion.
    llvm::DenseSet<Operation *> flatScopes = collectAIE4FlatScopes(op);

    std::optional<AIEVecTargetInfo> maybeTargetInfo =
        parseAIEVecTarget(aieTarget, op);
    if (!maybeTargetInfo.has_value())
      return signalPassFailure();
    AIEVecTargetInfo targetInfo = *maybeTargetInfo;
    if (targetInfo.isAIE2PS()) {
      op->emitError() << "AIEVec has no vector lowering for target 'aie2ps'";
      signalPassFailure();
      return;
    }

    if (!isVectorLoweringEnabled(targetInfo)) {
      op->emitError() << "Vector lowering is not enabled for target "
                      << getCanonicalAIEVecTargetName(targetInfo);
      return signalPassFailure();
    }

    populateAIEVecCommonConversionPatterns(patterns);
    configureAIEVecCommonLegalizations(target);
    if (targetInfo.isAIE1()) {
      populateAIEVecV1ConversionPatterns(patterns);
      configureAIEVecV1Legalizations(target);
    } else if (targetInfo.isAIE2()) {
      populateAIEVecV2ConversionPatterns(patterns);
      configureAIEVecV2Legalizations(target);
    } else if (targetInfo.isAIE2P()) {
      auto planner = std::make_shared<AIE2PShufflePlanner>();
      populateAIEVecV2PConversionPatterns(patterns, planner);
      configureAIEVecV2Legalizations(target);
      configureAIEVecV2PLegalizations(target, planner);
      // Last, as it overrides the predicates above for the ops it rewrites.
      AIE2PElementwise::markIllegal(target, &getContext());
    } else if (targetInfo.isAIE4()) {
      // Make masked operations tail-safe before the target lowering.
      if (failed(lowerAIE4Masks(op)))
        return signalPassFailure();
      populateAIEVecV4ConversionPatterns(patterns, flatScopes);
      configureAIEVecV4Legalizations(target);
    } else {
      llvm_unreachable("AIE version is misconfigured");
    }

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }
};

static std::unique_ptr<Pass>
createLowerVectorToAIEVec(const LowerVectorToAIEVecOptions &options) {
  return std::make_unique<LowerVectorToAIEVec>(options);
}

//===---------------------------------------------------------------------------
// Custom canonicalization passes
//===---------------------------------------------------------------------------

// This pass widens UPD ops to twice the width followed by an ext op of the
// bottom half. This can be used together with SimplifyUPDOpsPass to find
// additional common subexpressions with UPDs generated from unaligned
// `transfer_read` ops.
struct ExtendUPDOpsPass : PassWrapper<ExtendUPDOpsPass, OperationPass<>> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.add<ExpandUPDToUPDAndExtPattern>(patterns.getContext());
    target.addLegalDialect<aievec::AIEVecDialect>();
    target.addDynamicallyLegalOp<aievec::UPDOp>([](aievec::UPDOp op) {
      return op.getVector() ||
             (op->hasOneUse() && isa<aievec::UPDOp>(*op->getUsers().begin())) ||
             llvm::all_of(op->getUsers(),
                          [](Operation *op) { return isa<aievec::ExtOp>(op); });
    });

    if (auto *op = getOperation();
        failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

// This pass replaces wide UPD ops that are only used by a single ext op of
// the bottom half. This pass undos the work of ExtendUPDOpsPass.
// TODO: This pass can be extended to work with wide UPD ops that are used by
// TODO: a single ext op of the top half, which might be a good opportunity to
// TODO: further optimize wide UPDs.
struct SimplifyUPDOpsPass : PassWrapper<SimplifyUPDOpsPass, OperationPass<>> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.add<FuseExtIntoUPDPattern>(patterns.getContext());
    target.addLegalDialect<aievec::AIEVecDialect>();
    target.addDynamicallyLegalOp<aievec::ExtOp>([](aievec::ExtOp op) {
      auto *defOp = op.getSource().getDefiningOp();
      return !defOp || !isa<aievec::UPDOp>(defOp) || !defOp->hasOneUse() ||
             op.getIndex() != 0;
    });

    if (auto *op = getOperation();
        failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

//============================================================================//
//=============== Main Vector2AIEVec Pipeline Configuration ==================//
//============================================================================//

void xilinx::aievec::buildLowerVectorToAIEVec(
    OpPassManager &pm, const LowerVectorToAIEVecOptions &options) {
  // Add lowering from `Vector` to `AIEVec`
  pm.addPass(createLowerVectorToAIEVec(options));
  pm.addPass(createCanonicalizerPass());

  // Simplify UPD ops
  pm.addPass(std::make_unique<ExtendUPDOpsPass>());
  pm.addPass(createCSEPass());
  pm.addPass(std::make_unique<SimplifyUPDOpsPass>());
  pm.addPass(createCanonicalizerPass());
}
