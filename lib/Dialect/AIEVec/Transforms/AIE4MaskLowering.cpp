//===- AIE4MaskLowering.cpp - AIE4 tail predication -------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIE4 vector loads and stores access every lane and require alignment.
// Use them only for fully active, provably aligned accesses; otherwise use
// scalar accesses for active lanes. Arithmetic mask intrinsics do not
// predicate memory accesses.
//
// Transfers intersect nested masks with memory bounds. Prefix masks use
// bounded loops; arbitrary masks use packed bits with lane i in bit i.
// Masked reductions fill inactive lanes with the reduction identity.
// Lower create_mask to scalar arithmetic because Peano cannot legalize the
// vector compare emitted by generic lowering.
//
//===----------------------------------------------------------------------===//

#include "AIE4MaskLowering.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/Utils/TargetInfo.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace xilinx::aievec {
namespace {

// Masks up to 64 lanes use a single scalar carrier.
constexpr int64_t kMaxMaskLanes = 64;

bool isSupportedMaskType(VectorType ty) {
  return ty && !ty.isScalable() && ty.getRank() == 1 &&
         ty.getElementType().isInteger(1) && ty.getNumElements() >= 1 &&
         ty.getNumElements() <= kMaxMaskLanes;
}

IntegerType carrierType(OpBuilder &b, int64_t lanes) {
  return b.getIntegerType(lanes <= 32 ? 32 : 64);
}

uint64_t allOnes(int64_t lanes) {
  return lanes == 64 ? ~uint64_t(0) : (uint64_t(1) << lanes) - 1;
}

Value intConst(OpBuilder &b, Location loc, IntegerType ty, uint64_t v) {
  return arith::ConstantOp::create(
      b, loc, ty, b.getIntegerAttr(ty, APInt(ty.getWidth(), v)));
}

Value idxConst(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantIndexOp::create(b, loc, v);
}

// Clamp an index lane count to [0, lanes] in index arithmetic, before any
// narrowing cast.
Value clampCount(OpBuilder &b, Location loc, Value n, int64_t lanes) {
  Value c = arith::MaxSIOp::create(b, loc, n, idxConst(b, loc, 0));
  return arith::MinSIOp::create(b, loc, c, idxConst(b, loc, lanes));
}

// Bits for a prefix of `count` active lanes, count already in [0, lanes].
Value prefixBits(OpBuilder &b, Location loc, Value count, int64_t lanes,
                 IntegerType ty) {
  Value c = arith::IndexCastOp::create(b, loc, ty, count);
  Value one = intConst(b, loc, ty, 1);
  Value partial =
      arith::SubIOp::create(b, loc, arith::ShLIOp::create(b, loc, one, c), one);
  // A shift by the carrier width is poison; select the full mask instead.
  Value isFull = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, c,
                                       intConst(b, loc, ty, lanes));
  return arith::SelectOp::create(b, loc, isFull,
                                 intConst(b, loc, ty, allOnes(lanes)), partial);
}

// The active-lane count of a prefix mask, or null for other masks.
Value prefixCount(OpBuilder &b, Location loc, Value mask) {
  int64_t lanes = cast<VectorType>(mask.getType()).getNumElements();
  if (auto create = mask.getDefiningOp<vector::CreateMaskOp>())
    return clampCount(b, loc, create.getOperand(0), lanes);
  if (auto cst = mask.getDefiningOp<vector::ConstantMaskOp>())
    return idxConst(b, loc, std::min(cst.getMaskDimSizes()[0], lanes));
  return {};
}

// Pack a 1-D i1 mask into an integer carrier, lane i in bit i.
Value packMask(OpBuilder &b, Location loc, Value mask) {
  auto maskTy = cast<VectorType>(mask.getType());
  int64_t lanes = maskTy.getNumElements();
  IntegerType ty = carrierType(b, lanes);
  if (Value count = prefixCount(b, loc, mask))
    return prefixBits(b, loc, count, lanes, ty);
  auto laneBits = b.getIntegerType(lanes);
  Value word =
      vector::BitCastOp::create(b, loc, VectorType::get({1}, laneBits), mask);
  Value bits = vector::ExtractOp::create(b, loc, word, ArrayRef<int64_t>{0});
  if (laneBits != ty)
    bits = arith::ExtUIOp::create(b, loc, ty, bits);
  return bits;
}

// Reinterpret packed bits as a vector<lanes x i1> mask.
Value unpackMask(OpBuilder &b, Location loc, Value bits, int64_t lanes) {
  auto laneBits = b.getIntegerType(lanes);
  if (bits.getType() != laneBits)
    bits = arith::TruncIOp::create(b, loc, laneBits, bits);
  Value word =
      vector::BroadcastOp::create(b, loc, VectorType::get({1}, laneBits), bits);
  return vector::BitCastOp::create(
      b, loc, VectorType::get({lanes}, b.getI1Type()), word);
}

// Active lanes use an exclusive end `count` or packed `bits`, optionally
// skipping lanes before `begin` after transfer bounds checks.
struct ActiveLanes {
  Value count;
  Value bits;
  int64_t lanes;
  Value begin;

  Value isFull(OpBuilder &b, Location loc) const {
    Value full;
    if (count)
      full = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, count,
                                   idxConst(b, loc, lanes));
    else {
      auto ty = cast<IntegerType>(bits.getType());
      full = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bits,
                                   intConst(b, loc, ty, allOnes(lanes)));
    }
    if (begin) {
      Value startsAtZero = arith::CmpIOp::create(
          b, loc, arith::CmpIPredicate::eq, begin, idxConst(b, loc, 0));
      full = arith::AndIOp::create(b, loc, full, startsAtZero);
    }
    return full;
  }

  Value laneOn(OpBuilder &b, Location loc, Value lane) const {
    auto ty = cast<IntegerType>(bits.getType());
    Value shift = arith::IndexCastOp::create(b, loc, ty, lane);
    Value bit = arith::AndIOp::create(
        b, loc, arith::ShRUIOp::create(b, loc, bits, shift),
        intConst(b, loc, ty, 1));
    return arith::TruncIOp::create(b, loc, b.getI1Type(), bit);
  }

  // Emit `body` for every active lane, threading an optional value.
  Value
  forEach(OpBuilder &b, Location loc, Value init,
          function_ref<Value(OpBuilder &, Location, Value, Value)> body) const {
    SmallVector<Value> inits;
    if (init)
      inits.push_back(init);
    Value ub = count ? count : idxConst(b, loc, lanes);
    auto loop = scf::ForOp::create(
        b, loc, begin ? begin : idxConst(b, loc, 0), ub, idxConst(b, loc, 1),
        inits, [&](OpBuilder &fb, Location fl, Value lane, ValueRange acc) {
          Value in = init ? acc[0] : Value();
          if (count) {
            // A prefix needs no per-lane test.
            Value out = body(fb, fl, lane, in);
            scf::YieldOp::create(fb, fl, init ? ValueRange{out} : ValueRange{});
            return;
          }
          auto laneIf = scf::IfOp::create(
              fb, fl, init ? TypeRange{init.getType()} : TypeRange{},
              laneOn(fb, fl, lane), /*withElseRegion=*/bool(init));
          {
            OpBuilder::InsertionGuard g(fb);
            fb.setInsertionPointToStart(&laneIf.getThenRegion().front());
            Value out = body(fb, fl, lane, in);
            if (init) {
              scf::YieldOp::create(fb, fl, out);
              fb.setInsertionPointToStart(&laneIf.getElseRegion().front());
              scf::YieldOp::create(fb, fl, in);
            }
          }
          scf::YieldOp::create(fb, fl, laneIf.getResults());
        });
    return init ? loop.getResult(0) : Value();
  }
};

// True if `v` is provably a multiple of `m`.
bool isMultipleOf(Value v, int64_t m) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue() % m == 0;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *owner = arg.getOwner()->getParentOp();
    if (auto f = dyn_cast<scf::ForOp>(owner))
      return f.getInductionVar() == arg && isMultipleOf(f.getLowerBound(), m) &&
             isMultipleOf(f.getStep(), m);
    if (auto f = dyn_cast<affine::AffineForOp>(owner))
      return f.getInductionVar() == arg && f.hasConstantLowerBound() &&
             f.getConstantLowerBound() % m == 0 && f.getStepAsInt() % m == 0;
    return false;
  }
  Operation *def = v.getDefiningOp();
  if (auto add = dyn_cast_or_null<arith::AddIOp>(def))
    return isMultipleOf(add.getLhs(), m) && isMultipleOf(add.getRhs(), m);
  if (auto mul = dyn_cast_or_null<arith::MulIOp>(def))
    return isMultipleOf(mul.getLhs(), m) || isMultipleOf(mul.getRhs(), m);
  if (auto cast = dyn_cast_or_null<arith::IndexCastOp>(def))
    return isMultipleOf(cast.getIn(), m);
  return false;
}

// Prove alignment from the layout and indices: the last stride must be one,
// and the offset, outer strides, and last index must be multiples of `lanes`.
// The memref base is assumed vector-aligned, as for other AIE4 vector accesses.
bool isProvablyAligned(Value base, ValueRange indices, int64_t lanes) {
  auto memTy = cast<MemRefType>(base.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memTy.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) || offset % lanes != 0 ||
      strides.back() != 1)
    return false;
  for (int64_t s : ArrayRef<int64_t>(strides).drop_back())
    if (ShapedType::isDynamic(s) || s % lanes != 0)
      return false;
  return isMultipleOf(indices.back(), lanes);
}

SmallVector<Value> laneIndices(OpBuilder &b, Location loc, ValueRange indices,
                               Value lane) {
  SmallVector<Value> idx(indices.begin(), indices.end());
  idx.back() = arith::AddIOp::create(b, loc, idx.back(), lane);
  return idx;
}

// Load active lanes along the last memref dimension into `init`;
// other lanes keep `init` and are not read.
Value emitMaskedLoad(OpBuilder &b, Location loc, VectorType vecTy, Value base,
                     ValueRange indices, const ActiveLanes &active,
                     Value init) {
  auto scalarPath = [&](OpBuilder &sb, Location l) {
    return active.forEach(
        sb, l, init, [&](OpBuilder &ab, Location al, Value lane, Value acc) {
          Value x = memref::LoadOp::create(ab, al, base,
                                           laneIndices(ab, al, indices, lane));
          return vector::InsertOp::create(ab, al, x, acc, OpFoldResult(lane))
              .getResult();
        });
  };
  if (!isProvablyAligned(base, indices, vecTy.getNumElements()))
    return scalarPath(b, loc);
  auto ifOp = scf::IfOp::create(
      b, loc, active.isFull(b, loc),
      [&](OpBuilder &tb, Location l) {
        Value v = vector::LoadOp::create(tb, l, vecTy, base, indices);
        scf::YieldOp::create(tb, l, v);
      },
      [&](OpBuilder &eb, Location l) {
        scf::YieldOp::create(eb, l, scalarPath(eb, l));
      });
  return ifOp.getResult(0);
}

// Store the active lanes of `value`; other lanes are never written.
void emitMaskedStore(OpBuilder &b, Location loc, Value value, Value base,
                     ValueRange indices, const ActiveLanes &active) {
  auto scalarPath = [&](OpBuilder &sb, Location l) {
    active.forEach(
        sb, l, Value(), [&](OpBuilder &ab, Location al, Value lane, Value) {
          Value x =
              vector::ExtractOp::create(ab, al, value, OpFoldResult(lane));
          memref::StoreOp::create(ab, al, x, base,
                                  laneIndices(ab, al, indices, lane));
          return Value();
        });
  };
  auto vecTy = cast<VectorType>(value.getType());
  if (!isProvablyAligned(base, indices, vecTy.getNumElements())) {
    scalarPath(b, loc);
    return;
  }
  scf::IfOp::create(
      b, loc, active.isFull(b, loc),
      [&](OpBuilder &tb, Location l) {
        vector::StoreOp::create(tb, l, value, base, indices);
        scf::YieldOp::create(tb, l);
      },
      [&](OpBuilder &eb, Location l) {
        scalarPath(eb, l);
        scf::YieldOp::create(eb, l);
      });
}

ActiveLanes maskLanes(OpBuilder &b, Location loc, Value mask, int64_t lanes) {
  if (!mask)
    return {idxConst(b, loc, lanes), Value(), lanes};
  if (Value count = prefixCount(b, loc, mask))
    return {count, Value(), lanes};
  return {Value(), packMask(b, loc, mask), lanes};
}

// Require a rank-1 vector transfer along the last memref dimension.
LogicalResult checkTransfer(VectorTransferOpInterface op, VectorType vecTy,
                            StringRef what) {
  if (!isa<MemRefType>(op.getShapedType()))
    return op->emitError() << "AIE4: masked " << what
                           << " on a tensor has no tail-safe lowering";
  if (vecTy.isScalable() || vecTy.getRank() != 1 ||
      vecTy.getNumElements() > kMaxMaskLanes)
    return op->emitError() << "AIE4: masked " << what
                           << " requires a fixed 1-D vector of at most 64 "
                              "lanes";
  if (!op.getPermutationMap().isMinorIdentity())
    return op->emitError() << "AIE4: masked " << what
                           << " requires a minor identity permutation map";
  return success();
}

// Intersect masks without losing the prefix-loop optimization.
ActiveLanes intersectMasks(OpBuilder &b, Location loc, Value outer, Value inner,
                           int64_t lanes) {
  if (!outer || outer == inner)
    return maskLanes(b, loc, inner, lanes);
  if (!inner)
    return maskLanes(b, loc, outer, lanes);
  ActiveLanes a = maskLanes(b, loc, outer, lanes);
  ActiveLanes c = maskLanes(b, loc, inner, lanes);
  if (a.count && c.count)
    return {arith::MinSIOp::create(b, loc, a.count, c.count), Value(), lanes};
  IntegerType ty = carrierType(b, lanes);
  Value aBits = a.count ? prefixBits(b, loc, a.count, lanes, ty) : a.bits;
  Value cBits = c.count ? prefixBits(b, loc, c.count, lanes, ty) : c.bits;
  return {Value(), arith::AndIOp::create(b, loc, aBits, cBits), lanes};
}

// Limit both masks to the in-bounds lane interval, preserving lane positions.
ActiveLanes transferLanes(OpBuilder &b, Location loc,
                          VectorTransferOpInterface op, Value outerMask,
                          int64_t lanes) {
  ActiveLanes active = intersectMasks(b, loc, outerMask, op.getMask(), lanes);
  if (op.isDimInBounds(0))
    return active;
  unsigned dim = cast<MemRefType>(op.getShapedType()).getRank() - 1;
  Value size = memref::DimOp::create(b, loc, op.getBase(), dim);
  Value start = op.getIndices().back();
  Value zero = idxConst(b, loc, 0);
  // Bound the negative part before negation; subtract only nonnegative
  // indices from size so even extreme out-of-bounds indices cannot overflow.
  Value negativeStart = arith::MinSIOp::create(
      b, loc, arith::MaxSIOp::create(b, loc, start, idxConst(b, loc, -lanes)),
      zero);
  active.begin = arith::SubIOp::create(b, loc, zero, negativeStart);
  Value positiveStart = arith::MaxSIOp::create(b, loc, start, zero);
  Value remaining = clampCount(
      b, loc, arith::SubIOp::create(b, loc, size, positiveStart), lanes);
  Value end = arith::MinSIOp::create(
      b, loc, arith::AddIOp::create(b, loc, remaining, active.begin),
      idxConst(b, loc, lanes));
  if (active.count) {
    active.count = arith::MinSIOp::create(b, loc, active.count, end);
    return active;
  }
  active.bits = arith::AndIOp::create(
      b, loc, active.bits,
      prefixBits(b, loc, end, lanes, carrierType(b, lanes)));
  return active;
}

// Select float lanes through integer bitcasts to use the VSEL lowering.
Value selectLanes(OpBuilder &b, Location loc, Value mask, Value v,
                  Value other) {
  auto vecTy = cast<VectorType>(v.getType());
  Type elTy = vecTy.getElementType();
  if (!isa<FloatType>(elTy))
    return arith::SelectOp::create(b, loc, mask, v, other);
  auto intVecTy = VectorType::get(
      vecTy.getShape(), b.getIntegerType(elTy.getIntOrFloatBitWidth()));
  Value vi = vector::BitCastOp::create(b, loc, intVecTy, v);
  Value oi = vector::BitCastOp::create(b, loc, intVecTy, other);
  Value r = arith::SelectOp::create(b, loc, mask, vi, oi);
  return vector::BitCastOp::create(b, loc, vecTy, r);
}

LogicalResult lowerRead(IRRewriter &rw, Operation *root,
                        vector::TransferReadOp read, Value mask,
                        Value passthru) {
  VectorType vecTy = read.getVectorType();
  if (failed(checkTransfer(read, vecTy, "vector.transfer_read")))
    return failure();
  Location loc = read.getLoc();
  rw.setInsertionPoint(root);
  ActiveLanes active =
      transferLanes(rw, loc, read, mask, vecTy.getNumElements());
  // Lanes disabled by the outer mask use passthru when provided;
  // all other unloaded lanes keep padding.
  Value init = vector::BroadcastOp::create(rw, loc, vecTy, read.getPadding());
  if (passthru)
    init = read.isDimInBounds(0) && !read.getMask()
               ? passthru
               : selectLanes(rw, loc, mask, init, passthru);
  Value v = emitMaskedLoad(rw, loc, vecTy, read.getBase(), read.getIndices(),
                           active, init);
  rw.replaceOp(root, v);
  return success();
}

LogicalResult lowerWrite(IRRewriter &rw, Operation *root,
                         vector::TransferWriteOp write, Value mask) {
  VectorType vecTy = write.getVectorType();
  if (failed(checkTransfer(write, vecTy, "vector.transfer_write")))
    return failure();
  Location loc = write.getLoc();
  rw.setInsertionPoint(root);
  ActiveLanes active =
      transferLanes(rw, loc, write, mask, vecTy.getNumElements());
  emitMaskedStore(rw, loc, write.getVector(), write.getBase(),
                  write.getIndices(), active);
  rw.eraseOp(root);
  return success();
}

// Identity of a combining kind for element type `ty`, or null if none can
// be materialized.
TypedAttr reductionIdentity(OpBuilder &b, vector::CombiningKind kind, Type ty) {
  using K = vector::CombiningKind;
  if (auto ft = dyn_cast<FloatType>(ty)) {
    const llvm::fltSemantics &sem = ft.getFloatSemantics();
    switch (kind) {
    case K::ADD:
      return b.getFloatAttr(ft, APFloat::getZero(sem, /*Negative=*/true));
    case K::MUL:
      return b.getFloatAttr(ft, APFloat::getOne(sem));
    case K::MINIMUMF:
      return b.getFloatAttr(ft, APFloat::getInf(sem));
    case K::MAXIMUMF:
      return b.getFloatAttr(ft, APFloat::getInf(sem, /*Negative=*/true));
    default:
      // minnumf/maxnumf need a NaN identity (an all-NaN active set must give
      // NaN), which Peano cannot parse as a constant.
      return {};
    }
  }
  auto it = dyn_cast<IntegerType>(ty);
  if (!it)
    return {};
  unsigned w = it.getWidth();
  switch (kind) {
  case K::ADD:
  case K::OR:
  case K::XOR:
  case K::MAXUI:
    return b.getIntegerAttr(it, APInt::getZero(w));
  case K::MUL:
    return b.getIntegerAttr(it, APInt(w, 1));
  case K::AND:
  case K::MINUI:
    return b.getIntegerAttr(it, APInt::getAllOnes(w));
  case K::MINSI:
    return b.getIntegerAttr(it, APInt::getSignedMaxValue(w));
  case K::MAXSI:
    return b.getIntegerAttr(it, APInt::getSignedMinValue(w));
  default:
    return {};
  }
}

LogicalResult lowerReduction(IRRewriter &rw, vector::MaskOp maskOp,
                             Value source, vector::CombiningKind kind,
                             Value acc, arith::FastMathFlags fmf) {
  // The vector.mask verifier rejects a passthru on reductions.
  auto vecTy = cast<VectorType>(source.getType());
  if (vecTy.isScalable() || vecTy.getRank() != 1)
    return maskOp.emitError()
           << "AIE4: masked reductions require a fixed 1-D vector";
  TypedAttr id = reductionIdentity(rw, kind, vecTy.getElementType());
  if (!id)
    return maskOp.emitError()
           << "AIE4: masked " << vector::stringifyCombiningKind(kind)
           << " reduction has no supported identity for "
           << vecTy.getElementType();
  Location loc = maskOp.getLoc();
  rw.setInsertionPoint(maskOp);
  Value identity = arith::ConstantOp::create(rw, loc, vecTy,
                                             DenseElementsAttr::get(vecTy, id));
  Value selected = selectLanes(rw, loc, maskOp.getMask(), source, identity);
  auto red = vector::ReductionOp::create(rw, loc, kind, selected, acc, fmf);
  rw.replaceOp(maskOp, red.getResult());
  return success();
}

LogicalResult lowerMaskOp(IRRewriter &rw, vector::MaskOp maskOp) {
  Operation *inner = maskOp.getMaskableOp();
  if (!inner)
    return maskOp.emitError() << "AIE4: empty vector.mask is not supported";
  Value mask = maskOp.getMask();
  if (!isSupportedMaskType(cast<VectorType>(mask.getType())))
    return maskOp.emitError()
           << "AIE4: masks must be fixed 1-D with at most 64 lanes";

  if (auto read = dyn_cast<vector::TransferReadOp>(inner))
    return lowerRead(rw, maskOp, read, mask,
                     maskOp.hasPassthru() ? maskOp.getPassthru() : Value());
  if (auto write = dyn_cast<vector::TransferWriteOp>(inner)) {
    if (write.hasPureTensorSemantics())
      return maskOp.emitError()
             << "AIE4: masked vector.transfer_write on a tensor has no "
                "tail-safe lowering";
    return lowerWrite(rw, maskOp, write, mask);
  }
  if (auto red = dyn_cast<vector::ReductionOp>(inner))
    return lowerReduction(rw, maskOp, red.getVector(), red.getKind(),
                          red.getAcc(), red.getFastmath());
  if (auto red = dyn_cast<vector::MultiDimReductionOp>(inner)) {
    auto srcTy = cast<VectorType>(red.getSource().getType());
    if (srcTy.getRank() == 1 && !isa<VectorType>(red.getType()))
      return lowerReduction(rw, maskOp, red.getSource(), red.getKind(),
                            red.getAcc(), arith::FastMathFlags::none);
  }
  return maskOp.emitError() << "AIE4: masked '" << inner->getName()
                            << "' has no tail-safe lowering";
}

struct AIE4LowerVectorMasksPass
    : PassWrapper<AIE4LowerVectorMasksPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIE4LowerVectorMasksPass)

  AIE4LowerVectorMasksPass() = default;
  AIE4LowerVectorMasksPass(const AIE4LowerVectorMasksPass &pass)
      : PassWrapper(pass) {}
  explicit AIE4LowerVectorMasksPass(StringRef target) {
    aieTarget = target.str();
  }

  StringRef getArgument() const final { return "aie4-lower-vector-masks"; }
  StringRef getDescription() const final {
    return "Lower masked vector operations to tail-safe AIE4 code";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc("AIE target; only AIE4 targets are affected"),
      llvm::cl::init("aie4b")};

  void runOnOperation() override {
    // Other targets, and unknown spellings (reported by the target
    // lowering), are left alone.
    std::string error;
    std::optional<AIEVecTargetInfo> info = parseAIEVecTarget(aieTarget, error);
    if (!info || !info->isAIE4())
      return;
    if (failed(lowerAIE4Masks(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult lowerAIE4Masks(Operation *root) {
  SmallVector<Operation *> work;
  bool failedAny = false;
  // Pre-order, so that skipping a runtime sequence skips its body.
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // Runtime sequences are host control code, not core compute.
    if (isa<xilinx::AIE::RuntimeSequenceOp>(op))
      return WalkResult::skip();
    if (isa<vector::MaskOp, vector::MaskedLoadOp, vector::MaskedStoreOp,
            vector::CreateMaskOp>(op))
      work.push_back(op);
    else if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      if (read.getMask() && !op->getParentOfType<vector::MaskOp>())
        work.push_back(op);
    } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      if (write.getMask() && !op->getParentOfType<vector::MaskOp>())
        work.push_back(op);
    } else if (isa<vector::GatherOp, vector::ScatterOp, vector::ExpandLoadOp,
                   vector::CompressStoreOp>(op)) {
      op->emitError() << "AIE4: '" << op->getName()
                      << "' has no tail-safe lowering";
      failedAny = true;
    }
    return WalkResult::advance();
  });
  if (failedAny)
    return failure();

  IRRewriter rw(root->getContext());
  // Masked operations read create_mask operands directly; lower the
  // create_mask ops that still have users last.
  SmallVector<vector::CreateMaskOp> creates;
  for (Operation *op : work) {
    LogicalResult r = success();
    if (auto maskOp = dyn_cast<vector::MaskOp>(op)) {
      r = lowerMaskOp(rw, maskOp);
    } else if (auto load = dyn_cast<vector::MaskedLoadOp>(op)) {
      VectorType vecTy = load.getVectorType();
      if (!isSupportedMaskType(load.getMaskVectorType()) ||
          vecTy.getRank() != 1) {
        r = load.emitError() << "AIE4: vector.maskedload requires a fixed "
                                "1-D vector of at most 64 lanes";
      } else {
        rw.setInsertionPoint(load);
        Location loc = load.getLoc();
        ActiveLanes active =
            maskLanes(rw, loc, load.getMask(), vecTy.getNumElements());
        rw.replaceOp(load, emitMaskedLoad(rw, loc, vecTy, load.getBase(),
                                          load.getIndices(), active,
                                          load.getPassThru()));
      }
    } else if (auto store = dyn_cast<vector::MaskedStoreOp>(op)) {
      if (!isSupportedMaskType(store.getMaskVectorType()) ||
          store.getVectorType().getRank() != 1) {
        r = store.emitError() << "AIE4: vector.maskedstore requires a fixed "
                                 "1-D vector of at most 64 lanes";
      } else {
        rw.setInsertionPoint(store);
        Location loc = store.getLoc();
        ActiveLanes active = maskLanes(rw, loc, store.getMask(),
                                       store.getVectorType().getNumElements());
        emitMaskedStore(rw, loc, store.getValueToStore(), store.getBase(),
                        store.getIndices(), active);
        rw.eraseOp(store);
      }
    } else if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      r = lowerRead(rw, read, read, Value(), Value());
    } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      r = lowerWrite(rw, write, write, Value());
    } else if (auto create = dyn_cast<vector::CreateMaskOp>(op)) {
      creates.push_back(create);
    }
    if (failed(r))
      return failure();
  }

  for (vector::CreateMaskOp create : creates) {
    if (create->use_empty()) {
      rw.eraseOp(create);
      continue;
    }
    auto maskTy = create.getVectorType();
    if (!isSupportedMaskType(maskTy))
      return create.emitError()
             << "AIE4: vector.create_mask requires a fixed 1-D mask of at "
                "most 64 lanes";
    rw.setInsertionPoint(create);
    Location loc = create.getLoc();
    int64_t lanes = maskTy.getNumElements();
    Value count = clampCount(rw, loc, create.getOperand(0), lanes);
    Value bits = prefixBits(rw, loc, count, lanes, carrierType(rw, lanes));
    rw.replaceOp(create, unpackMask(rw, loc, bits, lanes));
  }
  return success();
}

std::unique_ptr<Pass> createAIE4LowerVectorMasksPass(StringRef aieTarget) {
  return std::make_unique<AIE4LowerVectorMasksPass>(aieTarget);
}

void registerAIE4LowerVectorMasksPass() {
  PassRegistration<AIE4LowerVectorMasksPass>();
}

} // namespace xilinx::aievec
