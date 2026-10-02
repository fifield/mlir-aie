//===- BdLowering.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Iterators.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <numeric>

using namespace mlir;

namespace xilinx::AIEX {

//===----------------------------------------------------------------------===//
// SsaStridePolicy: arith-emitting mirror of ConstStridePolicy.
//
// Arithmetic is i64, like ConstStridePolicy. Operands arrive zero-extended, so
// the comparisons are unsigned; a negative runtime value reads as a huge one,
// which the host-side guards in encodeBdCommon reject before the encoded
// fields are used.
//===----------------------------------------------------------------------===//

Value SsaStridePolicy::cst(int64_t c) const {
  return arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(c));
}

Value SsaStridePolicy::mul(Value v, int64_t c) const {
  return builder.createOrFold<arith::MulIOp>(loc, v, cst(c));
}

Value SsaStridePolicy::div(Value v, int64_t c) const {
  // Hardware granularity scaling is exact (verifier/guards enforce
  // divisibility).
  return builder.createOrFold<arith::DivUIOp>(loc, v, cst(c));
}

Value SsaStridePolicy::sub(Value v, int64_t c) const {
  return builder.createOrFold<arith::SubIOp>(loc, v, cst(c));
}

Value SsaStridePolicy::selectGT1(Value cond, Value t, Value e) const {
  Value gt = builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                 cond, cst(1));
  return builder.createOrFold<arith::SelectOp>(loc, gt, t, e);
}

Value SsaStridePolicy::selectGT0(Value cond, Value t, Value e) const {
  Value gt = builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                 cond, cst(0));
  return builder.createOrFold<arith::SelectOp>(loc, gt, t, e);
}

Value SsaStridePolicy::selectLt(Value a, Value b, Value t, Value e) const {
  Value lt =
      builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, a, b);
  return builder.createOrFold<arith::SelectOp>(loc, lt, t, e);
}

//===----------------------------------------------------------------------===//
// Host-side guards.
//===----------------------------------------------------------------------===//

LogicalResult emitRuntimeCheck(OpBuilder &builder, Location loc, Value cond,
                               const Twine &message) {
  if (auto c = getConstantIntValue(cond)) {
    if (*c)
      return success();
    return emitError(loc) << message;
  }
  cf::AssertOp::create(builder, loc, cond, builder.getStringAttr(message));
  return success();
}

void eraseDeadArith(Operation *root) {
  root->walk<WalkOrder::PostOrder, ReverseIterator>([](Operation *op) {
    if (isa<arith::ArithDialect>(op->getDialect()) && isOpTriviallyDead(op))
      op->erase();
  });
}

namespace {

// Folding unsigned i64 arithmetic for building guard conditions: over
// constant operands a condition folds to a constant, which emitRuntimeCheck
// drops (true) or reports at compile time (false).
// Arguments that build ops go in separate statements: C++ leaves their
// evaluation order, and so the emitted op order, to the compiler.
struct GuardBuilder {
  OpBuilder &b;
  Location loc;

  Value cst(uint64_t c) {
    return arith::ConstantOp::create(b, loc, b.getI64IntegerAttr((int64_t)c));
  }
  Value add(Value x, Value y) {
    return b.createOrFold<arith::AddIOp>(loc, x, y);
  }
  Value sub(Value x, uint64_t c) {
    return b.createOrFold<arith::SubIOp>(loc, x, cst(c));
  }
  Value mul(Value x, Value y) {
    return b.createOrFold<arith::MulIOp>(loc, x, y);
  }
  Value both(Value x, Value y) {
    return b.createOrFold<arith::AndIOp>(loc, x, y);
  }
  Value either(Value x, Value y) {
    return b.createOrFold<arith::OrIOp>(loc, x, y);
  }
  Value cmp(arith::CmpIPredicate pred, Value x, Value y) {
    return b.createOrFold<arith::CmpIOp>(loc, pred, x, y);
  }
  Value eq(Value x, Value y) { return cmp(arith::CmpIPredicate::eq, x, y); }
  Value eq(Value x, uint64_t c) { return eq(x, cst(c)); }
  Value ule(Value x, uint64_t c) {
    return cmp(arith::CmpIPredicate::ule, x, cst(c));
  }
  // 1 <= x <= hi as one unsigned compare: x - 1 wraps 0 to UINT64_MAX.
  Value inRange1(Value x, uint64_t hi) { return ule(sub(x, 1), hi - 1); }
  Value multipleOf(Value x, uint64_t d) {
    return eq(b.createOrFold<arith::RemUIOp>(loc, x, cst(d)), 0);
  }
  LogicalResult check(Value cond, const Twine &message) {
    return emitRuntimeCheck(b, loc, cond, message);
  }
};

} // namespace

Value getAsI64(OpBuilder &builder, Location loc, OpFoldResult ofr) {
  Type i64 = builder.getI64Type();
  if (auto c = getConstantIntValue(ofr))
    return Value(
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(i64, *c)));
  Value v = cast<Value>(ofr);
  if (v.getType().isIndex())
    return Value(arith::IndexCastUIOp::create(builder, loc, i64, v));
  unsigned bits = v.getType().getIntOrFloatBitWidth();
  if (bits < 64)
    return Value(arith::ExtUIOp::create(builder, loc, i64, v));
  if (bits > 64) {
    GuardBuilder g{builder, loc};
    Value max = arith::ConstantOp::create(
        builder, loc,
        IntegerAttr::get(v.getType(), APInt::getMaxValue(64).zext(bits)));
    if (failed(g.check(g.cmp(arith::CmpIPredicate::ule, v, max),
                       "a runtime DMA operand does not fit in 64 bits")))
      return {};
    return Value(arith::TruncIOp::create(builder, loc, i64, v));
  }
  return v;
}

//===----------------------------------------------------------------------===//
// Layout helpers.
//===----------------------------------------------------------------------===//

Value getAsValue(OpBuilder &builder, Location loc, OpFoldResult ofr,
                 Type intType) {
  if (auto constVal = getConstantIntValue(ofr))
    return arith::ConstantOp::create(builder, loc,
                                     IntegerAttr::get(intType, *constVal));
  Value val = cast<Value>(ofr);
  if (val.getType() != intType) {
    if (val.getType().isIndex())
      return arith::IndexCastUIOp::create(builder, loc, intType, val);
    unsigned valBits = val.getType().getIntOrFloatBitWidth();
    unsigned tgtBits = intType.getIntOrFloatBitWidth();
    if (valBits > tgtBits)
      val = arith::TruncIOp::create(builder, loc, intType, val);
    else
      val = arith::ExtUIOp::create(builder, loc, intType, val);
  }
  return val;
}

Value buildArgPlusValue(OpBuilder &builder, Location loc,
                        ArrayRef<OpFoldResult> elementOffsets,
                        ArrayRef<OpFoldResult> strides, int64_t elemWidthBytes,
                        int64_t baseByteOffset, uint32_t granuleBytes) {
  auto i32ty = builder.getIntegerType(32);

  // Fast path: everything constant -> fold to one arith.constant, matching the
  // static lowering byte-for-byte.
  bool allConst = llvm::all_of(elementOffsets,
                               [](OpFoldResult o) {
                                 return getConstantIntValue(o).has_value();
                               }) &&
                  llvm::all_of(strides, [](OpFoldResult s) {
                    return getConstantIntValue(s).has_value();
                  });
  if (allConst) {
    int64_t bytes = baseByteOffset;
    for (auto [o, s] : llvm::zip(elementOffsets, strides)) {
      auto oc = getConstantIntValue(o);
      auto sc = getConstantIntValue(s);
      assert(oc && sc && "allConst already verified these are constant");
      bytes += (*oc) * (*sc) * elemWidthBytes;
    }
    if (bytes > std::numeric_limits<uint32_t>::max() || bytes < 0)
      return Value(arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(builder.getIntegerType(64), bytes)));
    return Value(arith::ConstantOp::create(builder, loc,
                                           IntegerAttr::get(i32ty, bytes)));
  }

  // Runtime path, in i64: sum(offset[i] * stride[i]) * elemWidthBytes + base.
  // The range is the bounds guard's job (guardWithinHostBuffer); the patch
  // itself only needs the granule alignment the static verifier checks.
  GuardBuilder g{builder, loc};
  Value acc = g.cst(0);
  // A byte offset every term is a multiple of; when it is a whole granule the
  // alignment guard is statically true.
  uint64_t knownMultiple = (uint64_t)baseByteOffset;
  for (auto [o, s] : llvm::zip(elementOffsets, strides)) {
    Value ov = getAsI64(builder, loc, o);
    Value sv = getAsI64(builder, loc, s);
    if (!ov || !sv)
      return {};
    acc = g.add(acc, g.mul(ov, sv));
    uint64_t factor = (uint64_t)elemWidthBytes;
    for (OpFoldResult f : {o, s})
      if (auto c = getConstantIntValue(f))
        factor *= (uint64_t)*c;
    knownMultiple = std::gcd(knownMultiple, factor);
  }
  acc = g.mul(acc, g.cst(elemWidthBytes));
  if (baseByteOffset != 0)
    acc = g.add(acc, g.cst(baseByteOffset));
  if (granuleBytes > 1 && knownMultiple % granuleBytes != 0 &&
      failed(g.check(g.eq(builder.createOrFold<arith::AndIOp>(
                              loc, acc, g.cst(granuleBytes - 1)),
                          0),
                     "a runtime DMA offset is not " + Twine(granuleBytes) +
                         "-byte aligned")))
    return {};
  return acc;
}

LogicalResult guardWithinHostBuffer(
    OpBuilder &builder, Location loc, BaseMemRefType hostBufferType,
    int64_t baseByteOffset, int64_t elemWidthBytes,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> offsetStrides,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) {
  // Elements addressable past the base offset. Capped at 2^32 so every term
  // below is a product of two values under 2^32 and no sum can wrap 64 bits.
  constexpr uint64_t kMaxElems = 1ULL << 32;
  uint64_t cap = kMaxElems;
  if (hostBufferType.hasStaticShape()) {
    int64_t bytes = hostBufferType.getNumElements() *
                    (int64_t)hostBufferType.getElementTypeBitWidth() / 8;
    cap = bytes > baseByteOffset
              ? std::min<uint64_t>((bytes - baseByteOffset) / elemWidthBytes,
                                   kMaxElems)
              : 0;
  }
  if (cap == 0)
    return emitError(loc) << "DMA transfer starts past the end of its host "
                             "buffer";

  // The furthest element touched is sum(offset_k * offsetStride_k) +
  // sum((size_i - 1) * stride_i); it must be below `cap`. A term is in range
  // only if both factors are (a zero factor makes the other irrelevant), which
  // also keeps the product from wrapping.
  GuardBuilder g{builder, loc};
  Value ok, sum = g.cst(0);
  auto addTerm = [&](Value c, Value t) {
    Value cZero = g.eq(c, 0);
    Value tFits = g.either(cZero, g.ule(t, cap - 1));
    Value tZero = g.eq(t, 0);
    Value inRange = g.both(tFits, g.either(tZero, g.ule(c, cap - 1)));
    ok = ok ? g.both(ok, inRange) : inRange;
    sum = g.add(sum, g.mul(c, t));
  };
  for (auto [o, s] : llvm::zip(offsets, offsetStrides)) {
    Value ov = getAsI64(builder, loc, o);
    Value sv = getAsI64(builder, loc, s);
    if (!ov || !sv)
      return failure();
    addTerm(ov, sv);
  }
  for (auto [sz, st] : llvm::zip(sizes, strides)) {
    Value szv = getAsI64(builder, loc, sz);
    Value stv = getAsI64(builder, loc, st);
    if (!szv || !stv)
      return failure();
    addTerm(g.sub(szv, 1), stv);
  }
  Value cond = ok ? g.both(ok, g.ule(sum, cap - 1)) : g.ule(sum, cap - 1);
  return g.check(cond, "a runtime DMA access runs past the end of its " +
                           Twine(cap) + "-element host buffer");
}

Value buildBdWord(OpBuilder &builder, Location loc,
                  ArrayRef<std::tuple<Value, uint32_t, uint32_t>> fields) {
  auto i32ty = builder.getIntegerType(32);
  Value result =
      arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, 0));
  for (auto &[val, mask, shift] : fields) {
    Value masked = val;
    if (mask != 0xFFFFFFFF) {
      auto maskConst = arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(i32ty, (int64_t)(int32_t)mask));
      masked = builder.createOrFold<arith::AndIOp>(loc, masked, maskConst);
    }
    if (shift > 0) {
      auto shiftConst = arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(i32ty, shift));
      masked = builder.createOrFold<arith::ShLIOp>(loc, masked, shiftConst);
    }
    result = builder.createOrFold<arith::OrIOp>(loc, result, masked);
  }
  return result;
}

Value getBdRegisterBase(OpBuilder &builder, Location loc,
                        const AIE::AIETargetModel &targetModel, int tileCol,
                        int tileRow, OpFoldResult bdId, int channel,
                        AIE::DMAChannelDir direction) {
  auto i32ty = builder.getIntegerType(32);
  if (auto c = getConstantIntValue(bdId))
    return createConstantI32(builder, loc,
                             static_cast<uint32_t>(targetModel.getDmaBdAddress(
                                 tileCol, tileRow, *c, channel, direction)));
  // Runtime bd_id: base + bd_id * bdStride, with base/stride from the (linear)
  // target-model address function.
  uint64_t addrForId0 =
      targetModel.getDmaBdAddress(tileCol, tileRow, 0, channel, direction);
  uint64_t bdStride =
      targetModel.getDmaBdAddress(tileCol, tileRow, 1, channel, direction) -
      addrForId0;
  Value bdIdVal = getAsValue(builder, loc, bdId, i32ty);
  return arith::AddIOp::create(
      builder, loc, createConstantI32(builder, loc, addrForId0),
      arith::MulIOp::create(builder, loc, bdIdVal,
                            createConstantI32(builder, loc, bdStride)));
}

namespace {

// Arrays are innermost-first: [d0, d1, d2, iter]. Every value is i32, ready to
// pack into its BD field.
struct EncodedBd {
  Value hwS[4], hwT[4];  // granule-scaled wraps and -1-biased steps
  Value bufLen;          // buffer_length, in address-gen granules
  Value iterSizeField;   // iteration_size, zeroed for a pure repeat
  Value repeatCount;     // outer-dim repeat for the caller's queue push
  bool isLinear = false; // d0/d1/d2 folded into buffer_length
};

LogicalResult encodeBdCommon(OpBuilder &builder, Location loc,
                             const AIE::AIETargetModel &tm, int tileCol,
                             int tileRow, ArrayRef<OpFoldResult> mixedSizes,
                             ArrayRef<OpFoldResult> mixedStrides,
                             uint64_t elemWidth, OpFoldResult lenElems,
                             EncodedBd &out) {
  auto i32ty = builder.getIntegerType(32);
  uint32_t gran = tm.getAddressGenGranularity();
  SmallVector<OpFoldResult, 4> sizesRev(llvm::reverse(mixedSizes));
  SmallVector<OpFoldResult, 4> stridesRev(llvm::reverse(mixedStrides));
  auto cst = [&](OpFoldResult v) { return getConstantIntValue(v); };
  auto constEq = [&](OpFoldResult v, int64_t c) {
    auto k = cst(v);
    return k && *k == c;
  };
  // A transfer with no data layout transformation at all (mirrors
  // AIEX::isLinearTransfer). Canonicalization zeroes size-1 strides before this
  // runs, hence the stride == 0 tests.
  auto knownLinear = [&]() {
    return constEq(sizesRev[1], 1) && constEq(sizesRev[2], 1) &&
           constEq(stridesRev[0], 1) && constEq(stridesRev[1], 0) &&
           constEq(stridesRev[2], 0);
  };
  // A contiguous row-major scan: each outer stride is the product of the inner
  // sizes (mirrors AIEX::isContiguousTransfer). A runtime operand needed for
  // the test falls back to ND mode.
  auto knownContiguous = [&]() -> bool {
    auto s0 = cst(stridesRev[0]);
    if (!s0 || *s0 != 1)
      return false;
    auto sz0 = cst(sizesRev[0]);
    auto d1sz = cst(sizesRev[1]);
    auto d1st = cst(stridesRev[1]);
    if (!d1sz || *d1sz != 1)
      if (!sz0 || !d1st || *d1st != *sz0)
        return false;
    auto d2sz = cst(sizesRev[2]);
    auto d2st = cst(stridesRev[2]);
    if (!d2sz || *d2sz != 1) {
      auto prod01 =
          (sz0 && d1sz) ? std::optional<int64_t>(*sz0 * *d1sz) : std::nullopt;
      if (!prod01 || !d2st || *d2st != *prod01)
        return false;
    }
    return true;
  };
  // Folding d0/d1/d2 into buffer_length dodges the 10-bit d0 wrap, but only a
  // shim NOC tile's 32-bit length field can absorb a merely-contiguous scan
  // (17 bits on a mem tile, 14 on a core tile). The static path draws the line
  // in the same place, in AIEDMATasksToNPU.cpp's treatAsLinear.
  out.isLinear = knownLinear() ||
                 (tm.isShimNOCTile(tileCol, tileRow) && knownContiguous());
  bool isLinear = out.isLinear;

  Value inS[4], inT[4];
  for (int i = 0; i < 4; i++) {
    inS[i] = getAsI64(builder, loc, sizesRev[i]);
    inT[i] = getAsI64(builder, loc, stridesRev[i]);
    if (!inS[i] || !inT[i])
      return failure();
  }

  // Host-side guards: every size and stride must land in its BD field, checked
  // in the element domain before any scaling can wrap. Over constant operands
  // they fold away (the verifiers already checked those) or fail here.
  uint64_t ew = elemWidth;
  uint64_t wrapMax = (1ULL << tm.getDmaBdWrapBits(tileCol, tileRow)) - 1;
  uint64_t maxLen = tm.getDmaBdMaxLen(tileCol, tileRow);
  uint64_t maxStride =
      (1ULL << tm.getDmaBdStepBits(tileCol, tileRow)) * gran / ew;
  uint64_t maxIterations = 1ULL << tm.getDmaBdIterBits(tileCol, tileRow);
  uint64_t maxRepeats = tm.getMaxRepeatCount() + 1;
  GuardBuilder g{builder, loc};
  uint64_t sizeMax[4];
  auto checkSize = [&](int i, uint64_t hi) {
    sizeMax[i] = getConstantIntValue(sizesRev[i]).value_or(hi);
    return g.check(g.inRange1(inS[i], hi),
                   (i == 3 ? Twine("a runtime DMA repeat count")
                           : "a runtime DMA d" + Twine(i) + " size") +
                       " must be in [1:" + Twine(hi) + "]");
  };
  if (failed(checkSize(0, (isLinear ? maxLen : wrapMax) * gran / ew)) ||
      failed(checkSize(1, isLinear ? maxLen : wrapMax)) ||
      failed(checkSize(2, maxLen)) || failed(checkSize(3, maxRepeats)))
    return failure();
  Value pureRepeat = g.eq(inT[3], 0);
  if (failed(g.check(
          g.either(pureRepeat, g.ule(g.sub(inS[3], 1), maxIterations - 1)),
          "a runtime DMA iteration count must be in [1:" +
              Twine(maxIterations) + "]")))
    return failure();
  for (int i = 0; i < 4; i++) {
    if (i < 3 && (isLinear || (i == 0 && ew > gran)))
      continue;
    Value inRange =
        i == 3 ? g.ule(inT[3], maxStride) : g.inRange1(inT[i], maxStride);
    if (failed(g.check(g.either(g.ule(inS[i], 1), inRange),
                       (i == 3 ? Twine("a runtime DMA iteration stride")
                               : "a runtime DMA d" + Twine(i) + " stride") +
                           " must be in [" + Twine(i == 3 ? 0 : 1) + ":" +
                           Twine(maxStride) + "] when its size > 1")))
      return failure();
  }

  // A non-word element's innermost stride must be 1 (see verifyStridesWraps).
  if (ew != gran) {
    Value d0Unit = g.eq(inT[0], 1);
    if (failed(g.check(g.either(g.ule(inS[0], 1), d0Unit),
                       "a runtime DMA d0 stride must be 1 for " +
                           Twine(ew / 8) +
                           "-byte elements (the DMA moves whole " +
                           Twine(gran / 8) + "-byte granules)")))
      return failure();
  }

  // A size or stride whose byte extent is not a whole number of granules is
  // unrealizable (mirrors verifyStridesWraps).
  int64_t divisor = bdGranuleDivisor(elemWidth, gran);
  if (divisor > 1) {
    auto checkMultiple = [&](Value v, const Twine &what) {
      return g.check(g.multipleOf(v, divisor),
                     "a runtime DMA " + what + " must be a multiple of " +
                         Twine(divisor) + " elements (whole " +
                         Twine(gran / 8) + "-byte granules)");
    };
    if (failed(checkMultiple(inS[0], "d0 size")))
      return failure();
    for (int i = 1; i < 4; i++)
      if (failed(checkMultiple(inT[i], i == 3 ? Twine("iteration stride")
                                              : "d" + Twine(i) + " stride")))
        return failure();
  }

  // Compute the hardware sizes/strides via the shared encoder.
  Value hwS[4], hwT[4];
  SsaStridePolicy policy(builder, loc);
  encodeHardwareStridesWraps(policy, elemWidth, gran, inS, inT, hwS, hwT);

  // buffer_length (word[0]) is the d0*d1*d2 extent in granules. Every size is
  // already bounded by 2^32, so the d0*d1 product cannot wrap, and the final
  // product only counts once d0*d1 is in range. The guard is skipped when the
  // size bounds alone keep it in range. A dma_task's explicit `len` must agree
  // with it.
  Value d0d1 = g.mul(hwS[0], inS[1]);
  Value bufLen = g.mul(d0d1, inS[2]);
  uint64_t lenMax = llvm::SaturatingMultiply(
      llvm::SaturatingMultiply(sizeMax[0] * ew / gran, sizeMax[1]), sizeMax[2]);
  if (lenMax > maxLen) {
    Value lenFits = g.ule(bufLen, maxLen);
    if (bufLen != d0d1)
      lenFits = g.both(g.ule(d0d1, maxLen), lenFits);
    if (failed(g.check(lenFits, "a runtime DMA transfer exceeds the " +
                                    Twine(maxLen) +
                                    "-granule BD buffer_length")))
      return failure();
  }
  if (lenElems) {
    Value lenV = getAsI64(builder, loc, lenElems);
    if (!lenV)
      return failure();
    if (failed(g.check(g.inRange1(lenV, maxLen * gran / ew),
                       "a runtime DMA length must be in [1:" +
                           Twine(maxLen * gran / ew) + "] elements")) ||
        (divisor > 1 &&
         failed(g.check(g.multipleOf(lenV, divisor),
                        "a runtime DMA length must be a multiple of " +
                            Twine(divisor) + " elements (whole " +
                            Twine(gran / 8) + "-byte granules)"))))
      return failure();
    Value lenGranules =
        ew >= gran
            ? g.mul(lenV, g.cst(ew / gran))
            : builder.createOrFold<arith::DivUIOp>(loc, lenV, g.cst(gran / ew));
    if (failed(g.check(g.eq(lenGranules, bufLen),
                       "a runtime DMA length must equal the d0*d1*d2 extent "
                       "of its dimensions")))
      return failure();
  }

  auto asI32 = [&](Value v) {
    return builder.createOrFold<arith::TruncIOp>(loc, i32ty, v);
  };
  for (int i = 0; i < 4; i++) {
    out.hwS[i] = asI32(hwS[i]);
    out.hwT[i] = asI32(hwT[i]);
  }
  out.bufLen = asI32(bufLen);

  // iteration_size: a zero outer stride is a pure repeat (carried by
  // repeat_count), so the field must be 0 like AIEDmaToNpu; gate it on the
  // stride while leaving hwS[3] for repeatCount. hwT[3] already collapses to 0.
  out.iterSizeField = asI32(policy.selectGT0(inT[3], hwS[3], policy.cst(0)));

  // repeat_count for the queue push is the biased outer size (N > 1 ? N - 1 :
  // 0), matching the static path. A constant folds so the static push_queue
  // lowering can consume it; a runtime size yields the SSA hwS[3].
  if (auto outerConst = getConstantIntValue(sizesRev[3])) {
    int64_t r = *outerConst > 1 ? *outerConst - 1 : 0;
    out.repeatCount =
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, r));
  } else {
    out.repeatCount = out.hwS[3];
  }
  return success();
}

// dynamic-matches-static-words.mlir holds this to WriteBdToBlockWritePattern.
void packBdWords(OpBuilder &builder, Location loc,
                 const AIE::AIETargetModel &tm, const AIE::DmaBdLayout &layout,
                 const BdTemplateFields &f, const EncodedBd &e,
                 uint32_t burstLength, uint32_t axcache,
                 SmallVectorImpl<Value> &wordsOut) {
  using Field = std::tuple<Value, uint32_t, uint32_t>;
  SmallVector<uint32_t, 8> constWords(layout.numWords, 0);
  SmallVector<SmallVector<Field, 3>, 8> runtimeWords(layout.numWords);
  auto set = [&](const AIE::DmaBdField &field, uint64_t value) {
    if (field.exists())
      constWords[field.word] |= field.place(value);
  };
  auto add = [&](const AIE::DmaBdField &field, Value value) {
    if (field.exists())
      runtimeWords[field.word].push_back({value, field.mask(), field.shift});
  };

  set(layout.enablePacket, f.enable_packet);
  set(layout.packetType, f.packet_type);
  set(layout.packetId, f.packet_id);
  set(layout.outOfOrderId, f.out_of_order_id);
  set(layout.iterationCurrent, f.iteration_current);
  if (layout.burstLength.exists())
    set(layout.burstLength, AIE::getShimBurstLengthEncoding(tm, burstLength));
  set(layout.axcache, axcache);
  if (f.next_bd_id_val)
    add(layout.nextBd, f.next_bd_id_val);
  else
    set(layout.nextBd, f.next_bd_id);
  set(layout.useNextBd, f.use_next_bd);
  set(layout.validBd, 1);
  if (f.lock_rel_val_val)
    add(layout.lockRelValue, f.lock_rel_val_val);
  else
    set(layout.lockRelValue, f.lock_rel_val);
  set(layout.lockRelId, f.lock_rel_id);
  set(layout.lockAcqEnable, f.lock_acq_enable);
  if (f.lock_acq_val_val)
    add(layout.lockAcqValue, f.lock_acq_val_val);
  else
    set(layout.lockAcqValue, f.lock_acq_val);
  set(layout.lockAcqId, f.lock_acq_id);

  add(layout.bufferLength, e.bufLen);
  if (!e.isLinear) {
    add(layout.d0Size, e.hwS[0]);
    add(layout.d0Stride, e.hwT[0]);
    add(layout.d1Size, e.hwS[1]);
    add(layout.d1Stride, e.hwT[1]);
    add(layout.d2Stride, e.hwT[2]);
  }
  add(layout.iterationSize, e.iterSizeField);
  add(layout.iterationStride, e.hwT[3]);

  wordsOut.clear();
  for (auto [constWord, fields] : llvm::zip(constWords, runtimeWords)) {
    if (fields.empty()) {
      wordsOut.push_back(createConstantI32(builder, loc, constWord));
      continue;
    }
    auto [value, mask, shift] = fields.front();
    if (constWord == 0 && fields.size() == 1 && mask == 0xFFFFFFFF &&
        shift == 0) {
      wordsOut.push_back(value);
      continue;
    }
    Value word = buildBdWord(builder, loc, fields);
    if (constWord != 0)
      word = builder.createOrFold<arith::OrIOp>(
          loc, createConstantI32(builder, loc, constWord), word);
    wordsOut.push_back(word);
  }
}

} // namespace

LogicalResult
buildBdWords(OpBuilder &builder, Location loc,
             const AIE::AIETargetModel &targetModel, int tileCol, int tileRow,
             const BdTemplateFields &f, ArrayRef<OpFoldResult> mixedSizes,
             ArrayRef<OpFoldResult> mixedStrides, uint64_t elemWidth,
             uint32_t burstLength, uint32_t axcache, OpFoldResult lenElems,
             Value &repeatCountOut, SmallVectorImpl<Value> &wordsOut) {
  EncodedBd e;
  if (failed(encodeBdCommon(builder, loc, targetModel, tileCol, tileRow,
                            mixedSizes, mixedStrides, elemWidth, lenElems, e)))
    return failure();
  repeatCountOut = e.repeatCount;

  const AIE::DmaBdLayout *layout = targetModel.getDmaBdLayout(tileCol, tileRow);
  assert(layout && "buildBdWords called for a tile with no DMA BD layout "
                   "(rejected by the caller)");
  packBdWords(builder, loc, targetModel, *layout, f, e, burstLength, axcache,
              wordsOut);
  return success();
}

LogicalResult buildAIE4ShimBdWords(
    OpBuilder &builder, Location loc, const AIE::AIETargetModel &targetModel,
    const BdTemplateFields &f, ArrayRef<OpFoldResult> mixedSizes,
    ArrayRef<OpFoldResult> mixedStrides, uint64_t elemWidth,
    uint32_t burstLength, Value bufLenOverride, Value &repeatCountOut,
    SmallVectorImpl<Value> &wordsOut, bool emitD2Wrap) {
  assert(isa<AIE::AIE4TargetModel>(targetModel) &&
         "buildAIE4ShimBdWords called for a non-AIE4 target");
  auto i32ty = builder.getIntegerType(32);
  if (f.next_bd_id_val || f.lock_rel_val_val || f.lock_acq_val_val)
    return emitError(loc, "runtime next_bd and lock values are not supported "
                          "on AIE4 shim BDs.");

  // AIE4 shim BD field widths: 12-bit D0..D2 wraps, 22-bit steps, 6-bit
  // iteration wrap.
  constexpr uint64_t kWrapMax = 4095, kStepMax = 0x3fffff, kIterWrapMax = 63;
  uint32_t gran = targetModel.getAddressGenGranularity();
  uint64_t ew = elemWidth;
  GuardBuilder g{builder, loc};

  SmallVector<OpFoldResult, 5> sizesRev(llvm::reverse(mixedSizes));
  SmallVector<OpFoldResult, 5> stridesRev(llvm::reverse(mixedStrides));
  normalizeMixedBdDimVector(sizesRev, builder.getI64IntegerAttr(1));
  normalizeMixedBdDimVector(stridesRev, builder.getI64IntegerAttr(0));

  Value inS[5], inT[5];
  for (int i = 0; i < 5; i++) {
    inS[i] = getAsI64(builder, loc, sizesRev[i]);
    inT[i] = getAsI64(builder, loc, stridesRev[i]);
    if (!inS[i] || !inT[i])
      return failure();
  }

  // Every operand must stay inside the range the scaled 32-bit fields can
  // represent, so an out-of-range value never aliases a small hardware field.
  // A stride is only checked where its size > 1 (it is never applied
  // otherwise).
  uint64_t scale = ew / std::gcd(ew, (uint64_t)gran);
  uint64_t operandMax = (uint64_t)UINT32_MAX / scale;
  for (int i = 0; i < 5; ++i) {
    if (failed(g.check(g.inRange1(inS[i], operandMax),
                       "a runtime AIE4 shim DMA d" + Twine(i) +
                           " size must be in [1:" + Twine(operandMax) + "]")))
      return failure();
    Value strideOk = g.ule(inT[i], operandMax);
    if (i != 0)
      strideOk = g.either(g.ule(inS[i], 1), strideOk);
    if (failed(g.check(strideOk, "a runtime AIE4 shim DMA d" + Twine(i) +
                                     " stride exceeds the hardware range")))
      return failure();
  }

  // AIE4 has no D0_Stepsize field: D0 is always contiguous.
  if (!mixedSizes.empty() &&
      failed(g.check(g.eq(inT[0], 1),
                     "AIE4 D0 stride is fixed by hardware and must be 1.")))
    return failure();

  // Linear mode: a contiguous transfer (each outer stride == product of inner
  // sizes) folds D0..D3 into buffer_length, dodging the 12-bit wrap limits. A
  // runtime operand needed for the test falls back to ND mode.
  auto knownContiguous = [&]() -> bool {
    auto s0 = getConstantIntValue(stridesRev[0]);
    if (!s0 || *s0 != 1)
      return false;
    std::optional<int64_t> innerProduct = 1;
    for (int i = 0; i < 4; ++i) {
      auto sz = getConstantIntValue(sizesRev[i]);
      auto st = getConstantIntValue(stridesRev[i]);
      if (i > 0 && (!sz || *sz != 1) &&
          (!innerProduct || !st || *st != *innerProduct))
        return false;
      if (sz && innerProduct)
        *innerProduct *= *sz;
      else
        innerProduct.reset();
    }
    return true;
  };
  bool isLinear = mixedSizes.empty() || knownContiguous();

  // A stride must be positive where its size > 1; the iteration stride may be
  // zero (a pure repeat).
  for (int i = 1; i < 4; ++i)
    if (failed(g.check(g.either(g.ule(inS[i], 1), g.inRange1(inT[i], ~0ULL)),
                       "a runtime AIE4 shim DMA d" + Twine(i) +
                           " stride must be positive when its size > 1")))
      return failure();

  // A size or stride whose byte extent is not a whole number of granules is
  // unrealizable (mirrors verifyStridesWraps).
  int64_t divisor = bdGranuleDivisor(elemWidth, gran);
  if (divisor > 1) {
    auto checkMultiple = [&](Value v, const Twine &what) {
      return g.check(g.multipleOf(v, divisor),
                     "a runtime DMA " + what + " must be a multiple of " +
                         Twine(divisor) + " elements (whole " +
                         Twine(gran / 8) + "-byte granules)");
    };
    if (failed(checkMultiple(inS[0], "d0 size")))
      return failure();
    for (int i = 1; i < 5; i++)
      if (failed(checkMultiple(inT[i], i == 4 ? Twine("iteration stride")
                                              : "d" + Twine(i) + " stride")))
        return failure();
  }

  // Compute the hardware sizes/strides via the shared D3 encoder.
  Value hwS[5], hwT[5];
  SsaStridePolicy policy(builder, loc);
  encodeHardwareStridesWrapsD3(policy, elemWidth, gran, /*isAIE4=*/true, inS,
                               inT, hwS, hwT);

  // Every encoded field must fit. The iteration wrap only matters for an
  // advancing iteration; a pure repeat is carried by the queue push.
  for (int i = 0; i < 3; ++i) {
    bool hasWrap = !isLinear && (i < 2 || emitD2Wrap);
    if (hasWrap && failed(g.check(g.ule(hwS[i], kWrapMax),
                                  "a runtime AIE4 shim DMA d" + Twine(i) +
                                      " size exceeds its 12-bit BD field")))
      return failure();
  }
  uint64_t maxRepeats = targetModel.getMaxRepeatCount() + 1;
  if (failed(
          g.check(g.either(g.both(g.eq(inT[4], 0), g.ule(inS[4], maxRepeats)),
                           g.ule(hwS[4], kIterWrapMax)),
                  "a runtime AIE4 shim DMA iteration count exceeds the "
                  "hardware range")))
    return failure();
  for (int i = 1; i < 5; ++i) {
    if (i != 4 && isLinear)
      continue;
    if (failed(g.check(g.ule(hwT[i], kStepMax),
                       "a runtime AIE4 shim DMA d" + Twine(i) +
                           " stride exceeds its 22-bit BD field")))
      return failure();
  }

  // buffer_length is the full D0..D3 extent in granules, computed in i64 with
  // each step bounded so a runtime product cannot wrap around to a valid BD.
  Value bufLen = bufLenOverride;
  if (!bufLen) {
    uint64_t maxElems = (uint64_t)UINT32_MAX * gran / ew;
    Value extent = g.cst(1);
    for (int i = 0; i < 4; ++i) {
      Value size = inS[i];
      Value safeSize = builder.createOrFold<arith::SelectOp>(loc, g.eq(size, 0),
                                                             g.cst(1), size);
      Value limit =
          builder.createOrFold<arith::DivUIOp>(loc, g.cst(maxElems), safeSize);
      if (failed(g.check(g.cmp(arith::CmpIPredicate::ule, extent, limit),
                         "a runtime AIE4 shim DMA transfer exceeds the "
                         "32-bit BD buffer_length")))
        return failure();
      extent = g.mul(extent, size);
    }
    int64_t scaleGcd = std::gcd(ew, (uint64_t)gran);
    extent = g.mul(extent, g.cst(ew / scaleGcd));
    extent = builder.createOrFold<arith::DivUIOp>(loc, extent,
                                                  g.cst(gran / scaleGcd));
    bufLen = builder.createOrFold<arith::TruncIOp>(loc, i32ty, extent);
  }

  auto asI32 = [&](Value v) {
    return builder.createOrFold<arith::TruncIOp>(loc, i32ty, v);
  };
  Value zero = createConstantI32(builder, loc, 0);
  Value d0 = isLinear ? zero : asI32(hwS[0]);
  Value d1 = isLinear ? zero : asI32(hwS[1]);
  Value d2 = !isLinear && emitD2Wrap ? asI32(hwS[2]) : zero;
  auto orConst = [&](Value word, uint32_t c) {
    return builder.createOrFold<arith::OrIOp>(
        loc, word, createConstantI32(builder, loc, c));
  };

  wordsOut.assign(9, zero);
  // word[1] buffer_offset stays 0 (the address patch supplies the pointer).
  // word[2] buffer_length.
  wordsOut[2] = bufLen;
  // word[3] Lock_Acq_ID [31:27], Lock_Acq_Enable [26], Burst_Length [25:24],
  // D1_Wrap [23:12], D2_Wrap [11:0].
  wordsOut[3] = orConst(
      buildBdWord(builder, loc, {{d1, 0xfff, 12}, {d2, 0xfff, 0}}),
      ((f.lock_acq_id & 0x1f) << 27) | ((f.lock_acq_enable & 1) << 26) |
          ((AIE::getShimBurstLengthEncoding(targetModel, burstLength) & 3)
           << 24));
  // word[4] D0_Wrap [30:19], Lock_Rel_Value [18:12], Lock_Rel_ID [11:7],
  // Lock_Acq_Value [6:0].
  wordsOut[4] =
      orConst(buildBdWord(builder, loc, {{d0, 0xfff, 19}}),
              ((f.lock_rel_val & 0x7f) << 12) | ((f.lock_rel_id & 0x1f) << 7) |
                  (f.lock_acq_val & 0x7f));
  // word[5] Enable_Packet [31], Next_BD [30:27], Use_Next_BD [26],
  // D1_Stepsize [21:0].
  wordsOut[5] =
      orConst(buildBdWord(builder, loc,
                          {{isLinear ? zero : asI32(hwT[1]), kStepMax, 0}}),
              ((f.enable_packet & 1u) << 31) | ((f.next_bd_id & 0xf) << 27) |
                  ((f.use_next_bd & 1) << 26));
  // word[6] Iteration_Current [29:24], D2_Stepsize [21:0].
  wordsOut[6] =
      orConst(buildBdWord(builder, loc,
                          {{isLinear ? zero : asI32(hwT[2]), kStepMax, 0}}),
              (f.iteration_current & 0x3f) << 24);
  // word[7] Iteration_Wrap [27:22], Iteration_Stepsize [21:0]. A zero outer
  // stride is a pure repeat (carried by repeat_count), so the wrap is gated on
  // the stride; hwT[4] already collapses to 0.
  Value iterSize = asI32(policy.selectGT0(inT[4], hwS[4], policy.cst(0)));
  wordsOut[7] = buildBdWord(
      builder, loc, {{iterSize, 0x3f, 22}, {asI32(hwT[4]), kStepMax, 0}});
  // word[8] D3_Stepsize [31:10], Out_Of_Order_BD_ID [8:5], Packet_ID [4:0].
  wordsOut[8] =
      orConst(buildBdWord(builder, loc,
                          {{isLinear ? zero : asI32(hwT[3]), kStepMax, 10}}),
              ((f.out_of_order_id & 0xf) << 5) | (f.packet_id & 0x1f));

  // repeat_count for the queue push is the biased outer size (N > 1 ? N - 1 :
  // 0), matching the static path. A constant folds so the static push_queue
  // lowering can consume it; a runtime size yields the SSA hwS[4].
  if (auto outerConst = getConstantIntValue(sizesRev[4])) {
    int64_t r = *outerConst > 1 ? *outerConst - 1 : 0;
    repeatCountOut =
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, r));
  } else {
    repeatCountOut = asI32(hwS[4]);
  }
  return success();
}

} // namespace xilinx::AIEX
