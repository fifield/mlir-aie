//===- AIEVecToLLVMAIE4Mmac.cpp - AIE4 MMAC lowering ------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expand matmul and matrix regions into flat matrix-state ops, then lower to
// XLLVM.
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include <numeric>

namespace xilinx {
using namespace mlir; // For dialect names in generated getDependentDialects
#define GEN_PASS_DEF_AIEVECAIE4EXPANDMATMUL
#define GEN_PASS_DEF_AIEVECAIE4EXPANDMATRIXREGION
#include "aie/Conversion/Passes.h.inc"
} // namespace xilinx

using namespace mlir;

// Standalone llc callers must use the staging workarounds that aiecc applies
// in llcWorkaroundFlags (tools/aiecc/aiecc.cpp).

namespace xilinx::aievec {

//===----------------------------------------------------------------------===//
// Physical operand layout
//===----------------------------------------------------------------------===//
//
// Each B tile has 64 columns of 512 bits. Staging packs two columns per
// insert and commits all 32 chunks before the first row MMAC. int4 and fp8
// operands use byte vectors as register images; configuration fields are
// defined alongside the XLLVM intrinsic operations.

namespace {

struct MmacFormatInfo {
  Type elementType;
  int64_t k;
  unsigned elementBits;
};

MmacFormatInfo getInfo(MLIRContext *ctx, AIE4MatrixFormat format) {
  Builder b(ctx);
  switch (format) {
  case AIE4MatrixFormat::Bf16:
    return {b.getBF16Type(), 32, 16};
  case AIE4MatrixFormat::Int8:
    return {b.getIntegerType(8), 64, 8};
  case AIE4MatrixFormat::Int4:
    return {b.getIntegerType(4), 128, 4};
  case AIE4MatrixFormat::Fp8:
    return {Float8E4M3FNType::get(ctx), 64, 8};
  }
  llvm_unreachable("unknown AIE4 matrix format");
}

constexpr int64_t kColumns = 64;
constexpr int64_t kStagingChunkBits = 1024;
constexpr int64_t kColumnBits = 512;

/// Number of 1024-bit staging inserts that fill one dense B tile.
int64_t getStagingChunkCount(const MmacFormatInfo &info) {
  return kColumns * info.k * info.elementBits / kStagingChunkBits;
}

/// Pack an A row or B column for the intrinsic. On this little-endian
/// target, int4 element 2j occupies the low nibble of byte j; fp8 retains
/// its bit pattern. bf16 and i8 already have the required types.
Value toRegisterImage(OpBuilder &b, Location loc, Value v) {
  auto type = cast<VectorType>(v.getType());
  Type elementType = type.getElementType();
  if (elementType.isBF16() || elementType.isSignlessInteger(8))
    return v;
  auto i8Type = VectorType::get(
      {type.getNumElements() * elementType.getIntOrFloatBitWidth() / 8},
      b.getIntegerType(8));
  return vector::BitCastOp::create(b, loc, i8Type, v);
}

/// Write one 1024-bit staging chunk. `image` is vector<64xbf16> for bf16 and
/// vector<128xi8> otherwise; integer formats set crIntMMode from rhsSigned.
void emitInsertStaging(OpBuilder &b, Location loc, AIE4MatrixFormat format,
                       std::optional<bool> rhsSigned, Value image, Value slot) {
  switch (format) {
  case AIE4MatrixFormat::Bf16:
    xllvm::InsertStagingBf16I1024AIE4IntrOp::create(b, loc, image, slot);
    break;
  case AIE4MatrixFormat::Fp8:
    xllvm::InsertStagingFp8I1024AIE4IntrOp::create(b, loc, image, slot);
    break;
  case AIE4MatrixFormat::Int8:
    xllvm::InsertStagingIntI1024AIE4IntrOp::create(
        b, loc, image, slot,
        *rhsSigned ? xllvm::AIE4IntMMode::Int8 : xllvm::AIE4IntMMode::UInt8);
    break;
  case AIE4MatrixFormat::Int4:
    xllvm::InsertStagingIntI1024AIE4IntrOp::create(
        b, loc, image, slot,
        *rhsSigned ? xllvm::AIE4IntMMode::Int4 : xllvm::AIE4IntMMode::UInt4);
    break;
  }
}

/// Copy staging to the active matrix with the commit for `format`.
void emitCommit(OpBuilder &b, Location loc, AIE4MatrixFormat format) {
  switch (format) {
  case AIE4MatrixFormat::Bf16:
  case AIE4MatrixFormat::Int8:
    xllvm::StagingToMatrixI32768AIE4IntrOp::create(b, loc);
    break;
  case AIE4MatrixFormat::Int4:
    xllvm::StagingToMatrixI4I32768AIE4IntrOp::create(b, loc);
    break;
  case AIE4MatrixFormat::Fp8:
    xllvm::StagingToMatrixFP32768AIE4IntrOp::create(b, loc);
    break;
  }
}

xllvm::AIE4OperandSign toSign(std::optional<bool> isSigned) {
  return isSigned.value_or(false) ? xllvm::AIE4OperandSign::Signed
                                  : xllvm::AIE4OperandSign::Unsigned;
}

xllvm::AIE4MmacIntPrecision getIntPrecision(AIE4MatrixFormat format) {
  return format == AIE4MatrixFormat::Int8 ? xllvm::AIE4MmacIntPrecision::Int8x8
                                          : xllvm::AIE4MmacIntPrecision::Int4x4;
}

/// The static-conf row MMAC. With zeroAcc the hardware ignores `acc`, which
/// may be null; the intrinsic still takes an accumulator, so pass poison.
Value emitStaticMmac(OpBuilder &b, Location loc, AIE4MatrixFormat format,
                     Type resultType, Value lhsImage, Value acc, bool zeroAcc,
                     std::optional<bool> lhsSigned,
                     std::optional<bool> rhsSigned) {
  if (zeroAcc)
    acc = LLVM::PoisonOp::create(b, loc, resultType).getResult();
  switch (format) {
  case AIE4MatrixFormat::Bf16:
    return xllvm::MmacBf16I512Acc2048AIE4IntrOp::create(b, loc, resultType, acc,
                                                        lhsImage, zeroAcc);
  case AIE4MatrixFormat::Fp8:
    return xllvm::MmacFp8I512Acc2048AIE4IntrOp::create(
        b, loc, resultType, acc, lhsImage, zeroAcc,
        xllvm::AIE4MmacFp8EMode::All);
  case AIE4MatrixFormat::Int8:
  case AIE4MatrixFormat::Int4:
    return xllvm::MmacI512Acc2048AIE4IntrOp::create(
        b, loc, resultType, acc, lhsImage, zeroAcc, getIntPrecision(format),
        toSign(lhsSigned), toSign(rhsSigned));
  }
  llvm_unreachable("unknown AIE4 matrix format");
}

std::optional<bool> getOptionalBool(BoolAttr attr) {
  if (attr)
    return attr.getValue();
  return std::nullopt;
}

} // namespace

//===----------------------------------------------------------------------===//
// aievec.aie4.matmul -> aievec.aie4.matrix_region
//===----------------------------------------------------------------------===//

static void expandMatMul(RewriterBase &rewriter, AIE4MatMulOp op) {
  rewriter.setInsertionPoint(op);
  AIE4MatrixRegionOp region = buildAIE4MatMulRegion(
      rewriter, op.getLoc(), op.getLhs(), op.getRhs(), op.getAcc(),
      op.getLhsSignedAttr(), op.getRhsSignedAttr());
  rewriter.replaceOp(op, region.getResults());
}

LogicalResult expandAIE4MatMulOps(Operation *root) {
  SmallVector<AIE4MatMulOp> ops;
  root->walk([&](AIE4MatMulOp op) { ops.push_back(op); });
  IRRewriter rewriter(root->getContext());
  for (AIE4MatMulOp op : ops) {
    // The matrix_region verifier rejects a matmul in a region
    // body, so expansion never creates a nested region.
    assert(!op->getParentOfType<AIE4MatrixRegionOp>() &&
           "aievec.aie4.matmul inside a matrix region");
    int64_t m = cast<VectorType>(op.getLhs().getType()).getDimSize(0);
    if (m > kAIE4MatMulMaxRows)
      return op.emitOpError("has ")
             << m << " rows; lowering supports at most " << kAIE4MatMulMaxRows
             << " (one live accumulator register per row); tile M";
    expandMatMul(rewriter, op);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// aievec.aie4.matrix_region -> stage, commit, inlined body
//===----------------------------------------------------------------------===//
//
// Stage before the region body, inside enclosing K/N loops and outside row
// loops. MLIR effects and LLVM memory dependencies preserve matrix-state order.

/// True if the staging transpose of `rhs` will cancel or fold: `rhs` is a
/// [1, 0] transpose (B stored column-major) or a splat.
static bool isCheapToStage(Value rhs) {
  Operation *def = rhs.getDefiningOp();
  if (auto transpose = dyn_cast_or_null<vector::TransposeOp>(def))
    return transpose.getPermutation() == ArrayRef<int64_t>{1, 0};
  if (auto broadcast = dyn_cast_or_null<vector::BroadcastOp>(def))
    return !isa<VectorType>(broadcast.getSourceType());
  if (auto constant = dyn_cast_or_null<arith::ConstantOp>(def))
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
      return dense.isSplat();
  return false;
}

/// Stage logical B (K x 64) in column pairs, then commit the full tile.
/// int4 and fp8 chunks are staged as their vector<128xi8> byte image.
static void emitRegionStaging(OpBuilder &b, Location loc,
                              AIE4MatrixFormat format, BoolAttr rhsSigned,
                              Value rhs) {
  MmacFormatInfo info = getInfo(b.getContext(), format);
  assert(info.k * info.elementBits == kColumnBits &&
         "a B column is one 512-bit half of a staging chunk");

  // B^T: row n of the transpose is column n of B, k ascending.
  Value columns = vector::TransposeOp::create(b, loc, rhs, {1, 0});

  int64_t chunkCount = getStagingChunkCount(info);
  int64_t columnsPerChunk = kStagingChunkBits / kColumnBits;
  for (int64_t i = 0; i < chunkCount; ++i) {
    // Column 2i fills bits 511:0 and column 2i+1 bits 1023:512.
    Value lo = toRegisterImage(
        b, loc,
        vector::ExtractOp::create(b, loc, columns, columnsPerChunk * i));
    Value hi = toRegisterImage(
        b, loc,
        vector::ExtractOp::create(b, loc, columns, columnsPerChunk * i + 1));
    int64_t halfLanes = cast<VectorType>(lo.getType()).getNumElements();
    SmallVector<int64_t> concat(2 * halfLanes);
    std::iota(concat.begin(), concat.end(), 0);
    Value chunk = vector::ShuffleOp::create(b, loc, lo, hi, concat);
    Value slot = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(i));
    AIE4StageOp::create(b, loc, chunk, slot, format, rhsSigned);
  }
  AIE4CommitOp::create(b, loc, format, rhsSigned);
}

static void expandMatrixRegion(RewriterBase &rewriter,
                               AIE4MatrixRegionOp region) {
  if (!isCheapToStage(region.getRhs()))
    region.emitWarning("B tile is not produced by a vector.transpose [1, 0] "
                       "of a column-major tile; staging transposes it element "
                       "by element (about 44x the staging cost for one bf16 "
                       "tile). Supply B column-major so the transpose cancels");
  rewriter.setInsertionPoint(region);
  emitRegionStaging(rewriter, region.getLoc(), region.getFormat(),
                    region.getRhsSignedAttr(), region.getRhs());

  // Row MMACs read the matrix just committed. The region verifier rejects
  // isolated ops, so every row MMAC in the body is in this region's scope.
  Block &body = region.getBody().front();
  SmallVector<AIE4MmacOp> mmacs;
  body.walk([&](AIE4MmacOp mmac) { mmacs.push_back(mmac); });
  for (AIE4MmacOp mmac : mmacs) {
    rewriter.setInsertionPoint(mmac);
    auto active = AIE4MmacActiveOp::create(
        rewriter, mmac.getLoc(), mmac.getResult().getType(), mmac.getLhs(),
        mmac.getAcc(), /*zero_if=*/Value(), mmac.getFormat(),
        mmac.getLhsSignedAttr(), mmac.getRhsSignedAttr());
    rewriter.replaceOp(mmac, active.getResult());
  }

  // Inlining maps the block arguments to the inits, so yielded block
  // arguments become the corresponding inits.
  auto yield = cast<AIE4MatrixYieldOp>(body.getTerminator());
  rewriter.inlineBlockBefore(&body, region, region.getInits());
  SmallVector<Value> results = llvm::to_vector(yield.getValues());
  rewriter.eraseOp(yield);
  rewriter.replaceOp(region, results);
}

LogicalResult expandAIE4MatrixRegionOps(Operation *root) {
  SmallVector<AIE4MatrixRegionOp> regions;
  root->walk([&](AIE4MatrixRegionOp region) { regions.push_back(region); });
  IRRewriter rewriter(root->getContext());
  for (AIE4MatrixRegionOp region : regions)
    expandMatrixRegion(rewriter, region);
  return success();
}

//===----------------------------------------------------------------------===//
// Flat layer: stage, stage_stream, commit, mmac_active -> xllvm
//===----------------------------------------------------------------------===//
//
// Each flat op lowers in place to its XLLVM counterpart. Program order and
// the ops' unknown effects carry the matrix-state dependencies.

struct AIE4StageOpConversion : ConvertOpToLLVMPattern<AIE4StageOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AIE4StageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // A byte-image chunk (vector<128xi8>) is already the register image.
    Value image = toRegisterImage(rewriter, op.getLoc(), adaptor.getChunk());
    emitInsertStaging(rewriter, op.getLoc(), op.getFormat(),
                      getOptionalBool(op.getRhsSignedAttr()), image,
                      adaptor.getSlot());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIE4StageStreamOpConversion : ConvertOpToLLVMPattern<AIE4StageStreamOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AIE4StageStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto rawType = VectorType::get({32}, rewriter.getI32Type());
    Value raw = xllvm::GetSsAIE4IntrOp::create(rewriter, loc, rawType,
                                               adaptor.getPort());
    Type imageType = op.getFormat() == AIE4MatrixFormat::Bf16
                         ? VectorType::get({64}, rewriter.getBF16Type())
                         : VectorType::get({128}, rewriter.getI8Type());
    Value image = vector::BitCastOp::create(rewriter, loc, imageType, raw);
    emitInsertStaging(rewriter, loc, op.getFormat(),
                      getOptionalBool(op.getRhsSignedAttr()), image,
                      adaptor.getSlot());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIE4CommitOpConversion : ConvertOpToLLVMPattern<AIE4CommitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AIE4CommitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    emitCommit(rewriter, op.getLoc(), op.getFormat());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIE4MmacActiveOpConversion : ConvertOpToLLVMPattern<AIE4MmacActiveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AIE4MmacActiveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    AIE4MatrixFormat format = op.getFormat();
    Type resultType = op.getResult().getType();
    std::optional<bool> lhsSigned = getOptionalBool(op.getLhsSignedAttr());
    std::optional<bool> rhsSigned = getOptionalBool(op.getRhsSignedAttr());
    Value lhs = toRegisterImage(rewriter, loc, adaptor.getLhs());
    Value acc = adaptor.getAcc();
    Value zeroIf = adaptor.getZeroIf();

    // A constant predicate selects the static form: true is exactly the
    // no-acc form, false the with-acc form.
    if (!zeroIf || matchPattern(zeroIf, m_Zero()) ||
        matchPattern(zeroIf, m_One())) {
      bool zeroAcc = !acc || (zeroIf && matchPattern(zeroIf, m_One()));
      rewriter.replaceOp(op,
                         emitStaticMmac(rewriter, loc, format, resultType, lhs,
                                        acc, zeroAcc, lhsSigned, rhsSigned));
      return success();
    }

    // Run-time zero_acc (the verifier guarantees an integer format and an
    // accumulator): conf bit 0 is the predicate.
    uint32_t base = xllvm::encodeAIE4MmacIntConf(
        /*zeroAcc=*/false, getIntPrecision(format), toSign(lhsSigned),
        toSign(rhsSigned));
    Value baseConf = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(base));
    Value zeroBit =
        arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), zeroIf);
    Value conf = arith::OrIOp::create(rewriter, loc, baseConf, zeroBit);
    rewriter.replaceOpWithNewOp<xllvm::MmacI512Acc2048DynConfAIE4IntrOp>(
        op, resultType, acc, lhs, conf);
    return success();
  }
};

void populateAIEVecToLLVMAIE4MmacPatterns(mlir::LLVMTypeConverter &converter,
                                          mlir::RewritePatternSet &patterns) {
  patterns.add<AIE4StageOpConversion, AIE4StageStreamOpConversion,
               AIE4CommitOpConversion, AIE4MmacActiveOpConversion>(converter);
}

//===----------------------------------------------------------------------===//
// aievec-aie4-expand-matmul
//===----------------------------------------------------------------------===//

namespace {
struct AIEVecAIE4ExpandMatMulPass
    : xilinx::impl::AIEVecAIE4ExpandMatMulBase<AIEVecAIE4ExpandMatMulPass> {
  void runOnOperation() override {
    if (failed(expandAIE4MatMulOps(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIEVecAIE4ExpandMatMulPass() {
  return std::make_unique<AIEVecAIE4ExpandMatMulPass>();
}

//===----------------------------------------------------------------------===//
// aievec-aie4-expand-matrix-region
//===----------------------------------------------------------------------===//

namespace {
struct AIEVecAIE4ExpandMatrixRegionPass
    : xilinx::impl::AIEVecAIE4ExpandMatrixRegionBase<
          AIEVecAIE4ExpandMatrixRegionPass> {
  void runOnOperation() override {
    if (failed(expandAIE4MatrixRegionOps(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIEVecAIE4ExpandMatrixRegionPass() {
  return std::make_unique<AIEVecAIE4ExpandMatrixRegionPass>();
}

} // namespace xilinx::aievec
