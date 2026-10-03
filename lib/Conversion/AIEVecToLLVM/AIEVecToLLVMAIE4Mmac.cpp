//===- AIEVecToLLVMAIE4Mmac.cpp - AIE4 MMAC lowering ------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expand matmul into a matrix region of row MMACs, then lower the region
// to staging inserts, a commit, and the inlined body.
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
#include "mlir/IR/PatternMatch.h"

#include <numeric>

namespace xilinx {
using namespace mlir; // For dialect names in generated getDependentDialects
#define GEN_PASS_DEF_AIEVECAIE4EXPANDMATMUL
#include "aie/Conversion/Passes.h.inc"
} // namespace xilinx

using namespace mlir;

// Peano code generation requires
// -aie4postlegalizercustomcombiner-disable-rule=combine_staging_upd_load
// for dense staging and -disable-machine-cse for fp8 staging. aiecc adds
// these flags per core; standalone llc invocations must supply them too.

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

/// Stage logical B (K x 64) in column pairs, then commit the full tile.
void emitStagingAndCommit(OpBuilder &b, Location loc, AIE4MatrixFormat format,
                          std::optional<bool> rhsSigned, Value rhs) {
  MLIRContext *ctx = b.getContext();
  MmacFormatInfo info = getInfo(ctx, format);
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
    Value index = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(i));

    switch (format) {
    case AIE4MatrixFormat::Bf16:
      xllvm::InsertStagingBf16I1024AIE4IntrOp::create(b, loc, chunk, index);
      break;
    case AIE4MatrixFormat::Fp8:
      xllvm::InsertStagingFp8I1024AIE4IntrOp::create(b, loc, chunk, index);
      break;
    case AIE4MatrixFormat::Int8:
      xllvm::InsertStagingIntI1024AIE4IntrOp::create(
          b, loc, chunk, index,
          *rhsSigned ? xllvm::AIE4IntMMode::Int8 : xllvm::AIE4IntMMode::UInt8);
      break;
    case AIE4MatrixFormat::Int4:
      xllvm::InsertStagingIntI1024AIE4IntrOp::create(
          b, loc, chunk, index,
          *rhsSigned ? xllvm::AIE4IntMMode::Int4 : xllvm::AIE4IntMMode::UInt4);
      break;
    }
  }

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

} // namespace

//===----------------------------------------------------------------------===//
// aievec.aie4.matmul -> aievec.aie4.matrix_region
//===----------------------------------------------------------------------===//
//
// Static row indices avoid routing n-D vector accesses through memory.
// Unrolling keeps M accumulator rows live, so kAIE4MatMulMaxRows bounds M.
// Larger problems can tile M or use a row loop directly in matrix_region.

static void expandMatMul(RewriterBase &rewriter, AIE4MatMulOp op) {
  Location loc = op.getLoc();
  auto lhsType = cast<VectorType>(op.getLhs().getType());
  auto accType = cast<VectorType>(op.getAcc().getType());
  Type elementType = lhsType.getElementType();
  AIE4MatrixFormat format =
      elementType.isBF16()               ? AIE4MatrixFormat::Bf16
      : elementType.isSignlessInteger(8) ? AIE4MatrixFormat::Int8
      : elementType.isSignlessInteger(4) ? AIE4MatrixFormat::Int4
                                         : AIE4MatrixFormat::Fp8;
  int64_t m = lhsType.getDimSize(0);

  rewriter.setInsertionPoint(op);
  auto region = AIE4MatrixRegionOp::create(rewriter, loc, TypeRange{accType},
                                           op.getRhs(), ValueRange{op.getAcc()},
                                           format, op.getRhsSignedAttr());
  Block *body = rewriter.createBlock(&region.getBody(), {}, {accType}, {loc});
  Value acc = body->getArgument(0);
  Value result = acc;
  for (int64_t row = 0; row < m; ++row) {
    Value a = vector::ExtractOp::create(rewriter, loc, op.getLhs(), row);
    Value c = vector::ExtractOp::create(rewriter, loc, acc, row);
    Value r = AIE4MmacOp::create(rewriter, loc, c.getType(), a, c, format,
                                 op.getLhsSignedAttr(), op.getRhsSignedAttr());
    result = vector::InsertOp::create(rewriter, loc, r, result, row);
  }
  AIE4MatrixYieldOp::create(rewriter, loc, ValueRange{result});
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
// aievec.aie4.matrix_region -> staging, commit, inlined body
//===----------------------------------------------------------------------===//
//
// Stage and commit at the region's position, before its body. This keeps
// staging inside enclosing K/N loops and outside row loops in the body.
// Unknown MLIR effects and LLVM inaccessible-memory dependencies preserve
// matrix-state ordering.

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

struct AIE4MatrixRegionOpConversion
    : ConvertOpToLLVMPattern<AIE4MatrixRegionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AIE4MatrixRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCheapToStage(op.getRhs()))
      op.emitWarning("B tile is not produced by a vector.transpose [1, 0] of "
                     "a column-major tile; staging transposes it element by "
                     "element (about 44x the staging cost for one bf16 tile). "
                     "Supply B column-major so the transpose cancels");
    std::optional<bool> rhsSigned;
    if (BoolAttr attr = op.getRhsSignedAttr())
      rhsSigned = attr.getValue();
    emitStagingAndCommit(rewriter, op.getLoc(), op.getFormat(), rhsSigned,
                         adaptor.getRhs());

    Block &body = op.getBody().front();
    auto yield = cast<AIE4MatrixYieldOp>(body.getTerminator());
    // Yielded block arguments become the corresponding inits.
    SmallVector<Value> results;
    for (Value v : yield.getValues()) {
      auto arg = dyn_cast<BlockArgument>(v);
      results.push_back(arg && arg.getOwner() == &body
                            ? adaptor.getInits()[arg.getArgNumber()]
                            : v);
    }
    rewriter.inlineBlockBefore(&body, op, adaptor.getInits());
    rewriter.eraseOp(yield);
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// aievec.aie4.mmac -> xllvm MMAC
//===----------------------------------------------------------------------===//

struct AIE4MmacOpConversion : ConvertOpToLLVMPattern<AIE4MmacOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AIE4MmacOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = op.getResult().getType();
    Value lhs = toRegisterImage(rewriter, loc, adaptor.getLhs());
    // Without an accumulator the hardware zeroes it (conf bit 0); the
    // intrinsic still takes an accumulator operand, which it ignores.
    bool zeroAcc = !adaptor.getAcc();
    Value acc =
        zeroAcc ? LLVM::PoisonOp::create(rewriter, loc, resultType).getResult()
                : adaptor.getAcc();

    std::optional<bool> lhsSigned, rhsSigned;
    if (BoolAttr attr = op.getLhsSignedAttr())
      lhsSigned = attr.getValue();
    if (BoolAttr attr = op.getRhsSignedAttr())
      rhsSigned = attr.getValue();

    switch (op.getFormat()) {
    case AIE4MatrixFormat::Bf16:
      rewriter.replaceOpWithNewOp<xllvm::MmacBf16I512Acc2048AIE4IntrOp>(
          op, resultType, acc, lhs, zeroAcc);
      break;
    case AIE4MatrixFormat::Fp8:
      rewriter.replaceOpWithNewOp<xllvm::MmacFp8I512Acc2048AIE4IntrOp>(
          op, resultType, acc, lhs, zeroAcc, xllvm::AIE4MmacFp8EMode::All);
      break;
    case AIE4MatrixFormat::Int8:
    case AIE4MatrixFormat::Int4:
      rewriter.replaceOpWithNewOp<xllvm::MmacI512Acc2048AIE4IntrOp>(
          op, resultType, acc, lhs, zeroAcc,
          op.getFormat() == AIE4MatrixFormat::Int8
              ? xllvm::AIE4MmacIntPrecision::Int8x8
              : xllvm::AIE4MmacIntPrecision::Int4x4,
          toSign(lhsSigned), toSign(rhsSigned));
      break;
    }
    return success();
  }
};

void populateAIEVecToLLVMAIE4MmacPatterns(mlir::LLVMTypeConverter &converter,
                                          mlir::RewritePatternSet &patterns) {
  patterns.add<AIE4MatrixRegionOpConversion, AIE4MmacOpConversion>(converter);
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

} // namespace xilinx::aievec
