//===- AIEVecToLLVMAIE4.cpp - AIEVec to LLVM conversion for AIE4 ----------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// AIE4-specific AIEVec -> XLLVM conversion patterns.
//
// Kept separate from the monolithic AIEVecToLLVM.cpp deliberately: that file
// is ~6k lines of interleaved AIE1/AIE2/AIE2p patterns, and AIE4's intrinsic
// surface shares nothing with it.
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace xilinx::aievec {

// aievec.aie4.mul_ewise_f32 -> llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf
//
// Both sides are <32 x float>, so there is no accumulator widening, no
// UPS/SRS pair and no bitcasting to do: the only thing this pattern adds is
// the multiplier configuration word.
//
// Config word defined by the ISA (see XLLVMAIE4IntrOps.td for the field layout):
//   zeroAcc = true         -- dynamically zero the accumulator; this is a
//                             plain multiply, not a multiply-accumulate, so
//                             nothing must be carried in.
//   amode   = Float        -- floating-point arithmetic mode.
//   bmode   = Mode32x32    -- 32-bit x 32-bit operand lanes.
// Together these encode 0b11011 == 27.
//
// IMPORTANT: the Peano backend does not validate this word. Every value
// produces the same `vmul.f` instruction, with the word materialized verbatim
// into a register (`mova r8, #0x1b`). There is no verifier, no selection
// failure and no diagnostic for a wrong field -- only the numerical result
// changes. That is why the fields are modelled as verified attributes rather
// than an opaque i32, and why a numerical check on hardware is the only real
// validation.
//
// This was measured, not assumed. Flipping amode from Float(1) to Int(0)
// (conf 27 -> 25) still compiles clean, still emits the identical `vmul.f`,
// still exits 0 -- and silently returns garbage on NPU3 (e.g. 7.09 * -2.39
// gave -0 instead of -16.96), because the multiplier reinterprets the f32
// bit patterns as integers. That both confirms the ISA decode of
// amode and demonstrates that the only thing standing between a wrong field
// and wrong numbers is this MLIR-level typing.
struct AIE4MulEwiseF32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MulEwiseF32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MulEwiseF32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::MulEwiseFpConfAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        /*zeroAcc=*/true,
        /*amode=*/xllvm::AIE4AMode::Float,
        /*bmode=*/xllvm::AIE4BMode::Mode32x32);
    return success();
  }
};

// aievec.aie4.mac_ewise_i16_i32
//     -> llvm.aie4.I512.I512.ACC1024.mac.ewise.conf
//
// Encode 0x308: integer mode, 16x16 block mode, and signed inputs.
// Leave zeroAcc clear to preserve the incoming accumulator.
struct AIE4MacEwiseI16I32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MacEwiseI16I32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MacEwiseI16I32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::MacEwiseIntConfAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getAcc(),
        /*zeroAcc=*/false,
        /*amode=*/xllvm::AIE4AMode::Int,
        /*bmode=*/xllvm::AIE4BMode::Mode16x16,
        /*lhsSign=*/xllvm::AIE4OperandSign::Signed,
        /*rhsSign=*/xllvm::AIE4OperandSign::Signed);
    return success();
  }
};

// f32 MAC: conf 26 (0x1a), the f32 MUL configuration with zeroAcc cleared.
struct AIE4MacEwiseF32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MacEwiseF32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MacEwiseF32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::MacEwiseFpConfAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getAcc(),
        /*zeroAcc=*/false,
        /*amode=*/xllvm::AIE4AMode::Float,
        /*bmode=*/xllvm::AIE4BMode::Mode32x32);
    return success();
  }
};

// bf16 -> f32 MUL: conf 107 (0x6b). Float/Mode16x16 selects the 32-lane
// form; both operand format bits select bf16.
struct AIE4MulEwiseBf16F32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MulEwiseBf16F32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MulEwiseBf16F32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::MulEwiseBf16ConfAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        /*zeroAcc=*/true,
        /*amode=*/xllvm::AIE4AMode::Float,
        /*bmode=*/xllvm::AIE4BMode::Mode16x16,
        /*xIsBf16=*/true,
        /*yIsBf16=*/true);
    return success();
  }
};

// bf16 -> f32 MAC: conf 106 (0x6a), the MUL configuration with zeroAcc cleared.
struct AIE4MacEwiseBf16F32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MacEwiseBf16F32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MacEwiseBf16F32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::MacEwiseBf16ConfAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getAcc(),
        /*zeroAcc=*/false,
        /*amode=*/xllvm::AIE4AMode::Float,
        /*bmode=*/xllvm::AIE4BMode::Mode16x16,
        /*xIsBf16=*/true,
        /*yIsBf16=*/true);
    return success();
  }
};

struct AIE4MscEwiseF32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MscEwiseF32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MscEwiseF32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::MscEwiseFpConfAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getAcc(),
        /*zeroAcc=*/false,
        /*amode=*/xllvm::AIE4AMode::Float,
        /*bmode=*/xllvm::AIE4BMode::Mode32x32);
    return success();
  }
};

struct AIE4MscEwiseBf16F32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MscEwiseBf16F32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MscEwiseBf16F32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::MscEwiseBf16ConfAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getAcc(),
        /*zeroAcc=*/false,
        /*amode=*/xllvm::AIE4AMode::Float,
        /*bmode=*/xllvm::AIE4BMode::Mode16x16,
        /*xIsBf16=*/true,
        /*yIsBf16=*/true);
    return success();
  }
};

// aievec.aie4.srs_i32_i16 -> llvm.aie4.I512.v32.acc32.srs
//
// XLLVM materializes the verified shift and sign attributes as separate
// trailing i32 arguments.
struct AIE4SRSI32I16OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4SRSI32I16Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4SRSI32I16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::I512V32Acc32SrsAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource(), op.getShift(),
        op.getIsSigned() ? xllvm::AIE4SRSSign::Signed
                         : xllvm::AIE4SRSSign::Unsigned);
    return success();
  }
};

// aievec.aie4.ups_i16_i32 -> llvm.aie4.acc32.v32.I512.ups
//
// Mirrors AIE4SRSI32I16OpConversion: shift and sign are attribute-resident
// trailing arguments, not opaque operands.
struct AIE4UPSI16I32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4UPSI16I32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4UPSI16I32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::Acc32V32I512UpsAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource(), op.getShift(),
        op.getIsSigned() ? xllvm::AIE4UPSSign::Signed
                          : xllvm::AIE4UPSSign::Unsigned);
    return success();
  }
};

// aievec.aie4.unpack_i8_i16 -> llvm.aie4.unpack.I512.I16.I8
struct AIE4UnpackI8I16OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4UnpackI8I16Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4UnpackI8I16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::UnpackI512I16I8AIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource(),
        op.getIsSigned() ? xllvm::AIE4UPSSign::Signed
                         : xllvm::AIE4UPSSign::Unsigned);
    return success();
  }
};

// aievec.aie4.conv_bf16_f32 -> llvm.aie4.v32bf16.to.v32accfloat
//
// Plain operand forwarding: this intrinsic takes no attribute-resident
// config word, unlike SRS/UPS/MAC.
struct AIE4ConvBF16F32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4ConvBF16F32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4ConvBF16F32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::V32Bf16ToV32AccFloatAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource());
    return success();
  }
};

// aievec.aie4.conv_f32_bf16 -> llvm.aie4.v32accfloat.to.v32bfloat16
struct AIE4ConvF32BF16OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4ConvF32BF16Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4ConvF32BF16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::V32AccFloatToV32Bfloat16AIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource());
    return success();
  }
};

// aievec.aie4.conv_f32_i32 -> llvm.aie4.v32float.to.v32int32
//
// One instruction and no control-register traffic: rounding comes from the
// ambient crRnd, which AIECoreToStandard establishes at core entry.
struct AIE4ConvF32I32OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4ConvF32I32Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4ConvF32I32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::V32FloatToV32Int32AIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource(), /*scale=*/0,
        op.getIsSigned() ? xllvm::AIE4SRSSign::Signed
                         : xllvm::AIE4SRSSign::Unsigned);
    return success();
  }
};

// arith.truncf <32xf32> -> <32xbf16> -> llvm.aie4.v32accfloat.to.v32bfloat16
//
// The conversion rounds by the ambient crRnd; arith.truncf asks for round to
// nearest even.
struct AIE4TruncFBF16Conversion
    : mlir::ConvertOpToLLVMPattern<arith::TruncFOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static bool matches(arith::TruncFOp op) {
    auto srcTy = dyn_cast<VectorType>(op.getIn().getType());
    auto dstTy = dyn_cast<VectorType>(op.getType());
    if (!srcTy || !dstTy || srcTy.isScalable() || srcTy.getRank() != 1 ||
        srcTy.getShape()[0] != 32 || !srcTy.getElementType().isF32() ||
        !dstTy.getElementType().isBF16())
      return false;
    auto mode = op.getRoundingmode();
    return !mode || *mode == arith::RoundingMode::to_nearest_even;
  }

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matches(op))
      return failure();
    rewriter.replaceOpWithNewOp<xllvm::V32AccFloatToV32Bfloat16AIE4IntrOp>(
        op, op.getType(), adaptor.getIn());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Integer min/max/compare/select
//===----------------------------------------------------------------------===//

namespace {
// Intrinsic families shared by min/max, compare, and select lowering.
enum class AIE4VAddFamily { I8x64, I16x32, I16x64, I32x16, I32x32 };
} // namespace

static FailureOr<AIE4VAddFamily> classifyAIE4VAddShape(VectorType vecTy) {
  if (vecTy.getRank() != 1)
    return failure();
  unsigned elWidth = vecTy.getElementType().getIntOrFloatBitWidth();
  int64_t lanes = vecTy.getShape()[0];
  if (elWidth == 8 && lanes == 64)
    return AIE4VAddFamily::I8x64;
  if (elWidth == 16 && lanes == 32)
    return AIE4VAddFamily::I16x32;
  if (elWidth == 16 && lanes == 64)
    return AIE4VAddFamily::I16x64;
  if (elWidth == 32 && lanes == 16)
    return AIE4VAddFamily::I32x16;
  if (elWidth == 32 && lanes == 32)
    return AIE4VAddFamily::I32x32;
  return failure();
}

// Bridge scalar i64 masks and the intrinsic's vector<2xi32> mask type.
static Value bitcastIfNeeded(ConversionPatternRewriter &rewriter,
                              Location loc, Value v, Type destTy) {
  if (v.getType() == destTy)
    return v;
  return rewriter.create<LLVM::BitcastOp>(loc, destTy, v);
}

// Return both fields of the intrinsic result. Lt/Ge callers must leave the
// vector field unused because those instructions produce only a mask.
static std::pair<Value, Value>
buildAIE4VCmp(ConversionPatternRewriter &rewriter, Location loc,
              AIE4VAddFamily family, Value lhs, Value rhs,
              xllvm::AIE4OperandSign sign, xllvm::AIE4VCmpOpSel opsel) {
  // XLLVM requires the result struct type explicitly.
  Type i32Ty = rewriter.getI32Type();
  Type vec2xi32Ty = VectorType::get({2}, i32Ty);
  Value res;
  switch (family) {
  case AIE4VAddFamily::I8x64:
    res = rewriter.create<xllvm::VAdd8I512CmpAIE4IntrOp>(
        loc, LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                               {lhs.getType(), vec2xi32Ty}),
        lhs, rhs, sign, opsel);
    break;
  case AIE4VAddFamily::I16x32:
    res = rewriter.create<xllvm::VAdd16I512CmpAIE4IntrOp>(
        loc,
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                          {lhs.getType(), i32Ty}),
        lhs, rhs, sign, opsel);
    break;
  case AIE4VAddFamily::I16x64:
    res = rewriter.create<xllvm::VAdd16I1024CmpAIE4IntrOp>(
        loc, LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                               {lhs.getType(), vec2xi32Ty}),
        lhs, rhs, sign, opsel);
    break;
  case AIE4VAddFamily::I32x16:
    res = rewriter.create<xllvm::VAdd32I512CmpAIE4IntrOp>(
        loc,
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                          {lhs.getType(), i32Ty}),
        lhs, rhs, sign, opsel);
    break;
  case AIE4VAddFamily::I32x32:
    res = rewriter.create<xllvm::VAdd32I1024CmpAIE4IntrOp>(
        loc,
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                          {lhs.getType(), i32Ty}),
        lhs, rhs, sign, opsel);
    break;
  }
  auto structTy = cast<LLVM::LLVMStructType>(res.getType());
  Value vecResult = rewriter.create<LLVM::ExtractValueOp>(
      loc, structTy.getBody()[0], res, ArrayRef<int64_t>{0});
  Value maskResult = rewriter.create<LLVM::ExtractValueOp>(
      loc, structTy.getBody()[1], res, ArrayRef<int64_t>{1});
  return {vecResult, maskResult};
}

// `sel` must have the intrinsic family's i32 or vector<2xi32> mask type.
static Value buildAIE4VSel(ConversionPatternRewriter &rewriter, Location loc,
                           AIE4VAddFamily family, Value lhs, Value rhs,
                           Value sel) {
  switch (family) {
  case AIE4VAddFamily::I8x64:
    return rewriter.create<xllvm::VAdd8I512SelAIE4IntrOp>(loc, lhs.getType(),
                                                           lhs, rhs, sel);
  case AIE4VAddFamily::I16x32:
    return rewriter.create<xllvm::VAdd16I512SelAIE4IntrOp>(loc, lhs.getType(),
                                                            lhs, rhs, sel);
  case AIE4VAddFamily::I16x64:
    return rewriter.create<xllvm::VAdd16I1024SelAIE4IntrOp>(
        loc, lhs.getType(), lhs, rhs, sel);
  case AIE4VAddFamily::I32x16:
    return rewriter.create<xllvm::VAdd32I512SelAIE4IntrOp>(loc, lhs.getType(),
                                                            lhs, rhs, sel);
  case AIE4VAddFamily::I32x32:
    return rewriter.create<xllvm::VAdd32I1024SelAIE4IntrOp>(
        loc, lhs.getType(), lhs, rhs, sel);
  }
  llvm_unreachable("unhandled AIE4VAddFamily");
}

struct AIE4MinMaxOpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4MinMaxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4MinMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto family = classifyAIE4VAddShape(cast<VectorType>(op.getLhs().getType()));
    if (failed(family))
      return rewriter.notifyMatchFailure(op, "unsupported AIE4 vector shape");

    auto sign = op.getIsSigned() ? xllvm::AIE4OperandSign::Signed
                                  : xllvm::AIE4OperandSign::Unsigned;
    auto opsel = op.getKind() == aievec::AIE4MinMaxKind::Min
                     ? xllvm::AIE4VCmpOpSel::MinGe
                     : xllvm::AIE4VCmpOpSel::MaxLt;

    auto [vecResult, maskResult] =
        buildAIE4VCmp(rewriter, op.getLoc(), *family, adaptor.getLhs(),
                      adaptor.getRhs(), sign, opsel);
    (void)maskResult;
    rewriter.replaceOp(op, vecResult);
    return success();
  }
};

struct AIE4CmpOpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4CmpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto family = classifyAIE4VAddShape(cast<VectorType>(op.getLhs().getType()));
    if (failed(family))
      return rewriter.notifyMatchFailure(op, "unsupported AIE4 vector shape");

    auto sign = op.getIsSigned() ? xllvm::AIE4OperandSign::Signed
                                  : xllvm::AIE4OperandSign::Unsigned;
    auto opsel = op.getPredicate() == aievec::AIE4CmpPredicate::Lt
                     ? xllvm::AIE4VCmpOpSel::Lt
                     : xllvm::AIE4VCmpOpSel::Ge;

    auto [vecResult, maskResult] =
        buildAIE4VCmp(rewriter, op.getLoc(), *family, adaptor.getLhs(),
                      adaptor.getRhs(), sign, opsel);
    // Lt/Ge produce only the mask; leave the vector result unused.
    (void)vecResult;

    Type destTy = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOp(
        op, bitcastIfNeeded(rewriter, op.getLoc(), maskResult, destTy));
    return success();
  }
};

struct AIE4SelOpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4SelOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4SelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto family = classifyAIE4VAddShape(cast<VectorType>(op.getLhs().getType()));
    if (failed(family))
      return rewriter.notifyMatchFailure(op, "unsupported AIE4 vector shape");

    // Convert the scalar mask to the intrinsic's carrier of the same width.
    Value lhs = adaptor.getLhs();
    Type xllvmSelTy =
        (*family == AIE4VAddFamily::I8x64 || *family == AIE4VAddFamily::I16x64)
            ? (Type)VectorType::get({2}, rewriter.getI32Type())
            : (Type)rewriter.getI32Type();
    Value sel =
        bitcastIfNeeded(rewriter, op.getLoc(), adaptor.getSel(), xllvmSelTy);

    Value result = buildAIE4VSel(rewriter, op.getLoc(), *family, lhs,
                                 adaptor.getRhs(), sel);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Float min/max/compare
//===----------------------------------------------------------------------===//
//
// Native float operations dispatch by element format and lane count. Selectors
// also depend on the format; resolveAIE4VFCmpOpsel centralizes that mapping.

namespace {
enum class AIE4VFCmpFamily { Bf16x32, Bf16x64, Fp16x32, Fp16x64, Fp32x16, Fp32x32 };
// Native operations use the ordering selected by ambient crFPOrder.
enum class AIE4VFCmpOp { MinGe, MaxLt, Lt, Ge };
} // namespace

static FailureOr<AIE4VFCmpFamily> classifyAIE4VFCmpShape(VectorType vecTy) {
  if (vecTy.getRank() != 1)
    return failure();
  Type elTy = vecTy.getElementType();
  int64_t lanes = vecTy.getShape()[0];
  if (elTy.isBF16() && lanes == 32)
    return AIE4VFCmpFamily::Bf16x32;
  if (elTy.isBF16() && lanes == 64)
    return AIE4VFCmpFamily::Bf16x64;
  if (elTy.isF16() && lanes == 32)
    return AIE4VFCmpFamily::Fp16x32;
  if (elTy.isF16() && lanes == 64)
    return AIE4VFCmpFamily::Fp16x64;
  if (elTy.isF32() && lanes == 16)
    return AIE4VFCmpFamily::Fp32x16;
  if (elTy.isF32() && lanes == 32)
    return AIE4VFCmpFamily::Fp32x32;
  return failure();
}

// FP32 uses Ge=48 and Lt=49, reversing the selector order of BF16 and FP16.
static xllvm::AIE4VFCmpOpSel resolveAIE4VFCmpOpsel(AIE4VFCmpFamily family,
                                                    AIE4VFCmpOp op) {
  switch (family) {
  case AIE4VFCmpFamily::Bf16x32:
  case AIE4VFCmpFamily::Bf16x64:
    switch (op) {
    case AIE4VFCmpOp::MinGe:
      return xllvm::AIE4VFCmpOpSel::Bf16MinGe;
    case AIE4VFCmpOp::MaxLt:
      return xllvm::AIE4VFCmpOpSel::Bf16MaxLt;
    case AIE4VFCmpOp::Lt:
      return xllvm::AIE4VFCmpOpSel::Bf16Lt;
    case AIE4VFCmpOp::Ge:
      return xllvm::AIE4VFCmpOpSel::Bf16Ge;
    }
    break;
  case AIE4VFCmpFamily::Fp16x32:
  case AIE4VFCmpFamily::Fp16x64:
    switch (op) {
    case AIE4VFCmpOp::MinGe:
      return xllvm::AIE4VFCmpOpSel::Fp16MinGe;
    case AIE4VFCmpOp::MaxLt:
      return xllvm::AIE4VFCmpOpSel::Fp16MaxLt;
    case AIE4VFCmpOp::Lt:
      return xllvm::AIE4VFCmpOpSel::Fp16Lt;
    case AIE4VFCmpOp::Ge:
      return xllvm::AIE4VFCmpOpSel::Fp16Ge;
    }
    break;
  case AIE4VFCmpFamily::Fp32x16:
  case AIE4VFCmpFamily::Fp32x32:
    switch (op) {
    case AIE4VFCmpOp::MinGe:
      return xllvm::AIE4VFCmpOpSel::Fp32MinGe;
    case AIE4VFCmpOp::MaxLt:
      return xllvm::AIE4VFCmpOpSel::Fp32MaxLt;
    case AIE4VFCmpOp::Lt:
      return xllvm::AIE4VFCmpOpSel::Fp32Lt;
    case AIE4VFCmpOp::Ge:
      return xllvm::AIE4VFCmpOpSel::Fp32Ge;
    }
    break;
  }
  llvm_unreachable("unhandled AIE4VFCmpFamily/AIE4VFCmpOp");
}

// Build the XLLVM operation for this float shape and extract its vector and mask.
static std::pair<Value, Value>
buildAIE4VFCmp(ConversionPatternRewriter &rewriter, Location loc,
               AIE4VFCmpFamily family, Value lhs, Value rhs,
               xllvm::AIE4VFCmpOpSel opsel) {
  Type i32Ty = rewriter.getI32Type();
  Type vec2xi32Ty = VectorType::get({2}, i32Ty);
  Value res;
  switch (family) {
  case AIE4VFCmpFamily::Bf16x32:
    res = rewriter.create<xllvm::VAddBf16I512CmpAIE4IntrOp>(
        loc,
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                          {lhs.getType(), i32Ty}),
        lhs, rhs, opsel);
    break;
  case AIE4VFCmpFamily::Bf16x64:
    res = rewriter.create<xllvm::VAddBf16I1024CmpAIE4IntrOp>(
        loc, LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                               {lhs.getType(), vec2xi32Ty}),
        lhs, rhs, opsel);
    break;
  case AIE4VFCmpFamily::Fp16x32:
    res = rewriter.create<xllvm::VAddFp16I512CmpAIE4IntrOp>(
        loc,
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                          {lhs.getType(), i32Ty}),
        lhs, rhs, opsel);
    break;
  case AIE4VFCmpFamily::Fp16x64:
    res = rewriter.create<xllvm::VAddFp16I1024CmpAIE4IntrOp>(
        loc, LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                               {lhs.getType(), vec2xi32Ty}),
        lhs, rhs, opsel);
    break;
  case AIE4VFCmpFamily::Fp32x16:
    res = rewriter.create<xllvm::VAddFp32I512CmpAIE4IntrOp>(
        loc,
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                          {lhs.getType(), i32Ty}),
        lhs, rhs, opsel);
    break;
  case AIE4VFCmpFamily::Fp32x32:
    res = rewriter.create<xllvm::VAddFp32I1024CmpAIE4IntrOp>(
        loc,
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                          {lhs.getType(), i32Ty}),
        lhs, rhs, opsel);
    break;
  }
  auto structTy = cast<LLVM::LLVMStructType>(res.getType());
  Value vecResult = rewriter.create<LLVM::ExtractValueOp>(
      loc, structTy.getBody()[0], res, ArrayRef<int64_t>{0});
  Value maskResult = rewriter.create<LLVM::ExtractValueOp>(
      loc, structTy.getBody()[1], res, ArrayRef<int64_t>{1});
  return {vecResult, maskResult};
}

struct AIE4FMinMaxOpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4FMinMaxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4FMinMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto family =
        classifyAIE4VFCmpShape(cast<VectorType>(op.getLhs().getType()));
    if (failed(family))
      return rewriter.notifyMatchFailure(op, "unsupported AIE4 vector shape");

    auto vfOp = op.getKind() == aievec::AIE4MinMaxKind::Min
                    ? AIE4VFCmpOp::MinGe
                    : AIE4VFCmpOp::MaxLt;
    auto opsel = resolveAIE4VFCmpOpsel(*family, vfOp);

    auto [vecResult, maskResult] = buildAIE4VFCmp(
        rewriter, op.getLoc(), *family, adaptor.getLhs(), adaptor.getRhs(),
        opsel);
    (void)maskResult;
    rewriter.replaceOp(op, vecResult);
    return success();
  }
};

struct AIE4FCmpOpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4FCmpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4FCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto family =
        classifyAIE4VFCmpShape(cast<VectorType>(op.getLhs().getType()));
    if (failed(family))
      return rewriter.notifyMatchFailure(op, "unsupported AIE4 vector shape");

    auto vfOp = op.getPredicate() == aievec::AIE4CmpPredicate::Lt
                    ? AIE4VFCmpOp::Lt
                    : AIE4VFCmpOp::Ge;
    auto opsel = resolveAIE4VFCmpOpsel(*family, vfOp);

    auto [vecResult, maskResult] = buildAIE4VFCmp(
        rewriter, op.getLoc(), *family, adaptor.getLhs(), adaptor.getRhs(),
        opsel);
    // As with the integer compare, the vector result of a Lt/Ge opsel is
    // architecturally undefined; only the mask is meaningful.
    (void)vecResult;

    Type destTy = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOp(
        op, bitcastIfNeeded(rewriter, op.getLoc(), maskResult, destTy));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Vector reduction
//===----------------------------------------------------------------------===//
//
// Integer intrinsics require a seed, so supply zero for an unseeded op.
// Float intrinsics have separate seeded and unseeded forms.
struct AIE4ReduceOpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4ReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type elemTy =
        cast<VectorType>(op.getSource().getType()).getElementType();
    Type resTy = op.getResult().getType();
    Value src = adaptor.getSource();
    Value seed = adaptor.getSeed();

    if (elemTy.isInteger(32)) {
      if (!seed)
        seed = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                         (int32_t)0);
      switch (op.getKind()) {
      case aievec::AIE4ReduceKind::Add:
        rewriter.replaceOpWithNewOp<xllvm::AddReduceI1024AIE4IntrOp>(
            op, resTy, src, seed);
        break;
      case aievec::AIE4ReduceKind::Min:
        rewriter.replaceOpWithNewOp<xllvm::MinReduceI1024AIE4IntrOp>(
            op, resTy, src, seed);
        break;
      case aievec::AIE4ReduceKind::Max:
        rewriter.replaceOpWithNewOp<xllvm::MaxReduceI1024AIE4IntrOp>(
            op, resTy, src, seed);
        break;
      }
      return success();
    }

    assert(elemTy.isF32() && "AIE4ReduceOp::verify() should have rejected "
                              "any other element type");
    switch (op.getKind()) {
    case aievec::AIE4ReduceKind::Add:
      if (seed)
        rewriter.replaceOpWithNewOp<xllvm::AddReduceFP1024SclAIE4IntrOp>(
            op, resTy, src, seed);
      else
        rewriter.replaceOpWithNewOp<xllvm::AddReduceFP1024AIE4IntrOp>(
            op, resTy, src);
      break;
    case aievec::AIE4ReduceKind::Min:
      if (seed)
        rewriter.replaceOpWithNewOp<xllvm::MinReduceFP1024SclAIE4IntrOp>(
            op, resTy, src, seed);
      else
        rewriter.replaceOpWithNewOp<xllvm::MinReduceFP1024AIE4IntrOp>(
            op, resTy, src);
      break;
    case aievec::AIE4ReduceKind::Max:
      if (seed)
        rewriter.replaceOpWithNewOp<xllvm::MaxReduceFP1024SclAIE4IntrOp>(
            op, resTy, src, seed);
      else
        rewriter.replaceOpWithNewOp<xllvm::MaxReduceFP1024AIE4IntrOp>(
            op, resTy, src);
      break;
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Shuffle
//===----------------------------------------------------------------------===//
//
// The intrinsic uses vector<16xi32> for every mode. Bitcast other element
// types to that representation and cast the result back. Single-operand
// modes ignore xsrc1, so pass lhs for both intrinsic operands.
static xllvm::AIE4ShuffleMode
translateAIE4ShuffleMode(aievec::AIE4ShuffleMode mode) {
  switch (mode) {
  case aievec::AIE4ShuffleMode::T32_4X4:
    return xllvm::AIE4ShuffleMode::T32_4X4;
  case aievec::AIE4ShuffleMode::T8_8X8:
    return xllvm::AIE4ShuffleMode::T8_8X8;
  case aievec::AIE4ShuffleMode::T32_2X16_LO:
    return xllvm::AIE4ShuffleMode::T32_2X16_LO;
  case aievec::AIE4ShuffleMode::T32_2X16_HI:
    return xllvm::AIE4ShuffleMode::T32_2X16_HI;
  }
  llvm_unreachable("unhandled aievec::AIE4ShuffleMode");
}

struct AIE4ShuffleOpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type xllvmVecTy = VectorType::get({16}, rewriter.getI32Type());

    Value xsrc0 = bitcastIfNeeded(rewriter, loc, adaptor.getLhs(), xllvmVecTy);
    Value rhs = adaptor.getRhs();
    Value xsrc1 = bitcastIfNeeded(rewriter, loc, rhs ? rhs : adaptor.getLhs(),
                                   xllvmVecTy);

    Value shuffled = rewriter.create<xllvm::ShuffleAIE4IntrOp>(
        loc, xllvmVecTy, xsrc0, xsrc1, translateAIE4ShuffleMode(op.getMode()));

    rewriter.replaceOp(op, bitcastIfNeeded(rewriter, loc, shuffled,
                                            op.getResult().getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Approximate transcendentals
//===----------------------------------------------------------------------===//
//
// The result type selects the VEXP2 form: bf16 -> llvm.aie4.exp2,
// f32 -> llvm.aie4.exp2.bf20. The caller has already applied any input bias.
struct AIE4BiasedExp2OpConversion
    : mlir::ConvertOpToLLVMPattern<aievec::AIE4BiasedExp2Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4BiasedExp2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = cast<VectorType>(op.getResult().getType());
    if (resTy.getElementType().isF32())
      rewriter.replaceOpWithNewOp<xllvm::Exp2Bf20AIE4IntrOp>(
          op, resTy, adaptor.getSource());
    else
      rewriter.replaceOpWithNewOp<xllvm::Exp2AIE4IntrOp>(op, resTy,
                                                         adaptor.getSource());
    return success();
  }
};

struct AIE4TanhOpConversion : mlir::ConvertOpToLLVMPattern<aievec::AIE4TanhOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4TanhOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::TanhAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource());
    return success();
  }
};

// math.rsqrt -> llvm.aie4.invsqrt
//
// Takes a `math` op rather than an `aievec` one because no earlier pass
// rewrites math.rsqrt: this conversion is the whole lowering. AIE2p has the
// equivalent pattern in AIEVecToLLVM.cpp.
//
// INVSQRT is approximate and reads a subnormal input as zero, so, like
// exp2/tanh, it requires afn. There is no strict lowering.
//
// The vector case unrolls into scalar calls because the ISA has no vector
// invsqrt. Expensive per lane, and the only form available.
struct AIE4RsqrtOpConversion : mlir::ConvertOpToLLVMPattern<math::RsqrtOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp rsqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = rsqrtOp.getLoc();
    if (!arith::bitEnumContainsAll(rsqrtOp.getFastmath(),
                                   arith::FastMathFlags::afn)) {
      // Dialect conversion hides pattern-emitted errors; a warning keeps the
      // reason next to the legalization error.
      rsqrtOp.emitWarning()
          << "AIE4: math.rsqrt requires fastmath<afn>; AIE4 has only the "
             "approximate INVSQRT lowering and no strict lowering";
      return failure();
    }
    Type f32Ty = rewriter.getF32Type();
    Type operandType = adaptor.getOperand().getType();

    if (operandType.isF32()) {
      rewriter.replaceOpWithNewOp<xllvm::InvsqrtAIE4IntrOp>(
          rsqrtOp, f32Ty, adaptor.getOperand());
      return success();
    }

    auto vecTy = dyn_cast<VectorType>(operandType);
    if (!vecTy || !vecTy.getElementType().isF32() || vecTy.isScalable())
      return failure();

    // The type converter keeps multidimensional vector types, but the LLVM
    // operations below take only flat ones, so reshape around the unrolling.
    VectorType flatTy =
        vecTy.getRank() == 1
            ? vecTy
            : VectorType::get({vecTy.getNumElements()}, vecTy.getElementType());
    Value source = adaptor.getOperand();
    if (flatTy != vecTy)
      source = vector::ShapeCastOp::create(rewriter, loc, flatTy, source);

    Value result = LLVM::PoisonOp::create(rewriter, loc, flatTy);
    for (int64_t i = 0, n = flatTy.getNumElements(); i < n; ++i) {
      Value index = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                             rewriter.getI64IntegerAttr(i));
      Value elem =
          LLVM::ExtractElementOp::create(rewriter, loc, source, index);
      Value inv =
          xllvm::InvsqrtAIE4IntrOp::create(rewriter, loc, f32Ty, elem);
      result = LLVM::InsertElementOp::create(rewriter, loc, flatTy, result, inv,
                                             index);
    }

    if (flatTy != vecTy)
      result = vector::ShapeCastOp::create(rewriter, loc, vecTy, result);

    rewriter.replaceOp(rsqrtOp, result);
    return success();
  }
};

struct AIE4InvOpConversion : mlir::ConvertOpToLLVMPattern<aievec::AIE4InvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AIE4InvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xllvm::InvAIE4IntrOp>(
        op, op.getResult().getType(), adaptor.getSource());
    return success();
  }
};

void populateAIEVecToLLVMAIE4ConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<AIE4RsqrtOpConversion, AIE4MulEwiseF32OpConversion,
               AIE4MacEwiseF32OpConversion,
               AIE4MacEwiseI16I32OpConversion,
               AIE4MulEwiseBf16F32OpConversion, AIE4MacEwiseBf16F32OpConversion,
               AIE4MscEwiseF32OpConversion, AIE4MscEwiseBf16F32OpConversion,
               AIE4SRSI32I16OpConversion, AIE4UPSI16I32OpConversion,
               AIE4UnpackI8I16OpConversion,
               AIE4ConvBF16F32OpConversion, AIE4ConvF32BF16OpConversion,
               AIE4ConvF32I32OpConversion,
               AIE4MinMaxOpConversion, AIE4CmpOpConversion,
               AIE4SelOpConversion, AIE4FMinMaxOpConversion,
               AIE4FCmpOpConversion, AIE4ReduceOpConversion,
               AIE4ShuffleOpConversion, AIE4BiasedExp2OpConversion,
               AIE4TanhOpConversion, AIE4InvOpConversion>(converter);
  patterns.add<AIE4TruncFBF16Conversion>(converter);
}

void configureAIEVecToLLVMAIE4Legalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp op) {
    return !AIE4TruncFBF16Conversion::matches(op);
  });
}

} // namespace xilinx::aievec
