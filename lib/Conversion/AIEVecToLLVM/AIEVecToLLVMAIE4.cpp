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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

void populateAIEVecToLLVMAIE4ConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<AIE4MulEwiseF32OpConversion, AIE4MacEwiseI16I32OpConversion,
               AIE4SRSI32I16OpConversion>(converter);
}

} // namespace xilinx::aievec
