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

void populateAIEVecToLLVMAIE4ConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<AIE4MulEwiseF32OpConversion>(converter);
}

} // namespace xilinx::aievec
