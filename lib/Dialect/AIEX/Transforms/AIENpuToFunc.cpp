//===- AIEXDmaToNpu.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

static func::CallOp
convertOpToFunction(Operation *op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter,
                          StringRef fnName) {

  SmallVector<Type, 16> tys;
  SmallVector<Type, 1> retTys{};

  for (auto o : operands)
    tys.push_back(o.getType());

  AIE::DeviceOp device = op->getParentOfType<AIE::DeviceOp>();
  auto fn = device.lookupSymbol<func::FuncOp>(fnName);
  if (!fn) {
    auto fnTy = rewriter.getFunctionType(tys, retTys);
    fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    device.push_back(fn);
  }
  func::CallOp call = rewriter.replaceOpWithNewOp<func::CallOp>(
      op, retTys, SymbolRefAttr::get(fn), operands);
  return call;
}

struct NpuWriteBdToFuncPattern : public OpConversionPattern<NpuWriteBdOp> {
  using OpConversionPattern<NpuWriteBdOp>::OpConversionPattern;

  NpuWriteBdToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<NpuWriteBdOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWriteBdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 16> operands{adaptor.getOperands()};
    auto loc = op->getLoc();
    const AIE::AIETargetModel &tm =
        op->getParentOfType<AIE::DeviceOp>().getTargetModel();

    uint32_t bd_id = op.getBdId();
    uint32_t bd_addr = (op.getColumn() << tm.getColumnShift()) |
                       (op.getRow() << tm.getRowShift()) |
                       (0x1D000 + bd_id * 0x20);
    operands.push_back(rewriter.create<arith::ConstantIndexOp>(loc, bd_addr));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getBufferLength()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getBufferOffset()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getEnablePacket()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getOutOfOrderId()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getPacketId()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getPacketType()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getD0Size()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getD0Stride()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getD1Size()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getD1Stride()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getD2Stride()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getIterationCurrent()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getIterationSize()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getIterationStride()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getNextBd()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getUseNextBd()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getValidBd()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLockRelVal()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLockRelId()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLockAcqEnable()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLockAcqVal()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLockAcqId()));
    auto call =
        convertOpToFunction(op, operands, rewriter, "npu_writebd");
    if (call)
      return success();
    else
      return failure();
  }
};

struct NpuSyncToFuncPattern : public OpConversionPattern<NpuSyncOp> {
  using OpConversionPattern<NpuSyncOp>::OpConversionPattern;

  NpuSyncToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<NpuSyncOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 16> operands{adaptor.getOperands()};
    auto loc = op->getLoc();

    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getColumn()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getRow()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getDirection()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getChannel()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getColumnNum()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getRowNum()));
    auto call = convertOpToFunction(op, operands, rewriter, "npu_sync");
    if (call)
      return success();
    else
      return failure();
  }
};

struct NpuAddressPatchToFuncPattern
    : public OpConversionPattern<NpuAddressPatchOp> {
  using OpConversionPattern<NpuAddressPatchOp>::OpConversionPattern;

  NpuAddressPatchToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<NpuAddressPatchOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuAddressPatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 16> operands{adaptor.getOperands()};
    auto loc = op->getLoc();

    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getAddr()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getArgIdx()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getArgPlus()));
    auto call =
        convertOpToFunction(op, operands, rewriter, "address_patch");
    if (call)
      return success();
    else
      return failure();
  }
};

struct NpuWrite32ToFuncPattern : public OpConversionPattern<NpuWrite32Op> {
  using OpConversionPattern<NpuWrite32Op>::OpConversionPattern;

  NpuWrite32ToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<NpuWrite32Op>(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 16> operands{adaptor.getOperands()};
    auto loc = op->getLoc();
    const AIE::AIETargetModel &tm =
        op->getParentOfType<AIE::DeviceOp>().getTargetModel();
    std::optional<int> col = op.getColumn();
    std::optional<int> row = op.getRow();
    uint32_t addr = op.getAddress();
    if (col && row)
      addr = ((*col & 0xff) << tm.getColumnShift()) |
             ((*row & 0xff) << tm.getRowShift()) | (addr & 0xFFFFF);
    operands.push_back(rewriter.create<arith::ConstantIndexOp>(loc, addr));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getValue()));
    auto call =
        convertOpToFunction(op, operands, rewriter, "npu_write32");
    if (call)
      return success();
    else
      return failure();
  }
};

struct AIENpuToFuncPass : public AIENpuToFuncBase<AIENpuToFuncPass> {
  void runOnOperation() override {

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addIllegalDialect<AIEXDialect>();
    target.addLegalDialect<AIE::AIEDialect, memref::MemRefDialect,
                           scf::SCFDialect, func::FuncDialect,
                           arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<NpuWriteBdToFuncPattern, NpuSyncToFuncPattern,
                    NpuWrite32ToFuncPattern, NpuAddressPatchToFuncPattern>(
        &getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIENpuToFuncPass() {
  return std::make_unique<AIENpuToFuncPass>();
}
