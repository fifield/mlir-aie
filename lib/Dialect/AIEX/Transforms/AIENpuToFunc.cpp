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

static func::CallOp convertOpToFunction(Operation *op, ArrayRef<Value> operands,
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
    device.insert(device.getBody()->begin(), fn);
  }
  func::CallOp call = rewriter.replaceOpWithNewOp<func::CallOp>(
      op, retTys, SymbolRefAttr::get(fn), operands);
  return call;
}

namespace {

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
    auto call = convertOpToFunction(op, operands, rewriter, "npu_address_patch");
    if (call)
      return success();
    else
      return failure();
  }
};

struct NpuBlockWriteToFuncPattern
    : public OpConversionPattern<NpuBlockWriteOp> {
  using OpConversionPattern<NpuBlockWriteOp>::OpConversionPattern;

  NpuBlockWriteToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<NpuBlockWriteOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuBlockWriteOp op, OpAdaptor adaptor,
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

    // Value memref = op.getData();
    // int64_t width =
    // cast<MemRefType>(memref.getType()).getElementTypeBitWidth(); if (width !=
    // 32) {
    //     op.emitWarning("Only 32-bit data type is supported for now");
    //     return;
    // }

    // memref::GetGlobalOp getGlobal =
    // memref.getDefiningOp<memref::GetGlobalOp>(); if (!getGlobal) {
    //     op.emitError("Only MemRefs from memref.get_global are supported");
    //     return;
    // }

    // auto global = dyn_cast_if_present<memref::GlobalOp>(
    //     op->getParentOfType<AIE::DeviceOp>().lookupSymbol(getGlobal.getName()));
    // if (!global) {
    //     op.emitError("Global symbol not found");
    //     return;
    // }

    // auto initVal = global.getInitialValue();
    // if (!initVal) {
    //     op.emitError("Global symbol has no initial value");
    //     return;
    // }

    // auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
    // if (!data) {
    //     op.emitError("Global symbol initial value is not a dense int array");
    //     return;
    // }

    auto call = convertOpToFunction(op, operands, rewriter, "npu_blockwrite");
    if (call)
      return success();
    else
      return failure();
  }
};

struct NpuMaskWrite32ToFuncPattern
    : public OpConversionPattern<NpuMaskWrite32Op> {
  using OpConversionPattern<NpuMaskWrite32Op>::OpConversionPattern;

  NpuMaskWrite32ToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<NpuMaskWrite32Op>(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuMaskWrite32Op op, OpAdaptor adaptor,
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
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getMask()));
    auto call = convertOpToFunction(op, operands, rewriter, "npu_maskwrite32");
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
    auto call = convertOpToFunction(op, operands, rewriter, "npu_write32");
    if (call)
      return success();
    else
      return failure();
  }
};

struct RuntimeSequenceToFuncPattern
    : public OpConversionPattern<RuntimeSequenceOp> {
  using OpConversionPattern<RuntimeSequenceOp>::OpConversionPattern;

  RuntimeSequenceToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<RuntimeSequenceOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(RuntimeSequenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto name = op.getSymName();
    if (!name)
      return failure();

    SmallVector<Value> operands{adaptor.getOperands()};
    SmallVector<Type> argTypes;
    for (auto a : op.getBody().getArguments())
      argTypes.push_back(a.getType());

    IRMapping mapper;
    auto newFunc = rewriter.create<func::FuncOp>(
        op.getLoc(), name->str(),
        FunctionType::get(rewriter.getContext(), argTypes, {}));
    newFunc.setPrivate();
    rewriter.cloneRegionBefore(op.getBody(), newFunc.getBody(),
                               newFunc.getBody().begin(), mapper);
    rewriter.setInsertionPointToEnd(&newFunc.getBody().front());
    rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(), ValueRange({}));
    rewriter.eraseOp(op);
    return success();
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
    patterns.insert<NpuAddressPatchToFuncPattern, NpuBlockWriteToFuncPattern,
                    NpuMaskWrite32ToFuncPattern, NpuSyncToFuncPattern,
                    NpuWrite32ToFuncPattern, RuntimeSequenceToFuncPattern>(
        &getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIENpuToFuncPass() {
  return std::make_unique<AIENpuToFuncPass>();
}
