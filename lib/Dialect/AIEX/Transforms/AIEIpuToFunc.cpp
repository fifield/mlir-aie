//===- AIEXDmaToIpu.cpp -----------------------------------------*- C++ -*-===//
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

// static std::string getMangledType(const Type ty) {
//   std::stringstream ret;

//   if (const MemRefType mrt = ty.dyn_cast<const MemRefType>()) {
//     ret << "M";
//     ret << mrt.getMemorySpaceAsInt();
//     if (mrt.hasStaticShape()) {
//       auto shape = mrt.getShape();
//       for (auto s : shape)
//         ret << s << "x";
//     } else if (mrt.hasRank()) {
//       ret << "D" << mrt.getRank();
//     }
//     const Type elem = mrt.getElementType();
//     ret << getMangledType(elem);
//   } else if (FloatType ft = ty.dyn_cast<FloatType>()) {
//     ret << "F" << ft.getWidth();
//   } else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
//     ret << "I" << it.getWidth();
//   } else if (const IndexType it = ty.dyn_cast<const IndexType>()) {
//     ret << "I64";
//   } else {
//     Type t = ty;
//     t.dump();
//     assert(0 && "unhandled type in getMangledType");
//   }
//   return ret.str();
// }

static std::string getMangledFuncName(std::string prefix, FunctionType fnTy) {
  std::string sep = "_";

  // auto resultTy = fnTy.getResults();
  // auto operTy = fnTy.getInputs();

  std::string ret = prefix;
  // for (const Type t : resultTy)
  //   ret = ret + sep + "r" + getMangledType(t);
  // for (const Type t : operTy)
  //   ret = ret + sep + getMangledType(t);

  return ret;
}

static func::FuncOp getMangledFunction(AIE::DeviceOp device, std::string prefix,
                                       ArrayRef<Value> operands,
                                       ArrayRef<Type> retTys) {
  Builder builder(device);

  SmallVector<Type, 16> tys;
  for (auto o : operands)
    tys.push_back(o.getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(prefix, fnTy);
  auto fn = device.lookupSymbol<func::FuncOp>(fnName);

  if (!fn) {
    fn = func::FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    device.push_back(fn);
  }
  return fn;
}

static func::CallOp convertOpToFunction(Operation *op, ArrayRef<Value> operands,
                                        ConversionPatternRewriter &rewriter,
                                        StringRef fnName) {
  auto loc = op->getLoc();

  SmallVector<Value, 16> callops;
  SmallVector<Type, 1> retTys{};

  auto idTy = IntegerType::get(op->getContext(), 64);
  if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
    callops.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
  }

  for (auto o : operands) {
    // erase the size to reduce the number of manglings
    if (auto memrefTy = o.getType().dyn_cast<MemRefType>()) {
      auto t = MemRefType::get(
          std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
          memrefTy.getElementType(), memrefTy.getLayout(),
          memrefTy.getMemorySpace());
      callops.push_back(rewriter.create<memref::CastOp>(op->getLoc(), t, o));
    } else {
      callops.push_back(o);
    }
  }
  SmallVector<Type, 16> tys;
  for (auto o : callops)
    tys.push_back(o.getType());

  SmallVector<MemRefType, 16> real_result_tys;
  for (auto t : op->getResultTypes()) {
    if (auto memrefTy = t.dyn_cast<MemRefType>()) {
      auto mrt =
          MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                          memrefTy.getElementType(), memrefTy.getLayout(),
                          memrefTy.getMemorySpace());
      retTys.push_back(mrt);
      real_result_tys.push_back(memrefTy);
    } else {
      retTys.push_back(t);
    }
  }

  auto fn = getMangledFunction(op->getParentOfType<AIE::DeviceOp>(),
                               fnName.str(), callops, retTys);
  auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
      op, retTys, SymbolRefAttr::get(fn), callops);
  int real_result_idx = 0;
  int result_idx = 0;
  for (auto r : op->getResults()) {
    if (auto memrefTy = r.getType().dyn_cast<MemRefType>()) {
      auto t = real_result_tys[real_result_idx++];
      auto c = rewriter.create<memref::CastOp>(op->getLoc(), t,
                                               call.getResult(result_idx));
      r.replaceAllUsesWith(c.getResult());
    } else {
      r.replaceAllUsesWith(call.getResult(result_idx));
    }
    result_idx++;
  }
  return call;
}

static std::optional<AIE::ShimDMAAllocationOp>
getAllocOpForSymbol(AIE::DeviceOp dev, StringRef sym_name) {
  auto sym = dev.lookupSymbol(sym_name);
  if (!sym)
    return std::nullopt;

  auto uses = SymbolTable::getSymbolUses(sym, dev);
  for (auto use : *uses)
    if (auto infoOp = dyn_cast<AIE::ShimDMAAllocationOp>(use.getUser()))
      return infoOp;

  return std::nullopt;
}

namespace {

struct IpuWriteBdExShimTileToFuncPattern
    : public OpConversionPattern<IpuWriteBdExShimTileOp> {
  using OpConversionPattern<IpuWriteBdExShimTileOp>::OpConversionPattern;

  IpuWriteBdExShimTileToFuncPattern(MLIRContext *context,
                                    PatternBenefit benefit = 1)
      : OpConversionPattern<IpuWriteBdExShimTileOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(IpuWriteBdExShimTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getOperands()};
    auto loc = op->getLoc();

    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getColumn()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getColumnNum()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getDdrId()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getBdId()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getBufferLength()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getBufferOffset()));
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
        convertOpToFunction(op, operands, rewriter, "ipu_writebd_shimtile");
    if (call)
      return success();
    else
      return failure();
  }
};

struct IpuSyncToFuncPattern : public OpConversionPattern<IpuSyncOp> {
  using OpConversionPattern<IpuSyncOp>::OpConversionPattern;

  IpuSyncToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<IpuSyncOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(IpuSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getOperands()};
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
    auto call = convertOpToFunction(op, operands, rewriter, "ipu_sync");
    if (call)
      return success();
    else
      return failure();
  }
};

struct IpuWrite32ToFuncPattern : public OpConversionPattern<IpuWrite32Op> {
  using OpConversionPattern<IpuWrite32Op>::OpConversionPattern;

  IpuWrite32ToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<IpuWrite32Op>(context, benefit) {}

  LogicalResult
  matchAndRewrite(IpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 16> operands{adaptor.getOperands()};
    auto loc = op->getLoc();

    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getColumn()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getRow()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getAddress()));
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getValue()));
    auto call = convertOpToFunction(op, operands, rewriter, "ipu_write32");
    if (call)
      return success();
    else
      return failure();
  }
};

struct IpuDmaMemcpyNdToFuncPattern
    : public OpConversionPattern<IpuDmaMemcpyNdOp> {
  using OpConversionPattern<IpuDmaMemcpyNdOp>::OpConversionPattern;

  IpuDmaMemcpyNdToFuncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<IpuDmaMemcpyNdOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(IpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    auto loc = op->getLoc();
    
    operands.push_back(adaptor.getMemref());
    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev) {
      op.emitOpError("couldn't get DeviceOp");
      return failure();
    }

    auto infoOp = getAllocOpForSymbol(dev, op.getMetadata());
    if (!infoOp) {
      op.emitOpError("couldn't find shim_dma_allocation op");
      return failure();
    }

    // auto channelDir = infoOp->getChannelDir();
    // uint32_t ChannelId = infoOp->getChannelIndex();
    // bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    // int col = infoOp->getCol();

    SmallVector<arith::ConstantIndexOp, 3> strides = llvm::map_to_vector(
        llvm::reverse(op.getMixedStrides()), [&](OpFoldResult s) {
          return rewriter.create<arith::ConstantIndexOp>(
              loc, getConstantIntValue(s).value());
        });

    SmallVector<arith::ConstantIndexOp, 4> sizes = llvm::map_to_vector(
        llvm::reverse(op.getMixedSizes()), [&](OpFoldResult s) {
          return rewriter.create<arith::ConstantIndexOp>(
              loc, getConstantIntValue(s).value());
        });

    SmallVector<arith::ConstantIndexOp, 4> offsets = llvm::map_to_vector(
        llvm::reverse(op.getMixedOffsets()), [&](OpFoldResult s) {
          return rewriter.create<arith::ConstantIndexOp>(
              loc, getConstantIntValue(s).value());
        });

    operands.insert(operands.end(), offsets.begin(), offsets.end());
    operands.insert(operands.end(), sizes.begin(), sizes.end());
    operands.insert(operands.end(), strides.begin(), strides.end());

    //bool b = op.getIssueToken();
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, op.getIssueToken()));
    auto call =
        convertOpToFunction(op, operands, rewriter, "ipu_dma_memcpy_nd");
    if (call)
      return success();
    else
      return failure();
  }
};

struct AIEIpuToFuncPass : public AIEIpuToFuncBase<AIEIpuToFuncPass> {
  void runOnOperation() override {

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addIllegalDialect<AIEXDialect>();
    target.addLegalDialect<AIE::AIEDialect, memref::MemRefDialect,
                           scf::SCFDialect, func::FuncDialect,
                           arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<IpuWriteBdExShimTileToFuncPattern, IpuSyncToFuncPattern,
                    IpuWrite32ToFuncPattern, IpuDmaMemcpyNdToFuncPattern>(
        &getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEIpuToFuncPass() {
  return std::make_unique<AIEIpuToFuncPass>();
}
