//===- AIELowerMemcpy.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIETokenAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"
#include <string>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

static TileOp srcTileOp(xilinx::AIEX::MemcpyOp op) {
  return llvm::dyn_cast<xilinx::AIE::TileOp>(op.getSrcTile().getDefiningOp());
}
static TileOp dstTileOp(xilinx::AIEX::MemcpyOp op) {
  return llvm::dyn_cast<xilinx::AIE::TileOp>(op.getDstTile().getDefiningOp());
}

struct LowerAIEMemcpy : public OpConversionPattern<MemcpyOp> {
  using OpConversionPattern<MemcpyOp>::OpConversionPattern;

  LowerAIEMemcpy(MLIRContext *context, llvm::StringMap<LockOp> *tokenLocks,
                 PatternBenefit benefit = 1)
      : OpConversionPattern<MemcpyOp>(context, benefit),
        tokenLocks(tokenLocks) {}

  llvm::StringMap<LockOp> *tokenLocks;

  void createDMABlocksAndOps(MemOp &mem, StringRef tokenName, int acquireTknVal,
                             int releaseTknVal, Value buf, int offset, int len,
                             DMAChannelDir dmaDir, int channelIndex,
                             ConversionPatternRewriter &rewriter,
                             LockOp tokenLock) const {

    Region &r = mem.getBody();
    Block &endBlock = r.back();
    Block *dmaBlock = rewriter.createBlock(&endBlock);
    Block *bdBlock = rewriter.createBlock(&endBlock);

    rewriter.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(rewriter, rewriter.getUnknownLoc(), dmaDir, channelIndex,
                       /*repeatCount*/ 0, bdBlock, &endBlock);

    // Setup bd Block
    // It should contain locking operations (lock or token) as well as DMABD op
    // for specifying DMA Block description (which buffer type (A/B), transfer
    // length/address, etc.)
    rewriter.setInsertionPointToStart(bdBlock);
    if (tokenLock)
      UseLockOp::create(rewriter, rewriter.getUnknownLoc(),
                        tokenLock.getResult(), LockAction::AcquireGreaterEqual,
                        acquireTknVal);
    else
      UseTokenOp::create(rewriter, rewriter.getUnknownLoc(), tokenName,
                         acquireTknVal, LockAction::Acquire);
    DMABDOp::create(rewriter, rewriter.getUnknownLoc(), buf, offset, len);
    if (tokenLock)
      UseLockOp::create(rewriter, rewriter.getUnknownLoc(),
                        tokenLock.getResult(), LockAction::Release,
                        releaseTknVal);
    else
      UseTokenOp::create(rewriter, rewriter.getUnknownLoc(), tokenName,
                         releaseTknVal, LockAction::Release);
    NextBDOp::create(rewriter, rewriter.getUnknownLoc(), &endBlock);
  }

  ShimDMAAllocationOp getOrCreateShimAlloc(DeviceOp device, TileOp shimTile,
                                           DMAChannelDir dir,
                                           int64_t channelIdx,
                                           StringRef symName,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc) const {
    if (auto existing = ShimDMAAllocationOp::getForSymbol(device, symName))
      return existing;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(shimTile.getOperation());
    return rewriter.create<ShimDMAAllocationOp>(
        loc, rewriter.getStringAttr(symName), shimTile.getResult(),
        DMAChannelDirAttr::get(rewriter.getContext(), dir),
        rewriter.getI64IntegerAttr(channelIdx), rewriter.getBoolAttr(false),
        nullptr);
  }

  LogicalResult
  matchAndRewrite(MemcpyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcBuf = op.getSrcBuf();
    Value dstBuf = op.getDstBuf();

    auto bufferBaseAddress = [](Value v) -> int {
      if (auto bufOp = v.getDefiningOp<xilinx::AIE::BufferOp>()) {
        if (auto addr = bufOp.getAddress())
          return *addr;
      }
      return 0;
    };

    StringRef tokenName = op.getTokenName();
    int acquireTknVal = op.getAcquireTokenValue();
    int releaseTknVal = op.getReleaseTokenValue();
    int srcOffset = op.getSrcOffsetValue();
    int dstOffset = op.getDstOffsetValue();
    int srcLen = op.getSrcLenValue();
    int dstLen = op.getDstLenValue();

    TileOp srcTile = srcTileOp(op);
    TileOp dstTile = dstTileOp(op);

    MemOp srcMem = srcTile ? srcTile.getMemOp() : MemOp();
    MemOp dstMem = dstTile ? dstTile.getMemOp() : MemOp();

    // Handle shim/external endpoints by emitting an NPU DMA configured for the
    // shim channel and configure the non-shim endpoint with the token lock so
    // core-side DMA still participates in the semaphore protocol.
    if (!srcMem || !dstMem) {
      bool srcIsShim = srcTile && srcTile.isShimTile();
      bool dstIsShim = dstTile && dstTile.isShimTile();

      bool shimIsSource = srcIsShim || (!dstIsShim && !srcMem);
      TileOp shimTile = shimIsSource ? srcTile : dstTile;
      if (!shimTile || !(shimTile.isShimTile()))
        return op.emitOpError("expected shim tile endpoint");

      TileOp nonShimTile = shimIsSource ? dstTile : srcTile;
      LockOp tokenLock;
      if (tokenLocks) {
        auto lockIt = tokenLocks->find(tokenName);
        if (lockIt != tokenLocks->end())
          tokenLock = lockIt->second;
      }

      DMAChannelDir dir =
          shimIsSource ? DMAChannelDir::MM2S : DMAChannelDir::S2MM;
      // Each direction has its own channel space; use channel 0 for both.
      int64_t channelIdx = 0;
      int64_t baseAddr = bufferBaseAddress(shimIsSource ? srcBuf : dstBuf);
      int64_t offset = baseAddr + (shimIsSource ? srcOffset : dstOffset);
      int64_t len = shimIsSource ? srcLen : dstLen;
      Value memref = shimIsSource ? srcBuf : dstBuf;

      auto device = op->getParentOfType<DeviceOp>();
      if (!device) {
        return op.emitOpError("expected parent device for shim DMA lowering");
      }

      std::string symName = "shim_dma_" + tokenName.str();
      symName += "_" + std::to_string(shimTile.getCol());
      symName += "_" + std::to_string(shimTile.getRow());
      symName += "_" + std::to_string(channelIdx);
      ShimDMAAllocationOp shimAlloc = getOrCreateShimAlloc(
          device, shimTile, dir, channelIdx, symName, rewriter, op.getLoc());

      auto staticOffsets = rewriter.getDenseI64ArrayAttr(
          {0, 0, 0, static_cast<int64_t>(offset)});
      auto staticSizes =
          rewriter.getDenseI64ArrayAttr({1, 1, 1, static_cast<int64_t>(len)});
      auto staticStrides = rewriter.getDenseI64ArrayAttr({0, 0, 0, 1});
      auto zero = rewriter.getI64IntegerAttr(0);
      auto issueToken = rewriter.getBoolAttr(!shimIsSource);

      if (nonShimTile && nonShimTile.isShimTile())
        return op.emitOpError(
            "expected non-shim tile paired with shim endpoint");

      // Emit runtime writebd for the non-shim endpoint so its DMA uses the
      // lock shared with the core.
      if (tokenLock && nonShimTile) {
        int lockId = tokenLock.getLockIDValue();
        int bdLen = shimIsSource ? dstLen : srcLen;
        auto colAttr = rewriter.getI32IntegerAttr(nonShimTile.getCol());
        auto rowAttr = rewriter.getI32IntegerAttr(nonShimTile.getRow());
        bool nonShimIsDst = nonShimTile == dstTile;
        int bdId = nonShimIsDst ? 0 : 1;
        auto bdIdAttr = rewriter.getI32IntegerAttr(bdId);
        auto lengthAttr = rewriter.getI32IntegerAttr(bdLen);
        int bdOffset = shimIsSource ? dstOffset : srcOffset;
        int baseAddr = bufferBaseAddress(nonShimIsDst ? dstBuf : srcBuf);
        bdOffset += baseAddr;
        auto offsetAttr = rewriter.getI32IntegerAttr(bdOffset);
        auto zeroAttr = rewriter.getI32IntegerAttr(0);
        auto oneAttr = rewriter.getI32IntegerAttr(1);
        auto d0SizeAttr = rewriter.getI32IntegerAttr(bdLen);
        auto d0StrideAttr = zeroAttr;
        auto d1SizeAttr = oneAttr;
        auto d1StrideAttr = zeroAttr;
        auto d2SizeAttr = oneAttr;
        auto d2StrideAttr = zeroAttr;
        auto iterCurrentAttr = zeroAttr;
        auto iterSizeAttr = zeroAttr;
        auto iterStrideAttr = zeroAttr;
        auto nextBdAttr = zeroAttr;
        auto useNextBdAttr = zeroAttr;
        auto validBdAttr = oneAttr;
        // For shim->tile (input) we only release the lock for the core to
        // acquire. For tile->shim (output) we acquire the lock the core
        // released and then release to the next value.
        bool acquireLock = !shimIsSource;
        auto lockRelValAttr = rewriter.getI32IntegerAttr(releaseTknVal);
        auto lockRelIdAttr = rewriter.getI32IntegerAttr(lockId);
        auto lockAcqEnableAttr =
            rewriter.getI32IntegerAttr(acquireLock ? 1 : 0);
        auto lockAcqValAttr =
            rewriter.getI32IntegerAttr(acquireLock ? acquireTknVal : 0);
        auto lockAcqIdAttr =
            rewriter.getI32IntegerAttr(acquireLock ? lockId : 0);
        auto burstLenAttr = zeroAttr;

        rewriter.setInsertionPoint(op);
        rewriter.create<NpuWriteBdOp>(
            op.getLoc(), colAttr, bdIdAttr, lengthAttr, offsetAttr, zeroAttr,
            zeroAttr, zeroAttr, zeroAttr, d0SizeAttr, d0StrideAttr, d1SizeAttr,
            d1StrideAttr, d2SizeAttr, d2StrideAttr, iterCurrentAttr,
            iterSizeAttr, iterStrideAttr, nextBdAttr, rowAttr, useNextBdAttr,
            validBdAttr, lockRelValAttr, lockRelIdAttr, lockAcqEnableAttr,
            lockAcqValAttr, lockAcqIdAttr, zeroAttr, zeroAttr, zeroAttr,
            zeroAttr, zeroAttr, zeroAttr, burstLenAttr);

        auto dirAttr = DMAChannelDirAttr::get(
            rewriter.getContext(),
            nonShimIsDst ? DMAChannelDir::S2MM : DMAChannelDir::MM2S);
        auto channelAttr = zeroAttr;
        auto repeatCountAttr = zeroAttr;
        auto issueTokenAttr = rewriter.getBoolAttr(nonShimIsDst);
        rewriter.create<NpuPushQueueOp>(op.getLoc(), colAttr, rowAttr, dirAttr,
                                        channelAttr, issueTokenAttr,
                                        repeatCountAttr, bdIdAttr);
      }

      rewriter.setInsertionPoint(op);
      auto dma = rewriter.create<NpuDmaMemcpyNdOp>(
          op.getLoc(), memref, ValueRange{}, ValueRange{}, ValueRange{},
          staticOffsets, staticSizes, staticStrides, nullptr,
          SymbolRefAttr::get(shimAlloc.getSymNameAttr()),
          rewriter.getI64IntegerAttr(shimIsSource ? 0 : 1), issueToken, zero,
          zero, zero, zero, zero, zero, zero);
      if (issueToken.getValue()) {
        rewriter.setInsertionPointAfter(dma);
        rewriter.create<NpuDmaWaitOp>(
            op.getLoc(), SymbolRefAttr::get(shimAlloc.getSymNameAttr()));
      }
      rewriter.eraseOp(op);
      return success();
    }

    // Non-shim path: configure tile DMA BD in the runtime sequence and keep
    // existing lowering for the tile-side DMA graph. The BD references the
    // semaphore lock associated with this token.
    TileOp nonShimTile = dstTile;
    if (srcTile && !srcTile.isShimTile())
      nonShimTile = srcTile;
    LockOp tokenLock;
    if (tokenLocks) {
      auto lockIt = tokenLocks->find(tokenName);
      if (lockIt != tokenLocks->end())
        tokenLock = lockIt->second;
    }

    if (tokenLock && nonShimTile) {
      int lockId = tokenLock.getLockIDValue();
      int bdLen = (nonShimTile == dstTile) ? dstLen : srcLen;
      int bdOffset = (nonShimTile == dstTile) ? dstOffset : srcOffset;
      int baseAddr =
          bufferBaseAddress((nonShimTile == dstTile) ? dstBuf : srcBuf);
      bdOffset += baseAddr;
      auto colAttr = rewriter.getI32IntegerAttr(nonShimTile.getCol());
      auto rowAttr = rewriter.getI32IntegerAttr(nonShimTile.getRow());
      bool nonShimIsDst = nonShimTile == dstTile;
      int bdId = nonShimIsDst ? 0 : 1;
      auto bdIdAttr = rewriter.getI32IntegerAttr(bdId);
      auto lengthAttr = rewriter.getI32IntegerAttr(bdLen);
      auto offsetAttr = rewriter.getI32IntegerAttr(bdOffset);
      auto zeroAttr = rewriter.getI32IntegerAttr(0);
      auto oneAttr = rewriter.getI32IntegerAttr(1);
      auto d0SizeAttr = rewriter.getI32IntegerAttr(bdLen);
      auto d0StrideAttr = oneAttr;
      auto d1SizeAttr = oneAttr;
      auto d1StrideAttr = zeroAttr;
      auto d2SizeAttr = zeroAttr;
      auto d2StrideAttr = zeroAttr;
      auto iterCurrentAttr = zeroAttr;
      auto iterSizeAttr = zeroAttr;
      auto iterStrideAttr = zeroAttr;
      auto nextBdAttr = zeroAttr;
      auto useNextBdAttr = zeroAttr;
      auto validBdAttr = oneAttr;
      auto lockRelValAttr = rewriter.getI32IntegerAttr(releaseTknVal);
      auto lockRelIdAttr = rewriter.getI32IntegerAttr(lockId);
      auto lockAcqEnableAttr = oneAttr;
      auto lockAcqValAttr = rewriter.getI32IntegerAttr(acquireTknVal);
      auto lockAcqIdAttr = rewriter.getI32IntegerAttr(lockId);
      auto burstLenAttr = zeroAttr;

      rewriter.setInsertionPoint(op);
      rewriter.create<NpuWriteBdOp>(
          op.getLoc(), colAttr, bdIdAttr, lengthAttr, offsetAttr, zeroAttr,
          zeroAttr, zeroAttr, zeroAttr, d0SizeAttr, d0StrideAttr, d1SizeAttr,
          d1StrideAttr, d2SizeAttr, d2StrideAttr, iterCurrentAttr, iterSizeAttr,
          iterStrideAttr, nextBdAttr, rowAttr, useNextBdAttr, validBdAttr,
          lockRelValAttr, lockRelIdAttr, lockAcqEnableAttr, lockAcqValAttr,
          lockAcqIdAttr, zeroAttr, zeroAttr, zeroAttr, zeroAttr, zeroAttr,
          zeroAttr, burstLenAttr);

      auto dirAttr = DMAChannelDirAttr::get(rewriter.getContext(),
                                            nonShimIsDst ? DMAChannelDir::S2MM
                                                         : DMAChannelDir::MM2S);
      auto channelAttr = zeroAttr;
      auto repeatCountAttr = zeroAttr;
      auto issueTokenAttr = rewriter.getBoolAttr(nonShimIsDst);
      rewriter.create<NpuPushQueueOp>(op.getLoc(), colAttr, rowAttr, dirAttr,
                                      channelAttr, issueTokenAttr,
                                      repeatCountAttr, bdIdAttr);
    }

    // createDMABlocksAndOps(srcMem, tokenName, acquireTknVal, releaseTknVal,
    //                       srcBuf, srcOffset, srcLen, DMAChannelDir::MM2S, 0,
    //                       rewriter, tokenLock);
    // createDMABlocksAndOps(dstMem, tokenName, acquireTknVal, releaseTknVal,
    //                       dstBuf, dstOffset, dstLen, DMAChannelDir::S2MM, 0,
    //                       rewriter, tokenLock);

    rewriter.eraseOp(op);
    return success();
  }
};

struct AIELowerMemcpyPass : public AIELowerMemcpyBase<AIELowerMemcpyPass> {
  void runOnOperation() override {

    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());

    // Map logical tokens to physical semaphore locks (one per token) on the
    // destination/core tile. We allocate a lock ID per tile deterministically
    // so the subsequent writebd configuration can reference the same ID.
    llvm::StringMap<LockOp> tokenLocks;
    DenseMap<std::pair<int, int>, int> nextLockId;

    auto getOrCreateLockForToken = [&](StringRef tokenName,
                                       TileOp targetTile) -> LockOp {
      if (auto it = tokenLocks.find(tokenName); it != tokenLocks.end())
        return it->second;

      std::pair<int, int> tileKey{targetTile.getCol(), targetTile.getRow()};
      int &lockId = nextLockId[tileKey];
      // Create the lock with an explicit ID to keep the mapping stable.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(targetTile);
      std::string lockSym = (tokenName + "_lock").str();
      auto lock = builder.create<LockOp>(
          device.getLoc(), targetTile.getResult(),
          builder.getI32IntegerAttr(lockId), builder.getI32IntegerAttr(0),
          builder.getStringAttr(lockSym));
      lockId++;
      tokenLocks[tokenName] = lock;
      return lock;
    };

    // First walk memcpys to learn token → tile mapping and allocate locks.
    SmallVector<MemcpyOp> memcpyOps;
    device.walk([&](MemcpyOp op) { memcpyOps.push_back(op); });
    for (auto op : memcpyOps) {
      TileOp srcTile = dyn_cast<TileOp>(op.getSrcTile().getDefiningOp());
      TileOp dstTile = dyn_cast<TileOp>(op.getDstTile().getDefiningOp());
      // Prefer a non-shim/core tile for the semaphore; fall back to dst.
      TileOp targetTile = dstTile;
      if (srcTile && !srcTile.isShimTile())
        targetTile = srcTile;
      if (!targetTile)
        continue;
      getOrCreateLockForToken(op.getTokenName(), targetTile);
    }

    // Replace useToken with use_lock using AcquireGreaterEqual/Release on AIE2
    // counting semaphores so core code blocks on the same lock as DMA.
    device.walk([&](UseTokenOp op) {
      auto name = op.getTokenName();
      auto it = tokenLocks.find(name);
      if (it == tokenLocks.end())
        return;
      LockOp lock = it->second;
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(op);
      LockAction action =
          op.acquire() ? LockAction::AcquireGreaterEqual : LockAction::Release;
      builder.create<UseLockOp>(op.getLoc(), lock.getResult(), action,
                                op.getValue());
      op.erase();
    });

    // Setup FlowOps
    // Since memcpy moves data from one memory module to another, we use
    // WireBundle::DMA for both the source and the destination In addition, we
    // only have two DMA ports per each direction (MM2S/S2MM), and in a
    // circuit-switch mode, dest port/channel sharing is not possible.
    // Therefore, we will generate error if the number of logical flows
    // (streams) targeting the same destination (S2MM) is more than 2.
    // Walk the entire device to catch memcpy ops that may appear inside
    // runtime_sequences or other nested regions.
    DenseMap<Value, int> destChannel;
    for (auto op : memcpyOps) {
      builder.setInsertionPoint(device.getBody()->getTerminator());
      TileOp srcTile = dyn_cast<TileOp>(op.getSrcTile().getDefiningOp());
      TileOp dstTile = dyn_cast<TileOp>(op.getDstTile().getDefiningOp());
      // TODO: perhaps a better approach is to not assert here, but rather have
      // a subsequent pass that legally relocates the ports
      assert(destChannel[op.getDstTile()] <= 2 &&
             "Could not allocate more than two dest. channel when creating "
             "FlowOp");
      FlowOp::create(builder, builder.getUnknownLoc(), srcTile, WireBundle::DMA,
                     0, dstTile, WireBundle::DMA, destChannel[op.getDstTile()]);
      destChannel[op.getDstTile()]++;
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    target.addLegalOp<DMAStartOp>();
    target.addLegalOp<DMABDOp>();
    target.addLegalOp<UseTokenOp>();
    target.addLegalOp<UseLockOp>();
    target.addLegalOp<NextBDOp>();
    target.addLegalOp<NpuDmaMemcpyNdOp>();
    target.addLegalOp<NpuWriteBdOp>();
    target.addLegalOp<NpuPushQueueOp>();
    target.addLegalOp<NpuDmaWaitOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();

    patterns.insert<LowerAIEMemcpy>(&getContext(), &tokenLocks);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIELowerMemcpyPass() {
  return std::make_unique<AIELowerMemcpyPass>();
}
