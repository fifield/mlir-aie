//===- AIEDmaToNpu.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"
#include "aie/Dialect/AIEX/Utils/DmaQueueModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <cstdint>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDMATONPU
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct Write32SymToAddr : OpConversionPattern<NpuWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  Write32SymToAddr(MLIRContext *context, const AIE::NamedOpTable &names)
      : OpConversionPattern(context), names(names) {}

  const AIE::NamedOpTable &names;

  LogicalResult
  matchAndRewrite(NpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getBuffer())
      return failure();

    std::optional<uint32_t> address = op.getAbsoluteAddress(&names);
    if (!address.has_value()) {
      return failure();
    }

    Value addressVal = createConstantI32(rewriter, op->getLoc(), *address);
    rewriter.replaceOpWithNewOp<NpuWrite32Op>(
        op, addressVal, adaptor.getValue(), nullptr, nullptr, nullptr);
    return success();
  }
};

struct BlockWriteSymToAddr : OpConversionPattern<NpuBlockWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  BlockWriteSymToAddr(MLIRContext *context, const AIE::NamedOpTable &names)
      : OpConversionPattern(context), names(names) {}

  const AIE::NamedOpTable &names;

  LogicalResult
  matchAndRewrite(NpuBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getBuffer())
      return failure();

    std::optional<uint32_t> address = op.getAbsoluteAddress(&names);
    if (!address.has_value()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(op, *address, op.getData(),
                                                 nullptr, nullptr, nullptr);
    return success();
  }
};

struct MaskWrite32SymToAddr : OpConversionPattern<NpuMaskWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  MaskWrite32SymToAddr(MLIRContext *context, const AIE::NamedOpTable &names)
      : OpConversionPattern(context), names(names) {}

  const AIE::NamedOpTable &names;

  LogicalResult
  matchAndRewrite(NpuMaskWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getBuffer())
      return failure();

    std::optional<uint32_t> absoluteAddress = op.getAbsoluteAddress(&names);
    if (!absoluteAddress.has_value()) {
      return failure();
    }

    Value addressVal =
        createConstantI32(rewriter, op->getLoc(), *absoluteAddress);
    rewriter.replaceOpWithNewOp<NpuMaskWrite32Op>(
        op, addressVal, adaptor.getValue(), adaptor.getMask(), nullptr, nullptr,
        nullptr);
    return success();
  }
};

struct MaskPollSymToAddr : OpConversionPattern<NpuMaskPollOp> {
  using OpConversionPattern::OpConversionPattern;

  MaskPollSymToAddr(MLIRContext *context, const AIE::NamedOpTable &names)
      : OpConversionPattern(context), names(names) {}

  const AIE::NamedOpTable &names;

  LogicalResult
  matchAndRewrite(NpuMaskPollOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getBuffer())
      return failure();

    std::optional<uint32_t> absoluteAddress = op.getAbsoluteAddress(&names);
    if (!absoluteAddress)
      return failure();

    Value addressVal =
        createConstantI32(rewriter, op->getLoc(), *absoluteAddress);
    rewriter.replaceOpWithNewOp<NpuMaskPollOp>(
        op, addressVal, adaptor.getValue(), adaptor.getMask(), nullptr, nullptr,
        nullptr);
    return success();
  }
};

struct RtpToWrite32Pattern : OpConversionPattern<NpuWriteRTPOp> {
  using OpConversionPattern::OpConversionPattern;

  RtpToWrite32Pattern(MLIRContext *context, const AIE::NamedOpTable &names)
      : OpConversionPattern(context), names(names) {}

  const AIE::NamedOpTable &names;

  LogicalResult
  matchAndRewrite(NpuWriteRTPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto buffer = names.lookup<AIE::BufferOp>(op.getBufferAttr().getAttr());
    if (!buffer) {
      op->emitError("buffer '" + op.getBuffer() + "' not found in device");
      return failure();
    }

    auto bufferAddress = buffer.getAddress();
    if (!bufferAddress) {
      op->emitError("buffer must have address assigned");
      return failure();
    }
    AIE::TileOp tile = buffer.getTileOp();

    uint32_t idx = op.getIndex() * sizeof(uint32_t);
    uint32_t address = *bufferAddress + idx;

    NpuWrite32Op::create(rewriter, op->getLoc(),
                         createConstantI32(rewriter, op->getLoc(), address),
                         adaptor.getValue(), nullptr,
                         rewriter.getI32IntegerAttr(tile.getCol()),
                         rewriter.getI32IntegerAttr(tile.getRow()));

    rewriter.eraseOp(op);
    return success();
  }
};

struct PushQueuetoWrite32Pattern : OpConversionPattern<NpuPushQueueOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  PushQueuetoWrite32Pattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuPushQueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const auto &tm = AIE::getTargetModel(op);
    uint32_t ctrl_offset = tm.getDmaControlAddress(
        op.getColumn(), op.getRow(), op.getChannel(), op.getDirection());

    // control packet for issuing token
    if (op.getIssueToken()) {
      // set the task-complete-token controller ID field in the dma control
      // register
      AIE::TileOp shimTile = AIE::TileOp::getOrCreate(
          rewriter, op->getParentOfType<AIE::DeviceOp>(), op.getColumn(),
          op.getRow());
      if (shimTile->hasAttr("controller_id")) {
        AIE::PacketInfoAttr controller_id_attr =
            shimTile->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
        uint32_t data = controller_id_attr.assignedId() << 8;
        uint32_t mask = 0x00001F00;
        NpuMaskWrite32Op::create(
            rewriter, op->getLoc(),
            createConstantI32(rewriter, op->getLoc(), ctrl_offset),
            createConstantI32(rewriter, op->getLoc(), data),
            createConstantI32(rewriter, op->getLoc(), mask), nullptr, nullptr,
            nullptr);
      }
    }

    // the offset of the task queue register in the tile
    uint32_t queue_offset = ctrl_offset + 0x4;

    // Command word: bd_id in the START_BD_ID field, repeat_count [23:16],
    // issue-token bit [31]. START_BD_ID is 6 bits on a mem tile (48 BDs) and 4
    // bits on core/shim tiles, so mask to the tile class instead of a flat 0xF
    // (a flat 0xF silently truncates a mem-tile head bd_id >= 16).
    // bd_id and repeat_count may be runtime SSA; all-constant folds to one
    // constant (byte-identical to the static path), else built with arith.
    Location loc = op->getLoc();
    auto i32ty = rewriter.getIntegerType(32);
    uint32_t bdIdMask =
        tm.isMemTile(op.getColumn(), op.getRow()) ? 0x3Fu : 0xFu;
    std::optional<uint32_t> bd_id = getConstantIntOperand(op.getBdId());
    std::optional<uint64_t> repeat_cnt =
        getConstantInt64Operand(op.getRepeatCount());
    uint32_t issueBit = op.getIssueToken() ? 0x80000000 : 0;

    // A scratchpad repeat count: the host writes it already shifted into the
    // repeat field, and the update adds it to the rest of the command word.
    // The verifier holds bd_id to a constant multiple of 4, which the
    // update's clearing of bits [1:0] keeps.
    if (IntegerAttr idx = op.getRepeatStateTableIdxAttr()) {
      NpuUpdateFromScratchpadOp::create(
          rewriter, loc, static_cast<uint8_t>(idx.getUInt()),
          StateTableFunc::Incr,
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          /*func_arg=*/(*bd_id & bdIdMask) | issueBit,
          /*address=*/queue_offset, /*buffer=*/nullptr, /*column=*/nullptr,
          /*row=*/nullptr);
      rewriter.eraseOp(op);
      return success();
    }

    Value cmdVal;
    if (bd_id && repeat_cnt) {
      cmdVal = createConstantI32(rewriter, loc,
                                 (*bd_id & bdIdMask) |
                                     ((*repeat_cnt & 0xFF) << 16) | issueBit);
    } else {
      // (bd_id & bdIdMask) | ((repeat & 0xFF) << 16) | issueBit, as arith over
      // the runtime operands (a constant field folds to its contribution).
      // A runtime repeat_count is masked to its 8-bit field below, so an
      // out-of-range value would silently wrap to a different number of
      // executions. Refuse the dispatch instead (host-side guard); the
      // unsigned compare also rejects a negative count (zero executions).
      Value repeat;
      if (repeat_cnt) {
        repeat = createConstantI32(rewriter, loc, *repeat_cnt);
      } else {
        Value repeat64 =
            getAsI64(rewriter, loc, OpFoldResult(op.getRepeatCount()));
        if (!repeat64)
          return failure();
        uint32_t maxRepeat = tm.getMaxRepeatCount();
        Value inRange = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::ule, repeat64,
            arith::ConstantOp::create(rewriter, loc,
                                      rewriter.getI64IntegerAttr(maxRepeat)));
        if (failed(emitRuntimeCheck(
                rewriter, loc, inRange,
                "a runtime DMA repeat count exceeds the task queue's [0:" +
                    Twine(maxRepeat) + "] range (at most " +
                    Twine(maxRepeat + 1) + " executions)")))
          return failure();
        repeat = rewriter.createOrFold<arith::TruncIOp>(loc, i32ty, repeat64);
      }
      Value cmd = createConstantI32(rewriter, loc, issueBit);
      Value bdField = rewriter.createOrFold<arith::AndIOp>(
          loc, getAsValue(rewriter, loc, op.getBdId(), i32ty),
          createConstantI32(rewriter, loc, bdIdMask));
      cmd = rewriter.createOrFold<arith::OrIOp>(loc, cmd, bdField);
      Value masked = rewriter.createOrFold<arith::AndIOp>(
          loc, repeat, createConstantI32(rewriter, loc, 0xFF));
      Value shifted = rewriter.createOrFold<arith::ShLIOp>(
          loc, masked, createConstantI32(rewriter, loc, 16));
      cmdVal = rewriter.createOrFold<arith::OrIOp>(loc, cmd, shifted);
    }

    NpuWrite32Op::create(
        rewriter, op->getLoc(),
        createConstantI32(rewriter, op->getLoc(), queue_offset), cmdVal,
        nullptr, nullptr, nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct DmaToNpuPattern : OpConversionPattern<NpuDmaMemcpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  DmaToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Any runtime (SSA) offset/size/stride takes the dynamic path; a
    // fully-constant descriptor takes the static path below. The op verifier
    // has already enforced the supported scope for the dynamic case (shim NOC,
    // no padding, realizable/in-range constants).
    bool allOffsetsConstant =
        llvm::all_of(op.getMixedOffsets(),
                     [](OpFoldResult s) { return getConstantIntValue(s); });
    bool allSizesConstant =
        llvm::all_of(op.getMixedSizes(),
                     [](OpFoldResult s) { return getConstantIntValue(s); });
    bool allStridesConstant =
        llvm::all_of(op.getMixedStrides(),
                     [](OpFoldResult s) { return getConstantIntValue(s); });
    if (!allOffsetsConstant || !allSizesConstant || !allStridesConstant)
      return lowerDynamic(op, adaptor, rewriter);

    const auto &targetModel = AIE::getTargetModel(op);
    BaseMemRefType bufferType = op.getMemref().getType();
    auto *ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);

    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();

    AIE::ShimDMAAllocationOp infoOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, op.getMetadata().getRootReference());
    if (!infoOp) {
      return op->emitOpError("couldn't find shim_dma_allocation op.");
    }

    AIE::TileOp shimTile = infoOp.getTileOp();
    if (!shimTile) {
      return op->emitOpError(
          "shim_dma_allocation op must reference a valid TileOp.");
    }

    auto channelDir = infoOp.getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int tileCol = shimTile.getCol();
    int tileRow = shimTile.getRow();

    // initialize fields to zero
    auto column = zero;
    auto bd_id = zero;
    auto buffer_length = zero;
    auto buffer_offset = zero;
    auto enable_packet = zero;
    auto out_of_order_id = zero;
    auto packet_id = zero;
    auto packet_type = zero;
    auto d0_size = zero;
    auto d0_stride = zero;
    auto d1_size = zero;
    auto d1_stride = zero;
    auto d2_size = zero;
    auto d2_stride = zero;
    auto iteration_current = zero;
    auto iteration_size = zero;
    auto iteration_stride = zero;
    auto next_bd = zero;
    auto row = zero;
    auto use_next_bd = zero;
    auto valid_bd = zero;
    auto lock_rel_val = zero;
    auto lock_rel_id = zero;
    auto lock_acq_enable = zero;
    auto lock_acq_val = zero;
    auto lock_acq_id = zero;
    auto d0_zero_before = zero;
    auto d1_zero_before = zero;
    auto d2_zero_before = zero;
    auto d0_zero_after = zero;
    auto d1_zero_after = zero;
    auto d2_zero_after = zero;
    auto burst_length = zero;
    auto axcache = zero;

    auto issue_token = BoolAttr::get(ctx, false);
    auto repeat_count = zero;
    llvm::SmallVector<int64_t, 4> inputSizes = llvm::map_to_vector(
        llvm::reverse(op.getMixedSizes()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    llvm::SmallVector<int64_t, 4> inputStrides = llvm::map_to_vector(
        llvm::reverse(op.getMixedStrides()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    // A contiguous row-major ND access on a shim NOC tile is lowered to linear
    // mode (d0_size=d1_size=0) just like an already-canonical linear transfer.
    // This allows naturally-expressed multidimensional transfers (e.g., a 2D
    // image as [height, width]) without hitting the 10-bit ND wrap-size limit.
    bool isLinear = isLinearModeTransfer(targetModel, tileCol, tileRow,
                                         inputSizes, inputStrides);
    if (!isLinear && op.getLengthStateTableIdxAttr())
      AIE::placeRuntimeLengthDimension(inputSizes, inputStrides);
    llvm::SmallVector<int64_t, 4> sizes(4);
    llvm::SmallVector<int64_t, 4> strides(4);
    getHardwareStridesWraps(targetModel, op, bufferType, inputSizes,
                            inputStrides, sizes, strides);

    // column
    column = IntegerAttr::get(i32ty, tileCol);

    // row
    row = IntegerAttr::get(i32ty, tileRow);

    if (failed(verifyStridesWraps(
            op, [&] { return op->emitOpError(); }, bufferType, tileCol, tileRow,
            inputSizes, inputStrides, sizes, strides, isLinear))) {
      return failure();
    }

    // bd_id
    bd_id = IntegerAttr::get(i32ty, op.getId());

    // buffer_length
    uint64_t buffer_length_val = inputSizes[0] * op.getElementTypeBitwidth() /
                                 targetModel.getAddressGenGranularity();
    if (inputSizes.size() > 1) {
      for (size_t i = 1; i < std::min(inputSizes.size(), (size_t)3); i++) {
        buffer_length_val *= inputSizes[i];
      }
    }
    buffer_length = IntegerAttr::get(i32ty, buffer_length_val);

    // buffer_offset - zero because the complete address is set by the patch op
    buffer_offset = IntegerAttr::get(i32ty, 0);

    // enable_packet
    if (auto packetInfo =
            op.getPacket() ? op.getPacket() : infoOp.getPacket()) {
      enable_packet = IntegerAttr::get(i32ty, 1);
      packet_type = IntegerAttr::get(i32ty, packetInfo->getPktType());
      packet_id = IntegerAttr::get(i32ty, packetInfo->assignedId());
    }

    // out_of_order_id - stays 0; senders stamp it via aie.dma_bd,
    // dma_configure_task, or npu.writebd

    if (!isLinear) {
      // d0_size, d0_stride
      d0_size = IntegerAttr::get(i32ty, sizes[0]);
      d0_stride = IntegerAttr::get(i32ty, strides[0]);

      // d1_size, d1_stride
      d1_size = IntegerAttr::get(i32ty, sizes[1]);
      d1_stride = IntegerAttr::get(i32ty, strides[1]);

      // d2_stride
      d2_stride = IntegerAttr::get(i32ty, strides[2]);

      // d2_size
      if (targetModel.isMemTile(tileCol, 0)) // Need to be any row
        d2_size = IntegerAttr::get(i32ty, sizes[2]);
      else
        d2_size = IntegerAttr::get(i32ty, 0);
    }
    // iteration_current, iteration_size, iteration_stride, repeat_count
    if (inputSizes[3] > 1) {
      if (inputStrides[3] > 0) {
        iteration_size = IntegerAttr::get(i32ty, sizes[3]);
        iteration_stride = IntegerAttr::get(i32ty, strides[3]);
      } else {
        // We allow users to encode the repeat_count as a dimension 3 stride
        // of 0. This must lower to a iteration wrap of 0, so no stride is
        // ever added. We then repeat the BD using the repeat_count in
        // NpuPushQueueOp.
        iteration_size = zero;
        iteration_stride = zero;
      }
    }
    repeat_count = IntegerAttr::get(i32ty, sizes[3]);

    // next_bd

    // use_next_bd

    // valid_bd
    valid_bd = IntegerAttr::get(i32ty, 1);

    // lock_rel_val

    // lock_rel_id

    // lock_acq_enable

    // lock_acq_val

    // lock_acq_id

    // d0_zero_before
    d0_zero_before = IntegerAttr::get(i32ty, op.getD0ZeroBefore());

    // d1_zero_before
    d1_zero_before = IntegerAttr::get(i32ty, op.getD1ZeroBefore());

    // d2_zero_before
    d2_zero_before = IntegerAttr::get(i32ty, op.getD2ZeroBefore());

    // d0_zero_after
    d0_zero_after = IntegerAttr::get(i32ty, op.getD0ZeroAfter());

    // d1_zero_after
    d1_zero_after = IntegerAttr::get(i32ty, op.getD1ZeroAfter());

    // d2_zero_after
    d2_zero_after = IntegerAttr::get(i32ty, op.getD2ZeroAfter());

    // burst_size
    burst_length = IntegerAttr::get(i32ty, op.getBurstLength());

    // axcache; only meaningful on the AXI-MM side, so left unset elsewhere to
    // match the dma_task path (and NpuWriteBdOp's verifier).
    if (targetModel.isShimNOCTile(tileCol, tileRow))
      axcache = IntegerAttr::get(i32ty, op.getAxcacheOrDefault());
    else
      axcache = IntegerAttr();

    // Set the issue_token
    issue_token = BoolAttr::get(ctx, op.getIssueToken());
    // Earlier, all S2MM channels were implicitly assumed to issue a token.
    // This logic is kept for now for backward compatibility.
    if (!isMM2S)
      issue_token = BoolAttr::get(ctx, true);

    if (targetModel.isMemTile(tileCol, tileRow) && (!isMM2S) &&
        (op.getD0ZeroBefore() != 0 || op.getD0ZeroAfter() != 0 ||
         op.getD1ZeroBefore() != 0 || op.getD1ZeroAfter() != 0 ||
         op.getD2ZeroBefore() != 0 || op.getD2ZeroAfter() != 0)) {
      op->emitOpError("MemTile supports zero padding only on MM2S direction");
      return failure();
    }

    // write the buffer descriptor to the array
    auto channel = IntegerAttr::get(i32ty, infoOp.getChannelIndex());
    auto dir = IntegerAttr::get(
        i32ty,
        static_cast<std::underlying_type_t<AIE::DMAChannelDir>>(channelDir));
    NpuWriteBdOp::create(
        rewriter, op->getLoc(), column, row, channel, dir, bd_id, buffer_length,
        buffer_offset, enable_packet, out_of_order_id, packet_id, packet_type,
        d0_size, d0_stride, d1_size, d1_stride, d2_size, d2_stride,
        iteration_current, iteration_size, iteration_stride, next_bd,
        use_next_bd, valid_bd, lock_rel_val, lock_rel_id, lock_acq_enable,
        lock_acq_val, lock_acq_id, d0_zero_before, d1_zero_before,
        d2_zero_before, d0_zero_after, d1_zero_after, d2_zero_after,
        burst_length, axcache);

    // Resolve the buffer's runtime-sequence arg and emit the address patch
    // (plus any offset-state update).
    int arg_idx = -1;
    if (failed(emitBufferAddressPatch(op, adaptor, rewriter, tileCol, tileRow,
                                      infoOp.getChannelIndex(), channelDir,
                                      arg_idx)))
      return failure();

    // A length_state_table_idx adds the runtime length to the BD's
    // Buffer_Length, after the BD write above has set the static length. The
    // verifier requires a length_unit with it.
    auto lengthUnit = op.getLengthUnit();
    if (op.getLengthStateTableIdxAttr() && lengthUnit) {
      if (failed(emitUpdateBdLengthFromParameter(rewriter, op, bufferType,
                                                 *lengthUnit, targetModel,
                                                 tileCol, tileRow, op.getId())))
        return failure();
    }

    // push the patched bd onto the dma task queue. bd_id and repeat_count are
    // SSA operands; materialize them as constants here (the static path).
    NpuPushQueueOp::create(
        rewriter, op->getLoc(), column, row, infoOp.getChannelDirAttr(),
        infoOp.getChannelIndexAttr(), issue_token,
        createConstantI32(rewriter, op->getLoc(),
                          static_cast<uint32_t>(repeat_count.getInt())),
        createConstantI32(rewriter, op->getLoc(),
                          static_cast<uint32_t>(bd_id.getInt())),
        /*repeat_state_table_idx=*/nullptr);

    rewriter.eraseOp(op);
    return success();
  }

  // Resolve the runtime-sequence argument index the descriptor's buffer traces
  // back to, and emit the address-patch (plus an offset-state update, if the op
  // carries one) that binds the runtime buffer pointer into the BD. Shared by
  // the static and dynamic lowering paths, which are otherwise identical here.
  // A walk with any runtime offset, size or stride also gets a host-side check
  // that it stays inside the host buffer. On success `argIdx` receives the
  // resolved index.
  LogicalResult emitBufferAddressPatch(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                                       ConversionPatternRewriter &rewriter,
                                       int tileCol, int tileRow, int channel,
                                       int dir, int &argIdx) const {
    AIE::RuntimeSequenceOp seqOp =
        op->getParentOfType<AIE::RuntimeSequenceOp>();
    if (!seqOp)
      return op->emitOpError("NpuDmaMemcpyNdOps must have RuntimeSequenceOp "
                             "parent at time of lowering.");
    auto traceResult = traceSubviewToBlockArgument(adaptor.getMemref());
    if (!traceResult)
      return op->emitOpError(
          "memref must be a block argument or subview/cast/reinterpret_cast of "
          "a block argument with static offsets, sizes, and strides");
    std::optional<unsigned> hostIdx =
        getHostBufferArgIndex(traceResult->rootArg);
    if (!hostIdx || traceResult->rootArg.getOwner() != &seqOp.getBody().front())
      return failure();
    argIdx = static_cast<int>(*hostIdx);

    const auto &targetModel = AIE::getTargetModel(op);
    uint64_t patchAddr = targetModel.getDmaBdAddress(tileCol, tileRow,
                                                     op.getId(), channel, dir) +
                         targetModel.getDmaBdAddressOffset(tileCol, tileRow);

    // arg_plus is the buffer byte offset. Constant offsets fold to a constant
    // (byte-identical to the static path); a runtime offset operand is built
    // with arith so it flows into the patch instead of being rejected. The
    // subview trace contributes a constant base byte offset.
    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> strides = op.getMixedStrides();
    int64_t elemBytes = op.getElementTypeBitwidth() / 8;
    SmallVector<OpFoldResult> sizes = op.getMixedSizes();
    bool isRuntime =
        llvm::any_of(llvm::concat<OpFoldResult>(offsets, sizes, strides),
                     [](OpFoldResult v) { return !getConstantIntValue(v); });
    if (isRuntime && failed(guardWithinHostBuffer(
                         rewriter, op->getLoc(),
                         cast<BaseMemRefType>(traceResult->rootArg.getType()),
                         traceResult->offsetInBytes, elemBytes, offsets,
                         strides, sizes, strides)))
      return failure();
    Value argPlus = buildArgPlusValue(
        rewriter, op->getLoc(), offsets, strides, elemBytes,
        traceResult->offsetInBytes, targetModel.getAddressGenGranularity() / 8);
    if (!argPlus)
      return failure();
    NpuAddressPatchOp::create(rewriter, op->getLoc(), patchAddr,
                              /*addr_val=*/Value(), argIdx, argPlus);

    // If this DMA op has an offset_state_table_idx, emit an
    // update_from_scratchpad to add the runtime offset to the BD address
    // register (additive; applied after the base patch above).
    if (op.getOffsetStateTableIdxAttr()) {
      auto bufType = cast<BaseMemRefType>(op.getMemref().getType());
      if (failed(emitUpdateBdAddressFromOffsetParameter(rewriter, op, bufType,
                                                        patchAddr)))
        return failure();
    }
    return success();
  }

  // Lower a shim-NOC dma_memcpy_nd carrying runtime (SSA) offsets/sizes/
  // strides. The encoder runs the same arithmetic as the static path, so a
  // runtime value equal to a constant yields the same word. aiebu requires the
  // block-write to precede the address patch and cover the patched word.
  LogicalResult lowerDynamic(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    const auto &targetModel = AIE::getTargetModel(op);
    auto *ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    Location loc = op->getLoc();

    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();
    auto infoOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, op.getMetadata().getRootReference());
    if (!infoOp)
      return op->emitOpError("couldn't find shim_dma_allocation op.");
    AIE::TileOp shimTile = infoOp.getTileOp();
    if (!shimTile)
      return op->emitOpError(
          "shim_dma_allocation op must reference a valid TileOp.");
    auto channelDir = infoOp.getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int tileCol = shimTile.getCol();
    int tileRow = shimTile.getRow();

    // Packet / token setup (identical to the static path).
    auto column = IntegerAttr::get(i32ty, tileCol);
    auto row = IntegerAttr::get(i32ty, tileRow);
    BdTemplateFields fields;
    if (auto packetInfo =
            op.getPacket() ? op.getPacket() : infoOp.getPacket()) {
      fields.enable_packet = 1;
      fields.packet_type = packetInfo->getPktType();
      fields.packet_id = packetInfo->assignedId();
    }
    auto issue_token = BoolAttr::get(ctx, op.getIssueToken());
    if (!isMM2S)
      issue_token = BoolAttr::get(ctx, true);

    // buffer_length is the size-product here, hence no explicit length; the
    // encoder returns the hw repeat_count for the queue push.
    SmallVector<Value> words;
    Value repeatCount;
    if (failed(buildBdWords(rewriter, loc, targetModel, tileCol, tileRow,
                            fields, op.getMixedSizes(), op.getMixedStrides(),
                            op.getElementTypeBitwidth(), op.getBurstLength(),
                            op.getAxcacheOrDefault(),
                            /*lenElems=*/OpFoldResult(), repeatCount, words)))
      return failure();
    Value bdBase =
        getBdRegisterBase(rewriter, loc, targetModel, tileCol, tileRow,
                          rewriter.getI32IntegerAttr(op.getId()));
    NpuBlockWriteValuesOp::create(rewriter, loc, bdBase, words);

    // Address patch for the buffer pointer; emitBufferAddressPatch folds a
    // constant offset or builds a runtime arg_plus with arith as needed.
    int arg_idx = -1;
    if (failed(emitBufferAddressPatch(op, adaptor, rewriter, tileCol, tileRow,
                                      arg_idx)))
      return failure();

    NpuPushQueueOp::create(
        rewriter, loc, column, row, infoOp.getChannelDirAttr(),
        infoOp.getChannelIndexAttr(), issue_token, repeatCount,
        createConstantI32(rewriter, loc, op.getId()),
        /*repeat_state_table_idx=*/nullptr);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert NpuDmaWaitOp into NpuSyncOp by retrieving the necessary
/// information from the ShimDMAAllocationOp referenced through the
/// symbol argument of this op.
struct DmaWaitToSyncPattern : OpConversionPattern<NpuDmaWaitOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  DmaWaitToSyncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuDmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return op->emitError("couldn't find parent of type DeviceOp");

    AIE::ShimDMAAllocationOp shimDmaAllocOp =
        AIE::ShimDMAAllocationOp::getForSymbol(dev, op.getSymbol());
    if (!shimDmaAllocOp) {
      return op->emitError("couldn't find shim_dma_allocation op");
    }

    AIE::TileOp shimTile = shimDmaAllocOp.getTileOp();
    if (!shimTile) {
      return op->emitError(
          "shim_dma_allocation op must reference a valid TileOp");
    }

    // Create with `column_num == 1` and `row_num == 1` to check for a single
    // column and row.
    Location loc = op->getLoc();
    (void)rewriter.replaceOpWithNewOp<NpuSyncOp>(
        op, createConstantI32(rewriter, loc, shimTile.getCol()),
        createConstantI32(rewriter, loc, shimTile.getRow()),
        createConstantI32(
            rewriter, loc,
            static_cast<uint32_t>(shimDmaAllocOp.getChannelDir())),
        createConstantI32(rewriter, loc, shimDmaAllocOp.getChannelIndex()),
        createConstantI32(rewriter, loc, 1),
        createConstantI32(rewriter, loc, 1));

    return success();
  }
};

struct WriteBdToBlockWritePattern : OpConversionPattern<NpuWriteBdOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  WriteBdToBlockWritePattern(MLIRContext *context,
                             BlockwriteData &blockwriteData,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), blockwriteData(blockwriteData) {}

  BlockwriteData &blockwriteData;

  LogicalResult
  matchAndRewrite(NpuWriteBdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();
    int col = op.getColumn();
    int row = op.getRow();

    uint32_t bd_id = op.getBdId();
    auto dir = static_cast<AIE::DMAChannelDir>(op.getDirection());
    uint64_t bd_addr =
        tm.getDmaBdAddress(col, row, bd_id, op.getChannel(), dir);

    std::vector<uint32_t> words;
    if (isa<AIE::AIE4TargetModel>(tm) && tm.isMemTile(col, row)) {
      // AIE4 MemTile DMA - 11 words
      words.assign(11, 0);
      // DMA_S2MM_0_BD0_0
      // Base_Address [23:0]
      words[0] = op.getBufferOffset() & 0xffffff;

      // DMA_S2MM_0_BD0_1
      // Lock_Acq_Value [31:25], Buffer_Length [22:0]
      words[1] = (op.getLockAcqVal() & 0x7f) << 25;
      words[1] |= op.getBufferLength() & 0x7fffff;

      // DMA_S2MM_0_BD0_2
      // Lock_Acq_ID [29:20], Lock_Rel_Value [17:11], Lock_Rel_ID [10:1],
      // Lock_Acq_Enable [0]
      unsigned int acq_id = op.getLockAcqId();
      unsigned int rel_id = op.getLockRelId();
      if (auto base = tm.getLockLocalBaseIndex(col, row, col, row)) {
        acq_id += *base;
        rel_id += *base;
      }
      words[2] = (acq_id & 0x3ff) << 20;
      words[2] |= (op.getLockRelVal() & 0x7f) << 11;
      words[2] |= (rel_id & 0x3ff) << 1;
      words[2] |= op.getLockAcqEnable() & 0x1;

      // DMA_S2MM_0_BD0_3
      // D1_Stepsize [22:0]
      words[3] = op.getD1Stride() & 0x7fffff;

      // DMA_S2MM_0_BD0_4
      // D2_Stepsize [22:0]
      words[4] = op.getD2Stride() & 0x7fffff;

      // DMA_S2MM_0_BD0_5
      // D3_Stepsize [22:0]
      words[5] = 0; // D3_Stepsize not currently used

      // DMA_S2MM_0_BD0_6
      // TLAST_Suppress [31], Iteration_Stepsize [22:0]
      words[6] = 0; // TLAST_Suppress unused in S2MM
      words[6] |= op.getIterationStride() & 0x7fffff;

      // DMA_S2MM_0_BD0_7
      // Iteration_Current [29:22], Enable_Packet [21], Packet_ID [20:16],
      // D0_Pad_After [15:8], D0_Pad_Before [7:0]
      words[7] = (op.getIterationCurrent() & 0xff) << 22;
      words[7] |= (op.getEnablePacket() & 0x1) << 21;
      words[7] |= (op.getPacketId() & 0x1f) << 16;
      words[7] |= (op.getD0ZeroAfter() & 0xff) << 8;
      words[7] |= op.getD0ZeroBefore() & 0xff;

      // DMA_S2MM_0_BD0_8
      // D2_Pad_After [31:24], D2_Pad_Before [23:16], D1_Pad_After [15:8],
      // D1_Pad_Before [7:0]
      words[8] = (op.getD2ZeroAfter() & 0xff) << 24;
      words[8] |= (op.getD2ZeroBefore() & 0xff) << 16;
      words[8] |= (op.getD1ZeroAfter() & 0xff) << 8;
      words[8] |= op.getD1ZeroBefore() & 0xff;

      // DMA_S2MM_0_BD0_9
      // D1_Wrap [30:19], D0_Wrap [16:5], Out_Of_Order_BD_ID [3:0]
      words[9] = (op.getD1Size() & 0xfff) << 19;
      words[9] |= (op.getD0Size() & 0xfff) << 5;
      words[9] |= op.getOutOfOrderId() & 0xf;

      // DMA_S2MM_0_BD0_10
      // Next_BD [30:27], Use_Next_BD [26], D2_Wrap [25:14], Iteration_Wrap
      // [7:0]
      words[10] = (op.getNextBd() & 0xf) << 27;
      words[10] |= (op.getUseNextBd() & 0x1) << 26;
      words[10] |= (op.getD2Size() & 0xfff) << 14;
      words[10] |= op.getIterationSize() & 0xff;
    } else if (isa<AIE::AIE4TargetModel>(tm) && tm.isCoreTile(col, row)) {
      if (op.getD0ZeroBefore() || op.getD1ZeroBefore() ||
          op.getD2ZeroBefore() || op.getD0ZeroAfter() || op.getD1ZeroAfter() ||
          op.getD2ZeroAfter())
        return op->emitOpError("Zero padding is only available on MemTile");
      // AIE4 Tile DMA - 6 words
      words.assign(6, 0);
      // DMA_BD0_0
      // Base_Address [30:16], Buffer_Length [14:0]
      words[0] = ((op.getBufferOffset() / 4) & 0x7fff) << 16;
      words[0] |= op.getBufferLength() & 0x7fff;

      // DMA_BD0_1
      // Enable_Packet [30], Out_Of_Order_BD_ID [27:24], Packet_ID [23:19],
      // Packet_Type [18:16]
      words[1] = (op.getEnablePacket() & 0x1) << 30;
      words[1] |= (op.getOutOfOrderId() & 0xf) << 24;
      words[1] |= (op.getPacketId() & 0x1f) << 19;
      words[1] |= (op.getPacketType() & 0x7) << 16;

      // DMA_BD0_2
      // D1_Stepsize [30:16]
      words[2] = (op.getD1Stride() & 0x7fff) << 16;

      // DMA_BD0_3
      // Iteration_Current [31:26], D0_Wrap [25:17], D2_Stepsize [14:0]
      words[3] = (op.getIterationCurrent() & 0x3f) << 26;
      words[3] |= (op.getD0Size() & 0x1ff) << 17;
      words[3] |= op.getD2Stride() & 0x7fff;

      // DMA_BD0_4
      // D1_Wrap [31:23], Iteration_Wrap [22:17], Iteration_Stepsize [14:0]
      words[4] = (op.getD1Size() & 0x1ff) << 23;
      words[4] |= (op.getIterationSize() & 0x3f) << 17;
      words[4] |= op.getIterationStride() & 0x7fff;

      // DMA_BD0_5
      // TLAST_Suppress [31], Next_BD [30:27], Use_Next_BD [26],
      // Lock_Rel_Value [24:18], Lock_Rel_ID [16:13], Lock_Acq_Enable [12],
      // Lock_Acq_Value [11:5], Lock_Acq_ID [3:0]
      words[5] = 0; // TLAST_Suppress
      words[5] |= (op.getNextBd() & 0xf) << 27;
      words[5] |= (op.getUseNextBd() & 0x1) << 26;
      words[5] |= (op.getLockRelVal() & 0x7f) << 18;
      words[5] |= (op.getLockRelId() & 0xf) << 13;
      words[5] |= (op.getLockAcqEnable() & 0x1) << 12;
      words[5] |= (op.getLockAcqVal() & 0x7f) << 5;
      words[5] |= op.getLockAcqId() & 0xf;
    } else if (isa<AIE::AIE4TargetModel>(tm)) {
      if (!tm.isShimNOCTile(col, row))
        return op->emitOpError("has no buffer descriptor layout on this tile");
      if (op.getD0ZeroBefore() || op.getD1ZeroBefore() ||
          op.getD2ZeroBefore() || op.getD0ZeroAfter() || op.getD1ZeroAfter() ||
          op.getD2ZeroAfter())
        return op->emitOpError("Zero padding is only available on MemTile");
      if (op.getD0Size() > 0xfff)
        return op->emitError("D0 Size exceeds 12-bit AIE4 shim BD field");
      if (op.getD1Size() > 0xfff)
        return op->emitError("D1 Size exceeds 12-bit AIE4 shim BD field");
      if (op.getD2Size() > 0xfff)
        return op->emitError("D2 Size exceeds 12-bit AIE4 shim BD field");
      words.assign(9, 0);

      // DMA_BDX_0
      // TODO: AxUSER, KeyIdx, Base_Address_High
      words[0] = 0;

      // DMA_BDX_1
      words[1] = op.getBufferOffset();

      // DMA_BDX_2
      words[2] = op.getBufferLength();

      // DMA_BDX_3
      words[3] = (op.getLockAcqId() & 0x1f) << 27;
      words[3] |= (op.getLockAcqEnable() & 0x1) << 26;
      words[3] |= (getShimBurstLengthEncoding(tm, op.getBurstLength()) & 0x3)
                  << 24;
      words[3] |= (op.getD1Size() & 0xfff) << 12;
      words[3] |= op.getD2Size() & 0xfff;

      // DMA_BDX_4
      words[4] = (op.getD0Size() & 0xfff) << 19;
      words[4] |= (op.getLockRelVal() & 0x7f) << 12;
      words[4] |= (op.getLockRelId() & 0x1f) << 7;
      words[4] |= op.getLockAcqVal() & 0x7f;

      // DMA_BDX_5
      // TODO: AxQoS
      words[5] = (op.getEnablePacket() & 0x1) << 31;
      words[5] |= (op.getNextBd() & 0xf) << 27;
      words[5] |= (op.getUseNextBd() & 0x1) << 26;
      // words[5] |= (AxQoS & 0x2) << 22;
      words[5] |= op.getD1Stride() & 0x3fffff;

      // DMA_BDX_6
      // TODO: Enable_Compression, TLAST_Suppress, Data_Reuse
      words[6] = (op.getIterationCurrent() & 0x3f) << 24;
      words[6] |= op.getD2Stride() & 0x3fffff;

      // DMA_BDX_7
      // TODO: AxCache
      words[7] = (op.getIterationSize() & 0x3f) << 22;
      words[7] |= op.getIterationStride() & 0x3fffff;

      // DMA_BDX_8
      // TODO: D3_Stepsize, IO_Coherence
      words[8] = (op.getOutOfOrderId() & 0xf) << 5;
      words[8] |= op.getPacketId() & 0x1f;
    } else {
      const AIE::DmaBdLayout *layout = tm.getDmaBdLayout(col, row);
      if (!layout)
        return op->emitOpError("has no buffer descriptor layout on this tile");
      if (!tm.isMemTile(col, row) &&
          (op.getD0ZeroBefore() || op.getD1ZeroBefore() ||
           op.getD2ZeroBefore() || op.getD0ZeroAfter() || op.getD1ZeroAfter() ||
           op.getD2ZeroAfter()))
        return op->emitOpError("Zero padding is only available on MemTile");

      words.assign(layout->numWords, 0);
      auto set = [&](const AIE::DmaBdField &field, uint64_t value) {
        if (field.exists())
          words[field.word] |= field.place(value);
      };
      uint64_t bufferOffset = op.getBufferOffset();
      if (tm.isCoreTile(col, row))
        bufferOffset /= 4;
      set(layout->bufferLength, op.getBufferLength());
      set(layout->bufferOffset, bufferOffset);
      set(layout->enablePacket, op.getEnablePacket());
      set(layout->packetType, op.getPacketType());
      set(layout->packetId, op.getPacketId());
      set(layout->outOfOrderId, op.getOutOfOrderId());
      set(layout->d0Size, op.getD0Size());
      set(layout->d0Stride, op.getD0Stride());
      set(layout->d1Size, op.getD1Size());
      set(layout->d1Stride, op.getD1Stride());
      set(layout->d2Stride, op.getD2Stride());
      set(layout->iterationCurrent, op.getIterationCurrent());
      set(layout->iterationSize, op.getIterationSize());
      set(layout->iterationStride, op.getIterationStride());
      set(layout->d0ZeroBefore, op.getD0ZeroBefore());
      set(layout->d1ZeroBefore, op.getD1ZeroBefore());
      set(layout->d2ZeroBefore, op.getD2ZeroBefore());
      set(layout->d0ZeroAfter, op.getD0ZeroAfter());
      set(layout->d1ZeroAfter, op.getD1ZeroAfter());
      set(layout->d2ZeroAfter, op.getD2ZeroAfter());
      if (layout->burstLength.exists())
        set(layout->burstLength,
            getShimBurstLengthEncoding(tm, op.getBurstLength()));
      set(layout->axcache, op.getAxcacheOrDefault());
      set(layout->nextBd, op.getNextBd());
      set(layout->useNextBd, op.getUseNextBd());
      set(layout->validBd, op.getValidBd());
      set(layout->lockRelValue, op.getLockRelVal());
      set(layout->lockRelId, op.getLockRelId());
      set(layout->lockAcqEnable, op.getLockAcqEnable());
      set(layout->lockAcqValue, op.getLockAcqVal());
      set(layout->lockAcqId, op.getLockAcqId());
    }

    memref::GlobalOp global = nullptr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op->getParentOfType<AIE::RuntimeSequenceOp>());
      global = blockwriteData.getOrCreate(rewriter, op.getLoc(), words);
    }
    auto memref = memref::GetGlobalOp::create(
        rewriter, op.getLoc(), global.getType(), global.getName());

    (void)rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(
        op, rewriter.getUI32IntegerAttr(bd_addr), memref.getResult(), nullptr,
        nullptr, nullptr);
    return success();
  }
};

// Count all task starts at their common representation, after memcpy lowering
// but before pushes become register writes. This includes starts lowered by
// dma-tasks-to-npu and compiler-generated channel rearm pushes.
static void checkQueueDepth(AIE::DeviceOp device, bool enforceQueueDepth) {
  const AIE::AIETargetModel &tm = device.getTargetModel();
  auto effectOf = [&](Operation *op) -> QueueEffect {
    if (auto push = dyn_cast<NpuPushQueueOp>(op))
      return QueueEffect::push({static_cast<int>(push.getColumn()),
                                static_cast<int>(push.getRow()),
                                static_cast<int>(push.getDirection()),
                                static_cast<int>(push.getChannel())},
                               push.getIssueToken());
    if (auto sync = dyn_cast<NpuSyncOp>(op))
      if (std::optional<DmaQueueModel::ChannelKey> key = syncChannelKey(sync))
        return QueueEffect::await(*key);
    return {};
  };

  device.walk([&](AIE::RuntimeSequenceOp seq) {
    DmaQueueModel queue;
    // No dma_free_task survives to this lowering stage; nothing to check here.
    guardSequenceQueueDepth(seq.getBody(), queue, tm, enforceQueueDepth,
                            /*warnUnsafeBdReuse=*/false, effectOf);
    seq->removeAttr(queueDiagnosedAttr);
  });
}

struct AIEDmaToNpuPass : xilinx::AIEX::impl::AIEDmaToNpuBase<AIEDmaToNpuPass> {
  using Base = xilinx::AIEX::impl::AIEDmaToNpuBase<AIEDmaToNpuPass>;
  AIEDmaToNpuPass() = default;
  AIEDmaToNpuPass(const AIEDmaToNpuOptions &options) : Base(options) {}

  void runOnOperation() override {

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalOp<cf::AssertOp>();
    target.addLegalOp<AIE::BufferOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();
    target.addLegalOp<AIE::TileOp>();

    target.addIllegalOp<NpuDmaMemcpyNdOp>();
    target.addIllegalOp<NpuDmaWaitOp>();
    target.addIllegalOp<NpuWriteRTPOp>();
    target.addIllegalOp<NpuWriteBdOp>();
    target.addDynamicallyLegalOp<NpuWrite32Op>(
        [&](NpuWrite32Op op) { return !op.getBuffer(); });
    target.addDynamicallyLegalOp<NpuBlockWriteOp>(
        [&](NpuBlockWriteOp op) { return !op.getBuffer(); });
    target.addDynamicallyLegalOp<NpuMaskWrite32Op>(
        [&](NpuMaskWrite32Op op) { return !op.getBuffer(); });
    target.addDynamicallyLegalOp<NpuMaskPollOp>(
        [&](NpuMaskPollOp op) { return !op.getBuffer(); });

    BlockwriteData blockwriteData(device, "blockwrite_data_");
    AIE::NamedOpTable names(device);

    RewritePatternSet patterns(&getContext());
    patterns.insert<BlockWriteSymToAddr>(&getContext(), names);
    patterns.insert<DmaToNpuPattern>(&getContext());
    patterns.insert<DmaWaitToSyncPattern>(&getContext());
    patterns.insert<MaskWrite32SymToAddr>(&getContext(), names);
    patterns.insert<MaskPollSymToAddr>(&getContext(), names);
    patterns.insert<RtpToWrite32Pattern>(&getContext(), names);
    patterns.insert<Write32SymToAddr>(&getContext(), names);
    patterns.insert<WriteBdToBlockWritePattern>(&getContext(), blockwriteData);

    // The driver converts the illegal ops alone; started from the device, it
    // would visit every op of the runtime sequences.
    SmallVector<Operation *> illegal;
    device.walk([&](Operation *op) {
      if (target.isIllegal(op))
        illegal.push_back(op);
    });
    ConversionConfig config;
    config.foldingMode = DialectConversionFoldingMode::Never;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(illegal, target, std::move(patterns),
                                      config))) {
      signalPassFailure();
      return;
    }

    checkQueueDepth(device, enforceQueueDepth);

    target.addIllegalOp<NpuPushQueueOp>();
    RewritePatternSet pushPatterns(&getContext());
    pushPatterns.insert<PushQueuetoWrite32Pattern>(&getContext());
    SmallVector<Operation *> pushes;
    device.walk([&](NpuPushQueueOp op) { pushes.push_back(op); });
    if (failed(applyPartialConversion(pushes, target, std::move(pushPatterns),
                                      config)))
      signalPassFailure();

    eraseDeadArith(device);
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEDmaToNpuPass() {
  return std::make_unique<AIEDmaToNpuPass>();
}

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEDmaToNpuPass(const AIEDmaToNpuOptions &options) {
  return std::make_unique<AIEDmaToNpuPass>(options);
}
