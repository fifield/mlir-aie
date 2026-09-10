//===- AIEPlaceTiles.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallSet.h"

#include <cmath>
#include <map>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEPLACETILES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-place-tiles"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct ConvertLogicalTileToTile : OpConversionPattern<LogicalTileOp> {
  ConvertLogicalTileToTile(MLIRContext *context, DeviceOp &d, Placer &p,
                           PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), placer(p) {}

  LogicalResult
  matchAndRewrite(LogicalTileOp logicalTile, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto placement = placer.getPlacement(logicalTile);
    if (!placement)
      return logicalTile.emitError("no placement found for logical tile");

    // Handle merging multiple logical tiles to same physical tile
    TileOp tileOp = TileOp::getOrCreate(rewriter, device, placement->col,
                                        placement->row, logicalTile.getLoc());
    rewriter.modifyOpInPlace(tileOp, [&] {
      tileOp->setLoc(
          rewriter.getFusedLoc({tileOp.getLoc(), logicalTile.getLoc()}));
    });

    if (auto controllerId = logicalTile->getAttr("controller_id"))
      tileOp->setAttr("controller_id", controllerId);

    rewriter.replaceOp(logicalTile, tileOp.getResult());
    return success();
  }

private:
  DeviceOp device;
  Placer &placer;
};

// Emit the AIE4 control wiring: every shim tile used by the design must carry a
// controller_id PacketInfoAttr and participate in a Control32 -> Controller32
// packet flow back to the first shim's Controller32 channel 0. Without this the
// AIE4 runtime sequence has no control path back to the host and
// pyxrt.run.wait() never returns. AIE2 / AIE2P targets do not need it.
//
// This runs after tile placement so that logical tiles have been resolved to
// real, column-assigned (and merged) aie.tile ops. Doing the wiring earlier
// (e.g. in the IRON Python front-end on aie.logical_tile ops) does not work:
// unplaced logical tiles have no column to deduplicate on, and any attribute
// set on a logical tile is dropped when it is rewritten into a real tile.
//
// The emitted pattern:
//   shim.attributes["controller_id"] = #aie.packet_info<pkt_type=0, pkt_id=i>
//   aie.packet_flow(i) {
//     aie.packet_source<%shim, Control32 : ch>     // ch in {0, 1}
//     aie.packet_dest<%first_shim, Controller32 : 0>
//   }
//
// AIE4 shims have two BD pools (lower: BDs 0..15 on channels 0/1; upper: BDs
// 16..31). The lower pool returns DMA tokens via Control32:0 and the upper pool
// via Control32:1; both flows from a given shim share that shim's controller_id
// pkt_id, since the destination matches against the packet_id carried by the
// source tile, not by the source channel.
static void emitAIE4ControlWiring(DeviceOp device) {
  const AIETargetModel &targetModel = device.getTargetModel();
  if (!isa<AIE4TargetModel>(targetModel))
    return;

  // If the design already routes task-complete tokens back to a uC, it was
  // wired manually (e.g. the *_placed.py examples) -- leave it be.
  //
  // Test for the TCT route itself rather than for the presence of a
  // `controller_id` attribute. `controller_id` is a poor proxy: it is also set
  // on core and mem tiles, and the check runs before the shim-tile filter, so
  // keying off it skips the wiring for any design that carries a controller_id
  // but has no TCT route at all (see
  // test/place-tiles/test_aie4_control_wiring.mlir's
  // @controller_id_but_no_tct_route case).
  //
  // Note this cannot be reached from the standard aiecc pipeline by a
  // control-packet design: aie-place-tiles runs in getPlacementPipeline,
  // strictly before AIEAssignTileCtrlIDs / AIEGenerateColumnControlOverlay
  // stamp any controller_id. It only matters for input MLIR that already
  // carries one.
  for (auto pktFlow : device.getOps<PacketFlowOp>())
    for (auto dest : pktFlow.getOps<PacketDestOp>())
      if (dest.getBundle() == WireBundle::Controller32)
        return;

  // Collect the unique shim NOC/PL tiles used by the design, one per column and
  // in deterministic column order.
  std::map<int, TileOp> shimByCol;
  for (auto tile : device.getOps<TileOp>()) {
    if (!tile.isShimNOCorPLTile())
      continue;
    int col = tile.colIndex();
    if (!shimByCol.count(col))
      shimByCol[col] = tile;
  }
  if (shimByCol.empty())
    return;

  MLIRContext *ctx = device.getContext();

  // Assign controller_id [pkt_type=0, pkt_id=idx] per shim column, preserving
  // any id a previous pass already chose (the column control overlay routes
  // control packets by those ids, so they must not be renumbered here).
  llvm::SmallSet<int, 8> usedIds;
  for (auto &[col, tile] : shimByCol)
    if (auto info = tile->getAttrOfType<PacketInfoAttr>("controller_id"))
      usedIds.insert(static_cast<int>(info.getPktId()));
  int nextId = 0;
  auto allocPktId = [&]() {
    while (usedIds.count(nextId))
      ++nextId;
    usedIds.insert(nextId);
    return nextId;
  };

  SmallVector<TileOp> shims;
  SmallVector<int> shimPktIds;
  for (auto &[col, tile] : shimByCol) {
    int pktId;
    if (auto info = tile->getAttrOfType<PacketInfoAttr>("controller_id"))
      pktId = static_cast<int>(info.getPktId());
    else {
      pktId = allocPktId();
      tile->setAttr("controller_id",
                    PacketInfoAttr::get(ctx, /*pkt_type=*/0, /*pkt_id=*/pktId));
    }
    shims.push_back(tile);
    shimPktIds.push_back(pktId);
  }

  TileOp controller = shims.front();
  OpBuilder builder(ctx);
  for (int i = 0, e = shims.size(); i < e; ++i) {
    TileOp shim = shims[i];
    for (int ctrlCh : {0, 1}) {
      builder.setInsertionPoint(device.getBody()->getTerminator());
      // The flow ID must be the shim's controller_id pkt_id: the destination
      // matches on the packet id carried by the source tile.
      auto pktFlow =
          PacketFlowOp::create(builder, shim.getLoc(), /*ID=*/shimPktIds[i],
                               /*keep_pkt_header=*/nullptr,
                               /*ctrl_pkt_flow=*/nullptr);
      Block *body = builder.createBlock(&pktFlow.getPorts());
      builder.setInsertionPointToStart(body);
      PacketSourceOp::create(builder, shim.getLoc(), shim,
                             WireBundle::Control32, ctrlCh);
      PacketDestOp::create(builder, shim.getLoc(), controller,
                           WireBundle::Controller32, /*channel=*/0);
      EndOp::create(builder, shim.getLoc());
    }
  }
}

struct AIEPlaceTilesPass
    : xilinx::AIE::impl::AIEPlaceTilesBase<AIEPlaceTilesPass> {

  AIEPlaceTilesPass() = default;

  AIEPlaceTilesPass(const AIEPlaceTilesOptions &options)
      : AIEPlaceTilesBase(options) {}

  void runOnOperation() override {
    DeviceOp device = getOperation();

    // Create placer
    std::shared_ptr<Placer> placer;
    std::shared_ptr<SAPlacer> saPlacer;
    switch (clPlacerType) {
    case PlacerType::SequentialPlacer: {
      std::optional<int> coresPerCol = std::nullopt;
      if (clCoresPerCol >= 0)
        coresPerCol = clCoresPerCol;
      placer = std::make_shared<SequentialPlacer>(
          coresPerCol, clMergeLogicalTiles, clSpreadUnanchoredTiles);
      break;
    }
    case PlacerType::SAPlacer: {
      double effort = clSAEffort.getValue();
      if (!std::isfinite(effort) || effort <= 0) {
        device.emitError("sa-effort must be positive and finite, got ")
            << effort;
        return signalPassFailure();
      }
      SAConfig config;
      config.effort = effort;
      placer = saPlacer = std::make_shared<SAPlacer>(clSASeed, config);
      break;
    }
    }

    placer->initialize(device.getTargetModel());
    if (failed(placer->place(device)))
      return signalPassFailure();
    if (saPlacer)
      saFinalCost += saPlacer->getFinalCost();

    ConversionTarget target(getContext());
    target.addLegalOp<TileOp>();
    target.addIllegalOp<LogicalTileOp>();
    target.addLegalDialect<AIEDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertLogicalTileToTile>(device.getContext(), device,
                                           *placer);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      return signalPassFailure();

    // Generate objectfifo.allocate ops right after their respective objectfifo
    auto &allocs = placer->getAllocates();
    if (!allocs.empty()) {
      // Deduplicate: only emit one allocate per fifo
      llvm::DenseSet<Operation *> emitted;
      for (auto &[fifoOp, delegateTileID] : allocs) {
        if (emitted.count(fifoOp))
          continue;
        emitted.insert(fifoOp);

        auto ofOp = dyn_cast<ObjectFifoCreateOp>(fifoOp);
        if (!ofOp)
          continue;

        // Insert right after the objectfifo op
        OpBuilder builder(ofOp->getContext());
        builder.setInsertionPointAfter(ofOp);

        TileOp delegateTile =
            TileOp::getOrCreate(builder, device, delegateTileID.col,
                                delegateTileID.row, ofOp.getLoc());
        ObjectFifoAllocateOp::create(
            builder, ofOp.getLoc(),
            SymbolRefAttr::get(builder.getContext(), ofOp.getSymName()),
            delegateTile.getResult());
      }
    }

    // Tiles are now placed (and merged); emit AIE4 control wiring on the real
    // aie.tile ops. No-op for non-AIE4 targets.
    emitAIE4ControlWiring(device);
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEPlaceTilesPass(const AIEPlaceTilesOptions &options) {
  return std::make_unique<AIEPlaceTilesPass>(options);
}
