//===- AIEReserveRuntimeBDIDs.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Records compile-time-pinned runtime-sequence BD ids on each aie.tile so the
// static BD assignment avoids them: aiex.reserved_bd_ids for ids held in every
// pool of the tile, aiex.reserved_channel_bd_ids for one channel's pool only.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIERESERVERUNTIMEBDIDS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

struct AIEReserveRuntimeBDIDsPass
    : xilinx::AIEX::impl::AIEReserveRuntimeBDIDsBase<
          AIEReserveRuntimeBDIDsPass> {

  void runOnOperation() override {
    DeviceOp device = getOperation();

    const AIETargetModel &targetModel = device.getTargetModel();
    llvm::MapVector<TileOp, SmallVector<ReservedBdId>> reserved;
    forEachRuntimePinnedBdId(
        device, [&](Operation *, Operation *tile, std::optional<int> channel,
                    std::optional<DMAChannelDir> dir, int32_t id) {
          auto placed = dyn_cast<TileOp>(tile);
          if (!placed)
            return;
          // Reserve tile-wide unless that takes the id from other pools.
          ReservedBdId r{std::nullopt, id};
          BdIdGenerator gen(placed.getCol(), placed.getRow(), targetModel);
          if (channel && dir && id >= 0 &&
              !gen.holdsTileWide(*channel, *dir, static_cast<uint32_t>(id)))
            r.channel = std::make_pair(*channel, *dir);
          reserved[placed].push_back(r);
        });

    for (auto &[tile, ids] : reserved)
      setReservedBdIds(tile, ids);
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEReserveRuntimeBDIDsPass() {
  return std::make_unique<AIEReserveRuntimeBDIDsPass>();
}
