//===- AIEInsertTraceFlows.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Pass to insert packet flows and runtime sequence trace setup
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include <map>
#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct TraceInfo {
  TraceOp traceOp;
  TileOp tile;
  int packetId;
  TracePacketType packetType;
  WireBundle tracePort; // Trace:0 (core) or Trace:1 (mem)
  int traceChannel;     // Port number (0 for core, 1 for mem)
};

struct ShimInfo {
  TileOp shimTile;
  int channel;      // S2MM channel
  int bdId;         // Buffer descriptor ID
  int argIdx;       // Runtime sequence argument index
  std::vector<TraceInfo> traceSources; // All traces routed to this shim
};

} // namespace

struct AIEInsertTraceFlowsPass
    : AIEInsertTraceFlowsBase<AIEInsertTraceFlowsPass> {

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device);
    const auto &targetModel = device.getTargetModel();

    // Phase 1: Collect all trace operations
    SmallVector<TraceOp> traces;
    device.walk([&](TraceOp trace) { traces.push_back(trace); });

    if (traces.empty())
      return;

    // Phase 2: Analyze traces and allocate resources
    std::vector<TraceInfo> traceInfos;
    std::map<int, int> usedPacketIds; // col -> next available packet ID
    int nextPacketId = packetIdStart;

    for (auto trace : traces) {
      auto tile = cast<TileOp>(trace.getTile().getDefiningOp());
      int col = tile.getCol();

      // Find packet ID and type from trace body
      std::optional<int> packetId;
      TracePacketType packetType = TracePacketType::Core; // default
      for (auto &op : trace.getBody().getOps()) {
        if (auto packetOp = dyn_cast<TracePacketOp>(op)) {
          packetId = packetOp.getId();
          packetType = packetOp.getType();
          break;
        }
      }

      // Allocate packet ID if not specified
      if (!packetId) {
        if (usedPacketIds.find(col) == usedPacketIds.end()) {
          usedPacketIds[col] = nextPacketId;
        }
        packetId = usedPacketIds[col]++;
      }

      // Determine trace port based on packet type
      WireBundle tracePort = WireBundle::Trace;
      int traceChannel = 0;
      if (packetType == TracePacketType::Mem) {
        traceChannel = 1; // Mem trace uses port 1
      } else if (packetType == TracePacketType::Core) {
        traceChannel = 0; // Core trace uses port 0
      } else if (packetType == TracePacketType::MemTile) {
        traceChannel = 0; // MemTile trace uses port 0
      } else if (packetType == TracePacketType::ShimTile) {
        traceChannel = 0; // ShimTile trace uses port 0
      }

      TraceInfo info;
      info.traceOp = trace;
      info.tile = tile;
      info.packetId = *packetId;
      info.packetType = packetType;
      info.tracePort = tracePort;
      info.traceChannel = traceChannel;
      traceInfos.push_back(info);
    }

    // Phase 2b: Select shim tiles (minimize usage)
    std::map<int, ShimInfo> shimInfos; // col -> ShimInfo
    
    if (minimizeShims && preferSameColumn) {
      // Strategy: Group all traces by column, use one shim per column
      std::map<int, std::vector<TraceInfo>> tracesByCol;
      for (auto &info : traceInfos) {
        int col = info.tile.getCol();
        tracesByCol[col].push_back(info);
      }

      // For each column with traces, allocate a shim
      for (auto &[col, colTraces] : tracesByCol) {
        // Find shim tile for this column
        TileOp shimTile = nullptr;
        for (auto tile : device.getOps<TileOp>()) {
          if (tile.getCol() == col && tile.getRow() == 0) {
            shimTile = tile;
            break;
          }
        }

        if (!shimTile) {
          // Create shim tile if it doesn't exist
          builder.setInsertionPointToStart(&device.getRegion().front());
          shimTile = builder.create<TileOp>(device.getLoc(), col, 0);
        }

        ShimInfo shimInfo;
        shimInfo.shimTile = shimTile;
        shimInfo.channel = shimChannel;
        shimInfo.bdId = defaultBdId;
        shimInfo.argIdx = traceArgIdx;
        shimInfo.traceSources = colTraces;
        shimInfos[col] = shimInfo;
      }
    } else {
      // Fallback: Use one shim for all traces (column 0)
      int targetCol = 0;
      TileOp shimTile = nullptr;
      for (auto tile : device.getOps<TileOp>()) {
        if (tile.getCol() == targetCol && tile.getRow() == 0) {
          shimTile = tile;
          break;
        }
      }

      if (!shimTile) {
        builder.setInsertionPointToStart(&device.getRegion().front());
        shimTile = builder.create<TileOp>(device.getLoc(), targetCol, 0);
      }

      ShimInfo shimInfo;
      shimInfo.shimTile = shimTile;
      shimInfo.channel = shimChannel;
      shimInfo.bdId = defaultBdId;
      shimInfo.argIdx = traceArgIdx;
      shimInfo.traceSources = traceInfos;
      shimInfos[targetCol] = shimInfo;
    }

    // Phase 3: Insert packet flows
    // Insert before the device terminator
    Block &deviceBlock = device.getRegion().front();
    builder.setInsertionPoint(deviceBlock.getTerminator());

    for (auto &info : traceInfos) {
      // Find target shim for this trace
      int col = info.tile.getCol();
      ShimInfo &shimInfo = shimInfos[col];

      // Create packet flow
      auto packetFlowOp = builder.create<PacketFlowOp>(
          device.getLoc(), builder.getI8IntegerAttr(info.packetId), nullptr, nullptr);
      
      Block *flowBody = new Block();
      packetFlowOp.getPorts().push_back(flowBody);
      OpBuilder flowBuilder = OpBuilder::atBlockEnd(flowBody);

      // Add source
      flowBuilder.create<PacketSourceOp>(
          device.getLoc(), Value(info.tile.getResult()), info.tracePort, info.traceChannel);

      // Add destination
      flowBuilder.create<PacketDestOp>(
          device.getLoc(), Value(shimInfo.shimTile.getResult()), WireBundle::DMA, shimInfo.channel);

      // Add terminator
      flowBuilder.create<EndOp>(device.getLoc());

      // Add keep_pkt_header attribute
      packetFlowOp->setAttr("keep_pkt_header", builder.getBoolAttr(true));
    }

    // Phase 4: Insert runtime sequence operations
    // Find runtime sequence
    RuntimeSequenceOp runtimeSeq = nullptr;
    device.walk([&](RuntimeSequenceOp seq) {
      if (!runtimeSeq)
        runtimeSeq = seq;
      return WalkResult::advance();
    });

    if (!runtimeSeq) {
      // No runtime sequence found, nothing to insert
      return;
    }

    // Insert trace configuration at the beginning of runtime sequence
    Block &seqBlock = runtimeSeq.getBody().front();
    builder.setInsertionPointToStart(&seqBlock);

    // 4a. Insert trace.start_config for each trace
    for (auto &info : traceInfos) {
      builder.create<TraceStartConfigOp>(
          runtimeSeq.getLoc(),
          FlatSymbolRefAttr::get(builder.getContext(),
                                 info.traceOp.getSymNameAttr()));
    }

    // 4b. Insert per-tile timer controls
    std::set<std::pair<int, int>> processedTiles; // (col, row)
    for (auto &info : traceInfos) {
      int col = info.tile.getCol();
      int row = info.tile.getRow();
      
      if (processedTiles.find({col, row}) != processedTiles.end())
        continue;
      processedTiles.insert({col, row});

      // Compute timer control address
      uint32_t timerCtrlAddr = computeTimerCtrlAddress(
          info.tile, targetModel, info.packetType == TracePacketType::Mem);

      // Timer control value: BROADCAST_15 event (122 << 8 = 31232)
      uint32_t timerCtrlValue = 31232; // Event 122 (BROADCAST_15) << 8

      builder.create<xilinx::AIEX::NpuWrite32Op>(
          runtimeSeq.getLoc(), timerCtrlAddr, timerCtrlValue,
          nullptr, builder.getI32IntegerAttr(col), builder.getI32IntegerAttr(row));
    }

    // 4c-4f. Insert per-shim configurations
    for (auto &[col, shimInfo] : shimInfos) {
      int shimCol = shimInfo.shimTile.getCol();

      // 4c. Write buffer descriptor
      builder.create<xilinx::AIEX::NpuWriteBdOp>(
          runtimeSeq.getLoc(),
          shimCol,                      // column
          shimInfo.bdId,               // bd_id
          traceBufferSize,             // buffer_length
          0,                           // buffer_offset
          1,                           // enable_packet
          0,                           // out_of_order_id
          0,                           // packet_id (not used for reception)
          0,                           // packet_type (not used for reception)
          0, 0, 0, 0, 0, 0,           // d0_size, d0_stride, d1_size, d1_stride, d2_size, d2_stride
          0, 0, 0,                    // iteration_current, iteration_size, iteration_stride
          0,                           // next_bd
          0,                           // row
          0,                           // use_next_bd
          1,                           // valid_bd
          0, 0, 0, 0, 0,              // lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id
          0, 0, 0, 0, 0, 0,           // d0_zero_before, d1_zero_before, d2_zero_before, d0_zero_after, d1_zero_after, d2_zero_after
          traceBurstLength             // burst_length
      );

      // 4d. Address patch
      uint32_t bdAddress = computeBDAddress(shimCol, shimInfo.bdId, targetModel);
      builder.create<xilinx::AIEX::NpuAddressPatchOp>(
          runtimeSeq.getLoc(), bdAddress, shimInfo.argIdx, 0);

      // 4e. DMA channel configuration
      uint32_t ctrlAddr = computeCtrlAddress(DMAChannelDir::S2MM, shimInfo.channel);
      builder.create<xilinx::AIEX::NpuMaskWrite32Op>(
          runtimeSeq.getLoc(), ctrlAddr, 3840, 7936,  // value, mask
          nullptr, builder.getI32IntegerAttr(shimCol), builder.getI32IntegerAttr(0));

      // Push BD to task queue
      uint32_t taskQueueAddr = computeTaskQueueAddress(DMAChannelDir::S2MM, shimInfo.channel);
      uint32_t bdIdWithToken = (1U << 31) | shimInfo.bdId; // enable_token = 1
      builder.create<xilinx::AIEX::NpuWrite32Op>(
          runtimeSeq.getLoc(), taskQueueAddr, bdIdWithToken,
          nullptr, builder.getI32IntegerAttr(shimCol), builder.getI32IntegerAttr(0));

      // 4f. Shim timer and broadcast control
      // Shim timer control (USER_EVENT_1 = 127 << 8 = 32512)
      builder.create<xilinx::AIEX::NpuWrite32Op>(
          runtimeSeq.getLoc(), 212992, 32512,
          nullptr, builder.getI32IntegerAttr(shimCol), builder.getI32IntegerAttr(0));

      // Generate USER_EVENT_1
      builder.create<xilinx::AIEX::NpuWrite32Op>(
          runtimeSeq.getLoc(), 213068, 127,
          nullptr, builder.getI32IntegerAttr(shimCol), builder.getI32IntegerAttr(0));

      // Trigger broadcast
      builder.create<xilinx::AIEX::NpuWrite32Op>(
          runtimeSeq.getLoc(), 213000, 127,
          nullptr, builder.getI32IntegerAttr(shimCol), builder.getI32IntegerAttr(0));
    }
  }

private:
  // Compute buffer descriptor base address
  uint32_t computeBDAddress(int col, int bdId, const AIETargetModel &tm) {
    const uint32_t BD_BASE = 0x1D004;
    const uint32_t BD_STRIDE = 0x20;
    return (col << tm.getColumnShift()) | (BD_BASE + bdId * BD_STRIDE);
  }

  // Compute DMA task queue address
  uint32_t computeTaskQueueAddress(DMAChannelDir dir, int channel) {
    if (dir == DMAChannelDir::S2MM) {
      return (channel == 0) ? 0x1D204 : 0x1D20C;
    } else { // MM2S
      return (channel == 0) ? 0x1D214 : 0x1D21C;
    }
  }

  // Compute DMA control register address
  uint32_t computeCtrlAddress(DMAChannelDir dir, int channel) {
    if (dir == DMAChannelDir::S2MM) {
      return (channel == 0) ? 0x1D200 : 0x1D208;
    } else { // MM2S
      return (channel == 0) ? 0x1D210 : 0x1D218;
    }
  }

  // Compute timer control address based on tile type
  uint32_t computeTimerCtrlAddress(TileOp tile, const AIETargetModel &tm,
                                   bool isMemTrace) {
    int col = tile.getCol();
    int row = tile.getRow();
    
    if (tm.isShimNOCTile(col, row) || tm.isShimPLTile(col, row)) {
      return 0x34000; // 212992
    } else if (tm.isMemTile(col, row)) {
      return 0x94000;
    } else if (tm.isCoreTile(col, row)) {
      return isMemTrace ? 0x14000 : 0x34000;
    }
    return 0x34000; // default
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEInsertTraceFlowsPass() {
  return std::make_unique<AIEInsertTraceFlowsPass>();
}
