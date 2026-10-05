// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include <set>
#include <stdexcept>

using namespace xilinx;

void testDmaStreamPorts() {
  using namespace AIE;
  for (auto device : {AIEDevice::xcvc1902, AIEDevice::npu1, AIEDevice::npu2}) {
    const auto &model = getTargetModel(device);
    TileID core{0, static_cast<int>(model.getNumMemTileRows()) + 1};
    for (auto direction : {DMAChannelDir::MM2S, DMAChannelDir::S2MM}) {
      auto port = model.getDmaStreamPort(core, direction, 0);
      if (!port || *port != Port{WireBundle::DMA, 0})
        throw std::runtime_error("Incorrect DMA stream port");
      if (model.getDmaStreamPort(core, direction, -1) ||
          model.getDmaStreamPort(core, direction, 100) ||
          model.getDmaStreamPort({-1, 0}, direction, 0))
        throw std::runtime_error("Accepted invalid DMA stream port");
    }
    for (int col = 0; col < model.columns(); ++col)
      if (model.isShimPLTile(col, 0) &&
          model.getDmaStreamPort({col, 0}, DMAChannelDir::MM2S, 0))
        throw std::runtime_error("Accepted DMA on a PL-only shim");
    // Shim DMA endpoints precede routing through the shim mux.
    TileID shim{0, 0};
    while (!model.isShimNOCTile(shim.col, shim.row))
      ++shim.col;
    auto port = model.getDmaStreamPort(shim, DMAChannelDir::S2MM, 1);
    if (!port || *port != Port{WireBundle::DMA, 1})
      throw std::runtime_error("Incorrect logical shim DMA stream port");
  }
}

void testSpecializedDmaStreamPorts() {
  using namespace AIE;
  const auto &model = getTargetModel(AIEDevice::npu3);
  for (auto direction :
       {DMAChannelDir::ControlMM2S, DMAChannelDir::TraceS2MM}) {
    int count = direction == DMAChannelDir::ControlMM2S ? 2 : 1;
    WireBundle bundle = direction == DMAChannelDir::ControlMM2S
                            ? WireBundle::DMAControl
                            : WireBundle::DMATrace;
    for (int channel = 0; channel < count; ++channel) {
      auto port = model.getDmaStreamPort({0, 0}, direction, channel);
      if (!port || *port != Port{bundle, channel})
        throw std::runtime_error("Incorrect specialized DMA stream port");
    }
    if (model.getDmaStreamPort({0, 0}, direction, count) ||
        model.getDmaStreamPort({0, 1}, direction, 0) ||
        model.getDmaStreamPort({0, 2}, direction, 0) ||
        getTargetModel(AIEDevice::npu2).getDmaStreamPort({0, 0}, direction, 0))
      throw std::runtime_error("Accepted unsupported specialized DMA port");
  }
  if (model.getDmaStreamPort({0, 0}, DMAChannelDir::MM2S, 4) ||
      model.getDmaStreamPort({0, 0}, DMAChannelDir::S2MM, 2))
    throw std::runtime_error("Specialized DMA expanded ordinary DMA ports");
}

void testControlPortConnectivity() {
  using namespace AIE;
  const auto &model = getTargetModel(AIEDevice::npu3);
  // Control injection and trace reception are separate one-way endpoints.
  if (model.getNumSourceSwitchboxConnections(0, 0, WireBundle::DMAControl) !=
          2 ||
      model.getNumDestSwitchboxConnections(0, 0, WireBundle::DMAControl) != 0 ||
      model.getNumSourceSwitchboxConnections(0, 0, WireBundle::DMATrace) != 0 ||
      model.getNumDestSwitchboxConnections(0, 0, WireBundle::DMATrace) != 1 ||
      model.getNumSourceSwitchboxConnections(0, 0, WireBundle::TileControl) !=
          0 ||
      model.getNumDestSwitchboxConnections(0, 0, WireBundle::TileControl) != 2)
    throw std::runtime_error("Incorrect specialized port directionality");
  if (!model.isLegalTileConnection(0, 0, WireBundle::DMAControl, 0,
                                   WireBundle::North, 0) ||
      !model.isLegalTileConnection(0, 0, WireBundle::DMAControl, 1,
                                   WireBundle::North, 2) ||
      model.isLegalTileConnection(0, 0, WireBundle::DMAControl, 0,
                                  WireBundle::North, 2) ||
      model.isLegalTileConnection(0, 0, WireBundle::DMAControl, 0,
                                  WireBundle::DMATrace, 0))
    throw std::runtime_error("Incorrect control DMA connectivity");
}

void testDmaSelectorRegisters() {
  using namespace AIE;
  const auto &model = getTargetModel(AIEDevice::npu3);
  struct Case {
    DMAChannelDir direction;
    int channel, encoding;
    uint32_t bd, pool;
    uint64_t address;
    uint32_t control;
  };
  // Check selector encodings, shared data banks, and private control/trace
  // banks.
  const Case cases[] = {
      {DMAChannelDir::S2MM, 0, 0, 3, 0, 0x18090, 0x18540},
      {DMAChannelDir::S2MM, 1, 0, 19, 1, 0x58090, 0x58540},
      {DMAChannelDir::MM2S, 2, 1, 19, 1, 0x58090, 0x58550},
      {DMAChannelDir::ControlMM2S, 0, 2, 3, 200, 0x18390, 0x18560},
      {DMAChannelDir::ControlMM2S, 1, 2, 3, 201, 0x58390, 0x58560},
      {DMAChannelDir::TraceS2MM, 0, 3, 3, 300, 0x18510, 0x18548},
  };
  for (const auto &c : cases) {
    if (static_cast<int>(c.direction) != c.encoding)
      throw std::runtime_error("DMA selector encoding changed");
    if (model.getBdPoolId(0, 0, c.channel, c.direction) != c.pool ||
        model.getDmaBdAddress(0, 0, c.bd, c.channel, c.direction) !=
            c.address ||
        model.getDmaControlAddress(0, 0, c.channel, c.direction) != c.control)
      throw std::runtime_error("DMA selector selected the wrong register bank");
  }
}

void testDmaChannelSupport() {
  using namespace AIE;
  const auto &model = getTargetModel(AIEDevice::npu3);
  for (int col : {0, model.columns() - 1}) {
    for (int row = 0; row < model.rows(); ++row) {
      TileID tile{col, row};
      for (auto direction :
           {DMAChannelDir::MM2S, DMAChannelDir::S2MM,
            DMAChannelDir::ControlMM2S, DMAChannelDir::TraceS2MM}) {
        int count = 0;
        if (!isSpecializedShimDMAChannel(direction)) {
          if (model.isShimNOCTile(col, row))
            count = isMM2S(direction) ? 4 : 2;
          else if (model.isMemTile(col, row))
            count = isMM2S(direction) ? 10 : 8;
          else if (model.isCoreTile(col, row))
            count = isMM2S(direction) ? 1 : 2;
        } else if (model.isShimNOCTile(col, row)) {
          count = direction == DMAChannelDir::ControlMM2S ? 2 : 1;
        }
        if (model.getNumDmaChannels(tile, direction) !=
            static_cast<uint32_t>(count))
          throw std::runtime_error("Incorrect DMA channel count");
        for (int channel = -1; channel <= count + 1; ++channel) {
          bool expected = channel >= 0 && channel < count;
          if (model.isDmaChannelSupported(tile, direction, channel) !=
                  expected ||
              model.getDmaStreamPort(tile, direction, channel).has_value() !=
                  expected)
            throw std::runtime_error("Incorrect DMA channel support");
          if (!expected &&
              model.isBdChannelAccessible(col, row, 0, channel, direction))
            throw std::runtime_error("Accepted unsupported BD channel");
        }
      }
    }
  }
  for (TileID tile : {TileID{-1, 0}, TileID{model.columns(), 0}, TileID{0, -1},
                      TileID{0, model.rows()}})
    if (model.isDmaChannelSupported(tile, DMAChannelDir::MM2S, 0))
      throw std::runtime_error("Accepted DMA on an out-of-bounds tile");
  auto invalid = static_cast<DMAChannelDir>(99);
  if (model.getNumDmaChannels({0, 0}, invalid) != 0 ||
      model.getDmaStreamPort({0, 0}, invalid, 0))
    throw std::runtime_error("Accepted an invalid DMA selector");
  for (auto device : {AIEDevice::xcvc1902, AIEDevice::npu1, AIEDevice::npu2,
                      AIEDevice::xcve3858, AIEDevice::npu3}) {
    const auto &target = getTargetModel(device);
    int row = target.getNumMemTileRows() + 1;
    if (target.getDmaBdAddress(0, row, 3) !=
        target.getDmaBdAddress(0, row, 3, 0))
      throw std::runtime_error("Default BD channel changed");
  }
}

void testMemtileDmaBanks() {
  using namespace AIE;
  const auto &model = getTargetModel(AIEDevice::npu3);
  struct ChannelRegisters {
    DMAChannelDir direction;
    int channel;
    uint32_t bd0;
    uint32_t control;
  };
  // Cover every channel and both application banks.
  const ChannelRegisters channels[] = {
      {DMAChannelDir::S2MM, 0, 0x08000, 0x09E00},
      {DMAChannelDir::S2MM, 1, 0x08300, 0x09E08},
      {DMAChannelDir::S2MM, 2, 0x08600, 0x09E10},
      {DMAChannelDir::S2MM, 3, 0x08900, 0x09E18},
      {DMAChannelDir::S2MM, 4, 0x48000, 0x49E00},
      {DMAChannelDir::S2MM, 5, 0x48300, 0x49E08},
      {DMAChannelDir::S2MM, 6, 0x48600, 0x49E10},
      {DMAChannelDir::S2MM, 7, 0x48900, 0x49E18},
      {DMAChannelDir::MM2S, 0, 0x08C00, 0x09E60},
      {DMAChannelDir::MM2S, 1, 0x08F00, 0x09E68},
      {DMAChannelDir::MM2S, 2, 0x09200, 0x09E70},
      {DMAChannelDir::MM2S, 3, 0x09500, 0x09E78},
      {DMAChannelDir::MM2S, 4, 0x09800, 0x09E80},
      {DMAChannelDir::MM2S, 5, 0x48C00, 0x49E60},
      {DMAChannelDir::MM2S, 6, 0x48F00, 0x49E68},
      {DMAChannelDir::MM2S, 7, 0x49200, 0x49E70},
      {DMAChannelDir::MM2S, 8, 0x49500, 0x49E78},
      {DMAChannelDir::MM2S, 9, 0x49800, 0x49E80},
  };
  for (TileID tile : {TileID{0, 1}, TileID{2, 1}}) {
    uint64_t tileAddress = model.getTileAddress(tile.col, tile.row);
    std::set<uint64_t> bdAddresses;
    std::set<uint32_t> pools, controls;
    for (const auto &c : channels) {
      auto port = model.getDmaStreamPort(tile, c.direction, c.channel);
      if (!port || *port != Port{WireBundle::DMA, c.channel})
        throw std::runtime_error("Incorrect memtile DMA stream port");
      uint32_t pool =
          model.getBdPoolId(tile.col, tile.row, c.channel, c.direction);
      if (!pools.insert(pool).second ||
          model.getNumBDsInPool(tile.col, tile.row, pool) != 16 ||
          model.getBdPoolStartId(tile.col, tile.row, pool) != 0)
        throw std::runtime_error("Incorrect private memtile BD pool");
      uint32_t control = model.getDmaControlAddress(tile.col, tile.row,
                                                    c.channel, c.direction);
      if (control != (tileAddress | c.control) ||
          !controls.insert(control).second)
        throw std::runtime_error("Incorrect memtile DMA control register");
      for (uint32_t bd = 0; bd < 16; ++bd) {
        uint64_t address = model.getDmaBdAddress(tile.col, tile.row, bd,
                                                 c.channel, c.direction);
        if (address != (tileAddress | (c.bd0 + bd * 0x30)) ||
            !bdAddresses.insert(address).second ||
            !model.isBdChannelAccessible(tile.col, tile.row, bd, c.channel,
                                         c.direction))
          throw std::runtime_error("Incorrect or aliased memtile BD address");
      }
      if (model.isBdChannelAccessible(tile.col, tile.row, 16, c.channel,
                                      c.direction))
        throw std::runtime_error("Accepted out-of-bounds private BD ID");
    }
  }
}

int main() {
  testDmaStreamPorts();
  testSpecializedDmaStreamPorts();
  testControlPortConnectivity();
  testDmaSelectorRegisters();
  testDmaChannelSupport();
  testMemtileDmaBanks();
  return 0;
}
