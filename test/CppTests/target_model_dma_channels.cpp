//===- target_model_dma_channels.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DMA channel directions, incl. control and trace, across all targets.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

using namespace xilinx;
using namespace xilinx::AIE;

static void check(bool cond, const std::string &what) {
  if (!cond)
    throw std::runtime_error("Failed: " + what);
}

static const DMAChannelDir allDirections[] = {
    DMAChannelDir::S2MM, DMAChannelDir::MM2S, DMAChannelDir::ControlMM2S,
    DMAChannelDir::TraceS2MM};

static void testDirections() {
  // The integer values are part of the IR and of aiex.npu.sync.
  check(static_cast<int>(DMAChannelDir::S2MM) == 0, "S2MM = 0");
  check(static_cast<int>(DMAChannelDir::MM2S) == 1, "MM2S = 1");
  check(static_cast<int>(DMAChannelDir::ControlMM2S) == 2, "ControlMM2S = 2");
  check(static_cast<int>(DMAChannelDir::TraceS2MM) == 3, "TraceS2MM = 3");
  for (DMAChannelDir dir : allDirections) {
    std::string name = stringifyDMAChannelDir(dir).str();
    check(symbolizeDMAChannelDir(name) == dir, name + " round-trips");
    check(isMM2S(dir) != isS2MM(dir), name + " has one direction");
  }
  check(isMM2S(DMAChannelDir::MM2S) && isMM2S(DMAChannelDir::ControlMM2S),
        "MM2S directions");
  check(isS2MM(DMAChannelDir::S2MM) && isS2MM(DMAChannelDir::TraceS2MM),
        "S2MM directions");
  check(getDMAChannelBundle(DMAChannelDir::MM2S) == WireBundle::DMA &&
            getDMAChannelBundle(DMAChannelDir::S2MM) == WireBundle::DMA &&
            getDMAChannelBundle(DMAChannelDir::ControlMM2S) ==
                WireBundle::DMAControl &&
            getDMAChannelBundle(DMAChannelDir::TraceS2MM) ==
                WireBundle::DMATrace,
        "direction bundles");
}

// Shims count shim-mux DMA ports, other tiles switchbox DMA ports.
static uint32_t legacyNumDMAChannels(const AIETargetModel &tm, int col, int row,
                                     DMAChannelDir dir) {
  bool mm2s = dir == DMAChannelDir::MM2S;
  if (tm.hasShimMux(col, row))
    return mm2s ? tm.getNumSourceShimMuxConnections(col, row, WireBundle::DMA)
                : tm.getNumDestShimMuxConnections(col, row, WireBundle::DMA);
  return mm2s ? tm.getNumSourceSwitchboxConnections(col, row, WireBundle::DMA)
              : tm.getNumDestSwitchboxConnections(col, row, WireBundle::DMA);
}

// Pre-AIE4 targets have no control or trace DMA.
static void testExistingTargets() {
  const AIEDevice devices[] = {
      AIEDevice::xcvc1902,  AIEDevice::xcve2302,  AIEDevice::xcve2802,
      AIEDevice::npu1,      AIEDevice::npu1_1col, AIEDevice::npu1_2col,
      AIEDevice::npu1_3col, AIEDevice::npu2,      AIEDevice::npu2_1col,
      AIEDevice::npu2_4col, AIEDevice::xcve3858};
  for (AIEDevice device : devices) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string name = stringifyAIEDevice(device).str();
    for (int col = 0; col < tm.columns(); ++col)
      for (int row = 0; row < tm.rows(); ++row) {
        std::string at = name + " (" + std::to_string(col) + ", " +
                         std::to_string(row) + ")";
        for (DMAChannelDir dir : allDirections) {
          std::string what = at + " " + stringifyDMAChannelDir(dir).str();
          uint32_t expected = getDMAChannelBundle(dir) == WireBundle::DMA
                                  ? legacyNumDMAChannels(tm, col, row, dir)
                                  : 0;
          check(tm.getNumDMAChannels(col, row, dir) == expected,
                what + " channel count");
          for (int ch = -1; ch <= static_cast<int>(expected); ++ch)
            check(tm.isDmaChannelSupported({col, row}, dir, ch) ==
                      (ch >= 0 && ch < static_cast<int>(expected)),
                  what + " channel " + std::to_string(ch));
        }
      }
  }
}

// Off-device tiles and out-of-range directions (C API/Python pass integers)
// have no DMA channels.
static void testInvalidQueries() {
  const AIEDevice devices[] = {AIEDevice::xcvc1902, AIEDevice::xcve2802,
                               AIEDevice::npu1,     AIEDevice::npu2,
                               AIEDevice::xcve3858, AIEDevice::npu3};
  for (AIEDevice device : devices) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string name = stringifyAIEDevice(device).str();
    const std::pair<int, int> outside[] = {{-1, 0},
                                           {0, -1},
                                           {tm.columns(), 0},
                                           {0, tm.rows()},
                                           {tm.columns(), tm.rows()}};
    for (auto [col, row] : outside)
      for (DMAChannelDir dir : allDirections) {
        std::string what = name + " (" + std::to_string(col) + ", " +
                           std::to_string(row) + ") " +
                           stringifyDMAChannelDir(dir).str();
        check(tm.getNumDMAChannels(col, row, dir) == 0, what + " no channels");
        check(!tm.isDmaChannelSupported({col, row}, dir, 0),
              what + " channel 0 unsupported");
      }
    for (uint32_t raw : {4u, 7u, 0xffffffffu}) {
      auto dir = static_cast<DMAChannelDir>(raw);
      check(!symbolizeDMAChannelDir(raw), "not an enumerator");
      for (int col = 0; col < tm.columns(); ++col)
        for (int row = 0; row < tm.rows(); ++row) {
          std::string what = name + " (" + std::to_string(col) + ", " +
                             std::to_string(row) + ") direction " +
                             std::to_string(raw);
          check(tm.getNumDMAChannels(col, row, dir) == 0,
                what + " no channels");
          check(!tm.isDmaChannelSupported({col, row}, dir, 0),
                what + " channel 0 unsupported");
        }
    }
  }
}

// A PL-only shim has no shim DMA, but the count still reads its shim-mux DMA
// ports; this documents the value, not the hardware.
static void testPLShimDMAChannels() {
  const AIEDevice devices[] = {AIEDevice::xcvc1902, AIEDevice::xcve2302,
                               AIEDevice::xcve2802};
  for (AIEDevice device : devices) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string name = stringifyAIEDevice(device).str();
    int plShims = 0;
    for (int col = 0; col < tm.columns(); ++col) {
      if (!tm.isShimPLTile(col, 0))
        continue;
      ++plShims;
      std::string at = name + " PL shim (" + std::to_string(col) + ", 0)";
      check(tm.getNumDMAChannels(col, 0, DMAChannelDir::MM2S) == 2 &&
                tm.getNumDMAChannels(col, 0, DMAChannelDir::S2MM) == 2,
            at + " reports its shim mux's 2 DMA ports per direction");
      check(tm.getNumDMAChannels(col, 0, DMAChannelDir::ControlMM2S) == 0 &&
                tm.getNumDMAChannels(col, 0, DMAChannelDir::TraceS2MM) == 0,
            at + " has no control or trace DMA");
    }
    check(plShims > 0, name + " has PL-only shim tiles");
  }
  const AIEDevice npus[] = {AIEDevice::npu1,     AIEDevice::npu1_1col,
                            AIEDevice::npu2,     AIEDevice::npu2_1col,
                            AIEDevice::xcve3858, AIEDevice::npu3};
  for (AIEDevice device : npus) {
    const AIETargetModel &tm = getTargetModel(device);
    for (int col = 0; col < tm.columns(); ++col)
      for (int row = 0; row < tm.rows(); ++row)
        check(!tm.isShimPLTile(col, row),
              stringifyAIEDevice(device).str() + " has no PL-only shim");
  }
}

// A BD is accessible only from a channel the tile has.
static void testBdAccessibilityNeedsChannel() {
  const AIEDevice devices[] = {AIEDevice::xcvc1902, AIEDevice::xcve2802,
                               AIEDevice::npu1,     AIEDevice::npu2,
                               AIEDevice::xcve3858, AIEDevice::npu3};
  for (AIEDevice device : devices) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string name = stringifyAIEDevice(device).str();
    for (int col = 0; col < tm.columns(); ++col)
      for (int row = 0; row < tm.rows(); ++row) {
        uint32_t numBDs = tm.getNumBDs(col, row);
        for (DMAChannelDir dir : allDirections) {
          int channels = static_cast<int>(tm.getNumDMAChannels(col, row, dir));
          for (int ch = -1; ch <= channels; ++ch) {
            std::string what = name + " (" + std::to_string(col) + ", " +
                               std::to_string(row) + ") " +
                               stringifyDMAChannelDir(dir).str() + " " +
                               std::to_string(ch);
            bool exists = ch >= 0 && ch < channels;
            std::pair<uint32_t, uint32_t> range{0, 0};
            if (exists)
              range = tm.getBdIdRangeForChannel(col, row, ch, dir);
            auto pool = tm.getBdIdRangeForChannel(col, row, ch, dir);
            for (uint32_t bd = 0; bd < numBDs; ++bd) {
              check(tm.isBdChannelAccessible(col, row, bd, ch, dir) ==
                        (exists && bd >= range.first && bd < range.second),
                    what + " BD " + std::to_string(bd));
              check(tm.isBdInChannelPool(col, row, bd, ch, dir) ==
                        (bd >= pool.first && bd < pool.second),
                    what + " BD " + std::to_string(bd) + " in pool");
            }
          }
        }
        // Undirected: the channel in any data direction, every such pool.
        int maxChannels = static_cast<int>(
            std::max(tm.getNumDMAChannels(col, row, DMAChannelDir::MM2S),
                     tm.getNumDMAChannels(col, row, DMAChannelDir::S2MM)));
        for (int ch = -1; ch <= maxChannels; ++ch)
          for (uint32_t bd = 0; bd < numBDs; ++bd) {
            bool exists = false, inAll = true;
            for (DMAChannelDir dir : {DMAChannelDir::MM2S, DMAChannelDir::S2MM})
              if (tm.isDmaChannelSupported({col, row}, dir, ch)) {
                exists = true;
                auto [first, last] =
                    tm.getBdIdRangeForChannel(col, row, ch, dir);
                inAll &= bd >= first && bd < last;
              }
            check(tm.isBdChannelAccessible(col, row, bd, ch) ==
                      (exists && inAll),
                  name + " (" + std::to_string(col) + ", " +
                      std::to_string(row) + ") channel " + std::to_string(ch) +
                      " BD " + std::to_string(bd) + " undirected");
          }
      }
  }
  const AIETargetModel &npu3 = getTargetModel(AIEDevice::npu3);
  check(npu3.isBdChannelAccessible(0, 1, 0, 9, DMAChannelDir::MM2S) &&
            !npu3.isBdChannelAccessible(0, 1, 0, 10, DMAChannelDir::MM2S),
        "npu3 memtile MM2S 10 does not exist");
  check(npu3.isBdChannelAccessible(0, 0, 0, 1, DMAChannelDir::ControlMM2S) &&
            !npu3.isBdChannelAccessible(0, 0, 0, 2, DMAChannelDir::ControlMM2S),
        "npu3 shim ControlMM2S 2 does not exist");
  check(!npu3.isBdChannelAccessible(0, 2, 0, 0, DMAChannelDir::TraceS2MM),
        "npu3 core has no TraceS2MM");
  // MM2S-only channels: memtile MM2S 8-9, shim MM2S 2-3.
  check(npu3.getNumBDsForChannel(0, 1, 8) == 16 &&
            npu3.getNumBDsForChannel(0, 1, 9) == 16 &&
            npu3.getNumBDsForChannel(0, 1, 10) == 0,
        "npu3 memtile MM2S-only channels, undirected");
  check(npu3.getNumBDsForChannel(0, 0, 2) == 16 &&
            npu3.getNumBDsForChannel(0, 0, 3) == 16 &&
            npu3.getNumBDsForChannel(0, 0, 4) == 0,
        "npu3 shim MM2S-only channels, undirected");
}

// AIE4: each shim NOC tile has two ControlMM2S channels and one TraceS2MM
// channel, each with a private BD pool. No other tile has them.
static void testNPU3() {
  const AIEDevice devices[] = {AIEDevice::npu3, AIEDevice::npu3_1col,
                               AIEDevice::npu3_2col};
  const DMAChannelDir control = DMAChannelDir::ControlMM2S,
                      trace = DMAChannelDir::TraceS2MM;
  for (AIEDevice device : devices) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string name = stringifyAIEDevice(device).str();
    for (int col = 0; col < tm.columns(); ++col)
      for (int row = 0; row < tm.rows(); ++row) {
        std::string at = name + " (" + std::to_string(col) + ", " +
                         std::to_string(row) + ")";
        bool shim = tm.isShimNOCTile(col, row);
        check(tm.getNumDMAChannels(col, row, control) == (shim ? 2u : 0u),
              at + " ControlMM2S channels");
        check(tm.getNumDMAChannels(col, row, trace) == (shim ? 1u : 0u),
              at + " TraceS2MM channels");
        // Data DMA: shim 4/2, memtile 10/8, core 1/2.
        uint32_t mm2s = tm.getNumDMAChannels(col, row, DMAChannelDir::MM2S);
        uint32_t s2mm = tm.getNumDMAChannels(col, row, DMAChannelDir::S2MM);
        if (shim)
          check(mm2s == 4 && s2mm == 2, at + " shim data channels");
        else if (tm.isMemTile(col, row))
          check(mm2s == 10 && s2mm == 8, at + " memtile data channels");
        else if (tm.isCoreTile(col, row))
          check(mm2s == 1 && s2mm == 2, at + " core data channels");
        if (!tm.isCoreTile(col, row) && !tm.isMemTile(col, row) && !shim)
          continue;

        // The pools are disjoint and together hold every BD of the tile.
        std::set<uint32_t> pools;
        uint32_t total = 0;
        for (DMAChannelDir dir : allDirections)
          for (int ch = 0;
               ch < static_cast<int>(tm.getNumDMAChannels(col, row, dir));
               ++ch) {
            uint32_t pool = tm.getBdPoolId(col, row, ch, dir);
            if (pools.insert(pool).second)
              total += tm.getNumBDsInPool(col, row, pool);
          }
        check(total == tm.getTotalNumBDs(col, row), at + " pools add up");
        if (!shim)
          continue;

        uint64_t base = static_cast<uint64_t>(col) << tm.getColumnShift();
        for (int ch = 0; ch < 2; ++ch) {
          std::string what = at + " ControlMM2S " + std::to_string(ch);
          check(tm.getBdIdRangeForChannel(col, row, ch, control) ==
                    std::make_pair(0u, 8u),
                what + " BD ids");
          check(tm.getBdPoolId(col, row, ch, control) !=
                        tm.getBdPoolId(col, row, 1 - ch, control) &&
                    tm.getBdPoolId(col, row, ch, control) !=
                        tm.getBdPoolId(col, row, 0, DMAChannelDir::MM2S),
                what + " pool is private");
          uint64_t bank = ch == 0 ? 0x00018300 : 0x00058300;
          for (uint32_t bd = 0; bd < 8; ++bd)
            check(tm.getDmaBdAddress(col, row, bd, ch, control) ==
                      (base | (bank + bd * 0x30)),
                  what + " BD " + std::to_string(bd) + " address");
          check(tm.getDmaControlAddress(col, row, ch, control) ==
                    (base | (ch == 0 ? 0x00018560 : 0x00058560)),
                what + " control address");
        }
        check(tm.getBdIdRangeForChannel(col, row, 0, trace) ==
                  std::make_pair(0u, 4u),
              at + " TraceS2MM BD ids");
        check(tm.getBdPoolId(col, row, 0, trace) !=
                  tm.getBdPoolId(col, row, 0, DMAChannelDir::S2MM),
              at + " TraceS2MM pool is private");
        for (uint32_t bd = 0; bd < 4; ++bd)
          check(tm.getDmaBdAddress(col, row, bd, 0, trace) ==
                    (base | (0x00018480 + bd * 0x30)),
                at + " TraceS2MM BD " + std::to_string(bd) + " address");
        check(tm.getDmaControlAddress(col, row, 0, trace) ==
                  (base | 0x00018548),
              at + " TraceS2MM control address");
        check(tm.getDmaBdAddress(col, row, 17, 3, DMAChannelDir::MM2S) ==
                      (base | (0x00058000 + 1 * 0x30)) &&
                  tm.getDmaControlAddress(col, row, 0, DMAChannelDir::S2MM) ==
                      (base | 0x00018540),
              at + " data DMA addresses");
      }
  }
}

static void testBdIdGenerator() {
  const AIETargetModel &npu3 = getTargetModel(AIEDevice::npu3);
  const DMAChannelDir control = DMAChannelDir::ControlMM2S,
                      trace = DMAChannelDir::TraceS2MM,
                      mm2s = DMAChannelDir::MM2S;
  BdIdGenerator shim(1, 0, npu3);
  // Control and trace channels hand out ids independently of the data banks.
  check(shim.nextBdId(0, mm2s) == 0u && shim.nextBdId(0, control) == 0u &&
            shim.nextBdId(1, control) == 0u && shim.nextBdId(0, trace) == 0u,
        "npu3 shim private control and trace ids");
  for (uint32_t i = 1; i < 8; ++i)
    check(shim.nextBdId(0, control) == i, "npu3 ControlMM2S fills its pool");
  check(!shim.nextBdId(0, control), "npu3 ControlMM2S pool exhausted");
  for (uint32_t i = 1; i < 4; ++i)
    check(shim.nextBdId(0, trace) == i, "npu3 TraceS2MM fills its pool");
  check(!shim.nextBdId(0, trace), "npu3 TraceS2MM pool exhausted");
  check(shim.nextBdId(1, control) == 1u && shim.nextBdId(0, mm2s) == 1u,
        "npu3 other pools are unaffected");
  shim.freeBdId(0, control, 5);
  check(!shim.bdIdAlreadyAssigned(0, control, 5) &&
            shim.bdIdAlreadyAssigned(0, trace, 2),
        "npu3 free is per pool");

  // A per-channel id is taken only in that channel's pool.
  BdIdGenerator memtile(0, 1, npu3);
  memtile.assignBdId(0, DMAChannelDir::S2MM, 0);
  check(memtile.bdIdAlreadyAssigned(0, DMAChannelDir::S2MM, 0) &&
            !memtile.bdIdAlreadyAssigned(0, mm2s, 0) &&
            !memtile.bdIdAlreadyAssigned(1, DMAChannelDir::S2MM, 0),
        "npu3 memtile per-channel reservation");
  check(memtile.nextBdId(0, mm2s) == 0u &&
            memtile.nextBdId(0, DMAChannelDir::S2MM) == 1u,
        "npu3 memtile per-channel allocation");

  // A tile-wide id is taken in every pool.
  BdIdGenerator fresh(1, 0, npu3);
  fresh.assignBdId(2);
  check(fresh.bdIdAlreadyAssigned(0, control, 2) &&
            fresh.bdIdAlreadyAssigned(1, control, 2) &&
            fresh.bdIdAlreadyAssigned(0, trace, 2) &&
            fresh.bdIdAlreadyAssigned(0, mm2s, 2) &&
            !fresh.bdIdAlreadyAssigned(0, control, 3),
        "npu3 shim tile-wide reservation");
}

int main() {
  testDirections();
  testExistingTargets();
  testInvalidQueries();
  testPLShimDMAChannels();
  testBdAccessibilityNeedsChannel();
  testNPU3();
  testBdIdGenerator();
  return 0;
}
