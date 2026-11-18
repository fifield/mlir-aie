//===- target_model_bd_pools.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// BD pool queries of every target model, and the BD id generator.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace xilinx;
using namespace xilinx::AIE;

static void check(bool cond, const std::string &what) {
  if (!cond)
    throw std::runtime_error("Failed: " + what);
}

static const DMAChannelDir directions[] = {DMAChannelDir::MM2S,
                                           DMAChannelDir::S2MM};

// Pre-AIE4: AIE2 memtile even channels use BDs 0-23, odd 24-47; other tiles
// have one pool.
static bool legacyAccessible(const AIETargetModel &tm, int col, int row,
                             uint32_t bd, int channel) {
  if (tm.getTargetArch() == AIEArch::AIE1 || !tm.isMemTile(col, row))
    return true;
  return (channel & 1) == 0 ? bd < 24 : bd >= 24;
}

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
        if (!tm.isCoreTile(col, row) && !tm.isMemTile(col, row) &&
            !tm.isShimNOCTile(col, row))
          continue;
        std::string at = name + " (" + std::to_string(col) + ", " +
                         std::to_string(row) + ")";
        uint32_t numBDs = tm.getNumBDs(col, row);
        check(tm.getTotalNumBDs(col, row) == numBDs, at + " total BDs");
        bool memtile =
            tm.isMemTile(col, row) && tm.getTargetArch() != AIEArch::AIE1;
        for (DMAChannelDir dir : directions)
          for (int ch = 0;
               ch < static_cast<int>(tm.getNumDMAChannels(col, row, dir));
               ++ch) {
            std::string what = at + " channel " + std::to_string(ch);
            uint32_t pool = tm.getBdPoolId(col, row, ch, dir);
            check(pool == (memtile ? static_cast<uint32_t>(ch & 1) : 0),
                  what + " pool id");
            check(tm.getBdPoolStartId(col, row, pool) ==
                      (memtile && (ch & 1) ? 24 : 0),
                  what + " pool start");
            check(tm.getNumBDsInPool(col, row, pool) == (memtile ? 24 : numBDs),
                  what + " pool size");
            check(tm.getNumBDsForChannel(col, row, ch, dir) ==
                      tm.getNumBDsForChannel(col, row, ch),
                  what + " directed and undirected BD counts agree");
            for (uint32_t bd = 0; bd < numBDs; ++bd) {
              bool expected = legacyAccessible(tm, col, row, bd, ch);
              check(tm.isBdChannelAccessible(col, row, bd, ch, dir) == expected,
                    what + " BD " + std::to_string(bd));
              check(tm.isBdChannelAccessible(col, row, bd, ch) == expected,
                    what + " BD " + std::to_string(bd) + " undirected");
            }
          }
      }
  }
}

static void testNPU3() {
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu3);
  const DMAChannelDir mm2s = DMAChannelDir::MM2S, s2mm = DMAChannelDir::S2MM;

  check(tm.getTotalNumBDs(0, 0) == 52, "shim total BDs");
  check(tm.getTotalNumBDs(0, 1) == 16 * 18, "memtile total BDs");
  check(tm.getTotalNumBDs(0, 2) == 16, "core total BDs");

  for (DMAChannelDir dir : directions)
    for (int ch = 0; ch < static_cast<int>(tm.getNumDMAChannels(0, 2, dir));
         ++ch)
      check(tm.getBdPoolId(0, 2, ch, dir) == 0 &&
                tm.getBdIdRangeForChannel(0, 2, ch, dir) ==
                    std::make_pair(0u, 16u),
            "core pool");

  // Shim: two banks chosen by channel and direction.
  auto shimRange = [&](int ch, DMAChannelDir dir) {
    return tm.getBdIdRangeForChannel(0, 0, ch, dir);
  };
  check(shimRange(0, mm2s) == std::make_pair(0u, 16u) &&
            shimRange(1, mm2s) == std::make_pair(0u, 16u) &&
            shimRange(2, mm2s) == std::make_pair(16u, 32u) &&
            shimRange(3, mm2s) == std::make_pair(16u, 32u),
        "shim MM2S banks");
  check(shimRange(0, s2mm) == std::make_pair(0u, 16u) &&
            shimRange(1, s2mm) == std::make_pair(16u, 32u),
        "shim S2MM banks");
  check(tm.getBdPoolId(0, 0, 1, mm2s) == tm.getBdPoolId(0, 0, 0, s2mm) &&
            tm.getBdPoolId(0, 0, 1, s2mm) == tm.getBdPoolId(0, 0, 2, mm2s) &&
            tm.getBdPoolId(0, 0, 0, mm2s) != tm.getBdPoolId(0, 0, 1, s2mm),
        "shim pools are shared by bank");
  check(tm.isBdChannelAccessible(0, 0, 3, 1, mm2s) &&
            !tm.isBdChannelAccessible(0, 0, 3, 1, s2mm) &&
            tm.isBdChannelAccessible(0, 0, 20, 1, s2mm),
        "shim channel 1 depends on the direction");
  // The undirected form is conservative where the bank depends on direction.
  check(tm.getNumBDsForChannel(0, 0, 1) == 0 &&
            tm.getNumBDsForChannel(0, 0, 1, s2mm) == 16,
        "shim channel 1 undirected count");

  // Memtile: a private pool per channel and direction, channel-local ids.
  for (DMAChannelDir dir : directions)
    for (int ch = 0; ch < static_cast<int>(tm.getNumDMAChannels(0, 1, dir));
         ++ch) {
      std::string what = "memtile channel " + std::to_string(ch);
      check(tm.getBdIdRangeForChannel(0, 1, ch, dir) == std::make_pair(0u, 16u),
            what + " ids");
      for (DMAChannelDir otherDir : directions)
        for (int other = 0;
             other < static_cast<int>(tm.getNumDMAChannels(0, 1, otherDir));
             ++other)
          check((tm.getBdPoolId(0, 1, ch, dir) ==
                 tm.getBdPoolId(0, 1, other, otherDir)) ==
                    (ch == other && dir == otherDir),
                what + " pool is private");
    }
}

static void testBdIdGenerator() {
  const DMAChannelDir mm2s = DMAChannelDir::MM2S, s2mm = DMAChannelDir::S2MM;

  const AIETargetModel &npu1 = getTargetModel(AIEDevice::npu1);
  BdIdGenerator mem1(0, 1, npu1);
  check(mem1.nextBdId(0, s2mm) == 0u && mem1.nextBdId(1, mm2s) == 24u &&
            mem1.nextBdId(2, mm2s) == 1u && mem1.nextBdId(3, s2mm) == 25u,
        "npu1 memtile ids by parity");
  check(mem1.nextBdId(1, s2mm, 4) == 28u, "npu1 memtile aligned id");
  mem1.assignBdId(30);
  check(mem1.bdIdAlreadyAssigned(30) && mem1.bdIdAlreadyAssigned(1, mm2s, 30),
        "npu1 tile-wide reservation");
  check(mem1.nextBdId(1, mm2s) == 26u && mem1.nextBdId(1, mm2s) == 27u &&
            mem1.nextBdId(1, mm2s) == 29u,
        "npu1 reservation is skipped");

  const AIETargetModel &npu3 = getTargetModel(AIEDevice::npu3);
  BdIdGenerator mem3(0, 1, npu3);
  check(mem3.nextBdId(0, mm2s) == 0u && mem3.nextBdId(0, s2mm) == 0u &&
            mem3.nextBdId(1, mm2s) == 0u && mem3.nextBdId(0, mm2s) == 1u,
        "npu3 memtile private ids");
  for (uint32_t i = 2; i < 16; ++i)
    check(mem3.nextBdId(0, mm2s) == i, "npu3 memtile fills its pool");
  check(!mem3.nextBdId(0, mm2s), "npu3 memtile pool exhausted");
  check(mem3.nextBdId(9, mm2s) == 0u && mem3.nextBdId(7, s2mm) == 0u,
        "npu3 memtile other pools are free");
  mem3.freeBdId(0, mm2s, 5);
  check(!mem3.bdIdAlreadyAssigned(0, mm2s, 5) &&
            mem3.bdIdAlreadyAssigned(0, s2mm, 0),
        "npu3 memtile free is per pool");
  // A tile-wide id is taken in every pool.
  mem3.assignBdId(7);
  check(mem3.bdIdAlreadyAssigned(5, s2mm, 7) &&
            mem3.bdIdAlreadyAssigned(3, mm2s, 7),
        "npu3 memtile tile-wide reservation");

  // npu3 shim channel 1: low bank as MM2S, high bank as S2MM.
  BdIdGenerator shim3(0, 0, npu3);
  check(shim3.nextBdId(1, mm2s) == 0u && shim3.nextBdId(1, s2mm) == 16u &&
            shim3.nextBdId(0, mm2s) == 1u && shim3.nextBdId(2, mm2s) == 17u &&
            shim3.nextBdId(0, s2mm) == 2u && shim3.nextBdId(3, mm2s) == 18u,
        "npu3 shim banks");
}

using Group = DMAChannelAnalysis::BDGroup;
using User = DMAChannelAnalysis::BDUser;

static int numDmaChannels(const AIETargetModel &tm, int col, int row) {
  return std::max(
      tm.getNumSourceSwitchboxConnections(col, row, WireBundle::DMA),
      tm.getNumDestSwitchboxConnections(col, row, WireBundle::DMA));
}

static void testBdGroupsExistingTargets() {
  // Pre-AIE4 groups are channels with overlapping BD ids, both directions.
  const AIEDevice devices[] = {AIEDevice::xcvc1902, AIEDevice::xcve2802,
                               AIEDevice::npu1, AIEDevice::npu2,
                               AIEDevice::xcve3858};
  for (AIEDevice device : devices) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string name = stringifyAIEDevice(device).str();
    for (int col = 0; col < tm.columns(); ++col)
      for (int row = 0; row < tm.rows(); ++row) {
        int numChannels = numDmaChannels(tm, col, row);
        uint32_t numBDs = tm.getNumBDs(col, row);
        std::vector<std::pair<std::vector<bool>, std::vector<int>>> legacy;
        for (int channel = 0; channel < numChannels; ++channel) {
          std::vector<bool> ids(numBDs);
          for (uint32_t bd = 0; bd < numBDs; ++bd)
            ids[bd] = legacyAccessible(tm, col, row, bd, channel);
          std::pair<std::vector<bool>, std::vector<int>> merged{ids, {channel}};
          for (auto it = legacy.begin(); it != legacy.end();) {
            bool overlap = false;
            for (uint32_t bd = 0; bd < numBDs; ++bd)
              overlap |= it->first[bd] && merged.first[bd];
            if (!overlap) {
              ++it;
              continue;
            }
            for (uint32_t bd = 0; bd < numBDs; ++bd)
              merged.first[bd] = merged.first[bd] || it->first[bd];
            merged.second.insert(merged.second.end(), it->second.begin(),
                                 it->second.end());
            it = legacy.erase(it);
          }
          legacy.push_back(merged);
        }
        auto groups = DMAChannelAnalysis::bdGroups(tm, col, row, numChannels);
        std::string at = name + " (" + std::to_string(col) + ", " +
                         std::to_string(row) + ")";
        check(groups.size() == legacy.size(), at + " number of BD groups");
        for (size_t g = 0; g < groups.size(); ++g) {
          std::vector<int> channels = legacy[g].second;
          std::sort(channels.begin(), channels.end());
          std::vector<User> users;
          for (int channel : channels)
            for (DMAChannelDir dir : {DMAChannelDir::S2MM, DMAChannelDir::MM2S})
              users.push_back({dir, channel});
          int64_t size = 0;
          for (bool id : legacy[g].first)
            size += id;
          check(groups[g].size == size, at + " BD group size");
          check(std::vector<User>(groups[g].users.begin(),
                                  groups[g].users.end()) == users,
                at + " BD group channels");
        }
      }
  }
}

static void testBdGroupsNPU3() {
  const AIETargetModel &npu3 = getTargetModel(AIEDevice::npu3);
  auto groupOf = [](const llvm::SmallVector<Group> &groups, User user) {
    for (size_t g = 0; g < groups.size(); ++g)
      if (llvm::is_contained(groups[g].users, user))
        return static_cast<int>(g);
    return -1;
  };
  // Every memtile channel and direction plans against its own 16 BDs.
  auto mem =
      DMAChannelAnalysis::bdGroups(npu3, 0, 1, numDmaChannels(npu3, 0, 1));
  int memUsers = npu3.getNumDMAChannels(0, 1, DMAChannelDir::MM2S) +
                 npu3.getNumDMAChannels(0, 1, DMAChannelDir::S2MM);
  check(static_cast<int>(mem.size()) == memUsers,
        "npu3 memtile: one BD group per channel and direction");
  for (const Group &group : mem)
    check(group.size == 16 && group.users.size() == 1,
          "npu3 memtile BD group of 16");
  auto shim =
      DMAChannelAnalysis::bdGroups(npu3, 0, 0, numDmaChannels(npu3, 0, 0));
  int low = groupOf(shim, {DMAChannelDir::MM2S, 0});
  int high = groupOf(shim, {DMAChannelDir::S2MM, 1});
  check(low >= 0 && high >= 0 && low != high, "npu3 shim: two banks");
  check(groupOf(shim, {DMAChannelDir::MM2S, 1}) == low &&
            groupOf(shim, {DMAChannelDir::S2MM, 0}) == low,
        "npu3 shim low bank: MM2S 0-1 and S2MM 0");
  if (npu3.getNumDMAChannels(0, 0, DMAChannelDir::MM2S) > 2)
    check(groupOf(shim, {DMAChannelDir::MM2S, 2}) == high,
          "npu3 shim high bank: MM2S 2 and S2MM 1");
  for (const Group &group : shim)
    check(group.size == 16, "npu3 shim BD group of 16");
}

int main() {
  testExistingTargets();
  testNPU3();
  testBdIdGenerator();
  testBdGroupsExistingTargets();
  testBdGroupsNPU3();
  return 0;
}
