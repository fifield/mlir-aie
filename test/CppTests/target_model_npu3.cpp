//===- target_model_npu3.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The npu3 target models, including the queries that report "unsupported".

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <cstdint>
#include <stdexcept>
#include <string>

using namespace xilinx;
using namespace xilinx::AIE;

static void check(bool cond, const std::string &what) {
  if (!cond)
    throw std::runtime_error("Failed: " + what);
}

static constexpr uint64_t tileAddr(uint64_t col, uint64_t addrRow) {
  return (col << 25) | (addrRow << 20);
}

static void testDevices() {
  struct {
    AIEDevice device;
    int cols;
    bool virtualized;
  } devices[] = {{AIEDevice::npu3, 3, false},
                 {AIEDevice::npu3_1col, 1, true},
                 {AIEDevice::npu3_2col, 2, true}};
  for (auto [device, cols, virtualized] : devices) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string name = stringifyAIEDevice(device).str();
    check(tm.getTargetArch() == AIEArch::AIE4, name + " arch");
    check(llvm::isa<AIE4TargetModel>(tm), name + " isa<AIE4TargetModel>");
    check(llvm::isa<BaseNPU3TargetModel>(tm),
          name + " isa<BaseNPU3TargetModel>");
    check(llvm::isa<NPU3TargetModel>(tm) == !virtualized,
          name + " isa<NPU3TargetModel>");
    check(llvm::isa<VirtualizedNPU3TargetModel>(tm) == virtualized,
          name + " isa<VirtualizedNPU3TargetModel>");
    check(!llvm::isa<AIE1TargetModel, AIE2TargetModel, AIE2PSTargetModel>(tm),
          name + " !isa<AIE1/AIE2/AIE2PS>");
    check(tm.hasProperty(AIETargetModel::UsesSemaphoreLocks) &&
              tm.hasProperty(AIETargetModel::UsesMultiDimensionalBDs) &&
              tm.hasProperty(AIETargetModel::UsesBDIteration) &&
              tm.hasProperty(AIETargetModel::SupportsOutOfOrderDMA) &&
              tm.hasProperty(AIETargetModel::IsNPU),
          name + " properties");
    check(tm.hasProperty(AIETargetModel::IsVirtualized) == virtualized,
          name + " IsVirtualized");
    check(tm.columns() == cols && tm.rows() == 6, name + " shape");
    check(tm.getNumMemTileRows() == 1, name + " memtile rows");
    for (int c = 0; c < cols; ++c) {
      check(tm.isShimNOCTile(c, 0), name + " shim row");
      check(tm.isMemTile(c, 1), name + " memtile row");
      for (int r = 2; r < 6; ++r)
        check(tm.isCoreTile(c, r), name + " core rows");
    }
    check(tm.getNumControllersPerColumn() == 2 &&
              tm.getNumControllers() == static_cast<uint32_t>(2 * cols),
          name + " controllers");
    check(txnDeviceGen(tm) == 42, name + " txnDeviceGen");
    tm.validate();
  }
  check(txnDeviceGen(getTargetModel(AIEDevice::npu1)) == 3, "npu1 devGen");
  check(txnDeviceGen(getTargetModel(AIEDevice::npu2)) == 4, "npu2 devGen");
  check(getTargetModel(AIEDevice::npu2).getTargetArch() == AIEArch::AIE2p,
        "npu2 arch");
}

static void testMemoryAndLocks() {
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu3);
  // A core reaches only its own data memory.
  check(tm.isLegalMemAffinity(1, 3, 1, 3), "core own memory");
  for (auto [c, r] : {std::pair{0, 3}, {2, 3}, {1, 2}, {1, 4}})
    check(!tm.isLegalMemAffinity(1, 3, c, r), "core neighbour memory");
  check(tm.getLocalMemorySize() == 0x20000, "local memory size");
  check(tm.getProgramMemorySize() == 0x8000, "program memory size");
  check(tm.getMemTileSize() == 0x300000, "memtile size");
  check(tm.getAccumulatorCascadeSize() == 512, "cascade size");
  check(tm.getNumLocks(2, 2) == 16 && tm.getNumLocks(1, 1) == 64 &&
            tm.getNumLocks(1, 0) == 32,
        "lock counts");
  check(tm.getMaxLockValue() == 63, "max lock value");
  check(tm.getLocalLockAddress(1, {1, 2}) == 0xAE010u, "core lock address");
  check(tm.getLocalLockAddress(33, {1, 1}) == 0x50010u,
        "memtile upper-bank lock address");
  check(tm.getLocalLockAddress(17, {1, 0}) == 0x50010u,
        "shim upper-bank lock address");
  check(!tm.getLocalLockAddress(16, {1, 2}), "core lock out of range");
}

static void testDma() {
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu3);
  check(tm.getNumDMAChannels(0, 0, DMAChannelDir::MM2S) == 4 &&
            tm.getNumDMAChannels(0, 0, DMAChannelDir::S2MM) == 2,
        "shim DMA channels");
  check(tm.getNumDMAChannels(0, 1, DMAChannelDir::MM2S) == 10 &&
            tm.getNumDMAChannels(0, 1, DMAChannelDir::S2MM) == 8,
        "memtile DMA channels");
  check(tm.getNumDMAChannels(0, 2, DMAChannelDir::MM2S) == 1 &&
            tm.getNumDMAChannels(0, 2, DMAChannelDir::S2MM) == 2,
        "core DMA channels");

  check(tm.getNumBDs(0, 0) == 32 && tm.getNumBDs(0, 1) == 16 &&
            tm.getNumBDs(0, 2) == 16,
        "BD counts");
  check(tm.getBDMaxDims(0, 0) == 4 && tm.getBDMaxDims(0, 1) == 4 &&
            tm.getBDMaxDims(0, 2) == 3,
        "BD dimensions");
  check(tm.getMaxRepeatCount() == 4095, "repeat count");
  check(tm.getMaxOutOfOrderId() == 15, "out-of-order id");
  check(tm.getDmaTaskQueueDepth() == 0, "task queue depth unsupported");
  check(tm.getDmaBdLayout(0, 0) == nullptr, "no BD layout");

  // BD-to-channel access: answered only where it does not depend on the
  // direction.
  check(tm.isBdChannelAccessible(0, 2, 15, 0) &&
            !tm.isBdChannelAccessible(0, 2, 16, 0),
        "core BD access");
  check(tm.isBdChannelAccessible(0, 0, 0, 0) &&
            !tm.isBdChannelAccessible(0, 0, 16, 0),
        "shim channel 0 BD access");
  check(tm.isBdChannelAccessible(0, 0, 16, 2) &&
            tm.isBdChannelAccessible(0, 0, 31, 3) &&
            !tm.isBdChannelAccessible(0, 0, 0, 2),
        "shim channel 2/3 BD access");
  for (uint32_t bd = 0; bd < 32; ++bd)
    check(!tm.isBdChannelAccessible(0, 0, bd, 1),
          "shim channel 1 is unsupported");
  for (int ch = 0; ch < 10; ++ch)
    for (uint32_t bd = 0; bd < 16; ++bd)
      check(!tm.isBdChannelAccessible(0, 1, bd, ch),
            "memtile pools are unsupported");
  check(tm.getNumBDsForChannel(0, 0, 1) == 0 &&
            tm.getNumBDsForChannel(0, 1, 0) == 0,
        "unsupported channels have no BDs");
  check(tm.getNumBDsForChannel(0, 0, 0) == 16 &&
            tm.getNumBDsForChannel(0, 2, 0) == 16,
        "supported channels have 16 BDs");

  // Core row r sits at address row r + 4 (memtiles span five rows).
  check(tm.getDmaBdAddress(1, 2, 3, 0, DMAChannelDir::S2MM) ==
            (tileAddr(1, 6) | (0xAC000 + 3 * 0x20)),
        "core BD address");
  check(tm.getDmaBdAddress(2, 5, 0, 0, DMAChannelDir::MM2S) ==
            (tileAddr(2, 9) | 0xAC000),
        "top core BD address");
  check(tm.getDmaBdAddress(1, 1, 2, 6, DMAChannelDir::MM2S) ==
            (tileAddr(1, 1) | (0x40000 + 0x8C00 + 16 * 0x30 + 2 * 0x30)),
        "memtile MM2S BD address");
  check(tm.getDmaBdAddress(0, 1, 0, 3, DMAChannelDir::S2MM) ==
            (tileAddr(0, 1) | (0x8000 + 3 * 16 * 0x30)),
        "memtile S2MM BD address");
  check(tm.getDmaBdAddress(2, 0, 17, 3, DMAChannelDir::MM2S) ==
            (tileAddr(2, 0) | (0x58000 + 0x30)),
        "shim upper-bank BD address");
  check(tm.getDmaBdAddress(2, 0, 17, 1, DMAChannelDir::S2MM) ==
            (tileAddr(2, 0) | (0x58000 + 0x30)),
        "shim S2MM channel 1 BD address");
  check(tm.getDmaControlAddress(0, 3, 1, DMAChannelDir::S2MM) ==
            (tileAddr(0, 7) | 0xACE08),
        "core control address");
  check(tm.getDmaControlAddress(1, 1, 7, DMAChannelDir::MM2S) ==
            (tileAddr(1, 1) | (0x40000 + 0x9E60 + 2 * 8)),
        "memtile control address");
  check(tm.getDmaControlAddress(0, 0, 3, DMAChannelDir::MM2S) ==
            (tileAddr(0, 0) | 0x58558),
        "shim control address");
  check(tm.getLocalDmaControlAddress(0, 3, 1, DMAChannelDir::S2MM) == 0xACE08,
        "local control address");
}

static void testStreamSwitch() {
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu3);
  // 512-bit switch.
  check(tm.getNumSourceSwitchboxConnections(1, 2, WireBundle::South) == 8 &&
            tm.getNumDestSwitchboxConnections(1, 1, WireBundle::North) == 8,
        "memtile to core bus");
  check(tm.getNumSourceSwitchboxConnections(1, 0, WireBundle::DMAControl) ==
                2 &&
            tm.getNumDestSwitchboxConnections(1, 0, WireBundle::DMATrace) == 1,
        "shim control and trace DMA ports");
  check(
      tm.isLegalTileConnection(1, 2, WireBundle::DMA, 0, WireBundle::South, 0),
      "core DMA to South");
  check(!tm.isLegalTileConnection(1, 2, WireBundle::DMA, 0, WireBundle::South32,
                                  0),
        "no crossbar hop between the 512-bit and 32-bit switches");
  // 32-bit switch.
  check(tm.isLegalTileConnection(1, 2, WireBundle::South32, 0,
                                 WireBundle::North32, 1),
        "32-bit switch is fully connected");
  check(getConnectingBundle(WireBundle::North32) == WireBundle::South32 &&
            getConnectingBundle(WireBundle::NorthControl) ==
                WireBundle::SouthControl,
        "connecting bundles");
  // The event port map is not modelled.
  check(!tm.getStreamSwitchPortIndex(1, 2, WireBundle::DMA, 0, true),
        "stream switch port index unsupported");
  check(static_cast<int>(WireBundle::Controller32) == 11, "Controller32 = 11");
  check(static_cast<int>(WireBundle::DMAControl) == 24, "DMAControl = 24");
}

static void testStreamSwitchPortsMatchRegisterDatabase() {
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu3);

  check(tm.getNumDestSwitchboxConnections(1, 1, WireBundle::TileControl) == 1 &&
            tm.getNumDestSwitchboxConnections(1, 0, WireBundle::TileControl) ==
                1,
        "memtile and shim have one TileControl manager");
  check(tm.getNumSourceSwitchboxConnections(1, 1, WireBundle::Trace32) == 1 &&
            tm.getNumSourceSwitchboxConnections(1, 0, WireBundle::Trace32) == 1,
        "memtile and shim have one Trace32 subordinate");
  check(tm.getNumSourceSwitchboxConnections(1, 0, WireBundle::SouthControl) ==
            0,
        "shim has no SouthControl subordinate");
  check(tm.getNumSourceSwitchboxConnections(1, 2, WireBundle::West32) == 1 &&
            tm.getNumSourceSwitchboxConnections(1, 2, WireBundle::East32) == 1,
        "core has West32 and East32 subordinates");
}

int main() {
  testDevices();
  testMemoryAndLocks();
  testDma();
  testStreamSwitch();
  testStreamSwitchPortsMatchRegisterDatabase();
  return 0;
}
