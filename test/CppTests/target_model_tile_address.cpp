//===- target_model_tile_address.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The tile address map hooks on every target.

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

static std::string tileName(const std::string &device, int col, int row) {
  return device + " (" + std::to_string(col) + ", " + std::to_string(row) + ")";
}

// Every address in [base, base + span) belongs to tile (col, row).
static void checkWindow(const AIETargetModel &tm, int col, int row,
                        uint32_t span, const std::string &name) {
  uint32_t base = tm.getTileAddress(col, row);
  for (uint32_t offset : {0u, 4u, span / 2, span - 4}) {
    uint32_t address = base + offset;
    check(tm.getColumnFromAddress(address) == static_cast<uint32_t>(col),
          name + " column of +" + std::to_string(offset));
    check(tm.getRowFromAddress(address) == static_cast<uint32_t>(row),
          name + " row of +" + std::to_string(offset));
  }
}

// Pre-AIE4: one address row per tile, data memory at offset 0.
static void testOneRowPerTile() {
  for (AIEDevice device :
       {AIEDevice::xcvc1902, AIEDevice::xcve2302, AIEDevice::xcve2802,
        AIEDevice::npu1, AIEDevice::npu1_1col, AIEDevice::npu1_2col,
        AIEDevice::npu1_3col, AIEDevice::npu2, AIEDevice::npu2_1col,
        AIEDevice::npu2_4col, AIEDevice::xcve3858}) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string deviceName = stringifyAIEDevice(device).str();
    uint32_t span = 1u << tm.getRowShift();
    check(tm.getRowFieldMask() ==
              (1u << (tm.getColumnShift() - tm.getRowShift())) - 1,
          deviceName + " row field mask");
    for (int col = 0; col < tm.columns(); ++col)
      for (int row = 0; row < tm.rows(); ++row) {
        std::string name = tileName(deviceName, col, row);
        uint32_t legacy = ((col & 0xff) << tm.getColumnShift()) |
                          ((row & 0xff) << tm.getRowShift());
        check(tm.getTileAddress(col, row) == legacy, name + " tile address");
        checkWindow(tm, col, row, span, name);
        check(tm.getDataMemoryOffset(col, row) == 0,
              name + " data memory offset");
        if (tm.isCoreTile(col, row) || tm.isMemTile(col, row) ||
            tm.isShimNOCTile(col, row)) {
          uint32_t control =
              tm.getDmaControlAddress(col, row, 0, DMAChannelDir::S2MM);
          check(tm.getLocalDmaControlAddress(
                    col, row, 0, DMAChannelDir::S2MM) == (control & (span - 1)),
                name + " local DMA control address");
        }
      }
  }
}

// AIE4: a memtile spans five 1 MiB address rows and its data memory starts
// 1 MiB in, so memtile row m sits at address row 5m - 4 and core row r at
// address row r + 4.
static void testAIE4() {
  constexpr uint32_t mib = 0x100000;
  for (AIEDevice device :
       {AIEDevice::npu3, AIEDevice::npu3_1col, AIEDevice::npu3_2col}) {
    const AIETargetModel &tm = getTargetModel(device);
    std::string deviceName = stringifyAIEDevice(device).str();
    check(tm.getNumMemTileRows() == 1, deviceName + " one memtile row");
    for (int col = 0; col < tm.columns(); ++col) {
      uint32_t column = static_cast<uint32_t>(col) << 25;
      check(tm.getTileAddress(col, 0) == column, deviceName + " shim");
      check(tm.getTileAddress(col, 1) == (column | 1 * mib),
            deviceName + " memtile");
      for (int row = 2; row < tm.rows(); ++row)
        check(tm.getTileAddress(col, row) ==
                  (column | static_cast<uint32_t>(row + 4) * mib),
              tileName(deviceName, col, row) + " core tile");

      // The windows tile the column's address space without gaps.
      for (int row = 0; row < tm.rows(); ++row) {
        std::string name = tileName(deviceName, col, row);
        uint32_t span = tm.isMemTile(col, row) ? 5 * mib : mib;
        checkWindow(tm, col, row, span, name);
        if (row + 1 < tm.rows())
          check(tm.getTileAddress(col, row + 1) ==
                    tm.getTileAddress(col, row) + span,
                name + " next window follows");
        uint32_t dataOffset = tm.getDataMemoryOffset(col, row);
        if (tm.isMemTile(col, row)) {
          check(dataOffset == mib, name + " memtile data memory offset");
          check(dataOffset + tm.getMemTileSize() <= span,
                name + " memtile data memory fits its window");
        } else {
          check(dataOffset == 0, name + " data memory offset");
        }
      }

      check(tm.getRowFromAddress(static_cast<uint32_t>(
                tm.getDmaBdAddress(col, 1, 0, 0, DMAChannelDir::MM2S))) == 1,
            deviceName + " memtile BD in the memtile window");
      check(tm.getRowFromAddress(
                tm.getDmaControlAddress(col, 2, 0, DMAChannelDir::S2MM)) == 2,
            deviceName + " core DMA control in the core window");
      check(tm.getLocalDmaControlAddress(col, 1, 0, DMAChannelDir::S2MM) ==
                tm.getDmaControlAddress(col, 1, 0, DMAChannelDir::S2MM) -
                    (column | mib),
            deviceName + " memtile local DMA control address");
    }
  }
}

int main() {
  testOneRowPerTile();
  testAIE4();
  return 0;
}
