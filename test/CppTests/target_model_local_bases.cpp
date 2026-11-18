//===- target_model_local_bases.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Neighbour lock/memory windows and cascade width on npu1/npu2 and npu3.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

using namespace xilinx;
using namespace xilinx::AIE;

static void check(bool cond, const std::string &what) {
  if (!cond)
    throw std::runtime_error("Failed: " + what);
}

static std::string at(int lc, int lr, int c, int r) {
  return " (" + std::to_string(lc) + "," + std::to_string(lr) + ")->(" +
         std::to_string(c) + "," + std::to_string(r) + ")";
}

// On AIE2 a core reaches S/W/N/own memory; a memtile its W/E neighbours and
// itself.
static void testAIE2Unchanged(AIEDevice device) {
  const AIETargetModel &tm = getTargetModel(device);
  std::string name = stringifyAIEDevice(device).str();
  const uint32_t coreLocks = 16, memTileLocks = 64;
  const uint32_t memTileSize = tm.getMemTileSize();
  check(memTileSize == 0x80000, name + " memtile size");
  for (int lc = 0; lc < tm.columns(); lc++)
    for (int lr = 0; lr < tm.rows(); lr++)
      for (int c = 0; c < tm.columns(); c++)
        for (int r = 0; r < tm.rows(); r++) {
          std::optional<uint32_t> lock, mem;
          if (tm.isCoreTile(lc, lr)) {
            if (c == lc && r == lr - 1) {
              lock = 0;
              mem = 0x40000;
            } else if (c == lc - 1 && r == lr) {
              lock = coreLocks;
              mem = 0x50000;
            } else if (c == lc && r == lr + 1) {
              lock = 2 * coreLocks;
              mem = 0x60000;
            } else if (c == lc && r == lr) {
              lock = 3 * coreLocks;
              mem = 0x70000;
            }
          } else if (tm.isMemTile(lc, lr) && r == lr && c >= lc - 1 &&
                     c <= lc + 1) {
            lock = (c - lc + 1) * memTileLocks;
            mem = (c - lc + 1) * memTileSize;
          }
          check(tm.getLockLocalBaseIndex(lc, lr, c, r) == lock,
                name + " lock base" + at(lc, lr, c, r));
          check(tm.getMemLocalBaseAddress(lc, lr, c, r) == mem,
                name + " mem base" + at(lc, lr, c, r));
        }

  check(tm.getMemWest({1, 3}) == TileID{0, 3}, name + " west memory");
  check(tm.getMemNorth({1, 3}) == TileID{1, 4}, name + " north memory");
  check(tm.getMemSouth({1, 3}) == TileID{1, 2}, name + " south memory");
  check(tm.getMemEast({1, 3}) == TileID{1, 3}, name + " east memory");
  check(!tm.getMemSouth({1, 2}), name + " no memory south of core row 2");

  check(tm.getAccumulatorCascadeSize() == 512, name + " cascade width");
}

// AIE4 memtile DMA: 15 windows (+-7 columns, westmost first), 64 locks and
// 3 MiB each from 0xB00000. Core tiles keep AIE2-shaped lock windows.
static void testNPU3(AIEDevice device) {
  const AIETargetModel &tm = getTargetModel(device);
  std::string name = stringifyAIEDevice(device).str();
  check(tm.getMemTileSize() == 0x300000, name + " memtile size");
  for (int lc = 0; lc < tm.columns(); lc++)
    for (int c = 0; c < tm.columns(); c++) {
      uint32_t window = c - lc + 7;
      check(tm.getLockLocalBaseIndex(lc, 1, c, 1) == window * 64,
            name + " memtile lock base" + at(lc, 1, c, 1));
      check(tm.getMemLocalBaseAddress(lc, 1, c, 1) ==
                0xB00000 + window * 0x300000,
            name + " memtile mem base" + at(lc, 1, c, 1));
      for (int r : {0, 2})
        check(!tm.getLockLocalBaseIndex(lc, 1, c, r) &&
                  !tm.getMemLocalBaseAddress(lc, 1, c, r),
              name + " memtile reaches no other row" + at(lc, 1, c, r));
    }
  // Own memory at 0x2000000, own locks at 448.
  check(tm.getMemLocalBaseAddress(0, 1, 0, 1) == 0x2000000u,
        name + " own memtile window");
  check(tm.getLockLocalBaseIndex(0, 1, 0, 1) == 448u,
        name + " own memtile locks");
  check(!tm.getMemLocalBaseAddress(0, 1, -1, 1) &&
            !tm.getLockLocalBaseIndex(0, 1, tm.columns(), 1),
        name + " memtile outside the array");

  check(tm.getLockLocalBaseIndex(0, 3, 0, 3) == 48u, name + " core own locks");
  check(tm.getMemLocalBaseAddress(0, 3, 0, 3) == 0xE0000u,
        name + " core own memory");

  // A core's data memory map holds only its own memory.
  for (int c = 0; c < tm.columns(); c++)
    for (int r = 2; r < tm.rows(); r++) {
      check(!tm.getMemWest({c, r}) && !tm.getMemNorth({c, r}) &&
                !tm.getMemSouth({c, r}),
            name + " no neighbour memory" + at(c, r, c, r));
      check(tm.getMemEast({c, r}) == TileID{c, r},
            name + " own memory" + at(c, r, c, r));
    }

  check(tm.getAccumulatorCascadeSize() == 512, name + " cascade width");
}

int main() {
  testAIE2Unchanged(AIEDevice::npu1);
  testAIE2Unchanged(AIEDevice::npu1_2col);
  testAIE2Unchanged(AIEDevice::npu2);
  testNPU3(AIEDevice::npu3);
  testNPU3(AIEDevice::npu3_1col);
  testNPU3(AIEDevice::npu3_2col);
  return 0;
}
