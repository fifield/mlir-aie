//===- AIETargetModel.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <utility>

using namespace llvm;

namespace xilinx {
namespace AIE {

namespace {

std::string getModuleForTile(const AIETargetModel &model, TileID tile,
                             bool isMem) {
  if (model.isShimNOCorPLTile(tile.col, tile.row))
    return "shim";
  if (model.isMemTile(tile.col, tile.row))
    return "memory_tile";
  // In AIE4, core tiles have no separate memory module; Trace1 registers
  // are in the "core" module of the register database.
  AIEArch arch = model.getTargetArch();
  if (isMem && (arch == AIEArch::AIE4 || arch == AIEArch::AIE4z))
    return "core";
  return isMem ? std::string("memory") : std::string("core");
}

// Get module name for event lookups (events database uses different names)
std::string getModuleForTileEvents(const AIETargetModel &model, TileID tile,
                                   bool isMem) {
  if (model.isShimNOCorPLTile(tile.col, tile.row))
    return "pl"; // Events database uses "pl" for shim tiles
  if (model.isMemTile(tile.col, tile.row))
    return "mem_tile"; // Events database uses "mem_tile" instead of
                       // "memory_tile"
  // In AIE4, core tiles have no separate memory module; both trace units
  // (Trace0 and Trace1) use core module events.
  AIEArch arch = model.getTargetArch();
  if (isMem && (arch == AIEArch::AIE4 || arch == AIEArch::AIE4z))
    return "core";
  return isMem ? std::string("memory") : std::string("core");
}

} // namespace

AIETargetModel::~AIETargetModel() = default;

// Base class implementations for register database

std::unique_ptr<RegisterDatabase> AIETargetModel::loadRegisterDatabase() const {
  // Default: no register database available
  return nullptr;
}

const RegisterDatabase *AIETargetModel::getRegisterDatabase() const {
  std::call_once(regDBInitFlag, [this]() { regDB = loadRegisterDatabase(); });
  return regDB.get();
}

const RegisterInfo *AIETargetModel::lookupRegister(llvm::StringRef name,
                                                   TileID tile,
                                                   bool isMem) const {
  const auto *db = getRegisterDatabase();
  if (!db)
    return nullptr;
  return db->lookupRegister(name, getModuleForTile(*this, tile, isMem));
}

std::optional<uint32_t> AIETargetModel::lookupEvent(llvm::StringRef name,
                                                    TileID tile,
                                                    bool isMem) const {
  const auto *db = getRegisterDatabase();
  if (!db)
    return std::nullopt;
  return db->lookupEvent(name, getModuleForTileEvents(*this, tile, isMem));
}

std::optional<uint32_t>
AIETargetModel::encodeFieldValue(const BitFieldInfo &field,
                                 uint32_t value) const {
  const auto *db = getRegisterDatabase();
  if (!db)
    return std::nullopt;
  return db->encodeFieldValue(field, value);
}

std::optional<uint32_t>
AIETargetModel::getFieldMask(const BitFieldInfo &field) const {
  uint32_t width = field.getWidth();
  if (width == 0 || width > 32 || field.bit_end >= 32)
    return std::nullopt;

  uint64_t mask = (width == 32) ? 0xFFFFFFFFULL : ((1ULL << width) - 1ULL);
  mask <<= field.bit_start;
  if (mask > UINT32_MAX)
    return std::nullopt;

  return static_cast<uint32_t>(mask);
}

DenseMap<TileID, int>
AIETargetModel::getTileToControllerIdMap(bool columnWiseUniqueIDs) const {
  if (rows() <= 6) {
    // Controller id combinations chosen to avoid packet flow deadlock due to
    // single-level LUT bit-masking on packet header. Shim tile ids are < 16
    // because the TCT actor id field is only 4 bits.
    SmallVector<SmallVector<int>> validTileIds = {{15, 26, 27, 29, 30, 31},
                                                  {14, 21, 22, 23, 24, 25},
                                                  {13, 11, 17, 18, 19, 20},
                                                  {12, 28, 7, 8, 9, 10}};
    if (!columnWiseUniqueIDs && columns() > (int)validTileIds.size())
      llvm::report_fatal_error(
          "Device has " + llvm::Twine(columns()) + " columns but only " +
          llvm::Twine(validTileIds.size()) +
          " globally-unique controller ID sets are defined; "
          "use column-wise unique IDs mode.");
    DenseMap<TileID, int> tileIDMap;
    for (int col = 0; col < columns(); col++) {
      for (int row = 0; row < rows(); row++) {
        if (columnWiseUniqueIDs)
          tileIDMap[{col, row}] = validTileIds[0][row];
        else
          tileIDMap[{col, row}] = validTileIds[col][row];
      }
    }
    return tileIDMap;
  }

  // Devices with more than 6 rows use sequential assignment.
  assert(columnWiseUniqueIDs &&
         "Device has more tiles than candidate packet ids; "
         "use column-wise unique IDs mode.");
  DenseMap<TileID, int> tileIDMap;
  int unusedPacketIdFrom = 0;
  for (int col = 0; col < columns(); col++) {
    if (columnWiseUniqueIDs)
      unusedPacketIdFrom = 0;
    for (int row = 0; row < rows(); row++)
      tileIDMap[{col, row}] = unusedPacketIdFrom++;
  }
  return tileIDMap;
}

std::optional<uint32_t> AIETargetModel::resolvePortValue(llvm::StringRef value,
                                                         TileID tile,
                                                         bool master) const {
  auto colonPos = value.find(':');
  if (colonPos == StringRef::npos)
    return std::nullopt;

  StringRef portName = value.substr(0, colonPos);
  StringRef channelStr = value.substr(colonPos + 1);

  int channel;
  if (channelStr.getAsInteger(10, channel) || channel < 0)
    return std::nullopt;

  WireBundle bundle;
  if (portName.equals_insensitive("north")) {
    bundle = WireBundle::North;
  } else if (portName.equals_insensitive("south")) {
    bundle = WireBundle::South;
  } else if (portName.equals_insensitive("east")) {
    bundle = WireBundle::East;
  } else if (portName.equals_insensitive("west")) {
    bundle = WireBundle::West;
  } else if (portName.equals_insensitive("dma")) {
    bundle = WireBundle::DMA;
  } else if (portName.equals_insensitive("fifo")) {
    bundle = WireBundle::FIFO;
  } else if (portName.equals_insensitive("core")) {
    bundle = WireBundle::Core;
  } else if (portName.equals_insensitive("ctrl")) {
    bundle = WireBundle::TileControl;
  } else {
    return std::nullopt;
  }

  return getStreamSwitchPortIndex(tile.col, tile.row, bundle,
                                  static_cast<uint32_t>(channel), master);
}

///
/// AIE1 TargetModel
///

AIEArch AIE1TargetModel::getTargetArch() const { return AIEArch::AIE1; }

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE1TargetModel::getMemWest(TileID src) const {
  bool isEvenRow = ((src.row % 2) == 0);
  std::optional<TileID> ret;
  if (isEvenRow)
    ret = src;
  else
    ret = {src.col - 1, src.row};
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE1TargetModel::getMemEast(TileID src) const {
  bool isEvenRow = (src.row % 2) == 0;
  std::optional<TileID> ret;
  if (isEvenRow)
    ret = {src.col + 1, src.row};
  else
    ret = src;
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE1TargetModel::getMemNorth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row + 1});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

std::optional<TileID> AIE1TargetModel::getMemSouth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row - 1});
  // The first row doesn't have a tile memory south
  if (!isValidTile(*ret) || ret->row == 0)
    ret.reset();
  return ret;
}

bool AIE1TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  bool IsEvenRow = (srcRow % 2) == 0;
  return (IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (!IsEvenRow && isWest(srcCol, srcRow, dstCol, dstRow));
}

bool AIE1TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  bool IsEvenRow = (srcRow % 2) == 0;
  return (!IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (IsEvenRow && isEast(srcCol, srcRow, dstCol, dstRow));
}

bool AIE1TargetModel::isMemNorth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE1TargetModel::isMemSouth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE1TargetModel::isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                                         int memRow) const {
  bool IsEvenRow = ((coreRow % 2) == 0);

  bool IsMemWest = (isWest(coreCol, coreRow, memCol, memRow) && !IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) && IsEvenRow);

  bool IsMemEast = (isEast(coreCol, coreRow, memCol, memRow) && IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) && !IsEvenRow);

  bool IsMemNorth = isNorth(coreCol, coreRow, memCol, memRow);
  bool IsMemSouth = isSouth(coreCol, coreRow, memCol, memRow);

  return IsMemSouth || IsMemNorth || IsMemWest || IsMemEast;
}

uint64_t AIE1TargetModel::getDmaBdAddress(int col, int row, uint32_t bd_id,
                                          int channel,
                                          AIE::DMAChannelDir direction) const {
  uint32_t offset = 0;
  if (isShimNOCTile(col, row)) {
    offset = 0x0001D000 + (bd_id * 0x14);
  } else if (isCoreTile(col, row)) {
    offset = 0x0001D000 + (bd_id * 0x20);
  } else {
    llvm_unreachable(
        "AIE1TargetModel::getDmaBdAddress called for non-DMA tile");
  }
  return ((col & 0xff) << getColumnShift()) | ((row & 0x1f) << getRowShift()) |
         offset;
}

uint32_t AIE1TargetModel::getDmaBdAddressOffset(int col, int row) const {
  if (isShimNOCTile(col, row) || isCoreTile(col, row)) {
    return 0;
  }
  llvm_unreachable(
      "AIE1TargetModel::getDmaBdAddressOffset called for non-DMA tile");
}

uint32_t
AIE1TargetModel::getDmaControlAddress(int col, int row, int channel,
                                      AIE::DMAChannelDir direction) const {
  uint32_t offset = 0;
  if (isShimNOCTile(col, row))
    offset = 0x0001D140 + (channel * 0x8);
  else if (isCoreTile(col, row))
    offset = 0x0001DE00 + (channel * 0x8);
  else
    llvm_unreachable(
        "AIE1TargetModel::getDmaControlAddress called for non-DMA tile");

  if (direction == AIE::DMAChannelDir::MM2S)
    offset += 010;

  return ((col & 0xff) << getColumnShift()) | ((row & 0x1f) << getRowShift()) |
         offset;
}

uint32_t
AIE1TargetModel::getNumDestSwitchboxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 6;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 6;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::TileControl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  switch (bundle) {
  case WireBundle::Core:
  case WireBundle::DMA:
  case WireBundle::FIFO:
    return 2;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 6;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 4;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::TileControl:
    return 1;
  default:
    return 0;
  }
}

uint32_t
AIE1TargetModel::getNumSourceSwitchboxConnections(int col, int row,
                                                  WireBundle bundle) const {
  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 4;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 8;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::Trace:
      return 1;
    case WireBundle::TileControl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  switch (bundle) {
  case WireBundle::Core:
  case WireBundle::DMA:
  case WireBundle::FIFO:
    return 2;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 4;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 6;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::Trace:
    return 2;
  case WireBundle::TileControl:
    return 1;
  default:
    return 0;
  }
}
uint32_t
AIE1TargetModel::getNumDestShimMuxConnections(int col, int row,
                                              WireBundle bundle) const {
  if (isShimNOCorPLTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 6;
    case WireBundle::South:
      return 8; // Connection to the south port of the stream switch
    default:
      return 0;
    }
  return 0;
}
uint32_t
AIE1TargetModel::getNumSourceShimMuxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isShimNOCorPLTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 8;
    case WireBundle::South:
      return 6; // Connection to the south port of the stream switch
    default:
      return 0;
    }
  return 0;
}

bool AIE1TargetModel::isLegalTileConnection(int col, int row,
                                            WireBundle srcBundle, int srcChan,
                                            WireBundle dstBundle,
                                            int dstChan) const {
  // Check Channel Id within the range
  if (srcChan >= int(getNumSourceSwitchboxConnections(col, row, srcBundle)))
    return false;
  if (dstChan >= int(getNumDestSwitchboxConnections(col, row, dstBundle)))
    return false;

  // Memtile
  if (isMemTile(col, row)) {
    return false;
  }
  // Shimtile and Coretile currently share this trace-routing constraint;
  // flagged by bugprone-branch-clone -- unconfirmed whether that's
  // deliberate or a copy-paste that should differ per tile kind, needs
  // hardware-team confirmation before changing.
  // NOLINTBEGIN(bugprone-branch-clone)
  if (isShimNOCorPLTile(col, row)) {
    if (srcBundle == WireBundle::Trace)
      return dstBundle == WireBundle::South;
    return true;
  }
  // Coretile
  if (isCoreTile(col, row)) {
    if (srcBundle == WireBundle::Trace)
      return dstBundle == WireBundle::South;
    return true;
  }
  // NOLINTEND(bugprone-branch-clone)
  return false;
}

std::vector<std::pair<uint32_t, uint32_t>>
AIE1TargetModel::getShimBurstEncodingsAndLengths() const {
  return {std::pair(0, 64), std::pair(1, 128), std::pair(2, 256)};
}

std::optional<uint32_t>
AIE1TargetModel::getLocalLockAddress(uint32_t lockId, TileID tile) const {
  // This function is currently not supported for AIE1.
  // In order to be implemented for this target model, the interface
  // would need to change given the different way locks are written to in AIE1.
  return std::nullopt;
}

namespace {
namespace aie1_port_id {
namespace core {

// Slave port offset/size constants
static constexpr uint32_t S_CORE_OFFSET = 0;
static constexpr uint32_t S_CORE_SIZE = 2;
static constexpr uint32_t S_CTRL_OFFSET = 4;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_DMA_OFFSET = 2;
static constexpr uint32_t S_DMA_SIZE = 2;
static constexpr uint32_t S_EAST_OFFSET = 21;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 5;
static constexpr uint32_t S_FIFO_SIZE = 2;
static constexpr uint32_t S_NORTH_OFFSET = 17;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 7;
static constexpr uint32_t S_SOUTH_SIZE = 6;
static constexpr uint32_t S_TRACE_OFFSET = 25;
static constexpr uint32_t S_TRACE_SIZE = 2;
static constexpr uint32_t S_WEST_OFFSET = 13;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CORE_OFFSET = 0;
static constexpr uint32_t M_CORE_SIZE = 2;
static constexpr uint32_t M_CTRL_OFFSET = 4;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_DMA_OFFSET = 2;
static constexpr uint32_t M_DMA_SIZE = 2;
static constexpr uint32_t M_EAST_OFFSET = 21;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 5;
static constexpr uint32_t M_FIFO_SIZE = 2;
static constexpr uint32_t M_NORTH_OFFSET = 15;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 7;
static constexpr uint32_t M_SOUTH_SIZE = 4;
static constexpr uint32_t M_WEST_OFFSET = 11;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace core

namespace shim {

// Slave port offset/size constants
static constexpr uint32_t S_CTRL_OFFSET = 0;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_EAST_OFFSET = 19;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 1;
static constexpr uint32_t S_FIFO_SIZE = 2;
static constexpr uint32_t S_NORTH_OFFSET = 15;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 3;
static constexpr uint32_t S_SOUTH_SIZE = 8;
static constexpr uint32_t S_TRACE_OFFSET = 23;
static constexpr uint32_t S_TRACE_SIZE = 1;
static constexpr uint32_t S_WEST_OFFSET = 11;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CTRL_OFFSET = 0;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_EAST_OFFSET = 19;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 1;
static constexpr uint32_t M_FIFO_SIZE = 2;
static constexpr uint32_t M_NORTH_OFFSET = 13;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 3;
static constexpr uint32_t M_SOUTH_SIZE = 6;
static constexpr uint32_t M_WEST_OFFSET = 9;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace shim
} // namespace aie1_port_id
} // anonymous namespace

std::optional<uint32_t> AIE1TargetModel::getStreamSwitchPortIndex(
    int col, int row, WireBundle bundle, uint32_t port_num, bool master) const {
  if (master) {
    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie1_port_id::core::M_CORE_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::core::M_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie1_port_id::core::M_DMA_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::core::M_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::core::M_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::core::M_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::core::M_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::core::M_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCorPLTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::shim::M_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::shim::M_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::shim::M_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::shim::M_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::shim::M_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::shim::M_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  } else {
    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie1_port_id::core::S_CORE_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::core::S_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie1_port_id::core::S_DMA_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::core::S_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::core::S_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::core::S_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::core::S_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie1_port_id::core::S_TRACE_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::core::S_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCorPLTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::shim::S_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::shim::S_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::shim::S_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::shim::S_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::shim::S_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie1_port_id::shim::S_TRACE_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::shim::S_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }
}

///
/// AIE2 TargetModel
///

std::unique_ptr<RegisterDatabase>
AIE2TargetModel::loadRegisterDatabase() const {
  return RegisterDatabase::loadAIE2();
}

AIEArch AIE2TargetModel::getTargetArch() const { return AIEArch::AIE2; }

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE2TargetModel::getMemWest(TileID src) const {
  std::optional<TileID> ret({src.col - 1, src.row});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the east of the given tile (ie self), if
// it exists.
std::optional<TileID> AIE2TargetModel::getMemEast(TileID src) const {
  std::optional ret = src;
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the north of the given tile, if it
// exists.
std::optional<TileID> AIE2TargetModel::getMemNorth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row + 1});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

std::optional<TileID> AIE2TargetModel::getMemSouth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row - 1});
  // The first row doesn't have a tile memory south
  // Memtiles don't have memory adjacency to neighboring core tiles.
  if (!isValidTile(*ret) || ret->row == 0 || isMemTile(ret->col, ret->row))
    ret.reset();
  return ret;
}

bool AIE2TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isWest(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isMemNorth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isMemSouth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                                         int memRow) const {

  bool IsMemWest = isMemWest(coreCol, coreRow, memCol, memRow);
  bool IsMemEast = isMemEast(coreCol, coreRow, memCol, memRow);
  bool IsMemNorth = isMemNorth(coreCol, coreRow, memCol, memRow);
  bool IsMemSouth = isMemSouth(coreCol, coreRow, memCol, memRow);

  if (isMemTile(coreCol, coreRow))
    return isEast(coreCol, coreRow, memCol, memRow) ||
           isInternal(coreCol, coreRow, memCol, memRow) ||
           isWest(coreCol, coreRow, memCol, memRow);
  return (IsMemSouth && !isMemTile(memCol, memRow)) || IsMemNorth ||
         IsMemWest || IsMemEast;
}

uint64_t AIE2TargetModel::getDmaBdAddress(int col, int row, uint32_t bd_id,
                                          int channel,
                                          AIE::DMAChannelDir direction) const {
  uint64_t offset = 0;
  // Shimtile and Coretile currently compute the same offset; flagged by
  // bugprone-branch-clone -- unconfirmed whether the DMA BD register bank
  // genuinely shares this base across both tile kinds, needs hardware-team
  // confirmation before changing.
  // NOLINTBEGIN(bugprone-branch-clone)
  if (isShimNOCTile(col, row)) {
    offset = 0x0001D000 + bd_id * 0x20;
  } else if (isMemTile(col, row)) {
    offset = 0x000A0000 + bd_id * 0x20;
  } else if (isCoreTile(col, row)) {
    offset = 0x0001D000 + bd_id * 0x20;
  } else {
    llvm_unreachable(
        "AIE2TargetModel::getDmaBdAddress called for non-DMA tile");
  }
  // NOLINTEND(bugprone-branch-clone)
  return ((col & 0xff) << getColumnShift()) | ((row & 0x1f) << getRowShift()) |
         offset;
}

static DmaBdLayout makeAIE2ShimBdLayout() {
  DmaBdLayout l;
  l.numWords = 8;
  l.bufferLength = {0, 0, 32};
  l.bufferOffset = {1, 0, 32};
  l.enablePacket = {2, 30, 1};
  l.outOfOrderId = {2, 24, 6};
  l.packetId = {2, 19, 5};
  l.packetType = {2, 16, 3};
  l.d0Size = {3, 20, 10};
  l.d0Stride = {3, 0, 20};
  l.burstLength = {4, 30, 2};
  l.d1Size = {4, 20, 10};
  l.d1Stride = {4, 0, 20};
  l.axcache = {5, 24, 4};
  l.d2Stride = {5, 0, 20};
  l.iterationCurrent = {6, 26, 6};
  l.iterationSize = {6, 20, 6};
  l.iterationStride = {6, 0, 20};
  l.nextBd = {7, 27, 4};
  l.useNextBd = {7, 26, 1};
  l.validBd = {7, 25, 1};
  l.lockRelValue = {7, 18, 7};
  l.lockRelId = {7, 13, 4};
  l.lockAcqEnable = {7, 12, 1};
  l.lockAcqValue = {7, 5, 7};
  l.lockAcqId = {7, 0, 4};
  return l;
}

// D2_Wrap and D3_Stepsize are left out: the lowering never programs them, and
// the d2 extent is carried by buffer_length as on the other tiles.
static DmaBdLayout makeAIE2MemTileBdLayout() {
  DmaBdLayout l;
  l.numWords = 8;
  l.enablePacket = {0, 31, 1};
  l.packetType = {0, 28, 3};
  l.packetId = {0, 23, 5};
  l.outOfOrderId = {0, 17, 6};
  l.bufferLength = {0, 0, 17};
  l.d0ZeroBefore = {1, 26, 6};
  l.nextBd = {1, 20, 6};
  l.useNextBd = {1, 19, 1};
  l.bufferOffset = {1, 0, 19};
  l.d0Size = {2, 17, 10};
  l.d0Stride = {2, 0, 17};
  l.d1ZeroBefore = {3, 27, 5};
  l.d1Size = {3, 17, 10};
  l.d1Stride = {3, 0, 17};
  l.d2ZeroBefore = {4, 27, 4};
  l.d2Stride = {4, 0, 17};
  l.d2ZeroAfter = {5, 28, 4};
  l.d1ZeroAfter = {5, 23, 5};
  l.d0ZeroAfter = {5, 17, 6};
  l.iterationCurrent = {6, 23, 6};
  l.iterationSize = {6, 17, 6};
  l.iterationStride = {6, 0, 17};
  l.validBd = {7, 31, 1};
  l.lockRelValue = {7, 24, 7};
  l.lockRelId = {7, 16, 8};
  l.lockAcqEnable = {7, 15, 1};
  l.lockAcqValue = {7, 8, 7};
  l.lockAcqId = {7, 0, 8};
  return l;
}

// The core tile's buffer offset is its Base_Address field, in 32-bit words.
static DmaBdLayout makeAIE2CoreTileBdLayout() {
  DmaBdLayout l;
  l.numWords = 6;
  l.bufferOffset = {0, 14, 14};
  l.bufferLength = {0, 0, 14};
  l.enablePacket = {1, 30, 1};
  l.outOfOrderId = {1, 24, 6};
  l.packetId = {1, 19, 5};
  l.packetType = {1, 16, 3};
  l.d1Stride = {2, 13, 13};
  l.d0Stride = {2, 0, 13};
  l.d1Size = {3, 21, 8};
  l.d0Size = {3, 13, 8};
  l.d2Stride = {3, 0, 13};
  l.iterationCurrent = {4, 19, 6};
  l.iterationSize = {4, 13, 6};
  l.iterationStride = {4, 0, 13};
  l.nextBd = {5, 27, 4};
  l.useNextBd = {5, 26, 1};
  l.validBd = {5, 25, 1};
  l.lockRelValue = {5, 18, 7};
  l.lockRelId = {5, 13, 4};
  l.lockAcqEnable = {5, 12, 1};
  l.lockAcqValue = {5, 5, 7};
  l.lockAcqId = {5, 0, 4};
  return l;
}

const DmaBdLayout *AIE2TargetModel::getDmaBdLayout(AIETileType tileType) const {
  static const DmaBdLayout shim = makeAIE2ShimBdLayout();
  static const DmaBdLayout mem = makeAIE2MemTileBdLayout();
  static const DmaBdLayout core = makeAIE2CoreTileBdLayout();
  switch (tileType) {
  case AIETileType::ShimNOCTile:
    return &shim;
  case AIETileType::MemTile:
    return &mem;
  case AIETileType::CoreTile:
    return &core;
  default:
    return nullptr;
  }
}

uint32_t AIE2TargetModel::getDmaBdAddressOffset(int col, int row) const {
  if (isCoreTile(col, row))
    return 0x0;
  return 0x4;
}

uint32_t
AIE2TargetModel::getDmaControlAddress(int col, int row, int channel,
                                      AIE::DMAChannelDir direction) const {
  uint32_t offset = 0;
  if (isShimNOCTile(col, row)) {
    offset = 0x0001D200 + (channel * 0x8);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x10;
  } else if (isMemTile(col, row)) {
    offset = 0x000A0600 + (channel * 0x8);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x30;
  } else if (isCoreTile(col, row)) {
    offset = 0x0001DE00 + (channel * 0x8);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x10;
  } else {
    llvm_unreachable(
        "AIE2TargetModel::getDmaControlAddress called for non-DMA tile");
  }

  return ((col & 0xff) << getColumnShift()) | ((row & 0x1f) << getRowShift()) |
         offset;
}

std::optional<uint32_t>
AIE2TargetModel::getDmaStatusAddress(int col, int row, int channel,
                                     AIE::DMAChannelDir direction) const {
  // Status registers sit one word apart per channel, unlike the task-queue
  // control registers above, which are two words apart.
  uint32_t offset = 0;
  if (isShimNOCTile(col, row)) {
    offset = 0x0001D220 + (channel * 0x4);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x8;
  } else if (isMemTile(col, row)) {
    offset = 0x000A0660 + (channel * 0x4);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x20;
  } else if (isCoreTile(col, row)) {
    offset = 0x0001DF00 + (channel * 0x4);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x10;
  } else {
    return std::nullopt;
  }

  return ((col & 0xff) << getColumnShift()) | ((row & 0xff) << getRowShift()) |
         offset;
}

uint32_t
AIE2TargetModel::getNumDestSwitchboxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isMemTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
    case WireBundle::North:
      return 6;
    case WireBundle::South:
      return 4;
    case WireBundle::TileControl:
      return 1;
    default:
      return 0;
    }

  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      return 6;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 6;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::TileControl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  switch (bundle) {
  case WireBundle::Core:
    return 1;
  case WireBundle::DMA:
    return 2;
  case WireBundle::FIFO:
    return 1;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 6;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 4;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::TileControl:
    return 1;
  default:
    return 0;
  }
}

uint32_t
AIE2TargetModel::getNumSourceSwitchboxConnections(int col, int row,
                                                  WireBundle bundle) const {
  if (isMemTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 6;
    case WireBundle::North:
      return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::Trace:
    case WireBundle::TileControl:
      return 1;
    default:
      return 0;
    }

  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      return 4;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 8;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::Trace:
      return 1;
    case WireBundle::TileControl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  // compute/core tile
  switch (bundle) {
  case WireBundle::Core:
    return 1;
  case WireBundle::DMA:
    return 2;
  case WireBundle::FIFO:
    return 1;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 4;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 6;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::Trace:
    // Port 0: core trace. Port 1: memory trace.
    return 2;
  case WireBundle::TileControl:
    return 1;
  default:
    return 0;
  }
}

uint32_t
AIE2TargetModel::getNumDestShimMuxConnections(int col, int row,
                                              WireBundle bundle) const {
  if (isShimNOCorPLTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 6;
    case WireBundle::South:
      return 8; // Connection to the south port of the stream switch
    default:
      return 0;
    }

  return 0;
}

uint32_t
AIE2TargetModel::getNumSourceShimMuxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isShimNOCorPLTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 8;
    case WireBundle::South:
      return 6; // Connection to the south port of the stream switch
    default:
      return 0;
    }

  return 0;
}

bool AIE2TargetModel::isLegalTileConnection(int col, int row,
                                            WireBundle srcBundle, int srcChan,
                                            WireBundle dstBundle,
                                            int dstChan) const {
  // Check Channel Id within the range
  if (srcChan >= int(getNumSourceSwitchboxConnections(col, row, srcBundle)))
    return false;
  if (dstChan >= int(getNumDestSwitchboxConnections(col, row, dstBundle)))
    return false;

  // Lambda function to check if a bundle is in a list
  auto isBundleInList = [](WireBundle bundle,
                           std::initializer_list<WireBundle> bundles) {
    return llvm::find(bundles, bundle) != bundles.end();
  };

  // Memtile
  if (isMemTile(col, row)) {
    if (srcBundle == WireBundle::DMA) {
      if (dstBundle == WireBundle::DMA)
        return srcChan == dstChan;
      if (isBundleInList(dstBundle, {WireBundle::TileControl, WireBundle::South,
                                     WireBundle::North}))
        return true;
    }
    if (srcBundle == WireBundle::TileControl) {
      if (dstBundle == WireBundle::DMA)
        return dstChan == 5;
      if (isBundleInList(dstBundle, {WireBundle::South, WireBundle::North}))
        return true;
    }
    if (isBundleInList(srcBundle, {WireBundle::South, WireBundle::North})) {
      if (isBundleInList(dstBundle, {WireBundle::DMA, WireBundle::TileControl}))
        return true;
      if (isBundleInList(dstBundle, {WireBundle::South, WireBundle::North}))
        return srcChan == dstChan;
    }
    if (srcBundle == WireBundle::Trace) {
      if (dstBundle == WireBundle::DMA)
        return dstChan == 5;
      if (dstBundle == WireBundle::South)
        return true;
    }
  }
  // Shimtile
  else if (isShimNOCorPLTile(col, row)) {
    if (srcBundle == WireBundle::TileControl)
      return dstBundle != WireBundle::TileControl;
    if (isBundleInList(srcBundle, {WireBundle::FIFO, WireBundle::South}))
      return isBundleInList(dstBundle,
                            {WireBundle::TileControl, WireBundle::FIFO,
                             WireBundle::South, WireBundle::West,
                             WireBundle::North, WireBundle::East});
    if (isBundleInList(srcBundle,
                       {WireBundle::West, WireBundle::North, WireBundle::East}))
      return (srcBundle == dstBundle)
                 ? (srcChan == dstChan)
                 : isBundleInList(dstBundle,
                                  {WireBundle::TileControl, WireBundle::FIFO,
                                   WireBundle::South, WireBundle::West,
                                   WireBundle::North, WireBundle::East});
    if (srcBundle == WireBundle::Trace) {
      if (isBundleInList(dstBundle, {WireBundle::FIFO, WireBundle::South}))
        return true;
      if (isBundleInList(dstBundle, {WireBundle::West, WireBundle::East}))
        return dstChan == 0;
    }
  }
  // Coretile
  if (isCoreTile(col, row)) {
    if (isBundleInList(srcBundle,
                       {WireBundle::DMA, WireBundle::FIFO, WireBundle::South,
                        WireBundle::West, WireBundle::North, WireBundle::East}))
      if (isBundleInList(dstBundle, {WireBundle::Core, WireBundle::DMA,
                                     WireBundle::TileControl, WireBundle::FIFO,
                                     WireBundle::South, WireBundle::West,
                                     WireBundle::North, WireBundle::East}))
        return (srcBundle == dstBundle) ? (srcChan == dstChan) : true;
    if (srcBundle == WireBundle::Core)
      return dstBundle != WireBundle::Core;
    if (srcBundle == WireBundle::TileControl)
      return dstBundle != WireBundle::TileControl &&
             dstBundle != WireBundle::DMA;
    if (srcBundle == WireBundle::Trace) {
      if (dstBundle == WireBundle::DMA)
        return dstChan == 0;
      if (isBundleInList(dstBundle, {WireBundle::FIFO, WireBundle::South}))
        return true;
    }
  }
  return false;
}

std::vector<std::pair<uint32_t, uint32_t>>
AIE2TargetModel::getShimBurstEncodingsAndLengths() const {
  return {std::pair(0, 64), std::pair(1, 128), std::pair(2, 256)};
}

std::optional<uint32_t>
AIE2TargetModel::getLocalLockAddress(uint32_t lockId, TileID tile) const {
  uint32_t computeTileBaseAddress = 0x0001F000;
  uint32_t memTileBaseAddress = 0x000C0000;
  uint32_t shimTileBaseAddress = 0x00014000;
  uint32_t lockAddrOffset = 0x10;

  if (isCoreTile(tile.col, tile.row) &&
      lockId < AIETargetModel::getNumLocks(tile.col, tile.row))
    return computeTileBaseAddress + lockAddrOffset * lockId;

  if (isMemTile(tile.col, tile.row) &&
      lockId < AIETargetModel::getNumLocks(tile.col, tile.row))
    return memTileBaseAddress + lockAddrOffset * lockId;

  if (isShimNOCorPLTile(tile.col, tile.row) &&
      lockId < AIETargetModel::getNumLocks(tile.col, tile.row))
    return shimTileBaseAddress + lockAddrOffset * lockId;

  return std::nullopt;
}

namespace {
namespace aie2_port_id {
namespace core {

// Slave port offset/size constants
static constexpr uint32_t S_CORE_OFFSET = 0;
static constexpr uint32_t S_CORE_SIZE = 1;
static constexpr uint32_t S_CTRL_OFFSET = 3;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_DMA_OFFSET = 1;
static constexpr uint32_t S_DMA_SIZE = 2;
static constexpr uint32_t S_EAST_OFFSET = 19;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 4;
static constexpr uint32_t S_FIFO_SIZE = 1;
static constexpr uint32_t S_NORTH_OFFSET = 15;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 5;
static constexpr uint32_t S_SOUTH_SIZE = 6;
static constexpr uint32_t S_TRACE_OFFSET = 23;
static constexpr uint32_t S_TRACE_SIZE = 2;
static constexpr uint32_t S_WEST_OFFSET = 11;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CORE_OFFSET = 0;
static constexpr uint32_t M_CORE_SIZE = 1;
static constexpr uint32_t M_CTRL_OFFSET = 3;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_DMA_OFFSET = 1;
static constexpr uint32_t M_DMA_SIZE = 2;
static constexpr uint32_t M_EAST_OFFSET = 19;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 4;
static constexpr uint32_t M_FIFO_SIZE = 1;
static constexpr uint32_t M_NORTH_OFFSET = 13;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 5;
static constexpr uint32_t M_SOUTH_SIZE = 4;
static constexpr uint32_t M_WEST_OFFSET = 9;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace core

namespace mem {

// Slave port offset/size constants
static constexpr uint32_t S_CTRL_OFFSET = 6;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_DMA_OFFSET = 0;
static constexpr uint32_t S_DMA_SIZE = 6;
static constexpr uint32_t S_NORTH_OFFSET = 13;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 7;
static constexpr uint32_t S_SOUTH_SIZE = 6;
static constexpr uint32_t S_TRACE_OFFSET = 17;
static constexpr uint32_t S_TRACE_SIZE = 1;

// Master port offset/size constants
static constexpr uint32_t M_CTRL_OFFSET = 6;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_DMA_OFFSET = 0;
static constexpr uint32_t M_DMA_SIZE = 6;
static constexpr uint32_t M_NORTH_OFFSET = 11;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 7;
static constexpr uint32_t M_SOUTH_SIZE = 4;

} // namespace mem

namespace shim {

// Slave port offset/size constants
static constexpr uint32_t S_CTRL_OFFSET = 0;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_EAST_OFFSET = 18;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 1;
static constexpr uint32_t S_FIFO_SIZE = 1;
static constexpr uint32_t S_NORTH_OFFSET = 14;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 2;
static constexpr uint32_t S_SOUTH_SIZE = 8;
static constexpr uint32_t S_TRACE_OFFSET = 22;
static constexpr uint32_t S_TRACE_SIZE = 2;
static constexpr uint32_t S_WEST_OFFSET = 10;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CTRL_OFFSET = 0;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_EAST_OFFSET = 18;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 1;
static constexpr uint32_t M_FIFO_SIZE = 1;
static constexpr uint32_t M_NORTH_OFFSET = 12;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 2;
static constexpr uint32_t M_SOUTH_SIZE = 6;
static constexpr uint32_t M_WEST_OFFSET = 8;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace shim
} // namespace aie2_port_id
} // namespace

std::optional<uint32_t> AIE2TargetModel::getStreamSwitchPortIndex(
    int col, int row, WireBundle bundle, uint32_t port_num, bool master) const {

  if (master) {
    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie2_port_id::core::M_CORE_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::core::M_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::core::M_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::core::M_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::core::M_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::core::M_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::core::M_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::core::M_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isMemTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::mem::M_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::mem::M_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_DMA_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::mem::M_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::mem::M_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_SOUTH_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::shim::M_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::shim::M_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::shim::M_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::shim::M_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::shim::M_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::shim::M_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    }
    // Slave ports
  } else {

    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie2_port_id::core::S_CORE_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::core::S_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::core::S_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::core::S_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::core::S_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::core::S_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::core::S_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie2_port_id::core::S_TRACE_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::core::S_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isMemTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::mem::S_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::mem::S_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_DMA_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::mem::S_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::mem::S_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie2_port_id::mem::S_TRACE_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_TRACE_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::shim::S_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::shim::S_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::shim::S_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::shim::S_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::shim::S_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie2_port_id::shim::S_TRACE_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::shim::S_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

void AIETargetModel::validate() const {
  // Every tile in a shimtile row must be a shimtile, and can only be one type
  // of shim tile.
  for (int j = 0; j < columns(); j++) {
    assert(!isMemTile(j, 0) && (isShimPLTile(j, 0) || isShimNOCTile(j, 0)) &&
           !isCoreTile(j, 0));
    assert(isShimPLTile(j, 0) ^ isShimNOCTile(j, 0));
  }

  // Every tile in a memtile row must be a memtile.
  for (int i = 1; i < 1 + static_cast<int>(getNumMemTileRows()); i++)
    for (int j = 0; j < columns(); j++)
      assert(isMemTile(j, i) && !isShimPLTile(j, i) && !isShimNOCTile(j, i) &&
             !isCoreTile(j, i));

  // Every other tile is a coretile.
  for (int i = 1 + getNumMemTileRows(); i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(!isMemTile(j, i) && !isShimPLTile(j, i) && !isShimNOCTile(j, i) &&
             isCoreTile(j, i));

  // Looking North, buses must match
  for (int i = 0; i < rows() - 1; i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::North) ==
             getNumDestSwitchboxConnections(j, i + 1, WireBundle::South));
  // Looking South, buses must match
  for (int i = 1; i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::South) ==
             getNumDestSwitchboxConnections(j, i - 1, WireBundle::North));
  // Looking East, buses must match
  for (int i = 0; i < rows(); i++)
    for (int j = 0; j < columns() - 1; j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::East) ==
             getNumDestSwitchboxConnections(j + 1, i, WireBundle::West));
  // Looking West, buses must match
  for (int i = 0; i < rows(); i++)
    for (int j = 1; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::West) ==
             getNumDestSwitchboxConnections(j - 1, i, WireBundle::East));
  // Edges have no connections
  for (int j = 0; j < columns(); j++)
    assert(getNumSourceSwitchboxConnections(j, rows() - 1, WireBundle::North) ==
           0);
  for (int i = 0; i < rows(); i++)
    assert(getNumSourceSwitchboxConnections(columns() - 1, i,
                                            WireBundle::East) == 0);
  for (int i = 0; i < rows(); i++)
    assert(getNumSourceSwitchboxConnections(0, i, WireBundle::West) == 0);

  // FIFOS are consistent
  for (int i = 0; i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::FIFO) ==
             getNumDestSwitchboxConnections(j, i, WireBundle::FIFO));
}

std::optional<uint32_t>
AIETargetModel::getLockLocalBaseIndex(int localCol, int localRow, int lockCol,
                                      int lockRow) const {
  if (isCoreTile(localCol, localRow)) {
    if (isMemSouth(localCol, localRow, lockCol, lockRow))
      return 0;
    if (isMemWest(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow);
    if (isMemNorth(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow) * 2;
    if (isMemEast(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow) * 3;
  }

  if (isMemTile(localCol, localRow)) {
    if (isWest(localCol, localRow, lockCol, lockRow))
      return 0;
    if (isInternal(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow);
    if (isEast(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow) * 2;
  }

  return std::nullopt;
}

std::optional<uint32_t>
AIETargetModel::getMemLocalBaseAddress(int localCol, int localRow, int memCol,
                                       int memRow) const {
  if (isCoreTile(localCol, localRow)) {
    if (isMemSouth(localCol, localRow, memCol, memRow))
      return getMemSouthBaseAddress();
    if (isMemWest(localCol, localRow, memCol, memRow))
      return getMemWestBaseAddress();
    if (isMemNorth(localCol, localRow, memCol, memRow))
      return getMemNorthBaseAddress();
    if (isMemEast(localCol, localRow, memCol, memRow))
      return getMemEastBaseAddress();
  }

  if (isMemTile(localCol, localRow)) {
    if (isWest(localCol, localRow, memCol, memRow))
      return 0;
    if (isInternal(localCol, localRow, memCol, memRow))
      return getMemTileSize();
    if (isEast(localCol, localRow, memCol, memRow))
      return getMemTileSize() * 2;
  }

  return std::nullopt;
}

uint32_t AIETargetModel::getNumDMAChannels(int col, int row,
                                           AIE::DMAChannelDir dir) const {
  bool mm2s = dir == AIE::DMAChannelDir::MM2S;
  if (isShimNOCorPLTile(col, row))
    return mm2s ? getNumSourceShimMuxConnections(col, row, WireBundle::DMA)
                : getNumDestShimMuxConnections(col, row, WireBundle::DMA);
  return mm2s ? getNumSourceSwitchboxConnections(col, row, WireBundle::DMA)
              : getNumDestSwitchboxConnections(col, row, WireBundle::DMA);
}

bool AIETargetModel::isSupportedBlockFormat(std::string const &format) const {
  return false;
}

AIEArch BaseNPU2TargetModel::getTargetArch() const { return AIEArch::AIE2p; }

std::vector<std::pair<uint32_t, uint32_t>>
BaseNPU2TargetModel::getShimBurstEncodingsAndLengths() const {
  return {std::pair(0, 64), std::pair(1, 128), std::pair(2, 256),
          std::pair(3, 512)};
}

///
/// AIE4 TargetModel
///

std::unique_ptr<RegisterDatabase>
AIE4TargetModel::loadRegisterDatabase() const {
  return RegisterDatabase::loadAIE4();
}

std::unique_ptr<RegisterDatabase>
NPU3zTargetModel::loadRegisterDatabase() const {
  return RegisterDatabase::loadAIE4Z();
}

AIEArch AIE4TargetModel::getTargetArch() const { return AIEArch::AIE4; }

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE4TargetModel::getMemWest(TileID src) const {
  std::optional<TileID> ret({src.col - 1, src.row});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the east of the given tile (ie self), if
// it exists.
std::optional<TileID> AIE4TargetModel::getMemEast(TileID src) const {
  std::optional ret = src;
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the north of the given tile, if it
// exists.
std::optional<TileID> AIE4TargetModel::getMemNorth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row + 1});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

std::optional<TileID> AIE4TargetModel::getMemSouth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row - 1});
  // The first row doesn't have a tile memory south
  // Memtiles don't have memory adjacency to neighboring core tiles.
  if (!isValidTile(*ret) || ret->row == 0 || isMemTile(ret->col, ret->row))
    ret.reset();
  return ret;
}

bool AIE4TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isWest(srcCol, srcRow, dstCol, dstRow);
}

bool AIE4TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
}

bool AIE4TargetModel::isMemNorth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE4TargetModel::isMemSouth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE4TargetModel::isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                                         int memRow) const {

  if (isMemTile(coreCol, coreRow))
    return isEast(coreCol, coreRow, memCol, memRow) ||
           isInternal(coreCol, coreRow, memCol, memRow) ||
           isWest(coreCol, coreRow, memCol, memRow);

  // AIE4 compute tiles have no load/store interface to neighboring tile data
  // memory. Only the tile's own data memory is reachable (mapped via the
  // East/Internal direction by isMemEast). Cross-tile data movement must go
  // through the tile DMAs and stream switches.
  return isMemEast(coreCol, coreRow, memCol, memRow);
}

uint64_t AIE4TargetModel::getDmaBdAddress(int col, int row, uint32_t bd_id,
                                          int channel,
                                          AIE::DMAChannelDir direction) const {

  uint64_t offset = 0;
  if (isShimNOCTile(col, row)) {
    uint32_t numBds = 16;
    if (direction == AIE::DMAChannelDir::MM2S) {
      if (getTargetArch() == AIEArch::AIE4z) {
        offset = 0x00018C40;
      } else {
        if (channel < 2)
          offset = 0x00018000;
        else
          offset = 0x00058000;
        channel = 0;
      }
    } else {
      if (channel == 0) {
        offset = 0x00018000;
      } else if (channel == 1) {
        offset = 0x00058000;
        channel = 0;
      } else if (getTargetArch() == AIEArch::AIE4z) {
        offset = 0x00018640;
        channel -= 2;
      } else {
        llvm_unreachable("Unhandled channel and/or direction in "
                         "AIE4TargetModel::getDmaBdAddress");
      }
    }
    offset += (channel * numBds * 0x30) + (bd_id * 0x30);

  } else if (isMemTile(col, row)) {
    // AIE4 MemTile DMA BD address
    // Base address for S2MM channel 0
    offset = 0x00008000;
    // Each BD is 11 words (0x2C bytes), 16 BDs per channel
    if (direction == AIE::DMAChannelDir::MM2S) {
      offset = 0x00008C00; // MM2S base
    }
    offset += ((channel % 4) * 4 * 0x30) + (bd_id * 0x30);
    if (channel >= 4)
      offset += 0x40000;

  } else if (isCoreTile(col, row)) {
    // AIE4 Core Tile DMA BD address
    // Base address for DMA BDs
    offset = 0x000AC000;
    // Each BD is 6 words (0x18 bytes) at 0x20 alignment, 16 BDs total
    offset += bd_id * 0x20;

  } else {
    llvm_unreachable("unimplemented");
  }
  if (isCoreTile(col, row))
    row = row + (5 * getNumMemTileRows()) - 1; // account for memtile rows
  return ((col & 0xff) << getColumnShift()) | ((row & 0x1f) << getRowShift()) |
         offset;
}

uint32_t AIE4TargetModel::getDmaBdAddressOffset(int col, int row) const {
  if (isCoreTile(col, row))
    return 0x0; // 2
  if (isMemTile(col, row))
    return 0x0;
  // shim tile
  return 0x4;
}

uint32_t
AIE4TargetModel::getDmaControlAddress(int col, int row, int channel,
                                      AIE::DMAChannelDir direction) const {
  uint32_t offset = 0;
  if (isShimNOCTile(col, row)) {
    if (direction == AIE::DMAChannelDir::S2MM) {
      uint32_t o[] = {0x00018540, 0x00058540, 0x000185D0, 0x000185D8};
      offset = o[channel];
    } else {
      uint32_t o[] = {0x00018550, 0x00018558, 0x00058550, 0x00058558,
                      0x000185E0, 0x000185E8, 0x000185F0, 0x000185F8};
      offset = o[channel];
    }
  } else if (isMemTile(col, row)) {
    if (direction == AIE::DMAChannelDir::S2MM) {
      offset = 0x00009E00 + (channel * 0x8);
    } else {
      offset = 0x00009E60 + (channel * 0x8);
    }
    if (channel >= 4)
      offset += 0x40000;
  } else if (isCoreTile(col, row)) {
    if (direction == AIE::DMAChannelDir::S2MM) {
      offset = 0x000ACE00 + (channel * 0x8);
    } else {
      offset = 0x000ACE10 + (channel * 0x8);
    }
  } else {
    llvm_unreachable(
        "AIE4TargetModel::getDmaControlAddress unimplemented tile type");
  }
  if (isCoreTile(col, row))
    row = row + (5 * getNumMemTileRows()) - 1; // account for memtile rows
  return ((col & 0xff) << getColumnShift()) | ((row & 0x1f) << getRowShift()) |
         offset;
}

std::optional<uint32_t>
AIE4TargetModel::getLockLocalBaseIndex(int localCol, int localRow, int lockCol,
                                       int lockRow) const {
  if (isCoreTile(localCol, localRow)) {
    if (isMemSouth(localCol, localRow, lockCol, lockRow))
      return 0;
    if (isMemWest(localCol, localRow, lockCol, lockRow))
      return AIETargetModel::getNumLocks(localCol, localRow);
    if (isMemNorth(localCol, localRow, lockCol, lockRow))
      return AIETargetModel::getNumLocks(localCol, localRow) * 2;
    if (isMemEast(localCol, localRow, lockCol, lockRow))
      return AIETargetModel::getNumLocks(localCol, localRow) * 3;
  }

  if (isMemTile(localCol, localRow)) {
    if (localRow != lockRow)
      return std::nullopt;
    int colOffset = (lockCol - localCol) + 7;
    if (colOffset > 14 || colOffset < 0)
      return std::nullopt;
    return colOffset * AIETargetModel::getNumLocks(localCol, localRow);
  }

  return std::nullopt;
}

std::optional<uint32_t>
AIE4TargetModel::getMemLocalBaseAddress(int localCol, int localRow, int memCol,
                                        int memRow) const {
  if (isCoreTile(localCol, localRow)) {
    if (isMemSouth(localCol, localRow, memCol, memRow))
      return getMemSouthBaseAddress();
    if (isMemWest(localCol, localRow, memCol, memRow))
      return getMemWestBaseAddress();
    if (isMemNorth(localCol, localRow, memCol, memRow))
      return getMemNorthBaseAddress();
    if (isMemEast(localCol, localRow, memCol, memRow))
      return getMemEastBaseAddress();
  }

  if (isMemTile(localCol, localRow)) {
    if (localRow != memRow)
      return std::nullopt;
    int colOffset = (memCol - localCol) + 7;
    if (colOffset > 14 || colOffset < 0)
      return std::nullopt;
    uint32_t base = 0x00B00000;
    return base + (colOffset * getMemTileSize());
  }

  return std::nullopt;
}

// Manager ports
uint32_t
AIE4TargetModel::getNumDestSwitchboxConnections(int col, int row,
                                                WireBundle bundle) const {

  if (isMemTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
    case WireBundle::North:
      return 8;
    case WireBundle::South:
      return 4;
    case WireBundle::TileControl:
    case WireBundle::NorthControl:
    case WireBundle::Switch32:
      return 2;
    // 32 bit ports
    case WireBundle::South32:
    case WireBundle::North32:
    case WireBundle::Switch512:
      return 2;
    case WireBundle::West32:
    case WireBundle::East32:
    default:
      return 0;
    }

  if (isShimNOCTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      if (getTargetArch() == AIEArch::AIE4z)
        return 4;
      else
        return 2;
    case WireBundle::North:
      if (getTargetArch() == AIEArch::AIE4z)
        return 8;
      else
        return 4;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 2;
    }
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 2;
    }
    case WireBundle::TileControl:
    case WireBundle::NorthControl:
    case WireBundle::Switch32:
      return 2;
    // 32 bit ports
    case WireBundle::North32:
    case WireBundle::Switch512:
    case WireBundle::Controller32:
      return 2;
    case WireBundle::West32:
      if (col == 0)
        return 0;
      return 1;
    case WireBundle::East32:
      if (col == columns() - 1)
        return 0;
      return 1;
    case WireBundle::PLIO32:
    default:
      return 0;
    }

  switch (bundle) {
  case WireBundle::Core:
  case WireBundle::DMA:
  case WireBundle::NorthControl:
    return 2;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 8;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 2;
  }
  case WireBundle::South:
    return 4;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 2;
  }
  case WireBundle::TileControl:
  case WireBundle::Switch32:
    return 1;
  // 32 bit ports
  case WireBundle::South32:
  case WireBundle::North32:
    return 2;
  case WireBundle::West32:
  case WireBundle::East32:
  case WireBundle::Switch512:
    return 1;
  default:
    return 0;
  }
}

// Subordinate ports
uint32_t
AIE4TargetModel::getNumSourceSwitchboxConnections(int col, int row,
                                                  WireBundle bundle) const {

  if (isMemTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 10;
    case WireBundle::North:
      return 4;
    case WireBundle::South:
      if (getTargetArch() == AIEArch::AIE4z)
        return 8;
      else
        return 4;
    case WireBundle::SouthControl:
    case WireBundle::Switch32:
      return 2;
    // 32-bit ports
    case WireBundle::Control32:
    case WireBundle::Trace32:
    case WireBundle::South32:
    case WireBundle::North32:
    case WireBundle::Switch512:
      return 2;
    case WireBundle::West32:
    case WireBundle::East32:
    default:
      return 0;
    }

  if (isShimNOCTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      if (getTargetArch() == AIEArch::AIE4z)
        return 8;
      else
        return 4;
    case WireBundle::North:
      return 4;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 2;
    }
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 2;
    }
    case WireBundle::SouthControl:
    case WireBundle::Switch32:
      return 2;
    // 32-bit ports
    case WireBundle::Control32:
    case WireBundle::Trace32:
    case WireBundle::North32:
    case WireBundle::Switch512:
    case WireBundle::Controller32:
      return 2;
    case WireBundle::West32:
      if (col == 0)
        return 0;
      return 1;
    case WireBundle::East32:
      if (col == columns() - 1)
        return 0;
      return 1;
    default:
      return 0;
    }

  // compute/core tile
  switch (bundle) {
  case WireBundle::Core:
  case WireBundle::DMA:
    return 1;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 4;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 2;
  }
  case WireBundle::South:
    return 8;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 2;
  }
  case WireBundle::SouthControl:
    return 2;
  case WireBundle::Switch32:
    return 1;
  // 32-bit ports
  case WireBundle::Trace32:
  case WireBundle::South32:
  case WireBundle::North32:
    return 2;
  case WireBundle::Control32:
  case WireBundle::Switch512:
    return 1;
  case WireBundle::West32:
  case WireBundle::East32:
  default:
    return 0;
  }
}

uint32_t
AIE4TargetModel::getNumDestShimMuxConnections(int col, int row,
                                              WireBundle bundle) const {
  return getNumDestSwitchboxConnections(col, row, bundle);
}

uint32_t
AIE4TargetModel::getNumSourceShimMuxConnections(int col, int row,
                                                WireBundle bundle) const {
  return getNumSourceSwitchboxConnections(col, row, bundle);
}

// src == subordinate
// dst == manager
bool AIE4TargetModel::isLegalTileConnection(int col, int row,
                                            WireBundle srcBundle, int srcChan,
                                            WireBundle dstBundle,
                                            int dstChan) const {
  // Check Channel Id within the range
  if (srcChan >= int(getNumSourceSwitchboxConnections(col, row, srcBundle)))
    return false;
  if (dstChan >= int(getNumDestSwitchboxConnections(col, row, dstBundle)))
    return false;

  // Lambda function to check if a bundle is in a list
  auto isBundleInList = [](WireBundle bundle,
                           std::initializer_list<WireBundle> bundles) -> bool {
    return std::find(bundles.begin(), bundles.end(), bundle) != bundles.end();
  };

  auto isBundle32 = [&](WireBundle bundle) -> bool {
    return isBundleInList(
        bundle,
        {WireBundle::Control32, WireBundle::Controller32, WireBundle::Trace32,
         WireBundle::South32, WireBundle::West32, WireBundle::North32,
         WireBundle::East32, WireBundle::Switch512, WireBundle::PLIO32});
  };

  // 32-bit switch is fully connected
  if (isBundle32(srcBundle) && isBundle32(dstBundle))
    return true;

  // Memtile
  if (isMemTile(col, row)) {

    // DMA Subordinate
    if (srcBundle == WireBundle::DMA) {
      // DMA -> DMA
      if ((dstBundle == WireBundle::DMA) &&
          ((srcChan <= 3) || (srcChan >= 6 && srcChan <= 9)))
        return srcChan == dstChan;
      // DMA -> South
      if (dstBundle == WireBundle::South) {
        switch (srcChan) {
        case 0:
        case 2:
          return dstChan == 0;
        case 1:
        case 3:
          return dstChan == 1;
        case 5:
        case 7:
          return dstChan == 2;
        case 6:
        case 8:
          return dstChan == 3;
        case 4:
        case 9:
          return true;
        default:
          return false;
        }
      }
      // DMA -> North
      if (dstBundle == WireBundle::North) {
        if (srcChan == 4)
          return dstChan == 0;
        if (srcChan == 9)
          return dstChan == 4;
        return true;
      }
      // DMA -> Switch32
      if (dstBundle == WireBundle::Switch32)
        return true;
      // DMA -> NorthControl, Control
      if (isBundleInList(dstBundle,
                         {WireBundle::NorthControl, WireBundle::TileControl})) {
        if (srcChan == 3 || srcChan == 4)
          return dstChan == 0;
        if (srcChan == 9 || srcChan == 10)
          return dstChan == 1;
      }
      return false;
    }

    // South Subordinate
    if (srcBundle == WireBundle::South) {
      // South -> DMA
      if (dstBundle == WireBundle::DMA) {
        if (srcChan == 0)
          return dstChan == 0 || dstChan == 2 || dstChan == 3;
        if (srcChan == 1)
          return dstChan == 1 || dstChan == 2 || dstChan == 3;
        if (srcChan == 2)
          return dstChan == 4 || dstChan == 6 || dstChan == 7;
        if (srcChan == 3)
          return dstChan == 5 || dstChan == 6 || dstChan == 7;
      }
      // South -> South
      if (dstBundle == WireBundle::South)
        return srcChan == dstChan;
      // South -> North, Switch32
      if (isBundleInList(dstBundle, {WireBundle::North, WireBundle::Switch32}))
        return true;
      return false;
    }

    // North Subordinate
    if (srcBundle == WireBundle::North) {
      // North -> DMA
      if (dstBundle == WireBundle::DMA) {
        if (dstChan <= 1)
          return true;
        if (srcChan == 0)
          return dstChan == 2;
        if (srcChan == 1)
          return dstChan == 3;
        if (srcChan == 2)
          return dstChan == 4 || dstChan == 5 || dstChan == 6;
        if (srcChan == 3)
          return dstChan == 4 || dstChan == 5 || dstChan == 7;
      }
      // North -> North
      if (dstBundle == WireBundle::North) {
        if (srcChan == 0 && dstChan == 0)
          return true;
        if (srcChan == 1 && dstChan == 1)
          return true;
        if (srcChan == 2 && dstChan == 4)
          return true;
        if (srcChan == 3 && dstChan == 5)
          return true;
      }
      // North -> Control
      if (dstBundle == WireBundle::TileControl) {
        if (srcChan == 1 && dstChan == 0)
          return true;
        if (srcChan == 3)
          return true;
      }
      // North -> South, Switch32
      if (isBundleInList(dstBundle, {WireBundle::South, WireBundle::Switch32}))
        return true;
      return false;
    }

    // SouthControl Subordinate
    if (srcBundle == WireBundle::SouthControl) {
      // SouthControl -> DMA
      if (dstBundle == WireBundle::DMA) {
        if (srcChan == 0 && dstChan == 3)
          return true;
        if (srcChan == 1 && dstChan == 7)
          return true;
      }
      // SouthControl -> NorthControl
      if (dstBundle == WireBundle::NorthControl)
        return srcChan == dstChan;
      // SouthControl -> Control
      if (dstBundle == WireBundle::TileControl)
        return srcChan == 1 || dstChan == 0;
      // SouthControl -> Switch32
      if (dstBundle == WireBundle::Switch32)
        return true;
      return false;
    }

    // Switch32 Subordinate
    if (srcBundle == WireBundle::Switch32) {
      if (dstBundle == WireBundle::Switch32)
        return false;
      if (isBundleInList(dstBundle,
                         {WireBundle::TileControl, WireBundle::NorthControl}))
        return srcChan == dstChan;
      return true;
    }

    return false;
  }

  // Shimtile
  if (isShimNOCorPLTile(col, row)) {
    // DMA subordinate
    if (srcBundle == WireBundle::DMA) {
      if (dstBundle == WireBundle::DMA) {
        if (srcChan == 0 || srcChan == 2)
          return srcChan == dstChan;
        return false;
      }
      if (isBundleInList(dstBundle,
                         {WireBundle::NorthControl, WireBundle::TileControl})) {
        if (srcChan <= 1 && dstChan == 0)
          return true;
        if (srcChan >= 2 && dstChan == 1)
          return true;
        return false;
      }
      if (isBundleInList(dstBundle, {WireBundle::West, WireBundle::North,
                                     WireBundle::East, WireBundle::Switch32}))
        return true;
      return false;
    }
    // North, East, West subordinate
    if (isBundleInList(srcBundle, {WireBundle::West, WireBundle::North,
                                   WireBundle::East})) {
      if (srcBundle == dstBundle)
        return srcChan == dstChan;
      if (isBundleInList(dstBundle,
                         {WireBundle::DMA, WireBundle::West, WireBundle::North,
                          WireBundle::East, WireBundle::Switch32}))
        return true;
      if (dstBundle == WireBundle::TileControl) {
        if (dstChan == 1)
          return (srcBundle == WireBundle::North) && (srcChan == 3);
        if (srcBundle == WireBundle::North)
          return (srcChan == 1) || (srcChan == 3);
        return true;
      }
      if (dstBundle == WireBundle::NorthControl) {
        if (srcBundle != WireBundle::North)
          return true;
        if (srcChan == 3)
          return true;
        if (srcChan == 1 && dstChan == 0)
          return true;
        return false;
      }
      return false;
    }
    // Control mm2s subordinate
    if (srcBundle == WireBundle::TileControl) {
      if (isBundleInList(dstBundle, {WireBundle::West, WireBundle::East}))
        return (srcChan == 0) || (dstChan == 1);
      if (dstBundle == WireBundle::North) {
        if (dstChan <= 1)
          return srcChan == 0;
        if (dstChan >= 2)
          return srcChan == 1;
      }
      if (isBundleInList(dstBundle,
                         {WireBundle::NorthControl, WireBundle::TileControl,
                          WireBundle::Switch32})) {
        if (srcChan == dstChan)
          return true;
        if (dstBundle == WireBundle::TileControl)
          return dstChan == 0;
        return false;
      }
    }
    // Switch32 subordinate
    if (srcBundle == WireBundle::Switch32) {
      if (isBundleInList(dstBundle,
                         {WireBundle::DMA, WireBundle::West, WireBundle::North,
                          WireBundle::East, WireBundle::NorthControl}))
        return true;
      if (dstBundle == WireBundle::TileControl)
        return srcChan == dstChan;
    }
    // Switch512 subordinate (32-bit bridge to 512-bit switch)
    if (srcBundle == WireBundle::Switch512)
      return dstBundle == WireBundle::Switch32;
    return false;
  }

  // Coretile
  if (isCoreTile(col, row)) {
    // Core subordinate
    switch (srcBundle) {
    default:
      return false;
    case WireBundle::Core:
      if (dstBundle == WireBundle::DMA)
        return dstChan == 1;
      if (dstBundle == WireBundle::NorthControl)
        return dstChan == 0;
      if (isBundleInList(dstBundle,
                         {WireBundle::South, WireBundle::East, WireBundle::West,
                          WireBundle::North, WireBundle::TileControl,
                          WireBundle::Switch32}))
        return true;
      return false;
    // DMA subordinate
    case WireBundle::DMA:
      if (dstBundle == WireBundle::DMA)
        return dstChan == 0;
      if (isBundleInList(dstBundle,
                         {WireBundle::Core, WireBundle::South, WireBundle::East,
                          WireBundle::West, WireBundle::North,
                          WireBundle::NorthControl, WireBundle::TileControl,
                          WireBundle::Switch32}))
        return true;
      return false;
    // North, South subordinate
    case WireBundle::North:
    case WireBundle::South:
      if (srcBundle == dstBundle) {
        if (srcChan == 0 || srcChan == 1)
          return srcChan == dstChan;
        if (srcChan == 4 || srcChan == 5)
          return (srcChan - 2) == dstChan;
        return false;
      }
      if (isBundleInList(dstBundle,
                         {WireBundle::Core, WireBundle::DMA, WireBundle::West,
                          WireBundle::North, WireBundle::South,
                          WireBundle::East, WireBundle::Switch32}))
        return true;
      return false;
    // East, West subordinate
    case WireBundle::East:
    case WireBundle::West:
      if (dstBundle == srcBundle)
        return srcChan == dstChan;
      if (isBundleInList(dstBundle,
                         {WireBundle::Core, WireBundle::DMA, WireBundle::South,
                          WireBundle::East, WireBundle::West, WireBundle::North,
                          WireBundle::NorthControl, WireBundle::TileControl,
                          WireBundle::Switch32}))
        return true;
      return false;
    // SouthControl subordinate
    case WireBundle::SouthControl:
      if (isBundleInList(dstBundle, {WireBundle::Core, WireBundle::DMA}))
        return srcChan == dstChan;
      if (isBundleInList(dstBundle, {WireBundle::East, WireBundle::West}))
        return dstChan == 1 || srcChan == 0;
      if (dstBundle == WireBundle::NorthControl)
        return srcChan == 1 || dstChan == 0;
      if (isBundleInList(dstBundle,
                         {WireBundle::TileControl, WireBundle::Switch32}))
        return true;
      return false;
    // Switch512 subordinate (port on 32-bit switch to main switch)
    case (WireBundle::Switch512):
      return dstBundle == WireBundle::Switch32;
    // Switch32 subordinate (port on main switch; can route back to 32-bit
    // switch)
    case (WireBundle::Switch32):
      return (isBundleInList(
          dstBundle, {WireBundle::Core, WireBundle::DMA, WireBundle::South,
                      WireBundle::East, WireBundle::West, WireBundle::North,
                      WireBundle::NorthControl, WireBundle::TileControl,
                      WireBundle::Switch512}));
    }
  }
  return false;
}

bool BaseNPU2TargetModel::isSupportedBlockFormat(
    std::string const &format) const {
  std::set<std::string> supportedTypes = {"v8bfp16ebs8", "v16bfp16ebs16"};
  return static_cast<bool>(supportedTypes.find(format) != supportedTypes.end());
}

// AIE2PS overrides: shim DMA registers are at different offsets than AIE2.
// Memtile and core tile DMA addresses are identical to AIE2.
uint64_t
AIE2PSTargetModel::getDmaBdAddress(int col, int row, uint32_t bd_id,
                                   int channel,
                                   AIE::DMAChannelDir direction) const {
  if (isShimNOCTile(col, row)) {
    uint64_t offset = 0x00009000 + bd_id * 0x30;
    return ((col & 0xff) << getColumnShift()) |
           ((row & 0xff) << getRowShift()) | offset;
  }
  // Memtile and core tile use same addresses as AIE2.
  return AIE2TargetModel::getDmaBdAddress(col, row, bd_id, channel, direction);
}

uint32_t
AIE2PSTargetModel::getDmaControlAddress(int col, int row, int channel,
                                        AIE::DMAChannelDir direction) const {
  if (isShimNOCTile(col, row)) {
    uint32_t offset = 0x00009300 + (channel * 0x8);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x10;
    return ((col & 0xff) << getColumnShift()) |
           ((row & 0xff) << getRowShift()) | offset;
  }
  // Memtile and core tile use same addresses as AIE2.
  return AIE2TargetModel::getDmaControlAddress(col, row, channel, direction);
}

std::optional<uint32_t>
AIE2PSTargetModel::getDmaStatusAddress(int col, int row, int channel,
                                       AIE::DMAChannelDir direction) const {
  // The AIE2PS shim relocates its DMA registers (control is at 0x9300, not
  // 0x1D200), and the status block has not been confirmed against a register
  // spec. Report "unknown" rather than guess an address a maskpoll would then
  // read garbage from; memtile and core tiles match AIE2.
  if (isShimNOCTile(col, row))
    return std::nullopt;
  return AIE2TargetModel::getDmaStatusAddress(col, row, channel, direction);
}

// AIE2PS shim tiles have uC stream switch port for routing TCT to CERT uC.
uint32_t
AIE2PSTargetModel::getNumDestSwitchboxConnections(int col, int row,
                                                  WireBundle bundle) const {
  if (isShimNOCTile(col, row) && bundle == WireBundle::Controller32)
    return 1;
  return AIE2TargetModel::getNumDestSwitchboxConnections(col, row, bundle);
}

uint32_t
AIE2PSTargetModel::getNumSourceSwitchboxConnections(int col, int row,
                                                    WireBundle bundle) const {
  if (isShimNOCTile(col, row) && bundle == WireBundle::Controller32)
    return 1;
  return AIE2TargetModel::getNumSourceSwitchboxConnections(col, row, bundle);
}

std::optional<uint32_t> AIE2PSTargetModel::getStreamSwitchPortIndex(
    int col, int row, WireBundle bundle, uint32_t channel, bool master) const {
  // AIE2PS shim: Controller32 maps to hw master port 22 / slave port 23.
  if (isShimNOCTile(col, row) && bundle == WireBundle::Controller32) {
    if (channel > 0)
      return std::nullopt;
    return master ? 22 : 23;
  }
  return AIE2TargetModel::getStreamSwitchPortIndex(col, row, bundle, channel,
                                                   master);
}

std::vector<std::pair<uint32_t, uint32_t>>
AIE4TargetModel::getShimBurstEncodingsAndLengths() const {
  return {std::pair(0, 64), std::pair(1, 128), std::pair(2, 256),
          std::pair(3, 512)};
}

std::optional<uint32_t>
AIE4TargetModel::getLocalLockAddress(uint32_t lockId, TileID tile) const {
  uint32_t lockAddrOffset = 0x10;

  if (lockId >= AIETargetModel::getNumLocks(tile.col, tile.row))
    return std::nullopt;

  if (isCoreTile(tile.col, tile.row)) {
    // Compute-tile locks live in the tile's memory module at 0xAE000.
    uint32_t computeTileBaseAddress = 0x000AE000;
    return computeTileBaseAddress + lockAddrOffset * lockId;
  }

  if (isMemTile(tile.col, tile.row)) {
    // Memtile has two 32-lock banks (Lock_Request_A/B) at 0x10000 / 0x50000;
    // upper-bank offsets restart at 0.
    if (lockId < 32)
      return 0x00010000 + lockAddrOffset * lockId;
    return 0x00050000 + lockAddrOffset * (lockId - 32);
  }

  if (isShimNOCTile(tile.col, tile.row)) {
    // Shim has two 16-lock banks (Lock_Request_A/B) at 0x10000 / 0x50000;
    // upper-bank offsets restart at 0.
    if (lockId < 16)
      return 0x00010000 + lockAddrOffset * lockId;
    return 0x00050000 + lockAddrOffset * (lockId - 16);
  }

  return std::nullopt;
}

std::optional<uint32_t> BaseNPU3TargetModel::getStreamSwitchPortIndex(
    int col, int row, WireBundle bundle, uint32_t channel, bool master) const {
  // TODO: Implement AIE4 stream switch port index mapping.
  return std::nullopt;
}

} // namespace AIE
} // namespace xilinx
