//===- AIEAssignBufferDescriptorIDs.h ---------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H
#define AIE_ASSIGN_BUFFER_DESCRIPTOR_IDS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <set>
#include <utility>

using namespace xilinx::AIE;

// Hands out one tile's BD ids, tracked per BD pool so private pools can reuse
// ids.
struct BdIdGenerator {
  const int col;
  const int row;
  const AIETargetModel &targetModel;
  // (pool, id) pairs in use.
  std::set<std::pair<uint32_t, uint32_t>> alreadyAssigned;
  // Held by static DMAs; nextBdId skips them and freeBdId leaves them alone.
  std::set<std::pair<uint32_t, uint32_t>> reserved;

  BdIdGenerator(int col, int row, const AIETargetModel &targetModel);

  // The lowest free id in the channel's pool that is a multiple of
  // `alignment`.
  std::optional<uint32_t> nextBdId(int channelIndex, DMAChannelDir dir,
                                   uint32_t alignment = 1);

  // Act on the channel's pool; an id outside it is treated as tile-wide.
  void assignBdId(int channelIndex, DMAChannelDir dir, uint32_t bdId);
  bool bdIdAlreadyAssigned(int channelIndex, DMAChannelDir dir, uint32_t bdId);
  void freeBdId(int channelIndex, DMAChannelDir dir, uint32_t bdId);

  // Act on every pool holding `bdId`, for when the channel is unknown.
  void assignBdId(uint32_t bdId);
  bool bdIdAlreadyAssigned(uint32_t bdId);

  // Per-channel and tile-wide forms, as above.
  void reserveBdId(int channelIndex, DMAChannelDir dir, uint32_t bdId);
  void reserveBdId(uint32_t bdId);
  bool isReserved(int channelIndex, DMAChannelDir dir, uint32_t bdId) const;
  bool isReserved(uint32_t bdId) const;

  // Whether the per-channel form acts on every pool holding `bdId`.
  bool holdsTileWide(int channelIndex, DMAChannelDir dir, uint32_t bdId) const;

private:
  // The pools that hold `bdId`; a sentinel pool for an id in none of them.
  llvm::SmallVector<uint32_t, 2> poolsHolding(uint32_t bdId) const;
  std::optional<uint32_t> poolOf(int channelIndex, DMAChannelDir dir,
                                 uint32_t bdId) const;
};

// BD ids a runtime sequence pins, recorded on each aie.tile:
// `aiex.reserved_bd_ids` (every pool holding them) and
// `aiex.reserved_channel_bd_ids` (`<dir>_<channel>` -> ids in that pool).
namespace xilinx::AIE {

struct ReservedBdId {
  // None means every pool holding the id.
  std::optional<std::pair<int, DMAChannelDir>> channel;
  int32_t id;
};

llvm::SmallVector<ReservedBdId> getReservedBdIds(TileOp tile);

void setReservedBdIds(TileOp tile, llvm::ArrayRef<ReservedBdId> ids);

bool isReservedBdIdsAttr(llvm::StringRef name);

} // namespace xilinx::AIE

#endif
