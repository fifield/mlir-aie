//===- AIECertVerify.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECERTVERIFY
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "cert-verify"

namespace {

// The firmware indexes job contexts by job_id % 64 and reuses them on the next
// page, so job IDs on one page must differ modulo 64.
static LogicalResult checkJobSlots(AIE::DeviceOp device) {
  LogicalResult result = success();
  device.walk([&](AIEX::CertPageOp page) {
    llvm::DenseMap<unsigned, AIEX::CertJobOp> slots;
    for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
      unsigned slot = job.getJobId() % AIEX::CertJobOp::NumSlots;
      auto [it, inserted] = slots.try_emplace(slot, job);
      if (!inserted) {
        job.emitError() << "job ID " << job.getJobId()
                        << " collides with job ID " << it->second.getJobId()
                        << " modulo " << AIEX::CertJobOp::NumSlots
                        << " on the same page";
        result = failure();
      }
    }
  });
  return result;
}

// The uC (CERT group) id of a cert op is the group_id of its nearest enclosing
// cert.attach_to_group, or 0 if it is at device level.
static int getUcGroupId(Operation *op) {
  auto group = op->getParentOfType<AIEX::CertAttachToGroupOp>();
  return group ? group.getGroupID() : 0;
}

// G-localbar: within each uC scope, all cert.local_barrier ops that share a
// local_barrier_id must reside in the same cert.page (their participants must
// be co-resident on one page, else the firmware hangs).
static LogicalResult checkLocalBarrierColocation(AIE::DeviceOp device) {
  LogicalResult result = success();
  // Before page formation, a standalone job is its own co-location scope.
  std::map<std::pair<int, int>, Operation *> firstScope;
  device.walk(
      [&](AIEX::CertLocalBarrierOp barrier) {
        int uc = getUcGroupId(barrier);
        int id = static_cast<int>(barrier.getLocalBarrierId());
        Operation *scope = barrier->getParentOfType<AIEX::CertPageOp>();
        if (!scope)
          scope = barrier->getParentOfType<AIEX::CertJobOp>();
        auto key = std::make_pair(uc, id);
        auto it = firstScope.find(key);
        if (it == firstScope.end()) {
          firstScope[key] = scope;
          return;
        }
        if (it->second != scope) {
          barrier.emitError()
              << "cert.local_barrier "
              << stringifyCertLocalBarrierId(barrier.getLocalBarrierId())
              << " in uC " << uc
              << " must be on the same page as its other participants";
          result = failure();
        }
      });
  return result;
}

// Firmware keeps one outstanding TCT count per actor. Concurrent jobs waiting
// on the same (tile, channel) would over-wait; sequential waits within a job or
// on different pages are safe. Bare jobs use their own scope until
// legalization.
static LogicalResult checkWaitTctsUniqueness(AIE::DeviceOp device) {
  LogicalResult result = success();
  // (page scope, tile, channel) -> the job that already waits on that actor.
  std::map<std::tuple<Operation *, int, int>, Operation *> seen;
  device.walk([&](AIEX::CertWaitTCTSOp wait) {
    int tile = wait.getTileId();
    int channel = wait.getChannelId();
    Operation *job = wait->getParentOfType<AIEX::CertJobOp>();
    Operation *scope = wait->getParentOfType<AIEX::CertPageOp>();
    if (!scope)
      scope = job;
    auto key = std::make_tuple(scope, tile, channel);
    auto it = seen.find(key);
    if (it == seen.end()) {
      seen[key] = job;
      return;
    }
    // Same job: sequential within one thread of control, so legal.
    if (it->second == job)
      return;
    wait.emitError() << "more than one job on this cert.page waits on tile "
                     << tile << " channel " << channel << " in uC "
                     << getUcGroupId(wait)
                     << "; at most one job per page may wait_tcts on a given "
                        "(tile, channel)";
    result = failure();
  });
  return result;
}

static std::set<int> computePresentUcs(AIE::DeviceOp device) {
  std::set<int> present;
  device.walk([&](AIEX::CertAttachToGroupOp group) {
    present.insert(group.getGroupID());
  });
  if (!device.getBody()->getOps<AIEX::CertPageOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertJobOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertSectionOp>().empty())
    present.insert(0);
  return present;
}

// Firmware requires each controller to expose the same preemption IDs,
// including multiplicity. Compare against the lowest present controller ID.
static LogicalResult checkPreemptConsistency(AIE::DeviceOp device) {
  std::map<int, std::multiset<int>> preemptIds;
  std::map<int, AIEX::CertAttachToGroupOp> repAttach;
  std::set<int> present;

  // Every attach_to_group establishes a present uC.
  device.walk([&](AIEX::CertAttachToGroupOp group) {
    int uc = group.getGroupID();
    present.insert(uc);
    if (!repAttach.count(uc))
      repAttach[uc] = group;
  });

  // Device-level cert content lives on uC 0.
  if (!device.getBody()->getOps<AIEX::CertPageOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertJobOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertSectionOp>().empty())
    present.insert(0);

  // Collect the preempt ids per uC.
  device.walk([&](AIEX::CertPreemptOp preempt) {
    int uc = getUcGroupId(preempt);
    present.insert(uc);
    preemptIds[uc].insert(preempt.getId());
  });

  if (present.size() < 2)
    return success();

  int ref = *present.begin();
  const std::multiset<int> &refSet = preemptIds[ref];
  LogicalResult result = success();
  for (int uc : present) {
    if (uc == ref)
      continue;
    if (preemptIds[uc] != refSet) {
      Operation *loc = repAttach.count(uc) ? repAttach[uc].getOperation()
                                           : device.getOperation();
      loc->emitError()
          << "cert.preempt ids on uC " << uc << " differ from uC " << ref
          << "; all uCs must have the same preempt ids (G-preempt)";
      result = failure();
    }
  }
  return result;
}

// Group/placement ids must be valid microcontroller indices for the device:
// nonnegative, less than the number of uCs, and (defensively) < 32 so they are
// representable in a party mask. Rejecting invalid ids up front keeps later
// checks (e.g. the remote-barrier mask shift) free of undefined behavior.
static LogicalResult checkGroupIds(AIE::DeviceOp device) {
  const AIE::AIETargetModel &tm = device.getTargetModel();
  int64_t numUcs = std::min<int64_t>(tm.getNumControllers(), 32);
  LogicalResult result = success();
  auto checkId = [&](Operation *op, int64_t id, llvm::StringRef what) {
    if (id < 0 || id >= numUcs) {
      op->emitError() << what << " " << id << " is not a valid uC id [0, "
                      << numUcs << ")";
      result = failure();
    }
  };
  device.walk([&](AIEX::CertAttachToGroupOp g) {
    checkId(g, g.getGroupID(), "cert.attach_to_group group id");
  });
  // The verifier may run before placement lowering; check any leftover attrs.
  device.walk([&](AIEX::CertPageOp p) {
    if (auto pl = p.getPlacementAttr())
      checkId(p, pl.getInt(), "cert.page placement");
  });
  return result;
}

// G-uC / rendezvous: cross-uC remote barriers must actually rendezvous. Group
// occurrences by remote_barrier_id and require: all occurrences agree on
// party_mask; every mask bit names a present uC that has a matching occurrence
// (so nobody waits forever); each participating uC sets its own bit; and at
// most one occurrence per uC per id (one participant per column). Assumes group
// ids are already validated by checkGroupIds (shifts are still guarded).
static LogicalResult checkRemoteBarrierParties(AIE::DeviceOp device) {
  std::set<int> present = computePresentUcs(device);
  LogicalResult result = success();

  std::map<AIEX::CertRemoteBarrierId, SmallVector<AIEX::CertRemoteBarrierOp, 4>>
      byId;
  device.walk([&](AIEX::CertRemoteBarrierOp b) {
    byId[b.getRemoteBarrierId()].push_back(b);
  });

  for (auto &entry : byId) {
    StringRef id = stringifyCertRemoteBarrierId(entry.first);
    SmallVector<AIEX::CertRemoteBarrierOp, 4> &occs = entry.second;

    // All occurrences of an id must describe the same rendezvous.
    uint32_t mask = occs.front().getPartyMask();
    bool masksAgree = llvm::all_of(occs, [&](AIEX::CertRemoteBarrierOp b) {
      return b.getPartyMask() == mask;
    });
    if (!masksAgree) {
      for (auto b : occs)
        b.emitError() << "cert.remote_barrier " << id
                      << " occurrences disagree on party_mask";
      result = failure();
      continue;
    }

    // Participating uCs (from occurrences); own-bit + one-per-uC checks.
    std::map<int, int> occPerUc;
    for (auto b : occs) {
      int own = getUcGroupId(b);
      occPerUc[own]++;
      if (own >= 0 && own < 32 && !((mask >> own) & 1u)) {
        b.emitError() << "cert.remote_barrier " << id << " on uC " << own
                      << " excludes its own uC from party_mask";
        result = failure();
      }
    }
    for (auto &[uc, count] : occPerUc)
      if (count > 1) {
        occs.front().emitError()
            << "cert.remote_barrier " << id << " has " << count
            << " occurrences on uC " << uc
            << " (at most one participant per uC per rendezvous)";
        result = failure();
      }

    // Every mask bit must be a present uC that has a matching barrier.
    for (int bit = 0; bit < 32; ++bit) {
      if (!((mask >> bit) & 1u))
        continue;
      if (!present.count(bit)) {
        occs.front().emitError()
            << "cert.remote_barrier " << id << " party_mask references uC "
            << bit << " which is not present in the design";
        result = failure();
      } else if (!occPerUc.count(bit)) {
        occs.front().emitError()
            << "cert.remote_barrier " << id << " party_mask includes uC " << bit
            << " but that uC has no matching remote_barrier(" << id
            << ") to rendezvous with";
        result = failure();
      }
    }
  }
  return result;
}

// The register `op` writes, if any. cert.write32_d only reads registers.
static std::optional<uint32_t> registerWrittenBy(Operation &op) {
  if (auto o = dyn_cast<AIEX::CertUcDmaWriteDesOp>(op))
    return o.getWaitHandle();
  if (auto o = dyn_cast<AIEX::CertRead32Op>(op))
    return o.getValue();
  if (auto o = dyn_cast<AIEX::CertRead32DOp>(op))
    return o.getValue();
  return std::nullopt;
}

// A cert.wait_uc_dma handle must be a local register r0..r7 whose last write
// earlier in the same job is a cert.uc_dma_write_des. A read into that register
// in between replaces the handle.
static LogicalResult checkUcDmaWaitHandles(AIE::DeviceOp device) {
  LogicalResult result = success();
  device.walk([&](AIEX::CertJobOp job) {
    llvm::SmallDenseMap<uint32_t, Operation *> lastWrite;
    for (Operation &o : job.getBody().front()) {
      if (auto wait = dyn_cast<AIEX::CertWaitUcDmaOp>(o)) {
        uint32_t r = wait.getWaitHandle();
        auto it = lastWrite.find(r);
        if (r >= 8) {
          wait.emitError() << "cert.wait_uc_dma handle r" << r
                           << " is a shared register; the handle must be a "
                              "local register r0..r7 written by a "
                              "cert.uc_dma_write_des in the same job";
          result = failure();
        } else if (it == lastWrite.end()) {
          wait.emitError() << "cert.wait_uc_dma waits on r" << r
                           << ", which no earlier cert.uc_dma_write_des in "
                              "this job writes";
          result = failure();
        } else if (!isa<AIEX::CertUcDmaWriteDesOp>(it->second)) {
          wait.emitError() << "cert.wait_uc_dma waits on r" << r
                           << ", but its last write is a '"
                           << it->second->getName().getStringRef()
                           << "', which replaced the handle";
          result = failure();
        }
      }
      if (auto r = registerWrittenBy(o))
        lastWrite[*r] = &o;
    }
  });
  return result;
}

// Preemption rule 3: on restore, LOAD_LAST_PDI replays the page index the
// last LOAD_PDI on the uC recorded. cert never validates or clears that index,
// so a cert.preempt with no earlier load on its uC would replay page 0 of the
// command, or a stale page of another ELF. A configure job is not a load.
// Program order per uC: device-level control code is uC 0, each
// cert.attach_to_group(k) is uC k; section bodies run where they are loaded
// and are not part of it.
static LogicalResult checkPreemptAfterLoad(AIE::DeviceOp device) {
  std::set<int> loaded;
  LogicalResult result = success();
  device->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (isa<AIEX::CertSectionOp>(op))
      return WalkResult::skip();
    if (isa<AIEX::CertLoadPdiOp>(op))
      loaded.insert(getUcGroupId(op));
    if (auto preempt = dyn_cast<AIEX::CertPreemptOp>(op)) {
      int uc = getUcGroupId(op);
      if (!loaded.count(uc)) {
        preempt.emitError()
            << "cert.preempt needs an earlier cert.load_pdi on uC " << uc
            << ": on restore LOAD_LAST_PDI replays the page index of the last "
               "load, which is unset or stale without one";
        result = failure();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// The tile-relative offset of a memtile's data memory: AIE4 puts it after a
// 1 MiB register space (aie-rt XAIE4GBL_MEM_TILE_MODULE_DATAMEMORY), AIE2
// at the start of the tile. On AIE4 a memtile spans five 1 MiB address rows:
// registers, then the 3 MiB data window (memtile (0, 1): 0x200000 up to the
// last word at 0x4F_FFFC), then a 1 MiB hole. Writes into the hole are
// invalid anyway and are not reported here.
static uint32_t memTileDataOffset(const AIE::AIETargetModel &tm) {
  return isa<AIE::AIE4TargetModel>(tm) ? 0x00100000 : 0;
}

// Preemption rule 1: restore brings memtile data back from DDR and then
// replays the whole last-loaded section. A section that writes memtile data
// memory (a buffer initializer) would overwrite the restored data. Checks
// every section a cert.load_pdi names, in a device with a cert.preempt:
// cert.write32/maskwrite32, cert.write32_d, and the BDs of the uC-DMA chains
// it writes. A cert.write32_d with address_is_reg takes its address from a
// register at run time (cert dpu_control.c resolves it), so it cannot be
// shown to stay out of memtile data memory and is rejected too.
static LogicalResult checkNoMemtileDataInLoadedSections(AIE::DeviceOp device) {
  if (!device.walk([](AIEX::CertPreemptOp) { return WalkResult::interrupt(); })
           .wasInterrupted())
    return success();
  const AIE::AIETargetModel &tm = device.getTargetModel();
  uint32_t dataOffset = memTileDataOffset(tm);
  uint32_t dataSize = tm.getMemTileSize();
  // The offset into memtile data memory `[addr, addr + bytes)` writes, if it
  // overlaps it.
  auto memtileDataOffset = [&](uint64_t addr,
                               uint64_t bytes) -> std::optional<uint64_t> {
    if (addr > UINT32_MAX || dataSize == 0)
      return std::nullopt;
    int col = static_cast<int>(addr >> tm.getColumnShift());
    int row = static_cast<int>(tm.getRowFromAddress(addr));
    if (col >= tm.columns() || row >= tm.rows() || !tm.isMemTile(col, row))
      return std::nullopt;
    uint64_t off = addr - tm.getTileAddress(col, row);
    if (off + bytes <= dataOffset || off >= uint64_t(dataOffset) + dataSize)
      return std::nullopt;
    return off;
  };

  llvm::SmallPtrSet<Operation *, 4> loadedSections;
  device.walk([&](AIEX::CertLoadPdiOp load) {
    if (Operation *section =
            SymbolTable::lookupSymbolIn(device, load.getSymbolAttr()))
      loadedSections.insert(section);
  });

  LogicalResult result = success();
  auto report = [&](AIEX::CertSectionOp section, Operation *op, uint64_t addr) {
    op->emitError() << "section @" << section.getSymName()
                    << " writes memtile data memory at 0x"
                    << llvm::utohexstr(addr)
                    << "; a design with cert.preempt restores memtile data "
                       "before replaying this section, which would overwrite "
                       "it";
    result = failure();
  };
  auto reportRegisterAddress = [&](AIEX::CertSectionOp section,
                                   AIEX::CertWrite32DOp writeD) {
    writeD.emitError() << "section @" << section.getSymName()
                       << " has a cert.write32_d whose address is in register "
                          "$r"
                       << writeD.getAddress()
                       << ", which may be in memtile data memory; a design "
                          "with cert.preempt restores memtile data before "
                          "replaying this section, which would overwrite it";
    result = failure();
  };
  for (Operation *sectionOp : loadedSections) {
    auto section = cast<AIEX::CertSectionOp>(sectionOp);
    // A register address is known only at run time, so it may be anywhere.
    section.walk([&](AIEX::CertWrite32DOp writeD) {
      if (writeD.getAddressIsReg())
        reportRegisterAddress(section, writeD);
    });
    section.walk([&](Operation *op) {
      std::optional<uint64_t> addr;
      if (auto write = dyn_cast<AIEX::CertWrite32Op>(op))
        addr = write.getAddress();
      else if (auto mask = dyn_cast<AIEX::CertMaskWrite32Op>(op))
        addr = mask.getAddress();
      else if (auto writeD = dyn_cast<AIEX::CertWrite32DOp>(op)) {
        if (!writeD.getAddressIsReg())
          addr = writeD.getAddress();
      }
      if (addr && memtileDataOffset(*addr, 4)) {
        report(section, op, *addr);
        return;
      }
      FlatSymbolRefAttr chainRef;
      if (auto des = dyn_cast<AIEX::CertUcDmaWriteDesOp>(op))
        chainRef = des.getSymbolAttr();
      else if (auto sync = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(op))
        chainRef = sync.getSymbolAttr();
      if (!chainRef)
        return;
      auto chain = dyn_cast_or_null<AIEX::CertUcDmaChainOp>(
          SymbolTable::lookupSymbolIn(device, chainRef));
      if (!chain)
        return;
      chain.walk([&](AIEX::CertUcDmaBdOp bd) {
        uint64_t local = static_cast<uint32_t>(bd.getLocalAddress());
        uint64_t bytes = 4 * static_cast<uint64_t>(bd.getLength());
        if (memtileDataOffset(local, bytes))
          report(section, op, local);
      });
    });
  }
  return result;
}

struct AIECertVerifyPass
    : xilinx::AIEX::impl::AIECertVerifyBase<AIECertVerifyPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    if (failed(checkPreemptAfterLoad(device)))
      signalPassFailure();
    if (failed(checkNoMemtileDataInLoadedSections(device)))
      signalPassFailure();
    if (failed(checkJobSlots(device)))
      signalPassFailure();
    if (failed(checkGroupIds(device)))
      signalPassFailure();
    if (failed(checkLocalBarrierColocation(device)))
      signalPassFailure();
    if (failed(checkWaitTctsUniqueness(device)))
      signalPassFailure();
    if (failed(checkPreemptConsistency(device)))
      signalPassFailure();
    if (failed(checkRemoteBarrierParties(device)))
      signalPassFailure();
    if (failed(checkUcDmaWaitHandles(device)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIECertVerifyPass() {
  return std::make_unique<AIECertVerifyPass>();
}
