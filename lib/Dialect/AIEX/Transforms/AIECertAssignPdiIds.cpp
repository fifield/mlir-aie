//===- AIECertAssignPdiIds.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Derives cert.load_pdi ids from section contents so the firmware's
// last-loaded-id cache works across separately compiled ELFs.
// See AIECertAssignPdiIds in AIEXPasses.td for the canonical hash contract and
// the scope of collision checking.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECERTASSIGNPDIIDS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;

namespace {

// Bumped whenever the canonical encoding changes to separate hash domains.
constexpr llvm::StringLiteral kDomainTag = "mlir-aie.cert.pdi.v3";

// Record tags that delimit the structural walk. Every variable-length item is
// length-prefixed, so the stream is unambiguous.
enum Tag : uint8_t {
  TagOp = 0x10,
  TagOpEnd,
  TagRegion,
  TagRegionEnd,
  TagBlock,
  TagBlockEnd,
  TagAttr,
  TagIntAttr,
  TagStrAttr,
  TagUnitAttr,
  TagArrayAttr,
  TagDictAttr,
  TagDenseIntAttr,
  TagTextAttr,
  TagSection,
  TagChain,
  TagGlobal,
  TagUnresolvedSym,
};

using Digest = llvm::XXH128_hash_t;

// What a PDI id is derived from: the digest of the section's content, and the
// id, which hashes that digest under the domain tag.
struct SectionDigests {
  Digest content;
  Digest id;
};

class ByteStream {
public:
  void u8(uint8_t v) { bytes.push_back(v); }
  void u32(uint32_t v) {
    for (int i = 0; i < 4; ++i)
      bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void u64(uint64_t v) {
    for (int i = 0; i < 8; ++i)
      bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void str(StringRef s) {
    u32(static_cast<uint32_t>(s.size()));
    bytes.append(s.begin(), s.end());
  }
  void digest(const Digest &d) {
    u64(d.low64);
    u64(d.high64);
  }
  Digest hash() const { return llvm::xxh3_128bits(bytes.data(), bytes.size()); }
  llvm::SmallVector<uint8_t, 256> bytes;
};

class PdiHasher {
public:
  explicit PdiHasher(AIE::DeviceOp device)
      : printState(device->getContext(), canonicalPrintingFlags()) {
    // Section, chain and data symbols are resolved against every definition in
    // the device. The map is only ever looked up, never iterated.
    device.walk([&](Operation *op) {
      if (!isa<AIEX::CertSectionOp, AIEX::CertUcDmaChainOp, memref::GlobalOp>(
              op))
        return;
      if (auto name =
              op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        symbols.try_emplace(name.getValue(), op);
    });
  }

  // Digests of a section, memoized. Fails (after emitting a diagnostic) on a
  // reference cycle or an op with SSA operands.
  FailureOr<SectionDigests> sectionDigests(AIEX::CertSectionOp section) {
    Operation *key = section.getOperation();
    if (auto it = sectionMemo.find(key); it != sectionMemo.end())
      return it->second;
    if (!inProgress.insert(key).second) {
      section.emitError("cert.section @")
          << section.getSymName()
          << " references itself (through load_pdi/preempt); its PDI id "
             "cannot be derived from its content";
      return failure();
    }

    // The content: the section's attributes other than its label, and its
    // body.
    ByteStream content;
    content.u8(TagSection);
    bool ok = succeeded(hashAttrs(section, content)) &&
              succeeded(hashRegions(section, content));
    inProgress.erase(key);
    if (!ok)
      return failure();

    // The id: the content alone. Sections with equal content configure the
    // array identically, so sharing an id (and the firmware's skip) is right
    // whichever device they came from.
    SectionDigests d;
    d.content = content.hash();
    ByteStream id;
    id.str(kDomainTag);
    id.u8(TagSection);
    id.digest(d.content);
    d.id = id.hash();
    sectionMemo[key] = d;
    return d;
  }

private:
  // Printing flags that do not depend on global command-line options (e.g.
  // --mlir-elide-elementsattrs-if-larger), so the text fallback in hashAttr
  // always prints an attribute in full.
  static OpPrintingFlags canonicalPrintingFlags() {
    OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(std::numeric_limits<int64_t>::max());
    flags.printLargeElementsAttrWithHex(-1);
    flags.elideLargeResourceString(std::numeric_limits<int64_t>::max());
    return flags;
  }

  LogicalResult hashRegions(Operation *op, ByteStream &s) {
    s.u32(op->getNumRegions());
    for (Region &region : op->getRegions()) {
      s.u8(TagRegion);
      s.u32(static_cast<uint32_t>(region.getBlocks().size()));
      for (Block &block : region) {
        s.u8(TagBlock);
        s.u32(block.getNumArguments());
        for (Operation &inner : block)
          if (failed(hashOp(&inner, s)))
            return failure();
        s.u8(TagBlockEnd);
      }
      s.u8(TagRegionEnd);
    }
    return success();
  }

  LogicalResult hashOp(Operation *op, ByteStream &s) {
    if (op->getNumOperands() != 0)
      return op->emitError("cannot derive a cert PDI id: '")
             << op->getName() << "' has SSA operands";
    s.u8(TagOp);
    s.str(op->getName().getStringRef());
    s.u32(op->getNumResults());
    if (failed(hashAttrs(op, s)) || failed(hashRegions(op, s)))
      return failure();
    s.u8(TagOpEnd);
    return success();
  }

  // Attributes in DictionaryAttr (name-sorted) order.
  LogicalResult hashAttrs(Operation *op, ByteStream &s) {
    for (NamedAttribute named : op->getAttrDictionary()) {
      StringRef name = named.getName().getValue();
      // Compilation-chosen names of chains, data and sections are not hashed
      // (they are identified by content).
      if (name == SymbolTable::getSymbolAttrName())
        continue;
      // A nested load_pdi's id is derived from the section it names, which
      // the resolved symbol already covers.
      if (isa<AIEX::CertLoadPdiOp>(op) && name == "pdi_id")
        continue;
      s.u8(TagAttr);
      s.str(name);
      if (failed(hashAttr(op, named.getValue(), s)))
        return failure();
    }
    return success();
  }

  LogicalResult hashAttr(Operation *op, Attribute attr, ByteStream &s) {
    if (auto sym = dyn_cast<FlatSymbolRefAttr>(attr))
      return hashSymbol(op, sym.getValue(), s);
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      const APInt &v = intAttr.getValue();
      s.u8(TagIntAttr);
      s.u32(v.getBitWidth());
      for (unsigned i = 0, e = v.getNumWords(); i < e; ++i)
        s.u64(v.getRawData()[i]);
      return success();
    }
    if (auto str = dyn_cast<StringAttr>(attr)) {
      s.u8(TagStrAttr);
      s.str(str.getValue());
      return success();
    }
    if (isa<UnitAttr>(attr)) {
      s.u8(TagUnitAttr);
      return success();
    }
    if (auto array = dyn_cast<ArrayAttr>(attr)) {
      s.u8(TagArrayAttr);
      s.u32(array.size());
      for (Attribute elt : array)
        if (failed(hashAttr(op, elt, s)))
          return failure();
      return success();
    }
    if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
      s.u8(TagDictAttr);
      s.u32(dict.size());
      for (NamedAttribute named : dict) {
        s.str(named.getName().getValue());
        if (failed(hashAttr(op, named.getValue(), s)))
          return failure();
      }
      return success();
    }
    // Anything else (enums, types, ...): its printed form, which is stable and
    // contains no addresses. printState's flags never elide.
    std::string text;
    llvm::raw_string_ostream os(text);
    attr.print(os, printState);
    s.u8(TagTextAttr);
    s.str(text);
    return success();
  }

  // Hash what `name` defines rather than the name itself.
  LogicalResult hashSymbol(Operation *user, StringRef name, ByteStream &s) {
    Operation *def = symbols.lookup(name);
    if (!def) {
      // Not defined in this device (hand-written IR): the name is all there is.
      s.u8(TagUnresolvedSym);
      s.str(name);
      return success();
    }
    if (auto section = dyn_cast<AIEX::CertSectionOp>(def)) {
      FailureOr<SectionDigests> d = sectionDigests(section);
      if (failed(d))
        return failure();
      s.u8(TagSection);
      s.digest(d->id);
      return success();
    }
    FailureOr<Digest> d = definitionDigest(def);
    if (failed(d))
      return failure();
    s.u8(isa<AIEX::CertUcDmaChainOp>(def) ? TagChain : TagGlobal);
    s.digest(*d);
    return success();
  }

  // Digest of a uC-DMA chain or memref.global, memoized: one blockwrite
  // payload is typically named by several chains.
  FailureOr<Digest> definitionDigest(Operation *def) {
    if (auto it = memo.find(def); it != memo.end())
      return it->second;
    ByteStream s;
    if (auto chain = dyn_cast<AIEX::CertUcDmaChainOp>(def)) {
      // A chain's BDs: local_address, length, next_bd and the data behind
      // remote_address.
      s.u8(TagChain);
      if (failed(hashAttrs(chain, s)) || failed(hashRegions(chain, s)))
        return failure();
    } else {
      auto global = cast<memref::GlobalOp>(def);
      s.u8(TagGlobal);
      // The words AIETargetUcCert emits as `.long`: each element
      // zero-extended.
      auto init = global.getInitialValue();
      auto dense = init ? dyn_cast<DenseIntElementsAttr>(*init) : nullptr;
      if (dense) {
        s.u8(TagDenseIntAttr);
        s.u64(dense.getNumElements());
        for (const APInt &v : dense.getValues<APInt>())
          s.u64(v.getZExtValue());
      } else {
        if (failed(hashAttrs(global, s)))
          return failure();
      }
    }
    Digest d = s.hash();
    memo[def] = d;
    return d;
  }

  AsmState printState;
  llvm::StringMap<Operation *> symbols;
  llvm::DenseMap<Operation *, Digest> memo;
  llvm::DenseMap<Operation *, SectionDigests> sectionMemo;
  llvm::DenseSet<Operation *> inProgress;
};

// 32-bit id from a digest. Ids outside [1, mask-1] are reserved; on hitting one
// the digest's bytes are extended with 0x01 (once more per retry) and the
// result is hashed with xxh3-64 and folded again.
uint32_t foldToId(const Digest &d, unsigned bits) {
  uint32_t mask = bits >= 32 ? 0xFFFFFFFFu : ((1u << bits) - 1);
  auto fold = [&](uint64_t lo) {
    return (static_cast<uint32_t>(lo) ^ static_cast<uint32_t>(lo >> 32)) & mask;
  };
  uint32_t id = fold(d.low64);
  ByteStream s;
  s.digest(d);
  while (id == 0 || id == mask) {
    s.u8(0x01);
    id = fold(llvm::xxh3_64bits(s.bytes.data(), s.bytes.size()));
  }
  return id;
}

// The uCs each section can be loaded on. The firmware's skip-if-same-id cache
// is per uC, so that is the scope in which two ids may not collide. A
// reference (cert.load_pdi, or a cert.preempt's save/restore sections) outside
// any section runs on the uC of its enclosing cert.attach_to_group, or uC 0
// outside one. A reference inside a section runs on every uC that section is
// loaded on. A section no reference reaches gets the uC it is placed under.
llvm::DenseMap<Operation *, std::set<uint32_t>>
loadingUcs(AIE::DeviceOp device, const llvm::StringMap<Operation *> &sections) {
  // The uC an op runs on: its enclosing cert.attach_to_group, else the
  // `placement` of its enclosing cert.page if placement has not been lowered
  // to groups yet, else uC 0.
  auto groupOf = [](Operation *op) -> uint32_t {
    if (auto group = op->getParentOfType<AIEX::CertAttachToGroupOp>())
      return group.getGroupId();
    if (auto page = op->getParentOfType<AIEX::CertPageOp>())
      if (auto placement = page.getPlacement())
        return *placement;
    return 0;
  };

  llvm::DenseMap<Operation *, std::set<uint32_t>> ucs;
  // Section -> sections referenced from inside it, in IR order.
  llvm::DenseMap<Operation *, SmallVector<Operation *>> edges;
  SmallVector<Operation *> worklist;
  auto addRef = [&](Operation *user, StringRef name) {
    auto it = sections.find(name);
    if (it == sections.end())
      return;
    Operation *target = it->second;
    if (auto parent = user->getParentOfType<AIEX::CertSectionOp>()) {
      edges[parent.getOperation()].push_back(target);
      return;
    }
    if (ucs[target].insert(groupOf(user)).second)
      worklist.push_back(target);
  };
  device.walk([&](Operation *op) {
    if (auto load = dyn_cast<AIEX::CertLoadPdiOp>(op)) {
      addRef(op, load.getSymbol());
    } else if (auto preempt = dyn_cast<AIEX::CertPreemptOp>(op)) {
      addRef(op, preempt.getSaveSection());
      addRef(op, preempt.getRestoreSection());
    }
  });

  // Propagate along nested references until nothing changes.
  while (!worklist.empty()) {
    Operation *section = worklist.pop_back_val();
    auto it = edges.find(section);
    if (it == edges.end())
      continue;
    std::set<uint32_t> from = ucs[section];
    for (Operation *target : it->second) {
      std::set<uint32_t> &to = ucs[target];
      size_t before = to.size();
      to.insert(from.begin(), from.end());
      if (to.size() != before)
        worklist.push_back(target);
    }
  }

  for (auto &entry : sections) {
    std::set<uint32_t> &set = ucs[entry.second];
    if (set.empty())
      set.insert(groupOf(entry.second));
  }
  return ucs;
}

struct AIECertAssignPdiIdsPass
    : xilinx::AIEX::impl::AIECertAssignPdiIdsBase<AIECertAssignPdiIdsPass> {
  using Base =
      xilinx::AIEX::impl::AIECertAssignPdiIdsBase<AIECertAssignPdiIdsPass>;
  using Base::Base;

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    if (testIdBits < 2 || testIdBits > 32) {
      device.emitError("aie-cert-assign-pdi-ids: test-id-bits must be in "
                       "[2, 32], got ")
          << testIdBits.getValue();
      return signalPassFailure();
    }

    PdiHasher hasher(device);
    // Only sections some cert.load_pdi names get an id. Sections reachable
    // only through cert.preempt (save/restore) carry no id the firmware
    // compares, so they need not be unique; they are still hashed into the
    // digest of any section that references them.
    llvm::StringSet<> loaded;
    device.walk(
        [&](AIEX::CertLoadPdiOp load) { loaded.insert(load.getSymbol()); });

    // Sections are device-level symbols, so their labels are unique.
    llvm::StringMap<Operation *> sections;
    device.walk([&](AIEX::CertSectionOp section) {
      sections.try_emplace(section.getSymName(), section.getOperation());
    });
    llvm::DenseMap<Operation *, std::set<uint32_t>> ucs =
        loadingUcs(device, sections);

    // (uC, id) -> first loaded section with that id on that uC, and its
    // digests. std::map: ordered, so nothing here depends on hash-table
    // iteration.
    std::map<std::pair<uint32_t, uint32_t>,
             std::pair<SectionDigests, AIEX::CertSectionOp>>
        byUcAndId;
    llvm::StringMap<uint32_t> idByName;
    bool failedAny = false;

    device.walk([&](AIEX::CertSectionOp section) {
      if (!loaded.contains(section.getSymName()))
        return;
      FailureOr<SectionDigests> d = hasher.sectionDigests(section);
      if (failed(d)) {
        failedAny = true;
        return;
      }
      uint32_t id = foldToId(d->id, testIdBits);
      for (uint32_t uc : ucs[section.getOperation()]) {
        auto [it, inserted] = byUcAndId.try_emplace({uc, id}, *d, section);
        if (inserted)
          continue;
        // A shared id is fine, and lets the firmware skip the reload, exactly
        // when both sections have the same content.
        if (it->second.first.content == d->content)
          continue;
        AIEX::CertSectionOp first = it->second.second;
        InFlightDiagnostic diag = section.emitError();
        diag << "cert sections @" << first.getSymName() << " and @"
             << section.getSymName() << " are both loaded on uC " << uc
             << " and hash to the same PDI id "
             << llvm::formatv("{0:x8}", id).str()
             << " but differ in content; the firmware "
                "would skip loading one right after the other";
        diag.attachNote(first.getLoc())
            << "@" << first.getSymName() << " defined here";
        failedAny = true;
      }
      idByName.try_emplace(section.getSymName(), id);
    });
    if (failedAny)
      return signalPassFailure();

    device.walk([&](AIEX::CertLoadPdiOp load) {
      auto it = idByName.find(load.getSymbol());
      if (it == idByName.end()) {
        load.emitError("cert.load_pdi references @")
            << load.getSymbol()
            << ", which is not a cert.section in this device; its PDI id "
               "cannot be derived from its content";
        failedAny = true;
        return;
      }
      load.setPdiId(it->second);
    });
    if (failedAny)
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIECertAssignPdiIdsPass() {
  return std::make_unique<AIECertAssignPdiIdsPass>();
}

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIECertAssignPdiIdsPass(const AIECertAssignPdiIdsOptions &options) {
  return std::make_unique<AIECertAssignPdiIdsPass>(options);
}
