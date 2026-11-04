//===- AIECertPerSiteSections.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Gives every cert.load_pdi site a cert.section of its own, cloned from one
// legalized template per section. aiebu rejects two LOAD_PDIs of one label in
// a column and two section files of one name, so a program that loads a
// device twice (A -> B -> A, or the per-dispatch @empty_N of
// --expand-load-pdis) needs a distinct label and file per site. Cloning after
// cert-legalize-pages keeps every copy byte-identical, so
// aie-cert-assign-pdi-ids gives all copies the same id, and a repeated load is
// still skipped by the firmware.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringExtras.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECERTPERSITESECTIONS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIECertPerSiteSectionsPass
    : xilinx::AIEX::impl::AIECertPerSiteSectionsBase<
          AIECertPerSiteSectionsPass> {
  using Base = xilinx::AIEX::impl::AIECertPerSiteSectionsBase<
      AIECertPerSiteSectionsPass>;
  AIECertPerSiteSectionsPass() = default;
  AIECertPerSiteSectionsPass(const AIECertPerSiteSectionsOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    Block *body = device.getBody();

    // Load sites in program order. Section bodies are not program order (a
    // section runs where it is loaded), so the walk does not enter them; a
    // load inside one would make the copies of that section load further
    // copies, which is not supported.
    SmallVector<CertLoadPdiOp> sites;
    bool failedWalk = false;
    device->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (auto section = dyn_cast<CertSectionOp>(op)) {
        section.walk([&](CertLoadPdiOp nested) {
          nested.emitError("a cert.section that loads another section is not "
                           "supported with per-site sections");
          failedWalk = true;
        });
        return WalkResult::skip();
      }
      if (auto load = dyn_cast<CertLoadPdiOp>(op))
        sites.push_back(load);
      return WalkResult::advance();
    });
    if (failedWalk)
      return signalPassFailure();
    if (sites.empty())
      return;

    // A load needs a matching load on every participating uC (see
    // multi-uc); until those exist, per-site sections refuse such programs.
    if (!multiUc && llvm::any_of(body->getOps<CertAttachToGroupOp>(),
                                 [](CertAttachToGroupOp group) {
                                   return group.getGroupId() != 0;
                                 })) {
      sites.front().emitError(
          "per-site sections do not support programs on more than one "
          "microcontroller");
      return signalPassFailure();
    }

    // Group the sites by the section they load, in first-site order.
    llvm::MapVector<CertSectionOp, SmallVector<CertLoadPdiOp>> sitesOf;
    for (CertLoadPdiOp site : sites) {
      auto section = dyn_cast_or_null<CertSectionOp>(
          SymbolTable::lookupSymbolIn(device, site.getSymbolAttr()));
      if (!section || section->getParentOp() != device.getOperation()) {
        site.emitError("per-site sections need the loaded section @")
            << site.getSymbol() << " at device level";
        return signalPassFailure();
      }
      sitesOf[section].push_back(site);
    }

    OpBuilder builder(device.getContext());
    for (auto &[templ, templSites] : sitesOf) {
      // Canonical page-local IDs keep the section hash independent of the
      // surrounding program while preserving distinct firmware job slots.
      templ.walk([&](CertPageOp page) {
        uint32_t nextJobId = 1;
        for (auto job : page.getBody().front().getOps<CertJobOp>())
          job.setJobId(nextJobId++);
      });

      // aiebu folds labels to lower case, so the labels are lower case.
      std::string base = llvm::StringRef(templ.getSymName()).lower();
      for (auto [i, site] : llvm::enumerate(templSites)) {
        std::string label = base + "__site" + std::to_string(i);
        for (unsigned k = 0; SymbolTable::lookupSymbolIn(device, label); ++k)
          label = base + "__site" + std::to_string(i) + "_" + std::to_string(k);
        builder.setInsertionPoint(templ);
        auto copy = cast<CertSectionOp>(builder.clone(*templ));
        copy.setSymName(label);
        site.setSymbolAttr(FlatSymbolRefAttr::get(copy.getSymNameAttr()));
      }

      // A cert.preempt may still name the template as its save or restore
      // section.
      if (SymbolTable::symbolKnownUseEmpty(templ, device))
        templ.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIECertPerSiteSectionsPass() {
  return std::make_unique<AIECertPerSiteSectionsPass>();
}

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIECertPerSiteSectionsPass(
    const AIECertPerSiteSectionsOptions &options) {
  return std::make_unique<AIECertPerSiteSectionsPass>(options);
}
