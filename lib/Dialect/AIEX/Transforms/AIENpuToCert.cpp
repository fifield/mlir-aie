//===- AIENpuToCert.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <bitset>
#include <limits>
#include <map>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIENPUTOCERT
#define GEN_PASS_DEF_AIECERTPAGES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "npu-to-cert"

namespace {

// Trigger splitting below the 8 KB page limit to allow for estimated padding.
// An unsplittable page is an error only if it exceeds cert_page_limit.
static constexpr uint32_t cert_page_size = 8000;
static constexpr uint32_t cert_page_limit = 8192;

// Per-job control-code overhead: START_JOB (ISA_OPSIZE_START_JOB = 0x08) plus
// END_JOB (ISA_OPSIZE_END_JOB = 0x04), emitted around every job's body by
// emitJob.
static constexpr uint32_t cert_job_start_cost = 8;
static constexpr uint32_t cert_job_end_cost = 4;

// True if `seq` holds its device's generated configuration, i.e. carries the
// aie.generated_configure tag, either from convert-aie-to-transaction or from
// tagConfigureByName.
static bool isGeneratedConfigure(AIE::RuntimeSequenceOp seq) {
  return AIE::AIEDialect::GeneratedConfigureAttrHelper(seq.getContext())
      .isAttrPresent(seq);
}

// Hand-written IR may have no tag: there the sequence named `configure` is the
// configuration. Tag it, unless a sequence of the device is already tagged (the
// generated one was then named `configure0`, and `configure` is the user's).
// Runs on every device before any sequence is converted, so the choice does
// not depend on conversion order.
static void tagConfigureByName(AIE::DeviceOp device) {
  auto seqs = device.getBody()->getOps<AIE::RuntimeSequenceOp>();
  if (llvm::any_of(seqs, isGeneratedConfigure))
    return;
  for (auto seq : seqs)
    if (seq.getSymName() == "configure") {
      AIE::AIEDialect::GeneratedConfigureAttrHelper(device.getContext())
          .setAttr(seq, UnitAttr::get(device.getContext()));
      return;
    }
}

// The generated configuration sequence of `device`, or null if it has none.
static AIE::RuntimeSequenceOp findGeneratedConfigure(AIE::DeviceOp device) {
  for (auto seq : device.getBody()->getOps<AIE::RuntimeSequenceOp>())
    if (isGeneratedConfigure(seq))
      return seq;
  return nullptr;
}

// Returns the PDI id to use for a cert.load_pdi that references `deviceSymName`
// within `parentDevice`. If a load_pdi for that symbol already exists, its id
// is reused so repeated loads of the same PDI share an id; otherwise a fresh id
// (1-based, one past the current maximum) is assigned.
static uint32_t getOrAssignPdiId(AIE::DeviceOp parentDevice,
                                 StringRef deviceSymName) {
  uint32_t pdiId = 1;
  bool foundExisting = false;
  parentDevice.walk([&](AIEX::CertLoadPdiOp loadPdiOp) {
    if (loadPdiOp.getSymbol() == deviceSymName) {
      pdiId = loadPdiOp.getPdiId();
      foundExisting = true;
    } else if (!foundExisting) {
      pdiId = std::max(pdiId, loadPdiOp.getPdiId() + 1);
    }
  });
  return pdiId;
}

struct RuntimeSequenceToCertJob : OpConversionPattern<AIE::RuntimeSequenceOp> {
  using OpConversionPattern::OpConversionPattern;

  RuntimeSequenceToCertJob(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(AIE::RuntimeSequenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    bool isConfigure = isGeneratedConfigure(op);
    uint32_t newJobId = 1;
    if (!isConfigure) {
      uint32_t maxJobId = 1;
      op->getParentOp()->walk([&](AIEX::CertJobOp certJobOp) {
        maxJobId = std::max(maxJobId, certJobOp.getJobId());
      });
      newJobId = maxJobId + 1;
    }

    // Replace the runtime sequence with a job directly inside aie.device.
    // CertJobOp lists DeviceOp as an allowed parent in its ParentOneOf trait.
    // cert-legalize-pages later groups contiguous jobs into implicit pages.
    rewriter.setInsertionPoint(op);
    auto jobOp = AIEX::CertJobOp::create(rewriter, op.getLoc(), newJobId);

    // Tag the implicit configuration job so a later step can force it to be the
    // first page on uC0 and keep it out of user implicit pages.
    if (isConfigure)
      jobOp->setAttr("cert.configure", rewriter.getUnitAttr());

    // Clone runtime sequence body into job
    // Note: This preserves block arguments from the runtime sequence, which
    // will be present in the MLIR IR but are not emitted in the final assembly
    IRMapping remap;
    op.getRegion().cloneInto(&jobOp.getBody(), remap);
    AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), rewriter, op->getLoc());

    // Erase the original runtime sequence
    rewriter.eraseOp(op);

    return success();
  }
};

struct NpuWrite32ToCertWrite32 : OpConversionPattern<AIEX::NpuWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    std::optional<uint32_t> address =
        AIEX::getConstantIntOperand(op.getAddress());
    std::optional<uint32_t> value = AIEX::getConstantIntOperand(op.getValue());
    if (!address || !value)
      return op.emitOpError(
          "cannot lower to cert.write32 with non-constant address or value");

    // Get the absolute address, which properly handles row/col if present
    std::optional<uint32_t> absAddress = op.getAbsoluteAddress();
    if (!absAddress)
      return failure();

    uint32_t absoluteAddr = *absAddress;

    // If row and col are specified, validate that the address upper bits match
    std::optional<uint32_t> col = op.getColumn();
    std::optional<uint32_t> row = op.getRow();
    if (col && row) {
      const auto &tm = AIE::getTargetModel(op);
      uint32_t expectedUpperBits = ((*col & 0xff) << tm.getColumnShift()) |
                                   ((*row & 0xff) << tm.getRowShift());

      // Warn if the original address had non-zero upper bits that don't match
      uint32_t origAddress = *address;
      uint32_t origUpperBits = origAddress & ~0xfffff;
      if (origUpperBits != 0 && origUpperBits != expectedUpperBits) {
        op.emitWarning() << "address upper bits (0x"
                         << llvm::utohexstr(origUpperBits)
                         << ") don't match row=" << *row << " col=" << *col
                         << " computed bits (0x"
                         << llvm::utohexstr(expectedUpperBits) << ")";
      }
    }

    rewriter.replaceOpWithNewOp<AIEX::CertWrite32Op>(op, absoluteAddr, *value);
    return success();
  }
};

struct NpuMaskWrite32ToCertMaskWrite32
    : OpConversionPattern<AIEX::NpuMaskWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuMaskWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    std::optional<uint32_t> address =
        AIEX::getConstantIntOperand(op.getAddress());
    std::optional<uint32_t> mask = AIEX::getConstantIntOperand(op.getMask());
    std::optional<uint32_t> value = AIEX::getConstantIntOperand(op.getValue());
    if (!address || !mask || !value)
      return op.emitOpError(
          "cannot lower to cert.maskwrite32 with non-constant "
          "address, mask, or value");

    // Get the absolute address, which properly handles row/col if present
    std::optional<uint32_t> absAddress = op.getAbsoluteAddress();
    if (!absAddress)
      return failure();

    uint32_t absoluteAddr = *absAddress;

    // If row and col are specified, validate that the address upper bits match
    std::optional<uint32_t> col = op.getColumn();
    std::optional<uint32_t> row = op.getRow();
    if (col && row) {
      const auto &tm = AIE::getTargetModel(op);
      uint32_t expectedUpperBits = ((*col & 0xff) << tm.getColumnShift()) |
                                   ((*row & 0xff) << tm.getRowShift());

      // Warn if the original address had non-zero upper bits that don't match
      uint32_t origAddress = *address;
      uint32_t origUpperBits = origAddress & ~0xfffff;
      if (origUpperBits != 0 && origUpperBits != expectedUpperBits) {
        op.emitWarning() << "address upper bits (0x"
                         << llvm::utohexstr(origUpperBits)
                         << ") don't match row=" << *row << " col=" << *col
                         << " computed bits (0x"
                         << llvm::utohexstr(expectedUpperBits) << ")";
      }
    }

    rewriter.replaceOpWithNewOp<AIEX::CertMaskWrite32Op>(op, absoluteAddr,
                                                         *mask, *value);
    return success();
  }
};

struct NpuMaskPollToCertMaskPoll32 : OpConversionPattern<AIEX::NpuMaskPollOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuMaskPollOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<uint32_t> address =
        AIEX::getConstantIntOperand(op.getAddress());
    std::optional<uint32_t> mask = AIEX::getConstantIntOperand(op.getMask());
    std::optional<uint32_t> value = AIEX::getConstantIntOperand(op.getValue());
    if (!address || !mask || !value)
      return op.emitOpError("cannot lower to cert.maskpoll32 with non-constant "
                            "address, mask, or value");

    std::optional<uint32_t> absAddress = op.getAbsoluteAddress();
    if (!absAddress)
      return failure();

    rewriter.replaceOpWithNewOp<AIEX::CertMaskPoll32Op>(op, *absAddress, *mask,
                                                        *value);
    return success();
  }
};

struct NpuBlockWriteToCertUcDma : OpConversionPattern<AIEX::NpuBlockWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  mutable unsigned nextChainId = 0;

  LogicalResult
  matchAndRewrite(AIEX::NpuBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    memref::GetGlobalOp dataOperand =
        dyn_cast_or_null<memref::GetGlobalOp>(op.getData().getDefiningOp());
    if (!dataOperand)
      return failure();
    MemRefType dataType = cast<MemRefType>(dataOperand.getResult().getType());
    // Empty copies are no-ops; emitting a uC DMA descriptor would reference
    // an empty payload that the assembler cannot resolve.
    if (dataType.getNumElements() == 0) {
      rewriter.eraseOp(op);
      return success();
    }
    uint32_t dataSize = dataType.getNumElements();

    // Continue past previously allocated names instead of rescanning them.
    std::string symbolName = "chain_" + std::to_string(nextChainId++);
    while (op->getParentOfType<AIE::DeviceOp>().lookupSymbol(symbolName))
      symbolName = "chain_" + std::to_string(nextChainId++);

    // Create a new uc_dma_write_des_sync operation
    rewriter.replaceOpWithNewOp<AIEX::CertUcDmaWriteDesSyncOp>(op, symbolName);

    // Create the uc_dma_chain operation
    // Find the nearest device to insert the chain
    auto parentDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!parentDevice) {
      // No parent device - this shouldn't happen but handle gracefully
      return failure();
    }

    // Insert after the last existing uc_dma_chain (before any pages/jobs),
    // preserving the order of blockwrite ops. Globals are allowed to precede
    // the chains, so skip over them too.
    Block *deviceBody = parentDevice.getBody();
    Operation *insertAfter = nullptr;
    for (Operation &bodyOp : *deviceBody) {
      if (isa<AIEX::CertUcDmaChainOp>(bodyOp) || isa<memref::GlobalOp>(bodyOp))
        insertAfter = &bodyOp;
      else
        break;
    }
    if (insertAfter)
      rewriter.setInsertionPointAfter(insertAfter);
    else
      rewriter.setInsertionPointToStart(deviceBody);

    auto symbolAttr = rewriter.getStringAttr(symbolName);
    auto chainOp =
        AIEX::CertUcDmaChainOp::create(rewriter, op.getLoc(), symbolAttr);

    Block *bb = new Block();
    chainOp.getRegion().push_back(bb);
    rewriter.setInsertionPointToStart(bb);
    AIEX::CertUcDmaBdOp::create(rewriter, op.getLoc(), dataOperand.getName(),
                                op.getAddress(), dataSize, false);

    AIEX::CertUcDmaChainOp::ensureTerminator(chainOp.getBody(), rewriter,
                                             op->getLoc());
    return success();
  }
};

struct NpuSyncToCertWaitTCTS : OpConversionPattern<AIEX::NpuSyncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<uint32_t> rowOpt = AIEX::getConstantIntOperand(op.getRow());
    std::optional<uint32_t> colOpt =
        AIEX::getConstantIntOperand(op.getColumn());
    std::optional<uint32_t> channelOpt =
        AIEX::getConstantIntOperand(op.getChannel());
    std::optional<uint32_t> directionOpt =
        AIEX::getConstantIntOperand(op.getDirection());
    if (!rowOpt || !colOpt || !channelOpt || !directionOpt)
      return op.emitOpError(
          "cannot lower to cert.wait_tcts with non-constant sync parameters");
    uint32_t row = *rowOpt;
    uint32_t col = *colOpt;
    // These are the shift amounts from the tct packet format in the
    // architecture spec. The firmware expects the row and column packed and
    // shifted down to zero.
    const int row_id_shift = 16;
    const int col_id_shift = 21;
    uint16_t tile_id = col << (col_id_shift - row_id_shift) | row;
    uint32_t channel = *channelOpt;
    uint32_t direction = *directionOpt;

    const std::vector<int> chan2actor_shim_s2mm = {0, 2, 3, 4};
    const std::vector<int> chan2actor_shim_mm2s = {6, 7, 8, 9, 10, 11, 12, 13};
    // AIE4 MEM S2MM0 through S2MM7 are TCT actors 0 through 7.
    const std::vector<int> chan2actor_mem_s2mm = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<int> chan2actor_mem_mm2s = {16, 17, 18, 19, 20,
                                                  22, 23, 24, 25, 26};
    const std::vector<int> chan2actor_tile_s2mm = {0, 1};
    const std::vector<int> chan2actor_tile_mm2s = {6};
    // AIE4 shim control-MM2S and trace-S2MM channels are *not* part of the
    // regular MM2S/S2MM actor ranges. Ctrl_MM2S0/1 are actors 16/17 and
    // Trace_S2MM is actor 1.
    const std::vector<int> chan2actor_shim_ctrl_mm2s = {16, 17};
    const std::vector<int> chan2actor_shim_trace_s2mm = {1};

    const auto &tm = AIE::getTargetModel(op);
    auto dirEnum = static_cast<AIE::DMAChannelDir>(direction);
    const bool isS2MM = dirEnum == AIE::DMAChannelDir::S2MM;

    const std::vector<int> *chan2actor = nullptr;
    const char *dirName = isS2MM ? "S2MM" : "MM2S";
    if (dirEnum == AIE::DMAChannelDir::ControlMM2S ||
        dirEnum == AIE::DMAChannelDir::TraceS2MM) {
      // Control-MM2S and trace-S2MM only exist on the shim.
      if (!tm.isShimNOCTile(col, row))
        return op.emitOpError()
               << "direction " << stringifyDMAChannelDir(dirEnum)
               << " is only valid on a shim tile, but tile (" << col << ", "
               << row << ") is not one";
      if (dirEnum == AIE::DMAChannelDir::ControlMM2S) {
        chan2actor = &chan2actor_shim_ctrl_mm2s;
        dirName = "ControlMM2S";
      } else {
        chan2actor = &chan2actor_shim_trace_s2mm;
        dirName = "TraceS2MM";
      }
    } else if (tm.isCoreTile(col, row))
      chan2actor = isS2MM ? &chan2actor_tile_s2mm : &chan2actor_tile_mm2s;
    else if (tm.isMemTile(col, row))
      chan2actor = isS2MM ? &chan2actor_mem_s2mm : &chan2actor_mem_mm2s;
    else
      chan2actor = isS2MM ? &chan2actor_shim_s2mm : &chan2actor_shim_mm2s;

    size_t chanIdx = static_cast<size_t>(channel);
    if (chanIdx >= chan2actor->size()) {
      op.emitError("invalid DMA channel ")
          << channel << " for " << dirName
          << " direction in NpuSyncToCertWaitTCTS conversion";
      return failure();
    }

    uint8_t actor_id = static_cast<uint8_t>((*chan2actor)[chanIdx]);
    uint8_t num_tcts = 1;
    rewriter.replaceOpWithNewOp<AIEX::CertWaitTCTSOp>(op, tile_id, actor_id,
                                                      num_tcts);
    return success();
  }
};

struct NpuLoadPdiToCertLoadPdi : OpConversionPattern<AIEX::NpuLoadPdiOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuLoadPdiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the device reference
    auto deviceRef = op.getDeviceRef();
    if (!deviceRef)
      return failure();

    StringRef deviceSymName = *deviceRef;

    // Find parent device to get unique PDI ID
    auto parentDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!parentDevice)
      return failure();

    // Assign a PDI ID, reusing an existing one for this device if present.
    uint32_t pdiId = getOrAssignPdiId(parentDevice, deviceSymName);

    // Replace with cert.load_pdi
    rewriter.replaceOpWithNewOp<AIEX::CertLoadPdiOp>(
        op, rewriter.getUI32IntegerAttr(pdiId),
        FlatSymbolRefAttr::get(rewriter.getContext(), deviceSymName));

    return success();
  }
};

struct RunOpInlining : OpRewritePattern<AIEX::RunOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::RunOp op,
                                PatternRewriter &rewriter) const override {
    // Get the callee runtime sequence
    AIE::RuntimeSequenceOp calleeRuntimeSequence =
        op.getCalleeRuntimeSequenceOp();
    if (!calleeRuntimeSequence)
      return failure();

    // Get the callee body region
    Region &calleeBody = calleeRuntimeSequence.getBody();

    // Create argument mapping from run op arguments to callee parameters
    IRMapping argMap;
    ValueRange values = op.getArgs();
    if (values.size() != calleeBody.getNumArguments()) {
      op.emitError("number of run op arguments (")
          << values.size()
          << ") does not match number of callee runtime sequence "
             "arguments ("
          << calleeBody.getNumArguments() << ")";
      return failure();
    }
    for (unsigned i = 0, n = calleeBody.getNumArguments(); i < n; i++) {
      BlockArgument arg = calleeBody.getArgument(i);
      Value val = values[i];
      argMap.map(arg, val);
    }

    // Clone operations from callee into current location
    rewriter.setInsertionPoint(op);
    for (Operation &o : calleeBody.front().getOperations()) {
      // Skip the terminator
      if (isa<AIE::EndOp>(o))
        continue;
      rewriter.clone(o, argMap);
    }

    // Erase the run op
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConfigureOpToCertSection : OpRewritePattern<AIEX::ConfigureOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::ConfigureOp op,
                                PatternRewriter &rewriter) const override {
    // Get the referenced device
    AIE::DeviceOp referencedDevice = op.getReferencedDeviceOp();
    if (!referencedDevice)
      return failure();

    // Get the device's symbol name for the section
    StringRef deviceSymName = referencedDevice.getSymName();

    // Find the parent DeviceOp to insert the section into
    auto parentDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!parentDevice)
      return failure();

    // Check if section already exists (avoid creating duplicates)
    auto existingSection = dyn_cast_if_present<AIEX::CertSectionOp>(
        parentDevice.lookupSymbol(deviceSymName));
    if (existingSection) {
      // Section already exists, just create load_pdi at call site, reusing the
      // PDI ID already assigned to this section (or assigning a new one).
      uint32_t pdiId = getOrAssignPdiId(parentDevice, deviceSymName);

      rewriter.setInsertionPoint(op);
      AIEX::CertLoadPdiOp::create(
          rewriter, op.getLoc(), rewriter.getUI32IntegerAttr(pdiId),
          FlatSymbolRefAttr::get(rewriter.getContext(), deviceSymName));

      // Clone configure body operations after the load_pdi
      IRMapping bodyMapper;
      for (Operation &o : op.getRegion().front().getOperations()) {
        if (isa<AIE::EndOp>(o))
          continue;
        rewriter.clone(o, bodyMapper);
      }

      rewriter.eraseOp(op);
      return success();
    }

    // Create cert.section with the device symbol name
    rewriter.setInsertionPoint(parentDevice.getBody()->getTerminator());
    auto sectionOp = AIEX::CertSectionOp::create(
        rewriter, op.getLoc(), rewriter.getStringAttr(deviceSymName));

    // Create the section body with a page containing a job
    Block *sectionBlock = new Block();
    sectionOp.getBody().push_back(sectionBlock);
    rewriter.setInsertionPointToStart(sectionBlock);

    // Create page within section
    auto pageOp = AIEX::CertPageOp::create(rewriter, op.getLoc());
    Block *pageBlock = new Block();
    pageOp.getBody().push_back(pageBlock);
    rewriter.setInsertionPointToStart(pageBlock);

    // Create job within page - assign unique job ID across all devices
    // Find the maximum job ID in the parent device
    uint32_t maxJobId = 0;
    parentDevice.walk([&](AIEX::CertJobOp certJobOp) {
      maxJobId = std::max(maxJobId, certJobOp.getJobId());
    });
    uint32_t sectionJobId = maxJobId + 1;

    auto jobOp = AIEX::CertJobOp::create(rewriter, op.getLoc(), sectionJobId);
    Block *jobBlock = new Block();
    jobOp.getBody().push_back(jobBlock);
    rewriter.setInsertionPointToStart(jobBlock);

    // Find the "configure" sequence content from the referenced device
    // It could be either:
    // 1. Still the generated configuration sequence (if that device hasn't
    // been processed yet)
    // 2. Already converted to cert.page/cert.job (if that device was processed
    // first)

    AIE::RuntimeSequenceOp configureSeq =
        findGeneratedConfigure(referencedDevice);

    if (configureSeq) {
      // Clone operations from the "configure" runtime_sequence
      IRMapping mapper;
      for (Operation &o : configureSeq.getRegion().front().getOperations()) {
        if (isa<AIE::EndOp>(o))
          continue;
        rewriter.clone(o, mapper);
      }
    } else {
      // "configure" was already converted - find the first cert.page
      // (configuration page)
      AIEX::CertPageOp configPage = nullptr;
      for (auto &op : referencedDevice.getBody()->getOperations()) {
        if (auto page = dyn_cast<AIEX::CertPageOp>(op)) {
          configPage = page;
          break; // Take first page (configure)
        }
      }

      if (configPage) {
        // Clone operations from the config page's job
        IRMapping mapper;
        for (auto &op : configPage.getBody().front().getOperations()) {
          if (auto job = dyn_cast<AIEX::CertJobOp>(op)) {
            for (Operation &jobOp : job.getBody().front().getOperations()) {
              if (isa<AIE::EndOp>(jobOp))
                continue;
              rewriter.clone(jobOp, mapper);
            }
            break; // Only clone first job
          }
        }
      }
    }

    // Ensure terminators
    AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), rewriter, op.getLoc());
    AIEX::CertPageOp::ensureTerminator(pageOp.getBody(), rewriter, op.getLoc());
    AIEX::CertSectionOp::ensureTerminator(sectionOp.getBody(), rewriter,
                                          op.getLoc());

    // At the configure call site, replace with load_pdi, reusing the PDI ID
    // already assigned to this device if present (or assigning a new one).
    uint32_t pdiId = getOrAssignPdiId(parentDevice, deviceSymName);

    rewriter.setInsertionPoint(op);
    AIEX::CertLoadPdiOp::create(
        rewriter, op.getLoc(), rewriter.getUI32IntegerAttr(pdiId),
        FlatSymbolRefAttr::get(rewriter.getContext(), deviceSymName));

    // Clone configure body operations after the load_pdi
    // Operations that reference external values were not cloned into the
    // section, so we clone them here into the main control flow
    IRMapping bodyMapper;
    for (Operation &o : op.getRegion().front().getOperations()) {
      if (isa<AIE::EndOp>(o))
        continue;
      rewriter.clone(o, bodyMapper);
    }

    // Erase the original configure op
    rewriter.eraseOp(op);
    return success();
  }
};

// The blockwrite that loads the BD an address_patch patches: the nearest
// preceding blockwrite, when it targets the patched BD.
static AIEX::NpuBlockWriteOp
findPatchedBlockWrite(AIEX::NpuAddressPatchOp patch) {
  const auto &tm = AIE::getTargetModel(patch);
  uint32_t addr = patch.getAddr();
  int col = (addr >> tm.getColumnShift()) & 0x1f;
  int row = (addr >> tm.getRowShift()) & 0x1f;
  if (!tm.isValidTile({col, row}))
    return {};
  for (Block::iterator it(patch); it != patch->getBlock()->begin();) {
    auto blockWrite = dyn_cast<AIEX::NpuBlockWriteOp>(*--it);
    if (!blockWrite)
      continue;
    if (blockWrite.getAddress() + tm.getDmaBdAddressOffset(col, row) != addr)
      return {};
    return blockWrite;
  }
  return {};
}

// An address_patch writes buffer address + arg_plus into a BD register, but
// cert.apply_offset_57 adds the buffer address to the BD template in memory.
// So before conversion the patched template word must hold arg_plus, and the
// template must belong to its blockwrite alone: dma-to-npu shares identical
// templates between BDs, and a shared one would accumulate every BD's address.
static LogicalResult prepareAddressPatchTemplates(AIE::DeviceOp device) {
  LogicalResult result = success();
  SymbolTable symbols(device);
  // Count each global's symbol uses once, then keep the counts current as
  // getters are added and erased. Asking SymbolTable per patch walks the whole
  // device each time, which is quadratic in the number of patched BDs. With no
  // counts (an op with unknown symbol uses), every patched template is copied.
  std::optional<llvm::DenseMap<StringAttr, unsigned>> symbolUseCounts;
  if (auto uses = SymbolTable::getSymbolUses(&device.getRegion())) {
    symbolUseCounts.emplace();
    for (const SymbolTable::SymbolUse &use : *uses)
      ++(*symbolUseCounts)[use.getSymbolRef().getRootReference()];
  }
  SmallVector<AIEX::NpuAddressPatchOp> patches;
  device.walk([&](AIEX::NpuAddressPatchOp patch) { patches.push_back(patch); });
  for (AIEX::NpuAddressPatchOp patch : patches) {
    AIEX::NpuBlockWriteOp blockWrite = findPatchedBlockWrite(patch);
    if (!blockWrite)
      continue;
    auto getGlobal = dyn_cast_or_null<memref::GetGlobalOp>(
        blockWrite.getData().getDefiningOp());
    if (!getGlobal)
      continue;
    auto global = symbols.lookup<memref::GlobalOp>(getGlobal.getName());
    if (!global)
      continue;
    std::optional<Attribute> initialValue = global.getInitialValue();
    auto words = initialValue ? dyn_cast<DenseIntElementsAttr>(*initialValue)
                              : DenseIntElementsAttr();
    if (!words)
      continue;

    if (!patch.getArgIdx()) {
      patch.emitError("cert lowering requires an address_patch with arg_idx");
      result = failure();
      continue;
    }
    APInt argPlus;
    if (!matchPattern(patch.getArgPlus(), m_ConstantInt(&argPlus))) {
      patch.emitError("cert lowering requires a constant arg_plus: "
                      "apply_offset_57 cannot add a runtime offset");
      result = failure();
      continue;
    }
    // An i32 arg_plus is unsigned, so any value fits; a negative i64 has 64
    // active bits and is rejected.
    if (argPlus.getActiveBits() > 32) {
      patch.emitError("arg_plus ")
          << argPlus.getSExtValue()
          << " does not fit the BD's 32-bit address word";
      result = failure();
      continue;
    }
    unsigned word = (patch.getAddr() - blockWrite.getAddress()) / 4;
    if (word >= words.getNumElements())
      continue;

    // Give this blockwrite a private template if anything else uses it,
    // whether through another getter or through this one (after CSE).
    if (!symbolUseCounts ||
        symbolUseCounts->lookup(global.getSymNameAttr()) != 1 ||
        !getGlobal->hasOneUse()) {
      OpBuilder builder(global);
      auto copy = cast<memref::GlobalOp>(builder.clone(*global));
      // insert renames the copy, so count it under its new name.
      StringAttr copyName = symbols.insert(copy);
      OpBuilder atBlockWrite(blockWrite);
      auto privateGet =
          memref::GetGlobalOp::create(atBlockWrite, getGlobal.getLoc(),
                                      getGlobal.getType(), copy.getSymName());
      blockWrite.getDataMutable().assign(privateGet);
      if (symbolUseCounts)
        (*symbolUseCounts)[copyName] = 1;
      if (getGlobal->use_empty()) {
        if (symbolUseCounts)
          --(*symbolUseCounts)[global.getSymNameAttr()];
        getGlobal->erase();
      }
      global = copy;
    }

    auto values = llvm::to_vector(words.getValues<APInt>());
    values[word] = argPlus.zextOrTrunc(values[word].getBitWidth());
    global.setInitialValueAttr(
        DenseIntElementsAttr::get(words.getType(), values));
  }
  return result;
}

struct NpuAddressPatchToCertApplyOffset57
    : OpConversionPattern<AIEX::NpuAddressPatchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuAddressPatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIEX::NpuBlockWriteOp blockWriteOp = findPatchedBlockWrite(op);
    std::optional<uint32_t> argIdx = op.getArgIdx();
    if (!blockWriteOp)
      return failure();
    auto getGlobalOp =
        dyn_cast<memref::GetGlobalOp>(blockWriteOp.getData().getDefiningOp());
    if (!getGlobalOp)
      return failure();
    if (!argIdx)
      return op.emitOpError(
          "requires a resolved 'arg_idx' for CERT lowering; run "
          "-aie-resolve-address-patch-buffers first");
    if (*argIdx > std::numeric_limits<uint16_t>::max())
      return op.emitOpError("arg_idx exceeds the CERT 16-bit offset range");

    // replace the address with the new address
    rewriter.setInsertionPoint(blockWriteOp);
    rewriter.replaceOpWithNewOp<AIEX::CertApplyOffset57Op>(
        op, getGlobalOp.getName(), 1,
        static_cast<uint16_t>(*argIdx));
    return success();
  }
};

struct MergeConsecutiveCertUcDmaWriteDesSyncOps
    : OpRewritePattern<AIEX::CertUcDmaWriteDesSyncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::CertUcDmaWriteDesSyncOp op,
                                PatternRewriter &rewriter) const override {
    // Fusing the two enqueues sinks the earlier one to the later one's position
    // in the job, so every op scanned over is an op the earlier DMA is moved
    // past. Only ops with no architectural effect at all may be crossed; this
    // is an allowlist rather than a denylist so a cert op added to this dialect
    // later blocks the merge by default instead of silently becoming
    // reorderable with respect to a DMA enqueue. cert.nop is the only such op
    // today: it emits padding and carries no ordering, sync or side effect.
    auto isTransparentToMerge = [](Operation *o) {
      return isa<AIEX::CertNopOp>(o);
    };

    // Get the previous operation in the block
    Block::iterator it(op);
    AIEX::CertUcDmaWriteDesSyncOp prevWriteDesSync = nullptr;
    while (it != op->getBlock()->begin() && !prevWriteDesSync) {
      --it;
      Operation *prevOp = &*it;
      prevWriteDesSync = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(prevOp);
      if (!prevWriteDesSync && !isTransparentToMerge(prevOp))
        return failure();
    }
    if (!prevWriteDesSync)
      return failure();

    // find the uc_dma_chain
    StringRef sym_name = op.getSymbol();
    StringRef prev_sym_name = prevWriteDesSync.getSymbol();
    auto chain = dyn_cast_if_present<AIEX::CertUcDmaChainOp>(
        op->getParentOfType<AIE::DeviceOp>().lookupSymbol(sym_name));
    auto prevChain = dyn_cast_if_present<AIEX::CertUcDmaChainOp>(
        prevWriteDesSync->getParentOfType<AIE::DeviceOp>().lookupSymbol(
            prev_sym_name));
    if (!chain || !prevChain)
      return failure();

    // The merge rewrites @chain's BD list in place and deletes @prevChain, so
    // both symbols must be private to this pair of enqueues. Cert IR that is
    // hand-authored rather than lowered from npu ops may enqueue one chain from
    // several places: folding into a shared @chain would silently change those
    // other enqueues, and erasing a shared @prevChain would leave a dangling
    // symbol reference that neither the verifier nor the page cost model
    // notices. Decline the merge instead.
    auto deviceOp = op->getParentOfType<AIE::DeviceOp>();
    auto isSingleUse = [&](AIEX::CertUcDmaChainOp c) {
      auto uses = SymbolTable::getSymbolUses(c, deviceOp);
      return uses && llvm::hasSingleElement(*uses);
    };
    if (!isSingleUse(chain) || !isSingleUse(prevChain))
      return failure();

    // Compute the size of the current and previous chains. If their combined
    // data size is greater than the cert page size, then we cannot merge them.
    //
    // A chain costs 16 bytes of descriptor per BD on top of the BD's payload --
    // the same accounting updateCostForOp (the page splitter's authoritative
    // cost model) uses. Charging payload only under-reports a chain by 16 bytes
    // per BD, which for many small BDs is most of its real size.
    auto chainSize = [](AIEX::CertUcDmaChainOp c) {
      uint32_t size = 0;
      for (auto bdOp : c.getBody().front().getOps<AIEX::CertUcDmaBdOp>())
        size += bdOp.getLength() * sizeof(int) + 16; // payload + bd descriptor
      return size;
    };
    if ((chainSize(chain) + chainSize(prevChain)) >= cert_page_size)
      return failure();

    IRMapping map;
    rewriter.setInsertionPointToStart(&chain.getBody().front());
    for (auto &o : prevChain.getBody().front().getOperations()) {
      auto bdOp = dyn_cast<AIEX::CertUcDmaBdOp>(o);
      if (!bdOp)
        continue;
      AIEX::CertUcDmaBdOp::create(
          rewriter, bdOp.getLoc(), bdOp.getRemoteAddress(),
          bdOp.getLocalAddress(), bdOp.getLength(), true);
    }
    rewriter.eraseOp(prevChain);
    rewriter.eraseOp(prevWriteDesSync);
    return success();
  }
};

struct SplitNpuBlockWriteOpPattern : OpRewritePattern<AIEX::NpuBlockWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::NpuBlockWriteOp op,
                                PatternRewriter &rewriter) const override {

    memref::GetGlobalOp dataOperand =
        dyn_cast_or_null<memref::GetGlobalOp>(op.getData().getDefiningOp());
    if (!dataOperand)
      return failure();

    MemRefType dataType = cast<MemRefType>(dataOperand.getResult().getType());
    uint32_t dataSize = dataType.getNumElements();

    uint32_t dataSizeBytes = dataSize * sizeof(int);
    if (dataSizeBytes < cert_page_size)
      return failure();

    auto loc = op.getLoc();

    // Calculate split point (split roughly in half)
    uint32_t splitElements = dataSize / 2;
    uint32_t firstChunkSize = splitElements;
    uint32_t secondChunkSize = dataSize - splitElements;

    // Find the original memref.global operation
    auto deviceOp = op->getParentOfType<AIE::DeviceOp>();
    auto originalGlobal = dyn_cast_if_present<memref::GlobalOp>(
        deviceOp.lookupSymbol(dataOperand.getName()));
    if (!originalGlobal)
      return failure();

    // Get the original data attribute
    auto originalData = originalGlobal.getInitialValue();
    if (!originalData)
      return failure();

    auto denseData = dyn_cast<DenseIntElementsAttr>(*originalData);
    if (!denseData)
      return failure();

    // Split the data into two chunks
    auto dataValues = denseData.getValues<APInt>();
    std::vector<APInt> firstChunkData(dataValues.begin(),
                                      dataValues.begin() + firstChunkSize);
    std::vector<APInt> secondChunkData(dataValues.begin() + firstChunkSize,
                                       dataValues.end());

    // Create new global operations for the split data
    auto elementType = rewriter.getI32Type();
    auto firstChunkType = MemRefType::get({firstChunkSize}, elementType);
    auto secondChunkType = MemRefType::get({secondChunkSize}, elementType);
    TensorType firstTensorType =
        RankedTensorType::get({firstChunkSize}, elementType);
    TensorType secondTensorType =
        RankedTensorType::get({secondChunkSize}, elementType);

    auto firstChunkAttr =
        DenseIntElementsAttr::get(firstTensorType, firstChunkData);
    auto secondChunkAttr =
        DenseIntElementsAttr::get(secondTensorType, secondChunkData);

    // Generate unique names for the new globals
    std::string firstName = dataOperand.getName().str() + "_split_0";
    std::string secondName = dataOperand.getName().str() + "_split_1";

    // Ensure unique names
    int counter = 0;
    while (deviceOp.lookupSymbol(firstName)) {
      firstName =
          dataOperand.getName().str() + "_split_0_" + std::to_string(counter++);
    }
    counter = 0;
    while (deviceOp.lookupSymbol(secondName)) {
      secondName =
          dataOperand.getName().str() + "_split_1_" + std::to_string(counter++);
    }

    // Create the new global operations
    rewriter.setInsertionPoint(originalGlobal);
    memref::GlobalOp::create(rewriter, loc, firstName,
                             rewriter.getStringAttr("private"), firstChunkType,
                             firstChunkAttr, true, nullptr);

    memref::GlobalOp::create(rewriter, loc, secondName,
                             rewriter.getStringAttr("private"), secondChunkType,
                             secondChunkAttr, true, nullptr);

    // Create get_global operations for the new data
    rewriter.setInsertionPoint(op);

    auto firstGetGlobal =
        memref::GetGlobalOp::create(rewriter, loc, firstChunkType, firstName);
    auto secondGetGlobal =
        memref::GetGlobalOp::create(rewriter, loc, secondChunkType, secondName);

    uint32_t baseAddr = op.getAddress();

    AIEX::NpuBlockWriteOp::create(rewriter, loc, baseAddr,
                                  firstGetGlobal.getResult(), nullptr, nullptr,
                                  nullptr);

    AIEX::NpuBlockWriteOp::create(rewriter, loc, baseAddr + firstChunkSize * 4,
                                  secondGetGlobal.getResult(), nullptr, nullptr,
                                  nullptr);

    // Replace the original operation
    rewriter.eraseOp(op);

    LLVM_DEBUG(llvm::outs()
               << "Split NpuBlockWriteOp with data size: " << dataSizeBytes
               << " bytes into chunks of " << firstChunkSize << " and "
               << secondChunkSize << " elements\n");

    return success();
  }
};

struct AIENpuToCertPass
    : xilinx::AIEX::impl::AIENpuToCertBase<AIENpuToCertPass> {
  using Base = xilinx::AIEX::impl::AIENpuToCertBase<AIENpuToCertPass>;
  AIENpuToCertPass() = default;
  AIENpuToCertPass(const xilinx::AIEX::AIENpuToCertOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (failed(AIE::inlineWriteConfigs(moduleOp)))
      return signalPassFailure();

    // Collect all devices
    llvm::SmallVector<AIE::DeviceOp, 4> devices;
    for (auto &op : moduleOp.getBody()->getOperations()) {
      if (auto deviceOp = dyn_cast<AIE::DeviceOp>(op)) {
        devices.push_back(deviceOp);
      }
    }

    for (auto dev : devices)
      tagConfigureByName(dev);

    // Process main device first (named "main" or specified by --device-name)
    // so referenced devices' "configure" sequences can be found before they
    // get converted to cert.job
    AIE::DeviceOp mainDevice = nullptr;
    for (auto dev : devices) {
      if (dev.getSymName() == deviceName) {
        mainDevice = dev;
        break;
      }
    }

    if (mainDevice) {
      // Absorption below clones each referenced device's cert ops under the
      // main device and then erases the source device. The data those ops
      // address lives in per-device memref.globals, which the clone does not
      // carry: a uc_dma_bd whose remote_address is @config_blockwrite_data_0
      // resolves that symbol against its new parent, the main device, and
      // fails with "Global symbol not found". Move the globals up first, so
      // the clones are made against names that already resolve.
      //
      // Names collide by construction -- every device numbers its blockwrite
      // data from zero -- so rename on collision and rewrite the uses in the
      // source device before the clone is taken.
      migrateGlobalsIntoMainDevice(mainDevice, devices);
      processDevice(mainDevice);
    }

    if (!mainDevice) {
      moduleOp.emitError("no device found matching --device-name \"")
          << deviceName << "\"";
      signalPassFailure();
      return;
    }

    // Then process other devices
    for (auto dev : devices) {
      if (dev != mainDevice) {
        processDevice(dev);
      }
    }

    // Remove absorbed devices (keep only main device)
    // Referenced devices have been absorbed as cert.sections, so remove them
    llvm::SmallVector<AIE::DeviceOp, 4> devicesToRemove;
    for (auto dev : devices) {
      if (dev != mainDevice) {
        devicesToRemove.push_back(dev);
      }
    }

    for (auto dev : devicesToRemove) {
      dev.erase();
    }
  }

  // Hoist every non-main device's memref.global into the main device, keeping
  // the symbol names unique and the references inside the source device
  // pointing at the new names.
  void migrateGlobalsIntoMainDevice(AIE::DeviceOp mainDevice,
                                    ArrayRef<AIE::DeviceOp> devices) {
    Block *mainBody = mainDevice.getBody();
    if (!mainBody)
      return;
    MLIRContext *ctx = mainDevice.getContext();

    for (auto dev : devices) {
      if (dev == mainDevice)
        continue;
      llvm::SmallVector<memref::GlobalOp, 8> globals(
          dev.getOps<memref::GlobalOp>());
      for (auto global : globals) {
        StringRef oldName = global.getSymName();
        // The replacement has to be free in BOTH devices, not just in the
        // destination. Checking only the main device lets a name collide with
        // one already present in the source: renaming @x to @x_0 when the
        // source also defines @x_0 leaves two symbols of that name there, and
        // the later rename of the original @x_0 then walks both sets of uses
        // to the same global. The build still assembles -- it just addresses
        // the wrong payload.
        std::string newName = oldName.str();
        unsigned counter = 0;
        auto taken = [&](StringRef candidate) {
          if (mainDevice.lookupSymbol(candidate))
            return true;
          return candidate != oldName && dev.lookupSymbol(candidate);
        };
        while (taken(newName))
          newName = oldName.str() + "_" + std::to_string(counter++);

        if (newName != oldName) {
          // Rewrite the uses while they are still inside the source device;
          // once it is absorbed the clones carry whatever name is set here.
          if (failed(SymbolTable::replaceAllSymbolUses(
                  global, StringAttr::get(ctx, newName), dev))) {
            dev.emitError("could not rename global \"")
                << oldName << "\" while absorbing it into the main device";
            signalPassFailure();
            return;
          }
          global.setSymName(newName);
        }
        global->moveBefore(mainBody, mainBody->begin());
      }
    }
  }

  void processDevice(AIE::DeviceOp currentDevice) {
    auto moduleOp = currentDevice->getParentOfType<ModuleOp>();

    // Identify and convert referenced "configure" sequences early, before the
    // RuntimeSequenceOp conversion runs on other devices.
    if (moduleOp) {

      // Collect npu.load_pdi operations (before they are converted below)
      llvm::SmallVector<StringRef, 4> referencedDeviceSyms;
      currentDevice.walk([&](AIEX::NpuLoadPdiOp loadPdiOp) {
        if (loadPdiOp.getDeviceRef())
          referencedDeviceSyms.push_back(*loadPdiOp.getDeviceRef());
      });

      if (!referencedDeviceSyms.empty()) {
        OpBuilder builder(&getContext());

        for (StringRef refSymName : referencedDeviceSyms) {
          // Skip if section already exists
          if (currentDevice.lookupSymbol(refSymName))
            continue;

          // Find referenced device
          AIE::DeviceOp refDevice = nullptr;
          for (auto &op : moduleOp.getBody()->getOperations()) {
            if (auto dev = dyn_cast<AIE::DeviceOp>(op)) {
              if (dev.getSymName() == refSymName) {
                refDevice = dev;
                break;
              }
            }
          }

          if (!refDevice)
            continue;

          // Find the generated configuration sequence
          AIE::RuntimeSequenceOp configureSeq =
              findGeneratedConfigure(refDevice);

          if (!configureSeq)
            continue;

          // Create cert.section in current device
          builder.setInsertionPoint(currentDevice.getBody()->getTerminator());
          auto sectionOp =
              AIEX::CertSectionOp::create(builder, configureSeq.getLoc(),
                                          builder.getStringAttr(refSymName));

          Block *sectionBlock = new Block();
          sectionOp.getBody().push_back(sectionBlock);
          builder.setInsertionPointToStart(sectionBlock);

          // Create page
          auto pageOp =
              AIEX::CertPageOp::create(builder, configureSeq.getLoc());
          Block *pageBlock = new Block();
          pageOp.getBody().push_back(pageBlock);
          builder.setInsertionPointToStart(pageBlock);

          // Get unique job ID
          uint32_t maxJobId = 0;
          currentDevice.walk([&](AIEX::CertJobOp certJobOp) {
            maxJobId = std::max(maxJobId, certJobOp.getJobId());
          });

          // Create job
          auto jobOp = AIEX::CertJobOp::create(builder, configureSeq.getLoc(),
                                               maxJobId + 1);

          // Clone configure sequence body into job
          IRMapping mapper;
          configureSeq.getRegion().cloneInto(&jobOp.getBody(), mapper);

          // Ensure terminators
          AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), builder,
                                            configureSeq.getLoc());
          AIEX::CertPageOp::ensureTerminator(pageOp.getBody(), builder,
                                             configureSeq.getLoc());
          AIEX::CertSectionOp::ensureTerminator(sectionOp.getBody(), builder,
                                                configureSeq.getLoc());
        }
      }
    }

    // Inline RunOps first
    RewritePatternSet p_run_inline(&getContext());
    p_run_inline.insert<RunOpInlining>(&getContext());
    if (failed(applyPatternsGreedily(currentDevice, std::move(p_run_inline))))
      return signalPassFailure();

    // Then convert ConfigureOps to cert.section + cert.load_pdi
    RewritePatternSet p_configure(&getContext());
    p_configure.insert<ConfigureOpToCertSection>(&getContext());
    if (failed(applyPatternsGreedily(currentDevice, std::move(p_configure))))
      return signalPassFailure();

    ConversionTarget target(getContext());
    target.addIllegalOp<AIE::RuntimeSequenceOp>();

    target.addLegalOp<AIEX::CertApplyOffset57Op>();
    target.addLegalOp<AIEX::CertJobOp>();
    target.addLegalOp<AIEX::CertPageOp>();
    target.addLegalOp<AIEX::CertSectionOp>();
    target.addLegalOp<AIEX::CertLoadPdiOp>();
    target.addLegalOp<AIEX::CertMaskWrite32Op>();
    target.addLegalOp<AIEX::CertMaskPoll32Op>();
    target.addLegalOp<AIEX::CertUcDmaWriteDesSyncOp>();
    target.addLegalOp<AIEX::CertUcDmaChainOp>();
    target.addLegalOp<AIEX::CertUcDmaBdOp>();
    target.addLegalOp<AIEX::CertWrite32Op>();
    target.addLegalOp<AIEX::CertWaitTCTSOp>();
    target.addLegalOp<AIEX::ConfigureOp>(); // TODO: Convert in separate pass
    target.addLegalDialect<AIE::AIEDialect>();

    RewritePatternSet p0(&getContext());
    p0.insert<RuntimeSequenceToCertJob>(&getContext());

    if (failed(applyPartialConversion(currentDevice, target, std::move(p0))))
      return signalPassFailure();

    if (failed(prepareAddressPatchTemplates(currentDevice)))
      return signalPassFailure();

    target.addIllegalOp<AIEX::NpuAddressPatchOp>();

    // patch conversion must come before blockwrite conversion
    RewritePatternSet p1(&getContext());
    p1.insert<NpuAddressPatchToCertApplyOffset57>(&getContext());

    if (failed(applyPartialConversion(currentDevice, target, std::move(p1))))
      return signalPassFailure();

    // Split oversized NpuBlockWriteOps before lowering them to cert ops
    {
      RewritePatternSet p(&getContext());
      p.insert<SplitNpuBlockWriteOpPattern>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(p))))
        signalPassFailure();
    }

    target.addIllegalOp<AIEX::NpuBlockWriteOp>();
    target.addIllegalOp<AIEX::NpuBlockWriteValuesOp>();
    target.addIllegalOp<AIEX::NpuMaskWrite32Op>();
    target.addIllegalOp<AIEX::NpuMaskPollOp>();
    target.addIllegalOp<AIEX::NpuSyncOp>();
    target.addIllegalOp<AIEX::NpuWrite32Op>();
    target.addIllegalOp<AIEX::NpuLoadPdiOp>();

    RewritePatternSet p2(&getContext());
    p2.insert<NpuBlockWriteToCertUcDma>(&getContext());
    p2.insert<NpuMaskWrite32ToCertMaskWrite32>(&getContext());
    p2.insert<NpuMaskPollToCertMaskPoll32>(&getContext());
    p2.insert<NpuWrite32ToCertWrite32>(&getContext());
    p2.insert<NpuSyncToCertWaitTCTS>(&getContext());
    p2.insert<NpuLoadPdiToCertLoadPdi>(&getContext());

    if (failed(applyPartialConversion(currentDevice, target, std::move(p2))))
      return signalPassFailure();

    // Run the greedy driver with no patterns, purely for its DCE: p2
    // leaves each converted blockwrite's memref.get_global (and any constant
    // feeding it) dead, since the new op references the global by symbol
    // rather than by SSA value. CertJobOp's verifier rejects those ops
    RewritePatternSet cleanup(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(cleanup))))
      return signalPassFailure();

    // Convert referenced devices to cert.sections
    // Now that cert.load_pdi ops exist, find referenced devices and convert
    // their cert jobs into sections in this device
    {
      OpBuilder builder(&getContext());

      // Find all cert.load_pdi in current device
      llvm::SetVector<StringRef> referencedDeviceNames;
      currentDevice.walk([&](AIEX::CertLoadPdiOp loadPdiOp) {
        StringRef refSymName = loadPdiOp.getSymbol();
        if (!currentDevice.lookupSymbol(refSymName))
          referencedDeviceNames.insert(refSymName);
      });

      // For each referenced device, create a section with its cert jobs
      for (StringRef refSymName : referencedDeviceNames) {
        // Find the referenced device
        AIE::DeviceOp refDevice = nullptr;
        for (auto &op : moduleOp.getBody()->getOperations()) {
          if (auto dev = dyn_cast<AIE::DeviceOp>(op)) {
            if (dev.getSymName() == refSymName) {
              refDevice = dev;
              break;
            }
          }
        }

        if (!refDevice)
          continue;

        // Find all cert.page ops in the referenced device that are direct
        // children
        llvm::SmallVector<AIEX::CertPageOp, 4> certPages;
        for (auto &op : refDevice.getBody()->getOperations()) {
          if (auto pageOp = dyn_cast<AIEX::CertPageOp>(op)) {
            certPages.push_back(pageOp);
          }
        }

        if (certPages.empty())
          continue;

        // Create cert.section in current device
        builder.setInsertionPoint(currentDevice.getBody()->getTerminator());
        auto sectionOp = AIEX::CertSectionOp::create(
            builder, refDevice.getLoc(), builder.getStringAttr(refSymName));

        Block *sectionBlock = new Block();
        sectionOp.getBody().push_back(sectionBlock);

        // Clone all cert.page operations into the section
        IRMapping mapper;
        for (auto pageOp : certPages) {
          builder.setInsertionPointToEnd(sectionBlock);
          builder.clone(*pageOp.getOperation(), mapper);
        }

        // Ensure section terminator
        AIEX::CertSectionOp::ensureTerminator(sectionOp.getBody(), builder,
                                              refDevice.getLoc());
      }
    }

    // Clean up unused block arguments from cert.job operations
    currentDevice.walk([&](AIEX::CertJobOp jobOp) {
      Block &jobBlock = jobOp.getBody().front();

      // Remove unused block arguments (in reverse order)
      for (int i = jobBlock.getNumArguments() - 1; i >= 0; --i) {
        BlockArgument arg = jobBlock.getArgument(i);
        if (arg.use_empty()) {
          jobBlock.eraseArgument(i);
        }
      }
    });
  }
};

} // namespace

// Symbols already charged to the page currently being measured. The emitter
// writes one labelled block per chain and per payload global no matter how
// many instructions name it, and the assembler sizes the page from those
// labels -- so a symbol must be counted exactly once per page. Chains and
// payloads are tracked separately because two distinct chains may point at one
// shared global: their BDs are distinct labels and each costs 16 bytes, while
// the global they share is laid down once.
// Keyed on the resolved definitions rather than the reference strings, so the
// identity is the label the emitter will actually write.
struct ChainDataCostState {
  DenseSet<Operation *> countedChains;
  DenseSet<Operation *> countedPayloads;
};

// The page's data section carries the uC-DMA BDs of any chain referenced by
// UC_DMA_WRITE_DES or UC_DMA_WRITE_DES_SYNC, plus the globals those BDs point
// at. Both opcodes charge identical data; only their text size differs
// (ISA_OPSIZE_UC_DMA_WRITE_DES = 0x08 vs _SYNC = 0x04).
//
// Re-referencing a chain costs only the instruction, never the data again.
// Charging it twice inflated the estimate enough to reject pages the assembler
// accepts: two `uc_dma_write_des(@c)` over a 1200-word payload estimated 9700
// bytes against a real 4880.
static void addChainDataCost(StringRef sym_name, AIE::DeviceOp deviceOp,
                             uint32_t &data_cost, ChainDataCostState &seen) {
  auto chain = dyn_cast_if_present<AIEX::CertUcDmaChainOp>(
      deviceOp.lookupSymbol(sym_name));
  if (!chain)
    return;
  if (!seen.countedChains.insert(chain.getOperation()).second)
    return;
  for (auto bdOp : chain.getBody().front().getOps<AIEX::CertUcDmaBdOp>()) {
    data_cost += 16; // bd op
    StringRef data_sym_name = bdOp.getRemoteAddress();
    auto global = dyn_cast_if_present<memref::GlobalOp>(
        deviceOp.lookupSymbol(data_sym_name));
    if (!global)
      continue;
    if (!seen.countedPayloads.insert(global.getOperation()).second)
      continue;
    auto initVal = global.getInitialValue();
    if (!initVal)
      continue;
    auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
    if (!data)
      continue;
    data_cost += data.getNumElements() * 4; // 4 bytes per element
  }
}

// Instruction sizes are the authoritative ISA_OPSIZE_* values from
// AIEBU's specification/aie2ps/isa_stubs.h -- the same table the
// assembler encodes against -- so the running total here matches the bytes
// emitJob actually produces. Keep the two in sync when adding a cert op:
// an op the emitter can emit but this function ignores is counted as free,
// which makes the page look smaller than it is.
static void updateCostForOp(Operation &o, AIE::DeviceOp deviceOp,
                            uint32_t &text_cost, uint32_t &data_cost,
                            ChainDataCostState &seen) {
  // Several distinct op kinds share an encoded instruction size below --
  // an instruction-size table, not a copy-paste clone.
  // NOLINTBEGIN(bugprone-branch-clone)
  if (isa<AIEX::CertLocalBarrierOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_LOCAL_BARRIER
  } else if (isa<AIEX::CertRemoteBarrierOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_REMOTE_BARRIER
  } else if (isa<AIEX::CertRelAcqSyncOp>(o)) {
    text_cost += 12; // encoded REL_ACQ_SYNC instruction size
  } else if (isa<AIEX::CertWaitTCTSOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_WAIT_TCTS
  } else if (isa<AIEX::CertMaskWrite32Op, AIEX::CertMaskPoll32Op>(o)) {
    text_cost += 16; // ISA_OPSIZE_MASK_WRITE_32 / MASK_POLL_32
  } else if (isa<AIEX::CertWrite32Op>(o)) {
    text_cost += 12; // ISA_OPSIZE_WRITE_32
  } else if (isa<AIEX::CertApplyOffset57Op>(o)) {
    text_cost += 8; // ISA_OPSIZE_APPLY_OFFSET_57
  } else if (isa<AIEX::CertNopOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_NOP
  } else if (isa<AIEX::CertPreemptOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_PREEMPT
  } else if (isa<AIEX::CertLoadPdiOp>(o)) {
    text_cost += 12; // ISA_OPSIZE_LOAD_PDI
    // NOLINTEND(bugprone-branch-clone)
  } else if (auto syncOp = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC
    addChainDataCost(syncOp.getSymbol(), deviceOp, data_cost, seen);
  } else if (auto desOp = dyn_cast<AIEX::CertUcDmaWriteDesOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_UC_DMA_WRITE_DES
    addChainDataCost(desOp.getSymbol(), deviceOp, data_cost, seen);
  } else if (isa<AIEX::CertWaitUcDmaOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_WAIT_UC_DMA
  } else if (isa<AIEX::CertRead32Op>(o)) {
    text_cost += 8; // ISA_OPSIZE_READ_32
  } else if (isa<AIEX::CertRead32DOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_READ_32_D
  } else if (isa<AIEX::CertWrite32DOp>(o)) {
    text_cost += 12; // ISA_OPSIZE_WRITE_32_D
  } else if (isa<AIEX::CertPoll32Op>(o)) {
    text_cost += 12; // ISA_OPSIZE_POLL_32
  } else if (isa<AIEX::CertMaskPoll32Op>(o)) {
    text_cost += 16; // ISA_OPSIZE_MASK_POLL_32
  } else if (isa<AIEX::CertSaveTimestampsOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_SAVE_TIMESTAMPS
  } else if (isa<AIEX::CertSaveRegisterOp>(o)) {
    text_cost += 12; // ISA_OPSIZE_SAVE_REGISTER
  }
}

// True if `o` reads or writes any job-private local register r0..r7.
static bool opUsesLocalRegister(Operation &o) {
  auto isLocal = [](uint32_t r) { return r < 8; };
  if (auto op = dyn_cast<AIEX::CertRead32Op>(&o))
    return isLocal(op.getValue());
  if (auto op = dyn_cast<AIEX::CertRead32DOp>(&o))
    return isLocal(op.getAddress()) || isLocal(op.getValue());
  if (auto op = dyn_cast<AIEX::CertUcDmaWriteDesOp>(&o))
    return isLocal(op.getWaitHandle());
  if (auto op = dyn_cast<AIEX::CertWaitUcDmaOp>(&o))
    return isLocal(op.getWaitHandle());
  if (auto op = dyn_cast<AIEX::CertWrite32DOp>(&o))
    return (op.getAddressIsReg() && isLocal(op.getAddress())) ||
           (op.getValueIsReg() && isLocal(op.getValue()));
  return false;
}

// isa-spec.yaml -- "Register r0..r7 are local registers which are private
// to each job." SplitCertPageOpPattern rebuilds the two halves as two DISTINCT
// CertJobOps via cloneJobRange, so a local-register value produced on one side
// is not merely deferred across the .eop -- it is destroyed. That is different
// from the backward-dependency argument documented below in this file's split
// logic, which is sound for TCTs and memory: those survive because they live
// outside the job-private register file.
// Globals r8..r23 are "shared among all the jobs" (isa-spec.yaml) and
// survive a split, so the rule is register-class dependent. This takes the
// conservative form: a job that touches ANY local register is not a split
// candidate at all.
static bool jobUsesLocalRegister(AIEX::CertJobOp job) {
  if (job.getBody().empty())
    return false;
  for (Operation &o : job.getBody().front())
    if (opUsesLocalRegister(o))
      return true;
  return false;
}

// The job-private local registers r0..r7 that `o` reads, and those it writes.
// A register that is both read and written (cert.read32_d(1, 1)) appears in
// both. Globals r8..r23 are deliberately excluded: they survive a job
// boundary, so they never constrain a rewrite that splits a job.
static void collectLocalRegisterDefsUses(Operation &o,
                                         SmallVectorImpl<uint32_t> &uses,
                                         SmallVectorImpl<uint32_t> &defs) {
  auto addLocal = [](SmallVectorImpl<uint32_t> &v, uint32_t r) {
    if (r < 8)
      v.push_back(r);
  };
  if (auto op = dyn_cast<AIEX::CertRead32Op>(&o))
    addLocal(defs, op.getValue());
  else if (auto op = dyn_cast<AIEX::CertRead32DOp>(&o)) {
    addLocal(uses, op.getAddress());
    addLocal(defs, op.getValue());
  } else if (auto op = dyn_cast<AIEX::CertUcDmaWriteDesOp>(&o))
    addLocal(defs, op.getWaitHandle());
  else if (auto op = dyn_cast<AIEX::CertWaitUcDmaOp>(&o))
    addLocal(uses, op.getWaitHandle());
  else if (auto op = dyn_cast<AIEX::CertWrite32DOp>(&o)) {
    if (op.getAddressIsReg())
      addLocal(uses, op.getAddress());
    if (op.getValueIsReg())
      addLocal(uses, op.getValue());
  }
}

// The first local register in `ops` that is read before anything in `ops`
// writes it -- i.e. one whose reaching definition lies earlier in the job,
// ahead of wherever `ops` was cut from. Because r0..r7 are private to a job,
// moving `ops` into a NEW job destroys that value, so such a register makes
// the cut illegal. Returns nullopt when `ops` is self-contained.
static std::optional<uint32_t>
findLocalRegisterCrossingIn(ArrayRef<Operation *> ops) {
  llvm::SmallDenseSet<uint32_t> definedHere;
  SmallVector<uint32_t> uses, defs;
  for (Operation *o : ops) {
    uses.clear();
    defs.clear();
    collectLocalRegisterDefsUses(*o, uses, defs);
    for (uint32_t r : uses)
      if (!definedHere.contains(r))
        return r;
    definedHere.insert(defs.begin(), defs.end());
  }
  return std::nullopt;
}

static uint32_t estimateCost(AIEX::CertPageOp op, uint32_t split_target,
                             AIEX::CertJobOp &split_job,
                             Block::iterator &split_iter,
                             bool &found_split_point) {
  uint32_t text_cost = 32; // page header: `.align 16` padding plus `.eop`
  uint32_t data_cost = 0;
  found_split_point = false;
  AIE::DeviceOp deviceOp = op->getParentOfType<AIE::DeviceOp>();
  // Per-page: the assembler emits each chain/payload label once per page.
  ChainDataCostState seen;

  for (auto job : op.getBody().front().getOps<AIEX::CertJobOp>()) {
    // A job using job-private local registers cannot be split (its
    // local-register state does not survive across the new .eop), so it is
    // never a split candidate below. Cost accumulation still runs
    // unconditionally so the returned total remains accurate.
    bool splittable = !jobUsesLocalRegister(job);
    // START_JOB precedes the body, so it is already spent at every candidate
    // split point inside this job; END_JOB is charged after the body below.
    text_cost += cert_job_start_cost;
    for (auto &o : job.getBody().front().getOperations()) {
      Block::iterator current(&o);
      if (splittable && !found_split_point && !isa<AIE::EndOp>(o) &&
          current != job.getBody().front().begin() &&
          (text_cost + data_cost) >= split_target) {
        split_job = job;
        split_iter = current;
        found_split_point = true;
      }

      updateCostForOp(o, deviceOp, text_cost, data_cost, seen);

      if (splittable && !found_split_point &&
          (text_cost + data_cost) >= split_target) {
        Block::iterator next = current;
        ++next;
        if (next != job.getBody().front().end() && !isa<AIE::EndOp>(*next)) {
          split_job = job;
          split_iter = next;
          found_split_point = true;
        } else if (!isa<AIE::EndOp>(o) &&
                   current != job.getBody().front().begin()) {
          // Cutting after the last real op would leave an empty second half.
          // Cut before it; exclude the terminator to guarantee progress.
          split_job = job;
          split_iter = current;
          found_split_point = true;
        }
      }
    }
    text_cost += cert_job_end_cost;
  }
  return text_cost + data_cost;
}

// A cut before flat index p separates a span [lo, hi] when lo < p <= hi.
// Local barrier participants must share a page to avoid a firmware hang.
// Payload patches must share a page with all later uC-DMA pushes: each page has
// its own payload copy, so a push on another page would use unpatched addresses.
enum class SpanKind { LocalBarrier, PatchedPayload };

struct IndivisibleSpan {
  int lo, hi;
  SpanKind kind;
};

// Index non-terminator ops in textual order and record indivisible spans.
static void computeIndivisibleSpans(AIEX::CertPageOp page,
                                    std::map<Operation *, int> &flatIndex,
                                    SmallVectorImpl<IndivisibleSpan> &spans) {
  AIE::DeviceOp deviceOp = page->getParentOfType<AIE::DeviceOp>();
  // local_barrier_id -> [min,max] flat index of its participants.
  std::map<int, std::pair<int, int>> barriers;
  // Payload symbol -> earliest patch; patches accumulate and survive pushes.
  llvm::StringMap<int> firstPatch;

  // Collect payload symbols from the chain's BDs.
  auto payloadsOf = [&](StringRef chainSym) {
    SmallVector<StringRef> payloads;
    auto chain = dyn_cast_if_present<AIEX::CertUcDmaChainOp>(
        deviceOp ? deviceOp.lookupSymbol(chainSym) : nullptr);
    if (!chain || chain.getBody().empty())
      return payloads;
    for (auto bd : chain.getBody().front().getOps<AIEX::CertUcDmaBdOp>())
      payloads.push_back(bd.getRemoteAddress());
    return payloads;
  };

  int idx = 0;
  for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
    for (auto &o : job.getBody().front().getOperations()) {
      if (isa<AIE::EndOp>(o))
        continue;
      flatIndex[&o] = idx;
      if (auto bar = dyn_cast<AIEX::CertLocalBarrierOp>(o)) {
        int id = static_cast<int>(bar.getLocalBarrierId());
        auto it = barriers.find(id);
        if (it == barriers.end())
          barriers[id] = {idx, idx};
        else {
          it->second.first = std::min(it->second.first, idx);
          it->second.second = std::max(it->second.second, idx);
        }
      } else if (auto patch = dyn_cast<AIEX::CertApplyOffset57Op>(o)) {
        // Keep all patches preceding a push on the same page.
        firstPatch.try_emplace(patch.getSymbol(), idx);
      } else {
        StringRef chainSym;
        if (auto sync = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(o))
          chainSym = sync.getSymbol();
        else if (auto async = dyn_cast<AIEX::CertUcDmaWriteDesOp>(o))
          chainSym = async.getSymbol();
        if (!chainSym.empty())
          for (StringRef payload : payloadsOf(chainSym)) {
            auto it = firstPatch.find(payload);
            if (it == firstPatch.end())
              continue;
            spans.push_back({it->second, idx, SpanKind::PatchedPayload});
          }
      }
      ++idx;
    }
  }
  for (auto &e : barriers)
    spans.push_back({e.second.first, e.second.second, SpanKind::LocalBarrier});
}

// Return whether the cut separates a span, optionally reporting its kind.
static bool cutSeparatesSpan(ArrayRef<IndivisibleSpan> spans, int p,
                             SpanKind *kind = nullptr) {
  for (const IndivisibleSpan &s : spans) {
    if (s.lo < p && s.hi >= p) {
      if (kind)
        *kind = s.kind;
      return true;
    }
  }
  return false;
}

// Search for a split point whose cut keeps every indivisible span intact.
// Two kinds of position qualify, and the splitter can execute either:
//   - interior cut: at least one op of the same job precedes the position, so
//     the job is partitioned into an earlier and a later half;
//   - job-boundary cut: the position is a job's first op while an earlier job
//     already carries content, so that job moves whole to the later page and no
//     job is partitioned at all.
// Only the first kind is subject to the local-register rule, since only it
// tears a job in two: a job using r0..r7 offers no interior cut, but a cut on
// the boundary ahead of it is still available.
// What is never legal is a cut ahead of everything on the page: it leaves the
// earlier page empty, so the "split" makes no progress and the greedy driver
// repeats it forever. Among the legal candidates pick the one whose
// earlier-page cost is closest to `split_target`. Returns false if no legal
// split point exists (the caller then leaves the page intact).
static bool findLegalSplitPoint(AIEX::CertPageOp page, uint32_t split_target,
                                const std::map<Operation *, int> &flatIndex,
                                ArrayRef<IndivisibleSpan> spans,
                                AIEX::CertJobOp &split_job,
                                Block::iterator &split_iter) {
  AIE::DeviceOp deviceOp = page->getParentOfType<AIE::DeviceOp>();
  uint32_t text_cost = 32; // page header, mirrors estimateCost
  uint32_t data_cost = 0;
  ChainDataCostState seen; // mirrors estimateCost
  bool found = false;
  uint32_t best_delta = 0;
  // Whether some job already visited holds a real op, i.e. whether cutting at
  // the current job's first op would leave content on the earlier page.
  bool sawOpInEarlierJob = false;
  for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
    bool splittable = !jobUsesLocalRegister(job);
    text_cost += cert_job_start_cost; // mirrors estimateCost
    bool sawOpInJob = false;
    for (auto &o : job.getBody().front().getOperations()) {
      if (isa<AIE::EndOp>(o))
        continue;
      Block::iterator current(&o);
      // Candidate: cut before `o`, legal as an interior cut when an op of this
      // job precedes it, or as a job-boundary cut when an earlier job does.
      // Admitting the latter matters for a page built of small jobs whose sizes
      // only sum over the limit: no job has an interior point worth cutting at,
      // yet a cut between two whole jobs splits it perfectly well.
      //
      // The local-register constraint gates the interior case only: an
      // interior cut partitions this job, and its local-register state does not
      // survive the new .eop. A boundary cut moves the job whole to the later
      // page, so an unsplittable job is still a legal place to cut *ahead of*.
      if (sawOpInJob ? splittable : sawOpInEarlierJob) {
        auto fi = flatIndex.find(&o);
        int p = fi == flatIndex.end() ? 0 : fi->second;
        if (!cutSeparatesSpan(spans, p)) {
          uint32_t cost = text_cost + data_cost;
          uint32_t delta =
              cost > split_target ? cost - split_target : split_target - cost;
          if (!found || delta < best_delta) {
            found = true;
            best_delta = delta;
            split_job = job;
            split_iter = current;
          }
        }
      }
      updateCostForOp(o, deviceOp, text_cost, data_cost, seen);
      sawOpInJob = true;
    }
    text_cost += cert_job_end_cost; // mirrors estimateCost
    sawOpInEarlierJob |= sawOpInJob;
  }
  return found;
}

namespace {

// Pattern to isolate load_pdi and preempt operations into their own job and
// page According to CERT spec: "load_pdi and preempt should take one whole job
// which in turn should take one whole page"
struct IsolateFullPageOpsPattern : OpRewritePattern<AIEX::CertJobOp> {
  // See SplitCertPageOpPattern for why an error needs an out-param: a pattern
  // returning failure() is ordinary control flow and does not fail the pass.
  IsolateFullPageOpsPattern(MLIRContext *context, bool *hadError)
      : OpRewritePattern(context), hadError(hadError) {}

  // Create an empty, terminated cert.page directly after `after`, in the same
  // block. Staying in `after`'s block is what keeps a page that was moved into
  // a cert.attach_to_group by placement lowering on its uC.
  static AIEX::CertPageOp makePageAfter(PatternRewriter &rewriter,
                                        Operation *after, Location loc) {
    rewriter.setInsertionPointAfter(after);
    auto page = AIEX::CertPageOp::create(rewriter, loc);
    page.getBody().push_back(new Block());
    AIEX::CertPageOp::ensureTerminator(page.getBody(), rewriter, loc);
    return page;
  }

  LogicalResult matchAndRewrite(AIEX::CertJobOp jobOp,
                                PatternRewriter &rewriter) const override {

    // Find first load_pdi or preempt in this job
    Operation *fullPageOp = nullptr;
    Block::iterator fullPageOpIter;
    for (Block::iterator it = jobOp.getBody().front().begin();
         it != jobOp.getBody().front().end(); ++it) {
      if (isa<AIEX::CertLoadPdiOp, AIEX::CertPreemptOp>(*it)) {
        fullPageOp = &*it;
        fullPageOpIter = it;
        break;
      }
    }

    if (!fullPageOp)
      return failure(); // No full-page op in this job

    auto parentPage = jobOp->getParentOfType<AIEX::CertPageOp>();
    if (!parentPage)
      return failure(); // No parent page (unusual but skip)

    // Real ops in this job (the terminator does not count).
    size_t opCount = 0;
    for (Operation &op : jobOp.getBody().front().getOperations())
      if (!isa<AIE::EndOp>(op))
        opCount++;

    // Pages execute in order, so the rewrite has to preserve this job's
    // position among its *siblings*, not just relative to the parent page:
    // anchoring every new page on the parent page's boundary hoists the jobs
    // that surround this one across the isolated load_pdi/preempt. Nothing
    // downstream checks op order, so that reordering is silent.
    SmallVector<Operation *> leadingSiblings, trailingSiblings;
    bool pastSelf = false;
    for (Operation &op : parentPage.getBody().front()) {
      if (isa<AIE::EndOp>(op))
        continue;
      if (&op == jobOp.getOperation()) {
        pastSelf = true;
        continue;
      }
      (pastSelf ? trailingSiblings : leadingSiblings).push_back(&op);
    }

    // Already properly isolated: sole op of the sole job of its page.
    if (opCount == 1 && leadingSiblings.empty() && trailingSiblings.empty())
      return failure();

    auto loc = jobOp.getLoc();

    // The parent page stays where it is and keeps the leading siblings; every
    // page created below is chained after it, in program order.
    Operation *anchor = parentPage.getOperation();

    if (opCount == 1) {
      // The job is exactly the full-page op: it just needs a page of its own.
      auto isolatedPage = makePageAfter(rewriter, anchor, loc);
      jobOp->moveBefore(isolatedPage.getBody().front().getTerminator());
      anchor = isolatedPage;
    } else {
      // Mixed job: split it into up to three jobs, each on its own page, in
      // source order (ops before the full-page op, it, then the ops after).
      auto parentDevice = jobOp->getParentOfType<AIE::DeviceOp>();
      uint32_t maxJobId = 0;
      parentDevice.walk([&](AIEX::CertJobOp certJobOp) {
        maxJobId = std::max(maxJobId, certJobOp.getJobId());
      });
      uint32_t beforeJobId = jobOp.getJobId();
      uint32_t fullPageJobId = maxJobId + 1;
      uint32_t afterJobId = maxJobId + 2;

      SmallVector<Operation *> beforeOps, afterOps;
      for (Block::iterator it = jobOp.getBody().front().begin();
           it != fullPageOpIter; ++it)
        if (!isa<AIE::EndOp>(*it))
          beforeOps.push_back(&*it);
      Block::iterator afterStart = std::next(fullPageOpIter);
      for (Block::iterator it = afterStart; it != jobOp.getBody().front().end();
           ++it)
        if (!isa<AIE::EndOp>(*it))
          afterOps.push_back(&*it);

      // Isolation rebuilds this job as up to three DISTINCT jobs, so -- exactly
      // as for a page split -- a job-private local register r0..r7 written
      // before the full-page op is destroyed before the trailing ops can read
      // it. The size-driven splitter avoids this by declining to cut; isolation
      // is mandatory ("load_pdi and preempt should take one whole job"), so
      // there is nothing to decline and the only sound answer is to reject.
      //
      // The test is on values that actually cross the boundary rather than on
      // any local-register use at all: a local whose live range is wholly inside
      // one of the halves is unaffected, and the blunt form would reject working
      // designs that the splitter merely leaves unsplit.
      if (auto crossing = findLocalRegisterCrossingIn(afterOps)) {
        if (hadError)
          *hadError = true;
        jobOp.emitError()
            << "job " << jobOp.getJobId() << " mixes "
            << (isa<AIEX::CertLoadPdiOp>(fullPageOp) ? "cert.load_pdi"
                                                     : "cert.preempt")
            << " with other operations and must be broken into separate jobs, "
               "but register r"
            << *crossing
            << " is written before it and read after it. Registers r0..r7 are "
               "job-private, so that value does not survive; move the "
               "local-register sequence to one side of the "
            << (isa<AIEX::CertLoadPdiOp>(fullPageOp) ? "cert.load_pdi"
                                                     : "cert.preempt")
            << ", or use a global register r8..r23";
        return failure();
      }

      auto emitJobPage = [&](uint32_t jobId, ArrayRef<Operation *> ops) {
        auto page = makePageAfter(rewriter, anchor, loc);
        rewriter.setInsertionPoint(page.getBody().front().getTerminator());
        auto job = AIEX::CertJobOp::create(rewriter, loc, jobId);
        job.getBody().push_back(new Block());
        AIEX::CertJobOp::ensureTerminator(job.getBody(), rewriter, loc);
        for (Operation *op : ops)
          op->moveBefore(job.getBody().front().getTerminator());
        anchor = page;
      };

      if (!beforeOps.empty())
        emitJobPage(beforeJobId, beforeOps);
      emitJobPage(fullPageJobId, {fullPageOp});
      if (!afterOps.empty())
        emitJobPage(afterJobId, afterOps);

      rewriter.eraseOp(jobOp);
    }

    // The trailing siblings cannot stay on the parent page, which now precedes
    // the isolated page, so they move to a page of their own behind it.
    if (!trailingSiblings.empty()) {
      auto tailPage = makePageAfter(rewriter, anchor, loc);
      for (Operation *op : trailingSiblings)
        op->moveBefore(tailPage.getBody().front().getTerminator());
    }

    // If the isolated job was the first job on the page, moving or rebuilding
    // it above can leave the original page with only its implicit terminator.
    // Remove that rewrite artifact rather than emitting a stray empty .eop.
    if (llvm::all_of(parentPage.getBody().front(),
                     [](Operation &op) { return isa<AIE::EndOp>(op); }))
      rewriter.eraseOp(parentPage);

    return success();
  }

private:
  bool *hadError;
};

struct SplitCertPageOpPattern : OpRewritePattern<AIEX::CertPageOp> {
  // `hadError` is raised when the pattern emits an error diagnostic. A rewrite
  // pattern returning failure() is ordinary control flow -- it does NOT fail
  // the pass -- so without this the "cannot split" error below would print and
  // the oversized page would still reach codegen. The pass checks the flag
  // after applyPatternsGreedily and calls signalPassFailure().
  SplitCertPageOpPattern(MLIRContext *context, bool *hadError)
      : OpRewritePattern(context), hadError(hadError) {}

  LogicalResult matchAndRewrite(AIEX::CertPageOp op,
                                PatternRewriter &rewriter) const override {

    constexpr uint32_t split_threshold = cert_page_size;

    AIEX::CertJobOp split_job;
    Block::iterator split_iter;
    bool found_split_point = false;
    uint32_t cost = estimateCost(op, cert_page_size / 2, split_job, split_iter,
                                 found_split_point);
    LLVM_DEBUG(llvm::outs()
               << "Estimate cost for page: " << " is " << cost << "\n");

    if (cost < split_threshold)
      return failure();

    bool isImplicit = op->hasAttr("cert.implicit");

    std::map<Operation *, int> flatIndex;
    SmallVector<IndivisibleSpan> spans;
    computeIndivisibleSpans(op, flatIndex, spans);

    // estimateCost's search is greedy: it retains only the first position at
    // which the running cost crosses split_target and never backtracks, and it
    // only ever considers positions interior to the one job the crossing
    // happened in. So a page can be reported as offering no split point when a
    // cut is plainly available -- the crossing op alone in its job while an
    // earlier job has several ops, or a page of small jobs whose sizes merely
    // sum over the limit, where the cut belongs on a job boundary. Fall back to
    // the exhaustive search, which considers every position in the page and
    // keeps the legal one closest to split_target, before giving up.
    if (!found_split_point)
      found_split_point = findLegalSplitPoint(op, cert_page_size / 2, flatIndex,
                                              spans, split_job, split_iter);

    // Over the split trigger but with nowhere to cut. Every cut has to leave at
    // least one op on each side, so a page whose bulk sits in a single one-op
    // job offers none. Leave the page alone -- but if it is also over the hard
    // limit, it will not load, so say so instead of emitting a page the
    // firmware will reject.
    if (!found_split_point) {
      if (cost > cert_page_limit) {
        // Name the local-register constraint explicitly when that is what
        // ruled out every candidate cut; otherwise the page
        // genuinely offers no interior split point.
        AIEX::CertJobOp blocker;
        for (auto job : op.getBody().front().getOps<AIEX::CertJobOp>())
          if (jobUsesLocalRegister(job)) {
            blocker = job;
            break;
          }
        if (blocker)
          op.emitError()
              << "cert.page is an estimated " << cost << " bytes, over the "
              << cert_page_limit
              << "-byte microcontroller page limit, and cannot be split: job "
              << blocker.getJobId()
              << " uses job-private local registers r0..r7, whose values do "
                 "not survive a page split (the two halves become distinct "
                 "jobs); move the local-register sequence into a smaller job, "
                 "or use global registers r8..r23";
        else
          op.emitError() << "cert.page is an estimated " << cost
                         << " bytes, over the " << cert_page_limit
                         << "-byte microcontroller page limit, and offers no "
                            "split point (a split must leave at least one op on "
                            "each page); break the oversized job into smaller "
                            "jobs";
        *hadError = true;
      }
      return failure();
    }

    // Retarget cuts that separate a barrier group or a payload patch and push.
    if (!spans.empty()) {
      auto fi = flatIndex.find(&*split_iter);
      int p = fi == flatIndex.end() ? 0 : fi->second;
      SpanKind kind = SpanKind::LocalBarrier;
      if (cutSeparatesSpan(spans, p, &kind)) {
        if (!findLegalSplitPoint(op, cert_page_size / 2, flatIndex, spans,
                                 split_job, split_iter)) {
          // Keep indivisible pages whole while they fit the hardware limit.
          // Set hadError on overflow so the pass fails, not just this rewrite.
          if (cost > cert_page_limit) {
            StringRef what =
                kind == SpanKind::LocalBarrier
                    ? "a local_barrier participant set"
                    : "a cert.apply_offset_57 from the uC-DMA push that "
                      "consumes the payload it patches";
            StringRef implicitRemedy =
                kind == SpanKind::LocalBarrier
                    ? "co-locate the barrier participants"
                    : "co-locate each apply_offset_57 with its push";
            StringRef explicitRemedy =
                kind == SpanKind::LocalBarrier
                    ? "regroup its local_barrier participants"
                    : "regroup its apply_offset_57 ops and their pushes";
            if (isImplicit)
              op.emitError()
                  << "implicit cert.page is an estimated " << cost
                  << " bytes, over the " << cert_page_limit
                  << "-byte microcontroller page limit, and cannot be split "
                     "without separating "
                  << what << " across pages; insert an explicit cert.page "
                  << "boundary to " << implicitRemedy;
            else
              op.emitError()
                  << "cert.page is an estimated " << cost << " bytes, over the "
                  << cert_page_limit
                  << "-byte microcontroller page limit, and cannot be split "
                     "without separating "
                  << what << " across pages; reduce the page's contents or "
                  << explicitRemedy << " so that a legal split point exists";
            *hadError = true;
          }
          return failure();
        }
      }
    }

    // Splitting an implicit page turns jobs that were meant to run
    // cooperatively on one page into strictly sequential pages. Warn so the user
    // can insert an explicit boundary if the serialization is unintended.
    if (isImplicit)
      op.emitRemark()
          << "auto-split of implicit cert.page serializes cooperatively-"
             "scheduled jobs into separate sequential pages; insert an "
             "explicit "
             "cert.page boundary to control where the split occurs";

    auto loc = op.getLoc();

    // Preserve textual order so producers precede consumers across pages.
    // TCTs and memory survive page boundaries; local-register and payload
    // constraints are checked above.
    //
    // split_iter at split_job's first op means the cut falls on a job boundary:
    // split_job is not partitioned at all, it moves whole to the later page and
    // the earlier page is made of the jobs ahead of it. That is a real split
    // only because some earlier job carries content; the searches guarantee it,
    // and cutting ahead of everything would leave an empty earlier page and no
    // progress. Otherwise the cut is interior to split_job and, again by
    // construction, at least one op stays behind on the earlier page. Either
    // way the earlier page is non-empty and the split strictly makes progress.
    bool jobBoundaryCut = split_iter == split_job.getBody().front().begin();
    [[maybe_unused]] auto earlierJobHasRealOp = [&]() {
      for (auto job : op.getBody().front().getOps<AIEX::CertJobOp>()) {
        if (job == split_job)
          break;
        for (auto &o : job.getBody().front().getOperations())
          if (!isa<AIE::EndOp>(o))
            return true;
      }
      return false;
    };
    assert((!jobBoundaryCut || earlierJobHasRealOp()) &&
           "a job-boundary cut needs a preceding job with content, else the "
           "earlier page is empty and the split makes no progress");

    // The second half of a split job takes the next ID, which may collide
    // with another job on its page; allocateJobIds resolves that after all
    // splitting is done.
    auto cloneJobRange = [&](AIEX::CertJobOp sourceJob, uint32_t jobId,
                             Block::iterator begin, Block::iterator end) {
      auto newJobOp = AIEX::CertJobOp::create(rewriter, loc, jobId);
      Block *newJobBlock = new Block();
      newJobOp.getBody().push_back(newJobBlock);

      IRMapping mapper;
      Block &sourceBlock = sourceJob.getBody().front();
      for (BlockArgument arg : sourceBlock.getArguments()) {
        BlockArgument newArg =
            newJobBlock->addArgument(arg.getType(), arg.getLoc());
        mapper.map(arg, newArg);
      }

      rewriter.setInsertionPointToStart(newJobBlock);
      for (Block::iterator oi = begin; oi != end; ++oi) {
        if (!isa<AIE::EndOp>(*oi))
          rewriter.clone(*oi, mapper);
      }
      AIEX::CertJobOp::ensureTerminator(newJobOp.getBody(), rewriter, loc);
    };

    rewriter.setInsertionPoint(op);
    auto newPageOp0 = AIEX::CertPageOp::create(rewriter, loc);
    Block *newPageBlock0 = new Block();
    newPageOp0.getBody().push_back(newPageBlock0);

    auto newPageOp1 = AIEX::CertPageOp::create(rewriter, loc);
    Block *newPageBlock1 = new Block();
    newPageOp1.getBody().push_back(newPageBlock1);

    // Propagate the implicit tag so a later re-split of either fragment
    // still knows it came from an implicit page and diagnoses consistently.
    if (isImplicit) {
      newPageOp0->setAttr("cert.implicit", rewriter.getUnitAttr());
      newPageOp1->setAttr("cert.implicit", rewriter.getUnitAttr());
    }

    for (auto job : op.getBody().front().getOps<AIEX::CertJobOp>()) {
      if (job == split_job) {
        if (jobBoundaryCut) {
          // Whole job to the later page: partitioning it at its own first op
          // would leave an empty half behind on the earlier page.
          rewriter.setInsertionPointToEnd(newPageBlock1);
          cloneJobRange(job, job.getJobId(), job.getBody().front().begin(),
                        job.getBody().front().end());
          continue;
        }
        rewriter.setInsertionPointToEnd(newPageBlock0);
        cloneJobRange(job, job.getJobId(), job.getBody().front().begin(),
                      split_iter);

        rewriter.setInsertionPointToEnd(newPageBlock1);
        cloneJobRange(job, job.getJobId() + 1, split_iter,
                      job.getBody().front().end());
        continue;
      }

      if (job->isBeforeInBlock(split_job)) {
        rewriter.setInsertionPointToEnd(newPageBlock0);
        cloneJobRange(job, job.getJobId(), job.getBody().front().begin(),
                      job.getBody().front().end());
      } else {
        rewriter.setInsertionPointToEnd(newPageBlock1);
        cloneJobRange(job, job.getJobId(), job.getBody().front().begin(),
                      job.getBody().front().end());
      }
    }

    AIEX::CertPageOp::ensureTerminator(newPageOp0.getBody(), rewriter, loc);
    AIEX::CertPageOp::ensureTerminator(newPageOp1.getBody(), rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }

  bool *hadError;
};

// Form implicit pages: group maximal runs of contiguous top-level
// cert.job ops into a single cert.page. Only an explicit cert.page delimits a
// run; structural ops (tiles, buffers, locks, switchboxes, ...) do NOT. The
// implicit configuration job (tagged {cert.configure}) is kept isolated in its
// own page so it can be forced first on uC0 and never merged into a user
// implicit page. CertJobOp legally parents under DeviceOp, so bare device-level
// jobs are valid input here (produced by RuntimeSequenceToCertJob).
// attach_to_group(0) names uC 0, which is exactly the default (unspecified)
// group. It is therefore a placement no-op, not a page boundary: flatten it so
// its content joins the device-level group-0 stream and participates in
// implicit-page formation with adjacent default-group-0 content (rather than
// being isolated on its own page). Move each group-0 op out before the
// attach_to_group, preserving order, then erase the now-empty op.
static void inlineDefaultGroup(AIE::DeviceOp dev) {
  Block *body = dev.getBody();
  SmallVector<AIEX::CertAttachToGroupOp> zeroGroups;
  for (auto grp : body->getOps<AIEX::CertAttachToGroupOp>())
    if (grp.getGroupId() == 0)
      zeroGroups.push_back(grp);
  for (auto grp : zeroGroups) {
    SmallVector<Operation *> inner;
    for (Operation &op : grp.getBody().front())
      if (!isa<AIE::EndOp>(op))
        inner.push_back(&op);
    for (Operation *op : inner)
      op->moveBefore(grp.getOperation());
    grp.erase();
  }
}

// Group a block's maximal runs of contiguous top-level cert.job ops into
// implicit cert.page ops (only an explicit cert.page delimits; the
// {cert.configure} job stays isolated in its own page). Applied per uC stream.
static void formImplicitPagesInBlock(Block *body, OpBuilder &builder) {
  // Snapshot top-level ops; the body is mutated while iterating.
  SmallVector<Operation *> topOps;
  for (Operation &op : *body)
    topOps.push_back(&op);

  // `cert.implicit` marks a page formed from a run of *user* jobs -- a
  // boundary the author could have drawn but didn't. The splitter's diagnostics
  // are keyed off it and all advise editing page boundaries, so it must not be
  // set on a page the user cannot restructure. The compiler-synthesized
  // @configure page is exactly that case, and it is isolated by construction,
  // so the tag would buy nothing there anyway.
  auto openPage = [&](Location loc, Operation *before,
                      bool implicit) -> AIEX::CertPageOp {
    builder.setInsertionPoint(before);
    auto page = AIEX::CertPageOp::create(builder, loc);
    if (implicit)
      page->setAttr("cert.implicit", builder.getUnitAttr());
    Block *b = new Block();
    page.getBody().push_back(b);
    AIEX::CertPageOp::ensureTerminator(page.getBody(), builder, loc);
    return page;
  };

  // The (tile, channel) actors a job wait_tcts on, keyed exactly as
  // AIECertVerify's G-tct check (checkWaitTctsUniqueness) keys them so the two
  // stay consistent.
  auto waitTctsActors = [](AIEX::CertJobOp job) {
    std::set<std::pair<int, int>> actors;
    job.walk([&](AIEX::CertWaitTCTSOp wait) {
      actors.insert({wait.getTileId(), wait.getChannelId()});
    });
    return actors;
  };

  AIEX::CertPageOp currentPage; // null => no open implicit page
  // Actors already waited on by the jobs accumulated into `currentPage`.
  std::set<std::pair<int, int>> pageActors;
  auto closePage = [&]() {
    currentPage = nullptr;
    pageActors.clear();
  };

  for (Operation *op : topOps) {
    if (auto job = dyn_cast<AIEX::CertJobOp>(op)) {
      if (job->hasAttr("cert.configure")) {
        // Keep the configuration job isolated in its own page. Not tagged
        // implicit: the user did not author this job and cannot add a boundary
        // inside it, so the implicit-page remark would be noise on every
        // compile whose config transaction is large enough to split (and it
        // usually is).
        auto page = openPage(job.getLoc(), job, /*implicit=*/false);
        job->moveBefore(page.getBody().front().getTerminator());
        closePage();
        continue;
      }
      // G-tct: at most one job per page may wait on a given actor. Two jobs
      // that each legally wait on the same (tile, channel) -- an ordinary shape
      // for two runtime sequences sharing a shim channel -- are only legal
      // while they sit on separate pages, so grouping them would manufacture a
      // verifier error out of valid input. Cut a page boundary instead, exactly
      // as an over-size job would.
      auto actors = waitTctsActors(job);
      if (currentPage &&
          llvm::any_of(actors, [&](const std::pair<int, int> &actor) {
            return pageActors.count(actor);
          }))
        closePage();
      if (!currentPage)
        currentPage = openPage(job.getLoc(), job, /*implicit=*/true);
      job->moveBefore(currentPage.getBody().front().getTerminator());
      pageActors.insert(actors.begin(), actors.end());
    } else if (isa<AIEX::CertPageOp>(op)) {
      // An explicit page delimits the current implicit run.
      closePage();
    }
    // All other ops (structural / control containers) do not delimit a run.
  }
}

static void formImplicitPages(AIE::DeviceOp dev) {
  OpBuilder builder(dev.getContext());
  formImplicitPagesInBlock(dev.getBody(), builder);
  // Each other uC (attach_to_group) forms implicit pages within its own stream,
  // so adjacent jobs on that uC are cooperatively co-located too.
  SmallVector<AIEX::CertAttachToGroupOp> groups;
  for (auto grp : dev.getBody()->getOps<AIEX::CertAttachToGroupOp>())
    groups.push_back(grp);
  for (auto grp : groups)
    formImplicitPagesInBlock(&grp.getBody().front(), builder);
}

// Force the implicit configuration page to be the first page on uC0:
// @configure sets up the device (DMA/lock/core config), so it must strictly
// precede all other uC0 pages regardless of where it appeared textually. The
// config job is already isolated in its own page by formImplicitPages; here we
// just hoist that page to the front of the device body (structural ops are not
// control code, so position relative to them does not matter).
static void moveConfigPageFirst(AIE::DeviceOp dev) {
  Block *body = dev.getBody();
  AIEX::CertPageOp configPage;
  for (auto page : body->getOps<AIEX::CertPageOp>()) {
    for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
      if (job->hasAttr("cert.configure")) {
        configPage = page;
        break;
      }
    }
    if (configPage)
      break;
  }
  if (configPage && &body->front() != configPage.getOperation())
    configPage->moveBefore(&body->front());
}

// Lower the `placement` attribute on top-level cert.page ops to enclosing
// cert.attach_to_group ops. Placed pages (placement present and != 0)
// are moved, preserving IR order, under one cert.attach_to_group per distinct
// group id. Unplaced pages (and placement 0) stay at device level and are
// emitted as the implicit group 0. Nesting stays attach_to_group -> page -> job
// as the emitter expects.
static void lowerPagePlacementToGroups(AIE::DeviceOp dev) {
  Block *body = dev.getBody();

  // Collect placed top-level pages by resolved group id, in first-seen order.
  std::map<int32_t, SmallVector<AIEX::CertPageOp, 4>> byGroup;
  for (auto page : body->getOps<AIEX::CertPageOp>()) {
    if (auto placement = page.getPlacementAttr()) {
      auto gid = static_cast<int32_t>(placement.getInt());
      if (gid != 0)
        byGroup[gid].push_back(page);
    }
  }

  OpBuilder builder(dev.getContext());
  for (auto &entry : byGroup) {
    int32_t gid = entry.first;
    SmallVector<AIEX::CertPageOp, 4> &pages = entry.second;

    // Create the group op where the first placed page currently lives (avoids
    // assuming anything about a device terminator).
    builder.setInsertionPoint(pages.front());
    auto grp = AIEX::CertAttachToGroupOp::create(builder, dev.getLoc(), gid);
    Block *grpBlock = new Block();
    grp.getBody().push_back(grpBlock);

    // Move the placed pages into the group, in order, dropping the
    // now-redundant placement attribute (the group encodes it).
    for (AIEX::CertPageOp page : pages) {
      page->removeAttr("placement");
      page->moveBefore(grpBlock, grpBlock->end());
    }
    AIEX::CertAttachToGroupOp::ensureTerminator(grp.getBody(), builder,
                                                dev.getLoc());
  }
}

// The firmware indexes job contexts by job_id % 64, and a page's jobs finish
// before the next page starts. Keep each page's noncolliding IDs and move the
// colliding ones to free slots.
static LogicalResult allocateJobIds(AIE::DeviceOp device) {
  WalkResult result = device.walk([&](AIEX::CertPageOp page) {
    std::bitset<AIEX::CertJobOp::NumSlots> used;
    SmallVector<AIEX::CertJobOp> collisions;
    unsigned count = 0;
    for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
      unsigned slot = job.getJobId() % AIEX::CertJobOp::NumSlots;
      if (used.test(slot))
        collisions.push_back(job);
      used.set(slot);
      ++count;
    }
    if (count > AIEX::CertJobOp::NumSlots) {
      page.emitError() << "cert.page has " << count << " jobs, exceeding the "
                       << AIEX::CertJobOp::NumSlots
                       << " firmware job slots; reduce the jobs on this page";
      return WalkResult::interrupt();
    }
    unsigned slot = 0;
    for (auto job : collisions) {
      while (used.test(slot))
        ++slot;
      job.setJobId(slot);
      used.set(slot);
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

struct AIECertPagesPass
    : xilinx::AIEX::impl::AIECertPagesBase<AIECertPagesPass> {
  void runOnOperation() override {
    inlineDefaultGroup(getOperation());

    // Form implicit pages: group contiguous top-level jobs into pages; only an
    // explicit cert.page delimits. Replaces the old one-page-per-job
    // wrapping.
    formImplicitPages(getOperation());

    // Force the @configure page to be first on uC0.
    moveConfigPageFirst(getOperation());

    // Lower cert.page placement to cert.attach_to_group BEFORE isolate
    // and split. Those rewrites create replacement pages adjacent to the
    // original (same parent block) with the no-placement builder; running the
    // placement grouping first means placed pages already live inside their
    // cert.attach_to_group, so any pages the later rewrites spawn stay in the
    // same group by position. (Doing this last silently dropped placement from
    // split/isolated pages, emitting them on group 0.)
    lowerPagePlacementToGroups(getOperation());

    // First, isolate load_pdi and preempt operations into their own job/page.
    // `isolateError` mirrors `splitError` below: the pattern declines to
    // rewrite a job whose local-register values would not survive isolation,
    // and that diagnostic has to fail the pass rather than just print.
    bool isolateError = false;
    RewritePatternSet isolatePatterns(&getContext());
    isolatePatterns.insert<IsolateFullPageOpsPattern>(&getContext(),
                                                      &isolateError);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(isolatePatterns))))
      signalPassFailure();
    if (isolateError)
      signalPassFailure();

    // Then apply the page splitting pattern. `splitError` catches diagnostics
    // the pattern emits while declining to split (see SplitCertPageOpPattern):
    // applyPatternsGreedily reports only whether the rewrite converged, so an
    // illegal-to-split page would otherwise print an error and still compile.
    bool splitError = false;
    RewritePatternSet p1(&getContext());
    p1.insert<SplitCertPageOpPattern>(&getContext(), &splitError);
    if (failed(applyPatternsGreedily(getOperation(), std::move(p1))))
      signalPassFailure();
    if (splitError)
      signalPassFailure();

    if (failed(allocateJobIds(getOperation())))
      return signalPassFailure();

    // Add the merge pattern for CertUcDmaWriteDesSyncOps
    RewritePatternSet p3(&getContext());
    p3.insert<MergeConsecutiveCertUcDmaWriteDesSyncOps>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(p3))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> AIEX::createAIENpuToCertPass() {
  return std::make_unique<AIENpuToCertPass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
AIEX::createAIENpuToCertPass(const AIENpuToCertOptions &options) {
  return std::make_unique<AIENpuToCertPass>(options);
}

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIECertPagesPass() {
  return std::make_unique<AIECertPagesPass>();
}
