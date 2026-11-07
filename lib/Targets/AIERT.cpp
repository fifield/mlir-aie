//===- AIERT.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIERT.h"
#include "aie/Targets/AIETargetShared.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_dma.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_helper.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaie_locks.h"
#include "xaiengine/xaie_mem.h"
#include "xaiengine/xaie_noc.h"
#include "xaiengine/xaie_perfcnt.h"
#include "xaiengine/xaie_plif.h"
#include "xaiengine/xaie_reset.h"
#include "xaiengine/xaie_ss.h"
#include "xaiengine/xaie_tilectrl.h"
#include "xaiengine/xaie_txn.h"
#include "xaiengine/xaie_uc.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"
}

#include <filesystem>

#define AIERC_STR(x) x, #x
static const std::map<AieRC, std::string> AIERCTOSTR = {
    {AIERC_STR(XAIE_OK)},
    {AIERC_STR(XAIE_ERR)},
    {AIERC_STR(XAIE_INVALID_DEVICE)},
    {AIERC_STR(XAIE_INVALID_RANGE)},
    {AIERC_STR(XAIE_INVALID_ARGS)},
    {AIERC_STR(XAIE_INVALID_TILE)},
    {AIERC_STR(XAIE_ERR_STREAM_PORT)},
    {AIERC_STR(XAIE_INVALID_DMA_TILE)},
    {AIERC_STR(XAIE_INVALID_BD_NUM)},
    {AIERC_STR(XAIE_ERR_OUTOFBOUND)},
    {AIERC_STR(XAIE_INVALID_DATA_MEM_ADDR)},
    {AIERC_STR(XAIE_INVALID_ELF)},
    {AIERC_STR(XAIE_CORE_STATUS_TIMEOUT)},
    {AIERC_STR(XAIE_INVALID_CHANNEL_NUM)},
    {AIERC_STR(XAIE_INVALID_LOCK)},
    {AIERC_STR(XAIE_INVALID_DMA_DIRECTION)},
    {AIERC_STR(XAIE_INVALID_PLIF_WIDTH)},
    {AIERC_STR(XAIE_INVALID_LOCK_ID)},
    {AIERC_STR(XAIE_INVALID_LOCK_VALUE)},
    {AIERC_STR(XAIE_LOCK_RESULT_FAILED)},
    {AIERC_STR(XAIE_INVALID_DMA_DESC)},
    {AIERC_STR(XAIE_INVALID_ADDRESS)},
    {AIERC_STR(XAIE_FEATURE_NOT_SUPPORTED)},
    {AIERC_STR(XAIE_INVALID_BURST_LENGTH)},
    {AIERC_STR(XAIE_INVALID_BACKEND)},
    {AIERC_STR(XAIE_INSUFFICIENT_BUFFER_SIZE)},
    {AIERC_STR(XAIE_INVALID_API_POINTER)},
    {AIERC_STR(XAIE_NOT_SUPPORTED)},
    {AIERC_STR(XAIE_INVALID_APP_MODE)},
    {AIERC_STR(XAIE_ERR_MAX)}};
#undef AIERC_STR

static const std::map<xilinx::AIE::WireBundle, StrmSwPortType>
    WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE = {
        {xilinx::AIE::WireBundle::Core, StrmSwPortType::CORE},
        {xilinx::AIE::WireBundle::DMA, StrmSwPortType::DMA},
        {xilinx::AIE::WireBundle::DMATrace, StrmSwPortType::DMA_Trace},
        {xilinx::AIE::WireBundle::DMAControl, StrmSwPortType::DMA_CTRL},
        {xilinx::AIE::WireBundle::TileControl, StrmSwPortType::CTRL},
        {xilinx::AIE::WireBundle::FIFO, StrmSwPortType::FIFO},
        {xilinx::AIE::WireBundle::South, StrmSwPortType::SOUTH},
        {xilinx::AIE::WireBundle::West, StrmSwPortType::WEST},
        {xilinx::AIE::WireBundle::North, StrmSwPortType::NORTH},
        {xilinx::AIE::WireBundle::East, StrmSwPortType::EAST},
        // missing PLIO from WireBundle
        // missing NOC from WireBundle
        {xilinx::AIE::WireBundle::Trace, StrmSwPortType::TRACE},
        // {xilinx::AIE::WireBundle::UCTRLR, StrmSwPortType::UCTRLR},
        {xilinx::AIE::WireBundle::NorthControl, StrmSwPortType::NORTH_CTRL},
        {xilinx::AIE::WireBundle::SouthControl, StrmSwPortType::SOUTH_CTRL},
        {xilinx::AIE::WireBundle::Switch32, StrmSwPortType::SWITCH_32b},
        // {xilinx::AIE::WireBundle::DMA_CTRL, StrmSwPortType::DMA_CTRL},
        {xilinx::AIE::WireBundle::PLIO, StrmSwPortType::PL},
        // AIE4 specific 32b switch PortTypes
        {xilinx::AIE::WireBundle::Control32, StrmSwPortType::_32B_CTRL},
        {xilinx::AIE::WireBundle::Trace32, StrmSwPortType::_32B_TRACE},
        {xilinx::AIE::WireBundle::South32, StrmSwPortType::_32B_SOUTH},
        {xilinx::AIE::WireBundle::West32, StrmSwPortType::_32B_WEST},
        {xilinx::AIE::WireBundle::North32, StrmSwPortType::_32B_NORTH},
        {xilinx::AIE::WireBundle::East32, StrmSwPortType::_32B_EAST},
        {xilinx::AIE::WireBundle::Switch512, StrmSwPortType::_32B_SWITCH_512b},
        {xilinx::AIE::WireBundle::Controller32, StrmSwPortType::_32B_UCTRLR},
        {xilinx::AIE::WireBundle::PLIO32, StrmSwPortType::_32B_PL}};

static XAie_DmaDirection toXAieDmaDirection(xilinx::AIE::DMAChannelDir dir) {
  if (dir == xilinx::AIE::DMAChannelDir::ControlMM2S)
    return DMA_MM2S_CTRL;
  if (dir == xilinx::AIE::DMAChannelDir::TraceS2MM)
    return DMA_S2MM_TRACE;
  return xilinx::AIE::isS2MM(dir) ? DMA_S2MM : DMA_MM2S;
}

#ifndef NDEBUG

// https://stackoverflow.com/a/32230306
template <typename H1>
static llvm::raw_ostream &showAIEXRTArgs(llvm::raw_ostream &out,
                                         const char *label, H1 &&value) {
  return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
static llvm::raw_ostream &showAIEXRTArgs(llvm::raw_ostream &out,
                                         const char *label, H1 &&value,
                                         T &&...rest) {
  const char *pcomma = strchr(label, ',');
  return showAIEXRTArgs(out.write(label, pcomma - label)
                            << "=" << std::forward<H1>(value) << ',',
                        pcomma + 1, std::forward<T>(rest)...);
}

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const XAie_LocType &loc);

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const XAie_Lock &lock);

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const XAie_Packet &packet);

#define SHOW_AIERT_ARGS(os, ...) showAIEXRTArgs(os, #__VA_ARGS__, __VA_ARGS__)

// So that we can use the pattern if(auto r = TRY_XAIE_API...) { // r is nonzero
// }
static_assert(XAIE_OK == 0);

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                     \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERT_ARGS(llvm::dbgs(), __VA_ARGS__));                    \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +             \
                               AIERCTOSTR.at(r));                              \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERT_ARGS(llvm::dbgs(), __VA_ARGS__));                    \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      return OP.emitOpError() << #API " failed with " << AIERCTOSTR.at(r);     \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERT_ARGS(llvm::dbgs(), __VA_ARGS__));                    \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__)) {                                           \
      llvm::errs() << #API " failed with " << AIERCTOSTR.at(r);                \
      return failure();                                                        \
    }                                                                          \
  } while (0)

#else

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                     \
  do {                                                                         \
    if (auto r = API(__VA_ARGS__))                                             \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +             \
                               AIERCTOSTR.at(r));                              \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                                  \
  do {                                                                         \
    if (auto r = API(__VA_ARGS__))                                             \
      return OP.emitOpError() << #API " failed with " << AIERCTOSTR.at(r);     \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                                  \
  do {                                                                         \
    if (auto r = API(__VA_ARGS__)) {                                           \
      llvm::errs() << #API " failed with " << AIERCTOSTR.at(r);                \
      return failure();                                                        \
    }                                                                          \
  } while (0)

#endif

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "aie-aiert"

static bool hasTileControlMasterSet(Block &block) {
  return llvm::any_of(block.getOps<AIE::MasterSetOp>(), [](auto masterSetOp) {
    return masterSetOp.getDestBundle() == AIE::WireBundle::TileControl;
  });
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_LocType &loc) {
  os << "XAie_LocType(col: " << std::to_string(loc.Col)
     << ", row: " << std::to_string(loc.Row) << ")";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Lock &lock) {
  os << "XAie_Lock(id: " << std::to_string(lock.LockId)
     << ", val: " << std::to_string(lock.LockVal) << ")";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const XAie_Packet &packet) {
  os << "XAie_Packet(id: " << std::to_string(packet.PktId)
     << ", type: " << std::to_string(packet.PktType) << ")";
  return os;
}

#define XAIE_BASE_ADDR 0x40000000
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_PARTITION_BASE_ADDR 0x0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define EVEN_BD_NUM_START 0
#define ODD_BD_NUM_START 24

struct xilinx::AIE::AIERTControl::AIERtImpl {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  // One Location per recorded XAie_TxnCmd, indexed by position in
  // XAie_TxnInst::CmdBuf. Populated by TxnLocBracket scopes.
  std::vector<mlir::Location> txnCmdLocs;

  // One Location per operation in the exported transaction binary. Derived
  // from txnCmdLocs by exportSerializedTransaction().
  std::vector<mlir::Location> txnOpLocs;
};

// Walk DevInst->TxnList to find the active XAie_TxnInst (single-threaded
// compilation: at most one). Returns null if no transaction is being
// recorded. The XAie_TxnInst struct and TxnList layout are part of aie-rt's
// public xaiegbl.h, so this only depends on the public header.
static XAie_TxnInst *getActiveTxnInst(XAie_DevInst &devInst) {
  XAie_List *node = devInst.TxnList.Next;
  if (!node)
    return nullptr;
  return reinterpret_cast<XAie_TxnInst *>(reinterpret_cast<char *>(node) -
                                          offsetof(XAie_TxnInst, Node));
}

static uint32_t getActiveTxnNumCmds(XAie_DevInst &devInst) {
  XAie_TxnInst *inst = getActiveTxnInst(devInst);
  return inst ? inst->NumCmds : 0;
}

xilinx::AIE::TxnLocBracket::TxnLocBracket(AIERTControl &c, mlir::Location l)
    : ctl(c), loc(l), startCmds(c.getCurrentTxnNumCmds()) {}

xilinx::AIE::TxnLocBracket::~TxnLocBracket() {
  uint32_t endCmds = ctl.getCurrentTxnNumCmds();
  if (endCmds <= startCmds)
    return;
  // Only record meaningful locations. Falling through (no entry) lets the
  // downstream emitter use its fallback (device.getLoc()), which preserves
  // the pre-existing behavior for source ops that don't carry a real loc.
  if (mlir::isa<mlir::UnknownLoc>(loc))
    return;
  ctl.recordTxnLocRange(startCmds, endCmds, loc);
}

xilinx::AIE::AIERTControl::~AIERTControl() { XAie_Finish(&aiert->devInst); }

xilinx::AIE::AIERTControl::AIERTControl(const AIE::AIETargetModel &tm)
    : targetModel(tm), aiert(std::make_unique<AIERtImpl>()) {
  // The first column in the NPU lacks a shim tile.  AIE-RT exposes some of
  // the internals about how this is modeled in a somewhat awkward way.
  size_t partitionStartCol =
      tm.hasProperty(AIE::AIETargetModel::IsVirtualized) ? 1 : 0;
  size_t partitionNumCols = tm.columns();
  size_t deviceRows = tm.rows();
  size_t deviceCols = tm.columns() + partitionStartCol;

  // Don't put this in the target model, because it's XAIE specific.
  unsigned char devGen = XAIE_DEV_GEN_AIE;
  switch (tm.getTargetArch()) {
  case AIEArch::AIE1: // probably unreachable.
    devGen = XAIE_DEV_GEN_AIE;
    break;
  case AIEArch::AIE2:
    // FIXME: What if we don't have an IPU?  aie-rt
    // models non-IPU devices differently.
    devGen = XAIE_DEV_GEN_AIE2IPU;
    break;
  case AIEArch::AIE2p:
    devGen = XAIE_DEV_GEN_AIE2P_STRIX_B0;
    break;
  case AIEArch::AIE2ps:
    // AIE2PS configures the array through a CERT ELF instead of CDO or an
    // xclbin (see AIE2PSTargetModel), so it has no aie-rt device generation.
    // Listed here so that a new AIEArch still raises -Wswitch.
    llvm::report_fatal_error("aie-rt configuration is not supported for the "
                             "AIE2PS architecture");
  case AIEArch::AIE4:
    devGen = XAIE_DEV_GEN_AIE4;
    break;
  }
  aiert->configPtr = XAie_Config{
      /*AieGen*/ devGen,
      /*BaseAddr*/ 0,
      /*ColShift*/ static_cast<uint8_t>(tm.getColumnShift()),
      /*RowShift*/ static_cast<uint8_t>(tm.getRowShift()),
      /*NumRows*/ static_cast<uint8_t>(deviceRows),
      /*NumCols*/ static_cast<uint8_t>(deviceCols),
      /*ShimRowNum*/ XAIE_SHIM_ROW,
      /*MemTileRowStart*/ XAIE_MEM_TILE_ROW_START,
      /*MemTileNumRows*/ static_cast<uint8_t>(tm.getNumMemTileRows()),
      /*AieTileRowStart*/
      static_cast<uint8_t>(XAIE_MEM_TILE_ROW_START + tm.getNumMemTileRows()),
      /*AieTileNumRows*/
      static_cast<uint8_t>(tm.rows() - tm.getNumMemTileRows() - 1),
      /*PartProp*/ {},
      /*Backend*/ XAIE_IO_BACKEND_CDO};
  XAie_InstDeclare(_devInst, &aiert->configPtr);
  aiert->devInst = _devInst;
  TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &aiert->devInst,
                           XAIE_PARTITION_BASE_ADDR, partitionStartCol,
                           partitionNumCols);
  TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &aiert->devInst,
                           &aiert->configPtr);
  TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &aiert->devInst, NPI_ADDR);
}

LogicalResult xilinx::AIE::AIERTControl::setIOBackend(bool aieSim,
                                                      bool xaieDebug) {
  // Quoting: The instance of a device must be always declared using this
  // macro. In the future, the same macro will be expanded to
  // allocate more memory from the user application for resource
  // management.
  if (aieSim) {
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &aiert->devInst,
                             XAIE_IO_BACKEND_SIM);
  } else if (xaieDebug)
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &aiert->devInst,
                             XAIE_IO_BACKEND_DEBUG);
  else
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &aiert->devInst,
                             XAIE_IO_BACKEND_CDO);
  return success();
}

static LogicalResult
configureLocksInBdBlock(const AIE::AIETargetModel &targetModel,
                        XAie_DmaDesc &dmaTileBd, Block &block, int col, int row,
                        bool outOfOrder) {
  LLVM_DEBUG(llvm::dbgs() << "\nstart configuring bds\n");
  AIE::UseLockOp acquire, release;
  if (failed(AIE::verifyBdLockPair(block, outOfOrder, acquire, release)))
    return failure();
  std::optional<int> acqValue, relValue, acqLockId, relLockId;
  bool acqEn = false;

  // A memtile BD may use locks of its neighbours, so each lock ID is offset
  // by the base index of the tile that owns that lock.
  auto lockIdFor = [&](AIE::LockOp lock) {
    int id = lock.getLockIDValue();
    if (targetModel.isMemTile(col, row))
      if (auto offset = targetModel.getLockLocalBaseIndex(
              col, row, lock.colIndex(), lock.rowIndex()))
        id += *offset;
    return id;
  };
  for (auto op : block.getOps<AIE::UseLockOp>()) {
    auto lock = cast<AIE::LockOp>(op.getLock().getDefiningOp());
    switch (op.getAction()) {
    case AIE::LockAction::Acquire:
    case AIE::LockAction::AcquireGreaterEqual: {
      acqEn = op.getAcqEn();
      acqLockId = lockIdFor(lock);
      auto value = op.getConstantValue();
      if (failed(value))
        return failure();
      int32_t v = *value; // NOLINT(bugprone-unchecked-optional-access)
      if (op.acquireGE())
        v = -v;
      acqValue = v;
      break;
    }
    case AIE::LockAction::Release: {
      relLockId = lockIdFor(lock);
      auto value = op.getConstantValue();
      if (failed(value))
        return failure();
      relValue = value;
      break;
    }
    }
  }



  bool relEn = false;
  if (!acqLockId) {
    // Release-only BD (out-of-order, or a semaphore-lock sync BD): point the
    // disabled acquire lock at the release lock so XAie_LockInit gets a valid
    // id.
    acqLockId = relLockId;
    acqValue = 0;
    acqEn = false;
  }
  if (!relLockId) {
    // Acquire-only BD: a release value of 0 releases nothing.
    relLockId = acqLockId;
    relValue = 0;
  }
  assert(acqLockId && acqValue && relLockId && relValue &&
         "lock values must be resolved before init");
  XAie_Lock acqLock = XAie_LockInit(acqLockId.value(), acqValue.value());
  XAie_Lock relLock = XAie_LockInit(relLockId.value(), relValue.value());
  TRY_XAIE_API_EMIT_ERROR((*block.getOps<AIE::UseLockOp>().begin()),
                          dmaTileBd.DmaMod->SetLock, &dmaTileBd, acqLock,
                          relLock, acqEn, relEn);
  return success();
}

static LogicalResult
configureBdInBlock(const AIE::AIETargetModel &targetModel,
                   XAie_DevInst *devInst, XAie_DmaDesc &dmaTileBd, Block &block,
                   uint32_t channel, AIE::DMAChannelDir dir, int col, int row,
                   int bdId, std::optional<int> nextBdId) {
  std::optional<int> packetType;
  std::optional<int> packetID;

  // Below should go
  auto maybePacketOps = block.getOps<AIE::DMABDPACKETOp>();
  if (!maybePacketOps.empty()) {
    assert(llvm::range_size(maybePacketOps) == 1 &&
           "expected only one dma_bd_packet");
    auto packetOp = *maybePacketOps.begin();
    packetType = packetOp.getPacketType();
    packetID = packetOp.getPacketID();
  }

  auto bdOp = *block.getOps<AIE::DMABDOp>().begin();

  if (targetModel.isShimNOCTile(col, row)) {
    // write them out like this so they show up with names in debug prints
    uint8_t smid = 0;
    uint32_t burstLen =
        getShimBurstLengthBytes(targetModel, bdOp.getBurstLength());
    uint8_t qOs = 0;
    uint8_t cache = static_cast<uint8_t>(bdOp.getAxcacheOrDefault());
    uint8_t secure = 0;
    // workaround for aie-rt behavior in xaie_dma.c where it expects different
    // values for aie4
    if (isa<AIE::AIE4TargetModel>(targetModel))
      burstLen = burstLen / 4;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAxi, &dmaTileBd, smid,
                            burstLen / 16, qOs, cache, secure);
  }

  // get address from BufferOp (core,mem) or ExternalBufferOp (shim)
  uint64_t baseAddr = 0;
  if (targetModel.isShimNOCTile(col, row)) {
    auto bufferOp =
        cast<AIE::ExternalBufferOp>(bdOp.getBuffer().getDefiningOp());
    // external buffers aren't required to have an address here because the
    // address might get patched later or the default of zero might be a valid
    // address.
    if (auto addr = bufferOp.getAddress())
      baseAddr = *addr; // NOLINT(bugprone-unchecked-optional-access)
  } else {
    auto bufferOp = cast<AIE::BufferOp>(bdOp.getBuffer().getDefiningOp());
    auto addr = bufferOp.getAddress();
    if (!addr)
      return bufferOp.emitError("buffer must have address assigned");
    baseAddr = *addr; // NOLINT(bugprone-unchecked-optional-access)
  }

  if (targetModel.isMemTile(col, row)) {
    // check if buffer is allocated on the same memtile, the west, or the east
    // one
    auto bufferOp = cast<AIE::BufferOp>(bdOp.getBuffer().getDefiningOp());
    auto bufferRow = bufferOp.getTileOp().getRow();
    auto bufferCol = bufferOp.getTileOp().getCol();
    auto addrOffset =
        targetModel.getMemLocalBaseAddress(col, row, bufferCol, bufferRow);
    if (addrOffset)
      baseAddr += addrOffset.value();
  }

  // Runtime-valued sizes/strides are not supported on this static lowering
  // path.
  for (mlir::OpFoldResult s : bdOp.getMixedSizes())
    if (!mlir::getConstantIntValue(s))
      return bdOp->emitOpError(
          "runtime-valued BD size/stride is not supported on the static XAIE "
          "lowering path; use compile-time constant sizes");
  for (mlir::OpFoldResult s : bdOp.getMixedStrides())
    if (!mlir::getConstantIntValue(s))
      return bdOp->emitOpError(
          "runtime-valued BD size/stride is not supported on the static XAIE "
          "lowering path; use compile-time constant sizes");
  // The owning storage must outlive the ArrayRef used below.
  std::optional<llvm::SmallVector<AIE::BDDimLayoutAttr>> dimsStorage =
      bdOp.getConstantDimensions();
  if (!dimsStorage)
    return bdOp->emitOpError("internal error folding BD dimensions");
  std::optional<llvm::ArrayRef<AIE::BDDimLayoutAttr>> dims;
  if (!dimsStorage->empty())
    dims = llvm::ArrayRef<AIE::BDDimLayoutAttr>(*dimsStorage);
  uint64_t lenInBytes = bdOp.getLenInBytes();
  uint64_t basePlusOffsetInBytes = baseAddr + bdOp.getOffsetInBytes();
  if (!dims) {
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAddrLen, &dmaTileBd,
                            basePlusOffsetInBytes, lenInBytes);
  } else {
    llvm::SmallVector<XAie_DmaDimDesc, 4> dimDescs(dims->size());
    XAie_DmaTensor dmaTileBdTensor = {};
    dmaTileBdTensor.NumDim = dimDescs.size();
    dmaTileBdTensor.Dim = dimDescs.data();
    // libxaie requires stride in multiples of 32b
    double elementWidthIn32bWords =
        static_cast<double>(bdOp.getBufferElementTypeWidthInBytes()) / 4.0;
    for (size_t i = 0; i < dims->size(); i++) {
      // Pass down dimensions in reverse order; in the MLIR, this allows
      // us to specify step sizes/wraps in the same order as we would
      // access a multi-dim C array, with the highest dimension first.
      int j = dims->size() - i - 1;
      uint16_t size;
      uint32_t stride;
      if (j > 0) {
        stride = static_cast<uint32_t>(dims.value()[i].getStride() *
                                       elementWidthIn32bWords);
        size = dims.value()[i].getSize();
      } else {
        stride = dims.value()[i].getStride();
        size = static_cast<uint16_t>(dims.value()[i].getSize() *
                                     elementWidthIn32bWords);
      }
      stride = stride > 0 ? stride : 1;
      // Assume AIE-ML architecture (ie use AieMlDimDesc instead of AieDimDesc);
      // asserted in AIETranslateToCDODirect).
      dmaTileBdTensor.Dim[j].AieMlDimDesc = {stride, size};
    }
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetMultiDimAddr, &dmaTileBd,
                            &dmaTileBdTensor, basePlusOffsetInBytes,
                            lenInBytes);
  }

  // BD iteration.
  if (auto iter = bdOp.getIteration(); iter && iter->getSize() > 1) {
    if (iter->getStride() > 0) {
      double elementWidthIn32bWords =
          static_cast<double>(bdOp.getBufferElementTypeWidthInBytes()) / 4.0;
      uint32_t stepInWords =
          static_cast<uint32_t>(iter->getStride() * elementWidthIn32bWords);
      uint16_t wrap = static_cast<uint16_t>(iter->getSize());
      uint8_t iterCurr = static_cast<uint8_t>(iter->getCurrent());
      TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetBdIteration, &dmaTileBd,
                              stepInWords, wrap, iterCurr);
    }
  }

  // ND zero padding.
  std::optional<llvm::ArrayRef<AIE::BDPadLayoutAttr>> padDims =
      bdOp.getPadDimensions();

  if (padDims) {
    llvm::SmallVector<XAie_PadDesc, 4> padDescs(padDims->size());
    XAie_DmaPadTensor dmaPadTensor = {};
    dmaPadTensor.NumDim = padDescs.size();
    dmaPadTensor.PadDesc = padDescs.data();
    // libxaie requires stride in multiples of 32b
    double elementWidthIn32bWords =
        static_cast<double>(bdOp.getBufferElementTypeWidthInBytes()) / 4.0;
    for (size_t i = 0; i < padDims->size(); i++) {
      // Pass down dimensions in reverse order.
      int j = padDims->size() - i - 1;
      uint8_t before;
      uint8_t after;
      if (j > 0) {
        before = static_cast<uint8_t>(padDims.value()[i].getConstPadBefore());
        after = static_cast<uint8_t>(padDims.value()[i].getConstPadAfter());
      } else {
        before = static_cast<uint8_t>(padDims.value()[i].getConstPadBefore() *
                                      elementWidthIn32bWords);
        after = static_cast<uint8_t>(padDims.value()[i].getConstPadAfter() *
                                     elementWidthIn32bWords);
      }
      dmaPadTensor.PadDesc[j] = {before, after};
    }
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetPadding, &dmaTileBd,
                            &dmaPadTensor);
  }
  if (nextBdId) {
    auto enableNextBd = 1;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetNextBd, &dmaTileBd,
                            nextBdId.value(), enableNextBd);
  }

  if (auto packetInfo = bdOp.getPacket()) {
    packetType = packetInfo->getPktType();
    packetID = packetInfo->assignedId();
  }

  if (packetID) {
    if (!packetType)
      return bdOp.emitError("must have packetType with packetID");
    // getConstantLen() is nullopt for a runtime len operand, so this guard only
    // catches a compile-time-zero length. A runtime len that happens to be 0 is
    // not diagnosed here (runtime len is not yet reachable on this static
    // path).
    if (bdOp.getConstantLen() == 0)
      return bdOp.emitOpError(
          "For MM2S channels, if Buffer_Length=0 then Enable_Packet must be "
          "set to 0, otherwise behavior is undefined (according to the "
          "architecture specification)");
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetPkt, &dmaTileBd,
                            XAie_PacketInit(*packetID, *packetType));
  }
  if (std::optional<int32_t> oooId = bdOp.getOutOfOrderId())
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetOutofOrderBdId, &dmaTileBd,
                            static_cast<uint8_t>(*oooId));
  // In AIE4, Valid_BD is assumed always Enabled
  if (!isa<AIE::AIE4TargetModel>(targetModel))
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaEnableBd, &dmaTileBd);

  // If it's AIE4 mem tile, then it has a private bd pool and
  // XAie_DmaWriteBdPvtBuffPool aie-rt is used instead of XAie_DmaWriteBd
  auto tileLoc = XAie_TileLoc(col, row);
  if (isa<AIE::AIE4TargetModel>(targetModel) &&
      (targetModel.isMemTile(col, row) ||
       (targetModel.isShimNOCTile(col, row) &&
        (dir == AIE::DMAChannelDir::ControlMM2S ||
         dir == AIE::DMAChannelDir::TraceS2MM)))) {
    XAie_DmaDirection d = toXAieDmaDirection(dir);
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaWriteBdPvtBuffPool, devInst,
                            &dmaTileBd, tileLoc, channel, d, bdId);
  } else {
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaWriteBd, devInst, &dmaTileBd, tileLoc,
                            bdId);
  }
  LLVM_DEBUG(llvm::dbgs() << "\nend configuring bds\n");
  return success();
};

LogicalResult xilinx::AIE::AIERTControl::pushToBdQueueAndEnable(
    Operation &op, int col, int row, int chNum, const DMAChannelDir &channelDir,
    int bdId, int repeatCount, uint32_t padValue, bool outOfOrder,
    bool configureChannel, bool pushTask) {
  TxnLocBracket bracket(*this, op.getLoc());
  XAie_DmaDirection direction = toXAieDmaDirection(channelDir);
  auto tileLoc = XAie_TileLoc(col, row);
  if (configureChannel && padValue != 0 && direction == DMA_MM2S &&
      targetModel.isMemTile(col, row)) {
    TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaSetPadValue, &aiert->devInst, tileLoc,
                            static_cast<uint8_t>(chNum), padValue);
  }
  // in english repeat_count==0 means "do it once" and don't repeat but
  // libxaie treats repeat_count=1 as do it once.
  repeatCount += 1;
  if (outOfOrder) {
    if (configureChannel) {
      XAie_DmaChannelDesc dmaChannelDesc;
      TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelDescInit, &aiert->devInst,
                              &dmaChannelDesc, tileLoc);
      TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelEnOutofOrder, &dmaChannelDesc,
                              static_cast<uint8_t>(XAIE_ENABLE));
      TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaWriteChannel, &aiert->devInst,
                              &dmaChannelDesc, tileLoc,
                              static_cast<uint8_t>(chNum), direction);
    }
    // StartBd=0 (ignored) and skip the BD/channel validity check.
    // Disable TCT because the bd lookup id aliases the token bits.
    XAie_DmaDeclareQueueConfig(startQueueCfg, /*StartBd=*/0, repeatCount,
                               /*EnToken=*/static_cast<uint8_t>(XAIE_DISABLE),
                               static_cast<uint8_t>(XAIE_ENABLE));
    if (pushTask)
      TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelSetStartQueueGeneric,
                              &aiert->devInst, tileLoc, chNum, direction,
                              &startQueueCfg);
  } else if (pushTask) {
    bool enTokenIssue = tileLoc.Row == 0 &&
                        (direction == DMA_S2MM || direction == DMA_S2MM_TRACE);
    TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelSetStartQueue, &aiert->devInst,
                            tileLoc, chNum, direction, bdId, repeatCount,
                            enTokenIssue);
  }
  if (configureChannel && !isa<AIE4TargetModel>(targetModel))
    TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelEnable, &aiert->devInst, tileLoc,
                            chNum, direction);
  return success();
};

LogicalResult xilinx::AIE::AIERTControl::configureLocksAndBd(Block &block,
                                                             uint32_t channel,
                                                             DMAChannelDir dir,
                                                             int col, int row,
                                                             bool outOfOrder) {
  DMABDOp bd = *block.getOps<DMABDOp>().begin();
  auto bdId = bd.getBdId();
  assert(bdId.has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  TxnLocBracket bracket(*this, bd.getLoc());
  XAie_DmaDesc dmaTileBd;
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_EMIT_ERROR(bd, XAie_DmaDescInit, &aiert->devInst, &dmaTileBd,
                          tileLoc);
  if (!block.getOps<UseLockOp>().empty() &&
      failed(configureLocksInBdBlock(targetModel, dmaTileBd, block, col, row,
                                     outOfOrder)))
    return failure();
  // Out-of-order ignores Use_Next_BD; the IR chain exists only so every
  // receive BD gets configured.
  std::optional<int> nextBdId =
      outOfOrder ? std::optional<int>() : bd.getNextBdId();
  if (!block.getOps<DMABDOp>().empty() &&
      failed(configureBdInBlock(
          targetModel, &aiert->devInst, dmaTileBd, block, channel, dir, col,
          row, *bdId, // NOLINT(bugprone-unchecked-optional-access)
          nextBdId)))
    return failure();
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::initLocks(DeviceOp &targetOp) {
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
    if (!tileOp.isShimTile() && tileOp.getCoreOp() &&
        tileOp.getCoreOp().getElfFile()) {
      TxnLocBracket bracket(*this, tileOp.getLoc());
      TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreReset, &aiert->devInst, tileLoc);
      TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreUnreset, &aiert->devInst,
                              tileLoc);
      // Set locks to zero
      for (uint8_t l = 0; l < NUM_LOCKS; l++) {
        auto locInit = XAie_LockInit(l, 0);
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_LockSetValue, &aiert->devInst,
                                tileLoc, locInit);
      }
    }
  }

  // Set locks with explicit initializers
  targetOp.walkOutsideRuntimeSequences<LockOp>([&](LockOp lockOp) {
    TileOp tileOp = lockOp.getTileOp();
    if (lockOp.getLockID() && lockOp.getInit()) {
      TxnLocBracket bracket(*this, lockOp.getLoc());
      auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
      auto locInit = XAie_LockInit(*lockOp.getLockID(), *lockOp.getInit());
      TRY_XAIE_API_FATAL_ERROR(XAie_LockSetValue, &aiert->devInst, tileLoc,
                               locInit);
    } else if (zeroUninitializedLocks && lockOp.getLockID() &&
               !tileOp.isShimTile() && !tileOp.getCoreOp()) {
      // A declared lock without an initializer starts at zero rather than at
      // the count the previous dispatch left. Tiles with a core already zeroed
      // all their locks above.
      TxnLocBracket bracket(*this, lockOp.getLoc());
      auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
      auto locInit = XAie_LockInit(*lockOp.getLockID(), 0);
      TRY_XAIE_API_FATAL_ERROR(XAie_LockSetValue, &aiert->devInst, tileLoc,
                               locInit);
    } else
      LLVM_DEBUG(llvm::dbgs()
                 << "lock op missing either id or init" << lockOp << "\n");
  });
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::initBuffers(DeviceOp &targetOp) {
  // Set buffers with explicit initializers
  targetOp.walkOutsideRuntimeSequences<BufferOp>([&](BufferOp bufferOp) {
    auto initialValue = bufferOp.getInitialValue();
    if (!initialValue)
      return;
    TxnLocBracket bracket(*this, bufferOp.getLoc());
    mlir::DenseElementsAttr denseInit =
        dyn_cast<mlir::DenseElementsAttr>(initialValue.value());
    if (!denseInit)
      return;
    auto tileLoc = XAie_TileLoc(bufferOp.getTileOp().colIndex(),
                                bufferOp.getTileOp().rowIndex());
    std::optional<std::vector<char>> byteVec = denseAttrToBytes(denseInit);
    if (!byteVec) {
      llvm::outs() << "buffer op type not supported for initialization "
                   << bufferOp << "\n";
      return;
    }
    TRY_XAIE_API_FATAL_ERROR(XAie_DataMemBlockWrite, &aiert->devInst, tileLoc,
                             bufferOp.getAddress().value(), byteVec->data(),
                             byteVec->size());
  });
  return success();
}

// An AIE4 device's configuration is applied over whatever stream switch
// connections the previous device left. A connection only another device uses
// would fork this device's streams into a port nothing drains, so disable the
// connections of the other devices with the same target first. Control packet
// overlay connections stay: a caller that skips the overlay does not enable
// them again.
LogicalResult
xilinx::AIE::AIERTControl::disableOtherDeviceSwitches(DeviceOp &targetOp) {
  auto moduleOp = targetOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return success();
  for (auto otherOp : moduleOp.getOps<DeviceOp>()) {
    if (otherOp == targetOp || otherOp.getDevice() != targetOp.getDevice())
      continue;
    for (auto switchboxOp : otherOp.getOps<SwitchboxOp>()) {
      int32_t col = switchboxOp.colIndex();
      int32_t row = switchboxOp.rowIndex();
      XAie_LocType tileLoc = XAie_TileLoc(col, row);
      Block &b = switchboxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>()) {
        if (isControlPacketOverlay(connectOp))
          continue;
        TxnLocBracket bracket(*this, connectOp.getLoc());
        TRY_XAIE_API_EMIT_ERROR(
            switchboxOp, XAie_StrmConnCctDisable, &aiert->devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
            connectOp.sourceIndex(),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex());
      }
      for (auto masterSetOp : b.getOps<MasterSetOp>()) {
        if (isControlPacketOverlay(masterSetOp))
          continue;
        TxnLocBracket bracket(*this, masterSetOp.getLoc());
        TRY_XAIE_API_EMIT_ERROR(
            masterSetOp, XAie_StrmPktSwMstrPortDisable, &aiert->devInst,
            tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(masterSetOp.getDestBundle()),
            masterSetOp.destIndex());
      }
      for (auto packetRulesOp : b.getOps<PacketRulesOp>()) {
        if (isControlPacketOverlay(packetRulesOp))
          continue;
        TxnLocBracket bracket(*this, packetRulesOp.getLoc());
        auto rules = packetRulesOp.getRules().front().getOps<PacketRuleOp>();
        if (llvm::any_of(rules, [&](PacketRuleOp rule) {
              return isControlPacketOverlay(rule);
            })) {
          // A shared slave stays enabled for the installed overlay. Retire
          // only application slots, keeping every rule's physical slot index.
          int slot = 0;
          for (PacketRuleOp rule : rules) {
            if (!isControlPacketOverlay(rule))
              TRY_XAIE_API_EMIT_ERROR(rule, XAie_StrmPktSwSlaveSlotDisable,
                                      &aiert->devInst, tileLoc,
                                      WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                                          packetRulesOp.getSourceBundle()),
                                      packetRulesOp.sourceIndex(), slot);
            ++slot;
          }
          continue;
        }
        TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlavePortDisable,
                                &aiert->devInst, tileLoc,
                                WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                                    packetRulesOp.getSourceBundle()),
                                packetRulesOp.sourceIndex());
      }
    }
  }
  return success();
}

LogicalResult
xilinx::AIE::AIERTControl::configureSwitches(DeviceOp &targetOp,
                                             bool skipCtrlPktOverlay) {
  if (isa<AIE4TargetModel>(targetModel) &&
      failed(disableOtherDeviceSwitches(targetOp)))
    return failure();

  // StreamSwitch (switchbox) configuration
  for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
    int32_t col = switchboxOp.colIndex();
    int32_t row = switchboxOp.rowIndex();
    XAie_LocType tileLoc = XAie_TileLoc(col, row);
    assert(targetModel.hasProperty(AIETargetModel::IsNPU) &&
           "Only NPU currently supported");

    Block &b = switchboxOp.getConnections().front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (skipCtrlPktOverlay && isControlPacketOverlay(connectOp))
        continue;
      TxnLocBracket bracket(*this, connectOp.getLoc());
      TRY_XAIE_API_EMIT_ERROR(
          switchboxOp, XAie_StrmConnCctEnable, &aiert->devInst, tileLoc,
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
          connectOp.sourceIndex(),
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
          connectOp.destIndex());
    }

    for (auto masterSetOp : b.getOps<MasterSetOp>()) {
      if (skipCtrlPktOverlay && isControlPacketOverlay(masterSetOp))
        continue;
      TxnLocBracket bracket(*this, masterSetOp.getLoc());
      int mask = 0;
      int arbiter = -1;

      // MasterSetOp::verify guarantees every amsel names the same arbiter, so
      // folding them into one arbiter plus msel mask is safe here.
      for (auto val : masterSetOp.getAmsels()) {
        AMSelOp amsel = cast<AMSelOp>(val.getDefiningOp());
        arbiter = amsel.arbiterIndex();
        int msel = amsel.getMselValue();
        mask |= (1 << msel);
      }

      auto dropHeader = masterSetOp.keepsPktHeader()
                            ? XAIE_SS_PKT_DONOT_DROP_HEADER
                            : XAIE_SS_PKT_DROP_HEADER;
      TRY_XAIE_API_EMIT_ERROR(
          masterSetOp, XAie_StrmPktSwMstrPortEnable, &aiert->devInst, tileLoc,
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(masterSetOp.getDestBundle()),
          masterSetOp.destIndex(), dropHeader, arbiter, mask);
    }

    for (auto packetRulesOp : b.getOps<PacketRulesOp>()) {
      if (skipCtrlPktOverlay && isControlPacketOverlay(packetRulesOp))
        continue;
      TxnLocBracket bracket(*this, packetRulesOp.getLoc());
      int slot = 0;
      Block &block = packetRulesOp.getRules().front();
      // The overlay's rules take the first slots of a port they share with the
      // design's rules, and the overlay already enabled the port.
      bool overlayPort =
          skipCtrlPktOverlay &&
          llvm::any_of(block.getOps<PacketRuleOp>(), [&](PacketRuleOp rule) {
            return isControlPacketOverlay(rule);
          });
      for (auto slotOp : block.getOps<PacketRuleOp>()) {
        if (skipCtrlPktOverlay && isControlPacketOverlay(slotOp)) {
          slot++;
          continue;
        }
        AMSelOp amselOp = cast<AMSelOp>(slotOp.getAmsel().getDefiningOp());
        int arbiter = amselOp.arbiterIndex();
        int msel = amselOp.getMselValue();
        if (!overlayPort)
          TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlavePortEnable,
                                  &aiert->devInst, tileLoc,
                                  WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                                      packetRulesOp.getSourceBundle()),
                                  packetRulesOp.sourceIndex());
        auto packetInit = XAie_PacketInit(slotOp.valueInt(), /*PktType*/ 0);
        // TODO Need to better define packet id,type used here
        TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlaveSlotEnable,
                                &aiert->devInst, tileLoc,
                                WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                                    packetRulesOp.getSourceBundle()),
                                packetRulesOp.sourceIndex(), slot, packetInit,
                                slotOp.maskInt(), msel, arbiter);
        slot++;
      }
    }
  }

  for (auto muxOp : targetOp.getOps<ShimMuxOp>()) {
    // NOTE ShimMux always connects from the south as directions are
    // defined relative to the tile stream switch.
    auto tileLoc =
        XAie_TileLoc(muxOp.getTileOp().getCol(), muxOp.getTileOp().getRow());
    Block &b = muxOp.getConnections().front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (skipCtrlPktOverlay && overlay.isAttrPresent(connectOp))
        continue;
      TxnLocBracket bracket(*this, connectOp.getLoc());
      // demux!
      if (connectOp.getSourceBundle() == WireBundle::North)
        TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableAieToShimDmaStrmPort,
                                &aiert->devInst, tileLoc,
                                connectOp.sourceIndex());
      // mux
      if (connectOp.getDestBundle() == WireBundle::North)
        TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableShimDmaToAieStrmPort,
                                &aiert->devInst, tileLoc,
                                connectOp.destIndex());
    }
  }

  // Cascade configuration
  if (isa<AIE2TargetModel>(targetModel) || isa<AIE4TargetModel>(targetModel)) {
    for (auto configOp : targetOp.getOps<ConfigureCascadeOp>()) {
      TxnLocBracket bracket(*this, configOp.getLoc());
      TileOp tile = cast<TileOp>(configOp.getTile().getDefiningOp());
      auto tileLoc = XAie_TileLoc(tile.getCol(), tile.getRow());
      TRY_XAIE_API_EMIT_ERROR(
          targetOp, XAie_CoreConfigAccumulatorControl, &aiert->devInst, tileLoc,
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
              static_cast<WireBundle>(configOp.getInputDir())),
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
              static_cast<WireBundle>(configOp.getOutputDir())));
    }
  }

  return success();
}

LogicalResult
xilinx::AIE::AIERTControl::teardownSwitches(DeviceOp &targetOp,
                                            bool skipCtrlPktOverlay) {
  AIEDialect::IsCtrlPktOverlayAttrHelper overlay(targetOp.getContext());
  auto skip = [&](Operation *op) {
    return skipCtrlPktOverlay && overlay.isAttrPresent(op);
  };
  for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
    XAie_LocType tileLoc =
        XAie_TileLoc(switchboxOp.colIndex(), switchboxOp.rowIndex());
    Block &b = switchboxOp.getConnections().front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (skip(connectOp))
        continue;
      TxnLocBracket bracket(*this, connectOp.getLoc());
      TRY_XAIE_API_EMIT_ERROR(
          switchboxOp, XAie_StrmConnCctDisable, &aiert->devInst, tileLoc,
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
          connectOp.sourceIndex(),
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
          connectOp.destIndex());
    }
    for (auto masterSetOp : b.getOps<MasterSetOp>()) {
      if (skip(masterSetOp))
        continue;
      TxnLocBracket bracket(*this, masterSetOp.getLoc());
      TRY_XAIE_API_EMIT_ERROR(
          masterSetOp, XAie_StrmPktSwMstrPortDisable, &aiert->devInst, tileLoc,
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(masterSetOp.getDestBundle()),
          masterSetOp.destIndex());
    }
    for (auto packetRulesOp : b.getOps<PacketRulesOp>()) {
      auto rules = packetRulesOp.getRules().front().getOps<PacketRuleOp>();
      if (rules.empty() || skip(packetRulesOp))
        continue;
      TxnLocBracket bracket(*this, packetRulesOp.getLoc());
      StrmSwPortType port =
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(packetRulesOp.getSourceBundle());
      if (llvm::none_of(rules, skip))
        TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlavePortDisable,
                                &aiert->devInst, tileLoc, port,
                                packetRulesOp.sourceIndex());
      // A slot stays enabled when its port is disabled.
      for (auto [slot, rule] : llvm::enumerate(rules))
        if (!skip(rule))
          TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlaveSlotDisable,
                                  &aiert->devInst, tileLoc, port,
                                  packetRulesOp.sourceIndex(), slot);
    }
  }
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::disableCores(DeviceOp &targetOp) {
  for (auto coreOp : targetOp.getOps<CoreOp>()) {
    TileID tile = coreOp.getTileID();
    TxnLocBracket bracket(*this, coreOp.getLoc());
    TRY_XAIE_API_EMIT_ERROR(coreOp, XAie_CoreDisable, &aiert->devInst,
                            XAie_TileLoc(tile.col, tile.row));
  }
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetDmaChannels(DeviceOp &targetOp) {
  auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
  llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
  for (TileElement memOp : memOps) {
    TileID tile = memOp.getTileID();
    XAie_LocType tileLoc = XAie_TileLoc(tile.col, tile.row);
    llvm::SmallSet<std::pair<DMAChannelDir, int>, 4> reset;
    auto resetChannel = [&](Operation *op, DMAChannelDir dir,
                            int channel) -> LogicalResult {
      if (!reset.insert({dir, channel}).second)
        return success();
      TxnLocBracket bracket(*this, op->getLoc());
      XAie_DmaDirection direction =
          dir == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S;
      TRY_XAIE_API_EMIT_ERROR((*op), XAie_DmaChannelReset, &aiert->devInst,
                              tileLoc, channel, direction,
                              XAie_DmaChReset::DMA_CHANNEL_RESET);
      // Writing a fresh descriptor ends the reset and returns every field of
      // the control register, such as out-of-order mode, to its reset value.
      XAie_DmaChannelDesc defaults;
      TRY_XAIE_API_EMIT_ERROR((*op), XAie_DmaChannelDescInit, &aiert->devInst,
                              &defaults, tileLoc);
      TRY_XAIE_API_EMIT_ERROR((*op), XAie_DmaWriteChannel, &aiert->devInst,
                              &defaults, tileLoc, channel, direction);
      return success();
    };
    Region &region = memOp.getOperation()->getRegion(0);
    for (auto dmaOp : region.getOps<DMAOp>())
      if (failed(resetChannel(dmaOp, dmaOp.getChannelDir(),
                              dmaOp.getChannelIndex())))
        return failure();
    for (auto &[block, channel, dir] : getOrderedChainOfBlocks(&region))
      for (auto startOp : block->getOps<DMAStartOp>())
        if (failed(resetChannel(startOp, startOp.getChannelDir(),
                                startOp.getChannelIndex())))
          return failure();
  }
  return success();
}

LogicalResult
xilinx::AIE::AIERTControl::addInitConfig(DeviceOp &targetOp,
                                         bool skipCtrlPktOverlay) {

  if (failed(verifyDMAChannelsResolved(targetOp)))
    return failure();

  // Core tiles that receive control packets on their TileControl port need the
  // core processor bus enabled so control packets can reach the tile's
  // memory-mapped registers. (Isolation registers are deliberately left alone
  // on medusa; see "Don't set isolation registers on medusa".)
  if (isa<AIE4TargetModel>(targetOp.getTargetModel())) {
    for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
      int col = switchboxOp.colIndex();
      int row = switchboxOp.rowIndex();
      if (!targetModel.isCoreTile(col, row))
        continue;
      Block &b = switchboxOp.getConnections().front();
      if (!hasTileControlMasterSet(b))
        continue;
      constexpr uint32_t coreProcessorBusOffset = 0x000A3060;
      TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_Write32, &aiert->devInst,
                              targetModel.getTileAddress(col, row) +
                                  coreProcessorBusOffset,
                              /*Enable=*/1);
    }
  }
  if (failed(initLocks(targetOp))) {
    return failure();
  }

  if (failed(initBuffers(targetOp))) {
    return failure();
  }

  // A DMA whose lock is already satisfied streams as soon as it is enabled, so
  // its route must exist first.
  if (failed(configureSwitches(targetOp, skipCtrlPktOverlay))) {
    return failure();
  }

  auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
  llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
  llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());
  for (TileElement memOp : memOps) {
    int col = memOp.getTileID().col;
    int row = memOp.getTileID().row;

    // Get the region's entry block, then start traversing through the chain of
    // blocks.
    llvm::SetVector<std::tuple<Block *, uint32_t, DMAChannelDir>> blockVector =
        getOrderedChainOfBlocks(&memOp->getRegion(0));

    llvm::SmallPtrSet<Block *, 8> unchainedBlocks =
        collectUnchainedBlocks(blockVector);

    // handle DMA ops separately
    auto dmaOps =
        llvm::to_vector_of<DMAOp>(memOp->getRegion(0).getOps<DMAOp>());
    if (!dmaOps.empty()) {
      for (auto dmaOp : dmaOps)
        for (auto &bdRegion : dmaOp.getBds()) {
          Block &block = bdRegion.getBlocks().front();
          if (failed(configureLocksAndBd(block, dmaOp.getChannelIndex(),
                                         dmaOp.getChannelDir(), col, row,
                                         dmaOp.getOutOfOrder())))
            return failure();
        }
    } else {
      // Build a map from BD blocks to all (channel, dir) pairs that reference
      // them. Block merging (canonicalization) may cause multiple DMA channels
      // to share the same BD blocks, but each channel's private BD pool must
      // be configured separately.
      llvm::DenseMap<Block *, SmallVector<std::pair<uint32_t, DMAChannelDir>>>
          blockChannelMap;
      for (auto &t : blockVector) {
        Block *block = std::get<0>(t);
        for (auto startOp : block->getOps<DMAStartOp>()) {
          uint32_t ch = startOp.getChannelIndex();
          DMAChannelDir d = startOp.getChannelDir();
          // Walk the BD chain from this DMAStartOp's destination.
          Block *bdBlock = startOp.getDest();
          llvm::SmallPtrSet<Block *, 8> visited;
          while (bdBlock && visited.insert(bdBlock).second) {
            blockChannelMap[bdBlock].push_back({ch, d});
            auto successors = bdBlock->getTerminator()->getSuccessors();
            bdBlock = successors.empty() ? nullptr : successors[0];
          }
        }
      }

      for (auto &t : blockVector) {
        Block *block = std::get<0>(t);
        if (block->getOps<DMABDOp>().empty())
          continue;
        auto it = blockChannelMap.find(block);
        if (it != blockChannelMap.end()) {
          for (auto &[ch, d] : it->second) {
            if (failed(configureLocksAndBd(*block, ch, d, col, row,
                                           unchainedBlocks.contains(block))))
              return failure();
          }
        } else {
          uint32_t channel = std::get<1>(t);
          DMAChannelDir dir = std::get<2>(t);
          if (failed(configureLocksAndBd(*block, channel, dir, col, row,
                                         unchainedBlocks.contains(block))))
            return failure();
        }
      }
    }

    if (!dmaOps.empty())
      for (auto dmaOp : dmaOps) {
        auto &block = dmaOp.getBds().front().getBlocks().front();
        DMABDOp bd = *block.getOps<DMABDOp>().begin();
        auto bdId = bd.getBdId();
        assert(bdId.has_value() &&
               "bd_id assigned by configureLocksAndBd above");
        if (failed(pushToBdQueueAndEnable(
                *dmaOp.getOperation(), col, row, dmaOp.getChannelIndex(),
                dmaOp.getChannelDir(), bdId.value(), dmaOp.getRepeatCount(),
                dmaOp.getPadValue(), dmaOp.getOutOfOrder())))
          return failure();
      }
    else
      for (auto &t : blockVector) {
        Block *block = std::get<0>(t);
        for (auto op : block->getOps<DMAStartOp>()) {
          DMABDOp bd = *op.getDest()->getOps<DMABDOp>().begin();
          int chNum = op.getChannelIndex();
          auto channelDir = op.getChannelDir();
          auto bdId = bd.getBdId();
          assert(bdId.has_value() &&
                 "bd_id assigned by configureLocksAndBd above");
          if (failed(pushToBdQueueAndEnable(
                  *bd.getOperation(), col, row, chNum, channelDir, bdId.value(),
                  op.getRepeatCount(), op.getPadValue(), op.getOutOfOrder(),
                  /*configureChannel=*/true,
                  /*pushTask=*/!op.getCorePushes())))
            return failure();
        }
      }
  }

  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addCoreEnable(DeviceOp &targetOp) {
  // Start execution of all the cores that have an ELF loaded.
  // Skip cores without elf_file to avoid enabling a core that has no program
  // (which causes the core to run off the end of memory).
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
    if (!tileOp.isShimTile() && tileOp.getCoreOp() &&
        tileOp.getCoreOp().getElfFile()) {
      CoreOp coreOp = tileOp.getCoreOp();
      TxnLocBracket bracket(*this, coreOp.getLoc());
      // A core reaching its own tile's registers goes through the processor
      // bus, which starts disabled: an access made before this stalls the
      // core. Only the cores that ask for it pay for it.
      if (coreOp.getProcessorBus())
        TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreProcessorBusEnable,
                                &aiert->devInst, tileLoc);
      TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreEnable, &aiert->devInst,
                              tileLoc);
    }
  }
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addUcElf(uint32_t col,
                                                  const StringRef elfPath) {
  TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadUc, &aiert->devInst,
                              XAie_TileLoc(col, 0), elfPath.str().c_str());
  TRY_XAIE_API_LOGICAL_RESULT(XAie_Write32, &aiert->devInst,
                              (col << 25) | 0x0002C004, 1);
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addCoreElf(uint32_t col, uint32_t row,
                                                    const StringRef elfPath,
                                                    bool aieSim) {
  TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreDisable, &aiert->devInst,
                              XAie_TileLoc(col, row));
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelResetAll, &aiert->devInst,
                              XAie_TileLoc(col, row),
                              XAie_DmaChReset::DMA_CHANNEL_RESET);

  // loadSym: Load symbols from .map file. This argument is not used when
  // __AIESIM__ is not defined.
  TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, &aiert->devInst,
                              XAie_TileLoc(col, row), elfPath.str().c_str(),
                              /*loadSym*/ false);

  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelResetAll, &aiert->devInst,
                              XAie_TileLoc(col, row),
                              XAie_DmaChReset::DMA_CHANNEL_UNRESET);

  return success();
}

// Upper bound for each core reset readback. aie-rt does not serialize it:
// aiex.npu.maskpoll and cert MASK_POLL_32 carry no timeout.
static constexpr uint32_t kQuiescePollTimeoutUs = 1000;

// Partition-wide quiesce at the head of an AIE4 configuration. Every tile in
// the partition is visited, not only those the device declares: a previous
// design may have left cores running or DMA channels armed on tiles this one
// does not use. Each phase completes over all tiles before the next starts.
LogicalResult xilinx::AIE::AIERTControl::addColumnQuiesce(DeviceOp &targetOp,
                                                          DeviceOp incoming) {
  if (!isa<AIE4TargetModel>(targetModel))
    return targetOp.emitOpError(
        "the partition reset is only written as registers on AIE4; elsewhere "
        "aie-expand-load-pdi lowers aiex.npu.write_reset to the load of an "
        "empty device");
  TxnLocBracket bracket(*this, targetOp.getLoc());

  int numCols = targetModel.columns();
  int numRows = targetModel.rows();
  // Register offsets are tile-relative and identical in every column, so
  // resolve them once on column 0's first tile of each kind.
  auto firstRow = [&](auto isKind) -> std::optional<TileID> {
    for (int row = 0; row < numRows; row++)
      if (isKind(0, row))
        return TileID{0, row};
    return std::nullopt;
  };
  std::optional<TileID> coreTile = firstRow(
      [&](int col, int row) { return targetModel.isCoreTile(col, row); });
  std::optional<TileID> memTile = firstRow(
      [&](int col, int row) { return targetModel.isMemTile(col, row); });
  // Only NOC shims have DMA channels; PL shims have none to quiesce.
  std::optional<TileID> shimTile = firstRow(
      [&](int col, int row) { return targetModel.isShimNOCTile(col, row); });
  if (!coreTile || !memTile || !shimTile)
    return targetOp.emitOpError(
        "configure-time DMA reset needs core, memory and shim tiles");

  const RegisterInfo *coreCtl =
      targetModel.lookupRegister("Core_Control", *coreTile, /*isMem=*/false);
  if (!coreCtl)
    return targetOp.emitOpError(
        "register database has no Core_Control for core tiles");
  const BitFieldInfo *coreReset = coreCtl->getField("Reset");
  const BitFieldInfo *coreEnable = coreCtl->getField("Enable");
  if (!coreReset || !coreEnable)
    return targetOp.emitOpError(
        "Core_Control has no Reset or Enable field in the register database");
  std::optional<uint32_t> coreResetMask = targetModel.getFieldMask(*coreReset);
  std::optional<uint32_t> coreEnableMask =
      targetModel.getFieldMask(*coreEnable);
  if (!coreResetMask || !coreEnableMask)
    return targetOp.emitOpError(
        "Core_Control Reset or Enable field does not fit in 32 bits");
  // Reset asserted, enable clear; the readback checks both.
  auto coreAssert = targetModel.encodeFieldValue(*coreReset, 1);
  if (!coreAssert)
    return targetOp.emitOpError(
        "Core_Control Reset value does not fit in the register field");
  uint32_t corePollMask = *coreResetMask | *coreEnableMask;

  // The number of data channels in each direction of `tile`'s kind, counted
  // from the register database's DMA_{S2MM,MM2S}_<n>_Ctrl registers. aie-rt
  // numbers the shim's control-packet and trace channels after the data
  // channels, so only these may be reset.
  struct DmaChannels {
    int s2mm = 0, mm2s = 0;
  };
  auto countDmaChannels = [&](TileID tile, StringRef kind,
                              DmaChannels &channels) -> LogicalResult {
    for (auto [prefix, count] :
         {std::pair<StringRef, int *>{"DMA_S2MM_", &channels.s2mm},
          std::pair<StringRef, int *>{"DMA_MM2S_", &channels.mm2s}}) {
      while (targetModel.lookupRegister(
          (Twine(prefix) + Twine(*count) + "_Ctrl").str(), tile,
          /*isMem=*/true))
        ++*count;
      if (*count == 0)
        return targetOp.emitOpError() << "register database has no " << prefix
                                      << "0_Ctrl for " << kind << " tiles";
    }
    return success();
  };
  // The memtile MM2S channels' DMA_MM2S_<n>_Constant_Pad_Value registers.
  // Starting a BD queue entry only writes the pad value when it is non-zero,
  // so a previous design's pad would otherwise leak into this one. Probed by
  // name like the Ctrl registers and checked against the MM2S Ctrl count.
  SmallVector<uint32_t> memPadValue;
  for (int n = 0;; n++) {
    std::string name =
        (Twine("DMA_MM2S_") + Twine(n) + "_Constant_Pad_Value").str();
    const RegisterInfo *reg =
        targetModel.lookupRegister(name, *memTile, /*isMem=*/true);
    if (!reg)
      break;
    memPadValue.push_back(reg->offset);
  }
  if (memPadValue.empty())
    return targetOp.emitOpError(
        "register database has no DMA_MM2S_0_Constant_Pad_Value for memory "
        "tiles");
  auto memMm2sCtrl = [&](size_t n) {
    return targetModel.lookupRegister(
        (Twine("DMA_MM2S_") + Twine(n) + "_Ctrl").str(), *memTile,
        /*isMem=*/true);
  };
  if (!memMm2sCtrl(memPadValue.size() - 1) || memMm2sCtrl(memPadValue.size()))
    return targetOp.emitOpError(
        "register database has a different number of memory tile MM2S "
        "channels and constant pad value registers");
  // Phase S's registers: the stream switch registers of both the 512b data
  // switch and the 32b switch (port configuration, packet slots,
  // deterministic merge; see getStreamSwitchRegisters). All reset to 0, which
  // disables every port, slot and arbiter, so writing 0 returns the switch to
  // its fresh-context state. Grouped into runs of consecutive offsets; a run
  // never covers an offset without a register (the address map has holes,
  // and an access to one is a slave error).
  struct SwitchRun {
    uint32_t offset;
    uint32_t words;
  };
  //
  // A packet slot matters only on a subordinate port in packet mode, and S
  // takes every port out of packet mode. With the incoming device known, the
  // slots are cleared only on the tiles where its configuration puts ports in
  // packet mode again (it has packet rules there, the control-packet overlay's
  // included); elsewhere a stale slot stays inert.
  auto collectSwitchRuns =
      [&](TileID tile, StringRef kind, bool slots,
          SmallVectorImpl<SwitchRun> &runs) -> LogicalResult {
    for (const RegisterInfo *reg : targetModel.getStreamSwitchRegisters(tile)) {
      StringRef name = reg->name;
      if (!slots && name.contains("_Slot"))
        continue;
      uint64_t resetValue;
      if (reg->width != 32 ||
          StringRef(reg->reset).getAsInteger(0, resetValue) || resetValue != 0)
        return targetOp.emitOpError()
               << name << " of " << kind
               << " tiles is not a 32-bit register that resets to 0 in the "
                  "register database";
      if (!runs.empty() &&
          runs.back().offset + 4 * runs.back().words == reg->offset)
        runs.back().words++;
      else
        runs.push_back({reg->offset, 1});
    }
    if (runs.empty())
      return targetOp.emitOpError()
             << "register database has no stream switch registers for " << kind
             << " tiles";
    return success();
  };
  // Indexed by whether the runs include the slots.
  std::array<SmallVector<SwitchRun>, 2> coreSwitch, memSwitch, shimSwitch;
  for (bool slots : {false, true})
    if (failed(
            collectSwitchRuns(*coreTile, "core", slots, coreSwitch[slots])) ||
        failed(
            collectSwitchRuns(*memTile, "memory", slots, memSwitch[slots])) ||
        failed(collectSwitchRuns(*shimTile, "shim", slots, shimSwitch[slots])))
      return failure();
  std::set<std::pair<int, int>> packetTiles;
  if (incoming)
    for (auto switchbox : incoming.getOps<SwitchboxOp>())
      if (!switchbox.getConnections().front().getOps<PacketRulesOp>().empty())
        packetTiles.insert({switchbox.colIndex(), switchbox.rowIndex()});
  auto clearsSlots = [&](int col, int row) {
    return !incoming || packetTiles.count({col, row});
  };

  DmaChannels coreChannels, memChannels, shimChannels;
  if (failed(countDmaChannels(*coreTile, "core", coreChannels)) ||
      failed(countDmaChannels(*memTile, "memory", memChannels)) ||
      failed(countDmaChannels(*shimTile, "shim", shimChannels)))
    return failure();

  // Tiles, column-major with rows ascending.
  SmallVector<XAie_LocType> coreTiles, memTiles, shimTiles;
  // The core and memory tiles, with their data channel counts.
  SmallVector<std::pair<XAie_LocType, DmaChannels>> dmaTiles;
  // Every tile, with its stream switch register runs.
  SmallVector<std::pair<XAie_LocType, ArrayRef<SwitchRun>>> switchTiles;
  for (int col = 0; col < numCols; col++) {
    for (int row = 0; row < numRows; row++) {
      XAie_LocType tile = XAie_TileLoc(col, row);
      if (targetModel.isCoreTile(col, row)) {
        coreTiles.push_back(tile);
        dmaTiles.push_back({tile, coreChannels});
        switchTiles.push_back({tile, coreSwitch[clearsSlots(col, row)]});
      } else if (targetModel.isMemTile(col, row)) {
        memTiles.push_back(tile);
        dmaTiles.push_back({tile, memChannels});
        switchTiles.push_back({tile, memSwitch[clearsSlots(col, row)]});
      } else if (targetModel.isShimNOCTile(col, row)) {
        // Only NoC shim tiles: the register layout of a PL shim tile may
        // differ, and a write to an undefined offset is a slave error.
        shimTiles.push_back(tile);
        switchTiles.push_back({tile, shimSwitch[clearsSlots(col, row)]});
      }
    }
  }
  auto forEachChannel = [&](DmaChannels channels, auto fn) -> LogicalResult {
    for (auto [dir, count] : {std::pair{DMA_S2MM, channels.s2mm},
                              std::pair{DMA_MM2S, channels.mm2s}})
      for (int ch = 0; ch < count; ch++)
        if (failed(fn(static_cast<u8>(ch), dir)))
          return failure();
    return success();
  };
  // Writing a channel's default descriptor clears its Reset field and returns
  // every other field of its control register, such as out-of-order mode, to
  // its reset value (as AIERTControl::resetDmaChannels does for a teardown).
  auto writeDefaultChannel = [&](XAie_LocType tile, u8 ch,
                                 XAie_DmaDirection dir) -> LogicalResult {
    XAie_DmaChannelDesc defaults;
    TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_DmaChannelDescInit, &aiert->devInst,
                            &defaults, tile);
    TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_DmaWriteChannel, &aiert->devInst,
                            &defaults, tile, ch, dir);
    return success();
  };

  // Q0: hold every core in reset with its enable clear.
  for (XAie_LocType tile : coreTiles) {
    TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreReset, &aiert->devInst, tile);
    TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreDisable, &aiert->devInst, tile);
  }
  // QB: read each Core_Control back, so the core resets have reached the
  // array before any channel is reset.
  for (XAie_LocType tile : coreTiles)
    TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_MaskPoll, &aiert->devInst,
                            targetModel.getTileAddress(tile.Col, tile.Row) +
                                coreCtl->offset,
                            corePollMask, *coreAssert, kQuiescePollTimeoutUs);
  // S: clear every stream switch in the partition, one block set per run of
  // consecutive registers. The configuration reinstalls this device's routes
  // (control-packet overlay included) in configureSwitches. Stream data left
  // in switch FIFOs is not removed: the outgoing design must have drained its
  // streams.
  for (auto [tile, runs] : switchTiles)
    for (const SwitchRun &run : runs)
      TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_BlockSet32, &aiert->devInst,
                              targetModel.getTileAddress(tile.Col, tile.Row) +
                                  run.offset,
                              0, run.words);
  // Q1: assert the reset of every core-tile and memtile data channel, and
  // only then (Q2) release them, writing each channel's default mode.
  for (auto [tile, channels] : dmaTiles)
    if (failed(forEachChannel(
            channels, [&](u8 ch, XAie_DmaDirection dir) -> LogicalResult {
              TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_DmaChannelReset,
                                      &aiert->devInst, tile, ch, dir,
                                      DMA_CHANNEL_RESET);
              return success();
            })))
      return failure();
  for (auto [tile, channels] : dmaTiles)
    if (failed(forEachChannel(
            channels, [&](u8 ch, XAie_DmaDirection dir) -> LogicalResult {
              return writeDefaultChannel(tile, ch, dir);
            })))
      return failure();
  // Q2p: clear every memtile MM2S channel's constant pad value, which the
  // channel descriptor does not cover.
  for (XAie_LocType tile : memTiles)
    for (uint32_t offset : memPadValue)
      TRY_XAIE_API_EMIT_ERROR(
          targetOp, XAie_Write32, &aiert->devInst,
          targetModel.getTileAddress(tile.Col, tile.Row) + offset, 0);
  // Q2s: shim channels have no Reset field; return their modes to default.
  for (XAie_LocType tile : shimTiles)
    if (failed(forEachChannel(
            shimChannels, [&](u8 ch, XAie_DmaDirection dir) -> LogicalResult {
              return writeDefaultChannel(tile, ch, dir);
            })))
      return failure();
  // Q3: release the cores' reset and leave them disabled; the configuration
  // that follows enables the cores it loads.
  for (XAie_LocType tile : coreTiles)
    TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreUnreset, &aiert->devInst, tile);
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetPartition() {
  TRY_XAIE_API_LOGICAL_RESULT(XAie_ResetPartition, &aiert->devInst);
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetDMA(int col, int row, bool on) {
  auto tileLoc = XAie_TileLoc(col, row);
  XAie_DmaDesc dmaTileBd;
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaDescInit, &aiert->devInst, &dmaTileBd,
                              tileLoc);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaDisableBd, &dmaTileBd);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelResetAll, &aiert->devInst, tileLoc,
                              on ? XAie_DmaChReset::DMA_CHANNEL_UNRESET
                                 : XAie_DmaChReset::DMA_CHANNEL_RESET);
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetCore(int col, int row) {
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreReset, &aiert->devInst, tileLoc);
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetSwitch(int col, int row) {
  XAie_LocType tileLoc = XAie_TileLoc(col, row);

  // Reset all combinations of input/output routing in the switchbox
  for (auto endpoint_a : WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE) {
    for (auto endpoint_b : WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE) {
      unsigned n_a_connections = targetModel.getNumSourceSwitchboxConnections(
          col, row, endpoint_a.first);
      unsigned n_b_connections = targetModel.getNumDestSwitchboxConnections(
          col, row, endpoint_b.first);
      for (unsigned a_index = 0; a_index < n_a_connections; a_index++) {
        for (unsigned b_index = 0; b_index < n_b_connections; b_index++) {
          if (!targetModel.isLegalTileConnection(col, row, endpoint_a.first,
                                                 a_index, endpoint_b.first,
                                                 b_index)) {
            continue;
          }
          TRY_XAIE_API_FATAL_ERROR(XAie_StrmConnCctDisable, &aiert->devInst,
                                   tileLoc, endpoint_a.second, a_index,
                                   endpoint_b.second, b_index);
        }
      }
    }
  }

  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetCoreUnreset(int col, int row) {
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreUnreset, &aiert->devInst, tileLoc);
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetLock(int col, int row,
                                                   int lockId) {
  auto tileLoc = XAie_TileLoc(col, row);
  // Reset a single lock to value 0
  XAie_Lock lock;
  lock.LockId = lockId;
  lock.LockVal = 0;
  TRY_XAIE_API_LOGICAL_RESULT(XAie_LockSetValue, &aiert->devInst, tileLoc,
                              lock);
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetSwitchConnection(
    int col, int row, WireBundle sourceBundle, int sourceChannel,
    WireBundle destBundle, int destChannel) {
  auto tileLoc = XAie_TileLoc(col, row);

  // Helper lambda to map WireBundle to StrmSwPortType
  auto mapBundle = [](WireBundle bundle) -> StrmSwPortType {
    switch (bundle) {
    case WireBundle::Core:
      return CORE;
    case WireBundle::DMA:
      return DMA;
    case WireBundle::FIFO:
      return FIFO;
    case WireBundle::South:
      return SOUTH;
    case WireBundle::West:
      return WEST;
    case WireBundle::North:
      return NORTH;
    case WireBundle::East:
      return EAST;
    case WireBundle::Trace:
      return TRACE;
    default:
      return SOUTH;
    }
  };

  StrmSwPortType sourcePortType = mapBundle(sourceBundle);
  StrmSwPortType destPortType = mapBundle(destBundle);

  // Disconnect the specific connection from source to destination
  TRY_XAIE_API_LOGICAL_RESULT(XAie_StrmConnCctDisable, &aiert->devInst, tileLoc,
                              sourcePortType, sourceChannel, destPortType,
                              destChannel);

  return success();
}

LogicalResult xilinx::AIE::AIERTControl::resetPerfCounters(int col, int row) {
  auto tileLoc = XAie_TileLoc(col, row);
  // Reset performance counters in all modules
  // Try core module counters (if applicable)
  for (int counterId = 0; counterId < 4; counterId++) {
    // Ignore errors as not all tiles have all counter types
    (void)XAie_PerfCounterReset(&aiert->devInst, tileLoc, XAIE_CORE_MOD,
                                counterId);
  }
  // Try mem module counters
  for (int counterId = 0; counterId < 4; counterId++) {
    (void)XAie_PerfCounterReset(&aiert->devInst, tileLoc, XAIE_MEM_MOD,
                                counterId);
  }
  // Try PL module counters (for shim tiles)
  for (int counterId = 0; counterId < 4; counterId++) {
    (void)XAie_PerfCounterReset(&aiert->devInst, tileLoc, XAIE_PL_MOD,
                                counterId);
  }
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addAieElfs(DeviceOp &targetOp,
                                                    const StringRef elfPath,
                                                    bool aieSim) {
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    if (row == 0 && tileOp->hasAttr("elf_file")) {
      std::string fileName =
          tileOp->getAttrOfType<mlir::StringAttr>("elf_file").str();
      auto ps = std::filesystem::path::preferred_separator;
      fileName = elfPath.str() + std::string(1, ps) + fileName;
      std::cout << fileName << "\n";
      if (failed(addUcElf(col, fileName)))
        return failure();
      continue;
    }
    if (auto coreOp = tileOp.getCoreOp()) {
      std::string fileName;
      if (auto fileAttr = coreOp.getElfFile()) {
        fileName = fileAttr->str();
      } else {
        // Skip cores without elf_file (e.g., lightweight reset
        // devices that only need DMA/lock reconfiguration).
        continue;
      }
      // Check if fileName is already an absolute path.
      // If so, use it directly. Otherwise, concatenate with elfPath.
      std::string fullPath;
      if (std::filesystem::path(fileName).is_absolute()) {
        fullPath = fileName;
      } else {
        auto ps = std::filesystem::path::preferred_separator;
        fullPath = (llvm::Twine(elfPath) + std::string(1, ps) + fileName).str();
      }
      if (failed(addCoreElf(col, row, fullPath, aieSim)))
        return failure();
    }
  }
  return success();
}

void xilinx::AIE::AIERTControl::dmaUpdateBdAddr(int col, int row, size_t addr,
                                                size_t bdId) {
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_FATAL_ERROR(XAie_DmaUpdateBdAddr, &aiert->devInst, tileLoc, addr,
                           bdId);
}

void xilinx::AIE::AIERTControl::startTransaction() {
  aiert->txnCmdLocs.clear();
  aiert->txnOpLocs.clear();

  // Workaround this problem:
  //  [AIE ERROR] _XAie_LoadDataMemSection():337: Unable to turn ECC On for Data
  //  Memory XAie_LoadElf failed with XAIE_FEATURE_NOT_SUPPORTED
  TRY_XAIE_API_FATAL_ERROR(XAie_TurnEccOff, &aiert->devInst);

  TRY_XAIE_API_FATAL_ERROR(XAie_StartTransaction, &aiert->devInst,
                           XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
}

// Project the per-XAie_TxnCmd locations in `cmdLocs` onto the operations
// aie-rt's serializer actually emits, mirroring _XAie_TxnExportSerialized's
// command-to-operation mapping. That mapping is not 1:1: the serializer
// buffers block writes and collapses a run of address-contiguous
// XAIE_IO_BLOCKWRITE commands into a single block-write operation, holds that
// pending run across intervening DDR-patch commands (so a patch is emitted
// *before* the block write it follows in CmdBuf), and drops
// XAIE_IO_LOAD_PM_END_INTERNAL entirely. Everything else emits one operation
// in command order.
static std::vector<mlir::Location>
projectCmdLocsOntoSerializedOps(const XAie_TxnInst &inst,
                                const std::vector<mlir::Location> &cmdLocs) {
  std::vector<mlir::Location> opLocs;
  opLocs.reserve(inst.NumCmds);

  mlir::Location unknown = mlir::UnknownLoc::get(cmdLocs.front().getContext());
  bool pendingBlockWrite = false;
  mlir::Location pendingBlockWriteLoc = unknown;
  uint64_t blockWriteEndOff = 0;

  for (uint32_t i = 0; i < inst.NumCmds; ++i) {
    const XAie_TxnCmd &cmd = inst.CmdBuf[i];
    mlir::Location loc = i < cmdLocs.size() ? cmdLocs[i] : unknown;

    if (cmd.Opcode == XAIE_IO_BLOCKWRITE) {
      // A command that extends the run in flight emits nothing of its own;
      // the run keeps the location of the command that started it.
      if (!pendingBlockWrite || cmd.RegOff != blockWriteEndOff) {
        if (pendingBlockWrite)
          opLocs.push_back(pendingBlockWriteLoc);
        pendingBlockWrite = true;
        pendingBlockWriteLoc = loc;
      }
      blockWriteEndOff = cmd.RegOff + cmd.Size * sizeof(uint32_t);
      continue;
    }

    if (cmd.Opcode == XAIE_IO_LOAD_PM_END_INTERNAL)
      continue;

    if (pendingBlockWrite && cmd.Opcode != XAIE_IO_CUSTOM_OP_DDR_PATCH) {
      opLocs.push_back(pendingBlockWriteLoc);
      pendingBlockWrite = false;
    }
    opLocs.push_back(loc);
  }
  if (pendingBlockWrite)
    opLocs.push_back(pendingBlockWriteLoc);

  return opLocs;
}

std::vector<uint8_t> xilinx::AIE::AIERTControl::exportSerializedTransaction() {
  // Export the transactions to a binary buffer
  uint8_t *txn_ptr = XAie_ExportSerializedTransaction(&aiert->devInst, 0, 0);
  XAie_TxnHeader *hdr = (XAie_TxnHeader *)txn_ptr;
  std::vector<uint8_t> txn_data(txn_ptr, txn_ptr + hdr->TxnSize);
  free(txn_ptr);

  // Exporting leaves the recorded commands intact, so the loc projection can
  // still walk CmdBuf here. With nothing bracketed there is no location to
  // project and every emitted op falls back to the device location.
  XAie_TxnInst *inst = getActiveTxnInst(aiert->devInst);
  if (inst && !aiert->txnCmdLocs.empty())
    aiert->txnOpLocs =
        projectCmdLocsOntoSerializedOps(*inst, aiert->txnCmdLocs);

  return txn_data;
}

const std::vector<mlir::Location> &
xilinx::AIE::AIERTControl::getTxnOpLocs() const {
  return aiert->txnOpLocs;
}

uint32_t xilinx::AIE::AIERTControl::getCurrentTxnNumCmds() const {
  return getActiveTxnNumCmds(aiert->devInst);
}

void xilinx::AIE::AIERTControl::recordTxnLocRange(uint32_t startCmds,
                                                  uint32_t endCmds,
                                                  mlir::Location loc) {
  if (endCmds <= startCmds)
    return;
  if (aiert->txnCmdLocs.size() < endCmds)
    aiert->txnCmdLocs.resize(endCmds, mlir::UnknownLoc::get(loc.getContext()));
  for (uint32_t i = startCmds; i < endCmds; ++i)
    aiert->txnCmdLocs[i] = loc;
}
