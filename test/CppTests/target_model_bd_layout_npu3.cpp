//===- target_model_bd_layout_npu3.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 BD layouts vs aie-codegen's register database: word, shift and width
// of every modelled field, and absence of the fields AIE4 BDs lack.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "xaie4gbl_params.h"

#include <cstdint>
#include <stdexcept>
#include <string>

using namespace xilinx;
using namespace xilinx::AIE;

static void check(bool cond, const std::string &what) {
  if (!cond)
    throw std::runtime_error("Failed: " + what);
}

static void checkField(const std::string &what, const DmaBdField &field,
                       uint64_t word, uint64_t lsb, uint64_t width) {
  check(field.word == word && field.shift == lsb && field.width == width,
        what + ": layout {" + std::to_string(field.word) + ", " +
            std::to_string(field.shift) + ", " + std::to_string(field.width) +
            "}, aie-codegen {" + std::to_string(word) + ", " +
            std::to_string(lsb) + ", " + std::to_string(width) + "}");
}

// Field NAME of word W of the BD whose words are PREFIX##0, PREFIX##1, ...
#define FIELD(layout, member, PREFIX, W, NAME)                                 \
  checkField(#PREFIX #W "_" #NAME, (layout).member,                            \
             (PREFIX##W - PREFIX##0) / 4, PREFIX##W##_##NAME##_LSB,            \
             PREFIX##W##_##NAME##_WIDTH)

static void checkAbsent(const std::string &what, const DmaBdField &field) {
  check(!field.exists(), what + " should be absent");
}

// AIE4 BDs have no d0 step (always one granule) and no valid bit.
static void checkCommonAbsent(const std::string &name, const DmaBdLayout &l) {
  check(l.stepBias == 0, name + " steps are unbiased");
  checkAbsent(name + " d0Stride", l.d0Stride);
  checkAbsent(name + " validBd", l.validBd);
}

static void checkPadsAbsent(const std::string &name, const DmaBdLayout &l) {
  checkAbsent(name + " d0ZeroBefore", l.d0ZeroBefore);
  checkAbsent(name + " d1ZeroBefore", l.d1ZeroBefore);
  checkAbsent(name + " d2ZeroBefore", l.d2ZeroBefore);
  checkAbsent(name + " d0ZeroAfter", l.d0ZeroAfter);
  checkAbsent(name + " d1ZeroAfter", l.d1ZeroAfter);
  checkAbsent(name + " d2ZeroAfter", l.d2ZeroAfter);
}

static void testCoreTile(const AIETargetModel &tm) {
  const DmaBdLayout *l = tm.getDmaBdLayout(AIETileType::CoreTile);
  check(l, "core tile layout");
  check(l->numWords == 6, "core tile BD words");
  checkCommonAbsent("core", *l);
  checkPadsAbsent("core", *l);
#define CORE(member, W, NAME)                                                  \
  FIELD(*l, member, XAIE4GBL_MEMORY_MODULE_DMA_BD0_, W, NAME)
  CORE(bufferOffset, 0, BASE_ADDRESS);
  CORE(bufferLength, 0, BUFFER_LENGTH);
  CORE(enablePacket, 1, ENABLE_PACKET);
  CORE(outOfOrderId, 1, OUT_OF_ORDER_BD_ID);
  CORE(packetId, 1, PACKET_ID);
  CORE(packetType, 1, PACKET_TYPE);
  CORE(d1Stride, 2, D1_STEPSIZE);
  CORE(iterationCurrent, 3, ITERATION_CURRENT);
  CORE(d0Size, 3, D0_WRAP);
  CORE(d2Stride, 3, D2_STEPSIZE);
  CORE(d1Size, 4, D1_WRAP);
  CORE(iterationSize, 4, ITERATION_WRAP);
  CORE(iterationStride, 4, ITERATION_STEPSIZE);
  CORE(nextBd, 5, NEXT_BD);
  CORE(useNextBd, 5, USE_NEXT_BD);
  CORE(lockRelValue, 5, LOCK_REL_VALUE);
  CORE(lockRelId, 5, LOCK_REL_ID);
  CORE(lockAcqEnable, 5, LOCK_ACQ_ENABLE);
  CORE(lockAcqValue, 5, LOCK_ACQ_VALUE);
  CORE(lockAcqId, 5, LOCK_ACQ_ID);
#undef CORE
  checkAbsent("core d2Size", l->d2Size);
  checkAbsent("core d3Stride", l->d3Stride);
  checkAbsent("core burstLength", l->burstLength);
  checkAbsent("core axcache", l->axcache);
  check(XAIE4GBL_MEMORY_MODULE_DMA_BD1_0 - XAIE4GBL_MEMORY_MODULE_DMA_BD0_0 ==
            0x20,
        "core BD stride");
}

// Memtile S2MM and MM2S BDs share one layout.
#define MEM(prefix, member, W, NAME) FIELD(*l, member, prefix, W, NAME)
#define MEMTILE_FIELDS(prefix)                                                 \
  do {                                                                         \
    MEM(prefix, bufferOffset, 0, BASE_ADDRESS);                                \
    MEM(prefix, lockAcqValue, 1, LOCK_ACQ_VALUE);                              \
    MEM(prefix, bufferLength, 1, BUFFER_LENGTH);                               \
    MEM(prefix, lockAcqId, 2, LOCK_ACQ_ID);                                    \
    MEM(prefix, lockRelValue, 2, LOCK_REL_VALUE);                              \
    MEM(prefix, lockRelId, 2, LOCK_REL_ID);                                    \
    MEM(prefix, lockAcqEnable, 2, LOCK_ACQ_ENABLE);                            \
    MEM(prefix, d1Stride, 3, D1_STEPSIZE);                                     \
    MEM(prefix, d2Stride, 4, D2_STEPSIZE);                                     \
    MEM(prefix, d3Stride, 5, D3_STEPSIZE);                                     \
    MEM(prefix, iterationStride, 6, ITERATION_STEPSIZE);                       \
    MEM(prefix, iterationCurrent, 7, ITERATION_CURRENT);                       \
    MEM(prefix, enablePacket, 7, ENABLE_PACKET);                               \
    MEM(prefix, packetId, 7, PACKET_ID);                                       \
    MEM(prefix, d0ZeroAfter, 7, D0_PAD_AFTER);                                 \
    MEM(prefix, d0ZeroBefore, 7, D0_PAD_BEFORE);                               \
    MEM(prefix, d2ZeroAfter, 8, D2_PAD_AFTER);                                 \
    MEM(prefix, d2ZeroBefore, 8, D2_PAD_BEFORE);                               \
    MEM(prefix, d1ZeroAfter, 8, D1_PAD_AFTER);                                 \
    MEM(prefix, d1ZeroBefore, 8, D1_PAD_BEFORE);                               \
    MEM(prefix, d1Size, 9, D1_WRAP);                                           \
    MEM(prefix, d0Size, 9, D0_WRAP);                                           \
    MEM(prefix, outOfOrderId, 9, OUT_OF_ORDER_BD_ID);                          \
    MEM(prefix, nextBd, 10, NEXT_BD);                                          \
    MEM(prefix, useNextBd, 10, USE_NEXT_BD);                                   \
    MEM(prefix, d2Size, 10, D2_WRAP);                                          \
    MEM(prefix, iterationSize, 10, ITERATION_WRAP);                            \
  } while (0)

static void testMemTile(const AIETargetModel &tm) {
  const DmaBdLayout *l = tm.getDmaBdLayout(AIETileType::MemTile);
  check(l, "memtile layout");
  check(l->numWords == 11, "memtile BD words");
  checkCommonAbsent("memtile", *l);
  MEMTILE_FIELDS(XAIE4GBL_MEM_TILE_MODULE_DMA_S2MM_0_BD0_);
  MEMTILE_FIELDS(XAIE4GBL_MEM_TILE_MODULE_DMA_MM2S_0_BD0_);
  // The packet type is a channel setting (DMA_*_CTRL.Packet_Type).
  checkAbsent("memtile packetType", l->packetType);
  checkAbsent("memtile burstLength", l->burstLength);
  checkAbsent("memtile axcache", l->axcache);
  check(XAIE4GBL_MEM_TILE_MODULE_DMA_S2MM_0_BD1_0 -
                XAIE4GBL_MEM_TILE_MODULE_DMA_S2MM_0_BD0_0 ==
            0x30,
        "memtile BD stride");
}
#undef MEMTILE_FIELDS
#undef MEM

// Word 1 holds byte-address bits [31:2]; the address patch sets the high bits.
static void checkShimBufferOffset(const std::string &name,
                                  const DmaBdField &field, uint64_t word,
                                  uint64_t lsb, uint64_t width) {
  check(field.word == word && field.shift == 0 && field.width == 32 &&
            lsb + width == 32,
        name + " buffer offset");
}

static void testShimTile(const AIETargetModel &tm) {
  const DmaBdLayout *l = tm.getDmaBdLayout(AIETileType::ShimNOCTile);
  check(l, "shim layout");
  check(l->numWords == 9, "shim BD words");
  checkCommonAbsent("shim", *l);
  checkPadsAbsent("shim", *l);
  checkShimBufferOffset(
      "shim", l->bufferOffset,
      (XAIE4GBL_NOC_MODULE_DMA_BD0_1 - XAIE4GBL_NOC_MODULE_DMA_BD0_0) / 4,
      XAIE4GBL_NOC_MODULE_DMA_BD0_1_BASE_ADDRESS_LOW_LSB,
      XAIE4GBL_NOC_MODULE_DMA_BD0_1_BASE_ADDRESS_LOW_WIDTH);
#define SHIM(member, W, NAME)                                                  \
  FIELD(*l, member, XAIE4GBL_NOC_MODULE_DMA_BD0_, W, NAME)
  SHIM(bufferLength, 2, BUFFER_LENGTH);
  SHIM(lockAcqId, 3, LOCK_ACQ_ID);
  SHIM(lockAcqEnable, 3, LOCK_ACQ_ENABLE);
  SHIM(burstLength, 3, BURST_LENGTH);
  SHIM(d1Size, 3, D1_WRAP);
  SHIM(d2Size, 3, D2_WRAP);
  SHIM(d0Size, 4, D0_WRAP);
  SHIM(lockRelValue, 4, LOCK_REL_VALUE);
  SHIM(lockRelId, 4, LOCK_REL_ID);
  SHIM(lockAcqValue, 4, LOCK_ACQ_VALUE);
  SHIM(enablePacket, 5, ENABLE_PACKET);
  SHIM(nextBd, 5, NEXT_BD);
  SHIM(useNextBd, 5, USE_NEXT_BD);
  SHIM(d1Stride, 5, D1_STEPSIZE);
  SHIM(iterationCurrent, 6, ITERATION_CURRENT);
  SHIM(d2Stride, 6, D2_STEPSIZE);
  SHIM(axcache, 7, AXCACHE);
  SHIM(iterationSize, 7, ITERATION_WRAP);
  SHIM(iterationStride, 7, ITERATION_STEPSIZE);
  SHIM(d3Stride, 8, D3_STEPSIZE);
  SHIM(outOfOrderId, 8, OUT_OF_ORDER_BD_ID);
  SHIM(packetId, 8, PACKET_ID);
#undef SHIM
  checkAbsent("shim packetType", l->packetType);
  check(XAIE4GBL_NOC_MODULE_DMA_BD1_0 - XAIE4GBL_NOC_MODULE_DMA_BD0_0 == 0x30,
        "shim BD stride");
}

// ControlMM2S and TraceS2MM BDs: no addressing dimensions and no locks.
static void checkLinearAbsent(const std::string &name, const DmaBdLayout &l) {
  checkCommonAbsent(name, l);
  checkPadsAbsent(name, l);
  for (auto [field, what] : {std::pair{&l.d0Size, "d0Size"},
                             {&l.d1Size, "d1Size"},
                             {&l.d2Size, "d2Size"},
                             {&l.d1Stride, "d1Stride"},
                             {&l.d2Stride, "d2Stride"},
                             {&l.d3Stride, "d3Stride"},
                             {&l.lockAcqId, "lockAcqId"},
                             {&l.lockAcqEnable, "lockAcqEnable"},
                             {&l.lockAcqValue, "lockAcqValue"},
                             {&l.lockRelId, "lockRelId"},
                             {&l.lockRelValue, "lockRelValue"},
                             {&l.packetType, "packetType"}})
    checkAbsent(name + " " + what, *field);
}

#define LINEAR_FIELDS(prefix)                                                  \
  do {                                                                         \
    FIELD(*l, bufferLength, prefix, 2, BUFFER_LENGTH);                         \
    FIELD(*l, burstLength, prefix, 3, BURST_LENGTH);                           \
    FIELD(*l, nextBd, prefix, 5, NEXT_BD);                                     \
    FIELD(*l, useNextBd, prefix, 5, USE_NEXT_BD);                              \
    FIELD(*l, iterationCurrent, prefix, 6, ITERATION_CURRENT);                 \
    FIELD(*l, axcache, prefix, 7, AXCACHE);                                    \
    FIELD(*l, iterationSize, prefix, 7, ITERATION_WRAP);                       \
    FIELD(*l, iterationStride, prefix, 7, ITERATION_STEPSIZE);                 \
  } while (0)

static void testShimControlAndTrace(const AIETargetModel &tm) {
  check(tm.getDmaBdLayout(AIETileType::ShimNOCTile, DMAChannelDir::MM2S) ==
                tm.getDmaBdLayout(AIETileType::ShimNOCTile) &&
            tm.getDmaBdLayout(AIETileType::ShimNOCTile, DMAChannelDir::S2MM) ==
                tm.getDmaBdLayout(AIETileType::ShimNOCTile),
        "data channels use the tile's layout");
  check(!tm.getDmaBdLayout(AIETileType::CoreTile, DMAChannelDir::ControlMM2S) &&
            !tm.getDmaBdLayout(AIETileType::MemTile, DMAChannelDir::TraceS2MM),
        "only shims have control and trace BDs");

  const DmaBdLayout *l =
      tm.getDmaBdLayout(AIETileType::ShimNOCTile, DMAChannelDir::ControlMM2S);
  check(l, "control layout");
  check(l->numWords == 9, "control BD words");
  checkLinearAbsent("control", *l);
  check(tm.getDmaBdAddress(0, 0, 0, 0, DMAChannelDir::ControlMM2S) ==
                XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_0 &&
            tm.getDmaBdAddress(0, 0, 0, 1, DMAChannelDir::ControlMM2S) ==
                XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_1_BD0_0 &&
            tm.getDmaBdAddress(0, 0, 0, 0, DMAChannelDir::TraceS2MM) ==
                XAIE4GBL_NOC_MODULE_DMA_TRACE_S2MM_BD0_0,
        "control and trace BD bases");
  checkShimBufferOffset(
      "control", l->bufferOffset,
      (XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_1 -
       XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_0) /
          4,
      XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_1_BASE_ADDRESS_LOW_LSB,
      XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_1_BASE_ADDRESS_LOW_WIDTH);
  LINEAR_FIELDS(XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_);
  LINEAR_FIELDS(XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_1_BD0_);
  FIELD(*l, enablePacket, XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_, 5,
        ENABLE_PACKET);
  FIELD(*l, outOfOrderId, XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_, 8,
        OUT_OF_ORDER_BD_ID);
  FIELD(*l, packetId, XAIE4GBL_NOC_MODULE_DMA_CONTROL_MM2S_0_BD0_, 8,
        PACKET_ID);

  l = tm.getDmaBdLayout(AIETileType::ShimNOCTile, DMAChannelDir::TraceS2MM);
  check(l, "trace layout");
  check(l->numWords == 9, "trace BD words");
  checkLinearAbsent("trace", *l);
  checkShimBufferOffset(
      "trace", l->bufferOffset,
      (XAIE4GBL_NOC_MODULE_DMA_TRACE_S2MM_BD0_1 -
       XAIE4GBL_NOC_MODULE_DMA_TRACE_S2MM_BD0_0) /
          4,
      XAIE4GBL_NOC_MODULE_DMA_TRACE_S2MM_BD0_1_BASE_ADDRESS_LOW_LSB,
      XAIE4GBL_NOC_MODULE_DMA_TRACE_S2MM_BD0_1_BASE_ADDRESS_LOW_WIDTH);
  LINEAR_FIELDS(XAIE4GBL_NOC_MODULE_DMA_TRACE_S2MM_BD0_);
  checkAbsent("trace enablePacket", l->enablePacket);
  checkAbsent("trace outOfOrderId", l->outOfOrderId);
  checkAbsent("trace packetId", l->packetId);
}
#undef LINEAR_FIELDS

static void testLimits(const AIETargetModel &tm) {
  struct {
    AIETileType type;
    uint64_t maxLen;
    uint32_t wrapBits, stepBits, iterBits;
  } limits[] = {{AIETileType::CoreTile, (1u << 15) - 1, 9, 15, 6},
                {AIETileType::MemTile, (1u << 23) - 1, 12, 23, 8},
                {AIETileType::ShimNOCTile, 0xFFFFFFFFu, 12, 22, 6}};
  for (auto [type, maxLen, wrapBits, stepBits, iterBits] : limits) {
    std::string name = stringifyAIETileType(type).str();
    check(tm.getDmaBdMaxLen(type) == maxLen, name + " max length");
    check(tm.getDmaBdWrapBits(type) == wrapBits, name + " wrap bits");
    check(tm.getDmaBdStepBits(type) == stepBits, name + " step bits");
    check(tm.getDmaBdIterBits(type) == iterBits, name + " iteration bits");
    check(tm.getDmaBdStepBias(type) == 0, name + " step bias");
  }
  // Every pre-AIE4 layout keeps the -1 step bias.
  for (AIEDevice device : {AIEDevice::npu1, AIEDevice::npu2,
                           AIEDevice::xcve2802, AIEDevice::xcvc1902})
    for (AIETileType type : {AIETileType::ShimNOCTile, AIETileType::MemTile,
                             AIETileType::CoreTile})
      check(getTargetModel(device).getDmaBdStepBias(type) == 1,
            stringifyAIEDevice(device).str() + " step bias");
}

// A shim data BD id names its bank, so its address needs no channel.
static void testShimBdAddress(const AIETargetModel &tm) {
  for (uint32_t bd = 0; bd < 32; ++bd) {
    uint64_t expected = (bd < 16 ? XAIE4GBL_NOC_MODULE_DMA_BD0_0
                                 : XAIE4GBL_NOC_MODULE_DMA_BD0_0 + 0x40000) +
                        (bd % 16) * (XAIE4GBL_NOC_MODULE_DMA_BD1_0 -
                                     XAIE4GBL_NOC_MODULE_DMA_BD0_0);
    check(tm.getDmaBdAddress(0, 0, bd) == expected,
          "shim BD " + std::to_string(bd) + " address without a channel");
    for (DMAChannelDir dir : {DMAChannelDir::MM2S, DMAChannelDir::S2MM})
      for (int ch = 0; ch < static_cast<int>(tm.getNumDMAChannels(0, 0, dir));
           ++ch)
        if (tm.isBdChannelAccessible(0, 0, bd, ch, dir))
          check(tm.getDmaBdAddress(0, 0, bd, ch, dir) ==
                    tm.getDmaBdAddress(0, 0, bd),
                "shim BD " + std::to_string(bd) + " channel " +
                    std::to_string(ch) + " address");
  }
}

int main() {
  for (AIEDevice device :
       {AIEDevice::npu3, AIEDevice::npu3_1col, AIEDevice::npu3_2col}) {
    const AIETargetModel &tm = getTargetModel(device);
    testCoreTile(tm);
    testMemTile(tm);
    testShimTile(tm);
    testShimControlAndTrace(tm);
    testLimits(tm);
    testShimBdAddress(tm);
  }
  return 0;
}
