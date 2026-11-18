//===- bd_register_base.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// getBdRegisterBase() vs getDmaBdAddress(): private-pool BD ids (AIE4 memtile,
// ControlMM2S, TraceS2MM) are per channel; shim data BD ids name their bank.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <iostream>
#include <optional>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

int failures = 0;

void expect(bool ok, const std::string &what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

// Folds the address getBdRegisterBase() built, with `bdId` bound to `id`.
std::optional<int64_t> evaluate(Value v, Value bdId, int64_t id) {
  if (v == bdId)
    return id;
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    return cast<IntegerAttr>(c.getValue()).getInt();
  auto both = [&](Operation *op) -> std::optional<std::pair<int64_t, int64_t>> {
    auto l = evaluate(op->getOperand(0), bdId, id);
    auto r = evaluate(op->getOperand(1), bdId, id);
    if (!l || !r)
      return std::nullopt;
    return std::make_pair(*l, *r);
  };
  if (auto add = v.getDefiningOp<arith::AddIOp>())
    if (auto lr = both(add))
      return lr->first + lr->second;
  if (auto mul = v.getDefiningOp<arith::MulIOp>())
    if (auto lr = both(mul))
      return lr->first * lr->second;
  return std::nullopt;
}

struct Fixture {
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  OpBuilder builder;
  Location loc;
  Value runtimeId;

  Fixture() : builder(&context), loc(UnknownLoc::get(&context)) {
    context.getOrLoadDialect<arith::ArithDialect>();
    module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module->getBody());
    // Not a constant, so getBdRegisterBase() takes the runtime path.
    Value one =
        arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1));
    runtimeId = arith::AddIOp::create(builder, loc, one, one);
  }

  std::optional<int64_t> constant(const AIETargetModel &tm, int col, int row,
                                  int id, int channel, DMAChannelDir dir) {
    Value v =
        AIEX::getBdRegisterBase(builder, loc, tm, col, row,
                                builder.getI32IntegerAttr(id), channel, dir);
    if (auto c = v.getDefiningOp<arith::ConstantOp>())
      return static_cast<uint32_t>(cast<IntegerAttr>(c.getValue()).getInt());
    return std::nullopt;
  }

  std::optional<int64_t> runtime(const AIETargetModel &tm, int col, int row,
                                 int id, int channel, DMAChannelDir dir) {
    Value v = AIEX::getBdRegisterBase(builder, loc, tm, col, row,
                                      OpFoldResult(runtimeId), channel, dir);
    auto r = evaluate(v, runtimeId, id);
    if (!r)
      return std::nullopt;
    return static_cast<uint32_t>(*r);
  }
};

void testNpu3(Fixture &f) {
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu3);
  const int col = 0, shim = 0, mem = 1;
  using D = DMAChannelDir;

  for (auto [dir, channel, id] :
       {std::tuple{D::ControlMM2S, 0, 2}, std::tuple{D::ControlMM2S, 1, 2},
        std::tuple{D::TraceS2MM, 0, 3}}) {
    uint64_t want = tm.getDmaBdAddress(col, shim, id, channel, dir);
    auto got = f.constant(tm, col, shim, id, channel, dir);
    expect(got && uint64_t(*got) == want,
           std::string("npu3 shim ") + stringifyDMAChannelDir(dir).str() +
               " constant BD address");
    expect(want != tm.getDmaBdAddress(col, shim, id),
           "control/trace BD differs from the same-id data BD");
    for (int k : {0, 1}) {
      int rid = tm.getBdIdRangeForChannel(col, shim, channel, dir).first + k;
      auto r = f.runtime(tm, col, shim, rid, channel, dir);
      expect(r && uint64_t(*r) ==
                      tm.getDmaBdAddress(col, shim, rid, channel, dir),
             std::string("npu3 shim ") + stringifyDMAChannelDir(dir).str() +
                 " runtime BD address");
    }
  }
  // ControlMM2S channel 1 is in the upper bank.
  expect(f.constant(tm, col, shim, 0, 1, D::ControlMM2S) ==
             std::optional<int64_t>(0x58300),
         "ControlMM2S channel 1 BD 0 at 0x58300");
  expect(f.constant(tm, col, shim, 3, 0, D::TraceS2MM) ==
             std::optional<int64_t>(0x18480 + 3 * 0x30),
         "TraceS2MM BD 3 at 0x18510");

  // A runtime id on an upper-bank channel starts at 16.
  for (auto [dir, channel] : {std::pair{D::MM2S, 2}, std::pair{D::S2MM, 1}}) {
    for (int id : {16, 31}) {
      uint64_t want = tm.getDmaBdAddress(col, shim, id);
      expect(f.constant(tm, col, shim, id, channel, dir) ==
                 std::optional<int64_t>(want),
             "npu3 shim upper-bank data BD, with its channel");
      expect(f.constant(tm, col, shim, id, -1, dir) ==
                 std::optional<int64_t>(want),
             "npu3 shim upper-bank data BD, without a channel");
      expect(f.runtime(tm, col, shim, id, channel, dir) ==
                 std::optional<int64_t>(want),
             "npu3 shim upper-bank data BD, runtime id");
    }
  }

  // Memtile BD ids are channel-local.
  for (auto [dir, channel] : {std::pair{D::MM2S, 6}, std::pair{D::S2MM, 1}}) {
    uint64_t want = tm.getDmaBdAddress(col, mem, 3, channel, dir);
    expect(f.constant(tm, col, mem, 3, channel, dir) ==
               std::optional<int64_t>(want),
           "npu3 memtile constant BD address");
    expect(f.runtime(tm, col, mem, 3, channel, dir) ==
               std::optional<int64_t>(want),
           "npu3 memtile runtime BD address");
  }
}

void testNpu2(Fixture &f) {
  // Earlier targets: a BD id names a BD of the tile, whatever the channel.
  const AIETargetModel &tm = getTargetModel(AIEDevice::npu2);
  for (auto [row, channel, dir, id] :
       {std::tuple{0, 1, DMAChannelDir::MM2S, 5},
        std::tuple{0, 0, DMAChannelDir::S2MM, 15},
        std::tuple{1, 3, DMAChannelDir::S2MM, 30},
        std::tuple{2, 1, DMAChannelDir::MM2S, 7}}) {
    uint64_t want = tm.getDmaBdAddress(0, row, id);
    expect(f.constant(tm, 0, row, id, channel, dir) ==
               std::optional<int64_t>(want),
           "npu2 constant BD address");
    expect(f.constant(tm, 0, row, id, -1, dir) == std::optional<int64_t>(want),
           "npu2 constant BD address without a channel");
  }
}

} // namespace

int main() {
  Fixture f;
  testNpu3(f);
  testNpu2(f);
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "bd_register_base: all checks passed\n";
  return 0;
}
