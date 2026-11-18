//===- taken_bds_pools.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TakenBDs charges a pinned BD id once per pool. Each npu3 mem tile channel has
// a private pool (ids 0-15); npu2 pins go to the lowest channel of the pool.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include <cstdio>
#include <map>
#include <string>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

int failures = 0;

using Taken = std::map<DMAChannelAnalysis::BDUser, int64_t>;

std::string deviceWithPins(StringRef device,
                           ArrayRef<std::pair<int, int>> pins) {
  std::string tasks;
  for (auto [channel, id] : pins)
    tasks += "      %t" + std::to_string(channel) + "_" + std::to_string(id) +
             " = aiex.dma_configure_task(%m, S2MM, " + std::to_string(channel) +
             ") {\n        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = "
             "16) {bd_id = " +
             std::to_string(id) + " : i32}\n        aie.end\n      }\n";
  return "module {\n  aie.device(" + device.str() +
         ") {\n    %m = aie.tile(0, 1)\n    %buf = aie.buffer(%m) {address = "
         "0 : i32} : memref<16xi32>\n    aie.runtime_sequence() {\n" +
         tasks + "    }\n  }\n}\n";
}

void check(const char *what, StringRef device,
           ArrayRef<std::pair<int, int>> pins, const Taken &expected) {
  MLIRContext context;
  context.loadDialect<AIEDialect, AIEX::AIEXDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(deviceWithPins(device, pins), &context);
  if (!module) {
    std::fprintf(stderr, "FAIL: %s: does not parse\n", what);
    ++failures;
    return;
  }
  DeviceOp op = *module->getOps<DeviceOp>().begin();
  Taken taken = DMAChannelAnalysis::TakenBDs(op).at(op.getTargetModel(), 0, 1);
  if (taken == expected)
    return;
  std::fprintf(stderr, "FAIL: %s:", what);
  for (auto [user, bds] : taken)
    std::fprintf(stderr, " %s%d=%lld",
                 user.first == DMAChannelDir::S2MM ? "S2MM" : "MM2S",
                 user.second, static_cast<long long>(bds));
  std::fprintf(stderr, "\n");
  ++failures;
}

} // namespace

int main() {
  constexpr DMAChannelDir s2mm = DMAChannelDir::S2MM;
  check("npu3 pin on channel 1", "npu3", {{1, 0}}, {{{s2mm, 1}, 1}});
  check("npu3 same id on two channels", "npu3", {{0, 0}, {1, 0}},
        {{{s2mm, 0}, 1}, {{s2mm, 1}, 1}});
  // npu2: even channels share ids 0-23, odd channels 24-47.
  check("npu2 lowest channel of the pool", "npu2", {{2, 0}, {3, 30}},
        {{{s2mm, 0}, 1}, {{s2mm, 1}, 1}});
  check("npu2 same id on two channels of a pool", "npu2", {{0, 5}, {2, 5}},
        {{{s2mm, 0}, 1}});
  if (failures)
    return 1;
  std::printf("PASS\n");
  return 0;
}
