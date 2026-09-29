//===- AIENarrowCoreLoops.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include <cstdint>
#include <limits>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIENARROWCORELOOPS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx::AIE;

// The final induction-variable increment must fit too. Otherwise a loop
// ending near INT32_MAX could wrap and run forever after narrowing.
static bool canNarrow(scf::ForOp loop) {
  if (!loop.getInductionVar().getType().isIndex() || loop.getUnsignedCmp())
    return false;
  auto lb = getConstantIntValue(loop.getLowerBound());
  auto ub = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step || *lb < 0 || *ub <= *lb || *step <= 0 ||
      *ub > std::numeric_limits<int32_t>::max() ||
      *step > std::numeric_limits<int32_t>::max())
    return false;
  int64_t trip = (*ub - *lb + *step - 1) / *step;
  return *lb + trip * *step <= std::numeric_limits<int32_t>::max();
}

static void narrow(scf::ForOp loop) {
  OpBuilder b(loop);
  Location loc = loop.getLoc();
  Type i32 = b.getI32Type();
  auto c32 = [&](Value v) -> Value {
    return arith::ConstantOp::create(
        b, loc, b.getI32IntegerAttr(*getConstantIntValue(v)));
  };
  Value lb = c32(loop.getLowerBound());
  Value ub = c32(loop.getUpperBound());
  Value step = c32(loop.getStep());
  auto narrowed = scf::ForOp::create(b, loc, lb, ub, step, loop.getInitArgs(),
                                     /*bodyBuilder=*/nullptr);
  narrowed.getRegion().takeBody(loop.getRegion());
  Value iv = narrowed.getInductionVar();
  iv.setType(i32);
  if (!iv.use_empty()) {
    b.setInsertionPointToStart(narrowed.getBody());
    Value index =
        arith::IndexCastOp::create(b, loc, b.getIndexType(), iv).getResult();
    iv.replaceAllUsesExcept(index, index.getDefiningOp());
  }
  loop.replaceAllUsesWith(narrowed.getResults());
  loop.erase();
}

namespace {
struct AIENarrowCoreLoopsPass
    : xilinx::AIE::impl::AIENarrowCoreLoopsBase<AIENarrowCoreLoopsPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    if (device.getTargetModel().getTargetArch() != AIEArch::AIE4)
      return;
    SmallVector<scf::ForOp> loops;
    device.walk([&](scf::ForOp loop) {
      if (loop->getParentOfType<CoreOp>() ||
          loop->getParentOfType<func::FuncOp>())
        if (canNarrow(loop))
          loops.push_back(loop);
    });
    for (scf::ForOp loop : loops)
      narrow(loop);
  }
};
} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIENarrowCoreLoopsPass() {
  return std::make_unique<AIENarrowCoreLoopsPass>();
}
