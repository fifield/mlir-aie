//===- AIETargetXAIEV2.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2021-2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#include "aie/Targets/AIETargetShared.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace xilinx::AIE {

// This string is output at the top of the lowered C++ code.
const char *hsa_cpp_file_header = R"code(
// This file was auto-generated by aiecc.py --aie-generate-hsa

#ifndef MLIR_AIE_QUIET
#define __mlir_aie_verbose(x) x
#else
#define __mlir_aie_verbose(x)
#endif

)code";

std::optional<AIE::ShimDMAAllocationOp>
getAllocOpForSymbol(AIE::DeviceOp dev, StringRef sym_name) {
  auto sym = dev.lookupSymbol(sym_name);
  if (!sym)
    return std::nullopt;

  auto uses = SymbolTable::getSymbolUses(sym, dev);
  for (auto use : *uses)
    if (auto infoOp = dyn_cast<AIE::ShimDMAAllocationOp>(use.getUser()))
      return infoOp;

  return std::nullopt;
}

mlir::LogicalResult AIETranslateToHSA(ModuleOp module, raw_ostream &output) {

  DenseMap<TileID, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  if (module.getOps<DeviceOp>().empty())
    return module.emitOpError("expected AIE.device operation at toplevel");
  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  // Putting the standard header
  output << hsa_cpp_file_header;

  // Getting the sequence function op which contains the instructions
  auto sequenceOps = targetOp.getOps<AIEX::RuntimeSequenceOp>();
  if (sequenceOps.empty()) {
    // If no sequenceOp then just return
    return success();
  } else if (std::distance(sequenceOps.begin(), sequenceOps.end()) > 1) {
    return module.emitOpError("expected at most one sequence operation");
  }
  AIEX::RuntimeSequenceOp sequenceOp = *sequenceOps.begin();

  collectTiles(targetOp, tiles);
  collectBuffers(targetOp, buffers);

  // Generate dynamic data movement
  output << "void invoke_data_movement(hsa_queue_t *q, hsa_agent_t *a";

  // Looping over every Memcpy operation so we take the correct number of
  // buffers
  int num_ops = 0;
  for (auto op : sequenceOp.getOps<NpuDmaMemcpyNdOp>()) {
    // Getting the IDs of the buffers
    auto memref = op.getMemref();
    Block &entryBB =
        op->getParentOfType<AIEX::RuntimeSequenceOp>().getBody().front();
    int arg_idx = -1;
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
      if (entryBB.getArgument(i) == memref) {
        arg_idx = i;
        break;
      }
    }
    num_ops++;

    output << ", void *buf" << arg_idx;
  }

  output << ") {\n";

  output << "\tuint64_t wr_idx = 0;\n";
  output << "\tuint64_t packet_id = 0;\n";

  int op_count = 0;
  for (auto op : sequenceOp.getOps<NpuDmaMemcpyNdOp>()) {
    auto dev = sequenceOp->getParentOfType<AIE::DeviceOp>();
    if (!dev) {
      op.emitOpError("couldn't get DeviceOp");
      return failure();
    }

    auto infoOp = getAllocOpForSymbol(dev, op.getMetadata());
    if (!infoOp) {
      op.emitOpError("couldn't find shim_dma_allocation op");
      return failure();
    }

    auto channelDir = infoOp->getChannelDir();
    uint32_t ChannelId = infoOp->getChannelIndex();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int col = infoOp->getCol();
    bool isPlio = infoOp->getPlio();

    llvm::SmallVector<int64_t, 4> strides = llvm::map_to_vector(
        llvm::reverse(op.getMixedStrides()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    ::SmallVector<int64_t, 4> sizes = llvm::map_to_vector(
        llvm::reverse(op.getMixedSizes()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    ::SmallVector<int64_t, 4> offsets = llvm::map_to_vector(
        llvm::reverse(op.getMixedOffsets()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });

    // buffer_offset
    size_t stride = 1;
    size_t offset = 0;
    BaseMemRefType my_memref = op.getMemref().getType();
    auto shape = my_memref.getShape();
    size_t R = shape.size();
    size_t el_bit_width = op.getElementTypeBitwidth();
    assert(el_bit_width % 8 == 0 &&
           "Expected Memref element bitwidth to be multiple of 8.");
    size_t S = el_bit_width / 8;
    for (size_t i = 0; i < R; i++) {
      offset += offsets[i] * stride * S;
      stride *= shape[R - i - 1];
    }

    // Getting the ID of the buffer that we are using
    auto memref = op.getMemref();
    Block &entryBB =
        op->getParentOfType<AIEX::RuntimeSequenceOp>().getBody().front();
    int arg_idx = -1;
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
      if (entryBB.getArgument(i) == memref) {
        arg_idx = i;
        break;
      }
    }

    if (strides[0] != 1)
      return module.emitOpError("nd_memcpy inner-dimension stride != 1 is "
                                "unsupported by HSA target");

    // Writing the packet information to perform the DMA
    output << "\thsa_agent_dispatch_packet_t pkt" << op_count << " ;\n";
    output << "\twr_idx  = hsa_queue_add_write_index_relaxed(q, 1);\n";
    output << "\tpacket_id  = wr_idx % q->size;\n";
    output << "\tmlir_aie_packet_nd_memcpy(&pkt" << op_count
           << ", 0 /* herd_id */, " << col << " /* col */, " << isMM2S
           << " /* dir */, " << ChannelId
           << "/* channel */, 4 /* Burst length */, " << (isPlio ? 1 : 2)
           << " /* Memory space */, "
              "(uint64_t)buf"
           << arg_idx << " + " << offset << " /* Address */, " << sizes[0] * 4
           << " /* 1d_length */, " << (strides[1] ? sizes[1] : 1)
           << " /* 2d_length */, " << (strides[1] ? strides[1] * 4 : 0)
           << " /* 2d_stride */, " << (strides[2] ? sizes[2] : 1)
           << " /* 3d_length */, " << (strides[2] ? strides[2] * 4 : 0)
           << " /* 3d_stride */ , 1 /* 4d_length */, 0 /* 4d_stride */);\n";

    bool last_op = op_count == (num_ops - 1);
    // Only ring the doorbell on the last packet
    if (last_op) {
      output
          << "\tmlir_aie_queue_dispatch_and_wait(a, q, packet_id, wr_idx, &pkt"
          << op_count << ", false);\n\n";
    } else {
      output << "\thsa_amd_signal_create_on_agent(1, 0, nullptr, a, 0, &pkt"
             << op_count << ".completion_signal);\n";
      output << "\tmlir_aie_write_pkt<hsa_agent_dispatch_packet_t>(q, "
                "packet_id, &pkt"
             << op_count << ");\n\n";
    }

    op_count++;
  }

  // Waiting to make sure each DMA is complete
  for (int i = 0; i < op_count; i++) {
    output << "\twhile (hsa_signal_wait_scacquire(pkt" << i
           << ".completion_signal,\n";
    output << "\tHSA_SIGNAL_CONDITION_EQ, 0, 0x80000,\n";
    output << "\tHSA_WAIT_STATE_ACTIVE) != 0);\n";
  }

  // Destroying every signal that we created
  for (int i = 0; i < op_count; i++) {
    output << "\thsa_signal_destroy(pkt" << i << ".completion_signal);\n";
  }

  output << "}\n";

  return success();
}
} // namespace xilinx::AIE
