//===- AIEPasses.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_PASSES
#define AIEX_PASSES

include "mlir/Pass/PassBase.td"

def AIEXToStandard : Pass<"aiex-standard-lowering", "mlir::ModuleOp"> {
  let summary = "Lower AIEX operations";
  let description = [{

    AIEX Npu Ops are removed.

  }];

  let constructor = "xilinx::AIEX::createAIEXToStandardPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIECreateCores : Pass<"aie-create-cores", "AIE::DeviceOp"> {
  let summary = "Create CoreOp, MemOp, and FlowOp of AIE dialect";
  let description = [{
    Lower toplevel 'call' operations with implicit data movement into
    aie.core, aie.mem, and aie.flow operations with explicit data
    movement.  The tile tile associated with the operations is inferred
    based on the aie.x and aie.y attributes of the call operation. e.g.:
    ```
    func.func @aie_task(%arg0: memref<256xi32>, %arg1: i32) -> () {
      ...
      return
    }
    %a = constant 0 : i32
    func.call @aie_task(%buf, %a) { aie.x = 1, aie.y = 1 } : (memref<256xi32>, i32) -> ()

    ```
    will result in:
    ```
    %0 = AIE.tile(1, 1)
    %1 = AIE.buffer(%0) : memref<256xi32>
    %2 = AIE.buffer(%0) : memref<1xi32>
    %c0_i32 = constant 0 : i32
    %5 = AIE.core(%0) {
      ...
    }
    ```
    In this case, each argument to aie_task results in an explicit buffer accessed by the core.

  }];

  let constructor = "xilinx::AIEX::createAIECreateCoresPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIECreateLocks : Pass<"aie-create-locks", "AIE::DeviceOp"> {
  let summary = "Generate physical lock ops from logical lock (token) ops";
  let description = [{
    Tokens represent high-level buffer synchronization through a sequence of
    pipeline stages.  This pass lowers token operations into physical aie.lock
    operations.
  }];

  let constructor = "xilinx::AIEX::createAIECreateLocksPass()";
}

def AIEHerdRouting : Pass<"aie-herd-routing", "AIE::DeviceOp"> {
  let summary =       "Lowering herds with place and route ops to AIE cores, mems, and switchboxes";
  let description = [{
    An experimental pass which elaborates herd operations (e.g. aie.herd, aie.iter, aie.select)
    into an explicit representation (e.g. aie.core, aie.mem, etc.).
  }];

  let constructor = "xilinx::AIEX::createAIEHerdRoutingPass()";
}

def AIEBroadcastPacket : Pass<"aie-lower-broadcast-packet", "AIE::DeviceOp"> {
  let summary = "Replace combination of broadcast and packet-switch by packet-flow";
  let description = [{
    Replace combination of broadcast and packet-switch by packet-flow
  }];

  let constructor = "xilinx::AIEX::createAIEBroadcastPacketPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIEMulticast : Pass<"aie-lower-multicast", "AIE::DeviceOp"> {
  let summary = "Lower AIE.multicast operation to AIE.flow operations";
  let description = [{
    This pass replaces AIE.multicast operation with the equivalent number of AIE.flow
    operations. The lowered AIE.flow operations have the same source port but different
    destinations.
  }];

  let constructor = "xilinx::AIEX::createAIELowerMulticastPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIELowerMemcpy : Pass<"aie-lower-memcpy", "AIE::DeviceOp"> {
  let summary = "Lower aie.memcpy operations to Flows and DMA programs";
  let description = [{
    aie.memcpy operations are an experimental high-level abstraction which
    move data from one buffer to another.
    This pass lowers them into appropriate aie.flow and aie.mem operations.
  }];

  let constructor = "xilinx::AIEX::createAIELowerMemcpyPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIEDmaToNpu : Pass<"aie-dma-to-npu", "AIE::DeviceOp"> {
  let summary = "";
  let description = [{

  }];

  let constructor = "xilinx::AIEX::createAIEDmaToNpuPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIEMaterializeBDChains : Pass<"aie-materialize-bd-chains", "AIE::DeviceOp"> {
  let summary = "Concretize aie.bd_chain ops at aiex.start_task use sites";
  let description = [{
    Inlines pre-defined `aie.bd_chains` at `aiex.start_task` use sites to become `aiex.dma_configure_task` and `aiex.dma_start_task` ops.
  }];

  let constructor = "xilinx::AIEX::createAIEMaterializeBDChainsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIEAssignRuntimeSequenceBDIDs : Pass<"aie-assign-runtime-sequence-bd-ids", "AIE::DeviceOp"> {
  let summary = "Assign IDs to Buffer Descriptors Configured in the Runtime Sequence";

  let constructor = "xilinx::AIEX::createAIEAssignRuntimeSequenceBDIDsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIEDMATasksToNPU : Pass<"aie-dma-tasks-to-npu", "AIE::DeviceOp"> {
  let summary = "Lower configured DMA tasks to NPU instructions";

  let constructor = "xilinx::AIEX::createAIEDMATasksToNPUPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIESubstituteShimDMAAllocations : Pass<"aie-substitute-shim-dma-allocations", "AIE::DeviceOp"> {
  let summary = "Replace symbolic references to `aie.shim_dma_allocation` ops with their `(tile, direction, channel)` triple";

  let constructor = "xilinx::AIEX::createAIESubstituteShimDMAAllocationsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIECtrlPacketToDma : Pass<"aie-ctrl-packet-to-dma", "AIE::DeviceOp"> {
  let summary = "Lowers npu.control_packet op to npu.dma_memcpy_nd op";

  let constructor = "xilinx::AIEX::createAIECtrlPacketToDmaPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIECtrlPacketInferTiles : Pass<"aie-ctrl-packet-infer-tiles", "AIE::DeviceOp"> {
  let summary = "Infer aie.tile ops from aiex.control_packet addresses";

  let constructor = "xilinx::AIEX::createAIECtrlPacketInferTilesPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIELowerSetLock : Pass<"aie-lower-set-lock", "AIE::DeviceOp"> {
  let summary = "Lowers all aiex.set_lock operations to npu.write32Op";

  let constructor = "xilinx::AIEX::createAIELowerSetLockPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

def AIETransformBfpTypes : Pass<"aie-transform-bfp-types", "AIE::DeviceOp"> {
  let summary = "Transform bfp types to standard builtin types";

  let description = [{
      Lowers bfp types to standard builtin types.
      Adapts operations as necessary to accomodate for bfp data movement.
  }];

  let constructor = "xilinx::AIEX::createAIETransformBfpTypesPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
    "xilinx::AIEX::AIEXDialect",
  ];
}

#endif
