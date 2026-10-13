// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_DIALECT_XLLVM_IR_XLLVMTILEMEMORYINTROPS_TD
#define AIE_DIALECT_XLLVM_IR_XLLVMTILEMEMORYINTROPS_TD

include "aie/Dialect/XLLVM/IR/XLLVM.td"

// Thin wrappers around Peano's intrinsics. Keep unknown effects on all three
// operations: reads may observe changing device state, writes must not be
// eliminated, and the scheduling barrier must also order MLIR transformations.
// Addresses are already core-visible pointers, exactly as in the LLVM ABI.
class ReadTmIntrOp<string arch>
    : ExtIntrOpBase<"intr."#arch#".read.tm", arch#".read.tm", [], 1>,
      Arguments<(ins LLVM_DefaultPointer:$address)> {
  let results = (outs I32:$res);
  let summary =
      "Read a 32-bit tile-mapped register through the core processor bus";
  let description = [{
    Calls the target's `llvm.<arch>.read.tm` intrinsic. The operand is a
    core-visible pointer in address space zero, not an array address or tile
    register offset. The caller must translate the register address using the
    target's TM aperture and enable the processor bus before executing the core.
    See `docs/TileMappedAccess.md`.

    The access has unknown effects and is not speculatable. Use the target's
    `sched.barrier` intrinsic where scheduling separation is required.
  }];
  let assemblyFormat = "$address attr-dict `:` type($address)";
}

class WriteTmIntrOp<string arch>
    : ExtIntrOpBase<"intr."#arch#".write.tm", arch#".write.tm", [], 0>,
      Arguments<(ins I32:$value, LLVM_DefaultPointer:$address)> {
  let summary =
      "Write a 32-bit tile-mapped register through the core processor bus";
  let description = [{
    Calls the target's `llvm.<arch>.write.tm(value, address)` intrinsic.
    Address interpretation and processor-bus prerequisites are the same as
    `read.tm`. This is a full 32-bit register write, not a masked write.

    Unknown effects preserve execution and ordering in MLIR. Backend scheduling
    separation is explicit: place `sched.barrier` around accesses as required
    by the target and affected resource. Neither the write nor that barrier
    drains traffic in neighboring cores.
  }];
  // The base builder assumes a result; write.tm returns void.
  let llvmBuilder = [{
    ::xilinx::xllvm::createExternalLLVMIntrinsicCall(
        builder, moduleTranslation, &opInst, "llvm.}]#arch#[{.write.tm");
  }];
  let assemblyFormat = "$value `,` $address attr-dict `:` type($address)";
}

class SchedBarrierIntrOp<string arch>
    : ExtIntrOpBase<"intr."#arch#".sched.barrier", arch#".sched.barrier", [],
                    0>,
      Arguments<(ins)> {
  let summary = "Expose the target's compiler scheduling barrier";
  let description = [{
    Calls `llvm.<arch>.sched.barrier()`. This preserves a scheduling boundary
    through Peano code generation. It is a compiler barrier, not a device
    synchronization operation, a processor-bus enable, or a traffic drain.
  }];
  let llvmBuilder = [{
    ::xilinx::xllvm::createExternalLLVMIntrinsicCall(
        builder, moduleTranslation, &opInst, "llvm.}]#arch#[{.sched.barrier");
  }];
  let assemblyFormat = "attr-dict";
}

def ReadTmAIE2Op : ReadTmIntrOp<"aie2">;
def WriteTmAIE2Op : WriteTmIntrOp<"aie2">;
def SchedBarrierAIE2Op : SchedBarrierIntrOp<"aie2">;
def ReadTmAIE2pOp : ReadTmIntrOp<"aie2p">;
def WriteTmAIE2pOp : WriteTmIntrOp<"aie2p">;
def SchedBarrierAIE2pOp : SchedBarrierIntrOp<"aie2p">;
def ReadTmAIE4Op : ReadTmIntrOp<"aie4">;
def WriteTmAIE4Op : WriteTmIntrOp<"aie4">;
def SchedBarrierAIE4Op : SchedBarrierIntrOp<"aie4">;

#endif // AIE_DIALECT_XLLVM_IR_XLLVMTILEMEMORYINTROPS_TD
