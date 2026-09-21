//===- XLLVMAIE4IntrOps.td - XLLVM AIE4 intr. op defs. ----*- tablegen -*-====//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines external LLVM (XLLVM) intrinsic operations for AIE4 devices.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_IR_XLLVMAIE4INTROPS_TD
#define AIE_DIALECT_XLLVM_IR_XLLVMAIE4INTROPS_TD

include "aie/Dialect/XLLVM/IR/XLLVM.td"
include "aie/Dialect/XLLVM/IR/XLLVMTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// `AIEVecIntrOpBase` is defined in XLLVMAIE2IntrOps.td, which XLLVMOps.td
// includes ahead of this file. It is shared rather than duplicated so that
// the `llvm.<arch>.<mnemonic>` naming convention has exactly one definition.

// For AIE4 only.
class AIEVec4_IntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 1> :
    AIEVecIntrOpBase<"aie4", mnemonic, traits, numResults>;

//===----------------------------------------------------------------------===//
// AIE4 vector-MAC configuration word
//===----------------------------------------------------------------------===//
//
// AIE4's elementwise multiply/MAC intrinsics take a 32-bit "configuration"
// word that selects the multiplier's operating mode. The Peano backend does
// *not* validate this word: it materializes whatever value it is given into a
// register (`mova r8, #...`) and emits the same `vmul.f` instruction for every
// value. There is no verifier, no selection failure and no diagnostic on a
// malformed word -- a wrong field silently produces wrong numerical results.
//
// AIE2 models the equivalent word as an opaque `I32:$conf` operand. For AIE4
// we instead decompose it into typed, range-verified attributes, because MLIR
// is the only layer in the stack where a bad field can be caught at all.
//
// Field layout defined by the ISA, restricted to the bits this op uses:
//
//   bit    0   zero_acc   dynamically zero the accumulator before accumulating
//   bits 2:1   amode      arithmetic mode (operand data type)
//   bits 4:3   bmode      multiplier block mode (operand lane widths)
//
// The remaining bits (sign control etc.) are not meaningful for a
// floating-point elementwise multiply and are therefore not modelled; they are
// emitted as zero.

def AIE4_AMode_Int   : I32EnumAttrCase<"Int",    0, "int">;
def AIE4_AMode_Float : I32EnumAttrCase<"Float",  1, "float">;
def AIE4_AMode_Res2  : I32EnumAttrCase<"Res2",   2, "reserved2">;
def AIE4_AMode_Res3  : I32EnumAttrCase<"Res3",   3, "reserved3">;

def AIE4_AMode : I32EnumAttr<"AIE4AMode",
    "AIE4 vector multiplier arithmetic mode (conf bits 2:1)",
    [AIE4_AMode_Int, AIE4_AMode_Float, AIE4_AMode_Res2, AIE4_AMode_Res3]> {
  let cppNamespace = "::xilinx::xllvm";
}

def AIE4_BMode_8x8   : I32EnumAttrCase<"Mode8x8",   0, "m8x8">;
def AIE4_BMode_16x8  : I32EnumAttrCase<"Mode16x8",  1, "m16x8">;
def AIE4_BMode_16x16 : I32EnumAttrCase<"Mode16x16", 2, "m16x16">;
def AIE4_BMode_32x32 : I32EnumAttrCase<"Mode32x32", 3, "m32x32">;

def AIE4_BMode : I32EnumAttr<"AIE4BMode",
    "AIE4 vector multiplier block mode (conf bits 4:3)",
    [AIE4_BMode_8x8, AIE4_BMode_16x8, AIE4_BMode_16x16, AIE4_BMode_32x32]> {
  let cppNamespace = "::xilinx::xllvm";
}

// `I32EnumAttr` is IntegerAttr-backed (no dialect attribute registration
// needed) and auto-generates a verifier rejecting any value outside the
// enumerated cases. That is exactly the range check we want: unlike an opaque
// `I32:$conf`, an out-of-range amode/bmode cannot be constructed at all.

//===----------------------------------------------------------------------===//
// MUL
//===----------------------------------------------------------------------===//

// llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf
//     <32 x float> (<32 x float>, <32 x float>, i32)
//
// Elementwise single-precision multiply of two 1024-bit vector registers into
// a 1024-bit accumulator, which for f32 is the same <32 x float> shape.
//
// NOTE: this intrinsic's `properties:` list is empty in the Peano intrinsic
// table -- it carries no `IntrNoMem` -- so LLVM treats it as potentially
// side-effecting. We deliberately do NOT attach `Pure`/`NoMemoryEffect` here;
// claiming a purity the intrinsic does not declare would let MLIR hoist or CSE
// it more aggressively than LLVM itself will.
def MulEwiseFpConfAIE4IntrOp :
    AIEVec4_IntrOp<"I1024.I1024.ACC1024.fp.mul.ewise.conf",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
                   UnitAttr:$zeroAcc,
                   AIE4_AMode:$amode,
                   AIE4_BMode:$bmode)> {
  let summary = "AIE4 elementwise f32 vector multiply (vmul.f)";

  // The default `ExtIntrOpBase` llvmBuilder forwards `intrOp->getOperands()`
  // verbatim, which would drop the config word entirely (it lives in
  // attributes here, not operands). Encode the attributes into the i32 the
  // intrinsic actually expects and append it as the trailing argument.
  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf",
                      ::xilinx::xllvm::encodeAIE4MulConf(
                          op.getZeroAcc(), op.getAmode(), op.getBmode()));
    (void) inst;
    $res = inst;
  }];
}

#endif // AIE_DIALECT_XLLVM_IR_XLLVMAIE4INTROPS_TD
