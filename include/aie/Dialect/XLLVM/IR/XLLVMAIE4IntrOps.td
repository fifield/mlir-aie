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

// Block mode (conf bits 4:3): 1 selects i16 x i16 -> i32; 3 is used for f32.
// Modes 0 and 2 have no operand-width interpretation specified here.
def AIE4_BMode_Mode0  : I32EnumAttrCase<"Mode0",     0, "mode0">;
def AIE4_BMode_16x16  : I32EnumAttrCase<"Mode16x16", 1, "m16x16">;
def AIE4_BMode_Mode2  : I32EnumAttrCase<"Mode2",     2, "mode2">;
def AIE4_BMode_32x32  : I32EnumAttrCase<"Mode32x32", 3, "m32x32">;

def AIE4_BMode : I32EnumAttr<"AIE4BMode",
    "AIE4 vector multiplier block mode (conf bits 4:3)",
    [AIE4_BMode_Mode0, AIE4_BMode_16x16, AIE4_BMode_Mode2, AIE4_BMode_32x32]> {
  let cppNamespace = "::xilinx::xllvm";
}

// Integer operand signedness: conf bit 8 selects signed RHS, bit 9 signed LHS.
def AIE4_OperandSign_Unsigned : I32EnumAttrCase<"Unsigned", 0, "unsigned">;
def AIE4_OperandSign_Signed   : I32EnumAttrCase<"Signed",   1, "signed">;

def AIE4_OperandSign : I32EnumAttr<"AIE4OperandSign",
    "AIE4 integer multiplier per-operand signedness",
    [AIE4_OperandSign_Unsigned, AIE4_OperandSign_Signed]> {
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

//===----------------------------------------------------------------------===//
// AIE4 control-register side-effect resources
//===----------------------------------------------------------------------===//
//
// Share resource identities with AIEVec through AIE4CtrlRegResources.h.
def AIE4_CRSatResource     : Resource<"::xilinx::xllvm::AIE4CRSatResource">;
def AIE4_CRRndResource     : Resource<"::xilinx::xllvm::AIE4CRRndResource">;
def AIE4_CRSRSModeResource : Resource<"::xilinx::xllvm::AIE4CRSRSModeResource">;
def AIE4_SRSOverflowResource :
    Resource<"::xilinx::xllvm::AIE4SRSOverflowResource">;
def AIE4_CRVecSatResource  : Resource<"::xilinx::xllvm::AIE4CRVecSatResource">;

// Keep these effects consistent with the corresponding AIEVec operations.
defvar AIE4_SRSEffects = MemoryEffects<[
    MemRead<AIE4_CRSatResource>,
    MemRead<AIE4_CRRndResource>,
    MemRead<AIE4_CRSRSModeResource>,
    MemWrite<AIE4_SRSOverflowResource>]>;

defvar AIE4_MacEffects = MemoryEffects<[MemRead<AIE4_CRVecSatResource>]>;

//===----------------------------------------------------------------------===//
// MAC
//===----------------------------------------------------------------------===//

// llvm.aie4.I512.I512.ACC1024.mac.ewise.conf
//     <32 x i32> (<32 x i16>, <32 x i16>, <32 x i32>, i32)
//
// The multiplier widens i16 products into the i32 accumulator lanes.
def MacEwiseIntConfAIE4IntrOp :
    AIEVec4_IntrOp<"I512.I512.ACC1024.mac.ewise.conf",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>>, AIE4_MacEffects]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$lhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$rhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$acc,
                   UnitAttr:$zeroAcc,
                   AIE4_AMode:$amode,
                   AIE4_BMode:$bmode,
                   AIE4_OperandSign:$lhsSign,
                   AIE4_OperandSign:$rhsSign)> {
  let summary = "AIE4 elementwise i16 x i16 -> i32 vector MAC (vmac)";
  let description = [{
    Multiplies the input lanes and accumulates into i32 lanes according to
    the configuration attributes. `lhsSign` and `rhsSign` explicitly select
    each operand's signedness. The instruction reads `crVecSat`.
  }];

  // Materialize the configuration attributes as the trailing i32 argument.
  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I512.I512.ACC1024.mac.ewise.conf",
                      ::xilinx::xllvm::encodeAIE4MacConf(
                          op.getZeroAcc(), op.getAmode(), op.getBmode(),
                          op.getLhsSign(), op.getRhsSign()));
    (void) inst;
    $res = inst;
  }];
}

//===----------------------------------------------------------------------===//
// SRS
//===----------------------------------------------------------------------===//

// A separate i32 sign argument selects the srssign0 or srssign1 form.
def AIE4_SRSSign_Unsigned : I32EnumAttrCase<"Unsigned", 0, "unsigned">;
def AIE4_SRSSign_Signed   : I32EnumAttrCase<"Signed",   1, "signed">;

def AIE4_SRSSign : I32EnumAttr<"AIE4SRSSign",
    "AIE4 SRS signedness (selects the srssign0/srssign1 instruction form)",
    [AIE4_SRSSign_Unsigned, AIE4_SRSSign_Signed]> {
  let cppNamespace = "::xilinx::xllvm";
}

// llvm.aie4.I512.v32.acc32.srs
//     <32 x i16> (<32 x i32>, i32 shift, i32 sign)
//
// Narrows i32 accumulator lanes to i16 using the current rounding and
// saturation modes, and writes SRS overflow status.
def I512V32Acc32SrsAIE4IntrOp :
    AIEVec4_IntrOp<"I512.v32.acc32.srs",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>>, AIE4_SRSEffects]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$acc,
                   ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<31>]>:$shift,
                   AIE4_SRSSign:$sign)> {
  let summary = "AIE4 shift-round-saturate i32 accumulator -> i16 (vsrs.2x)";
  let description = [{
    Narrows the accumulator using SRS. The verified attributes `shift`
    (0..31) and `sign` (unsigned=0, signed=1) become trailing i32 arguments
    in that order.

    Rounding and saturation depend on `crRnd` and `crSat`; this op does not
    establish or restore those modes. It reads the mode registers and writes
    SRS overflow status.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I512.v32.acc32.srs",
                      ::llvm::ArrayRef<uint32_t>{
                          (uint32_t) op.getShift(),
                          (uint32_t) op.getSign()});
    (void) inst;
    $res = inst;
  }];
}

#endif // AIE_DIALECT_XLLVM_IR_XLLVMAIE4INTROPS_TD
