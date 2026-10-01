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
// Share the MAC resource identity with AIEVec through AIE4CtrlRegResources.h.
def AIE4_CRVecSatResource : Resource<"::xilinx::xllvm::AIE4CRVecSatResource">;

defvar AIE4_MacEffects = MemoryEffects<[MemRead<AIE4_CRVecSatResource>]>;

// SRS omits effect declarations so CSE and LICM treat it conservatively.

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

// llvm.aie4.I1024.I1024.ACC1024.fp.mac.ewise.conf
//     <32 x float> (<32 x float>, <32 x float>, <32 x float>, i32)
//
// Fused f32 MAC; LLVM argument order is (lhs, rhs, acc, conf).
// The float MUL/MAC intrinsics read crFPVMask/crVecRnd and update srFPVFlags,
// so none of these operations is Pure.
def MacEwiseFpConfAIE4IntrOp :
    AIEVec4_IntrOp<"I1024.I1024.ACC1024.fp.mac.ewise.conf",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
                   UnitAttr:$zeroAcc,
                   AIE4_AMode:$amode,
                   AIE4_BMode:$bmode)> {
  let summary = "AIE4 elementwise f32 vector multiply-accumulate (vmac.f)";

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I1024.I1024.ACC1024.fp.mac.ewise.conf",
                      ::xilinx::xllvm::encodeAIE4MulConf(
                          op.getZeroAcc(), op.getAmode(), op.getBmode()));
    (void) inst;
    $res = inst;
  }];
}

// llvm.aie4.I1024.I1024.ACC1024.fp.msc.ewise.conf
//     <32 x float> (<32 x float>, <32 x float>, <32 x float>, i32)
//
// Fused f32 multiply-subtract; LLVM argument order is (lhs, rhs, acc, conf).
def MscEwiseFpConfAIE4IntrOp :
    AIEVec4_IntrOp<"I1024.I1024.ACC1024.fp.msc.ewise.conf",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
                   UnitAttr:$zeroAcc,
                   AIE4_AMode:$amode,
                   AIE4_BMode:$bmode)> {
  let summary = "AIE4 elementwise f32 vector multiply-subtract (vmsc.f)";

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I1024.I1024.ACC1024.fp.msc.ewise.conf",
                      ::xilinx::xllvm::encodeAIE4MulConf(
                          op.getZeroAcc(), op.getAmode(), op.getBmode()));
    (void) inst;
    $res = inst;
  }];
}

//===----------------------------------------------------------------------===//
// bf16 x bf16 -> f32 accumulator MUL/MAC
//===----------------------------------------------------------------------===//
//
// This 32-lane form uses bf16 inputs and an f32 result. The generic bf16
// multiply uses a separate 64-lane form with a bf16 result.
//
// llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf
//     <32 x float> (<32 x bfloat>, <32 x bfloat>, i32)
def MulEwiseBf16ConfAIE4IntrOp :
    AIEVec4_IntrOp<"bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
                   UnitAttr:$zeroAcc,
                   AIE4_AMode:$amode,
                   AIE4_BMode:$bmode,
                   UnitAttr:$xIsBf16,
                   UnitAttr:$yIsBf16)> {
  let summary = "AIE4 elementwise bf16 x bf16 -> f32 vector multiply (vmul.f)";
  let description = [{
    The bf16 operands use 512-bit `x` registers; the f32 result uses a
    1024-bit `y` accumulator.
  }];
  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf",
                      ::xilinx::xllvm::encodeAIE4Bf16FpConf(
                          op.getZeroAcc(), op.getAmode(), op.getBmode(),
                          op.getXIsBf16(), op.getYIsBf16()));
    (void) inst;
    $res = inst;
  }];
}

// llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.mac.ewise.conf
//     <32 x float> (<32 x bfloat>, <32 x bfloat>, <32 x float>, i32)
def MacEwiseBf16ConfAIE4IntrOp :
    AIEVec4_IntrOp<"bf16.I512.bf16.I512.ACC1024.fp.mac.ewise.conf",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
                   UnitAttr:$zeroAcc,
                   AIE4_AMode:$amode,
                   AIE4_BMode:$bmode,
                   UnitAttr:$xIsBf16,
                   UnitAttr:$yIsBf16)> {
  let summary = "AIE4 elementwise bf16 x bf16 -> f32 vector MAC (vmac.f)";

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.mac.ewise.conf",
                      ::xilinx::xllvm::encodeAIE4Bf16FpConf(
                          op.getZeroAcc(), op.getAmode(), op.getBmode(),
                          op.getXIsBf16(), op.getYIsBf16()));
    (void) inst;
    $res = inst;
  }];
}

// llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.msc.ewise.conf
//     <32 x float> (<32 x bfloat>, <32 x bfloat>, <32 x float>, i32)
def MscEwiseBf16ConfAIE4IntrOp :
    AIEVec4_IntrOp<"bf16.I512.bf16.I512.ACC1024.fp.msc.ewise.conf",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
                   UnitAttr:$zeroAcc,
                   AIE4_AMode:$amode,
                   AIE4_BMode:$bmode,
                   UnitAttr:$xIsBf16,
                   UnitAttr:$yIsBf16)> {
  let summary = "AIE4 elementwise bf16 x bf16 -> f32 vector multiply-subtract (vmsc.f)";

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.msc.ewise.conf",
                      ::xilinx::xllvm::encodeAIE4Bf16FpConf(
                          op.getZeroAcc(), op.getAmode(), op.getBmode(),
                          op.getXIsBf16(), op.getYIsBf16()));
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
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>>]>,
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

//===----------------------------------------------------------------------===//
// UPS
//===----------------------------------------------------------------------===//

// The `sign` argument selects between `vups.2x ..., upssign0` (unsigned) and
// `..., upssign1` (signed) -- a separate i32 operand, not a packed field, just
// like SRS's `sign`. Verified by disassembly: `(v, 0, 1)` emits `upssign1`,
// `(v, 0, 0)` emits `upssign0`.
def AIE4_UPSSign_Unsigned : I32EnumAttrCase<"Unsigned", 0, "unsigned">;
def AIE4_UPSSign_Signed   : I32EnumAttrCase<"Signed",   1, "signed">;

def AIE4_UPSSign : I32EnumAttr<"AIE4UPSSign",
    "AIE4 UPS signedness (selects the upssign0/upssign1 instruction form)",
    [AIE4_UPSSign_Unsigned, AIE4_UPSSign_Signed]> {
  let cppNamespace = "::xilinx::xllvm";
}

// Widens i16 lanes, then shifts under the ambient UPS and saturation modes.
// Reads crSat/crUPSMode/upsSign and updates srUPS_of.
def Acc32V32I512UpsAIE4IntrOp :
    AIEVec4_IntrOp<"acc32.v32.I512.ups",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$src,
                   ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<31>]>:$shift,
                   AIE4_UPSSign:$sign)> {
  let summary = "AIE4 up-shift i16 vector -> i32 accumulator (vups.2x)";
  let description = [{
    `shift` is in 0..31. Automatic integer widening uses only shift zero;
    explicit nonzero shifts can overflow according to the ambient modes.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.acc32.v32.I512.ups",
                      ::llvm::ArrayRef<uint32_t>{
                          (uint32_t) op.getShift(),
                          (uint32_t) op.getSign()});
    (void) inst;
    $res = inst;
  }];
}

//===----------------------------------------------------------------------===//
// Unpack
//===----------------------------------------------------------------------===//

// llvm.aie4.unpack.I512.I16.I8 <32 x i16> (<32 x i8>, i32 sign)
//
// The backend sets the unpack size; the result is independent of caller mode
// state. AIE4_UPSSign also encodes unpack signedness: 0 = zero-extend,
// 1 = sign-extend.
def UnpackI512I16I8AIE4IntrOp :
    AIEVec4_IntrOp<"unpack.I512.I16.I8",
        [Pure, TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I8]>:$src,
                   AIE4_UPSSign:$sign)> {
  let summary = "AIE4 widen i8 vector -> i16 vector, exact (vunpack)";

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.unpack.I512.I16.I8",
                      ::llvm::ArrayRef<uint32_t>{(uint32_t) op.getSign()});
    (void) inst;
    $res = inst;
  }];
}

// Exact bf16 widening has no control-register dependencies.
def V32Bf16ToV32AccFloatAIE4IntrOp :
    AIEVec4_IntrOp<"v32bf16.to.v32accfloat",
        [Pure, TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$src)> {
  let summary = "AIE4 widen bf16 vector -> f32 accumulator, exact (vconv.fp32.bf16)";
}

// Narrowing reads crRnd/crFPConvSat/crF2FMask and updates srF2FFlags.
// Rounding is toward negative infinity when crRnd == 0; callers supply
// the ambient modes. This operation does not model arith.truncf semantics.
def V32AccFloatToV32Bfloat16AIE4IntrOp :
    AIEVec4_IntrOp<"v32accfloat.to.v32bfloat16",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 narrow f32 accumulator -> bf16 vector, ambient rounding (vconv.bf16.fp32)";
}

//===----------------------------------------------------------------------===//
// Float -> integer conversion
//===----------------------------------------------------------------------===//
//
// `scale` is a binary exponent applied before the conversion; `sign` selects
// the signed or unsigned form. Rounding of the discarded fraction comes from
// `crRnd`, so a caller with a rounding contract to honour has to establish it.

def V32FloatToV32Int32AIE4IntrOp :
    AIEVec4_IntrOp<"v32float.to.v32int32",
        [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src,
                   ConfinedAttr<I32Attr, [IntMinValue<-32>, IntMaxValue<31>]>:$scale,
                   AIE4_SRSSign:$sign)> {
  let summary = "AIE4 convert f32 vector -> i32 vector, ambient rounding";

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.v32float.to.v32int32",
                      ::llvm::ArrayRef<uint32_t>{
                          (uint32_t) op.getScale(),
                          (uint32_t) op.getSign()});
    (void) inst;
    $res = inst;
  }];
}


//===----------------------------------------------------------------------===//
// Integer min/max/compare/select ("vadd" opsel multiplexer)
//===----------------------------------------------------------------------===//
//
// The vadd family selects integer min/max/compare behavior with `opsel`:
//   {<VEC>, <MASK>} (<VEC> lhs, <VEC> rhs, i32 sign, i32 opsel, i32 fporder)
// These wrappers expose min/max and ordered comparisons. Integer patterns
// require fporder = 0. The 128-lane i8 form is unsupported because its mask
// exceeds the 64-bit carriers used by the AIEVec operations.
def AIE4_VCmpOpSel_MinGe : I32EnumAttrCase<"MinGe", 13, "min_ge">;
def AIE4_VCmpOpSel_MaxLt : I32EnumAttrCase<"MaxLt", 14, "max_lt">;
def AIE4_VCmpOpSel_Lt    : I32EnumAttrCase<"Lt",    15, "lt">;
def AIE4_VCmpOpSel_Ge    : I32EnumAttrCase<"Ge",    17, "ge">;

def AIE4_VCmpOpSel : I32EnumAttr<"AIE4VCmpOpSel",
    "AIE4 vadd-family opsel value selecting min/max/compare (int forms only)",
    [AIE4_VCmpOpSel_MinGe, AIE4_VCmpOpSel_MaxLt, AIE4_VCmpOpSel_Lt,
     AIE4_VCmpOpSel_Ge]> {
  let cppNamespace = "::xilinx::xllvm";
}

class AIE4VCmpIntrOp<string mnemonic, Type vecTy, Type maskTy> :
    AIEVec4_IntrOp<mnemonic,
        [TypeIs<"res", LLVM_StructOf<[vecTy, maskTy]>>], /*numResults=*/2>,
    Arguments<(ins vecTy:$lhs, vecTy:$rhs,
                   AIE4_OperandSign:$sign,
                   AIE4_VCmpOpSel:$opsel)> {
  let summary = "AIE4 vector min/max/compare (vmin_ge/vmax_lt/vlt/vge)";
  let description = [{
    The result struct contains a vector in field 0 and a packed per-lane
    mask in field 1. Field 0 holds min/max results for `MinGe`/`MaxLt` and
    must remain unused for `Lt`/`Ge`, which produce only a mask.

    `sign` selects `vaddSign0` (unsigned) or `vaddSign1` (signed), relying on
    their ABI values. The mask uses the native `vsel` selector type: `i32`
    for up to 32 lanes, or `vector<2xi32>` for 64 lanes.

    LLVM translation appends `sign`, `opsel`, and the required `fporder = 0`.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.}] # mnemonic # [{",
                      ::llvm::ArrayRef<uint32_t>{
                          (uint32_t) op.getSign(),
                          (uint32_t) op.getOpsel(),
                          0u});
    (void) inst;
    $res = inst;
  }];
}

def VAdd8I512CmpAIE4IntrOp :
    AIE4VCmpIntrOp<"vadd8.512",
        FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
        FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd16I512CmpAIE4IntrOp :
    AIE4VCmpIntrOp<"vadd16.512",
        FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>, I32>;
def VAdd16I1024CmpAIE4IntrOp :
    AIE4VCmpIntrOp<"vadd16.1024",
        FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
        FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd32I512CmpAIE4IntrOp :
    AIE4VCmpIntrOp<"vadd32.512",
        FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>, I32>;
def VAdd32I1024CmpAIE4IntrOp :
    AIE4VCmpIntrOp<"vadd32.1024",
        FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>, I32>;

// Select signature:
//   <VEC> (<VEC> lhs, <VEC> rhs, i32 sign, i32 opsel, <MASK> sel, i32 fporder)
// Translation inserts the required literals sign = 0, opsel = 2, fporder = 0.
// The mask appears between opsel and fporder, so it needs a dedicated builder.
class AIE4VSelIntrOp<string mnemonic, Type vecTy, Type maskTy> :
    AIEVec4_IntrOp<mnemonic, [TypeIs<"res", vecTy>]>,
    Arguments<(ins vecTy:$lhs, vecTy:$rhs, maskTy:$sel)> {
  let summary = "AIE4 vector lane select (vsel)";
  let description = [{
    Selects lane i of `lhs` when mask bit i is clear, or lane i of `rhs`
    when it is set.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4SelIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.}] # mnemonic # [{",
                      /*sign=*/0u);
    (void) inst;
    $res = inst;
  }];
}

def VAdd8I512SelAIE4IntrOp :
    AIE4VSelIntrOp<"vadd8.mask.512",
        FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
        FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd16I512SelAIE4IntrOp :
    AIE4VSelIntrOp<"vadd16.mask.512",
        FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>, I32>;
def VAdd16I1024SelAIE4IntrOp :
    AIE4VSelIntrOp<"vadd16.mask.1024",
        FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
        FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd32I512SelAIE4IntrOp :
    AIE4VSelIntrOp<"vadd32.mask.512",
        FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>, I32>;
def VAdd32I1024SelAIE4IntrOp :
    AIE4VSelIntrOp<"vadd32.mask.1024",
        FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>, I32>;

//===----------------------------------------------------------------------===//
// Float min/max/compare selectors for the vaddfloat-family intrinsics
//===----------------------------------------------------------------------===//
//
// The backend's literal fporder=0 does not set crFPOrder or model a register
// dependency. Native operations read ambient state through helper calls;
// source IEEE arithmetic uses integer comparisons to be independent of it.
// Selectors differ by element format; fp32 uses GE=48 and LT=49.
def AIE4_VFCmpOpSel_Bf16MinGe : I32EnumAttrCase<"Bf16MinGe", 20, "bf16_min_ge">;
def AIE4_VFCmpOpSel_Bf16MaxLt : I32EnumAttrCase<"Bf16MaxLt", 24, "bf16_max_lt">;
def AIE4_VFCmpOpSel_Bf16Lt    : I32EnumAttrCase<"Bf16Lt",    40, "bf16_lt">;
def AIE4_VFCmpOpSel_Bf16Ge    : I32EnumAttrCase<"Bf16Ge",    44, "bf16_ge">;
def AIE4_VFCmpOpSel_Fp16MinGe : I32EnumAttrCase<"Fp16MinGe", 21, "fp16_min_ge">;
def AIE4_VFCmpOpSel_Fp16MaxLt : I32EnumAttrCase<"Fp16MaxLt", 25, "fp16_max_lt">;
def AIE4_VFCmpOpSel_Fp16Lt    : I32EnumAttrCase<"Fp16Lt",    41, "fp16_lt">;
def AIE4_VFCmpOpSel_Fp16Ge    : I32EnumAttrCase<"Fp16Ge",    45, "fp16_ge">;
def AIE4_VFCmpOpSel_Fp32MinGe : I32EnumAttrCase<"Fp32MinGe", 36, "fp32_min_ge">;
def AIE4_VFCmpOpSel_Fp32MaxLt : I32EnumAttrCase<"Fp32MaxLt", 37, "fp32_max_lt">;
def AIE4_VFCmpOpSel_Fp32Ge    : I32EnumAttrCase<"Fp32Ge",    48, "fp32_ge">;
def AIE4_VFCmpOpSel_Fp32Lt    : I32EnumAttrCase<"Fp32Lt",    49, "fp32_lt">;

def AIE4_VFCmpOpSel : I32EnumAttr<"AIE4VFCmpOpSel",
    "AIE4 float min/max/compare selector for the operand element format",
    [AIE4_VFCmpOpSel_Bf16MinGe, AIE4_VFCmpOpSel_Bf16MaxLt,
     AIE4_VFCmpOpSel_Bf16Lt, AIE4_VFCmpOpSel_Bf16Ge,
     AIE4_VFCmpOpSel_Fp16MinGe, AIE4_VFCmpOpSel_Fp16MaxLt,
     AIE4_VFCmpOpSel_Fp16Lt, AIE4_VFCmpOpSel_Fp16Ge,
     AIE4_VFCmpOpSel_Fp32MinGe, AIE4_VFCmpOpSel_Fp32MaxLt,
     AIE4_VFCmpOpSel_Fp32Ge, AIE4_VFCmpOpSel_Fp32Lt]> {
  let cppNamespace = "::xilinx::xllvm";
}

def AIE4_Bf16VFCmpOpSel : ConfinedAttr<AIE4_VFCmpOpSel,
    [IntIsOneOf<[AIE4_VFCmpOpSel_Bf16MinGe.value,
                 AIE4_VFCmpOpSel_Bf16MaxLt.value,
                 AIE4_VFCmpOpSel_Bf16Lt.value,
                 AIE4_VFCmpOpSel_Bf16Ge.value]>]>;
def AIE4_Fp16VFCmpOpSel : ConfinedAttr<AIE4_VFCmpOpSel,
    [IntIsOneOf<[AIE4_VFCmpOpSel_Fp16MinGe.value,
                 AIE4_VFCmpOpSel_Fp16MaxLt.value,
                 AIE4_VFCmpOpSel_Fp16Lt.value,
                 AIE4_VFCmpOpSel_Fp16Ge.value]>]>;
def AIE4_Fp32VFCmpOpSel : ConfinedAttr<AIE4_VFCmpOpSel,
    [IntIsOneOf<[AIE4_VFCmpOpSel_Fp32MinGe.value,
                 AIE4_VFCmpOpSel_Fp32MaxLt.value,
                 AIE4_VFCmpOpSel_Fp32Ge.value,
                 AIE4_VFCmpOpSel_Fp32Lt.value]>]>;

def AIE4_CRFPOrderResource : Resource<"::xilinx::xllvm::AIE4CRFPOrderResource">;

class AIE4VFCmpIntrOp<string mnemonic, Type vecTy, Type maskTy,
                    Attr opselAttr> :
    AIEVec4_IntrOp<mnemonic,
        [MemoryEffects<[MemRead<AIE4_CRFPOrderResource>]>,
         TypeIs<"res", LLVM_StructOf<[vecTy, maskTy]>>], /*numResults=*/2>,
    Arguments<(ins vecTy:$lhs, vecTy:$rhs, opselAttr:$opsel)> {
  let summary = "AIE4 float vector min/max/compare (vmin_ge/vmax_lt/vlt/vge, float forms)";
  let description = [{
    Native float min/max/compare using ambient `crFPOrder`. The intrinsic
    receives literal zero for `sign` (ignored) and `fporder`; the backend
    does not change the ordering mode for that literal.

    Translation uses a `noinline` helper with `llvm.sideeffect` to preserve
    ordering across mode changes during optimization and scheduling. This
    adds call overhead until Peano models the intrinsic's FPOrder dependency.

    The vector result is undefined for `Lt`/`Ge` selectors.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4FPOrderIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.}] # mnemonic # [{",
                      (uint32_t) op.getOpsel());
    (void) inst;
    $res = inst;
  }];
}

def VAddBf16I512CmpAIE4IntrOp :
    AIE4VFCmpIntrOp<"vaddbfloat16.512",
        FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>, I32, AIE4_Bf16VFCmpOpSel>;
def VAddBf16I1024CmpAIE4IntrOp :
    AIE4VFCmpIntrOp<"vaddbfloat16.1024",
        FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>,
        FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>, AIE4_Bf16VFCmpOpSel>;
def VAddFp16I512CmpAIE4IntrOp :
    AIE4VFCmpIntrOp<"vaddfloat16.512",
        FixedVectorOfRankAndLengthAndType<[1], [32], [F16]>, I32, AIE4_Fp16VFCmpOpSel>;
def VAddFp16I1024CmpAIE4IntrOp :
    AIE4VFCmpIntrOp<"vaddfloat16.1024",
        FixedVectorOfRankAndLengthAndType<[1], [64], [F16]>,
        FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>, AIE4_Fp16VFCmpOpSel>;
def VAddFp32I512CmpAIE4IntrOp :
    AIE4VFCmpIntrOp<"vaddfloat.512",
        FixedVectorOfRankAndLengthAndType<[1], [16], [F32]>, I32, AIE4_Fp32VFCmpOpSel>;
def VAddFp32I1024CmpAIE4IntrOp :
    AIE4VFCmpIntrOp<"vaddfloat.1024",
        FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>, I32, AIE4_Fp32VFCmpOpSel>;

//===----------------------------------------------------------------------===//
// 1024-bit vector reductions (min/max/add, i32 and f32)
//===----------------------------------------------------------------------===//
//
// Only 1024-bit reductions are available. The intrinsics are `Pure` because
// their instructions neither access memory nor read control registers.
// Integer forms always take a seed; a literal zero folds to the unseeded
// instruction. Float forms use separate seeded (`.scl`) intrinsics.

// llvm.aie4.min.reduce.I1024 / llvm.aie4.max.reduce.I1024 /
// llvm.aie4.add.reduce.I1024
//     i32 (<32 x i32> src, i32 seed)
//
class AIE4IntReduceIntrOp<string mnemonic> :
    AIEVec4_IntrOp<mnemonic, [Pure, TypeIs<"res", I32>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$src, I32:$seed)> {
  let summary = "AIE4 i32 1024-bit vector reduction (vminr/vmaxr/vaddr)";
  let description = [{
    A literal zero seed selects the unseeded hardware form through
    `PatAbsMinMaxAddReduceZero`; other seeds select the scalar form.
  }];
}

def MinReduceI1024AIE4IntrOp : AIE4IntReduceIntrOp<"min.reduce.I1024">;
def MaxReduceI1024AIE4IntrOp : AIE4IntReduceIntrOp<"max.reduce.I1024">;
def AddReduceI1024AIE4IntrOp : AIE4IntReduceIntrOp<"add.reduce.I1024">;

// llvm.aie4.{min,max,add}.reduce.FP1024
//     float (<32 x float> src)
// llvm.aie4.{min,max,add}.reduce.FP1024.scl
//     float (<32 x float> src, float seed)
//
class AIE4FpReduceIntrOp<string mnemonic> :
    AIEVec4_IntrOp<mnemonic, [Pure, TypeIs<"res", F32>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 f32 1024-bit vector reduction, no seed (vminr.f/vmaxr.f/vaddr.f)";
}

class AIE4FpReduceSclIntrOp<string mnemonic> :
    AIEVec4_IntrOp<mnemonic, [Pure, TypeIs<"res", F32>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src, F32:$seed)> {
  let summary = "AIE4 f32 1024-bit vector reduction, with seed (vminr.f/vmaxr.f/vaddr.f)";
  let description = [{
    For `add`, VADDR.f accumulates in a wider adder and rounds once. Its
    result can differ from a sequential f32 sum.
  }];
}

def MinReduceFP1024AIE4IntrOp : AIE4FpReduceIntrOp<"min.reduce.FP1024">;
def MinReduceFP1024SclAIE4IntrOp : AIE4FpReduceSclIntrOp<"min.reduce.FP1024.scl">;
def MaxReduceFP1024AIE4IntrOp : AIE4FpReduceIntrOp<"max.reduce.FP1024">;
def MaxReduceFP1024SclAIE4IntrOp : AIE4FpReduceSclIntrOp<"max.reduce.FP1024.scl">;
def AddReduceFP1024AIE4IntrOp : AIE4FpReduceIntrOp<"add.reduce.FP1024">;
def AddReduceFP1024SclAIE4IntrOp : AIE4FpReduceSclIntrOp<"add.reduce.FP1024.scl">;

//===----------------------------------------------------------------------===//
// Shuffle
//===----------------------------------------------------------------------===//
//
// llvm.aie4.vshuffle: v16i32 (v16i32, v16i32, i32).
// Hardware takes mode in a register; this op uses a verified attribute to
// restrict calls to supported modes. Keep these encodings in sync with
// AIEVec's AIE4ShuffleMode and Peano's aie4/aie4_enums.h.
def AIE4_ShuffleMode_T32_4X4 :
    I32EnumAttrCase<"T32_4X4", 90, "t32_4x4">;
def AIE4_ShuffleMode_T8_8X8 :
    I32EnumAttrCase<"T8_8X8", 72, "t8_8x8">;
def AIE4_ShuffleMode_T32_2X16_LO :
    I32EnumAttrCase<"T32_2X16_LO", 66, "t32_2x16_lo">;
def AIE4_ShuffleMode_T32_2X16_HI :
    I32EnumAttrCase<"T32_2X16_HI", 67, "t32_2x16_hi">;

def AIE4_ShuffleMode : I32EnumAttr<"AIE4ShuffleMode",
    "AIE4 vshuffle mode (register operand, verified at compile time)",
    [AIE4_ShuffleMode_T32_4X4, AIE4_ShuffleMode_T8_8X8,
     AIE4_ShuffleMode_T32_2X16_LO, AIE4_ShuffleMode_T32_2X16_HI]> {
  let cppNamespace = "::xilinx::xllvm";
}

// The intrinsic is IntrNoMem and the ISA specifies no status/control
// register use, so the operation is pure.
def ShuffleAIE4IntrOp :
    AIEVec4_IntrOp<"vshuffle",
        [Pure, TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>:$xsrc0,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>:$xsrc1,
                   AIE4_ShuffleMode:$mode)> {
  let summary = "AIE4 512-bit vector shuffle/transpose (vshuffle)";
  let description = [{
    Shuffles two 512-bit vectors according to `mode`. The single-operand
    modes (`t32_4x4`, `t8_8x8`) ignore `xsrc1`; AIEVec lowering duplicates
    `xsrc0` into that operand.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.vshuffle",
                      (uint32_t) op.getMode());
    (void) inst;
    $res = inst;
  }];
}

//===----------------------------------------------------------------------===//
// Approximate transcendentals (vexp2, vtanh)
//===----------------------------------------------------------------------===//
//
// Pure: the intrinsics are IntrNoMem, raise no FP flags, and read no
// control registers. Callers supply
// VEXP2's input bias; see aievec.aie4.biased_exp2.

// llvm.aie4.exp2
//     <32 x bfloat> (<32 x float>)
def Exp2AIE4IntrOp :
    AIEVec4_IntrOp<"exp2",
        [Pure, TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 biased base-2 exponential, bf16 result (vexp2.bf16)";
}

// llvm.aie4.exp2.bf20
//     <32 x float> (<32 x float>)
def Exp2Bf20AIE4IntrOp :
    AIEVec4_IntrOp<"exp2.bf20",
        [Pure, TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 biased base-2 exponential, f32 result with 11 LSBs "
                "cleared (vexp2.fp32)";
}

// llvm.aie4.tanh
//     <32 x bfloat> (<32 x float>)
def TanhAIE4IntrOp :
    AIEVec4_IntrOp<"tanh",
        [Pure, TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>>]>,
    Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 hyperbolic tangent, bf16-truncated input (vtanh)";
}

//===----------------------------------------------------------------------===//
// Scalar reciprocal and reciprocal square root
//===----------------------------------------------------------------------===//

// Both read crFPNlfMask and write the srFPNlf status flags, so neither
// wrapper is Pure. Neither has a vector form; callers unroll.
def InvsqrtAIE4IntrOp :
    AIEVec4_IntrOp<"invsqrt", [TypeIs<"res", F32>]>,
    Arguments<(ins F32:$src)> {
  let summary = "AIE4 scalar f32 reciprocal square root (invsqrt)";
}

def InvAIE4IntrOp :
    AIEVec4_IntrOp<"inv", [TypeIs<"res", F32>]>,
    Arguments<(ins F32:$src)> {
  let summary = "AIE4 scalar f32 reciprocal (inv)";
}

#endif // AIE_DIALECT_XLLVM_IR_XLLVMAIE4INTROPS_TD
