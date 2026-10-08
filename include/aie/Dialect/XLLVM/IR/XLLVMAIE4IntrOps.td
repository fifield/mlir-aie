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
class AIEVec4_IntrOp<string mnemonic, list<Trait> traits = [],
                     int numResults = 1>
    : AIEVecIntrOpBase<"aie4", mnemonic, traits, numResults>;

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
// Configuration fields used by this operation:
//
//   bit    0   zero_acc   dynamically zero the accumulator before accumulating
//   bits 2:1   amode      arithmetic mode (operand data type)
//   bits 4:3   bmode      multiplier block mode (operand lane widths)
//
// The remaining bits (sign control etc.) are not meaningful for a
// floating-point elementwise multiply and are therefore not modelled; they are
// emitted as zero.

def AIE4_AMode_Int : I32EnumAttrCase<"Int", 0, "int">;
def AIE4_AMode_Float : I32EnumAttrCase<"Float", 1, "float">;
def AIE4_AMode_Res2 : I32EnumAttrCase<"Res2", 2, "reserved2">;
def AIE4_AMode_Res3 : I32EnumAttrCase<"Res3", 3, "reserved3">;

def AIE4_AMode
    : I32EnumAttr<"AIE4AMode",
                  "AIE4 vector multiplier arithmetic mode (conf bits 2:1)",
                  [AIE4_AMode_Int, AIE4_AMode_Float, AIE4_AMode_Res2,
                   AIE4_AMode_Res3]> {
  let cppNamespace = "::xilinx::xllvm";
}

// Block mode (conf bits 4:3): 1 selects i16 x i16 -> i32; 3 is used for f32.
// Modes 0 and 2 have no operand-width interpretation specified here.
def AIE4_BMode_Mode0 : I32EnumAttrCase<"Mode0", 0, "mode0">;
def AIE4_BMode_16x16 : I32EnumAttrCase<"Mode16x16", 1, "m16x16">;
def AIE4_BMode_Mode2 : I32EnumAttrCase<"Mode2", 2, "mode2">;
def AIE4_BMode_32x32 : I32EnumAttrCase<"Mode32x32", 3, "m32x32">;

def AIE4_BMode
    : I32EnumAttr<"AIE4BMode",
                  "AIE4 vector multiplier block mode (conf bits 4:3)",
                  [AIE4_BMode_Mode0, AIE4_BMode_16x16, AIE4_BMode_Mode2,
                   AIE4_BMode_32x32]> {
  let cppNamespace = "::xilinx::xllvm";
}

// Integer operand signedness: conf bit 8 selects signed RHS, bit 9 signed LHS.
def AIE4_OperandSign_Unsigned : I32EnumAttrCase<"Unsigned", 0, "unsigned">;
def AIE4_OperandSign_Signed : I32EnumAttrCase<"Signed", 1, "signed">;

def AIE4_OperandSign
    : I32EnumAttr<"AIE4OperandSign",
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
def MulEwiseFpConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "I1024.I1024.ACC1024.fp.mul.ewise.conf",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
          UnitAttr:$zeroAcc, AIE4_AMode:$amode, AIE4_BMode:$bmode)> {
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
def MacEwiseIntConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "I512.I512.ACC1024.mac.ewise.conf",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>>,
           AIE4_MacEffects]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$lhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$rhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$acc,
          UnitAttr:$zeroAcc, AIE4_AMode:$amode, AIE4_BMode:$bmode,
          AIE4_OperandSign:$lhsSign, AIE4_OperandSign:$rhsSign)> {
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
def MacEwiseFpConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "I1024.I1024.ACC1024.fp.mac.ewise.conf",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
          UnitAttr:$zeroAcc, AIE4_AMode:$amode, AIE4_BMode:$bmode)> {
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
def MscEwiseFpConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "I1024.I1024.ACC1024.fp.msc.ewise.conf",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
          UnitAttr:$zeroAcc, AIE4_AMode:$amode, AIE4_BMode:$bmode)> {
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
def MulEwiseBf16ConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
          UnitAttr:$zeroAcc, AIE4_AMode:$amode, AIE4_BMode:$bmode,
          UnitAttr:$xIsBf16, UnitAttr:$yIsBf16)> {
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
def MacEwiseBf16ConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "bf16.I512.bf16.I512.ACC1024.fp.mac.ewise.conf",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
          UnitAttr:$zeroAcc, AIE4_AMode:$amode, AIE4_BMode:$bmode,
          UnitAttr:$xIsBf16, UnitAttr:$yIsBf16)> {
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
def MscEwiseBf16ConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "bf16.I512.bf16.I512.ACC1024.fp.msc.ewise.conf",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
          FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc,
          UnitAttr:$zeroAcc, AIE4_AMode:$amode, AIE4_BMode:$bmode,
          UnitAttr:$xIsBf16, UnitAttr:$yIsBf16)> {
  let summary =
      "AIE4 elementwise bf16 x bf16 -> f32 vector multiply-subtract (vmsc.f)";

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
def AIE4_SRSSign_Signed : I32EnumAttrCase<"Signed", 1, "signed">;

def AIE4_SRSSign : I32EnumAttr<"AIE4SRSSign",
                               "AIE4 SRS signedness (selects the "
                               "srssign0/srssign1 instruction form)",
                               [AIE4_SRSSign_Unsigned, AIE4_SRSSign_Signed]> {
  let cppNamespace = "::xilinx::xllvm";
}

// llvm.aie4.I512.v32.acc32.srs
//     <32 x i16> (<32 x i32>, i32 shift, i32 sign)
//
// Narrows i32 accumulator lanes to i16 using the current rounding and
// saturation modes, and writes SRS overflow status.
def I512V32Acc32SrsAIE4IntrOp
    : AIEVec4_IntrOp<
          "I512.v32.acc32.srs",
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
    SRS overflow status. Peano sets `crSRSMode` to zero before the SRS.
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

def I512V64Acc32SrsAIE4IntrOp
    : AIEVec4_IntrOp<
          "I512.v64.acc32.srs",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [64], [I32]>:$acc,
          ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<59>]>:$shift,
          AIE4_SRSSign:$sign)> {
  let summary = "AIE4 shift-round-saturate i32 accumulator -> i8 (vsrs.4x)";
  let description = [{
    Narrows i32 lanes to i8. `shift` (0..59) and `sign` (unsigned=0,
    signed=1) become trailing i32 arguments. Left shifts are not supported.

    Reads `crRnd` and `crSat` without changing them and writes SRS overflow
    status. Peano sets `crSRSMode` to zero before the SRS.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I512.v64.acc32.srs",
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
def AIE4_UPSSign_Signed : I32EnumAttrCase<"Signed", 1, "signed">;

def AIE4_UPSSign : I32EnumAttr<"AIE4UPSSign",
                               "AIE4 UPS signedness (selects the "
                               "upssign0/upssign1 instruction form)",
                               [AIE4_UPSSign_Unsigned, AIE4_UPSSign_Signed]> {
  let cppNamespace = "::xilinx::xllvm";
}

// Widens i16 lanes, then shifts under the ambient UPS and saturation modes.
// Reads crSat/crUPSMode/upsSign and updates srUPS_of.
def Acc32V32I512UpsAIE4IntrOp
    : AIEVec4_IntrOp<
          "acc32.v32.I512.ups",
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
def UnpackI512I16I8AIE4IntrOp
    : AIEVec4_IntrOp<
          "unpack.I512.I16.I8",
          [Pure,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>>]>,
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
def V32Bf16ToV32AccFloatAIE4IntrOp
    : AIEVec4_IntrOp<
          "v32bf16.to.v32accfloat",
          [Pure,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(
          ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$src)> {
  let summary =
      "AIE4 widen bf16 vector -> f32 accumulator, exact (vconv.fp32.bf16)";
}

// Narrowing reads crRnd/crFPConvSat/crF2FMask and updates srF2FFlags.
// Rounding is toward negative infinity when crRnd == 0; callers supply
// the ambient modes. This operation does not model arith.truncf semantics.
def V32AccFloatToV32Bfloat16AIE4IntrOp
    : AIEVec4_IntrOp<
          "v32accfloat.to.v32bfloat16",
          [TypeIs<"res",
                  FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>>]>,
      Arguments<(
          ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 narrow f32 accumulator -> bf16 vector, ambient rounding "
                "(vconv.bf16.fp32)";
}

//===----------------------------------------------------------------------===//
// Float -> integer conversion
//===----------------------------------------------------------------------===//
//
// `scale` is a binary exponent applied before the conversion; `sign` selects
// the signed or unsigned form. Rounding of the discarded fraction comes from
// `crRnd`, so a caller with a rounding contract to honour has to establish it.

def V32FloatToV32Int32AIE4IntrOp
    : AIEVec4_IntrOp<
          "v32float.to.v32int32",
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
def AIE4_VCmpOpSel_Lt : I32EnumAttrCase<"Lt", 15, "lt">;
def AIE4_VCmpOpSel_Ge : I32EnumAttrCase<"Ge", 17, "ge">;

def AIE4_VCmpOpSel : I32EnumAttr<"AIE4VCmpOpSel",
                                 "AIE4 vadd-family opsel value selecting "
                                 "min/max/compare (int forms only)",
                                 [AIE4_VCmpOpSel_MinGe, AIE4_VCmpOpSel_MaxLt,
                                  AIE4_VCmpOpSel_Lt, AIE4_VCmpOpSel_Ge]> {
  let cppNamespace = "::xilinx::xllvm";
}

class AIE4VCmpIntrOp<string mnemonic, Type vecTy, Type maskTy>
    : AIEVec4_IntrOp<mnemonic, [TypeIs<"res", LLVM_StructOf<[vecTy, maskTy]>>],
                     /*numResults=*/2>,
      Arguments<(ins vecTy:$lhs, vecTy:$rhs, AIE4_OperandSign:$sign,
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
                      "llvm.aie4.}]#mnemonic#[{",
                      ::llvm::ArrayRef<uint32_t>{
                          (uint32_t) op.getSign(),
                          (uint32_t) op.getOpsel(),
                          0u});
    (void) inst;
    $res = inst;
  }];
}

def VAdd8I512CmpAIE4IntrOp
    : AIE4VCmpIntrOp<"vadd8.512",
                     FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                     FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd16I512CmpAIE4IntrOp
    : AIE4VCmpIntrOp<"vadd16.512",
                     FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>, I32>;
def VAdd16I1024CmpAIE4IntrOp
    : AIE4VCmpIntrOp<"vadd16.1024",
                     FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                     FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd32I512CmpAIE4IntrOp
    : AIE4VCmpIntrOp<"vadd32.512",
                     FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>, I32>;
def VAdd32I1024CmpAIE4IntrOp
    : AIE4VCmpIntrOp<"vadd32.1024",
                     FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>, I32>;

// Select signature:
//   <VEC> (<VEC> lhs, <VEC> rhs, i32 sign, i32 opsel, <MASK> sel, i32 fporder)
// Translation inserts the required literals sign = 0, opsel = 2, fporder = 0.
// The mask appears between opsel and fporder, so it needs a dedicated builder.
class AIE4VSelIntrOp<string mnemonic, Type vecTy, Type maskTy>
    : AIEVec4_IntrOp<mnemonic, [TypeIs<"res", vecTy>]>,
      Arguments<(ins vecTy:$lhs, vecTy:$rhs, maskTy:$sel)> {
  let summary = "AIE4 vector lane select (vsel)";
  let description = [{
    Selects lane i of `lhs` when mask bit i is clear, or lane i of `rhs`
    when it is set.
  }];

  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4SelIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.}]#mnemonic#[{",
                      /*sign=*/0u);
    (void) inst;
    $res = inst;
  }];
}

def VAdd8I512SelAIE4IntrOp
    : AIE4VSelIntrOp<"vadd8.mask.512",
                     FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                     FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd16I512SelAIE4IntrOp
    : AIE4VSelIntrOp<"vadd16.mask.512",
                     FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>, I32>;
def VAdd16I1024SelAIE4IntrOp
    : AIE4VSelIntrOp<"vadd16.mask.1024",
                     FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                     FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>>;
def VAdd32I512SelAIE4IntrOp
    : AIE4VSelIntrOp<"vadd32.mask.512",
                     FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>, I32>;
def VAdd32I1024SelAIE4IntrOp
    : AIE4VSelIntrOp<"vadd32.mask.1024",
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
def AIE4_VFCmpOpSel_Bf16Lt : I32EnumAttrCase<"Bf16Lt", 40, "bf16_lt">;
def AIE4_VFCmpOpSel_Bf16Ge : I32EnumAttrCase<"Bf16Ge", 44, "bf16_ge">;
def AIE4_VFCmpOpSel_Fp16MinGe : I32EnumAttrCase<"Fp16MinGe", 21, "fp16_min_ge">;
def AIE4_VFCmpOpSel_Fp16MaxLt : I32EnumAttrCase<"Fp16MaxLt", 25, "fp16_max_lt">;
def AIE4_VFCmpOpSel_Fp16Lt : I32EnumAttrCase<"Fp16Lt", 41, "fp16_lt">;
def AIE4_VFCmpOpSel_Fp16Ge : I32EnumAttrCase<"Fp16Ge", 45, "fp16_ge">;
def AIE4_VFCmpOpSel_Fp32MinGe : I32EnumAttrCase<"Fp32MinGe", 36, "fp32_min_ge">;
def AIE4_VFCmpOpSel_Fp32MaxLt : I32EnumAttrCase<"Fp32MaxLt", 37, "fp32_max_lt">;
def AIE4_VFCmpOpSel_Fp32Ge : I32EnumAttrCase<"Fp32Ge", 48, "fp32_ge">;
def AIE4_VFCmpOpSel_Fp32Lt : I32EnumAttrCase<"Fp32Lt", 49, "fp32_lt">;

def AIE4_VFCmpOpSel
    : I32EnumAttr<
          "AIE4VFCmpOpSel",
          "AIE4 float min/max/compare selector for the operand element format",
          [AIE4_VFCmpOpSel_Bf16MinGe, AIE4_VFCmpOpSel_Bf16MaxLt,
           AIE4_VFCmpOpSel_Bf16Lt, AIE4_VFCmpOpSel_Bf16Ge,
           AIE4_VFCmpOpSel_Fp16MinGe, AIE4_VFCmpOpSel_Fp16MaxLt,
           AIE4_VFCmpOpSel_Fp16Lt, AIE4_VFCmpOpSel_Fp16Ge,
           AIE4_VFCmpOpSel_Fp32MinGe, AIE4_VFCmpOpSel_Fp32MaxLt,
           AIE4_VFCmpOpSel_Fp32Ge, AIE4_VFCmpOpSel_Fp32Lt]> {
  let cppNamespace = "::xilinx::xllvm";
}

def AIE4_Bf16VFCmpOpSel
    : ConfinedAttr<
          AIE4_VFCmpOpSel, [IntIsOneOf<[AIE4_VFCmpOpSel_Bf16MinGe.value,
                                        AIE4_VFCmpOpSel_Bf16MaxLt.value,
                                        AIE4_VFCmpOpSel_Bf16Lt.value,
                                        AIE4_VFCmpOpSel_Bf16Ge.value]>]>;
def AIE4_Fp16VFCmpOpSel
    : ConfinedAttr<
          AIE4_VFCmpOpSel, [IntIsOneOf<[AIE4_VFCmpOpSel_Fp16MinGe.value,
                                        AIE4_VFCmpOpSel_Fp16MaxLt.value,
                                        AIE4_VFCmpOpSel_Fp16Lt.value,
                                        AIE4_VFCmpOpSel_Fp16Ge.value]>]>;
def AIE4_Fp32VFCmpOpSel
    : ConfinedAttr<
          AIE4_VFCmpOpSel, [IntIsOneOf<[AIE4_VFCmpOpSel_Fp32MinGe.value,
                                        AIE4_VFCmpOpSel_Fp32MaxLt.value,
                                        AIE4_VFCmpOpSel_Fp32Ge.value,
                                        AIE4_VFCmpOpSel_Fp32Lt.value]>]>;

def AIE4_CRFPOrderResource : Resource<"::xilinx::xllvm::AIE4CRFPOrderResource">;

class AIE4VFCmpIntrOp<string mnemonic, Type vecTy, Type maskTy, Attr opselAttr>
    : AIEVec4_IntrOp<mnemonic,
                     [MemoryEffects<[MemRead<AIE4_CRFPOrderResource>]>,
                      TypeIs<"res", LLVM_StructOf<[vecTy, maskTy]>>],
                     /*numResults=*/2>,
      Arguments<(ins vecTy:$lhs, vecTy:$rhs, opselAttr:$opsel)> {
  let summary = "AIE4 float vector min/max/compare (vmin_ge/vmax_lt/vlt/vge, "
                "float forms)";
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
                      "llvm.aie4.}]#mnemonic#[{",
                      (uint32_t) op.getOpsel());
    (void) inst;
    $res = inst;
  }];
}

def VAddBf16I512CmpAIE4IntrOp
    : AIE4VFCmpIntrOp<"vaddbfloat16.512",
                      FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>, I32,
                      AIE4_Bf16VFCmpOpSel>;
def VAddBf16I1024CmpAIE4IntrOp
    : AIE4VFCmpIntrOp<"vaddbfloat16.1024",
                      FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>,
                      FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>,
                      AIE4_Bf16VFCmpOpSel>;
def VAddFp16I512CmpAIE4IntrOp
    : AIE4VFCmpIntrOp<"vaddfloat16.512",
                      FixedVectorOfRankAndLengthAndType<[1], [32], [F16]>, I32,
                      AIE4_Fp16VFCmpOpSel>;
def VAddFp16I1024CmpAIE4IntrOp
    : AIE4VFCmpIntrOp<"vaddfloat16.1024",
                      FixedVectorOfRankAndLengthAndType<[1], [64], [F16]>,
                      FixedVectorOfRankAndLengthAndType<[1], [2], [I32]>,
                      AIE4_Fp16VFCmpOpSel>;
def VAddFp32I512CmpAIE4IntrOp
    : AIE4VFCmpIntrOp<"vaddfloat.512",
                      FixedVectorOfRankAndLengthAndType<[1], [16], [F32]>, I32,
                      AIE4_Fp32VFCmpOpSel>;
def VAddFp32I1024CmpAIE4IntrOp
    : AIE4VFCmpIntrOp<"vaddfloat.1024",
                      FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>, I32,
                      AIE4_Fp32VFCmpOpSel>;

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
class AIE4IntReduceIntrOp<string mnemonic>
    : AIEVec4_IntrOp<mnemonic, [Pure, TypeIs<"res", I32>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$src,
          I32:$seed)> {
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
class AIE4FpReduceIntrOp<string mnemonic>
    : AIEVec4_IntrOp<mnemonic, [Pure, TypeIs<"res", F32>]>,
      Arguments<(
          ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary =
      "AIE4 f32 1024-bit vector reduction, no seed (vminr.f/vmaxr.f/vaddr.f)";
}

class AIE4FpReduceSclIntrOp<string mnemonic>
    : AIEVec4_IntrOp<mnemonic, [Pure, TypeIs<"res", F32>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src,
          F32:$seed)> {
  let summary =
      "AIE4 f32 1024-bit vector reduction, with seed (vminr.f/vmaxr.f/vaddr.f)";
  let description = [{
    For `add`, VADDR.f accumulates in a wider adder and rounds once. Its
    result can differ from a sequential f32 sum.
  }];
}

def MinReduceFP1024AIE4IntrOp : AIE4FpReduceIntrOp<"min.reduce.FP1024">;
def MinReduceFP1024SclAIE4IntrOp
    : AIE4FpReduceSclIntrOp<"min.reduce.FP1024.scl">;
def MaxReduceFP1024AIE4IntrOp : AIE4FpReduceIntrOp<"max.reduce.FP1024">;
def MaxReduceFP1024SclAIE4IntrOp
    : AIE4FpReduceSclIntrOp<"max.reduce.FP1024.scl">;
def AddReduceFP1024AIE4IntrOp : AIE4FpReduceIntrOp<"add.reduce.FP1024">;
def AddReduceFP1024SclAIE4IntrOp
    : AIE4FpReduceSclIntrOp<"add.reduce.FP1024.scl">;

//===----------------------------------------------------------------------===//
// Shuffle
//===----------------------------------------------------------------------===//
//
// llvm.aie4.vshuffle: v16i32 (v16i32, v16i32, i32).
// Hardware takes mode in a register; this op uses a verified attribute to
// restrict calls to supported modes. Keep these encodings in sync with
// AIEVec's AIE4ShuffleMode and Peano's aie4/aie4_enums.h.
def AIE4_ShuffleMode_T32_4X4 : I32EnumAttrCase<"T32_4X4", 90, "t32_4x4">;
def AIE4_ShuffleMode_T8_8X8 : I32EnumAttrCase<"T8_8X8", 72, "t8_8x8">;
def AIE4_ShuffleMode_T32_2X16_LO
    : I32EnumAttrCase<"T32_2X16_LO", 66, "t32_2x16_lo">;
def AIE4_ShuffleMode_T32_2X16_HI
    : I32EnumAttrCase<"T32_2X16_HI", 67, "t32_2x16_hi">;

def AIE4_ShuffleMode
    : I32EnumAttr<
          "AIE4ShuffleMode",
          "AIE4 vshuffle mode (register operand, verified at compile time)",
          [AIE4_ShuffleMode_T32_4X4, AIE4_ShuffleMode_T8_8X8,
           AIE4_ShuffleMode_T32_2X16_LO, AIE4_ShuffleMode_T32_2X16_HI]> {
  let cppNamespace = "::xilinx::xllvm";
}

// No memory or status/control register effects.
def ShuffleAIE4IntrOp
    : AIEVec4_IntrOp<
          "vshuffle",
          [Pure,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>>]>,
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
def Exp2AIE4IntrOp
    : AIEVec4_IntrOp<"exp2", [Pure,
                              TypeIs<"res", FixedVectorOfRankAndLengthAndType<
                                                [1], [32], [BF16]>>]>,
      Arguments<(
          ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 biased base-2 exponential, bf16 result (vexp2.bf16)";
}

// llvm.aie4.exp2.bf20
//     <32 x float> (<32 x float>)
def Exp2Bf20AIE4IntrOp
    : AIEVec4_IntrOp<
          "exp2.bf20",
          [Pure,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>>]>,
      Arguments<(
          ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 biased base-2 exponential, f32 result with 11 LSBs "
                "cleared (vexp2.fp32)";
}

// llvm.aie4.tanh
//     <32 x bfloat> (<32 x float>)
def TanhAIE4IntrOp
    : AIEVec4_IntrOp<"tanh", [Pure,
                              TypeIs<"res", FixedVectorOfRankAndLengthAndType<
                                                [1], [32], [BF16]>>]>,
      Arguments<(
          ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$src)> {
  let summary = "AIE4 hyperbolic tangent, bf16-truncated input (vtanh)";
}

//===----------------------------------------------------------------------===//
// Scalar reciprocal and reciprocal square root
//===----------------------------------------------------------------------===//

// Both read crFPNlfMask and write the srFPNlf status flags, so neither
// wrapper is Pure. Neither has a vector form; callers unroll.
def InvsqrtAIE4IntrOp : AIEVec4_IntrOp<"invsqrt", [TypeIs<"res", F32>]>,
                        Arguments<(ins F32:$src)> {
  let summary = "AIE4 scalar f32 reciprocal square root (invsqrt)";
}

def InvAIE4IntrOp : AIEVec4_IntrOp<"inv", [TypeIs<"res", F32>]>,
                    Arguments<(ins F32:$src)> {
  let summary = "AIE4 scalar f32 reciprocal (inv)";
}

//===----------------------------------------------------------------------===//
// Stream reads
//===----------------------------------------------------------------------===//

// Match aie.get_stream's generic intrinsic signature to share its declaration.
def GetSsAIE4IntrOp
    : AIEVec4_IntrOp<
          "get.ss",
          [TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>>]>,
      Arguments<(ins I32:$port)> {
  let summary = "AIE4 read 1024 bits from a stream port (vmovb y, ss)";
  let description = [{
    Pops one 1024-bit word from input stream `port`, blocking until available.
    Lowers to `llvm.aie4.get.ss` and selects `vmovb y, ss`.
    Consuming stream data is a side effect.
  }];
}

//===----------------------------------------------------------------------===//
// Dense matrix MAC (MMAC), staging and commit
//===----------------------------------------------------------------------===//
//
// Dense MMACs multiply one activation row by the committed matrix. Peano
// models the matrix as inaccessible memory: MMACs read it, while staging
// inserts and commits write it. These dependencies order the LLVM calls.
// The MLIR ops retain unknown effects to preserve execution and ordering
// through generic MLIR transformations.

// Matrix-state classification: staging writers (za), commits (za -> zb) and
// MMACs (read zb). The C++ classes are shared with AIEVec. Every staging or
// active-matrix writer must carry AIE4StagingWriter or AIE4MatrixCommit so
// that matrix_region rejects operations that would clobber its resident
// tile. MMACs only read and are allowed in region bodies.
def AIE4StagingWriter : NativeOpTrait<"AIE4StagingWriter"> {
  let cppNamespace = "::xilinx::xllvm";
}
def AIE4MatrixCommit : NativeOpTrait<"AIE4MatrixCommit"> {
  let cppNamespace = "::xilinx::xllvm";
}
def AIE4MatrixReader : NativeOpTrait<"AIE4MatrixReader"> {
  let cppNamespace = "::xilinx::xllvm";
}

// Integer MMAC precision occupies conf bit 3; bit 4 is not part of it.
def AIE4_MmacIntPrecision_Int4 : I32EnumAttrCase<"Int4x4", 0, "int4x4">;
def AIE4_MmacIntPrecision_Int8 : I32EnumAttrCase<"Int8x8", 1, "int8x8">;

def AIE4_MmacIntPrecision
    : I32EnumAttr<"AIE4MmacIntPrecision",
                  "AIE4 dense integer MMAC precision (conf bit 3)",
                  [AIE4_MmacIntPrecision_Int4, AIE4_MmacIntPrecision_Int8]> {
  let cppNamespace = "::xilinx::xllvm";
}

// fp8 X-operand NaN processing (conf bits 11:10). Value 3 is not defined.
// Dense matrix multiplication uses `all`; the others serve the 8x8 and
// convolution expansion modes.
def AIE4_MmacFp8EMode_All : I32EnumAttrCase<"All", 0, "all">;
def AIE4_MmacFp8EMode_Xpnd8x8
    : I32EnumAttrCase<"Expand8x8", 1, "expand_8x8_8x8">;
def AIE4_MmacFp8EMode_Convo : I32EnumAttrCase<"ExpandConvo", 2, "expand_convo">;

def AIE4_MmacFp8EMode
    : I32EnumAttr<"AIE4MmacFp8EMode",
                  "AIE4 fp8 MMAC X-operand NaN processing (conf bits 11:10)",
                  [AIE4_MmacFp8EMode_All, AIE4_MmacFp8EMode_Xpnd8x8,
                   AIE4_MmacFp8EMode_Convo]> {
  let cppNamespace = "::xilinx::xllvm";
}

// crIntMMode bit 0 selects signed data; bit 1 selects int8/bf16 (set) or
// int4 (clear). Signed int4 staging converts two's complement to
// sign/magnitude, so the MMAC Z signedness must match the staging mode.
def AIE4_IntMMode_UInt4 : I32EnumAttrCase<"UInt4", 0, "uint4">;
def AIE4_IntMMode_Int4 : I32EnumAttrCase<"Int4", 1, "int4">;
def AIE4_IntMMode_UInt8 : I32EnumAttrCase<"UInt8", 2, "uint8">;
def AIE4_IntMMode_Int8 : I32EnumAttrCase<"Int8", 3, "int8">;

def AIE4_IntMMode
    : I32EnumAttr<"AIE4IntMMode",
                  "AIE4 integer staging mode written to crIntMMode",
                  [AIE4_IntMMode_UInt4, AIE4_IntMMode_Int4, AIE4_IntMMode_UInt8,
                   AIE4_IntMMode_Int8]> {
  let cppNamespace = "::xilinx::xllvm";
}

// llvm.aie4.bf16.I512.ACC2048.mmac
//     <64 x float> (<64 x float> acc, <32 x bfloat> x, i32 conf)
//
// One 1x32 bf16 row times the resident 32x64 bf16 matrix (AIE4b only).
// Only conf bit 0 (zero_acc) applies.
def MmacBf16I512Acc2048AIE4IntrOp
    : AIEVec4_IntrOp<
          "bf16.I512.ACC2048.mmac",
          [AIE4MatrixReader,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [64], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [64], [F32]>:$acc,
          FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
          UnitAttr:$zeroAcc)> {
  let summary = "AIE4 dense bf16 matrix MAC, one row (mmac.f ... zb)";
  let description = [{
    `acc + lhs x B`, where B is the committed bf16 matrix. With `zeroAcc`,
    the hardware replaces `acc` by zeros. Commit B with
    `staging.to.matrix.I32768`.
  }];
  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.bf16.I512.ACC2048.mmac",
                      op.getZeroAcc() ? 1u : 0u);
    (void) inst;
    $res = inst;
  }];
}

// llvm.aie4.I512.ACC2048.mmac
//     <64 x i32> (<64 x i32> acc, <64 x i8> x, i32 conf)
//
// Dense int8 (bmode = int8x8: 1x64 row, 64x64 matrix) and int4
// (bmode = int4x4: 1x128 row as packed nibbles, 128x64 matrix).
def MmacI512Acc2048AIE4IntrOp
    : AIEVec4_IntrOp<
          "I512.ACC2048.mmac",
          [AIE4MatrixReader,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [64], [I32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [64], [I32]>:$acc,
          FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>:$lhs,
          UnitAttr:$zeroAcc, AIE4_MmacIntPrecision:$bmode,
          AIE4_OperandSign:$xSign, AIE4_OperandSign:$zSign)> {
  let summary = "AIE4 dense integer matrix MAC, one row (mmac.i ... zb)";
  let description = [{
    `acc + lhs x B` in i32 lanes. `xSign` is the activation signedness (conf
    bit 9) and `zSign` the matrix signedness (bit 8). For int4, `zSign` must
    match the crIntMMode used to stage B.
  }];
  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I512.ACC2048.mmac",
                      ::xilinx::xllvm::encodeAIE4MmacIntConf(
                          op.getZeroAcc(), op.getBmode(), op.getXSign(),
                          op.getZSign()));
    (void) inst;
    $res = inst;
  }];
}

def MmacI512Acc2048DynConfAIE4IntrOp
    : AIEVec4_IntrOp<
          "I512.ACC2048.mmac.dyn_conf",
          [AIE4MatrixReader,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [64], [I32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [64], [I32]>:$acc,
          FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>:$lhs, I32:$conf)> {
  let summary = "AIE4 dense integer matrix MAC with a dynamic conf word";
  let description = [{
    Uses the same intrinsic as `intr.aie4.I512.ACC2048.mmac`, with an unchecked
    SSA `conf` operand for runtime choices such as zeroing the first K-tile.
    Only int8x8/int4x4 configurations are supported: bit 0 zero_acc, bit 3
    bmode, bit 8 zSign and bit 9 xSign; other bits must be zero.
    See `encodeAIE4MmacIntConf` for constant configurations.
  }];
  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createExternalLLVMIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.I512.ACC2048.mmac");
    (void) inst;
    $res = inst;
  }];
}

// llvm.aie4.fp8.I512.ACC2048.mmac
//     <64 x float> (<64 x float> acc, <64 x i8> x, i32 conf)
//
// One 1x64 fp8 (E4M3) row, carried as bytes, times the 64x64 fp8 matrix.
def MmacFp8I512Acc2048AIE4IntrOp
    : AIEVec4_IntrOp<
          "fp8.I512.ACC2048.mmac",
          [AIE4MatrixReader,
           TypeIs<"res", FixedVectorOfRankAndLengthAndType<[1], [64], [F32]>>]>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [64], [F32]>:$acc,
          FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>:$lhs,
          UnitAttr:$zeroAcc, AIE4_MmacFp8EMode:$emode)> {
  let summary = "AIE4 dense fp8 matrix MAC, one row (mmac.f ... f_zb)";
  let description = [{
    `acc + lhs x B` in f32 lanes, where `lhs` holds 64 fp8 (E4M3) values as
    bytes and B is the committed fp8 matrix. Commit B with
    `staging.to.matrix.FP32768`. With `zeroAcc`, the hardware replaces `acc`
    by zeros. `emode = all` (0) propagates a NaN in any of the 64 lhs
    elements to every output lane and is the mode for dense matrix
    multiplication. `expand_8x8_8x8` (1) and `expand_convo` (2) group NaN
    propagation according to their respective expansion layouts.
  }];
  let llvmBuilder = [{
    auto *inst = ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder,
                      moduleTranslation, &opInst,
                      "llvm.aie4.fp8.I512.ACC2048.mmac",
                      ::xilinx::xllvm::encodeAIE4MmacFp8Conf(
                          op.getZeroAcc(), op.getEmode()));
    (void) inst;
    $res = inst;
  }];
}

// llvm.aie4.insert.staging.bf16.I1024
//     void (<64 x bfloat> src, i32 index, i32 mode)
//
// bf16 staging fixes crIntMMode to 2 (int8/bf16 format, unsigned mode).
def InsertStagingBf16I1024AIE4IntrOp
    : AIEVec4_IntrOp<"insert.staging.bf16.I1024", [AIE4StagingWriter],
                     /*numResults=*/0>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>:$src,
          I32:$index)> {
  let summary =
      "AIE4 insert a 1024-bit bf16 chunk into matrix staging (vmov za[i], y)";
  let description = [{
    Writes 64 bf16 values (two 512-bit B columns) to staging chunk `index`
    (0..31). The translation passes crIntMMode = 2.
  }];
  let llvmBuilder = [{
    ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder, moduleTranslation,
        &opInst, "llvm.aie4.insert.staging.bf16.I1024",
        ::xilinx::xllvm::kAIE4IntMModeBf16);
  }];
}

// llvm.aie4.insert.staging.int.I1024
//     void (<128 x i8> src, i32 index, i32 mode)
//
def InsertStagingIntI1024AIE4IntrOp
    : AIEVec4_IntrOp<"insert.staging.int.I1024", [AIE4StagingWriter],
                     /*numResults=*/0>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [128], [I8]>:$src,
          I32:$index, AIE4_IntMMode:$mode)> {
  let summary = "AIE4 insert a 1024-bit integer chunk into matrix staging "
                "(vmov za[i], y)";
  let description = [{
    Writes 128 bytes (two 512-bit B columns) to staging chunk `index`
    (0..31) after setting crIntMMode to `mode`. int4 data is packed two
    nibbles per byte and converted from two's complement to sign/magnitude
    when `mode` is `int4`.
  }];
  let llvmBuilder = [{
    ::xilinx::xllvm::createAIE4ConfIntrinsicCall(builder, moduleTranslation,
        &opInst, "llvm.aie4.insert.staging.int.I1024",
        static_cast<uint32_t>(op.getMode()));
  }];
}

// llvm.aie4.insert.staging.fp8.I1024
//     void (<128 x i8> src, i32 index)
def InsertStagingFp8I1024AIE4IntrOp
    : AIEVec4_IntrOp<"insert.staging.fp8.I1024", [AIE4StagingWriter],
                     /*numResults=*/0>,
      Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [128], [I8]>:$src,
          I32:$index)> {
  let summary =
      "AIE4 insert a 1024-bit fp8 chunk into matrix staging (vmov f_za[i], y)";
  let description = [{
    Writes 128 fp8 (E4M3) values, carried as bytes, to chunk `index`
    (0..31) of the fp8 staging matrix `f_za`. crIntMMode is not used.
  }];
  let llvmBuilder = [{
    ::xilinx::xllvm::createExternalLLVMIntrinsicCall(builder,
        moduleTranslation, &opInst, "llvm.aie4.insert.staging.fp8.I1024");
  }];
}

// llvm.aie4.staging.to.matrix.{I32768,I4.I32768,FP32768}
//     void ()
//
// Commit staging with I32768 for int8/bf16, I4.I32768 for int4, or
// FP32768 for fp8.
class AIE4StagingToMatrixIntrOp<string mnemonic, string asm>
    : AIEVec4_IntrOp<mnemonic, [AIE4MatrixCommit],
                     /*numResults=*/0> {
  let summary = "AIE4 commit staging to the active matrix ("#asm#")";
  let description = [{
    Copies the whole staging matrix to the active matrix read by the MMACs.
    A dense tile writes all 32 staging chunks before the commit.
  }];
  let llvmBuilder = [{
    ::xilinx::xllvm::createExternalLLVMIntrinsicCall(builder,
        moduleTranslation, &opInst, "llvm.aie4.}]#mnemonic#[{");
  }];
}

def StagingToMatrixI32768AIE4IntrOp
    : AIE4StagingToMatrixIntrOp<"staging.to.matrix.I32768", "vmov zb, za">;
def StagingToMatrixI4I32768AIE4IntrOp
    : AIE4StagingToMatrixIntrOp<"staging.to.matrix.I4.I32768",
                                "vmov i4_zb, i4_za">;
def StagingToMatrixFP32768AIE4IntrOp
    : AIE4StagingToMatrixIntrOp<"staging.to.matrix.FP32768", "vmov f_zb, f_za">;

#endif // AIE_DIALECT_XLLVM_IR_XLLVMAIE4INTROPS_TD
