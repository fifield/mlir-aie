//===- AIEVecOps.td - AIE vector op definitions -----------*- tablegen -*-====//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines AIE vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_OPS
#define AIEVEC_OPS

include "aie/Dialect/AIE/IR/AIEAttrs.td"
include "aie/Dialect/AIEVec/IR/AIEVecAttributes.td"
include "aie/Dialect/AIEVec/IR/AIEVecTypes.td"
include "aie/Dialect/AIEVec/IR/AIEVecTypeConstraints.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for AIE dialect ops.
class AIEVec_Op<string mnemonic, list<Trait> traits = []>
    : Op<AIEVec_Dialect, mnemonic, traits> {
  // For every AIE vector op, there needs to be a:
  //   * void ${C++ class of Op}::print(OpAsmPrinter &p)
  //   * LogicalResult ${C++ class of Op}::verify()
  //   * ParseResult ${C++ class of Op}::parse(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AIEVec_AddElemOp
    : AIEVec_Op<"add_elem", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector add elem";
  let description = [{
    AMD-specific AIE2 intrinsic that allows you to perform addition operation
    on all types of vectors.`$result = `$lhs + $rhs`.
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}


// AIE4-specific. Deliberately a separate op from `aievec.mul_elem`, which
// carries the AIE2-only `isOperandResultTypePairValidForAIE2MulElem` type
// constraint (its legal lhs/rhs/acc combinations are AIE2 accumulator shapes,
// none of which describe AIE4's <32xf32> -> <32xf32> form).
//
// Unlike `aievec.mul_elem`, this op is NOT marked `Pure`. It lowers to
// `llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf`, whose intrinsic
// definition carries no `IntrNoMem`, so LLVM itself treats it as potentially
// side-effecting. Modelling it as pure here would let MLIR hoist/CSE it more
// aggressively than the backend will.
def AIEVec_AIE4MulEwiseF32Op:
  AIEVec_Op<"aie4.mul_ewise_f32", [
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$result)> {
  let summary = "AIE4 elementwise f32 vector multiply";
  let description = [{
    AIE4-specific elementwise multiply of two 1024-bit (`vector<32xf32>`)
    vectors: `$result = $lhs * $rhs`.

    Same type in and out: AIE4's 1024-bit f32 accumulator has the same shape as
    the input vector, so this needs no UPS/SRS pair and no shift/sign
    bookkeeping. Lowers to a single `vmul.f` instruction.
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 control-register side-effect resources
//===----------------------------------------------------------------------===//
//
// Share the MAC resource identity with XLLVM through AIE4CtrlRegResources.h.
def AIEVec_AIE4CRVecSatRes :
    Resource<"::xilinx::xllvm::AIE4CRVecSatResource">;

//===----------------------------------------------------------------------===//
// AIE4 integer MAC
//===----------------------------------------------------------------------===//

def AIEVec_AIE4MacEwiseI16I32Op:
  AIEVec_Op<"aie4.mac_ewise_i16_i32", [
    AllTypesMatch<["lhs", "rhs"]>,
    AllTypesMatch<["acc", "result"]>,
    MemoryEffects<[MemRead<AIEVec_AIE4CRVecSatRes>]>
  ]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$lhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$rhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$acc)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$result)> {
  let summary = "AIE4 elementwise integer multiply-accumulate";
  let description = [{
    Elementwise signed i16 x i16 multiply-accumulate. The multiplier widens
    each product to i32 and adds it to the corresponding accumulator lane.
    Inputs are `vector<32xi16>` (512 bits); the accumulator and result are
    `vector<32xi32>` (1024 bits).

    Lowers to `vmac`, which reads the vector saturation control `crVecSat`.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 f32 MAC
//===----------------------------------------------------------------------===//

def AIEVec_AIE4MacEwiseF32Op:
  AIEVec_Op<"aie4.mac_ewise_f32", [
    AllTypesMatch<["lhs", "rhs", "acc", "result"]>
  ]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$result)> {
  let summary = "AIE4 elementwise f32 vector multiply-accumulate";
  let description = [{
    Computes `$acc + ($lhs * $rhs)` with one rounding, using fixed
    `vector<32xf32>` operands and result. Lowers to `vmac.f`.

    Fusion of `arith.mulf` and `arith.addf` requires `contract` (or `fast`)
    on both operations because it can change rounding. Otherwise, the
    multiply lowers to `aievec.aie4.mul_ewise_f32` and the add stays separate.

    Reads `crFPVMask` and `crVecRnd` and updates `srFPVFlags`; not `Pure`.
    The VMAC unit is IEEE-compliant with two exceptions (AIE4 ISA v1.1
    section 18.4): subnormal inputs are clamped to a signed zero for every
    float type but fp8 and fp16, and subnormal results return zero. A
    consequence worth knowing is that subnormal times infinity becomes zero
    times infinity, which is NaN. `contract` permits contraction; it does
    not authorize that flush-to-zero behavior, and this op does not
    establish `crVecRnd` -- callers must arrange the rounding mode they
    need. This is the same contract as the native MUL/MAC forms.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `,` $acc attr-dict `:` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 bf16 x bf16 -> f32 MUL/MAC
//===----------------------------------------------------------------------===//

def AIEVec_AIE4MulEwiseBf16F32Op:
  AIEVec_Op<"aie4.mul_ewise_bf16_f32", [
    AllTypesMatch<["lhs", "rhs"]>
  ]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$result)> {
  let summary = "AIE4 elementwise bf16 x bf16 -> f32 vector multiply";
  let description = [{
    Multiplies fixed `vector<32xbf16>` operands and rounds the product to
    `vector<32xf32>`. Lowers to `vmul.f` with no intermediate bf16 rounding.

    Replaces `extf` + `extf` + `mulf` without contraction permission:
    bf16-to-f32 widening is exact, and the product is still rounded to f32.

    Reads `crFPVMask` and `crVecRnd` and updates `srFPVFlags`; not `Pure`.
    The VMAC unit is IEEE-compliant with two exceptions (AIE4 ISA v1.1
    section 18.4): subnormal inputs are clamped to a signed zero for every
    float type but fp8 and fp16, and subnormal results return zero. A
    consequence worth knowing is that subnormal times infinity becomes zero
    times infinity, which is NaN. `contract` permits contraction; it does
    not authorize that flush-to-zero behavior, and this op does not
    establish `crVecRnd` -- callers must arrange the rounding mode they
    need. This is the same contract as the native MUL/MAC forms.
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_AIE4MacEwiseBf16F32Op:
  AIEVec_Op<"aie4.mac_ewise_bf16_f32", [
    AllTypesMatch<["lhs", "rhs"]>,
    AllTypesMatch<["acc", "result"]>
  ]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$result)> {
  let summary = "AIE4 elementwise bf16 x bf16 -> f32 vector multiply-accumulate";
  let description = [{
    Computes `$acc + ($lhs * $rhs)` with one rounding. Inputs are fixed
    `vector<32xbf16>`; the accumulator and result are `vector<32xf32>`.
    Lowers to `vmac.f`.

    Fusion of a widened bf16 product with `arith.addf` requires `contract`
    (or `fast`) on both the multiply and add, as for `aie4.mac_ewise_f32`.

    Reads `crFPVMask` and `crVecRnd` and updates `srFPVFlags`; not `Pure`.
    The VMAC unit is IEEE-compliant with two exceptions (AIE4 ISA v1.1
    section 18.4): subnormal inputs are clamped to a signed zero for every
    float type but fp8 and fp16, and subnormal results return zero. A
    consequence worth knowing is that subnormal times infinity becomes zero
    times infinity, which is NaN. `contract` permits contraction; it does
    not authorize that flush-to-zero behavior, and this op does not
    establish `crVecRnd` -- callers must arrange the rounding mode they
    need. This is the same contract as the native MUL/MAC forms.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_AIE4MscEwiseF32Op:
  AIEVec_Op<"aie4.msc_ewise_f32", [
    AllTypesMatch<["lhs", "rhs", "acc", "result"]>
  ]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$lhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$rhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$result)> {
  let summary = "AIE4 elementwise f32 vector multiply-subtract";
  let description = [{
    Computes `$acc - ($lhs * $rhs)` with one rounding, using fixed
    `vector<32xf32>` operands and result. Lowers to `vmsc.f`.

    Fusion of `arith.mulf` and `arith.subf` requires `contract` (or `fast`)
    on both operations because it can change rounding. Subtraction does not
    commute, so only a product on the right-hand side fuses: `%acc - %a * %b`
    is a multiply-subtract, while `%a * %b - %acc` is not.

    Reads `crFPVMask` and `crVecRnd` and updates `srFPVFlags`; not `Pure`.
    The VMAC unit is IEEE-compliant with two exceptions (AIE4 ISA v1.1
    section 18.4): subnormal inputs are clamped to a signed zero for every
    float type but fp8 and fp16, and subnormal results return zero. A
    consequence worth knowing is that subnormal times infinity becomes zero
    times infinity, which is NaN. `contract` permits contraction; it does
    not authorize that flush-to-zero behavior, and this op does not
    establish `crVecRnd` -- callers must arrange the rounding mode they
    need. This is the same contract as the native MUL/MAC forms.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `,` $acc attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_AIE4MscEwiseBf16F32Op:
  AIEVec_Op<"aie4.msc_ewise_bf16_f32", [
    AllTypesMatch<["lhs", "rhs"]>,
    AllTypesMatch<["acc", "result"]>
  ]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$lhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$rhs,
                 FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$acc)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$result)> {
  let summary = "AIE4 elementwise bf16 x bf16 -> f32 vector multiply-subtract";
  let description = [{
    Computes `$acc - ($lhs * $rhs)` with one rounding. Inputs are fixed
    `vector<32xbf16>`; the accumulator and result are `vector<32xf32>`.
    Lowers to `vmsc.f`.

    Fusion of a widened bf16 product with `arith.subf` requires `contract`
    (or `fast`) on both the multiply and subtract, and only fuses a product
    on the right-hand side, as for `aie4.msc_ewise_f32`.

    Reads `crFPVMask` and `crVecRnd` and updates `srFPVFlags`; not `Pure`.
    The VMAC unit is IEEE-compliant with two exceptions (AIE4 ISA v1.1
    section 18.4): subnormal inputs are clamped to a signed zero for every
    float type but fp8 and fp16, and subnormal results return zero. A
    consequence worth knowing is that subnormal times infinity becomes zero
    times infinity, which is NaN. `contract` permits contraction; it does
    not authorize that flush-to-zero behavior, and this op does not
    establish `crVecRnd` -- callers must arrange the rounding mode they
    need. This is the same contract as the native MUL/MAC forms.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 SRS
//===----------------------------------------------------------------------===//

def AIEVec_AIE4SRSI32I16Op:
  AIEVec_Op<"aie4.srs_i32_i16", []>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$source,
                 ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<31>]>:$shift,
                 DefaultValuedAttr<BoolAttr, "true">:$isSigned)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$result)> {
  let summary = "AIE4 shift-round-saturate: i32 accumulator -> i16 vector";
  let description = [{
    Narrows a `vector<32xi32>` accumulator to `vector<32xi16>` using the AIE4
    SRS instruction. `$shift` is in 0..31; `$isSigned` selects the signed or
    unsigned instruction form.

    Rounding and saturation depend on the current `crRnd` and `crSat` modes.
    The backend sets the accumulator lane width and sign, but this op does
    not establish or restore rounding and saturation modes. Callers must
    arrange the modes required by their computation.

    Automatic `arith.shrsi`/`arith.trunci` fusion is disabled until a
    mode-preserving lowering can establish floor rounding and wrapping, then
    restore the caller's modes. Peano does not support the generic
    `vector<32xi32>` right shift, so it may fail backend legalization.

    The op reads the mode registers and writes the SRS overflow status flag;
    it must not be treated as pure.
  }];
  let assemblyFormat =
      "$source attr-dict `:` type($source) `,` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 UPS
//===----------------------------------------------------------------------===//

def AIEVec_AIE4UPSI16I32Op:
  AIEVec_Op<"aie4.ups_i16_i32", []>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$source,
                 ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<31>]>:$shift,
                 DefaultValuedAttr<BoolAttr, "true">:$isSigned)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>:$result)> {
  let summary = "AIE4 up-shift: i16 vector -> i32 accumulator";
  let description = [{
    Sign- or zero-extends `vector<32xi16>` to `vector<32xi32>`, then
    left-shifts by `$shift`. Reads `crSat`, `crUPSMode`, and the selected
    `upsSign0`/`upsSign1` field, and updates `srUPS_of`.

    Shift-zero widening is exact under the ABI signedness modes and is used
    for `arith.extsi`/`arith.extui`. Nonzero shifts can overflow; wrapping or
    saturation depends on the ambient modes. Lowers to `vups.2x` or the
    load-fused `vlda.ups.2x`.
  }];
  let assemblyFormat =
      "$source attr-dict `:` type($source) `,` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 i8 -> i16 unpack
//===----------------------------------------------------------------------===//

def AIEVec_AIE4UnpackI8I16Op:
  AIEVec_Op<"aie4.unpack_i8_i16", [Pure]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [I8]>:$source,
                 DefaultValuedAttr<BoolAttr, "true">:$isSigned)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>:$result)> {
  let summary = "AIE4 widen i8 vector -> i16 vector, exact";
  let description = [{
    Sign-extends `vector<32xi8>` to `vector<32xi16>` when `isSigned` is true
    (the default), and zero-extends otherwise. The result is exact and
    independent of caller-controlled mode state; the backend sets the unpack
    size. Lowers to `vunpack` or the load-fused `vldb.unpack.2x`.
  }];
  let assemblyFormat =
      "$source attr-dict `:` type($source) `,` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 bf16 <-> f32 accumulator conversion
//===----------------------------------------------------------------------===//
def AIEVec_AIE4ConvBF16F32Op:
  AIEVec_Op<"aie4.conv_bf16_f32", [Pure]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$source)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$result)> {
  let summary = "AIE4 widen bf16 vector -> f32 accumulator, exact";
  let description = [{
    Exactly widens `vector<32xbf16>` to `vector<32xf32>` without reading or
    updating control registers. Lowers to `vconv.fp32.bf16` or the
    load-fused `vlda.conv.fp32.bf16`.
  }];
  let assemblyFormat =
      "$source attr-dict `:` type($source) `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_AIE4ConvF32BF16Op:
  AIEVec_Op<"aie4.conv_f32_bf16", []>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$source)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$result)> {
  let summary = "AIE4 narrow f32 accumulator -> bf16 vector, ambient rounding";
  let description = [{
    Narrows `vector<32xf32>` to `vector<32xbf16>` using the ambient rounding
    and saturation modes. Reads `crRnd`, `crFPConvSat`, and `crF2FMask`, and
    updates `srF2FFlags`. With `crRnd == 0`, rounding is toward negative
    infinity. The operation does not establish or restore these modes and
    does not implement the rounding contract of `arith.truncf`.

    Lowers to `vconv.bf16.fp32` or the store-fused `vst.conv.bf16.fp32`.
  }];
  let assemblyFormat =
      "$source attr-dict `:` type($source) `,` type($result)";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// AIE4 integer min/max/compare/select
//===----------------------------------------------------------------------===//
//
// Integer min/max and compare use the ABI convention that vaddSign0 selects
// unsigned comparison and vaddSign1 selects signed comparison. These ops model
// fixed signedness and rely on those control bits retaining their ABI values.
// The supported shapes use at most 64 mask bits; vector<128xi8> is unsupported.
def AIEVec_AIE4MinMaxOp:
  AIEVec_Op<"aie4.minmax", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$lhs,
                 AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$rhs,
                 AIEVec_AIE4MinMaxKindAttr:$kind,
                 DefaultValuedAttr<BoolAttr, "true">:$isSigned)>,
  Results<(outs AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$result)> {
  let summary = "AIE4 native signed/unsigned vector min/max";
  let description = [{
    Computes per-lane minimum or maximum, selected by `$kind`, with signed
    or unsigned comparison selected by `$isSigned`. Lowers to `vmin_ge` or
    `vmax_lt`.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `` `[` `kind` `=` $kind `,` `signed` `=` $isSigned `]` attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_AIE4CmpOp:
  AIEVec_Op<"aie4.cmp", [
    Pure,
    AllTypesMatch<["lhs", "rhs"]>
  ]>,
  Arguments<(ins AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$lhs,
                 AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$rhs,
                 AIEVec_AIE4CmpPredicateAttr:$predicate,
                 DefaultValuedAttr<BoolAttr, "true">:$isSigned)>,
  Results<(outs AnyUnsignedInteger:$result)> {
  let summary = "AIE4 native vector compare, mask result";
  let description = [{
    Compares each lane using `lhs < rhs` (`lt`) or `lhs >= rhs` (`ge`),
    with signedness selected by `$isSigned`. Greater-than and less-or-equal
    comparisons use swapped operands. Equality predicates use generic lowering.

    Result bit i contains the comparison for lane i. The mask type is `ui64`
    for `vector<64xi8>` and `vector<64xi16>`, and `ui32` for `vector<32xi16>`,
    `vector<16xi32>`, and `vector<32xi32>`. For `vector<16xi32>`, only the low
    16 mask bits correspond to lanes. LLVM lowering represents 64-bit masks
    as `vector<2xi32>` register pairs.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `` `[` `pred` `=` $predicate `,` `signed` `=` $isSigned `]` attr-dict `:` type($lhs) `,` type($result)";
  let hasVerifier = 1;
}

def AIEVec_AIE4SelOp:
  AIEVec_Op<"aie4.sel", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$lhs,
                 AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$rhs,
                 AnyUnsignedInteger:$sel)>,
  Results<(outs AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I8]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [I16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>]>:$result)> {
  let summary = "AIE4 native lane-wise select with a packed mask";
  let description = [{
    Selects `$rhs[i]` when bit i of `$sel` is set, and `$lhs[i]` otherwise.
    The mask type follows the same vector-shape rules as `aievec.aie4.cmp`;
    bits above the lane count are ignored. Lowers to `vsel`.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `,` $sel attr-dict `:` type($lhs) `,` type($sel) `,` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE4 float min/max/compare
//===----------------------------------------------------------------------===//
//
// Native operations read ambient crFPOrder. Source IEEE arithmetic compares
// integer encodings to remain independent of the mode.
def AIEVec_AIE4CRFPOrderRes :
    Resource<"::xilinx::xllvm::AIE4CRFPOrderResource">;

def AIEVec_AIE4FMinMaxOp:
  AIEVec_Op<"aie4.fminmax", [
    MemoryEffects<[MemRead<AIEVec_AIE4CRFPOrderRes>]>,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [F32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>]>:$lhs,
                 AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [F32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>]>:$rhs,
                 AIEVec_AIE4MinMaxKindAttr:$kind)>,
  Results<(outs AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [F32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>]>:$result)> {
  let summary = "AIE4 native float vector min/max using ambient crFPOrder";
  let description = [{
    Computes per-lane native minimum or maximum. At `crFPOrder = 0`, NaNs
    propagate and -0 orders before +0. Other modes use the hardware ordering
    selected by the ambient register. This operation does not change the mode.

    Lowering uses a helper call to preserve ordering across mode changes until
    Peano models the intrinsic's FPOrder dependency. This adds call overhead.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `` `[` `kind` `=` $kind `]` attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_AIE4FCmpOp:
  AIEVec_Op<"aie4.fcmp", [
    MemoryEffects<[MemRead<AIEVec_AIE4CRFPOrderRes>]>,
    AllTypesMatch<["lhs", "rhs"]>
  ]>,
  Arguments<(ins AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [F32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>]>:$lhs,
                 AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [BF16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [64], [F16]>,
                   FixedVectorOfRankAndLengthAndType<[1], [16], [F32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>]>:$rhs,
                 AIEVec_AIE4CmpPredicateAttr:$predicate)>,
  Results<(outs AnyUnsignedInteger:$result)> {
  let summary = "AIE4 native float vector compare using ambient crFPOrder";
  let description = [{
    Computes per-lane native `lhs < rhs` (`lt`) or `lhs >= rhs` (`ge`). At
    `crFPOrder = 0`, both predicates are false for NaN operands, and signed
    zeros compare equal. Other modes use the hardware ordering selected by
    the ambient register. This operation does not change the mode.

    The unsigned result packs lane i into bit i: ui32 for 16 or 32 lanes,
    ui64 for 64 lanes. Only the low 16 bits are meaningful for 16 lanes.

    Lowering uses a helper call to preserve ordering across mode changes until
    Peano models the intrinsic's FPOrder dependency. This adds call overhead.
  }];
  let assemblyFormat =
      "$lhs `,` $rhs `` `[` `pred` `=` $predicate `]` attr-dict `:` type($lhs) `,` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE4 1024-bit vector reductions
//===----------------------------------------------------------------------===//
def AIEVec_AIE4ReduceOp:
  AIEVec_Op<"aie4.reduce", [Pure]>,
  Arguments<(ins AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [32], [I32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>]>:$source,
                 Optional<AnyTypeOf<[I32, F32]>>:$seed,
                 AIEVec_AIE4ReduceKindAttr:$kind)>,
  Results<(outs AnyTypeOf<[I32, F32]>:$result)> {
  let summary = "AIE4 native 1024-bit vector reduction (vminr/vmaxr/vaddr[.f])";
  let description = [{
    Reduces a fixed 32-lane i32 or f32 vector. An optional scalar seed
    participates as an additional value; its type and the result type match
    the vector element type. Integer min/max are signed, and floating min/max
    propagate NaNs.

    Native integer `add` saturates at the i32 limits; standard
    `vector.reduction<add>` wraps. Native f32 `add` accumulates in a wider
    adder and rounds once, so it may differ from an ordered f32 reduction.
  }];
  let assemblyFormat =
      "$kind $source (`,` $seed^)? attr-dict `:` type($source) (`,` type($seed)^)? `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE4 shuffle
//===----------------------------------------------------------------------===//
//
// VSHUFFLE takes 512-bit operands. Two-operand modes transpose their
// logical 1024-bit concatenation and return one 512-bit half.
def AIEVec_AIE4ShuffleVector : AllOfType<[
    FixedVectorOfRank<[1]>,
    VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16, F32]>],
    "fixed one-dimensional 512-bit vector of i8, i16, i32, bf16 or f32">;

def AIEVec_AIE4ShuffleOp:
  AIEVec_Op<"aie4.shuffle", [
    Pure,
    AllTypesMatch<["lhs", "result"]>,
    OptionalTypesMatchWith<"result and rhs have the same type", "result",
                           "rhs", "::llvm::cast<Type>($_self)">
  ]>,
  Arguments<(ins AIEVec_AIE4ShuffleVector:$lhs,
                 Optional<AIEVec_AIE4ShuffleVector>:$rhs,
                 AIEVec_AIE4ShuffleModeAttr:$mode)>,
  Results<(outs AIEVec_AIE4ShuffleVector:$result)> {
  let summary = "AIE4 512-bit vector shuffle/transpose (vshuffle)";
  let description = [{
    Shuffles fixed, one-dimensional 512-bit vectors using `llvm.aie4.vshuffle`.
    Mode encodings are specific to AIE4.

    `t32_4x4` and `t8_8x8` transpose a row-major square matrix in `lhs`
    and forbid `rhs`. `t32_2x16_lo` and `t32_2x16_hi` require `rhs` and interleave
    lanes 0-7 or 8-15, respectively, with `lhs` supplying the even lanes.
    All operands and the result must have the same type, and the element
    width must match the mode's `T<w>_` prefix.

    The operation is pure: the intrinsic has no memory effects, and the
    instruction neither reads nor writes status or control registers.
  }];
  let assemblyFormat =
      "$lhs (`,` $rhs^)? `` `[` `mode` `=` $mode `]` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE4 approximate transcendentals (VEXP2, VTANH)
//===----------------------------------------------------------------------===//
//
// Vector-to-AIEVec creates these ops only from exp2/tanh with fastmath<afn>.
// Pure: these instructions raise no FP exception flags and read no
// control registers.

def AIEVec_AIE4BiasedExp2Op:
  AIEVec_Op<"aie4.biased_exp2", [Pure]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$source)>,
  Results<(outs AnyTypeOf<[
                   FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>,
                   FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>]>:$result)> {
  let summary = "AIE4 approximate biased base-2 exponential (vexp2)";
  let description = [{
    Approximates `2^t` per lane, where `t = floor((x - 895) * 2^11) / 2^11`.
    Callers computing `2^x` add 895 + 2^-12 for an f32 result or the
    `aie4_nlf.h` bias 895.001 for a bf16 result.

    The result type selects the instruction form:
    - `vector<32xbf16>`: VEXP2.bf16. Faithful rounding selects a bf16
      neighbor bracketing `2^t`; it need not select the nearest one.
    - `vector<32xf32>`: VEXP2.fp32 (`llvm.aie4.exp2.bf20`), an f32 with the
      11 least significant bits cleared, error <= 1 fp16 ULP.

    NaN inputs produce a fixed NaN. `t >= 128` produces +Inf;
    `t <= -134` (bf16) or `t < -138` (f32) produces 0.
  }];
  let assemblyFormat =
      "$source attr-dict `:` type($source) `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_AIE4TanhOp:
  AIEVec_Op<"aie4.tanh", [Pure]>,
  Arguments<(ins FixedVectorOfRankAndLengthAndType<[1], [32], [F32]>:$source)>,
  Results<(outs FixedVectorOfRankAndLengthAndType<[1], [32], [BF16]>:$result)> {
  let summary = "AIE4 approximate hyperbolic tangent (vtanh)";
  let description = [{
    Truncates each f32 lane to bf16, then approximates `tanh` with relative
    error below 8.3%, measured after truncation. +/-Inf map
    to +/-1. NaN maps to a sign-dependent NaN: 0x7F81 for a positive-sign
    input, 0xFF81 for a negative-sign input.
  }];
  let assemblyFormat =
      "$source attr-dict `:` type($source) `,` type($result)";
  let hasVerifier = 0;
}

// Not Pure: INV writes the srFPNlf exception status flags.
def AIEVec_AIE4InvOp:
  AIEVec_Op<"aie4.inv", []>,
  Arguments<(ins F32:$source)>,
  Results<(outs F32:$result)> {
  let summary = "AIE4 approximate scalar f32 reciprocal (inv)";
  let description = [{
    Approximates `1.0 / x` with the scalar INV instruction. For normal inputs
    and results, the error is at most 1 ULP. Denormal inputs are treated as
    zero and denormal results are flushed to zero. Vector-to-AIEVec creates
    this op only from `arith.divf` carrying `fastmath<afn>` with a constant
    1.0 numerator.
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_SubElemOp:
  AIEVec_Op<"sub_elem", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs)>,
  Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector sub elem";
  let description = [{
    AMD-specific AIE2 intrinsic that allows you to perform substraction operation
    on all types of vectors.`$result = `$lhs - $rhs`.
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_FMAElemOp
    : AIEVec_Op<"mac_elem", [Pure]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs,
          AnyVectorOfNonZeroRank:$acc,
          DefaultValuedAttr<BoolAttr, "false">:$fmsub)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 element-wise vector fused multiply-add";
  let description = [{
    AMD-specific multiply-add operation. It multiplies two 1-D vectors in the same channel,
    and adds the result to an accumulator.
    `$result = `$lhs * $rhs + $acc`.
    Note: the same operator can be used as fmsub operator by setting the
    'fmsub' bool to true.
  }];
  let builders = [OpBuilder<
      (ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Value":$acc,
          "bool":$fmsub),
      [{build($_builder, $_state, acc.getType(), lhs, rhs, acc,
            fmsub);}]>];
  let extraClassDeclaration = [{
    // Get the attribute names
    llvm::StringRef getSubAttrName() { return "fmsub"; }
  }];
}

def AIEVec_MulElemOp
    : AIEVec_Op<"mul_elem", [Pure, SameTypeOperands, SameOperandsShape,
                             SameOperandsAndResultShape,
                             isOperandResultTypePairValidForAIE2MulElem<
                                 "lhs", "rhs", "result">]>,
      Arguments<(ins VectorOfLengthAndType<[16, 32, 64], [I8, I16, I32, BF16,
                                                          F32]>:$lhs,
          VectorOfLengthAndType<[16, 32, 64], [I8, I16, I32, BF16, F32]>:$rhs)>,
      Results<(
          outs VectorOfLengthAndType<[16, 32, 64], [I32, I64, F32]>:$result)> {
  let summary = "AIE2 vector element-wise multiply";
  let description = [{
    AMD-specific multiply operation that multiplies two 1-D vectors in the same channel.
    The vector sizes are at least 512 bits.
    `$result = `$lhs * $rhs`.
    Currently, the following are the supported type combinations:
        lhs                | rhs                | Accumulator
      :------------------:|:------------------:|:-----------------:
        `vector<32xi8>`    | `vector<32xi8>`    | `vector<32xi32>`
        `vector<32xi16>`   | `vector<32xi16>`   | `vector<32xi32>`
        `vector<32xbf16>`  | `vector<32xbf16>`  | `vector<32xf32>` (AIE2P)
        `vector<64xbf16>`  | `vector<64xbf16>`  | `vector<64xf32>` (AIE2P)
        `vector<16xi32>`   | `vector<16xi32>`   | `vector<16xi64>`
        `vector<16xbf16>`  | `vector<16xbf16>`  | `vector<16xf32>`
        `vector<16xf32>`   | `vector<16xf32>`   | `vector<16xf32>`'
  }];
}

def AIEVec_BroadcastOp
    : AIEVec_Op<"broadcast", [Pure]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source,
          DefaultValuedAttr<ConfinedAttr<AIEI8Attr, [IntNonNegative]>,
                            "0">:$idx)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 broadcast";
  let description = [{
    AMD-specific broadcast intrinsic. Extract element index from vector and broadcasts its
    value to all lanes of the vector.
    `$result = broadcast($source, $idx)`
  }];
  let builders = [OpBuilder<
      (ins "mlir::Value":$source, "int32_t":$idx),
      [{build($_builder, $_state, source.getType(), source, idx);}]>];
}

def AIEVec_BroadcastScalarOp
    : AIEVec_Op<"broadcast_scalar", [Pure]>,
      Arguments<(ins AnyTypeOf<[BF16, F32, I32, I16, I8]>:$source)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 broadcast scalar";
  let description = [{
    AMD-specific broadcast scalar intrinsic. Broadcasts input value to all vector lanes.
    `$result = broadcast_scalar($source)`
  }];
  let builders = [OpBuilder<
      (ins "mlir::Value":$source),
      [{build($_builder, $_state, source.getType(), source);}]>];
}

def AIEVec_UPSOp
    : AIEVec_Op<"ups", [Pure]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source,
          DefaultValuedAttr<ConfinedAttr<AIEI8Attr, [IntNonNegative]>,
                            "0">:$shift)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE ups";
  let description = [{
    AMD-specific upshift intrinsic. Moves data from AIE vector data type
    to accumulator data type. The adjustment in precision is controlled by
    the shift parameter.
    `$result = ups($source, $shift)`
  }];
  let builders = [OpBuilder<
      (ins "mlir::Value":$source, "int8_t":$shift),
      [{build($_builder, $_state, source.getType(), source, shift);}]>];
  let hasFolder = 1;
}

def AIEVec_CastOp : AIEVec_Op<"cast", [Pure]>,
                    Arguments<(ins AnyVectorOfNonZeroRank:$source,
                        DefaultValuedAttr<BoolAttr, "false">:$isResAcc)>,
                    Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE cast";
  let description = [{
    AIE2 cast intrinsic. Cast values from source data type to result data types.
    `$result = cast($source, isResAcc)`
  }];
  let builders = [OpBuilder<
      (ins "mlir::Value":$source, "bool":$isResAcc),
      [{build($_builder, $_state, source.getType(), source, isResAcc);}]>];
  let hasFolder = 1;
}

def AIEVec_SRSOp
    : AIEVec_Op<"srs", [Pure]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source, AnyInteger:$shift,
          DefaultValuedAttr<
                     ConfinedAttr<AIEI32Attr, [IntMinValue<0>, IntMaxValue<1>]>,
                     "1">:$sign)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE srs";
  let description = [{
    AMD-specific shift-round-saturate intrinsic. Moves values from
    accumulator data type to AIE vector data types. The adjustment in
    precision is controlled by the shift parameter. The sign parameter
    controls signed (1) vs unsigned (0) saturation.
    `$result = srs($source, $shift, $sign)`
  }];
  let hasFolder = 1;
}

def AIEVec_UPDOp
    : AIEVec_Op<"upd", [Pure, AttrSizedOperandSegments]>,
      Arguments<(ins AnyShaped:$source, Variadic<Index>:$indices,
          DefaultValuedAttr<AIEI32Attr, "0">:$offset,
          DefaultValuedAttr<
                     ConfinedAttr<AIEI8Attr, [IntMinValue<0>, IntMaxValue<1>]>,
                     "0">:$index,
          Optional<AnyVectorOfNonZeroRank>:$vector)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE upd";
  let description = [{
    AMD-specific update intrinsic. General upd intrinsic updates contiguous
    lanes of the result vector from a smaller source vector. This form of
    upd intrinsic combines the load of data from memory into a vector
    register, and then updating the lanes of the result vector using it.
    `$result = upd($source[$indices], $offset, $index)`
  }];
  let builders = [OpBuilder<
      (ins "mlir::Type":$resultType, "mlir::Value":$source,
          "mlir::ValueRange":$indices, "int32_t":$offset, "int8_t":$index),
      [{build($_builder, $_state, resultType, source, indices,
                   offset, index, nullptr);}]>];
}

def AIEVec_ConcatOp
    : AIEVec_Op<"concat", [Pure, InferTypeOpAdaptor, ]>,
      Arguments<(ins Variadic<AnyVectorOfNonZeroRank>:$sources)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE concat";
  let description = [{
    AMD-specific concat intrinsic. Concatenates two or more smaller
    vectors into a bigger vector. The verifier confirms that all the
    input vectors have the same number of lanes.
    `$result = concat($sources[0], $sources[1], ...)`
  }];
}

def AIEVec_ExtOp : AIEVec_Op<"ext", [Pure]>,
                   Arguments<(ins AnyVectorOfNonZeroRank:$source,
                       ConfinedAttr<AIEI8Attr, [IntNonNegative]>:$index)>,
                   Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE ext";
  let description = [{
    AMD-specific vector extract intrinsic. Selects contiguous lanes from
    the source vector, and transfers the data from those lanes to the
    result. The source divides into equal parts as wide as the result, and
    index selects one of them: index 0 extracts the lowest part, index 1 the
    next one, and so on. The verifier requires index to be less than the
    number of parts.
    `$result = ext($source, $index)`
  }];
}

def AIEVec_PackOp : AIEVec_Op<"pack", [Pure]>,
                    Arguments<(ins AnyVectorOfNonZeroRank:$source)>,
                    Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE pack";
  let description = [{
    AMD-specific pack intrinsic. Pack a vector of 16-bit values into
    a vector of 8-bit values.
    `$result = pack($source)`
  }];
}

def AIEVec_UnpackOp : AIEVec_Op<"unpack", [Pure]>,
                      Arguments<(ins AnyVectorOfNonZeroRank:$source)>,
                      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE unpack";
  let description = [{
    AMD-specific unpack intrinsic. Unpack a vector of 8-bit values into
    a vector of 16-bit values.
    `$result = unpack($source)`
  }];
}

def AIEVec_ShiftOp
    : AIEVec_Op<"shift", [Pure]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs,
          I32:$shift, DefaultValuedAttr<BoolAttr, "false">:$isAcc)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 concat and shift";
  let description = [{
    AMD-specific shift intrinsic. Concatenates two
    vectors into a bigger vector, interprets them as a vector of 128 bytes
    and returns v1::v2[shift: shift+64]. `shift` is the number of bytes to
    be shifted. The verifier confirms that all the input and result vectors
    have the same number of lanes and element types.
    `$result = shift($lhs, $rhs, $shift)`
  }];
}

def AIEVec_LegacyShuffleOp : AIEVec_Op<"legacyshuffle", [Pure]>,
                             Arguments<(ins AnyVectorOfNonZeroRank:$source,
                                 DefaultValuedAttr<AIEI32Attr, "0">:$mode)>,
                             Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 shuffle";
  let description = [{
    AMD-specific vector shuffle intrinsic by a specific shuffle mode.
    `$result = shuffle($source, $mode)`
  }];
}

def AIEVec_MulConvOp
    : AIEVec_Op<"mul_conv", [Pure]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs,
          DefaultValuedAttr<AIEI32Attr, "0">:$M,
          DefaultValuedAttr<AIEI32Attr, "0">:$N)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 multiply convolution";
  let description = [{
    AMD-specific multiply convolution intrinsic. Multiply convolution operation
    of (M x N)matrix with (N x 1)kernel.
    `$result = mul_convMxN($lhs, $rhs)`
  }];
  let builders = [OpBuilder<
      (ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Type":$accType,
          "int32_t":$M, "int32_t":$N),
      [{build($_builder, $_state, accType, lhs, rhs, M, N);}]>];
}

def AIEVec_FMAConvOp
    : AIEVec_Op<"fma_conv", [Pure]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs,
          AnyVectorOfNonZeroRank:$acc, DefaultValuedAttr<AIEI32Attr, "0">:$M,
          DefaultValuedAttr<AIEI32Attr, "0">:$N,
          DefaultValuedAttr<BoolAttr, "false">:$fmsub)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 multiply accumulate convolution";
  let description = [{
    AMD-specific multiply accumulate convolution intrinsic. Multiply accumulate convolution
    operation of (M x N)matrix with (N x 1)kernel.
    `$result = mac_convMxN($lhs, $rhs, $acc)`
  }];

  let extraClassDeclaration = [{
     // Get the attributes name
     llvm::StringRef getSubAttrName() { return "fmsub"; }
   }];
}

def AIEVec_MinOp
    : AIEVec_Op<"min", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector minimum";
  let description = [{
    AMD-specific intrinsic that calculates the minimum between two input vectors.
    `$result = min(`$lhs, $rhs`).
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_MaxOp
    : AIEVec_Op<"max", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector maximum";
  let description = [{
    AMD-specific intrinsic that calculates the maximum between two input vectors.
    `$result = max(`$lhs, $rhs`).
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_CmpOp
    : AIEVec_Op<"cmp", [Pure, AllTypesMatch<["lhs", "rhs"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs,
          DefaultValuedStrAttr<StrAttr, "">:$pred)>,
      Results<(outs AnyUnsignedInteger:$result)> {
  let summary = "AIE vector comparison";
  let description = [{
    AMD-specific intrinsic that performs element-wise comparisonof two input vectors.
    The attribute predicate defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`)
    -   not equal (mnemonic: `"ne"`)
    -   signed less than (mnemonic: `"slt"`)
    -   unsigned less than (mnemonic: `"ult"`)
    -   signed less than or equal (mnemonic: `"sle"`)
    -   unsigned less than or equal (mnemonic: `"ule"`)
    -   signed greater than (mnemonic: `"sgt"`)
    -   unsigned greater than (mnemonic: `"ugt"`)
    -   signed greater than or equal (mnemonic: `"sge"`)
    -   unsigned greater than or equal (mnemonic: `"uge"`)
  }];

  let assemblyFormat = "$lhs `,` $rhs ` ` `{` `pred` `=` $pred attr-dict `}` "
                       "`:` type($lhs) `,` type($rhs)  `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_SelOp
    : AIEVec_Op<"sel", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs,
          AnyUnsignedInteger:$sel)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector lane wise selection";
  let description = [{
    AMD-specific intrinsic that performs lane wise selection between two input vectors, if a bit of sel is zero, the lane of vector lhs is selected, else the lane of vector rhs is selected.
    `$result = sel(`$lhs, $rhs, $sel`).
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $sel attr-dict `:` type($lhs) `,` "
                       "type($rhs) `,` type($sel) `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_ExtElemOp
    : AIEVec_Op<"ext_elem", [Pure, AllElementTypesMatch<["source", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source, I32:$index)>,
      Results<(outs AnyTypeOf<[BF16, F32, I32, I16, I8]>:$result)> {
  let summary = "AIE extract element";
  let description = [{AMD - specific extract element
                  intrinsic. Extract element determined by index from vector.
    `$result = ext_elem($source, $index)`.}];
  let assemblyFormat = "$source `,` $index attr-dict `:` type($source) `,` "
                       "type($index) `,` type($result)";
  let hasVerifier = 1;
}

def AIEVec_NegOp
    : AIEVec_Op<"neg", [Pure, AllTypesMatch<["source", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector negative";
  let description = [{
    AMD-specific intrinsic that negates the vector and returns the result.
    `$result = neg(`$source`).
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_ExpOp
    : AIEVec_Op<"exp", [Pure, AllTypesMatch<["source", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector exponential";
  let description = [{
    AMD-specific intrinsic that computes the exponential of the input vector.
    For AIE2P, this will be lowered to the exp2 intrinsic using the identity
    exp(x) = exp2(x * log2(e)).
    `$result = exp(`$source`).
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_TanhOp
    : AIEVec_Op<"tanh", [Pure, AllTypesMatch<["source", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector hyperbolic tangent";
  let description = [{
    AMD-specific intrinsic that computes the hyperbolic tangent of the input
    vector. For AIE2P, this is lowered to the hardware tanh intrinsic.
    `$result = tanh(`$source`).
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_Exp2Op
    : AIEVec_Op<"exp2", [Pure, AllTypesMatch<["source", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$source)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE vector base-2 exponential";
  let description = [{
    AMD-specific intrinsic that computes 2 raised to each element of the input
    vector. For AIE2P, this is lowered to the hardware exp2 intrinsic.
    `$result = exp2(`$source`).
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_InvOp
    : AIEVec_Op<"inv", [Pure, AllTypesMatch<["source", "result"]>]>,
      Arguments<(
          ins AnyTypeOf<[F32,
                         VectorOfLengthAndType<[16, 32], [F32]>]>:$source)>,
      Results<(
          outs AnyTypeOf<[F32,
                          VectorOfLengthAndType<[16, 32], [F32]>]>:$result)> {
  let summary = "AIE inverse";
  let description = [{
    AMD-specific intrinsic that computes the inverse (1/x) of a scalar f32
    or a vector of f32 elements.
    For AIE2P, scalar f32 is lowered to the inv intrinsic, and vector f32
    is unrolled into scalar inv intrinsic calls.
    `$result = inv(`$source`) = 1.0 / $source`.
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_BxorOp
    : AIEVec_Op<"bxor", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]>,
      Arguments<(
          ins VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$lhs,
          VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$rhs)>,
      Results<(outs VectorOfBitWidthAndElementTypes<512, [I8, I16, I32,
                                                          BF16]>:$result)> {
  let summary = "AIE vector bitwise xor";
  let description = [{
    AMD-specific intrinsic that computes bitwise xor of two vectors and returns
    the result.
    `$result = bxor(`$lhs, $rhs`).
  }];
  let assemblyFormat = [{$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
                         `,` type($result)}];
  let hasVerifier = 0;
}

def AIEVec_BnegOp
    : AIEVec_Op<"bneg", [Pure, AllTypesMatch<["source", "result"]>]>,
      Arguments<(ins VectorOfBitWidthAndElementTypes<512, [I8, I16, I32,
                                                           BF16]>:$source)>,
      Results<(outs VectorOfBitWidthAndElementTypes<512, [I8, I16, I32,
                                                          BF16]>:$result)> {
  let summary = "AIE vector bitwise negation";
  let description = [{
    AMD-specific intrinsic that computes bitwise negation of a vector and
    returns the result.
    `$result = bneg(`$source`).
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_BorOp
    : AIEVec_Op<"bor", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]>,
      Arguments<(
          ins VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$lhs,
          VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$rhs)>,
      Results<(outs VectorOfBitWidthAndElementTypes<512, [I8, I16, I32,
                                                          BF16]>:$result)> {
  let summary = "AIE vector bitwise or";
  let description = [{
    AMD-specific intrinsic that computes bitwise or of two vectors and returns
    the result.
    `$result = bor(`$lhs, $rhs`).
  }];
  let assemblyFormat = [{$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
                         `,` type($result)}];
  let hasVerifier = 0;
}

def AIEVec_BandOp
    : AIEVec_Op<"band", [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]>,
      Arguments<(
          ins VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$lhs,
          VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$rhs)>,
      Results<(outs VectorOfBitWidthAndElementTypes<512, [I8, I16, I32,
                                                          BF16]>:$result)> {
  let summary = "AIE vector bitwise and";
  let description = [{
    AMD-specific intrinsic that computes bitwise and of two vectors and returns
    the result.
    `$result = band(`$lhs, $rhs`).
  }];
  let assemblyFormat = [{$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
                         `,` type($result)}];
  let hasVerifier = 0;
}

def AIEVec_MatMulOp
    : AIEVec_Op<"matmul", [Pure, AllRanksMatch<["lhs", "rhs", "acc"]>,
                           AllTypesMatch<["acc", "result"]>]>,
      Arguments<(ins AIE2MatMulLHS:$lhs, AIE2MatMulRHS:$rhs, AIE2MatMulACC:$acc,
          UnitAttr:$lhs_transposed, UnitAttr:$rhs_transposed)>,
      Results<(outs AIE2MatMulACC:$result)> {
  let summary = "AIE2 matrix-multiply and accummulate";
  let description = [{
    AMD AIEv2-specific intrinsic that performs a matrix multiplications
    between `lhs` and `rhs`, and accumulates the result in `acc`.

    Operand signedness is carried in the integer element types: a signed
    element type (`si8`) sets the MAC `signX`/`signY` bit, an unsigned type
    (`ui8`) clears it. A signless element type (`i8`) is treated as signed.
    Each operand is resolved independently, and `aievec.matmul_aie2p` follows
    the same rule.

    Currently, this intrinsic supports the following type combinations:

         lhs                | rhs                | Accumulator
        :------------------:|:------------------:|:-----------------:
         `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
         `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
         `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
         `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
         `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
         `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
         `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
         `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
         `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
         `vector<4x8xbf16>` | `vector<8x4xbf16>` | `vector<4x4xf32>`

    The table gives `lhs` as `[m, k]` and `rhs` as `[k, n]`. With
    `lhs_transposed`, `lhs` holds the `[k, m]` transpose instead, and with
    `rhs_transposed`, `rhs` holds the `[n, k]` transpose; the shapes change
    accordingly. The lowering transposes an operand with one shuffle when the
    multiply reads it the other way. An operand that needs a transpose no
    shuffle mode provides does not verify.
  }];
  let assemblyFormat = [{$lhs `,` $rhs `,` $acc
                         (`lhs_transposed` $lhs_transposed^)?
                         (`rhs_transposed` $rhs_transposed^)?
                         attr-dict `:` type($lhs) `,` type($rhs) `into`
                         type($acc)}];
  let hasVerifier = 1;
}

def AIEVec_MatMulOp_AIE2P
    : AIEVec_Op<"matmul_aie2p", [Pure, AllRanksMatch<["lhs", "rhs", "acc"]>,
                                 AllTypesMatch<["acc", "result"]>]>,
      Arguments<(ins AIE2PMatMulLHS:$lhs, AIE2PMatMulRHS:$rhs,
          AIE2PMatMulACC:$acc, UnitAttr:$lhs_transposed,
          UnitAttr:$rhs_transposed)>,
      Results<(outs AIE2PMatMulACC:$result)> {
  let summary = "AIE2P matrix-multiply and accummulate";
  let description = [{
    AMD AIEv2P-specific intrinsic that performs a matrix multiplication
    between `lhs` and `rhs`, and accumulates the result in `acc`.

    Operand signedness is carried in the integer element types exactly as for
    `aievec.matmul`: `si8` sets the MAC `signX`/`signY` bit, `ui8` clears it,
    and a signless element type is treated as signed.

    Currently, this intrinsic supports the following type combinations:

         lhs                 | rhs                 | Accumulator
        :-------------------:|:-------------------:|:------------------:
         `vector<8x8xbf16>`  | `vector<8x8xbf16>`  | `vector<8x8xf32>`
         `vector<4x8xbf16>`  | `vector<8x4xbf16>`  | `vector<4x4xf32>`
         `vector<4x8xbf16>`  | `vector<8x8xbf16>`  | `vector<4x8xf32>`
         `vector<8x1xbf16>`  | `vector<1x8xbf16>`  | `vector<8x8xf32>`
         `vector<8x8xbf16>`  | `vector<8x4xbf16>`  | `vector<8x4xf32>`
         `vector<8x8xi8>`    | `vector<8x8xi8>`    | `vector<8x8xi32>`
         `vector<8x2xi16>`   | `vector<2x8xi16>`   | `vector<8x8xi32>`

    `lhs_transposed` and `rhs_transposed` mean what they do for
    `aievec.matmul`. The bf16 multiplies through bfp16 (an `rhs` of
    `vector<8x8xbf16>`) read `rhs` as `[n, k]`, and the elementwise
    `vector<8x8xbf16>` times `vector<8x4xbf16>` multiply reads `lhs` as
    `[k, m]`; an operand stored that way needs no shuffle. Any other
    transposed operand takes one shuffle, or the `t16_8x8_lo`/`t16_8x8_hi`
    pair for an 8x8 bf16 tile.
  }];
  let assemblyFormat = [{$lhs `,` $rhs `,` $acc
                         (`lhs_transposed` $lhs_transposed^)?
                         (`rhs_transposed` $rhs_transposed^)?
                         attr-dict `:` type($lhs) `,` type($rhs) `into`
                         type($acc)}];
  let hasVerifier = 1;
}

def AIEVec_ToBFP16Op
    : AIEVec_Op<"to_bfp16", [Pure]>,
      Arguments<(ins VectorOfRankAndType<[2], [BF16]>:$source,
          UnitAttr:$transpose)>,
      Results<(outs VectorOfLengthAndType<[64], [I8]>:$mantissas,
          VectorOfLengthAndType<[8], [I8]>:$exponents)> {
  let summary = "AIE2P bf16 tile to bfp16ebs8";
  let description = [{
    AMD AIEv2P-specific conversion of a bf16 tile to bfp16ebs8, the block
    floating point format of the `8x8_8x8T` matrix multiplication mode, in
    which each row of 8 values shares one exponent. The tile is `source`, or
    its transpose if `transpose` is set, and has 4 or 8 rows of 8 values.
    `mantissas` holds the 8-bit mantissas of 8 rows and `exponents` their
    exponents; for a 4-row tile, the last 4 rows are undefined.

    `aievec.matmul_aie2p` on bf16 tiles is lowered as this conversion of each
    operand followed by `aievec.matmul_aie2p_bfp16`.
  }];
  let assemblyFormat = [{$source (`transpose` $transpose^)? attr-dict `:`
                         type($source) `->` type($mantissas) `,`
                         type($exponents)}];
  let hasVerifier = 1;
}

def AIEVec_LoadBFP16Op
    : AIEVec_Op<"load_bfp16">,
      Arguments<(ins Arg<MemRefRankOf<[I8], [1]>, "", [MemRead]>:$base,
          Index:$offset)>,
      Results<(outs VectorOfLengthAndType<[64], [I8]>:$mantissas,
          VectorOfLengthAndType<[8], [I8]>:$exponents)> {
  let summary = "AIE2P load of a bfp16ebs8 block";
  let description = [{
    AMD AIEv2P-specific load of the 72-byte bfp16ebs8 block at byte `offset`
    of `base`: 8 rows, each a shared exponent byte followed by 8 mantissa
    bytes. The result has the form `aievec.to_bfp16` produces. Blocks are
    not aligned to vector registers; they are read through the block-vector
    load FIFO.
  }];
  let assemblyFormat = [{$base `[` $offset `]` attr-dict `:` type($base) `->`
                         type($mantissas) `,` type($exponents)}];
  let hasVerifier = 0;
}

def AIEVec_FifoStartOp
    : AIEVec_Op<"fifo_start", [Pure]>,
      Arguments<(ins MemRefRankOf<[I8], [1]>:$base, Index:$offset)>,
      Results<(outs LLVM_DefaultPointer:$next,
          VectorOfLengthAndType<[32], [I32]>:$state, I32:$count)> {
  let summary = "AIE2P start of a block-vector FIFO stream";
  let description = [{
    AMD AIEv2P-specific start of a stream on a block-vector load or store
    FIFO at byte `offset` of `base`. The FIFO starts empty. `next` (the
    address of the next memory access), `state` and `count` hold the FIFO
    state and are passed to the first
    `aievec.fifo_ld_pop_bfp16` or `aievec.fifo_st_push_bfp16` of the
    stream.
  }];
  let assemblyFormat = [{$base `[` $offset `]` attr-dict `:` type($base) `->`
                         type($state)}];
  let hasVerifier = 0;
}

def AIEVec_FifoStPushBFP16Op
    : AIEVec_Op<"fifo_st_push_bfp16", [MemoryEffects<[MemWrite]>]>,
      Arguments<(ins LLVM_DefaultPointer:$addr,
          VectorOfLengthAndType<[32], [I32]>:$state, I32:$count,
          VectorOfLengthAndType<[64], [I8]>:$mantissas,
          VectorOfLengthAndType<[8], [I8]>:$exponents)>,
      Results<(outs LLVM_DefaultPointer:$next,
          VectorOfLengthAndType<[32], [I32]>:$state_out, I32:$count_out)> {
  let summary = "AIE2P next bfp16ebs8 block of a block-vector store stream";
  let description = [{
    AMD AIEv2P-specific write of `mantissas` and `exponents`, as produced by
    `aievec.to_bfp16`, as the next 72-byte bfp16ebs8 block of the store
    stream whose state is `addr`, `state` and `count`. The FIFO is flushed
    after the write. Returns the updated state. The blocks can be read back
    with `aievec.load_bfp16` or `aievec.fifo_ld_pop_bfp16`.
  }];
  let assemblyFormat = [{$addr `,` $state `,` $count `,` $mantissas `,`
                         $exponents attr-dict `:` type($state) `,`
                         type($mantissas) `,` type($exponents) `->`
                         type($state_out)}];
  let hasVerifier = 0;
}

def AIEVec_FifoLdPopBFP16Op
    : AIEVec_Op<"fifo_ld_pop_bfp16", [MemoryEffects<[MemRead]>]>,
      Arguments<(ins LLVM_DefaultPointer:$addr,
          VectorOfLengthAndType<[32], [I32]>:$state, I32:$count,
          UnitAttr:$no_fill)>,
      Results<(outs LLVM_DefaultPointer:$next,
          VectorOfLengthAndType<[32], [I32]>:$state_out, I32:$count_out,
          VectorOfLengthAndType<[64], [I8]>:$mantissas,
          VectorOfLengthAndType<[8], [I8]>:$exponents)> {
  let summary = "AIE2P next bfp16ebs8 block of a block-vector load stream";
  let description = [{
    AMD AIEv2P-specific read of the next 72-byte bfp16ebs8 block of the load
    stream whose state is `addr`, `state` and `count`. Returns the block in
    the form `aievec.to_bfp16` produces, and the updated state. A stream
    needs one memory load per block, whereas `aievec.load_bfp16` refills
    the FIFO for every block.

    A pop consumes 576 bits and loads 512, so the FIFO is refilled before
    each pop unless `no_fill` is set. One fill provides enough data for the
    8 pops that follow it, as long as nothing else uses the FIFO in
    between.
  }];
  let assemblyFormat = [{$addr `,` $state `,` $count (`no_fill` $no_fill^)?
                         attr-dict `:`
                         type($state) `->` type($state_out) `,` type($mantissas)
                         `,` type($exponents)}];
  let hasVerifier = 0;
}

def AIEVec_MatMulBFP16Op_AIE2P
    : AIEVec_Op<"matmul_aie2p_bfp16", [Pure, AllTypesMatch<["acc", "result"]>]>,
      Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs_mantissas,
          VectorOfLengthAndType<[8], [I8]>:$lhs_exponents,
          VectorOfLengthAndType<[64], [I8]>:$rhs_mantissas,
          VectorOfLengthAndType<[8], [I8]>:$rhs_exponents,
          AnyTypeOf<[VectorOfShapeAndType<[8, 8], F32>,
                     VectorOfShapeAndType<[4, 8], F32>]>:$acc)>,
      Results<(outs AnyTypeOf<[VectorOfShapeAndType<[8, 8], F32>,
                               VectorOfShapeAndType<[4, 8], F32>]>:$result)> {
  let summary = "AIE2P bfp16 matrix-multiply and accumulate";
  let description = [{
    AMD AIEv2P-specific `8x8_8x8T` multiply-accumulate of two bfp16 tiles as
    produced by `aievec.to_bfp16`. Computes `acc + lhs * transpose(rhs)`,
    where `lhs` is the [m, k] tile and `rhs` the [n, k] tile. `acc` is
    [m, n]; when it has 4 rows, only the first 4 rows of `lhs` are used.
  }];
  let assemblyFormat = [{$lhs_mantissas `,` $lhs_exponents `,` $rhs_mantissas
                         `,` $rhs_exponents `,` $acc attr-dict `:`
                         type($lhs_mantissas) `,` type($lhs_exponents) `,`
                         type($rhs_mantissas) `,` type($rhs_exponents) `into`
                         type($acc)}];
  let hasVerifier = 0;
}

def AIEVec_ShuffleOp
    : AIEVec_Op<"shuffle", [Pure, AllTypesMatch<["lhs", "result"]>,
                            OptionalTypesMatchWith<
                                "result and rhs have the same type", "result",
                                "rhs", "::llvm::cast<Type>($_self)">]>,
      Arguments<(
          ins VectorOfBitWidthAndElementTypes<
              512, [I8, I16, I32, I64, I128, I256, I512, BF16, F32]>:$lhs,
          Optional<VectorOfBitWidthAndElementTypes<
              512, [I8, I16, I32, I64, I128, I256, I512, BF16, F32]>>:$rhs,
          AIEVec_ShuffleModeAttr:$mode)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 shuffle";
  let description = [{
    AMD AIEv2-specific vector shuffle. It performs a shuffle of the elements of
    1 or 2 input vectors using the specified shuffle mode. The shuffle mode is
    specified as:

      `t<width>_<r>x<c>(_(hi|lo))?`

    where `<width>` is the bitwidth of the vector element type, `<r>` and `<c>`
    are the number of rows and columns that will be transposed to perform the
    shuffle, and, for modes that require two 512-bit vectors, `hi` and `lo`
    indicate which part of the resulting extended 1024-bit vector will be
    assembled and returned.

    E.g.: `t32_4x8` would take two 512-bit vectors, `lhs` and `rhs`, with 16
    elements of 32 bits each. The resulting vector would contain either the
    least (`lo`) or most (`hi`) significant 16 elements of the 32 element vector
    that would result from selecting, out of the concatenated vectors `lhs:rhs`,
    8 blocks of 4 elements, each block taking one of every 8 elements starting
    from the block index.

    That is, for two `vector<16xi32>` operands containing:
    ```
    lhs = [0,   1,  2,  3, ..., 15]
    rhs = [17, 18, 19, 20, ..., 31]
    ```

    The first 8 blocks would be:
    ```
    b0 = [0,  8, 16, 24]
    b1 = [1,  9, 17, 25]
    b2 = [2, 10, 18, 26]
    b3 = [3, 11, 19, 27]
       ...
    b7 = [7, 15, 23, 31]
    ```

    `t32_4x8_lo` would return first four blocks:
    ```
    result = [0, 8, 16, 24, 1, 9, 17, 25, ..., 3, 11, 19, 27]
    ```

    And `t32_4x8_hi` would return the last four blocks:
    ```
    result = [4, 12, 20, 28, 5, 13, 21, 29, ..., 7, 15, 24, 31]
    ```

    It can be seen as flattened 4x8 matrix, split in two 16-element halfs, being
    tranposed to a 8x4 arrangement. In the example above:

    ```
    lhs = [ 0,  1,  2,  3,  4,  5,  6,  7]
          [ 8,  9, 10, 11, 12, 13, 14, 15]
    rhs = [16, 17, 18, 19, 20, 21, 22, 23]
          [24, 25, 26, 27, 28, 29, 30, 31]
    ```

    Would result in:
    ```
    t32_4x8_lo = [0,  8, 16, 24]
                 [1,  9, 17, 25]
                 [2, 10, 18, 26]
                 [3, 11, 19, 27]
    t32_4x8_hi = [4, 12, 20, 28]
                 [5, 13, 21, 29]
                 [6, 14, 22, 30]
                 [7, 15, 23, 31]
    ```

    A special mode, `t16_1x2_flip`, swaps each pair of elements in a vector with
    32 16-bit elements. E.g.:
    ```
    lhs = [0, 1, 2, 3, ..., 28, 29, 30, 31]
    ```
    Would result in:
    ```
    t16_1x2_flip = [1, 0, 3, 2, ..., 29, 28, 31, 30]
    ```

    In general, mode `t<w>_<r>x<c>` reads an `r`x`c` row-major matrix of
    `w`-bit elements and returns its transpose: `out[j*r+i] = in[i*c+j]`. A
    mode whose matrix spans 1024 bits reads `lhs:rhs` and returns the `lo` or
    `hi` 512 bits. A mode whose matrix spans fewer than 512 bits reads `lhs`
    only and transposes each such block of it on its own. A mode
    `t<w>_1x<c>_flip` reverses the order of each group of `c` elements.

    The modes 48 to 55 (`t32_8x2`, `t32_2x8`, `t64_4x2`, `t64_2x4`,
    `t16_8x8_lo`, `t16_8x8_hi`, `t32_1x16_flip` and `t8_1x4_flip`) exist on
    AIE2P only. Inside an `aie.device` of another architecture they are
    rejected.

    The list of supported shuffle modes, required operands, and associated
    vector types are the following:

         Shuffle Mode       | Operands           | Types Supported
        :------------------:|:------------------:|:------------------:
         t8_8x4             | `lhs`              | `vector<64xi8>`
         t8_4x8             | ^                  | ^
         t8_8x8             | ^                  | ^
         t8_16x4            | ^                  | ^
         t8_4x16            | ^                  | ^
         t8_1x4_flip (AIE2P)| ^                  | ^
         t8_64x2_lo         | `lhs` & `rhs`      | ^
         t8_64x2_hi         | ^                  | ^
         t8_2x64_lo         | ^                  | ^
         t8_2x64_hi         | ^                  | ^
         t16_4x2            | `lhs`              | `vector<32xi16>` or `vector<32xbf16>`
         t16_2x4            | ^                  | ^
         t16_4x4            | ^                  | ^
         t16_8x2            | ^                  | ^
         t16_2x8            | ^                  | ^
         t16_8x4            | ^                  | ^
         t16_4x8            | ^                  | ^
         t16_16x2           | ^                  | ^
         t16_2x16           | ^                  | ^
         t16_1x2_flip       | ^                  | ^
         t16_32x2_lo        | `lhs` & `rhs`      | ^
         t16_32x2_hi        | ^                  | ^
         t16_2x32_lo        | ^                  | ^
         t16_2x32_hi        | ^                  | ^
         t16_16x4_lo        | ^                  | ^
         t16_16x4_hi        | ^                  | ^
         t16_4x16_lo        | ^                  | ^
         t16_4x16_hi        | ^                  | ^
         t16_8x8_lo (AIE2P) | ^                  | ^
         t16_8x8_hi (AIE2P) | ^                  | ^
         t32_4x4            | `lhs`              | `vector<16xi32>` or `vector<16xf32>`
         t32_8x2 (AIE2P)    | ^                  | ^
         t32_2x8 (AIE2P)    | ^                  | ^
         t32_1x16_flip (AIE2P) | ^               | ^
         t32_16x2_lo        | `lhs` & `rhs`      | ^
         t32_16x2_hi        | ^                  | ^
         t32_2x16_lo        | ^                  | ^
         t32_2x16_hi        | ^                  | ^
         t32_8x4_lo         | ^                  | ^
         t32_8x4_hi         | ^                  | ^
         t32_4x8_lo         | ^                  | ^
         t32_4x8_hi         | ^                  | ^
         t64_4x2 (AIE2P)    | `lhs`              | `vector<8xi64>`
         t64_2x4 (AIE2P)    | ^                  | ^
         t64_8x2_lo         | `lhs` & `rhs`      | ^
         t64_8x2_hi         | ^                  | ^
         t64_2x8_lo         | ^                  | ^
         t64_2x8_hi         | ^                  | ^
         t128_4x2_lo        | ^                  | `vector<4xi128>`
         t128_4x2_hi        | ^                  | ^
         t128_2x4_lo        | ^                  | ^
         t128_2x4_hi        | ^                  | ^
         t256_2x2_lo        | ^                  | `vector<2xi256>`
         t256_2x2_hi        | ^                  | ^
         t512_1x2_lo        | ^                  | `vector<1xi512>`
         t512_1x2_hi        | ^                  | ^
  }];
  let assemblyFormat = [{$lhs (`,` $rhs^)? $mode attr-dict `:` type($result)}];
  let hasVerifier = 1;
}

#endif // AIEVEC_OPS
