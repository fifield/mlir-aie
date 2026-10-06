//===- AIEVecAttributes.td - AIE vector attributes def. ----*- tablegen
//-*-====//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines AIE vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECATTRIBUTES_TD
#define AIE_DIALECT_AIEVEC_IR_AIEVECATTRIBUTES_TD

include "aie/Dialect/AIEVec/IR/AIEVecDialect.td"
include "mlir/IR/EnumAttr.td"

// Shuffle modes for shuffle ops.
def SHUFFLE_MODE_T8_64X2_LO : I32EnumAttrCase<"T8_64X2_LO", 0, "t8_64x2_lo">;
def SHUFFLE_MODE_T8_64X2_HI : I32EnumAttrCase<"T8_64X2_HI", 1, "t8_64x2_hi">;
def SHUFFLE_MODE_T16_32X2_LO : I32EnumAttrCase<"T16_32X2_LO", 2, "t16_32x2_lo">;
def SHUFFLE_MODE_T16_32X2_HI : I32EnumAttrCase<"T16_32X2_HI", 3, "t16_32x2_hi">;
def SHUFFLE_MODE_T32_16X2_LO : I32EnumAttrCase<"T32_16X2_LO", 4, "t32_16x2_lo">;
def SHUFFLE_MODE_T32_16X2_HI : I32EnumAttrCase<"T32_16X2_HI", 5, "t32_16x2_hi">;
def SHUFFLE_MODE_T64_8X2_LO : I32EnumAttrCase<"T64_8X2_LO", 6, "t64_8x2_lo">;
def SHUFFLE_MODE_T64_8X2_HI : I32EnumAttrCase<"T64_8X2_HI", 7, "t64_8x2_hi">;
def SHUFFLE_MODE_T128_4X2_LO : I32EnumAttrCase<"T128_4X2_LO", 8, "t128_4x2_lo">;
def SHUFFLE_MODE_T128_4X2_HI : I32EnumAttrCase<"T128_4X2_HI", 9, "t128_4x2_hi">;
def SHUFFLE_MODE_T256_2X2_LO
    : I32EnumAttrCase<"T256_2X2_LO", 10, "t256_2x2_lo">;
def SHUFFLE_MODE_T256_2X2_HI
    : I32EnumAttrCase<"T256_2X2_HI", 11, "t256_2x2_hi">;
def SHUFFLE_MODE_T128_2X4_LO
    : I32EnumAttrCase<"T128_2X4_LO", 12, "t128_2x4_lo">;
def SHUFFLE_MODE_T128_2X4_HI
    : I32EnumAttrCase<"T128_2X4_HI", 13, "t128_2x4_hi">;
def SHUFFLE_MODE_T64_2X8_LO : I32EnumAttrCase<"T64_2X8_LO", 14, "t64_2x8_lo">;
def SHUFFLE_MODE_T64_2X8_HI : I32EnumAttrCase<"T64_2X8_HI", 15, "t64_2x8_hi">;
def SHUFFLE_MODE_T32_2X16_LO
    : I32EnumAttrCase<"T32_2X16_LO", 16, "t32_2x16_lo">;
def SHUFFLE_MODE_T32_2X16_HI
    : I32EnumAttrCase<"T32_2X16_HI", 17, "t32_2x16_hi">;
def SHUFFLE_MODE_T16_2X32_LO
    : I32EnumAttrCase<"T16_2X32_LO", 18, "t16_2x32_lo">;
def SHUFFLE_MODE_T16_2X32_HI
    : I32EnumAttrCase<"T16_2X32_HI", 19, "t16_2x32_hi">;
def SHUFFLE_MODE_T8_2X64_LO : I32EnumAttrCase<"T8_2X64_LO", 20, "t8_2x64_lo">;
def SHUFFLE_MODE_T8_2X64_HI : I32EnumAttrCase<"T8_2X64_HI", 21, "t8_2x64_hi">;
def SHUFFLE_MODE_T512_1X2_LO
    : I32EnumAttrCase<"T512_1X2_LO", 22, "t512_1x2_lo">;
def SHUFFLE_MODE_T512_1X2_HI
    : I32EnumAttrCase<"T512_1X2_HI", 23, "t512_1x2_hi">;
def SHUFFLE_MODE_T16_16X4_LO
    : I32EnumAttrCase<"T16_16X4_LO", 24, "t16_16x4_lo">;
def SHUFFLE_MODE_T16_16X4_HI
    : I32EnumAttrCase<"T16_16X4_HI", 25, "t16_16x4_hi">;
def SHUFFLE_MODE_T16_4X16_LO
    : I32EnumAttrCase<"T16_4X16_LO", 26, "t16_4x16_lo">;
def SHUFFLE_MODE_T16_4X16_HI
    : I32EnumAttrCase<"T16_4X16_HI", 27, "t16_4x16_hi">;
def SHUFFLE_MODE_T16_8X4 : I32EnumAttrCase<"T16_8X4", 28, "t16_8x4">;
def SHUFFLE_MODE_T16_4X8 : I32EnumAttrCase<"T16_4X8", 29, "t16_4x8">;
def SHUFFLE_MODE_T32_8X4_LO : I32EnumAttrCase<"T32_8X4_LO", 30, "t32_8x4_lo">;
def SHUFFLE_MODE_T32_8X4_HI : I32EnumAttrCase<"T32_8X4_HI", 31, "t32_8x4_hi">;
def SHUFFLE_MODE_T32_4X8_LO : I32EnumAttrCase<"T32_4X8_LO", 32, "t32_4x8_lo">;
def SHUFFLE_MODE_T32_4X8_HI : I32EnumAttrCase<"T32_4X8_HI", 33, "t32_4x8_hi">;
def SHUFFLE_MODE_T32_4X4 : I32EnumAttrCase<"T32_4X4", 34, "t32_4x4">;
def SHUFFLE_MODE_T8_8X8 : I32EnumAttrCase<"T8_8X8", 35, "t8_8x8">;
def SHUFFLE_MODE_T8_16X4 : I32EnumAttrCase<"T8_16X4", 36, "t8_16x4">;
def SHUFFLE_MODE_T8_4X16 : I32EnumAttrCase<"T8_4X16", 37, "t8_4x16">;
def SHUFFLE_MODE_T16_1X2_flip
    : I32EnumAttrCase<"T16_1X2_flip", 38, "t16_1x2_flip">;
def SHUFFLE_MODE_T16_4X4 : I32EnumAttrCase<"T16_4X4", 39, "t16_4x4">;
def SHUFFLE_MODE_T16_4X2 : I32EnumAttrCase<"T16_4X2", 40, "t16_4x2">;
def SHUFFLE_MODE_T16_2X4 : I32EnumAttrCase<"T16_2X4", 41, "t16_2x4">;
def SHUFFLE_MODE_T16_8X2 : I32EnumAttrCase<"T16_8X2", 42, "t16_8x2">;
def SHUFFLE_MODE_T16_2X8 : I32EnumAttrCase<"T16_2X8", 43, "t16_2x8">;
def SHUFFLE_MODE_T16_16X2 : I32EnumAttrCase<"T16_16X2", 44, "t16_16x2">;
def SHUFFLE_MODE_T16_2X16 : I32EnumAttrCase<"T16_2X16", 45, "t16_2x16">;
def SHUFFLE_MODE_T8_8X4 : I32EnumAttrCase<"T8_8X4", 46, "t8_8x4">;
def SHUFFLE_MODE_T8_4X8 : I32EnumAttrCase<"T8_4X8", 47, "t8_4x8">;
// Modes 48 to 55 exist on AIE2P only.
def SHUFFLE_MODE_T32_8X2 : I32EnumAttrCase<"T32_8X2", 48, "t32_8x2">;
def SHUFFLE_MODE_T32_2X8 : I32EnumAttrCase<"T32_2X8", 49, "t32_2x8">;
def SHUFFLE_MODE_T64_4X2 : I32EnumAttrCase<"T64_4X2", 50, "t64_4x2">;
def SHUFFLE_MODE_T64_2X4 : I32EnumAttrCase<"T64_2X4", 51, "t64_2x4">;
def SHUFFLE_MODE_T16_8X8_LO : I32EnumAttrCase<"T16_8X8_LO", 52, "t16_8x8_lo">;
def SHUFFLE_MODE_T16_8X8_HI : I32EnumAttrCase<"T16_8X8_HI", 53, "t16_8x8_hi">;
def SHUFFLE_MODE_T32_1X16_flip
    : I32EnumAttrCase<"T32_1X16_flip", 54, "t32_1x16_flip">;
def SHUFFLE_MODE_T8_1X4_flip
    : I32EnumAttrCase<"T8_1X4_flip", 55, "t8_1x4_flip">;

def ShuffleMode
    : I32EnumAttr<
          "ShuffleMode", "Shuffle mode for AIEVec shuffle operations",
          [SHUFFLE_MODE_T8_64X2_LO, SHUFFLE_MODE_T8_64X2_HI,
           SHUFFLE_MODE_T16_32X2_LO, SHUFFLE_MODE_T16_32X2_HI,
           SHUFFLE_MODE_T32_16X2_LO, SHUFFLE_MODE_T32_16X2_HI,
           SHUFFLE_MODE_T64_8X2_LO, SHUFFLE_MODE_T64_8X2_HI,
           SHUFFLE_MODE_T128_4X2_LO, SHUFFLE_MODE_T128_4X2_HI,
           SHUFFLE_MODE_T256_2X2_LO, SHUFFLE_MODE_T256_2X2_HI,
           SHUFFLE_MODE_T128_2X4_LO, SHUFFLE_MODE_T128_2X4_HI,
           SHUFFLE_MODE_T64_2X8_LO, SHUFFLE_MODE_T64_2X8_HI,
           SHUFFLE_MODE_T32_2X16_LO, SHUFFLE_MODE_T32_2X16_HI,
           SHUFFLE_MODE_T16_2X32_LO, SHUFFLE_MODE_T16_2X32_HI,
           SHUFFLE_MODE_T8_2X64_LO, SHUFFLE_MODE_T8_2X64_HI,
           SHUFFLE_MODE_T512_1X2_LO, SHUFFLE_MODE_T512_1X2_HI,
           SHUFFLE_MODE_T16_16X4_LO, SHUFFLE_MODE_T16_16X4_HI,
           SHUFFLE_MODE_T16_4X16_LO, SHUFFLE_MODE_T16_4X16_HI,
           SHUFFLE_MODE_T16_8X4, SHUFFLE_MODE_T16_4X8, SHUFFLE_MODE_T32_8X4_LO,
           SHUFFLE_MODE_T32_8X4_HI, SHUFFLE_MODE_T32_4X8_LO,
           SHUFFLE_MODE_T32_4X8_HI, SHUFFLE_MODE_T32_4X4, SHUFFLE_MODE_T8_8X8,
           SHUFFLE_MODE_T8_16X4, SHUFFLE_MODE_T8_4X16,
           SHUFFLE_MODE_T16_1X2_flip, SHUFFLE_MODE_T16_4X4,
           SHUFFLE_MODE_T16_4X2, SHUFFLE_MODE_T16_2X4, SHUFFLE_MODE_T16_8X2,
           SHUFFLE_MODE_T16_2X8, SHUFFLE_MODE_T16_16X2, SHUFFLE_MODE_T16_2X16,
           SHUFFLE_MODE_T8_8X4, SHUFFLE_MODE_T8_4X8, SHUFFLE_MODE_T32_8X2,
           SHUFFLE_MODE_T32_2X8, SHUFFLE_MODE_T64_4X2, SHUFFLE_MODE_T64_2X4,
           SHUFFLE_MODE_T16_8X8_LO, SHUFFLE_MODE_T16_8X8_HI,
           SHUFFLE_MODE_T32_1X16_flip, SHUFFLE_MODE_T8_1X4_flip]> {
  let cppNamespace = "::xilinx::aievec";
  let genSpecializedAttr = 0;
}

// See python/dialects/AIEVecEnumBinding.td for the Python binding exclusion.
#ifndef AIEVEC_PYTHON_ENUM_BINDINGS
def AIEVec_ShuffleModeAttr : EnumAttr<AIEVec_Dialect, ShuffleMode, "mode"> {
  let assemblyFormat = "`[` $value `]`";
}
#endif

//===----------------------------------------------------------------------===//
// AIE4 integer min/max/compare/select
//===----------------------------------------------------------------------===//

def AIE4_MinMaxKind_Min : I32EnumAttrCase<"Min", 0, "min">;
def AIE4_MinMaxKind_Max : I32EnumAttrCase<"Max", 1, "max">;
def AIE4MinMaxKind
    : I32EnumAttr<"AIE4MinMaxKind", "AIE4 native min/max selector",
                  [AIE4_MinMaxKind_Min, AIE4_MinMaxKind_Max]> {
  let cppNamespace = "::xilinx::aievec";
  let genSpecializedAttr = 0;
}

def AIEVec_AIE4MinMaxKindAttr
    : EnumAttr<AIEVec_Dialect, AIE4MinMaxKind, "aie4_minmax_kind">;

def AIE4_CmpPredicate_Lt : I32EnumAttrCase<"Lt", 0, "lt">;
def AIE4_CmpPredicate_Ge : I32EnumAttrCase<"Ge", 1, "ge">;
def AIE4CmpPredicate
    : I32EnumAttr<"AIE4CmpPredicate",
                  "AIE4 native less-than or greater-or-equal predicate",
                  [AIE4_CmpPredicate_Lt, AIE4_CmpPredicate_Ge]> {
  let cppNamespace = "::xilinx::aievec";
  let genSpecializedAttr = 0;
}

def AIEVec_AIE4CmpPredicateAttr
    : EnumAttr<AIEVec_Dialect, AIE4CmpPredicate, "aie4_cmp_predicate">;

//===----------------------------------------------------------------------===//
// AIE4 1024-bit vector reductions
//===----------------------------------------------------------------------===//

def AIE4_ReduceKind_Add : I32EnumAttrCase<"Add", 0, "add">;
def AIE4_ReduceKind_Min : I32EnumAttrCase<"Min", 1, "min">;
def AIE4_ReduceKind_Max : I32EnumAttrCase<"Max", 2, "max">;
def AIE4ReduceKind
    : I32EnumAttr<"AIE4ReduceKind",
                  "AIE4 native 1024-bit vector-to-scalar reduction kind",
                  [AIE4_ReduceKind_Add, AIE4_ReduceKind_Min,
                   AIE4_ReduceKind_Max]> {
  let cppNamespace = "::xilinx::aievec";
  let genSpecializedAttr = 0;
}

def AIEVec_AIE4ReduceKindAttr
    : EnumAttr<AIEVec_Dialect, AIE4ReduceKind, "aie4_reduce_kind">;

//===----------------------------------------------------------------------===//
// AIE4 shuffle
//===----------------------------------------------------------------------===//
//
// AIE4 encodings from Peano's aie4/aie4_enums.h, checked against the ISA.
// AIE2 uses different encodings, so these modes need a separate enum.
// Keep values in sync with the XLLVM AIE4ShuffleMode enum.
//
// T32_4X4 and T8_8X8 transpose one 512-bit vector. T32_2X16_LO/HI
// interleave the lower/upper eight lanes of two 16-lane vectors.
def AIE4_ShuffleMode_T32_4X4 : I32EnumAttrCase<"T32_4X4", 90, "t32_4x4">;
def AIE4_ShuffleMode_T8_8X8 : I32EnumAttrCase<"T8_8X8", 72, "t8_8x8">;
def AIE4_ShuffleMode_T32_2X16_LO
    : I32EnumAttrCase<"T32_2X16_LO", 66, "t32_2x16_lo">;
def AIE4_ShuffleMode_T32_2X16_HI
    : I32EnumAttrCase<"T32_2X16_HI", 67, "t32_2x16_hi">;

def AIE4ShuffleMode
    : I32EnumAttr<"AIE4ShuffleMode", "AIE4 vector shuffle/transpose mode",
                  [AIE4_ShuffleMode_T32_4X4, AIE4_ShuffleMode_T8_8X8,
                   AIE4_ShuffleMode_T32_2X16_LO,
                   AIE4_ShuffleMode_T32_2X16_HI]> {
  let cppNamespace = "::xilinx::aievec";
  let genSpecializedAttr = 0;
}

def AIEVec_AIE4ShuffleModeAttr
    : EnumAttr<AIEVec_Dialect, AIE4ShuffleMode, "aie4_shuffle_mode">;

//===----------------------------------------------------------------------===//
// AIE4 dense matrix MAC formats
//===----------------------------------------------------------------------===//

// Operand format of a dense MMAC tile. Each format fixes the A row width, the
// B tile shape, and the accumulator element type (see aievec.aie4.mmac).
def AIE4_MatrixFormat_Bf16 : I32EnumAttrCase<"Bf16", 0, "bf16">;
def AIE4_MatrixFormat_Int8 : I32EnumAttrCase<"Int8", 1, "int8">;
def AIE4_MatrixFormat_Int4 : I32EnumAttrCase<"Int4", 2, "int4">;
def AIE4_MatrixFormat_Fp8 : I32EnumAttrCase<"Fp8", 3, "fp8">;
def AIE4MatrixFormat
    : I32EnumAttr<"AIE4MatrixFormat", "AIE4 dense MMAC operand format",
                  [AIE4_MatrixFormat_Bf16, AIE4_MatrixFormat_Int8,
                   AIE4_MatrixFormat_Int4, AIE4_MatrixFormat_Fp8]> {
  let cppNamespace = "::xilinx::aievec";
  let genSpecializedAttr = 0;
}

def AIEVec_AIE4MatrixFormatAttr
    : EnumAttr<AIEVec_Dialect, AIE4MatrixFormat, "aie4_matrix_format"> {
  let assemblyFormat = "$value";
}

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECATTRIBUTES_TD
