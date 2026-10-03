//===- AIEVecAIE4Ops.td - AIE4 matrix MAC op definitions --*- tablegen -*-====//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// AIE4 dense MMAC operations. The pure matmul op has explicit operands;
// matrix_region and row MMACs represent a resident B tile and its uses.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECAIE4OPS_TD
#define AIE_DIALECT_AIEVEC_IR_AIEVECAIE4OPS_TD

//===----------------------------------------------------------------------===//
// Semantic layer
//===----------------------------------------------------------------------===//

def AIEVec_AIE4MatMulOp
    : AIEVec_Op<"aie4.matmul", [Pure, AllTypesMatch<["acc", "result"]>]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs,
          AnyVectorOfNonZeroRank:$acc, OptionalAttr<BoolAttr>:$lhs_signed,
          OptionalAttr<BoolAttr>:$rhs_signed)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE4 matrix multiply-accumulate over one native B tile";
  let description = [{
    Computes `result[m][n] = acc[m][n] + sum_k lhs[m][k] * rhs[k][n]` with
    the numerics of the AIE4 dense MMAC for the operand format:

    | lhs | rhs (one B tile) | acc / result |
    |-----|------------------|--------------|
    | `vector<Mx32xbf16>` | `vector<32x64xbf16>` | `vector<Mx64xf32>` |
    | `vector<Mx64xi8>` | `vector<64x64xi8>` | `vector<Mx64xi32>` |
    | `vector<Mx128xi4>` | `vector<128x64xi4>` | `vector<Mx64xi32>` |
    | `vector<Mx64xf8E4M3FN>` | `vector<64x64xf8E4M3FN>` | `vector<Mx64xf32>` |

    `M` rows share one B tile. Producers tile larger K or N dimensions into
    separate ops, carrying the accumulator between K tiles. Lowering
    unrolls at most 64 rows (`kAIE4MatMulMaxRows`) to bound register pressure;
    producers must tile larger M dimensions.

    Integer formats require `lhs_signed` and `rhs_signed`, which select the
    X and Z operand signedness independently. Floating formats reject them.

    Converting a floating-point `vector.contract` requires its `afn`
    fastmath flag (`fast` includes it). Shape alone is insufficient because
    MMAC uses hardware accumulation order and rounding. Integer contractions
    need no fastmath permission.

    Numerical behavior:

    * bf16 MMAC accepts subnormal bf16 multiplicands but flushes subnormal
      f32 accumulator inputs and results to zero. Rewrites must preserve
      this behavior: `matmul(0, b, acc)` cannot generally fold to `acc`.
      Constant folding requires a bit-exact model of the hardware numerics.
    * int8 and int4 products are exact in i32. Accumulation wraps modulo
      2^32, matching `arith.addi` and `arith.muli` on operands extended
      according to `lhs_signed` and `rhs_signed`.
    * fp8 uses E4M3FN with `emode = all`: a NaN anywhere in the lhs row
      propagates to every result lane. This op makes no guarantees for
      NaNs in the matrix operand or for fp8 subnormal inputs.
    * Floating-point exception flags in `srFPMFlags` are not modeled or
      preserved. Results depend only on operands and attributes, without
      reading rounding or saturation mode registers, so the op is `Pure`.

    Example:

    ```mlir
    %r = aievec.aie4.matmul %a, %b, %c
        : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
    ```
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` "
                       "type($rhs) `into` type($acc)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Stateful layer: region-scoped matrix residency
//===----------------------------------------------------------------------===//

def AIEVec_AIE4MatrixRegionOp
    : AIEVec_Op<"aie4.matrix_region", [SingleBlock]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$rhs, Variadic<AnyType>:$inits,
          AIEVec_AIE4MatrixFormatAttr:$format,
          OptionalAttr<BoolAttr>:$rhs_signed)>,
      Results<(outs Variadic<AnyType>:$results)> {
  let summary = "AIE4 region with one resident MMAC matrix";
  let description = [{
    Makes `rhs` the active MMAC matrix for the body. Every
    `aievec.aie4.mmac` nested in the body multiplies by it. `rhs` is one
    native B tile with the logical K x 64 shape of `aievec.aie4.matmul`.
    Integer formats require `rhs_signed`.

    `inits` become the entry block arguments, typically the accumulator
    rows. The body ends with `aievec.aie4.matrix_yield`, whose operands
    become the results. Other values may be used from above.

    Region constraints:

    * Every `aievec.aie4.mmac` must be inside a region with the same format
      and B signedness. Regions cannot nest because there is one active
      matrix.
    * Calls, `llvm.call_intrinsic`, `llvm.inline_asm`, and operations with
      `AIE4MatrixStateWriter` are forbidden at every depth: they may clobber
      the resident matrix. `aievec.aie4.matmul` is also forbidden because
      its lowering stages another B tile.
    * Unregistered operations are rejected because their effects are
      unknown. Isolated-from-above operations are rejected because their
      bodies may execute outside the region.
    * The region and row MMACs have unknown memory effects. Generic MLIR
      optimizations must preserve their execution and ordering.

    Lowering stages all 64 B columns with 32 inserts of 1024 bits, commits
    the tile, and inlines the body at the region's position. The commit is
    `staging.to.matrix.I32768` for bf16 and int8, `.I4.I32768` for int4,
    and `.FP32768` for fp8.

    Example:

    ```mlir
    %out = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>,
        %c : vector<4x64xf32> -> vector<4x64xf32> {
    ^bb0(%acc: vector<4x64xf32>):
      %a0 = vector.extract %a[0] : vector<32xbf16> from vector<4x32xbf16>
      %c0 = vector.extract %acc[0] : vector<64xf32> from vector<4x64xf32>
      %r0 = aievec.aie4.mmac bf16 %a0, %c0
          : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
      %n = vector.insert %r0, %acc [0] : vector<64xf32> into vector<4x64xf32>
      aievec.aie4.matrix_yield %n : vector<4x64xf32>
    }
    ```
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $format $rhs `:` type($rhs) (`,` $inits^ `:` type($inits))?
    (`->` type($results)^)? attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def AIEVec_AIE4MatrixYieldOp
    : AIEVec_Op<"aie4.matrix_yield", [Pure, Terminator,
                                      HasParent<"AIE4MatrixRegionOp">]>,
      Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Terminator of aievec.aie4.matrix_region";
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let hasVerifier = 0;
}

def AIEVec_AIE4MmacOp : AIEVec_Op<"aie4.mmac", []>,
                        Arguments<(ins AnyVectorOfNonZeroRank:$lhs,
                            Optional<AnyVectorOfNonZeroRank>:$acc,
                            AIEVec_AIE4MatrixFormatAttr:$format,
                            OptionalAttr<BoolAttr>:$lhs_signed,
                            OptionalAttr<BoolAttr>:$rhs_signed)>,
                        Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE4 one-row MMAC with the enclosing region's matrix";
  let description = [{
    Computes `result[n] = acc[n] + sum_k lhs[k] * B[k][n]`, where B is the
    matrix of the enclosing `aievec.aie4.matrix_region`. There is no B
    operand; the op reads hidden matrix state and is therefore not `Pure`.

    `lhs` is a rank-1 K-element vector; `acc` and `result` are rank-1
    64-element vectors. Element types and numerical behavior follow
    `aievec.aie4.matmul` for the selected format.

    Without `acc`, the hardware replaces the accumulator by zeros (conf bit 0,
    `zero_acc`). This is not the same as adding a +0.0 accumulator for the
    sign of zero.

    Integer formats require `lhs_signed` (X, conf bit 9) and `rhs_signed`
    (Z, conf bit 8). `rhs_signed` must match the region, because int4
    staging converts B according to its signedness.

    All formats set `zero_acc` when `acc` is absent. Lowering uses:

    | format | XLLVM op | additional conf fields |
    |--------|----------|-------------|
    | bf16 | `bf16.I512.ACC2048.mmac` | none |
    | int8 | `I512.ACC2048.mmac` | bmode = int8x8, xSign, zSign |
    | int4 | `I512.ACC2048.mmac` | bmode = int4x4, xSign, zSign |
    | fp8  | `fp8.I512.ACC2048.mmac` | emode = all |
  }];
  let assemblyFormat = [{
    $format $lhs (`,` $acc^)? attr-dict `:` type($lhs) (`,` type($acc)^)?
    `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECAIE4OPS_TD
