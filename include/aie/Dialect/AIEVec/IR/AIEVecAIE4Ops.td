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

    A floating-point `vector.contract` converts to this op only if it
    carries `afn` (`fast` includes it). The AIE4 conversion treats `afn` on
    a floating-point contraction as an opt-in to the MMAC numerical behavior
    below, including its reduction order, rounding, and subnormal handling.
    This is an AIE4 lowering convention; `afn` does not generally permit
    reassociation. `reassoc`, `contract`, and `nsz` alone do not enable this
    conversion. Integer contractions need no flag: their wrapping arithmetic
    is the hardware's.

    Numerical behavior:

    * bf16 (measured on NPU3). Products are exact; subnormal multiplicands
      are accepted. The 32 products form two blocks of 16. Within each block,
      products are aligned to the largest product exponent, individually
      rounded to nearest even at 23 fractional binary digits, and then summed
      without normalization. The two block sums and the accumulator are
      aligned to their largest exponent (a block that cancels to zero still
      counts), individually rounded the same way, summed, and rounded to f32.
      Subnormal accumulator inputs and results flush to +0; -0 is never
      produced; a NaN result can be 0x7F800001 or 0xFF800001. Rewrites must
      preserve this:
      `matmul(0, b, acc)` cannot generally fold to `acc`.
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
      `AIE4StagingWriter` or `AIE4MatrixCommit` are forbidden at every
      depth: they may clobber the resident matrix. `aievec.aie4.matmul` is
      also forbidden because its lowering stages another B tile, and
      `aievec.aie4.mmac_active` because bodies use `aievec.aie4.mmac`.
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
    `zero_acc`). For bf16 this equals a +0.0 accumulator, since the bf16
    MMAC never produces -0; for fp8 the sign of zero is not characterized.

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

//===----------------------------------------------------------------------===//
// Flat layer: program-ordered matrix state
//===----------------------------------------------------------------------===//
//
// Unknown memory effects preserve program order for staging (za) and active
// matrix (zb) access. The traits are shared with the corresponding XLLVM ops.

def AIEVec_AIE4StagingWriterTrait : NativeOpTrait<"AIE4StagingWriter"> {
  let cppNamespace = "::xilinx::xllvm";
}
def AIEVec_AIE4MatrixCommitTrait : NativeOpTrait<"AIE4MatrixCommit"> {
  let cppNamespace = "::xilinx::xllvm";
}
def AIEVec_AIE4MatrixReaderTrait : NativeOpTrait<"AIE4MatrixReader"> {
  let cppNamespace = "::xilinx::xllvm";
}

def AIEVec_AIE4StageOp
    : AIEVec_Op<"aie4.stage", [AIEVec_AIE4StagingWriterTrait]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$chunk, I32:$slot,
          AIEVec_AIE4MatrixFormatAttr:$format,
          OptionalAttr<BoolAttr>:$rhs_signed)> {
  let summary = "AIE4 write one 1024-bit chunk to matrix staging slot `slot`";
  let description = [{
    Writes `chunk` to staging slot `slot` at this point in program order.
    The slot keeps its value until the next write to it or until another
    operation clobbers the staging matrix. No enclosing region is required;
    `aievec.aie4.commit` copies the staged image to the active matrix.

    A slot holds two 512-bit B columns: column 2i occupies bits 511:0 and
    column 2i+1 bits 1023:512, where i is the slot. 32 slots fill one dense
    B tile. A runtime slot outside [0, 32) has undefined behavior.

    `format` and `rhs_signed` select the crIntMMode written by the insert.
    Integer formats require `rhs_signed`; signed int4 staging converts two's
    complement to sign/magnitude, so the MMAC Z signedness must match it.
    Floating formats reject `rhs_signed`.

    | format | `chunk` |
    |--------|---------|
    | bf16 | `vector<64xbf16>` |
    | int8 | `vector<128xi8>` |
    | int4 | `vector<256xi4>`, or its byte image `vector<128xi8>` (element 2j in the low nibble of byte j) |
    | fp8  | `vector<128xf8E4M3FN>`, or its byte image `vector<128xi8>` |

    Example:

    ```mlir
    aievec.aie4.stage int8 %chunk, %slot {rhs_signed = true}
        : vector<128xi8>, i32
    ```
  }];
  let assemblyFormat =
      "$format $chunk `,` $slot attr-dict `:` type($chunk) `,` type($slot)";
  let hasVerifier = 1;
}

def AIEVec_AIE4StageStreamOp
    : AIEVec_Op<"aie4.stage_stream", [AIEVec_AIE4StagingWriterTrait]>,
      Arguments<(ins I32:$port, I32:$slot, AIEVec_AIE4MatrixFormatAttr:$format,
          OptionalAttr<BoolAttr>:$rhs_signed)> {
  let summary =
      "AIE4 pop 1024 bits from input stream `port` into staging slot `slot`";
  let description = [{
    Performs a blocking direct-mode read of 1024 bits from core input stream
    `port` (0 or 1) and writes them to staging slot `slot`, with the slot
    layout, `format` and `rhs_signed` rules of `aievec.aie4.stage`.

    Sequential reads of one stream are ordered by their effects. Mixing
    these reads with VLD.POP FIFO reads of the same stream before TLAST is
    invalid. A short transfer that ends in TLAST is zero-padded.

    Example:

    ```mlir
    aievec.aie4.stage_stream int8 %port, %slot {rhs_signed = true}
        : i32, i32
    ```
  }];
  let assemblyFormat = [{
    $format $port `,` $slot attr-dict `:` type($port) `,`
    type($slot)
  }];
  let hasVerifier = 1;
}

def AIEVec_AIE4CommitOp
    : AIEVec_Op<"aie4.commit", [AIEVec_AIE4MatrixCommitTrait]>,
      Arguments<(ins AIEVec_AIE4MatrixFormatAttr:$format,
          OptionalAttr<BoolAttr>:$rhs_signed)> {
  let summary = "AIE4 copy the staging image to the active matrix";
  let description = [{
    Copies all 32 staging slots to the active matrix read by
    `aievec.aie4.mmac_active`. The commit is `staging.to.matrix.I32768` for
    bf16 and int8, `.I4.I32768` for int4, and `.FP32768` for fp8.

    Integer formats require `rhs_signed`, which must be the signedness the
    slots were staged with. It does not change the lowering.

    Example:

    ```mlir
    aievec.aie4.commit int8 {rhs_signed = true}
    ```
  }];
  let assemblyFormat = "$format attr-dict";
  let hasVerifier = 1;
}

def AIEVec_AIE4MmacActiveOp
    : AIEVec_Op<"aie4.mmac_active", [AttrSizedOperandSegments,
                                     AIEVec_AIE4MatrixReaderTrait]>,
      Arguments<(ins AnyVectorOfNonZeroRank:$lhs,
          Optional<AnyVectorOfNonZeroRank>:$acc, Optional<I1>:$zero_if,
          AIEVec_AIE4MatrixFormatAttr:$format,
          OptionalAttr<BoolAttr>:$lhs_signed,
          OptionalAttr<BoolAttr>:$rhs_signed)>,
      Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE4 one-row MMAC with the currently committed matrix";
  let description = [{
    Computes `result[n] = acc[n] + sum_k lhs[k] * B[k][n]`, where B is the
    matrix most recently committed by `aievec.aie4.commit` (or a raw XLLVM
    commit) in program order, i.e. row x zb. Operand types and numerics are
    those of `aievec.aie4.mmac` and `aievec.aie4.matmul` for `format`.

    Without `acc`, the hardware replaces the accumulator by zeros
    (`zero_acc`). With `zero_if %z`, `%z = true` replaces `acc` by zeros at
    run time; this applies only to the integer formats and requires `acc`.

    If the committed matrix does not have this op's format and, for int4,
    B signedness, the results are unspecified but fixed values, not poison.

    The op has no structural parent requirement. `matrix_region` bodies
    reject it; region bodies use `aievec.aie4.mmac`.

    Example:

    ```mlir
    %r = aievec.aie4.mmac_active int8 %x, %acc zero_if %first
        {lhs_signed = true, rhs_signed = true}
        : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    ```
  }];
  let assemblyFormat = [{
    $format $lhs (`,` $acc^)? (`zero_if` $zero_if^)? attr-dict `:` type($lhs)
    (`,` type($acc)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECAIE4OPS_TD
