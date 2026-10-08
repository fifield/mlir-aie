//===-- Passes.td - AIE vector pass definition file --------*- tablegen -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains definitions for passes within the AIEVec/ directory.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
#define AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SplitVectorLoadUpsChains : Pass<"aievec-split-load-ups-chains"> {
  let summary = "Split 1024-bit vector loads and stores whose halves are "
                "widened or narrowed separately";
  let description = [{
    On AIE2P, splits a 1024-bit vector access into two 512-bit accesses when
    the value is only used, or produced, one half at a time:
    - vector.load feeding aievec.ups (v64i16 -> v64i32);
    - aievec.srs (v64i32 -> v64i16) feeding vector.store;
    - vector.load of v64bf16 feeding only the LHS of bf16 8x8x8
      aievec.matmul_aie2p ops, whose lowering widens the LHS in two halves.
    Each half then gets its own memory access instead of being extracted from
    the full-width value with shuffles.
  }];
  let constructor = "xilinx::aievec::createSplitVectorLoadUpsChainsPass()";
  let dependentDialects = ["mlir::vector::VectorDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::affine::AffineDialect",
                           "xilinx::aievec::AIEVecDialect"];
}

def AIEVecGrantContractAfn : Pass<"aievec-grant-contract-afn"> {
  let summary =
      "Add `afn` to the fastmath flags of floating-point vector.contract ops";
  let description = [{
    Adds `afn` to the fastmath flags of every `vector.contract` whose
    accumulator has a floating-point element type, keeping the flags it
    already has. Integer contractions are left alone.

    The AIE4 conversion treats `afn` on a floating-point contraction as an
    opt-in to MMAC numerics (see `aievec.aie4.matmul`), including hardware
    reduction order, rounding, and subnormal handling. This is an AIE4
    lowering convention; `afn` does not generally permit reassociation.
    This pass records that opt-in in the IR for sources that cannot
    annotate their contractions directly.

    Off by default. The `contract-afn` option of the
    `canonicalize-vector-for-aievec` and `convert-vector-to-aievec`
    pipelines runs it first.
  }];
  let constructor = "xilinx::aievec::createGrantContractAfnPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::vector::VectorDialect"];
}

def AIEVecAIE4CheckStaging
    : Pass<"aievec-aie4-check-staging", "mlir::ModuleOp"> {
  let summary = "Check AIE4 flat staging/commit provenance and matmul "
                "co-existence";
  let description = [{
    Checks programs that use the flat AIE4 matrix-state ops
    (`aievec.aie4.stage`, `stage_stream`, `commit`, `mmac_active`).

    Part 1 (co-existence) rejects an `aievec.aie4.matmul` in an execution
    scope (an `aie.core`, a function, or another isolated op) that also
    holds flat ops: generic transforms may move the pure matmul between a
    prime and its commit.

    Part 2 (provenance) runs a must-defined dataflow analysis over each
    scope that holds a flat op. Every `commit` must read 32 staging slots
    written with its format (and int4 signedness) on every path, and every
    `mmac_active` must read a matrix committed with its format on every
    path. Calls, inline assembly, unregistered ops and raw XLLVM staging or
    commit ops clobber the state. Only `scf.for`, `scf.if` and
    `aievec.aie4.matrix_region` may surround matrix-state ops. Calls count
    as matrix-state ops, so in a scope with flat ops a call inside, e.g.,
    `scf.while` is rejected too. Scopes without flat ops are not analysed.

    aiecc runs part 1 before vectorization and both parts around the
    matmul expansion, as errors.
  }];
  let options =
      [Option<"phase", "phase", "std::string", /*default=*/"\"all\"",
              "\"coexistence\" (part 1), \"provenance\" (part 2) or \"all\" "
              "(both, part 1 first)">,
       Option<"strict", "strict", "bool", /*default=*/"true",
              "Report findings as errors and fail the pass; false reports "
              "warnings">,
  ];
  let dependentDialects = ["xilinx::aievec::AIEVecDialect"];
}

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
