//===- target-parsing-aie4.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aie4 and aie4b both resolve to aie4b
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4" 2>%t.aie4-canon.err
// RUN: not aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" 2>%t.aie4b-canon.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4-canon.err > %t.aie4-canon-normalized.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4b-canon.err > %t.aie4b-canon-normalized.err
// RUN: diff %t.aie4-canon-normalized.err %t.aie4b-canon-normalized.err

// RUN: not aie-opt %s --lower-vector-to-aievec="aie-target=aie4" 2>%t.aie4-lower.err
// RUN: not aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" 2>%t.aie4b-lower.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4-lower.err > %t.aie4-lower-normalized.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4b-lower.err > %t.aie4b-lower-normalized.err
// RUN: diff %t.aie4-lower-normalized.err %t.aie4b-lower-normalized.err

// RUN: not aie-opt %s --optimize-aievec="aie-target=aie4" 2>%t.aie4-opt.err
// RUN: not aie-opt %s --optimize-aievec="aie-target=aie4b" 2>%t.aie4b-opt.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4-opt.err > %t.aie4-opt-normalized.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4b-opt.err > %t.aie4b-opt-normalized.err
// RUN: diff %t.aie4-opt-normalized.err %t.aie4b-opt-normalized.err

// RUN: not aie-opt %s --convert-vector-to-aievec="aie-target=aie4" 2>%t.aie4-conv.err
// RUN: not aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" 2>%t.aie4b-conv.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4-conv.err > %t.aie4-conv-normalized.err
// RUN: sed -E 's/target aie4b?/target AIE4/' %t.aie4b-conv.err > %t.aie4b-conv-normalized.err
// RUN: diff %t.aie4-conv-normalized.err %t.aie4b-conv-normalized.err

func.func @f() {
  return
}
