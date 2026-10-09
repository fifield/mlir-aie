//===- chess_intrinsic_wrapper.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
/// This file contains symbols that are used in Chess compilation to implement
/// intrinsics.  It tries to solve the problem that the Chess compiler does not
/// implement 'standard' llvm intrinsics (which contain '.') as a separator.
/// Rather than implement a fragile mechanism that relies on a particular
/// encoding this file is compiled with the Chess compiler to generate the
/// proper encoding and then this file is linked with code generated with
/// llvm-style intrinsics. Note that the Chess frontend replaces '.' with '___'
/// when parsing .ll code containing standard intrinsic names, so these symbols
/// are defined that way.

extern "C" void llvm___aie4___acquire(unsigned id, unsigned val) {
  acquire_equal(id, val);
}
extern "C" void llvm___aie4___release(unsigned id, unsigned val) {
  release(id, val);
}
extern "C" void llvm___aie___event0() { event0(); }
extern "C" void llvm___aie___event1() { event1(); }

// Map the Peano register IDs emitted by core lowering to Chess setters.
extern "C" void llvm___aie4___set___ctrl___reg(int reg, int val) {
  switch (reg) {
  case 24: // crVecRnd
    set_vec_rnd(val);
    break;
  case 45: // crSat
    set_satmode(val);
    break;
  case 46: // crRnd
    set_rnd(val);
    break;
  case 53: // crF2BConvSat
    set_fp2bfp_conv_sat_conf(val);
    break;
  default:
    break;
  }
}
