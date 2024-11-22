//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.
//
//===----------------------------------------------------------------------===//

// #include <stdio.h>
#include <stdint.h>

extern float vecA[256];
extern int8_t vecB[256];

#define LOCK_OFFSET 48

extern "C" void core_1_3() {

  set_rnd( rnd_sym_inf );
  set_sat();
  
  acquire_greater_equal(LOCK_OFFSET + 3, 1);

  v32accfloat * pA = ( v32accfloat * ) vecA;
  v32float8 * pO = ( v32float8 * ) vecB;

  for ( int i = 0; i < 256/8; i++ ) {
    v32float8 result = to_v32float8( *pA++ , 0 );   // Second parameter is a mystery ...
    *pO++ = result;
  }


  release(LOCK_OFFSET + 5, 1);
}
