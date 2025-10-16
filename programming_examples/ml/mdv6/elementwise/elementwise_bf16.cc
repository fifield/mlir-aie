//===- elementwise_bf16.cc ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

//*****************************************************************************
// Element-wise Addition - bfloat16
// out = a + b
//*****************************************************************************
void add_bf16_scalar(bfloat16 *a, bfloat16 *b, bfloat16 *out, const int32_t size) {
  event0();

  for (int i = 0; i < size; i++) {
    float a_val = (float)a[i];
    float b_val = (float)b[i];
    out[i] = (bfloat16)(a_val + b_val);
  }

  event1();
}

//*****************************************************************************
// Element-wise Maximum - bfloat16
// out = max(a, b)
//*****************************************************************************
void max_bf16_scalar(bfloat16 *a, bfloat16 *b, bfloat16 *out, const int32_t size) {
  event0();

  for (int i = 0; i < size; i++) {
    float a_val = (float)a[i];
    float b_val = (float)b[i];
    out[i] = (bfloat16)((a_val > b_val) ? a_val : b_val);
  }

  event1();
}

//*****************************************************************************
// Element-wise Multiplication - bfloat16
// out = a * b
//*****************************************************************************
void mul_bf16_scalar(bfloat16 *a, bfloat16 *b, bfloat16 *out, const int32_t size) {
  event0();

  for (int i = 0; i < size; i++) {
    float a_val = (float)a[i];
    float b_val = (float)b[i];
    out[i] = (bfloat16)(a_val * b_val);
  }

  event1();
}

//*****************************************************************************
// Scalar Addition - bfloat16
// out = a + scalar
//*****************************************************************************
void add_scalar_bf16_scalar(bfloat16 *a, float scalar, bfloat16 *out, const int32_t size) {
  event0();

  for (int i = 0; i < size; i++) {
    float a_val = (float)a[i];
    out[i] = (bfloat16)(a_val + scalar);
  }

  event1();
}

extern "C" {

void add_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *out, int32_t size) {
  add_bf16_scalar(a, b, out, size);
}

void max_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *out, int32_t size) {
  max_bf16_scalar(a, b, out, size);
}

void mul_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *out, int32_t size) {
  mul_bf16_scalar(a, b, out, size);
}

void add_scalar_bf16(bfloat16 *a, float scalar, bfloat16 *out, int32_t size) {
  add_scalar_bf16_scalar(a, scalar, out, size);
}

} // extern "C"
