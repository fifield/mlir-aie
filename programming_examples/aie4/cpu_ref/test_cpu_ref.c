//===- test_cpu_ref.c - Unit tests for CPU reference library -----*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "fp_types.h"
#include "int_types.h"
#include "mx_types.h"
#include "reference_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (cond) {                                                                \
      tests_passed++;                                                          \
    } else {                                                                   \
      tests_failed++;                                                          \
      printf("FAIL: %s (line %d)\n", msg, __LINE__);                           \
    }                                                                          \
  } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                             \
  CHECK(fabsf((a) - (b)) <= (tol), msg)

// ==================== FP8 E4M3 tests ====================

void test_fp8_e4m3(void) {
  // Zero
  CHECK_NEAR(fp8_e4m3_to_float(0x00), 0.0f, 0.0f, "fp8 zero");
  CHECK_NEAR(fp8_e4m3_to_float(0x80), -0.0f, 0.0f, "fp8 neg zero");

  // Max positive: exp=14, mant=7 → 2^7 * 1.875 = 240
  // Actually: exp=14, bias=7, so 2^(14-7) * (1 + 7/8) = 2^7 * 1.875 = 240
  float max_pos = fp8_e4m3_to_float(0x77); // 0_1110_111
  CHECK_NEAR(max_pos, 240.0f, 0.01f, "fp8 max positive (0x77)");

  // exp=15, mant=6 (0x7E) → 2^(15-7) * (1 + 6/8) = 256 * 1.75 = 448
  float big = fp8_e4m3_to_float(0x7E);
  CHECK_NEAR(big, 448.0f, 0.01f, "fp8 448 (0x7E)");

  // NaN
  CHECK(isnan(fp8_e4m3_to_float(0x7F)), "fp8 NaN 0x7F");
  CHECK(isnan(fp8_e4m3_to_float(0xFF)), "fp8 NaN 0xFF");

  // 1.0 = 0_0111_000 = 0x38
  CHECK_NEAR(fp8_e4m3_to_float(0x38), 1.0f, 0.0f, "fp8 1.0");

  // -1.0 = 1_0111_000 = 0xB8
  CHECK_NEAR(fp8_e4m3_to_float(0xB8), -1.0f, 0.0f, "fp8 -1.0");

  // Round-trip
  for (int i = 0; i < 256; i++) {
    if (i == 0x7F || i == 0xFF)
      continue; // NaN
    float f = fp8_e4m3_to_float((uint8_t)i);
    uint8_t back = float_to_fp8_e4m3(f);
    float f2 = fp8_e4m3_to_float(back);
    CHECK_NEAR(f, f2, fabsf(f) * 0.01f + 1e-10f, "fp8 round-trip");
  }
}

// ==================== BF16 tests ====================

void test_bf16(void) {
  CHECK_NEAR(bf16_to_float(0x0000), 0.0f, 0.0f, "bf16 zero");
  CHECK_NEAR(bf16_to_float(0x3F80), 1.0f, 0.0f, "bf16 1.0");
  CHECK_NEAR(bf16_to_float(0xBF80), -1.0f, 0.0f, "bf16 -1.0");
  CHECK_NEAR(bf16_to_float(0x4000), 2.0f, 0.0f, "bf16 2.0");

  // Round-trip
  float vals[] = {0.0f, 1.0f, -1.0f, 3.14f, -0.5f, 100.0f, 1e10f};
  for (int i = 0; i < 7; i++) {
    uint16_t b = float_to_bf16(vals[i]);
    float f = bf16_to_float(b);
    CHECK_NEAR(f, vals[i], fabsf(vals[i]) * 0.02f + 1e-10f,
               "bf16 round-trip");
  }
}

// ==================== FP4 E2M1 tests ====================

void test_fp4_e2m1(void) {
  CHECK_NEAR(fp4_e2m1_to_float(0x0), 0.0f, 0.0f, "fp4 zero");
  CHECK_NEAR(fp4_e2m1_to_float(0x1), 0.5f, 0.0f, "fp4 subnorm 0.5");
  CHECK_NEAR(fp4_e2m1_to_float(0x2), 1.0f, 0.0f, "fp4 1.0");
  CHECK_NEAR(fp4_e2m1_to_float(0x3), 1.5f, 0.0f, "fp4 1.5");
  CHECK_NEAR(fp4_e2m1_to_float(0x4), 2.0f, 0.0f, "fp4 2.0");
  CHECK_NEAR(fp4_e2m1_to_float(0x5), 3.0f, 0.0f, "fp4 3.0");
  CHECK_NEAR(fp4_e2m1_to_float(0x6), 4.0f, 0.0f, "fp4 4.0");
  CHECK_NEAR(fp4_e2m1_to_float(0x7), 6.0f, 0.0f, "fp4 6.0 (max)");
  CHECK_NEAR(fp4_e2m1_to_float(0x8), -0.0f, 0.0f, "fp4 -0");
  CHECK_NEAR(fp4_e2m1_to_float(0xF), -6.0f, 0.0f, "fp4 -6.0");
}

// ==================== FP6 E2M3 tests ====================

void test_fp6_e2m3(void) {
  CHECK_NEAR(fp6_e2m3_to_float(0x00), 0.0f, 0.0f, "fp6 zero");
  CHECK_NEAR(fp6_e2m3_to_float(0x08), 1.0f, 0.0f, "fp6 1.0");
  CHECK_NEAR(fp6_e2m3_to_float(0x1F), 7.5f, 0.0f, "fp6 7.5 (max)");
  CHECK_NEAR(fp6_e2m3_to_float(0x28), -1.0f, 0.0f, "fp6 -1.0");
}

// ==================== INT4 tests ====================

void test_int4(void) {
  uint8_t packed[4];
  memset(packed, 0, 4);

  // Pack and extract
  int4_pack(packed, 0, 3);
  int4_pack(packed, 1, -4);
  int4_pack(packed, 2, 7);
  int4_pack(packed, 3, -8);

  CHECK(int4_extract(packed, 0) == 3, "int4 extract 0");
  CHECK(int4_extract(packed, 1) == -4, "int4 extract 1");
  CHECK(int4_extract(packed, 2) == 7, "int4 extract 2");
  CHECK(int4_extract(packed, 3) == -8, "int4 extract 3");
}

// ==================== MX format tests ====================

void test_mxfp8(void) {
  float input[32];
  for (int i = 0; i < 32; i++)
    input[i] = (float)(i - 16) * 0.5f;

  uint8_t mantissas[32];
  uint8_t scale;
  float_to_mxfp8_block(input, mantissas, &scale);

  float output[32];
  mxfp8_block_to_float(mantissas, scale, output);

  int errors = ref_compare_float(input, output, 32, 0.1f, 0.5f);
  CHECK(errors == 0, "mxfp8 round-trip");
}

void test_mxfp4(void) {
  float input[32];
  for (int i = 0; i < 32; i++)
    input[i] = (float)(i - 16) * 0.25f;

  uint8_t packed[16];
  uint8_t scale;
  float_to_mxfp4_block(input, packed, &scale);

  float output[32];
  mxfp4_block_to_float(packed, scale, output);

  // MXFP4 has very low precision, so use generous tolerance
  int errors = ref_compare_float(input, output, 32, 0.5f, 1.0f);
  CHECK(errors <= 2, "mxfp4 round-trip (generous tolerance)");
}

void test_mxint8(void) {
  float input[8] = {0.5f, -0.5f, 1.0f, -1.0f, 0.25f, -0.25f, 1.5f, -1.5f};

  int8_t elements[8];
  uint8_t scale;
  float_to_mxint8_block(input, elements, &scale);

  float output[8];
  mxint8_block_to_float(elements, scale, output);

  int errors = ref_compare_float(input, output, 8, 0.05f, 0.02f);
  CHECK(errors == 0, "mxint8 round-trip");
}

// ==================== Reference ops tests ====================

void test_ref_vmul_int8(void) {
  int8_t a[] = {1, 2, 3, -4, 5, 6, 7, -8};
  int8_t b[] = {2, 3, 4, 5, -6, 7, 8, 9};
  int32_t out[8];
  ref_vmul_int8(a, b, out, 8);
  CHECK(out[0] == 2, "vmul_int8[0]");
  CHECK(out[1] == 6, "vmul_int8[1]");
  CHECK(out[3] == -20, "vmul_int8[3]");
  CHECK(out[7] == -72, "vmul_int8[7]");
}

void test_ref_matmul_int8(void) {
  // 2x2 * 2x2 = 2x2
  int8_t a[] = {1, 2, 3, 4};   // [[1,2],[3,4]]
  int8_t b[] = {5, 6, 7, 8};   // [[5,6],[7,8]]
  int32_t c[] = {0, 0, 0, 0};
  ref_matmul_int8(a, b, c, 2, 2, 2);
  CHECK(c[0] == 19, "matmul_int8[0,0]");  // 1*5 + 2*7 = 19
  CHECK(c[1] == 22, "matmul_int8[0,1]");  // 1*6 + 2*8 = 22
  CHECK(c[2] == 43, "matmul_int8[1,0]");  // 3*5 + 4*7 = 43
  CHECK(c[3] == 50, "matmul_int8[1,1]");  // 3*6 + 4*8 = 50
}

// ==================== Sparse tests ====================

void test_sparse_unpack(void) {
  // Create a simple sparse column: 32 groups * 2 = 64 mantissa bytes
  // All encoded as 000 (select x[0], x[1])
  uint8_t sparse[76];
  memset(sparse, 0, 76);

  // Fill mantissa bytes with values 0..63
  for (int i = 0; i < 64; i++)
    sparse[i] = (uint8_t)(i + 1);

  // Sparsity mask: all zeros = encoding 000 = select x[0],x[1]
  // This means for each quartet, positions 0 and 1 get the mantissa values

  int8_t dense[128];
  sparse_int8_unpack_column(sparse, dense);

  // Group 0: dense[0]=mantissa[0]=1, dense[1]=mantissa[1]=2, dense[2]=0, dense[3]=0
  CHECK(dense[0] == 1, "sparse[0]");
  CHECK(dense[1] == 2, "sparse[1]");
  CHECK(dense[2] == 0, "sparse[2]");
  CHECK(dense[3] == 0, "sparse[3]");
}

// ==================== Main ====================

int main(void) {
  printf("Running CPU reference library tests...\n\n");

  test_fp8_e4m3();
  test_bf16();
  test_fp4_e2m1();
  test_fp6_e2m3();
  test_int4();
  test_mxfp8();
  test_mxfp4();
  test_mxint8();
  test_ref_vmul_int8();
  test_ref_matmul_int8();
  test_sparse_unpack();

  printf("\n%d tests passed, %d tests failed\n", tests_passed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}
