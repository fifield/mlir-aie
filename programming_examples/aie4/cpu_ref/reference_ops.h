//===- reference_ops.h - CPU reference compute ops for AIE4 ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// CPU reference implementations for all AIE4b compute operations.
// Used to verify AIE4 kernel outputs in test harnesses.
//
//===----------------------------------------------------------------------===//

#ifndef AIE4_CPU_REF_REFERENCE_OPS_H
#define AIE4_CPU_REF_REFERENCE_OPS_H

#include "fp_types.h"
#include "int_types.h"
#include "mx_types.h"
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==================== Elementwise Multiply (VMAC unit) ====================

static inline void ref_vmul_int8(const int8_t *a, const int8_t *b,
                                 int32_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int32_t)a[i] * (int32_t)b[i];
}

static inline void ref_vmac_int8(const int8_t *a, const int8_t *b,
                                 const int32_t *acc, int32_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int32_t)a[i] * (int32_t)b[i] + acc[i];
}

static inline void ref_vmul_int16(const int16_t *a, const int16_t *b,
                                  int32_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int32_t)a[i] * (int32_t)b[i];
}

static inline void ref_vmac_int16(const int16_t *a, const int16_t *b,
                                  const int32_t *acc, int32_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int32_t)a[i] * (int32_t)b[i] + acc[i];
}

static inline void ref_vmul_int32x16(const int32_t *a, const int16_t *b,
                                     int64_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int64_t)a[i] * (int64_t)b[i];
}

static inline void ref_vmac_int32x16(const int32_t *a, const int16_t *b,
                                     const int64_t *acc, int64_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int64_t)a[i] * (int64_t)b[i] + acc[i];
}

static inline void ref_vmul_int32(const int32_t *a, const int32_t *b,
                                  int64_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int64_t)a[i] * (int64_t)b[i];
}

static inline void ref_vmac_int32(const int32_t *a, const int32_t *b,
                                  const int64_t *acc, int64_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (int64_t)a[i] * (int64_t)b[i] + acc[i];
}

// ==================== Elementwise FP Multiply ====================

static inline void ref_vmul_fp8(const uint8_t *a, const uint8_t *b, float *out,
                                int n) {
  for (int i = 0; i < n; i++)
    out[i] = fp8_e4m3_to_float(a[i]) * fp8_e4m3_to_float(b[i]);
}

static inline void ref_vmac_fp8(const uint8_t *a, const uint8_t *b,
                                const float *acc, float *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = fp8_e4m3_to_float(a[i]) * fp8_e4m3_to_float(b[i]) + acc[i];
}

static inline void ref_vmul_fp16(const uint16_t *a, const uint16_t *b,
                                 float *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = fp16_to_float(a[i]) * fp16_to_float(b[i]);
}

static inline void ref_vmul_bf16(const uint16_t *a, const uint16_t *b,
                                 float *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = bf16_to_float(a[i]) * bf16_to_float(b[i]);
}

static inline void ref_vmac_bf16(const uint16_t *a, const uint16_t *b,
                                 const float *acc, float *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = bf16_to_float(a[i]) * bf16_to_float(b[i]) + acc[i];
}

static inline void ref_vmul_fp32(const float *a, const float *b, float *out,
                                 int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

static inline void ref_vmac_fp32(const float *a, const float *b,
                                 const float *acc, float *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] * b[i] + acc[i];
}

static inline void ref_vmul_int32_fp32(const int32_t *a, const float *b,
                                       float *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = (float)a[i] * b[i];
}

// ==================== Vector Add/Sub ====================

static inline void ref_vadd_int8(const int8_t *a, const int8_t *b, int8_t *out,
                                 int n) {
  for (int i = 0; i < n; i++) {
    int16_t sum = (int16_t)a[i] + (int16_t)b[i];
    if (sum > 127)
      sum = 127;
    if (sum < -128)
      sum = -128;
    out[i] = (int8_t)sum;
  }
}

static inline void ref_vadd_int16(const int16_t *a, const int16_t *b,
                                  int16_t *out, int n) {
  for (int i = 0; i < n; i++) {
    int32_t sum = (int32_t)a[i] + (int32_t)b[i];
    if (sum > 32767)
      sum = 32767;
    if (sum < -32768)
      sum = -32768;
    out[i] = (int16_t)sum;
  }
}

static inline void ref_vadd_int32(const int32_t *a, const int32_t *b,
                                  int32_t *out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}

static inline void ref_vadd_fp32(const float *a, const float *b, float *out,
                                 int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}

// ==================== Matrix Multiply Reference ====================
// C[M,N] += A[M,K] @ B[K,N], column-major B

static inline void ref_matmul_int8(const int8_t *a, const int8_t *b,
                                   int32_t *c, int M, int K, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int32_t acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        acc += (int32_t)a[m * K + k] * (int32_t)b[k * N + n];
      }
      c[m * N + n] = acc;
    }
  }
}

static inline void ref_matmul_int4(const uint8_t *a_packed,
                                   const uint8_t *b_packed, int32_t *c, int M,
                                   int K, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int32_t acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        int8_t av = int4_extract(a_packed, m * K + k);
        int8_t bv = int4_extract(b_packed, k * N + n);
        acc += (int32_t)av * (int32_t)bv;
      }
      c[m * N + n] = acc;
    }
  }
}

static inline void ref_matmul_fp8(const uint8_t *a, const uint8_t *b, float *c,
                                  int M, int K, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        acc += fp8_e4m3_to_float(a[m * K + k]) *
               fp8_e4m3_to_float(b[k * N + n]);
      }
      c[m * N + n] = acc;
    }
  }
}

static inline void ref_matmul_bf16(const uint16_t *a, const uint16_t *b,
                                   float *c, int M, int K, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        acc += bf16_to_float(a[m * K + k]) * bf16_to_float(b[k * N + n]);
      }
      c[m * N + n] = acc;
    }
  }
}

// ==================== MX Format Matrix Multiply ====================
// These operate on packed MX blocks.
// A is row-major, B is column-major (as stored in Z register).
// Block structure: for MXFP8, each 32-element block is 32 mantissa + 1 scale.

static inline void ref_matmul_mxfp8(const uint8_t *a_packed,
                                    const uint8_t *b_packed, float *c, int M,
                                    int K, int N) {
  // Unpack A and B to float, then matmul
  int a_blocks = (M * K + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
  int b_blocks = (K * N + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;

  float a_float[M * K];
  float b_float[K * N];
  (void)a_blocks;
  (void)b_blocks;

  // Unpack A
  for (int blk = 0; blk < (M * K) / MXFP8_BLOCK_SIZE; blk++) {
    const uint8_t *mantissas = a_packed + blk * MXFP8_BLOCK_BYTES;
    uint8_t scale = mantissas[MXFP8_BLOCK_SIZE];
    mxfp8_block_to_float(mantissas, scale, &a_float[blk * MXFP8_BLOCK_SIZE]);
  }

  // Unpack B
  for (int blk = 0; blk < (K * N) / MXFP8_BLOCK_SIZE; blk++) {
    const uint8_t *mantissas = b_packed + blk * MXFP8_BLOCK_BYTES;
    uint8_t scale = mantissas[MXFP8_BLOCK_SIZE];
    mxfp8_block_to_float(mantissas, scale, &b_float[blk * MXFP8_BLOCK_SIZE]);
  }

  // Standard matmul in fp32
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        acc += a_float[m * K + k] * b_float[k * N + n];
      }
      c[m * N + n] = acc;
    }
  }
}

static inline void ref_matmul_mxfp4(const uint8_t *a_packed,
                                    const uint8_t *b_packed, float *c, int M,
                                    int K, int N) {
  float a_float[M * K];
  float b_float[K * N];

  // MXFP4: 16 mantissa bytes + 1 scale byte per 32-element block
  int bytes_per_block = MXFP4_MANTISSA_BYTES + 1;

  for (int blk = 0; blk < (M * K) / MXFP4_BLOCK_SIZE; blk++) {
    const uint8_t *block = a_packed + blk * bytes_per_block;
    uint8_t scale = block[MXFP4_MANTISSA_BYTES];
    mxfp4_block_to_float(block, scale, &a_float[blk * MXFP4_BLOCK_SIZE]);
  }

  for (int blk = 0; blk < (K * N) / MXFP4_BLOCK_SIZE; blk++) {
    const uint8_t *block = b_packed + blk * bytes_per_block;
    uint8_t scale = block[MXFP4_MANTISSA_BYTES];
    mxfp4_block_to_float(block, scale, &b_float[blk * MXFP4_BLOCK_SIZE]);
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        acc += a_float[m * K + k] * b_float[k * N + n];
      }
      c[m * N + n] = acc;
    }
  }
}

static inline void ref_matmul_mxfp6(const uint8_t *a_packed,
                                    const uint8_t *b_packed, float *c, int M,
                                    int K, int N) {
  float a_float[M * K];
  float b_float[K * N];

  int bytes_per_block = MXFP6_MANTISSA_BYTES + 1;

  for (int blk = 0; blk < (M * K) / MXFP6_BLOCK_SIZE; blk++) {
    const uint8_t *block = a_packed + blk * bytes_per_block;
    uint8_t scale = block[MXFP6_MANTISSA_BYTES];
    mxfp6_block_to_float(block, scale, &a_float[blk * MXFP6_BLOCK_SIZE]);
  }

  for (int blk = 0; blk < (K * N) / MXFP6_BLOCK_SIZE; blk++) {
    const uint8_t *block = b_packed + blk * bytes_per_block;
    uint8_t scale = block[MXFP6_MANTISSA_BYTES];
    mxfp6_block_to_float(block, scale, &b_float[blk * MXFP6_BLOCK_SIZE]);
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        acc += a_float[m * K + k] * b_float[k * N + n];
      }
      c[m * N + n] = acc;
    }
  }
}

static inline void ref_matmul_mxint8(const uint8_t *a_packed,
                                     const uint8_t *b_packed, float *c, int M,
                                     int K, int N) {
  float a_float[M * K];
  float b_float[K * N];

  for (int blk = 0; blk < (M * K) / MXINT8_BLOCK_SIZE; blk++) {
    const uint8_t *block = a_packed + blk * MXINT8_BLOCK_BYTES;
    uint8_t scale = block[MXINT8_BLOCK_SIZE];
    mxint8_block_to_float((const int8_t *)block, scale,
                          &a_float[blk * MXINT8_BLOCK_SIZE]);
  }

  for (int blk = 0; blk < (K * N) / MXINT8_BLOCK_SIZE; blk++) {
    const uint8_t *block = b_packed + blk * MXINT8_BLOCK_BYTES;
    uint8_t scale = block[MXINT8_BLOCK_SIZE];
    mxint8_block_to_float((const int8_t *)block, scale,
                          &b_float[blk * MXINT8_BLOCK_SIZE]);
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = c[m * N + n];
      for (int k = 0; k < K; k++) {
        acc += a_float[m * K + k] * b_float[k * N + n];
      }
      c[m * N + n] = acc;
    }
  }
}

// ==================== Sparse Matrix Multiply ====================

static inline void ref_matmul_int8_sparse(const int8_t *a,
                                          const uint8_t *b_sparse, int32_t *c,
                                          int M, int K_dense, int N) {
  // K_dense = 128 (the uncompressed row length)
  // b_sparse is stored column-by-column, 76 bytes per column
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int32_t acc = c[m * N + n];
      int8_t dense_col[128];
      sparse_int8_unpack_column(&b_sparse[n * SPARSE_BYTES_PER_COL], dense_col);
      for (int k = 0; k < K_dense; k++) {
        acc += (int32_t)a[m * K_dense + k] * (int32_t)dense_col[k];
      }
      c[m * N + n] = acc;
    }
  }
}

static inline void ref_matmul_fp8_sparse(const uint8_t *a,
                                         const uint8_t *b_sparse, float *c,
                                         int M, int K_dense, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = c[m * N + n];
      // Unpack sparse column (same layout as int8 sparse, but fp8 mantissas)
      int8_t raw_col[128];
      sparse_int8_unpack_column(&b_sparse[n * SPARSE_BYTES_PER_COL], raw_col);
      for (int k = 0; k < K_dense; k++) {
        acc += fp8_e4m3_to_float(a[m * K_dense + k]) *
               fp8_e4m3_to_float((uint8_t)raw_col[k]);
      }
      c[m * N + n] = acc;
    }
  }
}

// ==================== Comparison Utilities ====================

static inline int ref_compare_int32(const int32_t *expected,
                                    const int32_t *actual, int n, int verbose) {
  int errors = 0;
  (void)verbose;
  for (int i = 0; i < n; i++) {
    if (expected[i] != actual[i])
      errors++;
  }
  return errors;
}

static inline int ref_compare_float(const float *expected, const float *actual,
                                    int n, float rel_tol, float abs_tol) {
  int errors = 0;
  for (int i = 0; i < n; i++) {
    float diff = fabsf(expected[i] - actual[i]);
    float mag = fabsf(expected[i]);
    if (diff > abs_tol && diff > rel_tol * mag) {
      errors++;
    }
  }
  return errors;
}

#ifdef __cplusplus
}
#endif

#endif // AIE4_CPU_REF_REFERENCE_OPS_H
