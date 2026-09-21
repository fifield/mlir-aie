//===- mx_types.h - OCP MicroXcaling format utilities --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// CPU reference for OCP MicroXcaling (MX) block floating-point formats
// as used by AIE4b (according to the architecture specification):
//
//   MXFP8:  block_size=32, E8M0 shared exponent, E4M3 elements
//   MXFP6:  block_size=32, E8M0 shared exponent, E2M3 elements
//   MXFP4:  block_size=32, E8M0 shared exponent, E2M1 elements
//   MXINT8: block_size=8,  E8M0 shared exponent, S8 elements (6 frac bits)
//
// Memory layout for MXFP8/MXFP6/MXFP4 on AIE4:
//   Per 128-element vector (4 blocks of 32):
//     MXFP8: 128 mantissa bytes + 4 scale bytes = 132 bytes
//     MXFP4: 64 mantissa bytes (packed) + 4 scale bytes = 68 bytes
//     MXFP6: 96 mantissa bytes (packed) + 4 scale bytes = 100 bytes
//
// E8M0 shared exponent: unsigned 8-bit, bias=127, value = 2^(exp - 127)
//   exp=0 means scale = 2^(-127) (underflow), exp=255 means NaN scale.
//
//===----------------------------------------------------------------------===//

#ifndef AIE4_CPU_REF_MX_TYPES_H
#define AIE4_CPU_REF_MX_TYPES_H

#include "fp_types.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==================== E8M0 shared exponent ====================
// Unsigned 8-bit exponent with bias 127.
// Scale = 2^(exp - 127). exp=0 → 2^-127, exp=255 → NaN.

static inline float e8m0_to_scale(uint8_t exp) {
  if (exp == 255)
    return NAN;
  return exp2f((float)exp - 127.0f);
}

static inline uint8_t float_to_e8m0(float max_abs) {
  if (max_abs == 0.0f || isnan(max_abs))
    return 127; // scale = 1.0
  int exp = (int)ceilf(log2f(max_abs));
  int biased = exp + 127;
  if (biased < 0)
    biased = 0;
  if (biased > 254)
    biased = 254;
  return (uint8_t)biased;
}

// ==================== MXFP8 ====================
// Block of 32 E4M3 elements + 1 E8M0 scale byte.
// value[i] = fp8_e4m3_to_float(element[i]) * 2^(scale - 127)

#define MXFP8_BLOCK_SIZE 32
#define MXFP8_BLOCK_BYTES (MXFP8_BLOCK_SIZE + 1) // 33 bytes per block

static inline void mxfp8_block_to_float(const uint8_t *mantissas,
                                        uint8_t shared_exp, float *out) {
  float scale = e8m0_to_scale(shared_exp);
  for (int i = 0; i < MXFP8_BLOCK_SIZE; i++) {
    out[i] = fp8_e4m3_to_float(mantissas[i]) * scale;
  }
}

static inline void float_to_mxfp8_block(const float *in, uint8_t *mantissas,
                                        uint8_t *shared_exp) {
  // Find max absolute value to determine shared exponent
  float max_abs = 0.0f;
  for (int i = 0; i < MXFP8_BLOCK_SIZE; i++) {
    float a = fabsf(in[i]);
    if (a > max_abs)
      max_abs = a;
  }

  *shared_exp = float_to_e8m0(max_abs);
  float inv_scale = exp2f(127.0f - (float)*shared_exp);

  for (int i = 0; i < MXFP8_BLOCK_SIZE; i++) {
    mantissas[i] = float_to_fp8_e4m3(in[i] * inv_scale);
  }
}

// ==================== MXFP6 ====================
// Block of 32 E2M3 elements (6 bits each) + 1 E8M0 scale byte.
// Packed: 32 * 6 = 192 bits = 24 bytes of mantissa data.
// value[i] = fp6_e2m3_to_float(element[i]) * 2^(scale - 127)
// Internally processed as MXFP8 by AIE4b (upconverted to E4M3).

#define MXFP6_BLOCK_SIZE 32
#define MXFP6_MANTISSA_BYTES 24 // 32 * 6 bits = 192 bits = 24 bytes

static inline uint8_t mxfp6_extract(const uint8_t *packed, int index) {
  int bit_offset = index * 6;
  int byte_idx = bit_offset >> 3;
  int bit_pos = bit_offset & 7;
  uint16_t bits =
      (uint16_t)packed[byte_idx] | ((uint16_t)packed[byte_idx + 1] << 8);
  return (uint8_t)((bits >> bit_pos) & 0x3F);
}

static inline void mxfp6_pack(uint8_t *packed, int index, uint8_t value) {
  int bit_offset = index * 6;
  int byte_idx = bit_offset >> 3;
  int bit_pos = bit_offset & 7;
  uint16_t mask = ~((uint16_t)0x3F << bit_pos);
  uint16_t bits =
      (uint16_t)packed[byte_idx] | ((uint16_t)packed[byte_idx + 1] << 8);
  bits = (bits & mask) | ((uint16_t)(value & 0x3F) << bit_pos);
  packed[byte_idx] = (uint8_t)bits;
  packed[byte_idx + 1] = (uint8_t)(bits >> 8);
}

static inline void mxfp6_block_to_float(const uint8_t *packed_mantissas,
                                        uint8_t shared_exp, float *out) {
  float scale = e8m0_to_scale(shared_exp);
  for (int i = 0; i < MXFP6_BLOCK_SIZE; i++) {
    uint8_t elem = mxfp6_extract(packed_mantissas, i);
    out[i] = fp6_e2m3_to_float(elem) * scale;
  }
}

static inline void float_to_mxfp6_block(const float *in,
                                        uint8_t *packed_mantissas,
                                        uint8_t *shared_exp) {
  float max_abs = 0.0f;
  for (int i = 0; i < MXFP6_BLOCK_SIZE; i++) {
    float a = fabsf(in[i]);
    if (a > max_abs)
      max_abs = a;
  }

  *shared_exp = float_to_e8m0(max_abs);
  float inv_scale = exp2f(127.0f - (float)*shared_exp);

  memset(packed_mantissas, 0, MXFP6_MANTISSA_BYTES);
  for (int i = 0; i < MXFP6_BLOCK_SIZE; i++) {
    mxfp6_pack(packed_mantissas, i, float_to_fp6_e2m3(in[i] * inv_scale));
  }
}

// ==================== MXFP4 ====================
// Block of 32 E2M1 elements (4 bits each) + 1 E8M0 scale byte.
// Packed: 32 * 4 = 128 bits = 16 bytes of mantissa data.
// value[i] = fp4_e2m1_to_float(element[i]) * 2^(scale - 127)
// Internally processed as MXFP8 by AIE4b (upconverted to E4M3).
//
// AIE4 memory layout per 128 elements (from fp4_matmul.cc):
//   68 bytes = 64 packed mantissa bytes + 4 scale bytes

#define MXFP4_BLOCK_SIZE 32
#define MXFP4_MANTISSA_BYTES 16 // 32 * 4 bits = 128 bits = 16 bytes

static inline uint8_t mxfp4_extract(const uint8_t *packed, int index) {
  uint8_t byte = packed[index >> 1];
  return (index & 1) ? (byte >> 4) : (byte & 0x0F);
}

static inline void mxfp4_pack(uint8_t *packed, int index, uint8_t value) {
  int byte_idx = index >> 1;
  if (index & 1) {
    packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((value & 0x0F) << 4);
  } else {
    packed[byte_idx] = (packed[byte_idx] & 0xF0) | (value & 0x0F);
  }
}

static inline void mxfp4_block_to_float(const uint8_t *packed_nibbles,
                                        uint8_t shared_exp, float *out) {
  float scale = e8m0_to_scale(shared_exp);
  for (int i = 0; i < MXFP4_BLOCK_SIZE; i++) {
    uint8_t elem = mxfp4_extract(packed_nibbles, i);
    out[i] = fp4_e2m1_to_float(elem) * scale;
  }
}

static inline void float_to_mxfp4_block(const float *in,
                                        uint8_t *packed_nibbles,
                                        uint8_t *shared_exp) {
  float max_abs = 0.0f;
  for (int i = 0; i < MXFP4_BLOCK_SIZE; i++) {
    float a = fabsf(in[i]);
    if (a > max_abs)
      max_abs = a;
  }

  *shared_exp = float_to_e8m0(max_abs);
  float inv_scale = exp2f(127.0f - (float)*shared_exp);

  memset(packed_nibbles, 0, MXFP4_MANTISSA_BYTES);
  for (int i = 0; i < MXFP4_BLOCK_SIZE; i++) {
    mxfp4_pack(packed_nibbles, i, float_to_fp4_e2m1(in[i] * inv_scale));
  }
}

// ==================== MXINT8 ====================
// Block of 8 signed int8 elements (6 fractional bits) + 1 E8M0 scale byte.
// value[i] = (int8_t)element[i] / 64.0f * 2^(scale - 127)
// Max negative = -2 (0x80 = -128, -128/64 = -2.0), asymmetric range.
// The ISA defines: "MXINT8 with element type S8 (note 6 fractional bits)"

#define MXINT8_BLOCK_SIZE 8
#define MXINT8_BLOCK_BYTES (MXINT8_BLOCK_SIZE + 1) // 9 bytes per block

static inline void mxint8_block_to_float(const int8_t *elements,
                                         uint8_t shared_exp, float *out) {
  float scale = e8m0_to_scale(shared_exp);
  for (int i = 0; i < MXINT8_BLOCK_SIZE; i++) {
    out[i] = ((float)elements[i] / 64.0f) * scale;
  }
}

static inline void float_to_mxint8_block(const float *in, int8_t *elements,
                                         uint8_t *shared_exp) {
  float max_abs = 0.0f;
  for (int i = 0; i < MXINT8_BLOCK_SIZE; i++) {
    float a = fabsf(in[i]);
    if (a > max_abs)
      max_abs = a;
  }

  *shared_exp = float_to_e8m0(max_abs);
  float inv_scale = exp2f(127.0f - (float)*shared_exp);

  for (int i = 0; i < MXINT8_BLOCK_SIZE; i++) {
    float scaled = in[i] * inv_scale * 64.0f;
    int val = (int)roundf(scaled);
    if (val > 127)
      val = 127;
    if (val < -128)
      val = -128;
    elements[i] = (int8_t)val;
  }
}

#ifdef __cplusplus
}
#endif

#endif // AIE4_CPU_REF_MX_TYPES_H
