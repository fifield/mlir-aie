//===- fp_types.h - Floating-point type conversions for AIE4 ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// CPU reference implementations for all AIE4 floating-point formats:
//   - FP8 E4M3 (OCP standard, sign:1 + exp:4 + mant:3, bias=7)
//   - FP16 IEEE (sign:1 + exp:5 + mant:10, bias=15)
//   - BF16 (sign:1 + exp:8 + mant:7, bias=127)
//   - FP6 E2M3 (sign:1 + exp:2 + mant:3, bias=1, no inf/NaN)
//   - FP4 E2M1 (sign:1 + exp:2 + mant:1, bias=1, no inf/NaN)
//
//===----------------------------------------------------------------------===//

#ifndef AIE4_CPU_REF_FP_TYPES_H
#define AIE4_CPU_REF_FP_TYPES_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==================== FP8 E4M3 (OCP standard) ====================
// Used by AIE4 VMAC fp8 and MMAC fp8 modes.
// sign(1) + exp(4) + mant(3), bias = 7
// No infinity. NaN = 0x7F and 0xFF (exp=0xF, mant=0x7).
// Subnormals clamped to zero on AIE4 MAC input.
// Max positive = 448.0 (exp=14, mant=7: 2^7 * 1.875)
// Min positive normal = 2^-6 = 0.015625
// Min positive subnormal = 2^-9 = 0.001953125

static inline float fp8_e4m3_to_float(uint8_t x) {
  uint32_t sign = (x >> 7) & 1;
  uint32_t exp = (x >> 3) & 0x0F;
  uint32_t mant = x & 0x07;

  if (exp == 0x0F && mant == 0x07)
    return NAN;

  float result;
  if (exp == 0) {
    // Subnormal: value = (-1)^s * 2^(1-bias) * (mant/8)
    result = ldexpf((float)mant / 8.0f, 1 - 7);
  } else {
    // Normal: value = (-1)^s * 2^(exp-bias) * (1 + mant/8)
    result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
  }
  return sign ? -result : result;
}

static inline uint8_t float_to_fp8_e4m3(float x) {
  if (isnan(x))
    return 0x7F;

  uint32_t bits;
  memcpy(&bits, &x, 4);
  uint32_t sign = (bits >> 31) & 1;
  int32_t fp32_exp = ((bits >> 23) & 0xFF) - 127;

  if (x == 0.0f)
    return (uint8_t)(sign << 7);

  float abs_x = fabsf(x);

  // Clamp to max representable: exp=15, mant=6 → 448.0
  if (abs_x > 448.0f)
    return (uint8_t)((sign << 7) | (15 << 3) | 6); // 0x7E / 0xFE

  // Handle subnormals (exp would be <= 0)
  if (fp32_exp < -6) {
    if (fp32_exp < -9)
      return (uint8_t)(sign << 7);
    int mant = (int)roundf(abs_x * 512.0f); // abs_x / 2^-9
    if (mant > 7)
      mant = 7;
    if (mant <= 0)
      return (uint8_t)(sign << 7);
    return (uint8_t)((sign << 7) | mant);
  }

  // Normal range: biased exponent
  int exp8 = fp32_exp + 7;
  if (exp8 < 1)
    exp8 = 1;

  // Round mantissa from 23 bits to 3 bits
  float frac = abs_x / exp2f((float)(exp8 - 7)) - 1.0f;
  int mant3 = (int)roundf(frac * 8.0f);
  if (mant3 > 7) {
    mant3 = 0;
    exp8++;
  }
  if (exp8 > 15) {
    return (uint8_t)((sign << 7) | (15 << 3) | 6); // clamp to max
  }

  // Avoid NaN encoding (exp=15, mant=7)
  if (exp8 == 15 && mant3 == 7)
    mant3 = 6;

  return (uint8_t)((sign << 7) | (exp8 << 3) | mant3);
}

// ==================== FP16 IEEE half ====================
// sign(1) + exp(5) + mant(10), bias = 15

static inline float fp16_to_float(uint16_t x) {
  uint32_t sign = (x >> 15) & 1;
  uint32_t exp = (x >> 10) & 0x1F;
  uint32_t mant = x & 0x3FF;

  if (exp == 0x1F) {
    if (mant == 0)
      return sign ? -INFINITY : INFINITY;
    return NAN;
  }

  float result;
  if (exp == 0) {
    result = ldexpf((float)mant / 1024.0f, 1 - 15);
  } else {
    result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
  }
  return sign ? -result : result;
}

static inline uint16_t float_to_fp16(float x) {
  if (isnan(x))
    return 0x7E00;
  if (isinf(x))
    return (x > 0) ? 0x7C00 : 0xFC00;

  uint32_t bits;
  memcpy(&bits, &x, 4);
  uint32_t sign = (bits >> 31) & 1;
  int32_t fp32_exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t fp32_mant = bits & 0x7FFFFF;

  if (x == 0.0f)
    return (uint16_t)(sign << 15);

  if (fp32_exp > 15)
    return (uint16_t)((sign << 15) | 0x7C00);

  if (fp32_exp < -24)
    return (uint16_t)(sign << 15);

  if (fp32_exp < -14) {
    // Subnormal fp16
    int shift = -14 - fp32_exp;
    uint32_t full_mant = (1 << 23) | fp32_mant;
    uint32_t mant10 = (full_mant + (1 << (12 + shift))) >> (13 + shift);
    if (mant10 > 0x3FF)
      mant10 = 0x3FF;
    return (uint16_t)((sign << 15) | mant10);
  }

  int exp16 = fp32_exp + 15;
  uint32_t mant10 = (fp32_mant + (1 << 12)) >> 13;
  if (mant10 > 0x3FF) {
    mant10 = 0;
    exp16++;
    if (exp16 > 30)
      return (uint16_t)((sign << 15) | 0x7C00);
  }

  return (uint16_t)((sign << 15) | (exp16 << 10) | mant10);
}

// ==================== BF16 (bfloat16) ====================
// sign(1) + exp(8) + mant(7), bias = 127
// Same exponent range as fp32, truncated mantissa.

static inline float bf16_to_float(uint16_t x) {
  uint32_t fp32_bits = (uint32_t)x << 16;
  float result;
  memcpy(&result, &fp32_bits, 4);
  return result;
}

static inline uint16_t float_to_bf16(float x) {
  uint32_t bits;
  memcpy(&bits, &x, 4);
  // Round to nearest even
  uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
  return (uint16_t)((bits + rounding_bias) >> 16);
}

// ==================== FP6 E2M3 (AIE4b only) ====================
// sign(1) + exp(2) + mant(3), bias = 1
// No infinity, no NaN. All 64 encodings are valid numbers.
// Max = 7.5, min normal = 1.0, min subnormal = 0.125

static inline float fp6_e2m3_to_float(uint8_t x) {
  uint32_t sign = (x >> 5) & 1;
  uint32_t exp = (x >> 3) & 0x03;
  uint32_t mant = x & 0x07;

  float result;
  if (exp == 0) {
    result = ldexpf((float)mant / 8.0f, 1 - 1);
  } else {
    result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 1);
  }
  return sign ? -result : result;
}

static inline uint8_t float_to_fp6_e2m3(float x) {
  uint32_t sign = (x < 0) ? 1 : 0;
  float abs_x = fabsf(x);

  if (abs_x == 0.0f)
    return (uint8_t)(sign << 5);

  // Clamp to max
  if (abs_x > 7.5f)
    abs_x = 7.5f;

  // Try subnormal
  if (abs_x < 1.0f) {
    int mant = (int)roundf(abs_x * 8.0f);
    if (mant > 7)
      mant = 7;
    if (mant <= 0)
      return (uint8_t)(sign << 5);
    return (uint8_t)((sign << 5) | mant);
  }

  // Normal
  int exp6 = (int)floorf(log2f(abs_x));
  if (exp6 > 2)
    exp6 = 2;
  exp6 += 1; // add bias
  float frac = abs_x / exp2f((float)(exp6 - 1)) - 1.0f;
  int mant = (int)roundf(frac * 8.0f);
  if (mant > 7) {
    mant = 0;
    exp6++;
  }
  if (exp6 > 3)
    return (uint8_t)((sign << 5) | (3 << 3) | 7);

  return (uint8_t)((sign << 5) | (exp6 << 3) | mant);
}

// ==================== FP4 E2M1 (AIE4b only) ====================
// sign(1) + exp(2) + mant(1), bias = 1
// No infinity, no NaN. 16 encodings total.
// Max = 6.0, min normal = 1.0, min subnormal = 0.5

static inline float fp4_e2m1_to_float(uint8_t x) {
  uint32_t sign = (x >> 3) & 1;
  uint32_t exp = (x >> 1) & 0x03;
  uint32_t mant = x & 0x01;

  float result;
  if (exp == 0) {
    result = ldexpf((float)mant / 2.0f, 1 - 1);
  } else {
    result = ldexpf(1.0f + (float)mant / 2.0f, (int)exp - 1);
  }
  return sign ? -result : result;
}

static inline uint8_t float_to_fp4_e2m1(float x) {
  uint32_t sign = (x < 0) ? 1 : 0;
  float abs_x = fabsf(x);

  if (abs_x == 0.0f)
    return (uint8_t)(sign << 3);

  if (abs_x > 6.0f)
    abs_x = 6.0f;

  // Find best encoding by exhaustive search (only 8 positive values)
  float best_err = INFINITY;
  uint8_t best = 0;
  for (uint8_t code = 0; code < 8; code++) {
    float val = fp4_e2m1_to_float(code);
    float err = fabsf(val - abs_x);
    if (err < best_err) {
      best_err = err;
      best = code;
    }
  }
  return (uint8_t)((sign << 3) | best);
}

#ifdef __cplusplus
}
#endif

#endif // AIE4_CPU_REF_FP_TYPES_H
