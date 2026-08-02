//===- int_types.h - Integer packing utilities for AIE4 --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE4_CPU_REF_INT_TYPES_H
#define AIE4_CPU_REF_INT_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- INT4 pack/unpack (signed 2's complement, 4 bits) ---
// Packed layout: two int4 values per byte, lower nibble first.
// index 0 → bits [3:0] of byte 0, index 1 → bits [7:4] of byte 0, etc.

static inline int8_t int4_extract(const uint8_t *packed, int index) {
  uint8_t byte = packed[index >> 1];
  uint8_t nibble = (index & 1) ? (byte >> 4) : (byte & 0x0F);
  return (int8_t)((nibble ^ 0x08) - 0x08); // sign-extend from 4 bits
}

static inline void int4_pack(uint8_t *packed, int index, int8_t value) {
  int byte_idx = index >> 1;
  uint8_t nibble = (uint8_t)(value & 0x0F);
  if (index & 1) {
    packed[byte_idx] = (packed[byte_idx] & 0x0F) | (nibble << 4);
  } else {
    packed[byte_idx] = (packed[byte_idx] & 0xF0) | nibble;
  }
}

static inline uint8_t int4_saturate(int32_t value) {
  if (value > 7)
    return 7;
  if (value < -8)
    return (uint8_t)(-8 & 0x0F);
  return (uint8_t)(value & 0x0F);
}

// --- Sparse quartet encoding (MMAC sparse modes) ---
// Each column in sparse matrix: 64 mantissa bytes + 12 sparsity bytes = 76 bytes.
// Sparsity: 32 groups of 4 elements, 3 bits each select 2 of 4.
// Encoding:
//   001 → x[0],x[2]   010 → x[0],x[3]   011 → x[1],x[2]
//   100 → x[1],x[3]   101 → x[2],x[3]   000 → x[0],x[1]

#define SPARSE_BYTES_PER_COL 76
#define SPARSE_MANTISSA_BYTES 64
#define SPARSE_MASK_BYTES 12

static inline void sparse_quartet_indices(uint8_t encoding, int *i0, int *i1) {
  switch (encoding & 0x07) {
  case 0:
    *i0 = 0;
    *i1 = 1;
    break;
  case 1:
    *i0 = 0;
    *i1 = 2;
    break;
  case 2:
    *i0 = 0;
    *i1 = 3;
    break;
  case 3:
    *i0 = 1;
    *i1 = 2;
    break;
  case 4:
    *i0 = 1;
    *i1 = 3;
    break;
  case 5:
    *i0 = 2;
    *i1 = 3;
    break;
  default:
    *i0 = 0;
    *i1 = 1;
    break;
  }
}

// Unpack one column of sparse int8 data (76 bytes) into dense (128 values).
// dense_col must have room for 128 int8_t values.
static inline void sparse_int8_unpack_column(const uint8_t *sparse_col,
                                             int8_t *dense_col) {
  const int8_t *mantissas = (const int8_t *)sparse_col;
  const uint8_t *mask = sparse_col + SPARSE_MANTISSA_BYTES;

  // Zero the dense output
  for (int i = 0; i < 128; i++)
    dense_col[i] = 0;

  // 32 groups of 4, each group selects 2 non-zero from quartet
  // Sparsity mask: 32 * 3 bits = 96 bits = 12 bytes, packed LSB first
  int mantissa_idx = 0;
  int bit_offset = 0;
  for (int g = 0; g < 32; g++) {
    int byte_idx = bit_offset >> 3;
    int bit_pos = bit_offset & 7;
    uint16_t bits = (uint16_t)mask[byte_idx] | ((uint16_t)mask[byte_idx + 1] << 8);
    uint8_t encoding = (bits >> bit_pos) & 0x07;
    bit_offset += 3;

    int i0, i1;
    sparse_quartet_indices(encoding, &i0, &i1);
    dense_col[g * 4 + i0] = mantissas[mantissa_idx++];
    dense_col[g * 4 + i1] = mantissas[mantissa_idx++];
  }
}

#ifdef __cplusplus
}
#endif

#endif // AIE4_CPU_REF_INT_TYPES_H
