//===- test.cpp - Test harness for matmul_mxfp6 ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <stdint.h>
#include <stdio.h>

#include "memory_allocator.h"
#include "test_utils.h"

#include "../cpu_ref/fp_types.h"
#include "../cpu_ref/mx_types.h"

constexpr unsigned M = 4;
constexpr unsigned N = 64;
constexpr unsigned K = 256;
constexpr unsigned K_OUTER = 4;
constexpr unsigned K_INNER = 64;
constexpr unsigned MX_VEC_BYTES = 100; // 96 packed mantissa + 4 exponents
constexpr unsigned MANTISSA_BYTES = 96; // 128 * 6 / 8 = 96 bytes
constexpr unsigned A_MX_BYTES = K_OUTER * (M / 2) * MX_VEC_BYTES; // 800
constexpr unsigned B_MX_BYTES = K_OUTER * 32 * MX_VEC_BYTES;      // 12800

static void pack_mxfp6_vector(const float *elements, uint8_t *packed) {
  unsigned pos = 0;
  for (unsigned b = 0; b < 4; b++) {
    uint8_t packed_mantissas[MXFP6_MANTISSA_BYTES];
    uint8_t shared_exp;
    float_to_mxfp6_block(&elements[b * 32], packed_mantissas, &shared_exp);
    packed[pos++] = shared_exp;
    memcpy(&packed[pos], packed_mantissas, MXFP6_MANTISSA_BYTES);
    pos += MXFP6_MANTISSA_BYTES;
  }
}

static void unpack_mxfp6_vector(const uint8_t *packed, float *elements) {
  unsigned pos = 0;
  for (unsigned b = 0; b < 4; b++) {
    uint8_t shared_exp = packed[pos++];
    mxfp6_block_to_float(&packed[pos], shared_exp, &elements[b * 32]);
    pos += MXFP6_MANTISSA_BYTES;
  }
}

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "matmul_mxfp6",
      "MMAC MXFP6 matmul (4x256 * 256x64), fp32 output");

  float A_float[M * K];
  float B_float[K * N];

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (unsigned i = 0; i < M * K; i++)
    A_float[i] = dis(gen);
  for (unsigned i = 0; i < K * N; i++)
    B_float[i] = dis(gen);

  uint8_t A_mx[A_MX_BYTES];
  uint8_t B_mx[B_MX_BYTES];
  memset(A_mx, 0, sizeof(A_mx));
  memset(B_mx, 0, sizeof(B_mx));

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned v = 0; v < M / 2; v++) {
      float vec_elements[128];
      unsigned row0 = 2 * v, row1 = 2 * v + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned j = 0; j < K_INNER; j++) {
        vec_elements[j] = A_float[row0 * K + k_start + j];
        vec_elements[K_INNER + j] = A_float[row1 * K + k_start + j];
      }
      unsigned offset = (k * (M / 2) + v) * MX_VEC_BYTES;
      pack_mxfp6_vector(vec_elements, &A_mx[offset]);
    }
  }

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned j = 0; j < 32; j++) {
      float vec_elements[128];
      unsigned col0 = 2 * j;
      unsigned col1 = 2 * j + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned kk = 0; kk < K_INNER; kk++) {
        vec_elements[kk] = B_float[(k_start + kk) * N + col0];
        vec_elements[K_INNER + kk] = B_float[(k_start + kk) * N + col1];
      }
      unsigned offset = (k * 32 + j) * MX_VEC_BYTES;
      pack_mxfp6_vector(vec_elements, &B_mx[offset]);
    }
  }

  float A_decoded[M * K], B_decoded[K * N];

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned v = 0; v < M / 2; v++) {
      float vec_elements[128];
      unsigned offset = (k * (M / 2) + v) * MX_VEC_BYTES;
      unpack_mxfp6_vector(&A_mx[offset], vec_elements);
      unsigned row0 = 2 * v, row1 = 2 * v + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned j = 0; j < K_INNER; j++) {
        A_decoded[row0 * K + k_start + j] = vec_elements[j];
        A_decoded[row1 * K + k_start + j] = vec_elements[K_INNER + j];
      }
    }
  }

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned j = 0; j < 32; j++) {
      float vec_elements[128];
      unsigned offset = (k * 32 + j) * MX_VEC_BYTES;
      unpack_mxfp6_vector(&B_mx[offset], vec_elements);
      unsigned col0 = 2 * j;
      unsigned col1 = 2 * j + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned kk = 0; kk < K_INNER; kk++) {
        B_decoded[(k_start + kk) * N + col0] = vec_elements[kk];
        B_decoded[(k_start + kk) * N + col1] = vec_elements[K_INNER + kk];
      }
    }
  }

  float C_ref[M * N];
  memset(C_ref, 0, sizeof(C_ref));
  for (unsigned m = 0; m < M; m++)
    for (unsigned n = 0; n < N; n++)
      for (unsigned k = 0; k < K; k++)
        C_ref[m * N + n] += A_decoded[m * K + k] * B_decoded[k * N + n];

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;

  uint64_t tile_mem_a = ((col << 25) | ((row + 4) << 20)) + 1024;
  uint32_t *a_words = (uint32_t *)A_mx;
  for (unsigned i = 0; i < A_MX_BYTES / 4; i++)
    mlir_aie_sim_write32(tile_mem_a + i * 4, a_words[i]);

  uint64_t tile_mem_b = ((col << 25) | ((row + 4) << 20)) + 32768;
  uint32_t *b_words = (uint32_t *)B_mx;
  for (unsigned i = 0; i < B_MX_BYTES / 4; i++)
    mlir_aie_sim_write32(tile_mem_b + i * 4, b_words[i]);

  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  mlir_aie_sim_write32(tile_lock_1, 1);

  printf("Waiting for matmul to complete...\n");
  uint64_t tile_lock_3 = ((col << 25) | ((row + 4) << 20)) + 0x000AE030;
  while (mlir_aie_sim_read32(tile_lock_3) < 1)
    ;

  uint64_t tile_mem_c = ((col << 25) | ((row + 4) << 20)) + 65536;
  float C_hw[M * N];
  for (unsigned i = 0; i < M * N; i++) {
    uint32_t bits = mlir_aie_sim_read32(tile_mem_c + i * 4);
    memcpy(&C_hw[i], &bits, 4);
  }

  uint32_t errs = 0;
  float max_rel_err = 0.0f;
  for (unsigned i = 0; i < M * N; i++) {
    float diff = fabsf(C_hw[i] - C_ref[i]);
    float ref_abs = fabsf(C_ref[i]);
    float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
    if (rel_err > max_rel_err)
      max_rel_err = rel_err;

    if (rel_err > 0.1f && diff > 0.1f) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ [%d][%d]: got %.6f, expected %.6f (rel=%.6f)\n",
               i / N, i % N, C_hw[i], C_ref[i], rel_err);
    }
  }

  printf("Max relative error: %.8f\n", max_rel_err);

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, M * N); return -1; }
}
