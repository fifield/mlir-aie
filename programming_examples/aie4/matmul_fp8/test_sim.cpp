//===- test.cpp - Test harness for matmul_fp8 -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verifies MMAC fp8(E4M3)*fp8->fp32 matrix multiply on AIE4.
// C[4x64] += A[4x256] * B[256x64]. Direct tile memory access.
//
// Data layout (K_outer-blocked, matching kernel access pattern):
//   A[ko * M * K_INNER + m * K_INNER + k]  — row-major within each block
//   B[ko * N * K_INNER + n * K_INNER + k]  — column-major within each block
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

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "matmul_fp8",
      "MMAC fp8*fp8->fp32 matmul (4x256 * 256x64)");

  constexpr unsigned M = 4;
  constexpr unsigned N = 64;
  constexpr unsigned K_INNER = 64;
  constexpr unsigned K_OUTER = 4;

  uint8_t A[K_OUTER * M * K_INNER];
  uint8_t B[K_OUTER * N * K_INNER];
  float C_ref[M * N];

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-4.0f, 4.0f);

  for (unsigned i = 0; i < K_OUTER * M * K_INNER; i++)
    A[i] = float_to_fp8_e4m3(dis(gen));
  for (unsigned i = 0; i < K_OUTER * N * K_INNER; i++)
    B[i] = float_to_fp8_e4m3(dis(gen));

  memset(C_ref, 0, sizeof(C_ref));
  for (unsigned ko = 0; ko < K_OUTER; ko++)
    for (unsigned m = 0; m < M; m++)
      for (unsigned n = 0; n < N; n++)
        for (unsigned k = 0; k < K_INNER; k++)
          C_ref[m * N + n] +=
              fp8_e4m3_to_float(A[ko * M * K_INNER + m * K_INNER + k]) *
              fp8_e4m3_to_float(B[ko * N * K_INNER + n * K_INNER + k]);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;

  uint64_t tile_mem_a = ((col << 25) | ((row + 4) << 20)) + 1024;
  uint32_t *a_words = (uint32_t *)A;
  for (unsigned i = 0; i < K_OUTER * M * K_INNER / 4; i++)
    mlir_aie_sim_write32(tile_mem_a + i * 4, a_words[i]);

  uint64_t tile_mem_b = ((col << 25) | ((row + 4) << 20)) + 32768;
  uint32_t *b_words = (uint32_t *)B;
  for (unsigned i = 0; i < K_OUTER * N * K_INNER / 4; i++)
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

    if (rel_err > 1e-3f && diff > 1e-3f) {
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
