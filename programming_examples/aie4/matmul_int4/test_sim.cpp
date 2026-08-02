//===- test.cpp - Test harness for matmul_int4 ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verifies MMAC int4*int4->int32 matrix multiply on AIE4.
// C[4x64] += A[4x512] * B[512x64]. Direct tile memory access.
//
// Data layout (K_outer-blocked, matching kernel access pattern):
//   A[ko * M * K_INNER + m * K_INNER + k]  — row-major within each block
//   B[ko * N * K_INNER + n * K_INNER + k]  — column-major within each block
// K_INNER=128 for int4 (MMAC 128x64). Elements packed 2 per byte, low nibble first.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <stdint.h>
#include <stdio.h>

#include "memory_allocator.h"
#include "test_utils.h"

#include "../cpu_ref/int_types.h"

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "matmul_int4",
      "MMAC int4*int4->int32 matmul (4x512 * 512x64)");

  constexpr unsigned M = 4;
  constexpr unsigned N = 64;
  constexpr unsigned K_INNER = 128;
  constexpr unsigned K_OUTER = 4;
  constexpr unsigned TOTAL_A = K_OUTER * M * K_INNER;
  constexpr unsigned TOTAL_B = K_OUTER * N * K_INNER;

  int8_t A_vals[TOTAL_A];
  int8_t B_vals[TOTAL_B];
  int32_t C_ref[M * N];

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-7, 7);

  for (unsigned i = 0; i < TOTAL_A; i++)
    A_vals[i] = (int8_t)dis(gen);
  for (unsigned i = 0; i < TOTAL_B; i++)
    B_vals[i] = (int8_t)dis(gen);

  uint8_t A_packed[TOTAL_A / 2];
  uint8_t B_packed[TOTAL_B / 2];

  memset(A_packed, 0, sizeof(A_packed));
  for (unsigned i = 0; i < TOTAL_A; i++)
    int4_pack(A_packed, i, A_vals[i]);
  memset(B_packed, 0, sizeof(B_packed));
  for (unsigned i = 0; i < TOTAL_B; i++)
    int4_pack(B_packed, i, B_vals[i]);

  memset(C_ref, 0, sizeof(C_ref));
  for (unsigned ko = 0; ko < K_OUTER; ko++)
    for (unsigned m = 0; m < M; m++)
      for (unsigned n = 0; n < N; n++)
        for (unsigned k = 0; k < K_INNER; k++)
          C_ref[m * N + n] +=
              (int32_t)A_vals[ko * M * K_INNER + m * K_INNER + k] *
              (int32_t)B_vals[ko * N * K_INNER + n * K_INNER + k];

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;

  uint64_t tile_mem_a = ((col << 25) | ((row + 4) << 20)) + 1024;
  uint32_t *a_words = (uint32_t *)A_packed;
  for (unsigned i = 0; i < TOTAL_A / 2 / 4; i++)
    mlir_aie_sim_write32(tile_mem_a + i * 4, a_words[i]);

  uint64_t tile_mem_b = ((col << 25) | ((row + 4) << 20)) + 32768;
  uint32_t *b_words = (uint32_t *)B_packed;
  for (unsigned i = 0; i < TOTAL_B / 2 / 4; i++)
    mlir_aie_sim_write32(tile_mem_b + i * 4, b_words[i]);

  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  mlir_aie_sim_write32(tile_lock_1, 1);

  printf("Waiting for matmul to complete...\n");
  uint64_t tile_lock_3 = ((col << 25) | ((row + 4) << 20)) + 0x000AE030;
  while (mlir_aie_sim_read32(tile_lock_3) < 1)
    ;

  uint64_t tile_mem_c = ((col << 25) | ((row + 4) << 20)) + 65536;
  int32_t C_hw[M * N];
  for (unsigned i = 0; i < M * N; i++)
    C_hw[i] = (int32_t)mlir_aie_sim_read32(tile_mem_c + i * 4);

  uint32_t errs = 0;
  for (unsigned i = 0; i < M * N; i++) {
    if (C_hw[i] != C_ref[i]) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ [%d][%d]: got %d, expected %d\n",
               i / N, i % N, C_hw[i], C_ref[i]);
    }
  }

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, M * N); return -1; }
}
