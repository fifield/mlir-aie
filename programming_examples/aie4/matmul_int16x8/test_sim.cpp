//===- test.cpp - Test harness for matmul_int16x8 -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verifies MMAC int16*int8->int32 mixed-precision matrix multiply on AIE4.
// C[4x64] += A[4x128] * B[128x64]. Direct tile memory access.
//
// Data layout (K_outer-blocked, matching kernel access pattern):
//   A[ko * M * K_INNER + m * K_INNER + k]  — row-major, int16
//   B[ko * N * K_INNER + n * K_INNER + k]  — column-major, int8
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

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "matmul_int16x8",
      "MMAC int16*int8->int32 mixed matmul (4x128 * 128x64)");

  constexpr unsigned M = 4;
  constexpr unsigned N = 64;
  constexpr unsigned K_INNER = 64;
  constexpr unsigned K_OUTER = 2;

  int16_t A[K_OUTER * M * K_INNER];
  int8_t B[K_OUTER * N * K_INNER];
  int32_t C_ref[M * N];

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis_a(-127, 127);
  std::uniform_int_distribution<int> dis_b(-127, 127);

  for (unsigned i = 0; i < K_OUTER * M * K_INNER; i++)
    A[i] = (int16_t)dis_a(gen);
  for (unsigned i = 0; i < K_OUTER * N * K_INNER; i++)
    B[i] = (int8_t)dis_b(gen);

  memset(C_ref, 0, sizeof(C_ref));
  for (unsigned ko = 0; ko < K_OUTER; ko++)
    for (unsigned m = 0; m < M; m++)
      for (unsigned n = 0; n < N; n++)
        for (unsigned k = 0; k < K_INNER; k++)
          C_ref[m * N + n] += (int32_t)A[ko * M * K_INNER + m * K_INNER + k] *
                               (int32_t)B[ko * N * K_INNER + n * K_INNER + k];

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;

  // A: int16, 2 bytes each. Total = K_OUTER * M * K_INNER * 2 bytes
  uint64_t tile_mem_a = ((col << 25) | ((row + 4) << 20)) + 1024;
  uint32_t *a_words = (uint32_t *)A;
  for (unsigned i = 0; i < K_OUTER * M * K_INNER / 2; i++)
    mlir_aie_sim_write32(tile_mem_a + i * 4, a_words[i]);

  // B: int8, 1 byte each.
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

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("FAIL: %d/%d errors\n", errs, M * N);
    return -1;
  }
}
