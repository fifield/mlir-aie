//===- test.cpp - Test harness for vmul_int8x8 ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
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

#define VECTOR_SIZE 256

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "vmul_int8x8",
      "Elementwise int8*int8->int32 VMAC test");

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-128, 127);

  int8_t input_a[VECTOR_SIZE];
  int8_t input_b[VECTOR_SIZE];
  int32_t golden[VECTOR_SIZE];

  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = (int8_t)dis(gen);
    input_b[i] = (int8_t)dis(gen);
    golden[i] = (int32_t)input_a[i] * (int32_t)input_b[i];
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;
  uint64_t tile_base = (col << 25) | ((row + 4) << 20);

  // Write A (256 bytes at offset 1024)
  uint32_t *a_words = (uint32_t *)input_a;
  for (unsigned i = 0; i < VECTOR_SIZE / 4; i++)
    mlir_aie_sim_write32(tile_base + 1024 + i * 4, a_words[i]);

  // Write B (256 bytes at offset 1280)
  uint32_t *b_words = (uint32_t *)input_b;
  for (unsigned i = 0; i < VECTOR_SIZE / 4; i++)
    mlir_aie_sim_write32(tile_base + 1280 + i * 4, b_words[i]);

  // Release input lock
  uint64_t tile_lock_1 = tile_base + 0x000AE010;
  mlir_aie_sim_write32(tile_lock_1, 1);

  // Wait for output lock
  printf("Waiting for result...\n");
  uint64_t tile_lock_3 = tile_base + 0x000AE030;
  while (mlir_aie_sim_read32(tile_lock_3) < 1)
    ;

  // Read C (256 int32 at offset 2304)
  int32_t output[VECTOR_SIZE];
  for (unsigned i = 0; i < VECTOR_SIZE; i++) {
    uint32_t bits = mlir_aie_sim_read32(tile_base + 2304 + i * 4);
    memcpy(&output[i], &bits, 4);
  }

  uint32_t errs = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    if (output[i] != golden[i]) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got %d, expected %d (a=%d, b=%d)\n",
               i, output[i], golden[i], input_a[i], input_b[i]);
    }
  }

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, VECTOR_SIZE); return -1; }
}
