//===- test.cpp - Test harness for vmul_int32x16 --------------*- C++ -*-===//
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
      "vmul_int32x16",
      "Elementwise int32*int16->int32 VMAC test (16 lanes)");

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dis_a(-1000, 1000);
  std::uniform_int_distribution<int> dis_b(-1000, 1000);

  int32_t input_a[VECTOR_SIZE];
  int16_t input_b[VECTOR_SIZE];
  int32_t golden[VECTOR_SIZE];

  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = dis_a(gen);
    input_b[i] = (int16_t)dis_b(gen);
    golden[i] = input_a[i] * (int32_t)input_b[i];
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;
  uint64_t tile_base = (col << 25) | ((row + 4) << 20);

  // Write A: 256 int32 = 1024 bytes = 256 words at offset 1024
  uint32_t *a_words = (uint32_t *)input_a;
  for (unsigned i = 0; i < 256; i++)
    mlir_aie_sim_write32(tile_base + 1024 + i * 4, a_words[i]);

  // Write B: 256 int16 = 512 bytes = 128 words at offset 2048
  uint32_t *b_words = (uint32_t *)input_b;
  for (unsigned i = 0; i < 128; i++)
    mlir_aie_sim_write32(tile_base + 2048 + i * 4, b_words[i]);

  // Release input lock (lock 1)
  uint64_t tile_lock_1 = tile_base + 0x000AE010;
  mlir_aie_sim_write32(tile_lock_1, 1);

  // Wait for output lock (lock 3)
  printf("Waiting for result...\n");
  uint64_t tile_lock_3 = tile_base + 0x000AE030;
  while (mlir_aie_sim_read32(tile_lock_3) < 1)
    ;

  // Read C: 256 int32 = 1024 bytes = 256 words at offset 2560
  int32_t output[VECTOR_SIZE];
  for (unsigned i = 0; i < 256; i++) {
    uint32_t bits = mlir_aie_sim_read32(tile_base + 2560 + i * 4);
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
