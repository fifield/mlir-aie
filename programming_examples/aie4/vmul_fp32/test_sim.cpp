//===- test.cpp - Test harness for vmul_fp32 -----------------*- C++ -*-===//
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

#define VECTOR_SIZE 256

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "vmul_fp32",
      "Elementwise fp32*fp32->fp32 VMAC test");

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

  float input_a[VECTOR_SIZE];
  float input_b[VECTOR_SIZE];
  float golden[VECTOR_SIZE];

  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = dis(gen);
    input_b[i] = dis(gen);
    golden[i] = input_a[i] * input_b[i];
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;
  uint64_t tile_base = (col << 25) | ((row + 4) << 20);

  // Write A: 256 fp32 = 1024 bytes = 256 words at offset 1024
  uint32_t a_words[VECTOR_SIZE];
  memcpy(a_words, input_a, VECTOR_SIZE * sizeof(float));
  for (unsigned i = 0; i < 256; i++)
    mlir_aie_sim_write32(tile_base + 1024 + i * 4, a_words[i]);

  // Write B: 256 fp32 = 1024 bytes = 256 words at offset 2048
  uint32_t b_words[VECTOR_SIZE];
  memcpy(b_words, input_b, VECTOR_SIZE * sizeof(float));
  for (unsigned i = 0; i < 256; i++)
    mlir_aie_sim_write32(tile_base + 2048 + i * 4, b_words[i]);

  // Release input lock (lock 1)
  uint64_t tile_lock_1 = tile_base + 0x000AE010;
  mlir_aie_sim_write32(tile_lock_1, 1);

  // Wait for output lock (lock 3)
  printf("Waiting for result...\n");
  uint64_t tile_lock_3 = tile_base + 0x000AE030;
  while (mlir_aie_sim_read32(tile_lock_3) < 1)
    ;

  // Read C: 256 fp32 = 1024 bytes = 256 words at offset 3072
  float output[VECTOR_SIZE];
  for (unsigned i = 0; i < 256; i++) {
    uint32_t bits = mlir_aie_sim_read32(tile_base + 3072 + i * 4);
    memcpy(&output[i], &bits, 4);
  }

  uint32_t errs = 0;
  float max_rel_err = 0.0f;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    float diff = fabsf(output[i] - golden[i]);
    float ref_abs = fabsf(golden[i]);
    float rel_err = (ref_abs > 1e-30f) ? diff / ref_abs : diff;
    if (rel_err > max_rel_err)
      max_rel_err = rel_err;

    if (rel_err > 1e-6f) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got %.8f, expected %.8f (rel_err=%.10f)\n",
               i, output[i], golden[i], rel_err);
    }
  }

  printf("Max relative error: %.10f\n", max_rel_err);

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, VECTOR_SIZE); return -1; }
}
