//===- test.cpp - Test harness for vmul_bf16_64lane ----------*- C++ -*-===//
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

#define VECTOR_SIZE 256

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "vmul_bf16_64lane",
      "Elementwise bf16*bf16->bf16 VMAC 64-lane test");

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

  uint16_t input_a[VECTOR_SIZE];
  uint16_t input_b[VECTOR_SIZE];
  uint16_t golden[VECTOR_SIZE];

  for (int i = 0; i < VECTOR_SIZE; i++) {
    float fa = dis(gen);
    float fb = dis(gen);
    input_a[i] = float_to_bf16(fa);
    input_b[i] = float_to_bf16(fb);
    float qa = bf16_to_float(input_a[i]);
    float qb = bf16_to_float(input_b[i]);
    golden[i] = float_to_bf16(qa * qb);
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;
  uint64_t tile_base = (col << 25) | ((row + 4) << 20);

  // Write A: 256 bf16 = 512 bytes = 128 words at offset 1024
  uint32_t *a_words = (uint32_t *)input_a;
  for (unsigned i = 0; i < 128; i++)
    mlir_aie_sim_write32(tile_base + 1024 + i * 4, a_words[i]);

  // Write B: 256 bf16 = 512 bytes = 128 words at offset 1536
  uint32_t *b_words = (uint32_t *)input_b;
  for (unsigned i = 0; i < 128; i++)
    mlir_aie_sim_write32(tile_base + 1536 + i * 4, b_words[i]);

  // Release input lock (lock 1)
  uint64_t tile_lock_1 = tile_base + 0x000AE010;
  mlir_aie_sim_write32(tile_lock_1, 1);

  // Wait for output lock (lock 3)
  printf("Waiting for result...\n");
  uint64_t tile_lock_3 = tile_base + 0x000AE030;
  while (mlir_aie_sim_read32(tile_lock_3) < 1)
    ;

  // Read C: 256 bf16 = 512 bytes = 128 words at offset 2048
  uint16_t output[VECTOR_SIZE];
  for (unsigned i = 0; i < 128; i++) {
    uint32_t bits = mlir_aie_sim_read32(tile_base + 2048 + i * 4);
    output[i * 2] = (uint16_t)(bits & 0xFFFF);
    output[i * 2 + 1] = (uint16_t)(bits >> 16);
  }

  uint32_t errs = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    float out_f = bf16_to_float(output[i]);
    float ref_f = bf16_to_float(golden[i]);
    float diff = fabsf(out_f - ref_f);
    float ref_abs = fabsf(ref_f);
    float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;

    bool pass = (ref_abs < 1e-6f) ? (diff < 1e-3f) : (rel_err < 1e-2f);
    if (!pass) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got 0x%04x=%.4f, expected 0x%04x=%.4f "
               "(rel_err=%.6f)\n",
               i, output[i], out_f, golden[i], ref_f, rel_err);
    }
  }

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, VECTOR_SIZE); return -1; }
}
