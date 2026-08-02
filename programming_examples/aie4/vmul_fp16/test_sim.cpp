//===- test.cpp - Test harness for vmul_fp16 -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verifies elementwise fp16 * fp16 -> fp32 on AIE4 VMAC unit.
// Writes data directly to tile memory, runs kernel, reads results back.
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
  test_utils::print_test_banner("vmul_fp16",
                                "Elementwise fp16*fp16->fp32 VMAC test");

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

  uint16_t input_a[VECTOR_SIZE];
  uint16_t input_b[VECTOR_SIZE];
  float golden[VECTOR_SIZE];

  for (int i = 0; i < VECTOR_SIZE; i++) {
    float fa = dis(gen);
    float fb = dis(gen);
    input_a[i] = float_to_fp16(fa);
    input_b[i] = float_to_fp16(fb);
    float qa = fp16_to_float(input_a[i]);
    float qb = fp16_to_float(input_b[i]);
    golden[i] = qa * qb;
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;
  uint64_t tile_base = (col << 25) | ((row + 4) << 20);

  // Write A (512 bytes = 128 words at offset 1024)
  uint32_t *a_words = (uint32_t *)input_a;
  for (unsigned i = 0; i < 128; i++)
    mlir_aie_sim_write32(tile_base + 1024 + i * 4, a_words[i]);

  // Write B (512 bytes = 128 words at offset 1536)
  uint32_t *b_words = (uint32_t *)input_b;
  for (unsigned i = 0; i < 128; i++)
    mlir_aie_sim_write32(tile_base + 1536 + i * 4, b_words[i]);

  // Release input lock
  uint64_t tile_lock_1 = tile_base + 0x000AE010;
  mlir_aie_sim_write32(tile_lock_1, 1);

  // Wait for output lock
  printf("Waiting for result...\n");
  uint64_t tile_lock_3 = tile_base + 0x000AE030;
  while (mlir_aie_sim_read32(tile_lock_3) < 1)
    ;

  // Read C (256 floats = 256 words at offset 2048)
  float output[VECTOR_SIZE];
  for (unsigned i = 0; i < VECTOR_SIZE; i++) {
    uint32_t bits = mlir_aie_sim_read32(tile_base + 2048 + i * 4);
    memcpy(&output[i], &bits, 4);
  }

  uint32_t errs = 0;
  float max_rel_err = 0.0f;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    float diff = fabsf(output[i] - golden[i]);
    float ref_abs = fabsf(golden[i]);
    float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;

    if (rel_err > max_rel_err)
      max_rel_err = rel_err;

    bool pass = (ref_abs < 1e-6f) ? (diff < 1e-3f) : (rel_err < 1e-3f);
    if (!pass) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got %.6f, expected %.6f (rel_err=%.6f)\n",
               i, output[i], golden[i], rel_err);
    }
  }

  printf("Max relative error: %.8f\n", max_rel_err);

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, VECTOR_SIZE); return -1; }
}
