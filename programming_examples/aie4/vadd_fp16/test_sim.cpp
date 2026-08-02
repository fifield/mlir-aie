//===- test.cpp - Test harness for vadd_fp16 ------------------*- C++ -*-===//
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

static uint16_t float_to_fp16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  uint32_t sign = (bits >> 31) & 1;
  int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (bits >> 13) & 0x3FF;
  if (exp <= 0) {
    exp = 0;
    mant = 0;
  }
  if (exp >= 31) {
    exp = 31;
    mant = 0;
  }
  return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

static float fp16_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0)
    return sign ? -0.0f : 0.0f;
  if (exp == 31)
    return sign ? -INFINITY : INFINITY;
  uint32_t bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

int main(int argc, char *argv[]) {

  test_utils::print_test_banner("vadd_fp16",
                                "Elementwise fp16+fp16->fp16 test");

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

  uint16_t input_a[VECTOR_SIZE], input_b[VECTOR_SIZE], golden[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = float_to_fp16(dis(gen));
    input_b[i] = float_to_fp16(dis(gen));
    float sum = fp16_to_float(input_a[i]) + fp16_to_float(input_b[i]);
    golden[i] = float_to_fp16(sum);
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;
  uint64_t base = (col << 25) | ((row + 4) << 20);

  uint64_t tile_mem_a = base + 1024;
  uint32_t *a_words = (uint32_t *)input_a;
  for (unsigned i = 0; i < VECTOR_SIZE / 2; i++)
    mlir_aie_sim_write32(tile_mem_a + i * 4, a_words[i]);

  uint64_t tile_mem_b = base + 2048;
  uint32_t *b_words = (uint32_t *)input_b;
  for (unsigned i = 0; i < VECTOR_SIZE / 2; i++)
    mlir_aie_sim_write32(tile_mem_b + i * 4, b_words[i]);

  mlir_aie_sim_write32(base + 0x000AE010, 1);

  printf("Waiting for vadd to complete...\n");
  while (mlir_aie_sim_read32(base + 0x000AE030) < 1)
    ;

  uint64_t tile_mem_c = base + 4096;
  uint16_t output[VECTOR_SIZE];
  for (unsigned i = 0; i < VECTOR_SIZE / 2; i++) {
    uint32_t word = mlir_aie_sim_read32(tile_mem_c + i * 4);
    output[2 * i] = (uint16_t)(word & 0xFFFF);
    output[2 * i + 1] = (uint16_t)(word >> 16);
  }

  uint32_t errs = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    int32_t diff = (int32_t)output[i] - (int32_t)golden[i];
    if (diff < 0)
      diff = -diff;
    if (diff > 1) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got 0x%04x (%.6f), expected 0x%04x (%.6f)\n", i,
               output[i], fp16_to_float(output[i]), golden[i],
               fp16_to_float(golden[i]));
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("FAIL: %d/%d errors\n", errs, VECTOR_SIZE);
    return -1;
  }
}
