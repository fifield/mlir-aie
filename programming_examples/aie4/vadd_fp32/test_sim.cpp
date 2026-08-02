//===- test.cpp - Test harness for vadd_fp32 ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "memory_allocator.h"
#include "test_utils.h"

#define VECTOR_SIZE 256
#define VERBOSE_TEST 0

int main(int argc, char *argv[]) {

  test_utils::print_test_banner("vadd_fp32", "Elementwise fp32+fp32->fp32 test");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_a, handle_b, handle_c;
  uint64_t col = 0, row;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

  float input_a[VECTOR_SIZE], input_b[VECTOR_SIZE], golden[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = dis(gen);
    input_b[i] = dis(gen);
    golden[i] = input_a[i] + input_b[i];
  }

  uint32_t *ddr_ptr_a = mlir_aie_mem_alloc(&xaie, handle_a, 1024);
  memcpy(ddr_ptr_a, input_a, VECTOR_SIZE * sizeof(float));
  mlir_aie_sync_mem_dev(handle_a);

  uint32_t *ddr_ptr_b = mlir_aie_mem_alloc(&xaie, handle_b, 1024);
  memcpy(ddr_ptr_b, input_b, VECTOR_SIZE * sizeof(float));
  mlir_aie_sync_mem_dev(handle_b);

  uint32_t *ddr_ptr_c = mlir_aie_mem_alloc(&xaie, handle_c, 1024);
  for (int i = 0; i < 1024; i++) ddr_ptr_c[i] = 0xdeadbeef;
  mlir_aie_sync_mem_dev(handle_c);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  row = 0;
  mlir_aie_sim_write32(((col << 25) | (row << 20)) + 0x00010010, 1);
  mlir_aie_sim_write32(((col << 25) | (row << 20)) + 0x00010030, 1);

  printf("Waiting for output data...\n");
  test_utils::wait_on_lock((col << 25) | 0x00010050, 1);

  uint32_t *ddr_output = mlir_aie_sync_mem_cpu(handle_c);
  float *output = (float *)ddr_output;

  uint32_t errs = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    uint32_t out_bits, ref_bits;
    memcpy(&out_bits, &output[i], 4);
    memcpy(&ref_bits, &golden[i], 4);
    int32_t ulp_diff = (int32_t)out_bits - (int32_t)ref_bits;
    if (ulp_diff < 0) ulp_diff = -ulp_diff;

    if (ulp_diff > 0) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got %.8f, expected %.8f\n",
               i, output[i], golden[i]);
    }
  }

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, VECTOR_SIZE); return -1; }
}
