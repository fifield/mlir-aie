//===- test.cpp - Test harness for vadd_int8 ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
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

  test_utils::print_test_banner("vadd_int8", "Elementwise int8+int8->int8 test");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_a, handle_b, handle_c;
  uint64_t col = 0, row;

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-64, 63);

  int8_t input_a[VECTOR_SIZE], input_b[VECTOR_SIZE], golden[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = (int8_t)dis(gen);
    input_b[i] = (int8_t)dis(gen);
    golden[i] = (int8_t)(input_a[i] + input_b[i]);
  }

  uint32_t *ddr_ptr_a = mlir_aie_mem_alloc(&xaie, handle_a, 1024);
  memcpy(ddr_ptr_a, input_a, VECTOR_SIZE);
  for (int rep = 1; rep < 1024 * 4 / VECTOR_SIZE; rep++)
    memcpy((uint8_t *)ddr_ptr_a + rep * VECTOR_SIZE, input_a, VECTOR_SIZE);
  mlir_aie_sync_mem_dev(handle_a);

  uint32_t *ddr_ptr_b = mlir_aie_mem_alloc(&xaie, handle_b, 1024);
  memcpy(ddr_ptr_b, input_b, VECTOR_SIZE);
  for (int rep = 1; rep < 1024 * 4 / VECTOR_SIZE; rep++)
    memcpy((uint8_t *)ddr_ptr_b + rep * VECTOR_SIZE, input_b, VECTOR_SIZE);
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
  int8_t *output = (int8_t *)ddr_output;

  uint32_t errs = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    if (output[i] != golden[i]) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got %d, expected %d (a=%d, b=%d)\n",
               i, output[i], golden[i], input_a[i], input_b[i]);
    } else if (VERBOSE_TEST) {
      printf("  OK  @ %d: %d == %d\n", i, output[i], golden[i]);
    }
  }

  if (!errs) { printf("PASS!\n"); return 0; }
  else { printf("FAIL: %d/%d errors\n", errs, VECTOR_SIZE); return -1; }
}
