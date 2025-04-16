//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#define VERBOSE_TEST 0

uint32_t makePackedBfloat16(float lower, float upper) {
  uint32_t to_return = 0x00;

  uint32_t lower_bit_pattern = *(uint32_t *)(&lower);
  uint32_t upper_bit_pattern = *(uint32_t *)(&upper);

  uint32_t lower_as_bfloat = (lower_bit_pattern & 0xffff0000) >> 16;
  uint32_t upper_as_bfloat = (upper_bit_pattern & 0xffff0000) >> 16;
  to_return = (upper_as_bfloat << 16) | (lower_as_bfloat);
  return to_return;
}

float getUpper(uint32_t word) {
  uint32_t masked = (word & 0xFFFF0000);
  float upper = *(float *)(&masked);
  return upper;
}

float getLower(uint32_t word) {
  uint32_t masked = (word & 0x0000FFFF) << 16;
  float lower = *(float *)(&masked);
  return lower;
}

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "14_bf16_rope",
      "Give me some rope I'm coming loose, I'm pulling for you now");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_in;
  ext_mem_model_t handle_table;
  ext_mem_model_t handle_out;

  uint64_t col = 0;
  uint64_t row;

  ////////////////////////////////////////////////////
  // SETUP EXTERNAL MEMORY

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dis(-2.0f, 2.0f);

  float input[4096];
  float weights[4096];
  float golden[4096];

  // Write an input pattern to DDR
  int *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle_in, 2048);
  for (int i = 0; i < 2048; i++) {
    input[i * 2] = dis(gen);
    input[i * 2 + 1] = dis(gen);
    ddr_ptr_in[i] = makePackedBfloat16(input[i * 2], input[i * 2 + 1]);
  }
  mlir_aie_sync_mem_dev(handle_in);

  int *ddr_table_in = mlir_aie_mem_alloc(&xaie, handle_table, 2048);
  for (int i = 0; i < 2048; i++) {
    weights[i * 2] = dis(gen);
    weights[i * 2 + 1] = dis(gen);
    ddr_table_in[i] = makePackedBfloat16(weights[i * 2], weights[i * 2 + 1]);
  }
  mlir_aie_sync_mem_dev(handle_table);

  for (int i = 0; i < 4096; i++)
    golden[i] = input[i] * weights[i];

  // Write an output pattern to DDR which will get overwritten
  int *ddr_ptr_out = mlir_aie_mem_alloc(&xaie, handle_out, 2048);
  for (int i = 0; i < 2048; i++)
    ddr_ptr_out[i] = 0xbadbad00 + i;
  mlir_aie_sync_mem_dev(handle_out);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 512, 4, 640, 4);
  }

  ////////////////////////////////////////////////////
  // CONFIGURATION

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  ////////////////////////////////////////////////////
  // TEST START

  // At this point, we need to toggle the shimtile locks to indicate that data
  // has been written and can be read by the DMA These are the
  // shim_tile_input_read_lock which the MM2S channel0 and 1 are blocked on
  // aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)

  row = 0;
  uint64_t shim_tile_lock_1 = ((col << 25) | (row << 20)) + 0x00010010;
  // There's 1 tile of data there now, go for it
  mlir_aie_sim_write32(shim_tile_lock_1, 1);

  uint64_t shim_tile_lock_3 = ((col << 25) | (row << 20)) + 0x00010030;
  // There's 1 tile of data there now, go for it
  mlir_aie_sim_write32(shim_tile_lock_3, 1);

  if (VERBOSE_TEST) {
    printf("Test just started, possibly not completed yet\n");
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  ////////////////////////////////////////////////////
  // CHECKING FOR TEST COMPLETION

  printf("Waiting on lock 5 in the shim to indicate data has been written\n");
  uint64_t shim_tile_lock_5 = (col << 25) | 0x00010050;
  test_utils::wait_on_lock(shim_tile_lock_5, 1);

  // When your lock logic goes wrong, this code below can help
  // printf("Going to wait for 10,000 cycles\n");
  // test_utils::wait_for_cycles(0, 2, 160000);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  // EXECUTION COMPLETE

  uint32_t errs = 0;

  // Lets test the inputs
  // We're only looking for rough equality because the golden values are float
  // but the DDR contains their bfloat equivalents
  // These test seems redundant but we do want to detect if somehow the output
  // overwrites and destroys the inputs

  mlir_aie_sync_mem_cpu(handle_in);
  uint32_t *ddr_input = reinterpret_cast<uint32_t *>(handle_in.virtualAddr);

  for (int i = 0; i < 128; i++) {
    uint32_t word = ddr_input[i];
    float lower = getLower(word);
    float upper = getUpper(word);
    if (!(test_utils::nearly_equal(lower, input[i * 2], 0.005, FLT_MIN))) {
      errs++;
      printf(" IN_I: error @ %d : %f != %f\n", i * 2, lower, input[i * 2]);
    } else if (VERBOSE_TEST)
      printf(" IN_I:   yes @ %d :%f ~= %f\n", i * 2, lower, input[i * 2]);

    if (!(test_utils::nearly_equal(upper, input[i * 2 + 1], 0.005, FLT_MIN))) {
      errs++;
      printf(" IN_I: error @ %d : %f != %f\n", i * 2 + 1, upper,
             input[i * 2 + 1]);
    } else if (VERBOSE_TEST)
      printf(" IN_I:   yes @ %d :%f ~= %f\n", i * 2 + 1, upper,
             input[i * 2 + 1]);
  }

  mlir_aie_sync_mem_cpu(handle_table);
  uint32_t *ddr_table_input =
      reinterpret_cast<uint32_t *>(handle_table.virtualAddr);

  for (int i = 0; i < 128; i++) {
    uint32_t word = ddr_table_input[i];
    float lower = getLower(word);
    float upper = getUpper(word);
    if (!(test_utils::nearly_equal(lower, weights[i * 2], 0.005, FLT_MIN))) {
      errs++;
      printf(" IN_W: error @ %d : %f != %f\n", i * 2, lower, weights[i * 2]);
    } else if (VERBOSE_TEST)
      printf(" IN_W:   yes @ %d :%f ~= %f\n", i * 2, lower, weights[i * 2]);

    if (!(test_utils::nearly_equal(upper, weights[i * 2 + 1], 0.005,
                                   FLT_MIN))) {
      errs++;
      printf(" IN_W: error @ %d : %f != %f\n", i * 2 + 1, upper,
             weights[i * 2 + 1]);
    } else if (VERBOSE_TEST)
      printf(" IN_W:   yes @ %d :%f ~= %f\n", i * 2 + 1, upper,
             weights[i * 2 + 1]);
  }

  // Now lets test the output
  mlir_aie_sync_mem_cpu(handle_out);
  uint32_t *ddr_output = reinterpret_cast<uint32_t *>(handle_out.virtualAddr);
  for (int i = 0; i < 2048; i++) {
    uint32_t word = ddr_output[i];
    float lower = getLower(word);
    float upper = getUpper(word);

    if (test_utils::nearly_equal(lower, golden[i * 2], 0.010, FLT_MIN)) {
      if (VERBOSE_TEST)
        printf("OUT:   yes @ %d : %f == %f\n", i, lower, golden[i * 2]);
    } else {
      errs++;
      printf("OUT: error @ %d : %f != %f\n", i, lower, golden[i * 2]);
    }

    if (test_utils::nearly_equal(upper, golden[i * 2 + 1], 0.010, FLT_MIN)) {
      if (VERBOSE_TEST)
        printf("OUT:   yes @ %d : %f == %f\n", i, upper, golden[i * 2 + 1]);
    } else {
      errs++;
      printf("OUT: error @ %d : %f != %f\n", i, upper, golden[i * 2 + 1]);
    }
  }

  // We check the first 128 inputs, the first 128 weights and all 2048 outputs
  int tests = 128 + 128 + 2048;
  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", tests - errs, tests);
    return -1;
  }
}