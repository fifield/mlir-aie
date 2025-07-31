//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include "memory_allocator.h"
#include "test_utils.h"

void populate_expected(int *buf) {
  const int stepsizes[3] = {2, 1, 16};
  const int wraps[3] = {8, 2, 8};
  int l = 0;
  for (int i = 0; i < wraps[2]; i++) {
    for (int j = 0; j < wraps[1]; j++) {
      for (int k = 0; k < wraps[0]; k++) {
        int i_ = i * stepsizes[2];
        int j_ = j * stepsizes[1];
        int k_ = k * stepsizes[0];
        buf[l] = i_ + j_ + k_;
        l++;
      }
    }
  }
}

void print_buf(int *buf) {
  for (int i = 0; i < 128; i++) {
    printf("%3d ", buf[i]);
    if (0 == (i + 1) % 8) {
      printf("\n");
    }
  }
}

int main(int argc, char *argv[]) {
  int errors = 0;
  int seen[128];
  int expected[128];

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t tile_col = 3;
  uint64_t tile_row = 4;

  while (true) {
    uint64_t tile_lock_1 = ((tile_col << 25) | (tile_row << 20)) + 0x0001F010;
    uint32_t lock_value = mlir_aie_sim_read32(tile_lock_1);
    printf("Lock value: %u\n", lock_value);
    for (int i = 0; i < 128; i++) {
      seen[i] = mlir_aie_sim_read32(
          (tile_col << 25) | (tile_row << 20) + 1024 + i * sizeof(uint32_t));
      // printf("Data value: %d %u\n", i, seen[i]);
    }
    if (lock_value >= 1)
      break;
    sleep(1);
  }

  // Compare results.
  populate_expected(expected);
  for (int i = 0; i < 128; i++) {
    if (expected[i] != seen[i]) {
      errors = 1;
      printf("Mismatch at index %d: %d != %d.\n", i, expected[i], seen[i]);
    }
  }
  if (0 == errors) {
    print_buf(seen);
    printf("PASS!\n");
    return 0;
  } else {
    printf("Expected:\n");
    print_buf(expected);
    printf("But got:\n");
    print_buf(seen);
    printf("FAIL.\n");
    return 1;
  }
}
