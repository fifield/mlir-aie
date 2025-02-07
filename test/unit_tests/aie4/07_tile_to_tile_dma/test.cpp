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
#include <vector>

#include <stdint.h>
#include <stdio.h>

#include "memory_allocator.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {
  printf("Hello, world.\n");

  uint64_t col = 0;
  uint64_t row = 4;
  uint64_t memtile_data_mem = (col << 25) | ((row + 4) << 20) + 2048;
  for (int i = 0; i < 256; i++)
    mlir_aie_sim_write32(memtile_data_mem + (4 * i), 0xdeadbeef + i);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  row = 3;
  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  int timeout = 0;
  while (1) {
    uint32_t l = mlir_aie_sim_read32(tile_lock_1);
    if (l >= 1)
      break;
  }

  uint64_t tile_data_mem = ((col << 25) | ((row + 4) << 20)) + 2048;
  uint32_t output[256];
  int errs = 0;
  for (int i = 0; i < 256; i++) {
    output[i] = mlir_aie_sim_read32(tile_data_mem + (4 * i));
    if (output[i] != 0xdeadbeef + i) {
      errs++;
      printf("error @ %d : %x\n", i, output[i]);
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail.\n");
    return -1;
  }
}