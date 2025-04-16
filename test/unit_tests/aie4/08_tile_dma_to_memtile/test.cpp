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
  printf("Hello, world.  I see you there\n");

  uint64_t buffer_size = 512;
  uint64_t output_size = 4;
  uint64_t col = 0;

  uint32_t lock_id;
  uint64_t row;

  // Stomp over the data in the memory tile and an overrun pattern
  row = 1;
  uint64_t memtile_data_mem = (col << 25) | (row << 20) + 0x00100000 + 2048;
  for (int i = 0; i < buffer_size + 4; i++) {
    mlir_aie_sim_write32(memtile_data_mem + (4 * i), 0xbaadbaad);
  }

  // Set the AIE tile memory to a known ascending pattern
  row = 2;
  uint64_t tile_data_mem = ((col << 25) | ((row + 4) << 20)) + 2048;
  for (int i = 0; i < buffer_size; i++)
    mlir_aie_sim_write32(tile_data_mem + (4 * i), 0xbeefcafe + i);
  printf("Tile memory stomped.\n");

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  printf("Configuration written.\n");

  row = 2;
  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  int timeout = 0;
  while (1) {
    uint32_t l = mlir_aie_sim_read32(tile_lock_1);
    if (l >= 2)
      break;
  }

  printf("The tile memory lock has toggled\n");

  row = 1;
  while (1) {
    uint64_t mem_tile_lock_1 = ((col << 25) | (row << 20)) + 0x00010010;
    uint32_t l = mlir_aie_sim_read32(mem_tile_lock_1);
    if (l >= 2)
      break;
  }
  printf("The memory tile lock has toggled\n");

  // Check the memory tile for the ascending pattern
  row = 1;
  uint32_t output[buffer_size];
  int errs = 0;
  for (int i = 0; i < buffer_size; i++) {
    output[i] = mlir_aie_sim_read32(memtile_data_mem + (4 * i));
    if (output[i] != 0xbeefcafe + i) {
      errs++;
      printf("error @ %d : %x\n", i, output[i]);
    }
  }
  for (int i = 0; i < 4; i++) {
    uint32_t over =
        mlir_aie_sim_read32(memtile_data_mem + (4 * (buffer_size + i)));
    if (over != 0xbaadbaad) {
      errs++;
      printf("error @ over %d : %x\n", i, over);
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