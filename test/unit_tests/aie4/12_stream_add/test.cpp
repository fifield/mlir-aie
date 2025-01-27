//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <stdio.h>

#include "test_utils.h"

extern "C" {

void NpuWrite32(uint64_t Addr, unsigned int Data);
uint NpuRead32(uint64_t Addr);
void NpuWriteTransactionBinary(unsigned char *data, size_t nbytes);
}

int main(int argc, char *argv[]) {
  printf("Running...\n");

  uint64_t buffer_size = 2048;
  uint64_t tile_size = 32;
  uint64_t output_size = 4;
  uint64_t col = 0;

  uint32_t lock_id;
  uint64_t row;

  // We want to leave some space for the overrun pattern on A and B
  uint64_t input_a_offset = 0;
  uint64_t input_b_offset = 16896; // 16384 + 512
  uint64_t output_offset = 33792;  // 2*(16384 + 512)

  uint32_t input_a_pattern_seed = 0x10000042;
  uint32_t input_b_pattern_seed = 0x02200001;
  uint32_t input_a_pattern_stomp = 0xbaadbaad;
  uint32_t input_b_pattern_stomp = 0xdaabdaab;
  uint32_t output_pattern_stomp = 0x00badbad;

  // Stomp over the data in the memory tiles and an overrun pattern
  row = 1;
  uint64_t input_a_memtile_data_mem =
      (col << 25) | (row << 20) + 0x00100000 + input_a_offset;
  uint64_t input_b_memtile_data_mem =
      (col << 25) | (row << 20) + 0x00100000 + input_b_offset;
  uint64_t output_memtile_data_mem =
      (col << 25) | (row << 20) + 0x00100000 + output_offset;

  printf("The buffer size is %ld\n", buffer_size);

  for (int i = 0; i < buffer_size; i++) {
    NpuWrite32(input_a_memtile_data_mem + (4 * i), input_a_pattern_seed + i);
    NpuWrite32(input_b_memtile_data_mem + (4 * i), input_b_pattern_seed + i);
    NpuWrite32(output_memtile_data_mem + (4 * i), output_pattern_stomp);
  }
  for (int i = 0; i < 4; i++) {
    NpuWrite32(input_a_memtile_data_mem + (4 * (buffer_size + i)),
               input_a_pattern_stomp);
    NpuWrite32(input_b_memtile_data_mem + (4 * (buffer_size + i)),
               input_b_pattern_stomp);
    NpuWrite32(output_memtile_data_mem + (4 * (buffer_size + i)),
               output_pattern_stomp);
  }

  printf("Memory tile has been stomped.\n");

  // Check the stomps
  row = 1;
  for (int i = 0; i < output_size; i++) {
    uint32_t in_a = NpuRead32(input_a_memtile_data_mem + (4 * i));
    uint32_t in_b = NpuRead32(input_b_memtile_data_mem + (4 * i));
    uint32_t out = NpuRead32(output_memtile_data_mem + (4 * i));
    printf("input_a[%d] = 0x%08x, input_b[%d] = 0x%08x, output[%d] = 0x%08x\n",
           i, in_a, i, in_b, i, out);
  }

  // Stomp over the AIE tile memory
  row = 2;
  uint64_t tile_ping_in_a_buffer = ((col << 25) | ((row + 4) << 20)) + 2048;
  uint64_t tile_pong_in_a_buffer = ((col << 25) | ((row + 4) << 20)) + 2304;
  uint64_t tile_ping_out_buffer = ((col << 25) | ((row + 4) << 20)) + 3072;
  uint64_t tile_pong_out_buffer = ((col << 25) | ((row + 4) << 20)) + 3328;
  for (int i = 0; i < tile_size; i++) {
    NpuWrite32(tile_ping_in_a_buffer + (4 * i), 0xbad11100 + i);
    NpuWrite32(tile_pong_in_a_buffer + (4 * i), 0xbad22200 + i);
    NpuWrite32(tile_ping_out_buffer + (4 * i), 0xbad55500 + i);
    NpuWrite32(tile_pong_out_buffer + (4 * i), 0xbad66600 + i);
  }

  printf("Tile memory stomped.\n");

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  NpuWriteTransactionBinary(reinterpret_cast<uint8_t *>(data.data()),
                            data.size() * sizeof(uint32_t));

  // Toggle the memtile locks to indicate that data has been written and can be
  // read by the DMA
  row = 1;
  uint64_t mem_tile_lock_1 = ((col << 25) | (row << 20)) + 0x00010010;
  NpuWrite32(mem_tile_lock_1, 1); // There's 1 tile of data there now, go for it
  uint64_t mem_tile_lock_3 = ((col << 25) | (row << 20)) + 0x00010030;
  NpuWrite32(mem_tile_lock_3, 1); // There's 1 tile of data there now, go for it

  printf("Locks toggled\n");

  row = 1;
  while (1) {
    uint64_t mem_tile_lock_5 = ((col << 25) | (row << 20)) + 0x00010050;
    uint32_t l = NpuRead32(mem_tile_lock_5);
    if (l >= 1)
      break;
  }

  printf("The memory tile lock has toggled\n");

  // Check the memory tile for the ascending pattern
  row = 1;
  uint32_t output[buffer_size];
  int errs = 0;
  for (int i = 0; i < buffer_size; i++) {
    output[i] = NpuRead32(output_memtile_data_mem + (4 * i));
    uint32_t expected = input_a_pattern_seed + i + input_b_pattern_seed + i;
    if (output[i] != expected) {
      errs++;
      printf("error @ %d : %x != %x (actual != expected)\n", i, output[i],
             expected);
    } else {
      printf(" good @ %d : %x == %x (actual == expected)\n", i, output[i],
             expected);
    }
  }

  if (errs) {
    printf("failed.\n");
    return -1;
  }

  printf("PASS!\n");
  return 0;
}