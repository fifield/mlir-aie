//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for NUMA cross-column memory access detection (9 columns)
// Each column C reads from its own segment and writes to column (C+1)%9.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "memory_allocator.h"
#include "test_utils.h"

#define VERBOSE_TEST 0
#define NUM_COLS 9
#define BUF_SIZE 256

int main(int argc, char *argv[]) {

  test_utils::print_test_banner("18_numa_cross_column",
                                "Testing NUMA cross-column memory access (9 columns)");

  aie_libxaie_ctx_t xaie;

  // NUMA segment layout (32MB per column):
  //   Column 0: 0x80000000   Column 1: 0x82000000   Column 2: 0x84000000
  //   Column 3: 0x86000000   Column 4: 0x88000000   Column 5: 0x8A000000
  //   Column 6: 0x8C000000   Column 7: 0x8E000000   Column 8: 0x90000000
  //
  // Each column C:
  //   Input  at base[C]          (local read)
  //   Output at base[(C+1)%9] + 0x400  (cross-column write)

  ext_mem_model_t handle_in[NUM_COLS];
  ext_mem_model_t handle_out[NUM_COLS];
  ext_mem_model_t handle_pad[NUM_COLS];
  uint32_t *ddr_in[NUM_COLS];
  uint32_t *ddr_out[NUM_COLS];

  ////////////////////////////////////////////////////
  // SETUP EXTERNAL MEMORY
  // Allocations are sequential from 0x80000000.
  // Each 32MB column segment contains:
  //   offset 0x000: input buffer (1KB)
  //   offset 0x400: output buffer (1KB) — written by prev column
  //   rest: padding to reach next segment

  for (int c = 0; c < NUM_COLS; c++) {
    // Input buffer at base[c] (256 x 4 = 1024 bytes = 0x400)
    ddr_in[c] = mlir_aie_mem_alloc(&xaie, handle_in[c], BUF_SIZE);
    for (int i = 0; i < BUF_SIZE; i++)
      ddr_in[c][i] = 0xC0000000 | (c << 16) | i;
    mlir_aie_sync_mem_dev(handle_in[c]);

    // Output buffer at base[c] + 0x400 (written by column (c+8)%9)
    ddr_out[c] = mlir_aie_mem_alloc(&xaie, handle_out[c], BUF_SIZE);
    for (int i = 0; i < BUF_SIZE; i++)
      ddr_out[c][i] = 0xBAD00000 | (c << 16) | i;
    mlir_aie_sync_mem_dev(handle_out[c]);

    // Padding to fill rest of this column's 32MB segment
    // Already used: 0x800 (2KB). Need: 0x2000000 - 0x800 = 0x1FFF800 bytes
    // But skip padding after the last column
    if (c < NUM_COLS - 1) {
      uint32_t pad_words = (0x2000000 - 0x800) / 4;
      uint32_t *pad = mlir_aie_mem_alloc(&xaie, handle_pad[c], pad_words);
      mlir_aie_sync_mem_dev(handle_pad[c]);
    }
  }

  ////////////////////////////////////////////////////
  // CONFIGURATION

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  ////////////////////////////////////////////////////
  // TEST START - Trigger all columns

  for (int c = 0; c < NUM_COLS; c++) {
    // Release shim lock 1 (in_read) to start MM2S
    uint64_t lock_addr = ((uint64_t)c << 25) | 0x00010010;
    mlir_aie_sim_write32(lock_addr, 1);
  }

  if (VERBOSE_TEST) {
    printf("Test started - all %d columns active\n", NUM_COLS);
  }

  ////////////////////////////////////////////////////
  // CHECKING FOR TEST COMPLETION

  for (int c = 0; c < NUM_COLS; c++) {
    // Wait on shim lock 3 (out_read)
    uint64_t lock_addr = ((uint64_t)c << 25) | 0x00010030;
    test_utils::wait_on_lock(lock_addr, 1);
  }

  if (VERBOSE_TEST) {
    for (int c = 0; c < NUM_COLS; c++)
      test_utils::print_shim_tile_status(c);
  }

  ////////////////////////////////////////////////////
  // VALIDATION
  // Column C writes its passthrough output to column (C+1)%9's output buffer.
  // So column (C+1)%9's output should contain column C's input pattern.

  uint32_t errs = 0;

  for (int c = 0; c < NUM_COLS; c++) {
    // Validate input buffer unchanged (local read)
    uint32_t *in_data = mlir_aie_sync_mem_cpu(handle_in[c]);
    for (int i = 0; i < BUF_SIZE; i++) {
      uint32_t expected = 0xC0000000 | (c << 16) | i;
      if (in_data[i] != expected) {
        errs++;
        if (errs <= 10)
          printf("Col%d IN error @ %d : expected 0x%x, got 0x%x\n",
                 c, i, expected, in_data[i]);
      }
    }

    // Validate output buffer (cross-column write from column (c+8)%9)
    int src_col = (c + NUM_COLS - 1) % NUM_COLS;
    uint32_t *out_data = mlir_aie_sync_mem_cpu(handle_out[(src_col + 1) % NUM_COLS]);
    for (int i = 0; i < BUF_SIZE; i++) {
      uint32_t expected = 0xC0000000 | (src_col << 16) | i;
      if (out_data[i] != expected) {
        errs++;
        if (errs <= 10)
          printf("Col%d OUT error @ %d : expected 0x%x (from col%d), got 0x%x\n",
                 (src_col + 1) % NUM_COLS, i, expected, src_col, out_data[i]);
      }
    }
  }

  if (!errs) {
    printf("PASS!\n");
    printf("Check simulator output above for NUMA cross-column access statistics\n");
    return 0;
  } else {
    printf("FAIL (%d errors)\n", errs);
    return -1;
  }
}
