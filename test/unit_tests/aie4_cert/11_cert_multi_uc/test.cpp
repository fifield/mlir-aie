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
#include <unistd.h>

#include "cert_utils.h"
#include "memory_allocator.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle;
  uint32_t *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle, 256);
  for (int i = 0; i < 256; i++)
    ddr_ptr_in[i] = 0xdeadbeef + i;
  mlir_aie_sync_mem_dev(handle);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  constexpr uint32_t columns = 2;
  void *queue[columns];
  ext_mem_model_t queue_h[columns];
  for (uint32_t c = 0; c < columns; c++) {
    uint32_t fw_state = 0;
    do {
      fw_state = cert_utils::get_fw_state(c, 0);
      std::cout << "fw_state: " << fw_state << "\n";
    } while (fw_state != 0);

    queue[c] = cert_utils::setup_queue(&xaie, c, queue_h[c], 64);
  }

  ext_mem_model_t signal_h[columns];

  for (uint32_t c = 0; c < columns; c++) {
    std::vector<uint8_t> control_code =
        cert_utils::load_ctrl_pages_from_elf("aie4.elf", c);
    signal_h[c] =
        cert_utils::enqueue_control_code(&xaie, queue[c], control_code);
    mlir_aie_sync_mem_dev(queue_h[c]);
  }

  for (uint32_t c = 0; c < columns; c++) {
    cert_utils::set_partition_size(c, 4 /* partition_size */);
    cert_utils::set_alive(c, 0);
  }

  // wait for the completion signal to be set to state 'completed'
  uint64_t completion = 0;
  uint64_t completions[columns] = {0};
  uint32_t num_completions = 0;
  while (num_completions < columns) {
    for (uint32_t c = 0; c < columns; c++) {
      while (completions[c] == 0) {
        completions[c] =
            *reinterpret_cast<uint64_t *>(mlir_aie_sync_mem_cpu(signal_h[c]));
        std::cout << "completion signal: " << c << ": " << completions[c]
                  << "\n";
        if (!completions[c])
          sleep(1);
        else
          num_completions++;
      }
    }
  }

  int errs = 0;
  for (uint32_t c = 0; c < columns; c++) {
    if (completions[c] != 4) {
      std::cout << "packet failed.\n";
      errs++;
    }
  }

  // col 0 row 3 lock 5
  uint64_t col = 0;
  uint64_t row = 3;
  uint64_t tile_lock = ((col << 25) | (row + 4 << 20)) + 0xAE050;
  uint32_t l = mlir_aie_sim_read32(tile_lock);
  if (l != 1) {
    std::cout << "lock (0,3) 5 failed.\n";
    errs++;
  }

  // col 1 row 2 lock 2
  col = 1;
  row = 2;
  tile_lock = ((col << 25) | (row + 4 << 20)) + 0xAE020;
  l = mlir_aie_sim_read32(tile_lock);
  if (l != 4) {
    std::cout << "lock (1,2) 2 failed: " << l << ".\n";
    errs++;
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d).\n", errs);
    return -1;
  }
}