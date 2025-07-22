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
  printf("Hello, world.\n");

  aie_libxaie_ctx_t xaie;

  // allocate data buffer and init to non-zero values
  ext_mem_model_t data_h;
  uint32_t *data_in = mlir_aie_mem_alloc(&xaie, data_h, 256 * 2);
  for (int i = 0; i < 256 * 2; i++)
    data_in[i] = 0x12345678 + i;
  mlir_aie_sync_mem_dev(data_h);

  const uint32_t column = 0;
  uint64_t row = 1;

  // write data to the memtile buffer
  uint64_t memtile_mem = ((column << 25) | (row << 20)) + 0x00100000 + 2048;
  for (int i = 0; i < 256; i++) {
    mlir_aie_sim_write32(memtile_mem + (i * 4), 0xdeadbeef + i);
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint32_t fw_state = 0;
  do {
    fw_state = cert_utils::get_fw_state(column, 0);
    std::cout << "fw_state: " << fw_state << "\n";
  } while (fw_state != 0);

  ext_mem_model_t queue_h;
  void *queue = cert_utils::setup_queue(&xaie, column, queue_h, 64);

  // load control code
  std::vector<uint8_t> control_code =
      cert_utils::load_ctrl_pages_from_elf("aie4.elf");

  std::vector<ext_mem_model_t> args;
  args.push_back(data_h);

  ext_mem_model_t signal_h =
      cert_utils::enqueue_control_code(&xaie, queue, control_code, args);
  mlir_aie_sync_mem_dev(queue_h);

  const uint32_t partition_size = 3;
  cert_utils::set_partition_size(column, partition_size);

  cert_utils::set_alive(column, 0);

  // wait for the completion signal to be set to state 'completed'
  uint64_t completion = 0;
  while (completion == 0) {
    completion = *reinterpret_cast<uint64_t *>(mlir_aie_sync_mem_cpu(signal_h));
    std::cout << "completion signal: " << completion << "\n";
    if (!completion)
      sleep(1);
  }

  if (completion != 4) {
    std::cout << "packet failed.\n";
    return 1;
  }

  uint32_t *data_out = mlir_aie_sync_mem_cpu(data_h);
  int errs = 0;
  // start checking at offset 4, it should be the data sent from the memtile
  for (int i = 8; i < 256; i++) {
    if (data_out[i] != 0xdeadbeef + (i - 8)) {
      errs++;
      printf("error @ %d : %x\n", i, data_out[i]);
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", 252 - errs, 252);
    return -1;
  }
}