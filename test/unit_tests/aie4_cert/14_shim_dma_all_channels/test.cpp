//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices Inc.
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

static constexpr int data_size = 8192;

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t xaie;

  std::vector<ext_mem_model_t> buffers(2);

  // Allocate DDR buffer for input data (8 columns x 1024 words)
  uint32_t *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, buffers[0], data_size);
  for (int i = 0; i < data_size; i++)
    ddr_ptr_in[i] = i + 1;
  mlir_aie_sync_mem_dev(buffers[0]);

  // Allocate DDR buffer for output data
  uint32_t *ddr_ptr_out = mlir_aie_mem_alloc(&xaie, buffers[1], data_size);
  for (int i = 0; i < data_size; i++)
    ddr_ptr_out[i] = 0;
  mlir_aie_sync_mem_dev(buffers[1]);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  const uint32_t column = 0;
  uint32_t fw_state = 0;
  do {
    fw_state = cert_utils::get_fw_state(column, 0);
    std::cout << "fw_state: " << fw_state << "\n";
  } while (fw_state != 0xffe);

  ext_mem_model_t queue_h;
  void *queue = cert_utils::setup_queue(&xaie, column, queue_h, 64);

  std::vector<uint8_t> control_code =
      cert_utils::load_ctrl_pages_from_elf("aie4.elf");

  ext_mem_model_t signal_h =
      cert_utils::enqueue_control_code(&xaie, queue, control_code, buffers);
  mlir_aie_sync_mem_dev(queue_h);

  const uint32_t partition_size = 9;
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

  // Sync output buffer from device
  mlir_aie_sync_mem_cpu(buffers[1]);

  int errors = 0;
  for (uint32_t i = 0; i < data_size; i++) {
    uint32_t ref = i + 1;
    if (ddr_ptr_out[i] != ref) {
      errors++;
      std::cout << "Error at index " << i << ": expected " << ref << ", got "
                << ddr_ptr_out[i] << std::endl;
    }
  }

  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
