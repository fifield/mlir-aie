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
  int *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle, 3 * 256 * sizeof(int));
  for (int i = 0; i < 256 * 2; i++)
    ddr_ptr_in[i] = 5 + i;
  for (int i = 256 * 2; i < 256 * 3; i++)
    ddr_ptr_in[i] = (0xdead << 16) | (i - (256 * 2) + 1);
  mlir_aie_sync_mem_dev(handle);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t column = 0;
  uint32_t fw_state = 0;
  do {
    fw_state = cert_utils::get_fw_state(column, 0);
    std::cout << "fw_state: " << fw_state << "\n";
  } while (fw_state != 0);

  ext_mem_model_t queue_h;
  void *queue = cert_utils::setup_queue(&xaie, column, queue_h, 64);

  std::string control_code_path = "project1/ctrltext.0.0.bin";
  std::vector<uint8_t> control_code =
      test_utils::load_binary(control_code_path);
  std::string control_data_path = "project1/ctrldata.0.0.bin";
  std::vector<uint8_t> control_data =
      test_utils::load_binary(control_data_path);

  // Concatenate control_code and control_data into a single vector, since that
  // is what the firmware expects in memory.
  control_code.insert(control_code.end(), control_data.begin(),
                      control_data.end());
  std::vector<uint32_t> args(6, 0);
  args[0] = handle.physicalAddr - 0x80000000;
  args[2] = args[0] + (256 * sizeof(int32_t));
  args[4] = args[2] + (256 * sizeof(int32_t));
  ext_mem_model_t signal_h =
      cert_utils::enqueue_control_code(&xaie, queue, control_code, args);
  mlir_aie_sync_mem_dev(queue_h);

  const uint32_t partition_size = 3;
  cert_utils::set_partition_size(column, partition_size);

  cert_utils::set_alive(column, 0);

  // wait for the completion signal to be set to state 'completed'
  uint32_t completion = 0;
  uint32_t count = 0;
  while (completion != 4 /* HSA_CMD_STATE_COMPLETED */) {
    mlir_aie_sync_mem_cpu(signal_h);
    completion = *reinterpret_cast<uint32_t *>(signal_h.virtualAddr);
    std::cout << "completion signal: " << completion << "\n";
    if (completion == 5 /* HSA_CMD_STATE_ERROR */)
      break;
    if (completion == 6 /* HSA_CMD_STATE_ABORT */)
      break;
    if (completion >= 8 /* timeout, noresponse, skerror, skcrashed */)
      break;
    sleep(1);
    if (count++ > 300)
      break;
  }

  if (completion != 4) {
    std::cout << "packet failed.\n";
    // return 1;
  }

  mlir_aie_sync_mem_cpu(handle);
  uint32_t *output = reinterpret_cast<uint32_t *>(handle.virtualAddr);
  int errs = 0;
  for (int i = 0; i < 16; i++) {
    printf("error @ %d : %x\n", i, output[i]);
  }
  for (int i = 0; i < 256; i++) {
    if (output[i + 512] != (5 + i) * (256 + 5 + i)) {
      errs++;
      printf("error @ %d : %x\n", i, output[i + 512]);
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", 256 - errs, 256);
    return -1;
  }
}