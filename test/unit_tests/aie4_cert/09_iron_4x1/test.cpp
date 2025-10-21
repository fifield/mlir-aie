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

static constexpr int data_size = 8 * 1024;

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t xaie;
  std::vector<ext_mem_model_t> buffers(3);
  uint32_t *ddr_ptr_in0 = mlir_aie_mem_alloc(&xaie, buffers[0], data_size);
  uint32_t *ddr_ptr_in1 = mlir_aie_mem_alloc(&xaie, buffers[1], data_size);
  uint32_t *ddr_ptr_out = mlir_aie_mem_alloc(&xaie, buffers[2], data_size);
  for (int i = 0; i < data_size; i++) {
    ddr_ptr_in0[i] = 5 + i;
    ddr_ptr_in1[i] = data_size + 5 + i;
    ddr_ptr_out[i] = (0xdead << 16) | (i - (data_size * 2) + 1);
  }
  for (auto &h : buffers)
    mlir_aie_sync_mem_dev(h);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t column = 0;
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

  uint32_t *output = mlir_aie_sync_mem_cpu(buffers[2]);
  int errs = 0;
  for (int i = 0; i < data_size; i++) {
    if (output[i] != (5 + i) * (data_size + 5 + i)) {
      errs++;
      printf("error @ %d : %x != %x\n", i, output[i],
             (5 + i) * (data_size + 5 + i));
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", data_size - errs, data_size);
    return -1;
  }
}