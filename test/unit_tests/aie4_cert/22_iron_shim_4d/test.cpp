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

static constexpr int data_size = 64;

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t xaie;
  std::vector<ext_mem_model_t> buffers(2);
  uint32_t *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, buffers[0], data_size);
  uint32_t *ddr_ptr_out = mlir_aie_mem_alloc(&xaie, buffers[1], data_size);
  for (int i = 0; i < data_size; i++) {
    ddr_ptr_in[i] = i + 1;
    ddr_ptr_out[i] = 0xdeadbeef;
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

  // The shim MM2S read the input with the single-transfer 4D access pattern
  // sizes=[1,2,2,2,8], strides=[0,32,8,16,1] (order [iteration,d3,d2,d1,d0]),
  // a bijection of [0,64). The passthrough wrote the streamed elements back in
  // order, so ddr_out[k] == ddr_in[addr(k)] == addr(k) + 1, where k follows the
  // nested d3,d2,d1,d0 loops (no iteration: outermost size is 1).
  uint32_t *output = mlir_aie_sync_mem_cpu(buffers[1]);
  int errs = 0;
  int k = 0;
  for (int i3 = 0; i3 < 2; i3++)          // d3, stride 32
    for (int i2 = 0; i2 < 2; i2++)        // d2, stride 8
      for (int i1 = 0; i1 < 2; i1++)      // d1, stride 16
        for (int i0 = 0; i0 < 8; i0++) {  // d0, stride 1
          int addr = 32 * i3 + 8 * i2 + 16 * i1 + i0;
          uint32_t ref = static_cast<uint32_t>(addr + 1);
          if (output[k] != ref) {
            errs++;
            if (errs <= 16)
              printf("error @ %d : %x != %x\n", k, output[k], ref);
          }
          k++;
        }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", data_size - errs, data_size);
    return -1;
  }
}
