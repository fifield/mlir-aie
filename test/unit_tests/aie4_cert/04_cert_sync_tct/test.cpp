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
#include <unistd.h>

#include "cert_utils.h"
#include "memory_allocator.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t xaie;

  printf("Testing cert sync\n");

  ext_mem_model_t handle_ddr;
  handle_ddr.physicalAddr = 0;
  uint32_t *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle_ddr, 256);
  for (int i = 0; i < 256; i++)
    ddr_ptr_in[i] = 0x12345678 + i;
  mlir_aie_sync_mem_dev(handle_ddr);

  const uint32_t column = 0;
  uint64_t row = 2;

  uint64_t tile_data_mem = ((column << 25) | ((row + 4) << 20)) + 2048;
  for (int i = 0; i < 256; i++) {
    mlir_aie_sim_write32(tile_data_mem + (i * 4), 0xdeadbeef + i);
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

  std::vector<uint8_t> control_code =
      cert_utils::load_ctrl_pages_from_elf("aie4.elf");

  ext_mem_model_t signal_h =
      cert_utils::enqueue_control_code(&xaie, queue, control_code);
  mlir_aie_sync_mem_dev(queue_h);

  const uint32_t partition_size = 3;
  cert_utils::set_partition_size(column, partition_size);

  cert_utils::set_alive(column, 0);

  // wait for the completion signal to be set to state 'completed'
  uint32_t completion = 0;
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
  row = 0;
  uint64_t shim_lock_1 = ((column << 25) | (row << 20)) + 0x00010010;
  while (1) {
    uint32_t l = mlir_aie_sim_read32(shim_lock_1);
    if (l >= 1)
      break;
  }

  uint32_t *output = mlir_aie_sync_mem_cpu(handle_ddr);
  int errs = 0;
  for (int i = 0; i < 256; i++) {
    if (output[i] != 0xdeadbeef + i) {
      errs++;
      printf("error @ %d : %x\n", i, output[i]);
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