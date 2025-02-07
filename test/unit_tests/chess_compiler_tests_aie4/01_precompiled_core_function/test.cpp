//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {
  std::cout << "test start.\n";

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 1;
  uint64_t row = 3;

  uint64_t tile_data_mem = ((col << 25) | ((row + 4) << 20)) + 1024 + 1024;
  for (int i = 0; i < 256; i++) {
    mlir_aie_sim_write32(tile_data_mem + (i * 4), i + 1);
  }

  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  uint64_t tile_lock_3 = ((col << 25) | ((row + 4) << 20)) + 0x000AE030;

  std::cout << "Read buffer lock\n";
  uint32_t lock1 = mlir_aie_sim_read32(tile_lock_1);
  if (lock1 != 0) {
    std::cout << "Unexpected initial lock value: " << lock1 << "\n";
    return -1;
  }

  std::cout << "Assign buffer lock\n";
  mlir_aie_sim_write32(tile_lock_1, 1);

  std::cout << "Wait for completion\n";
  while (1) {
    uint32_t lock3 = mlir_aie_sim_read32(tile_lock_3);
    if (lock3 >= 1)
      break;
  }
  std::cout << "Done running\n";

  tile_data_mem = ((col << 25) | ((row + 4) << 20)) + 32768 + 1024;
  int32_t result = mlir_aie_sim_read32(tile_data_mem);
  std::cout << "result " << result << "\n";

  int errors = 42 - result;
  if (!errors) {
    std::cout << "PASS!\n";
    return 0;
  } else {
    std::cout << "failed.\n";
    return -1;
  }
}
