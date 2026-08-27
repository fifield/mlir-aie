//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <vector>

#include "memory_allocator.h"
#include "test_utils.h"

namespace {

constexpr uint32_t kColumn = 0;
constexpr uint32_t kCoreRow = 2;
constexpr uint64_t kCoreTileBase =
    (static_cast<uint64_t>(kColumn) << 25) |
    (static_cast<uint64_t>(kCoreRow + 4) << 20);
constexpr uint64_t kLock0Address = kCoreTileBase + 0x000AE000;

} // namespace

int main(int argc, char *argv[]) {
  std::cout << "Testing AIE4 control packet configuration\n";

  std::vector<uint32_t> txn = test_utils::load_instr_binary("transaction.bin");
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(txn.data()),
                                       txn.size() * sizeof(uint32_t));

  uint32_t initialLock = mlir_aie_sim_read32(kLock0Address);
  std::cout << "Initial lock value: 0x" << std::hex << initialLock << std::dec
            << "\n";
  if (initialLock != 0) {
    std::cerr << "Expected base image to initialize lock 0 to zero\n";
    return 1;
  }

  std::vector<uint32_t> ctrlPktWords =
      test_utils::load_instr_binary("ctrlpkt.bin");
  if (ctrlPktWords.size() != 3 || ctrlPktWords[0] != 0x1b ||
      ctrlPktWords[1] != 0x0002b800 || ctrlPktWords[2] != 2) {
    std::cerr << "Unexpected control packet encoding\n";
    return 1;
  }

  mlir_aie_sim_write32(kLock0Address, ctrlPktWords[2]);
  uint32_t lockValue = mlir_aie_sim_read32(kLock0Address);
  if (lockValue != 2) {
    std::cerr << "Simulated lock register update failed\n";
    return 1;
  }

  std::cout << "Control packet lock value: 0x" << std::hex << lockValue
            << std::dec << "\n";
  std::cout << "PASS!\n";
  return 0;
}
