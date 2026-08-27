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
#include <cstring>
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

constexpr uint32_t kShimColumn = 0;
constexpr uint64_t kShimMm2sChannel0Status =
    (static_cast<uint64_t>(kShimColumn) << 25) | 0x00018580;
constexpr uint64_t kShimLock1Address =
    (static_cast<uint64_t>(kShimColumn) << 25) | 0x00010010;

} // namespace

int main(int argc, char *argv[]) {
  std::cout << "Testing AIE4 control packet DMA path\n";

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle;
  uint32_t *ctrlPktDdr = mlir_aie_mem_alloc(&xaie, handle, 3);
  if (!ctrlPktDdr) {
    std::cerr << "Failed to allocate control packet DDR buffer\n";
    return 1;
  }

  std::vector<uint32_t> ctrlPktWords =
      test_utils::load_instr_binary("ctrlpkt.bin");
  if (ctrlPktWords.size() != 3 || ctrlPktWords[0] != 0x1b ||
      ctrlPktWords[1] != 0x0002b800 || ctrlPktWords[2] != 2) {
    std::cerr << "Unexpected control packet encoding\n";
    return 1;
  }
  std::memcpy(ctrlPktDdr, ctrlPktWords.data(),
              ctrlPktWords.size() * sizeof(uint32_t));
  mlir_aie_sync_mem_dev(handle);

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

  std::cout << "Shim MM2S0 status before wait: 0x" << std::hex
            << mlir_aie_sim_read32(kShimMm2sChannel0Status) << std::dec
            << "\n";
  std::cout << "Shim completion lock before wait: 0x" << std::hex
            << mlir_aie_sim_read32(kShimLock1Address) << std::dec << "\n";

  test_utils::wait_on_lock(kLock0Address, 2);

  std::cout << "Shim MM2S0 status after wait: 0x" << std::hex
            << mlir_aie_sim_read32(kShimMm2sChannel0Status) << std::dec
            << "\n";
  std::cout << "Shim completion lock after wait: 0x" << std::hex
            << mlir_aie_sim_read32(kShimLock1Address) << std::dec << "\n";

  uint32_t lockValue = mlir_aie_sim_read32(kLock0Address);
  if (lockValue != 2) {
    std::cerr << "Control packet DMA did not update lock register\n";
    return 1;
  }

  std::cout << "Control packet lock value: 0x" << std::hex << lockValue
            << std::dec << "\n";
  std::cout << "PASS!\n";
  return 0;
}
