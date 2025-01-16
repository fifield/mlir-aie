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

#include "memory_allocator.h"
#include "test_utils.h"

extern "C" {

typedef unsigned int uint;
void NpuWrite32(uint64_t Addr, uint Data);
uint NpuRead32(uint64_t Addr);
void NpuWriteTransactionBinary(unsigned char *data, size_t nbytes);
}

int main(int argc, char *argv[]) {
  printf("Hello, world.\n");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_ddr;
  handle_ddr.physicalAddr = 0;
  int *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle_ddr, 256 * sizeof(int));
  for (int i = 0; i < 256; i++)
    ddr_ptr_in[i] = 0x12345678 + i;
  mlir_aie_sync_mem_dev(handle_ddr);

  uint64_t col = 0;
  uint64_t row = 1;
  uint64_t tile_data_mem = ((col << 25) | (row << 20)) + 0x00100000 + 2048;
  for (int i = 0; i < 256; i++) {
    NpuWrite32(tile_data_mem + (i * 4), 0xdeadbeef + i);
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  NpuWriteTransactionBinary(reinterpret_cast<uint8_t *>(data.data()),
                            data.size() * sizeof(uint32_t));

  row = 0;
  uint64_t shim_lock_1 = ((col << 25) | (row << 20)) + 0x00010010;
  while (1) {
    uint32_t l = NpuRead32(shim_lock_1);
    if (l >= 1)
      break;
  }

  mlir_aie_sync_mem_cpu(handle_ddr);
  uint32_t *output = reinterpret_cast<uint32_t *>(handle_ddr.virtualAddr);
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