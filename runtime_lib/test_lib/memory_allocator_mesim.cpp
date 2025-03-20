//===- memory_allocator_mesim.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023-2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"

#include <assert.h>
#include <iostream>

extern "C" {
void mesim_ReadGM(uint64_t address, void *data, uint64_t size);
void mesim_WriteGM(uint64_t address, void *data, uint64_t size);

uint32_t mesim_Read32(uint64_t addr);
void mesim_Write32(uint64_t addr, uint32_t Data);

void mlir_aie_sim_write32(uint64_t addr, uint32_t data) {
  mesim_Write32(addr, data);
}

uint32_t mlir_aie_sim_read32(uint64_t addr) { return mesim_Read32(addr); }
}

// static variable for tracking current DDR physical addr during MESIM
static uint64_t nextPhysicalAddr = 0x80000000;

int *mlir_aie_mem_alloc(aie_libxaie_ctx_t *_xaie, ext_mem_model_t &handle,
                        int size) {
  uint64_t size_bytes = size * sizeof(int);
  handle.virtualAddr = std::malloc(size_bytes);
  if (handle.virtualAddr) {
    handle.size = size_bytes;
    // assign physical space in SystemC DDR memory controller
    handle.physicalAddr = nextPhysicalAddr;
    // adjust nextPhysicalAddr to the next 128-bit aligned address
    nextPhysicalAddr = nextPhysicalAddr + size_bytes;
    uint64_t gapToAligned = nextPhysicalAddr % 16; // 16byte (128bit)
    if (gapToAligned > 0)
      nextPhysicalAddr += (16 - gapToAligned);
  } else {
    printf("Failed to allocate %d bytes of memory.\n", size_bytes);
    return nullptr;
  }

  _xaie->allocations.push_back(handle);

  return (int *)handle.virtualAddr;
}

void mlir_aie_sync_mem_cpu(ext_mem_model_t &handle) {
  mesim_ReadGM(handle.physicalAddr, handle.virtualAddr, handle.size);
}

void mlir_aie_sync_mem_dev(ext_mem_model_t &handle) {
  mesim_WriteGM(handle.physicalAddr, handle.virtualAddr, handle.size);
}

u64 mlir_aie_get_device_address(aie_libxaie_ctx_t *_xaie, void *VA) {
  std::cout << "get_device_address: " << _xaie << " VA " << std::hex << VA
            << std::dec << "\n";
  for (auto i : _xaie->allocations) {
    std::cout << "get_device_address: virtual address " << std::hex
              << i.virtualAddr << ", physical address " << i.physicalAddr
              << ", size " << std::dec << i.size << std::endl;

    if (i.virtualAddr == VA)
      return i.physicalAddr;
  }
  printf("ERROR: cannot get device address for allocation!\n");
  assert(false);
  return 0;
}