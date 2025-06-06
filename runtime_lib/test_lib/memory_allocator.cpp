//===- memory_allocator.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"
#include <assert.h>
#include <iostream>

extern "C" {

void ess_WriteGM(uint64_t addr, const void *data, uint64_t size);
void ess_ReadGM(uint64_t addr, void *data, uint64_t size);
}

extern "C" {
  
  void ess_Write32(uint64_t addr, uint32_t Data);
  uint32_t ess_Read32(uint64_t addr);
}

#define XAIE_BASE_ADDR 0x20000000000

void mlir_aie_sim_write32(uint64_t addr, uint32_t data) {
  ess_Write32(XAIE_BASE_ADDR + addr, data);
}

uint32_t mlir_aie_sim_read32(uint64_t addr) {
  return ess_Read32(XAIE_BASE_ADDR + addr);
}

// static variable for tracking current DDR physical addr during AIESIM
static uint64_t nextAlignedAddr = 0x80000000;

uint32_t *mlir_aie_mem_alloc(aie_libxaie_ctx_t *_xaie, ext_mem_model_t &handle,
                             int size) {
  uint64_t size_bytes = size * sizeof(int);
  handle.virtualAddr = std::malloc(size_bytes);
  if (handle.virtualAddr) {
    handle.size = size_bytes;
    // assign physical space in SystemC DDR memory controller
    handle.physicalAddr = nextAlignedAddr;
    // adjust nextAlignedAddr to the next 128-bit aligned address
    nextAlignedAddr = nextAlignedAddr + size_bytes;
    uint64_t gapToAligned = nextAlignedAddr % 16; // 16byte (128bit)
    if (gapToAligned > 0)
      nextAlignedAddr += (16 - gapToAligned);
  } else {
    printf("ExtMemModel: Failed to allocate %ld memory.\n", size_bytes);
    return nullptr;
  }

  ctx->allocations.push_back(handle);

  return (uint32_t *)handle.virtualAddr;
}

uint32_t *mlir_aie_sync_mem_cpu(ext_mem_model_t &handle) {
  aiesim_ReadGM(handle.physicalAddr, handle.virtualAddr, handle.size);
  return (uint32_t *)handle.virtualAddr;
}

void mlir_aie_sync_mem_dev(ext_mem_model_t &handle) {
  ess_WriteGM(handle.physicalAddr, handle.virtualAddr, handle.size);
}

u64 mlir_aie_get_device_address(aie_libxaie_ctx_t *ctx, void *VA) {
  std::cout << "get_device_address: " << ctx << " VA " << std::hex << VA
            << std::dec << "\n";
  for (auto i : ctx->allocations) {
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