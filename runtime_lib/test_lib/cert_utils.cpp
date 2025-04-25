//===- cert_utils.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cert_utils.h"

#include <cassert>
#include <cstring>

namespace {

struct host_queue_header_t {
  uint64_t read_index;
  struct {
    uint16_t major;
    uint16_t minor;
  } version;
  uint32_t capacity; // Queue capacity, must be a power of two.
  uint64_t write_index;
  uint64_t data_address;
};

} // namespace

static const uint64_t uc_p_dm_addr = 0x28000;

void *cert_utils::setup_queue(aie_libxaie_ctx_t *xaie, uint32_t column,
                              ext_mem_model_t &h, uint32_t queue_size) {

  assert(!column && "column must be 0");

  // allocate a buffer to hold the queue
  void *queue_allocation = reinterpret_cast<void *>(mlir_aie_mem_alloc(
      xaie, h, sizeof(host_queue_header_t) + (queue_size * 64)));

  host_queue_header_t *host_queue =
      reinterpret_cast<host_queue_header_t *>(queue_allocation);

  void *packet_buffer =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t>(queue_allocation) +
                               sizeof(host_queue_header_t));

  // initialize the queue packets to zero
  std::memset(packet_buffer, 0, queue_size * 64);

  // set host_queue address field to be the buffer address
  host_queue->capacity = queue_size;
  host_queue->data_address = reinterpret_cast<uint64_t>(packet_buffer);

  uint64_t hsa_addr_high_offset = 0xc;
  uint64_t hsa_addr_low_offset = 0x10;
  // The cert firmware sets the axi offset register to 0x80000000 out of the box
  // and in order to make that work the external memory base address is the same
  // in the simulator. The allocator also starts at that offset (so that sync
  // works), so the offset must be subtracted here because the controller will
  // dereference it and the axi offset will get added.
  uint64_t host_queue_paddr = h.physicalAddr - 0x80000000;
  mlir_aie_sim_write32(uc_p_dm_addr + hsa_addr_high_offset,
                       host_queue_paddr >> 32);
  mlir_aie_sim_write32(uc_p_dm_addr + hsa_addr_low_offset,
                       host_queue_paddr & 0xffffffff);

  return queue_allocation;
}

// TODO: fixup the address calculations for columns other than zero

void cert_utils::set_alive(uint32_t column, uint32_t id) {
  assert(!column && "column must be 0");
  assert(!id && "id must be 0");
  uint64_t alive_offset = 0x0;
  uint32_t alive_magic = 0x404C5645;
  mlir_aie_sim_write32(uc_p_dm_addr + alive_offset, alive_magic);
}

void cert_utils::set_partition_size(uint32_t column, uint32_t partition_size) {
  assert(!column && "column must be 0");
  assert(partition_size && "partition size must be non-zero");
  uint64_t partition_size_offset = 0x8;
  mlir_aie_sim_write32(uc_p_dm_addr + partition_size_offset, partition_size);
}

uint32_t cert_utils::get_fw_state(uint32_t column, uint32_t id) {
  assert(!column && "column must be 0");
  assert(!id && "id must be 0");
  const uint64_t fw_state_offset = 0xa0; // offsetof(handshake_t, vm);
  return mlir_aie_sim_read32(uc_p_dm_addr + fw_state_offset);
}