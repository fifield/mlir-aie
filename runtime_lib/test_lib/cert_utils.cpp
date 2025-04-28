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

enum host_queue_packet_type {
  HOST_QUEUE_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  HOST_QUEUE_PACKET_TYPE_INVALID = 1,
};

struct exec_buf {
  uint16_t cu_index;
  uint16_t reserved0;
  uint32_t dpu_control_code_host_addr_low;
  uint32_t dpu_control_code_host_addr_high;
  uint16_t args_len;
  uint16_t reserved1;
  uint32_t args_host_addr_low;
  uint32_t args_host_addr_high;
};
typedef exec_buf exec_buf_t;

struct common_header {
  union {
    struct {
      uint16_t type : 8;
      uint16_t barrier : 1;
      uint16_t acquire_fence_scope : 2;
      uint16_t release_fence_scope : 2;
    };
    uint16_t header;
  };
  uint16_t opcode;
  uint16_t count;
  uint8_t distribute;
  uint8_t indirect;
};

struct xrt_packet_header {
  struct common_header common_header;
  uint64_t completion_signal;
};
typedef struct xrt_packet_header xrt_packet_header_t;

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

  host_queue->read_index = 0;
  host_queue->write_index = 0;

  uint64_t packet_buffer = reinterpret_cast<uint64_t>(queue_allocation) +
                           sizeof(host_queue_header_t);

  // initialize the packets to zero and set their type to invalid (1)
  std::memset(reinterpret_cast<void *>(packet_buffer), 0, queue_size * 64);
  for (int i = 0; i < queue_size; i++) {
    uint16_t *type = reinterpret_cast<uint16_t *>((i * 64) + packet_buffer);
    *type = HOST_QUEUE_PACKET_TYPE_INVALID;
  }
  // set host_queue address field to be the buffer address
  uint64_t buffer_paddr =
      (h.physicalAddr - 0x80000000) + sizeof(host_queue_header_t);
  host_queue->data_address = buffer_paddr;

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

ext_mem_model_t
cert_utils::enqueue_control_code(aie_libxaie_ctx_t *xaie, void *queue,
                                 std::vector<uint16_t> &control_code) {
  host_queue_header_t *host_queue =
      reinterpret_cast<host_queue_header_t *>(queue);

  uint64_t write_index = host_queue->write_index;
  host_queue->write_index = (write_index + 1) % host_queue->capacity;

  void *packet_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uint64_t>(host_queue) + sizeof(host_queue_header_t) +
      (64 * write_index));

  // create an xrt hsa packet in the packet buffer
  xrt_packet_header_t *xrt_header = (xrt_packet_header_t *)packet_buffer;
  xrt_header->common_header.type = HOST_QUEUE_PACKET_TYPE_VENDOR_SPECIFIC;
  xrt_header->common_header.opcode = 1; // exec_buf
  xrt_header->common_header.count = sizeof(exec_buf_t);

  // allocate completion signal and set it to 0
  ext_mem_model_t signal_h;
  uint32_t *completion_signal =
      (uint32_t *)mlir_aie_mem_alloc(xaie, signal_h, sizeof(uint32_t));
  *completion_signal = 0;
  mlir_aie_sync_mem_dev(signal_h);
  xrt_header->completion_signal = signal_h.physicalAddr - 0x80000000;

  // allocate, populate and sync control code buffer
  ext_mem_model_t control_code_h;
  uint16_t *control_code_buf = reinterpret_cast<uint16_t *>(mlir_aie_mem_alloc(
      xaie, control_code_h, control_code.size() * sizeof(uint16_t)));

  std::memcpy(control_code_buf, control_code.data(),
              control_code.size() * sizeof(uint16_t));
  mlir_aie_sync_mem_dev(control_code_h);

  exec_buf_t *execution_buffer = reinterpret_cast<exec_buf_t *>(
      (uint8_t *)packet_buffer + sizeof(xrt_packet_header_t));

  uint64_t control_code_paddr = control_code_h.physicalAddr - 0x80000000;
  execution_buffer->dpu_control_code_host_addr_high =
      reinterpret_cast<uint64_t>(control_code_paddr) >> 32;
  execution_buffer->dpu_control_code_host_addr_low =
      reinterpret_cast<uint64_t>(control_code_paddr) & 0xffffffff;

  return signal_h;
}