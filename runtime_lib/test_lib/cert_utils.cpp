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
#include <fcntl.h>
#include <gelf.h>
#include <iostream>
#include <libelf.h>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

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

  // allocate a buffer to hold the queue
  void *queue_allocation = reinterpret_cast<void *>(mlir_aie_mem_alloc(
      xaie, h,
      (128 + sizeof(host_queue_header_t) + (queue_size * 64) + 3) / 4));
  // The firmware does not like the queue to be at address 0x80000000 (zero), so
  // we add 128 bytes to the physical address to avoid that, since the queue is
  // often the first thing allocated in host code. An extra 128-bytes is
  // allocated above.
  if (h.physicalAddr == 0x80000000) {
    h.physicalAddr += 128;
    h.size -= 128;
  }

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
  uint64_t uc_addr = (column << 25) | uc_p_dm_addr;
  // if (id)
  //   uc_addr += 0x00040000;
  mlir_aie_sim_write32(uc_addr + hsa_addr_high_offset, host_queue_paddr >> 32);
  mlir_aie_sim_write32(uc_addr + hsa_addr_low_offset,
                       host_queue_paddr & 0xffffffff);

  return queue_allocation;
}

// TODO: fixup the address calculations for columns other than zero

void cert_utils::set_alive(uint32_t column, uint32_t id) {
  uint64_t alive_offset = 0x0;
  uint32_t alive_magic = 0x404C5645;
  uint64_t uc_addr = (column << 25) | uc_p_dm_addr;
  if (id)
    uc_addr += 0x00040000;
  mlir_aie_sim_write32(uc_addr + alive_offset, alive_magic);
}

void cert_utils::set_partition_size(uint32_t column, uint32_t partition_size) {
  uint64_t partition_size_offset = 0x8;
  uint64_t uc_addr = (column << 25) | uc_p_dm_addr;
  // if (id)
  // uc_addr += 0x00040000;
  mlir_aie_sim_write32(uc_addr + partition_size_offset, partition_size);
}

uint32_t cert_utils::get_fw_state(uint32_t column, uint32_t id) {
  const uint64_t fw_state_offset = 0xa0; // offsetof(handshake_t, vm);
  uint64_t uc_addr = (column << 25) | uc_p_dm_addr;
  if (id)
    uc_addr += 0x00040000;
  return mlir_aie_sim_read32(uc_addr + fw_state_offset);
}

ext_mem_model_t
cert_utils::enqueue_control_code(aie_libxaie_ctx_t *xaie, void *queue,
                                 std::vector<uint8_t> &control_code,
                                 std::vector<ext_mem_model_t> args) {
  std::vector<uint32_t> int_args;
  for (auto &a : args) {
    int_args.push_back(a.physicalAddr - 0x80000000); // addr low
    int_args.push_back(0);                           // addr_ high
  }
  return enqueue_control_code(xaie, queue, control_code, int_args);
}

ext_mem_model_t
cert_utils::enqueue_control_code(aie_libxaie_ctx_t *xaie, void *queue,
                                 std::vector<uint8_t> &control_code,
                                 std::vector<uint32_t> args) {
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
  uint64_t *completion_signal = (uint64_t *)mlir_aie_mem_alloc(
      xaie, signal_h, sizeof(uint64_t) / sizeof(uint32_t));
  *completion_signal = 0;
  mlir_aie_sync_mem_dev(signal_h);
  xrt_header->completion_signal = signal_h.physicalAddr - 0x80000000;

  // allocate, populate and sync control code buffer
  ext_mem_model_t control_code_h;
  uint8_t *control_code_buf = reinterpret_cast<uint8_t *>(
      mlir_aie_mem_alloc(xaie, control_code_h, (control_code.size() + 3) / 4));

  std::memcpy(control_code_buf, control_code.data(), control_code.size());
  mlir_aie_sync_mem_dev(control_code_h);

  exec_buf_t *execution_buffer = reinterpret_cast<exec_buf_t *>(
      (uint8_t *)packet_buffer + sizeof(xrt_packet_header_t));
  memset(execution_buffer, 0, sizeof(exec_buf_t));

  uint64_t control_code_paddr = control_code_h.physicalAddr - 0x80000000;
  execution_buffer->dpu_control_code_host_addr_high =
      reinterpret_cast<uint64_t>(control_code_paddr) >> 32;
  execution_buffer->dpu_control_code_host_addr_low =
      reinterpret_cast<uint64_t>(control_code_paddr) & 0xffffffff;

  // allocate, populate and sync args buffer
  if (args.size()) {
    ext_mem_model_t args_h;
    size_t args_len = 2 + args.size();
    uint32_t *args_buf = mlir_aie_mem_alloc(xaie, args_h, args_len);

    args_buf[0] = execution_buffer->dpu_control_code_host_addr_low;
    args_buf[1] = execution_buffer->dpu_control_code_host_addr_high;
    int idx = 2;
    for (uint32_t a : args)
      args_buf[idx++] = a;
    mlir_aie_sync_mem_dev(args_h);

    uint64_t args_paddr = args_h.physicalAddr - 0x80000000;
    execution_buffer->args_host_addr_high =
        reinterpret_cast<uint64_t>(args_paddr) >> 32;
    execution_buffer->args_host_addr_low =
        reinterpret_cast<uint64_t>(args_paddr) & 0xffffffff;
    execution_buffer->args_len = args_len * sizeof(uint32_t);
  }
  return signal_h;
}

// Extract control sections from an ELF file into a map of of section names
// and their corresponding byte arrays.
std::map<std::string, std::vector<uint8_t>>
extract_ctrl_sections(const std::string &elf_path) {
  std::map<std::string, std::vector<uint8_t>> ctrl_sections;
  if (elf_version(EV_CURRENT) == EV_NONE) {
    std::cerr << "ELF library initialization failed!" << std::endl;
    return ctrl_sections;
  }

  int fd = open(elf_path.c_str(), O_RDONLY, 0);
  if (fd < 0) {
    std::cerr << "Failed to open ELF file: " << elf_path << std::endl;
    return ctrl_sections;
  }

  Elf *elf = elf_begin(fd, ELF_C_READ, nullptr);
  if (!elf) {
    std::cerr << "elf_begin() failed!" << std::endl;
    close(fd);
    return ctrl_sections;
  }

  size_t shstrndx;
  if (elf_getshdrstrndx(elf, &shstrndx) != 0) {
    std::cerr << "elf_getshdrstrndx() failed!" << std::endl;
    elf_end(elf);
    close(fd);
    return ctrl_sections;
  }

  Elf_Scn *scn = nullptr;
  while ((scn = elf_nextscn(elf, scn)) != nullptr) {
    GElf_Shdr shdr;
    if (gelf_getshdr(scn, &shdr) != &shdr)
      continue;
    const char *name = elf_strptr(elf, shstrndx, shdr.sh_name);
    if (!name)
      continue;
    if (strstr(name, "ctrltext") || strstr(name, "ctrldata")) {
      Elf_Data *data = elf_getdata(scn, nullptr);
      if (data && data->d_buf && data->d_size > 0) {
        const uint8_t *buf = reinterpret_cast<const uint8_t *>(data->d_buf);
        std::cout << "Extracted section: " << name << std::endl;
        ctrl_sections.insert(
            {name, std::vector<uint8_t>(buf, buf + data->d_size)});
      }
    }
  }

  elf_end(elf);
  close(fd);
  return ctrl_sections;
}

std::vector<uint8_t>
cert_utils::load_ctrl_pages_from_elf(const std::string &elf_path) {
  std::vector<uint8_t> ctrl_pages;
  auto sections = extract_ctrl_sections(elf_path);
  for (unsigned i = 0; i < sections.size() / 2; i++) {
    const auto &text_section = sections[".ctrltext.0." + std::to_string(i)];
    const auto &data_section = sections[".ctrldata.0." + std::to_string(i)];
    ctrl_pages.insert(ctrl_pages.end(), text_section.begin(),
                      text_section.end());
    ctrl_pages.insert(ctrl_pages.end(), data_section.begin(),
                      data_section.end());
  }
  return ctrl_pages;
}