//===- cert_utils.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _CERT_UTILS_H_
#define _CERT_UTILS_H_

#include "memory_allocator.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cert_utils {

// Allocate a buffer to hold the queue and initialize the queue header.
// Write the queue address to the hsa_addr fields in the cert handshake struct.
void *setup_queue(aie_libxaie_ctx_t *xaie, uint32_t column, ext_mem_model_t &h,
                  uint32_t queue_size);

// Set the alive bits in the handshake struct to the magic value, signaling to
// the cert firmware that it should start running.
void set_alive(uint32_t column, uint32_t id);

// Set the partition size in the cert handshake struct.
void set_partition_size(uint32_t column, uint32_t partition_size);

// Read the firmware state from the cert handshake struct.
uint32_t get_fw_state(uint32_t column, uint32_t id);

// Enqueue control code buffer, return a handle to the completion signal.
// Arguments are raw 32-bit integers passed directly to the runtime
ext_mem_model_t enqueue_control_code(aie_libxaie_ctx_t *xaie, void *queue,
                                     std::vector<uint8_t> &control_code,
                                     std::vector<uint32_t> args);

// Enqueue control code buffer, return a handle to the completion signal.
// Arguments are ext_mem_model_t handles (buffers) populated by
// mlir_aie_mem_alloc.
ext_mem_model_t enqueue_control_code(aie_libxaie_ctx_t *xaie, void *queue,
                                     std::vector<uint8_t> &control_code,
                                     std::vector<ext_mem_model_t> args = {});

std::vector<uint8_t> load_ctrl_pages_from_elf(const std::string &elf_path,
                                              uint32_t id = 0);

} // namespace cert_utils

#endif // _CERT_UTILS_H_
