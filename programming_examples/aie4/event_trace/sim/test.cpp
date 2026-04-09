//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Mesimulator host code for event_trace example.
// Runs vector-scalar-multiply with trace collection via mesimulator.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <unistd.h>

#include "cert_utils.h"
#include "memory_allocator.h"
#include "test_utils.h"

// Buffer sizes (in elements)
static constexpr int IN_ELEMS = 8192;                       // in: 8192 x i32
static constexpr int FACT_ELEMS = 1;                        // infactor: 1 x i32
static constexpr int OUT_ELEMS = 8192;                      // out: 8192 x i32
static constexpr int TRACE_BYTES = 8192 * sizeof(uint32_t); // trace buffer

int main(int argc, char *argv[]) {
  aie_libxaie_ctx_t xaie;

  // Allocate and initialize data buffers.
  // Order matches runtime_sequence args: arg0=in, arg1=infactor, arg2=out,
  // arg3=trace
  std::vector<ext_mem_model_t> buffers(4);

  uint32_t *in_ptr = mlir_aie_mem_alloc(&xaie, buffers[0], IN_ELEMS);
  uint32_t *fact_ptr = mlir_aie_mem_alloc(&xaie, buffers[1], FACT_ELEMS);
  uint32_t *out_ptr = mlir_aie_mem_alloc(&xaie, buffers[2], OUT_ELEMS);
  uint32_t *trace_ptr =
      mlir_aie_mem_alloc(&xaie, buffers[3], TRACE_BYTES / sizeof(uint32_t));

  // Initialize inputs
  for (int i = 0; i < IN_ELEMS; i++)
    in_ptr[i] = i + 1;
  fact_ptr[0] = 3; // scale factor
  memset(out_ptr, 0, OUT_ELEMS * sizeof(uint32_t));
  memset(trace_ptr, 0, TRACE_BYTES);

  for (auto &h : buffers)
    mlir_aie_sync_mem_dev(h);

  // Load and submit transaction binary
  std::string txn_path = "project/transaction.bin";
  std::vector<uint32_t> txn_data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(
      reinterpret_cast<uint8_t *>(txn_data.data()),
      txn_data.size() * sizeof(uint32_t));

  // Wait for firmware to become ready
  const uint64_t column = 0;
  uint32_t fw_state = 0;
  do {
    fw_state = cert_utils::get_fw_state(column, 0);
    std::cout << "fw_state: " << fw_state << "\n";
  } while (fw_state != 0xffe);

  // Set up command queue
  ext_mem_model_t queue_h;
  void *queue = cert_utils::setup_queue(&xaie, column, queue_h, 64);

  // Load and enqueue control code (with data buffer args for address patching)
  std::vector<uint8_t> control_code =
      cert_utils::load_ctrl_pages_from_elf("project/aie4.elf");
  ext_mem_model_t signal_h =
      cert_utils::enqueue_control_code(&xaie, queue, control_code, buffers);
  mlir_aie_sync_mem_dev(queue_h);

  // Set partition size (col 0..2 = 3 columns)
  const uint32_t partition_size = 3;
  cert_utils::set_partition_size(column, partition_size);
  cert_utils::set_alive(column, 0);

  // Wait for completion signal
  uint64_t completion = 0;
  while (completion == 0) {
    completion = *reinterpret_cast<uint64_t *>(mlir_aie_sync_mem_cpu(signal_h));
    std::cout << "completion signal: " << completion << "\n";
    if (!completion)
      sleep(1);
  }

  if (completion != 4) {
    std::cout << "packet failed (completion=" << completion << ").\n";
    return 1;
  }

  // Verify output: out[i] should equal (i+1) * 3
  uint32_t *output = mlir_aie_sync_mem_cpu(buffers[2]);
  int errs = 0;
  for (int i = 0; i < OUT_ELEMS; i++) {
    uint32_t expected = (i + 1) * 3;
    if (output[i] != expected) {
      if (errs < 10)
        printf("error @ %d: got %u, expected %u\n", i, output[i], expected);
      errs++;
    }
  }

  // Write trace buffer to file for offline analysis
  uint32_t *trace =
      reinterpret_cast<uint32_t *>(mlir_aie_sync_mem_cpu(buffers[3]));
  std::ofstream trace_file("trace.bin", std::ios::binary);
  trace_file.write(reinterpret_cast<char *>(trace), TRACE_BYTES);
  trace_file.close();
  printf("trace written to trace.bin (%d bytes)\n", TRACE_BYTES);

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", OUT_ELEMS - errs, OUT_ELEMS);
    return -1;
  }
}
