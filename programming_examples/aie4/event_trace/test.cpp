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
// XRT host code for event_trace example.
// Runs vector-scalar-multiply, optionally with trace collection.
//
// Usage: ./test.exe [elf_file]
//   elf_file defaults to "event_trace.elf" (with trace)
//   Use "base.elf" to run without trace
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Buffer sizes -- must match the runtime_sequence arguments in aie_trace.mlir
constexpr int IN_ELEMS = 4096;
constexpr int FACT_ELEMS = 1;
constexpr int OUT_ELEMS = 4096;

constexpr size_t IN_SIZE = IN_ELEMS * sizeof(int32_t);
constexpr size_t FACT_SIZE = FACT_ELEMS * sizeof(int32_t);
constexpr size_t OUT_SIZE = OUT_ELEMS * sizeof(int32_t);
// buffer_size on aie.trace.host_config is in bytes, not elements
constexpr size_t TRACE_SIZE = 8192;

int main(int argc, const char *argv[]) {
  std::string elfFile = (argc > 1) ? argv[1] : "event_trace.elf";
  bool enableTrace = (elfFile.find("base") == std::string::npos);

  std::cout << "ELF: " << elfFile << " (trace "
            << (enableTrace ? "enabled" : "disabled") << ")\n";

  // Initialize input data
  std::vector<int32_t> vec_in(IN_ELEMS);
  std::vector<int32_t> vec_factor(FACT_ELEMS);
  std::vector<int32_t> vec_out(OUT_ELEMS);

  for (int i = 0; i < IN_ELEMS; i++)
    vec_in[i] = i + 1;
  vec_factor[0] = 3; // scale factor

  // Set up XRT
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  // Allocate buffers
  xrt::bo bo_in = xrt::ext::bo{device, IN_SIZE};
  xrt::bo bo_factor = xrt::ext::bo{device, FACT_SIZE};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_SIZE};

  // Map and fill buffers
  char *buf_in = bo_in.map<char *>();
  char *buf_factor = bo_factor.map<char *>();
  char *buf_out = bo_out.map<char *>();

  memcpy(buf_in, vec_in.data(), IN_SIZE);
  memcpy(buf_factor, vec_factor.data(), FACT_SIZE);
  memset(buf_out, 0, OUT_SIZE);

  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_factor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Run kernel
  auto t_start = std::chrono::high_resolution_clock::now();
  if (enableTrace) {
    xrt::bo bo_trace = xrt::ext::bo{device, TRACE_SIZE};
    char *buf_trace = bo_trace.map<char *>();
    memset(buf_trace, 0, TRACE_SIZE);

    kernel(bo_in, bo_factor, bo_out, bo_trace).wait2();

    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Write trace buffer to file for offline analysis.  parse.py expects text:
    // one zero-padded 8-digit hex word per line (same as
    // test_utils::write_out_trace), not raw binary.
    std::ofstream trace_file("trace.bin");
    const uint32_t *words = bo_trace.map<uint32_t *>();
    for (size_t i = 0; i < TRACE_SIZE / sizeof(uint32_t); i++)
      trace_file << std::setfill('0') << std::setw(8) << std::hex << words[i]
                 << std::endl;
    trace_file.close();
    printf("trace written to trace.bin (%zu bytes)\n", TRACE_SIZE);
  } else {
    kernel(bo_in, bo_factor, bo_out).wait2();
  }
  auto t_stop = std::chrono::high_resolution_clock::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Verify output: out[i] should equal (i+1) * 3
  memcpy(vec_out.data(), buf_out, OUT_SIZE);
  int errs = 0;
  for (int i = 0; i < OUT_ELEMS; i++) {
    int32_t expected = (i + 1) * 3;
    if (vec_out[i] != expected) {
      if (errs < 10)
        printf("error @ %d: got %d, expected %d\n", i, vec_out[i], expected);
      errs++;
    }
  }

  // Report results
  float time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  std::cout << "Elapsed time: " << std::fixed << std::setprecision(0)
            << std::setw(8) << time << " us" << std::endl;

  if (errs == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    printf("fail (%d/%d).\n", OUT_ELEMS - errs, OUT_ELEMS);
    return 1;
  }
}
