//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstring>
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

#define DTYPE int32_t
constexpr size_t DATA_COUNT = 64;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(DTYPE);

int main(int argc, const char *argv[]) {
  // Set up XRT
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"cert_multi_uc.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  // Initialize output buffer with sentinel values
  DTYPE *buf_out = bo_out.map<DTYPE *>();
  for (size_t i = 0; i < DATA_COUNT; i++)
    buf_out[i] = 0xdeadbeef;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Run - CERT jobs on two controllers coordinate via remote barriers, then
  // write to lock registers to release the cores. Each core fills a buffer
  // with a known pattern and sends it via objectfifo to shim DMA.
  auto t_start = std::chrono::high_resolution_clock::now();
  kernel(bo_out).wait2();
  auto t_stop = std::chrono::high_resolution_clock::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Validate output
  // Core 0 (tile 0,3) produces values 0..31
  // Core 1 (tile 1,2) produces values 100..131
  int errs = 0;
  for (int i = 0; i < 32; i++) {
    if (buf_out[i] != i) {
      std::cout << "Core 0 mismatch at [" << i << "]: got " << buf_out[i]
                << ", expected " << i << std::endl;
      errs++;
    }
  }
  for (int i = 0; i < 32; i++) {
    if (buf_out[32 + i] != 100 + i) {
      std::cout << "Core 1 mismatch at [" << (32 + i) << "]: got "
                << buf_out[32 + i] << ", expected " << (100 + i) << std::endl;
      errs++;
    }
  }

  // Report results
  float time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  std::cout << "Elapsed time for kernel execution: " << std::fixed
            << std::setprecision(0) << std::setw(8) << time << " us"
            << std::endl;
  if (errs == 0) {
    std::cout << "PASS!" << std::endl;
  } else {
    std::cout << "fail (" << errs << ")." << std::endl;
  }

  return (errs == 0 ? 0 : 1);
}
