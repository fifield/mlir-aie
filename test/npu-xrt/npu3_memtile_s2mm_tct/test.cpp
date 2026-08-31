//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bound the waits so a regression in the compiler's MEM S2MM actor mapping or
// TCT return routing reports a failure instead of hanging the NPU3 test worker
// indefinitely.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

constexpr size_t DATA_COUNT = 256;
constexpr size_t BUFFER_SIZE = DATA_COUNT * sizeof(uint32_t);
constexpr auto WAIT_TIMEOUT = std::chrono::seconds(5);

} // namespace

int main() {
  try {
    auto device = xrt::device(0);
    xrt::elf elf{"memtile_s2mm_tct.elf"};
    xrt::hw_context context{device, elf};
    auto kernel = xrt::ext::kernel(context, "main:sequence");
    xrt::bo input = xrt::ext::bo{device, BUFFER_SIZE};

    uint32_t *data = input.map<uint32_t *>();
    for (size_t i = 0; i < DATA_COUNT; ++i)
      data[i] = 0xa5e40000u + static_cast<uint32_t>(i);
    input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(input);
    if (run.wait2(WAIT_TIMEOUT) == std::cv_status::timeout) {
      std::cout << "FAIL: command timed out waiting for MEM-tile S2MM actors "
                   "0, 1, 2, and 7\n";
      return 1;
    }

    std::cout << "PASS: MEM-tile S2MM actors 0, 1, 2, and 7 satisfied their "
                 "WAIT_TCTS operations\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: NPU3 dispatch failed while testing WAIT_TCTS: "
              << error.what() << '\n';
    return 2;
  }
}
