//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <control.elf>\n";
    return 1;
  }

  constexpr size_t wordCount = 64;
  constexpr uint32_t sentinel = 0xDEADBEEF;
  constexpr uint32_t guard = 0x0BADF00D;
  auto device = xrt::device(0);
  auto context = xrt::hw_context(device, xrt::elf{argv[1]});
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo output = xrt::ext::bo{device, wordCount * sizeof(uint32_t)};
  auto *words = output.map<uint32_t *>();
  std::fill_n(words, wordCount, sentinel);
  output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(output);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cerr << "FAIL: nonzero-group uC DMA readback timed out after 10s\n";
    return 1;
  }
  output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  unsigned errors = 0;
  for (size_t i = 0; i < wordCount; ++i) {
    uint32_t expected = guard;
    if (i < 16 || (i >= 32 && i < 48)) {
      size_t payloadIndex = i % 32;
      expected = (payloadIndex < 8 ? 0x11110000 : 0x22220000) +
                 static_cast<uint32_t>(payloadIndex % 8 + 1);
    }
    if (words[i] != expected) {
      ++errors;
      std::cerr << "word[" << i << "]: got 0x" << std::hex << words[i]
                << ", expected 0x" << expected << std::dec << '\n';
    }
  }
  if (errors)
    return 1;
  std::cout << "PASS: group 2 uC DMA payloads and guards match (64 words)\n";
  return 0;
}
