//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host side of the NPU3 out-of-order S2MM tests. Sender s sends input slice
// s (64 words) with BD lookup index (s + 1) % 4, so the merged output holds
// slice s in slot (s + 1) % 4. Usage: test.exe <elf> <words>, where <words>
// is 256, or 264 when the design also loops back an 8-word sentinel.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

constexpr size_t NUM_SENDERS = 4;
constexpr size_t SLICE = 64;
constexpr size_t DATA_COUNT = NUM_SENDERS * SLICE;
constexpr uint32_t POISON = 0xdeadbeefu;
constexpr auto WAIT_TIMEOUT = std::chrono::seconds(5);

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <elf> <words>\n";
    return 2;
  }
  const std::string elfName = argv[1];
  const size_t count = std::strtoul(argv[2], nullptr, 0);
  if (count < DATA_COUNT) {
    std::cerr << "ERROR: <words> must be at least " << DATA_COUNT << '\n';
    return 2;
  }
  try {
    auto device = xrt::device(0);
    xrt::elf elf{elfName};
    xrt::hw_context context{device, elf};
    auto kernel = xrt::ext::kernel(context, "main:sequence");
    xrt::bo input = xrt::ext::bo{device, count * sizeof(uint32_t)};
    xrt::bo output = xrt::ext::bo{device, count * sizeof(uint32_t)};

    uint32_t *in = input.map<uint32_t *>();
    uint32_t *out = output.map<uint32_t *>();
    for (size_t i = 0; i < count; ++i) {
      in[i] = 0x00a40000u + static_cast<uint32_t>(i);
      out[i] = POISON;
    }
    input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(input, output);
    if (run.wait2(WAIT_TIMEOUT) == std::cv_status::timeout) {
      std::cout << "FAIL: " << elfName
                << " timed out waiting for the out-of-order merge\n";
      return 1;
    }
    output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (size_t s = 0; s < NUM_SENDERS; ++s) {
      size_t slot = (s + 1) % NUM_SENDERS;
      for (size_t i = 0; i < SLICE; ++i) {
        uint32_t expected = in[s * SLICE + i];
        uint32_t got = out[slot * SLICE + i];
        if (got != expected && errors++ < 16)
          std::cout << "slot " << slot << " [" << i << "]: expected 0x"
                    << std::hex << expected << " (sender " << std::dec << s
                    << "), got 0x" << std::hex << got << std::dec << '\n';
      }
    }
    for (size_t i = DATA_COUNT; i < count; ++i)
      if (out[i] != in[i] && errors++ < 16)
        std::cout << "sentinel [" << i - DATA_COUNT << "]: expected 0x"
                  << std::hex << in[i] << ", got 0x" << out[i] << std::dec
                  << '\n';

    if (errors) {
      std::cout << "FAIL: " << elfName << ": " << errors << " mismatches\n";
      return 1;
    }
    std::cout << "PASS: " << elfName
              << ": every packet landed in the BD its lookup index named\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << elfName << ": " << error.what() << '\n';
    return 2;
  }
}
