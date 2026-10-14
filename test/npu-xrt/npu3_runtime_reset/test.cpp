// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: test.exe core|dma");
    const std::string mode = argv[1];
    if (mode != "core" && mode != "dma")
      throw std::runtime_error("unknown test mode: " + mode);
    const int length = mode == "core" ? 16 : 8;
    const int dispatches = mode == "core" ? 1 : 8;

    auto device = xrt::device(0);
    auto elf = xrt::elf("reset.elf");
    auto context = xrt::hw_context(device, elf);
    auto kernel = xrt::ext::kernel(context, "main:seq");
    xrt::bo outputBo = xrt::ext::bo(device, length * sizeof(uint32_t));
    auto *output = outputBo.map<uint32_t *>();

    // Keep one context/ELF loaded and fill a sentinel before each dispatch.
    for (int dispatch = 0; dispatch < dispatches; ++dispatch) {
      std::fill(output, output + length, 0xdeadbeef);
      outputBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto run = kernel(outputBo);
      if (run.wait2(std::chrono::seconds(5)) == std::cv_status::timeout) {
        run.abort();
        throw std::runtime_error("timeout at dispatch " +
                                 std::to_string(dispatch));
      }
      outputBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      for (int i = 0; i < length; ++i) {
        // The core must emit the initialized value 41, reset, then emit 42
        // while retaining its counter in data memory. DMA must discard the bad
        // BD.
        uint32_t expected = mode == "dma" ? 100 + i : 41 + i / 8;
        if (output[i] != expected) {
          std::cerr << mode << " dispatch " << dispatch << " element " << i
                    << ": expected " << expected << ", got " << output[i]
                    << '\n';
          return 1;
        }
      }
    }
    std::cout << "PASS: " << mode << " (" << dispatches << " dispatches)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
