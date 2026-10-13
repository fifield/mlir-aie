// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

int main() {
  constexpr int iterations = 8;
  auto device = xrt::device(0);
  xrt::elf elf{"cascade_flip.elf"};
  xrt::hw_context context(device, elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo input = xrt::ext::bo{device, iterations * 16 * sizeof(int32_t)};
  xrt::bo output = xrt::ext::bo{device, iterations * 64 * sizeof(int32_t)};
  auto *src = input.map<int32_t *>();
  auto *dst = output.map<int32_t *>();
  for (int launch = 0; launch < 3; ++launch) {
    for (int i = 0; i < iterations * 16; ++i)
      src[i] = launch * 10000 + i;
    std::fill(dst, dst + iterations * 64, -1);
    input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kernel(input, output).wait2();
    output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < iterations; ++i) {
      for (int lane = 0; lane < 16; ++lane) {
        if (dst[i * 64 + lane] != src[i * 16 + lane] ||
            dst[i * 64 + 16 + lane] != src[i * 16 + lane] + 1000 ||
            dst[i * 64 + 32 + lane] != 2 * src[i * 16 + lane] + 1000) {
          std::cerr << "cascade mismatch at launch " << launch << ", iteration "
                    << i << ", lane " << lane << "\n";
          return 1;
        }
      }
      if (dst[i * 64 + 48] != 0 || dst[i * 64 + 49] != 3 ||
          dst[i * 64 + 50] != 0) {
        std::cerr << "direction readback mismatch\n";
        return 1;
      }
    }
  }
  std::cout << "PASS: North/West cascade flips, 8 iterations x 3 launches\n";
}
