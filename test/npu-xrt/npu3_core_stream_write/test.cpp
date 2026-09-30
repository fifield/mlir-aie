// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Check all words emitted by mixed scalar and vector stream writes.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr size_t WORDS = 64;

int main() {
  std::vector<int32_t> in(WORDS), out(WORDS, -1);
  for (size_t i = 0; i < WORDS; i++)
    in[i] = int32_t(0x1000 + i);

  // The core writes input ranges [0, 8), [16, 32), [8, 16), then [32, 64).
  std::vector<int32_t> expected;
  expected.insert(expected.end(), in.begin(), in.begin() + 8);
  expected.insert(expected.end(), in.begin() + 16, in.begin() + 32);
  expected.insert(expected.end(), in.begin() + 8, in.begin() + 16);
  expected.insert(expected.end(), in.begin() + 32, in.end());

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"stream_write.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_in = xrt::ext::bo{device, WORDS * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, WORDS * sizeof(int32_t)};
  memcpy(bo_in.map<char *>(), in.data(), WORDS * sizeof(int32_t));
  memcpy(bo_out.map<char *>(), out.data(), WORDS * sizeof(int32_t));
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Bound the host wait in case a stream write or DMA transfer stalls.
  auto run = kernel(bo_in, bo_out);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "fail: run did not complete within 10s" << std::endl;
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(out.data(), bo_out.map<char *>(), WORDS * sizeof(int32_t));

  int errors = 0;
  for (size_t i = 0; i < WORDS; i++) {
    if (out[i] != expected[i]) {
      if (errors < 16)
        std::cout << "out[" << i << "] = 0x" << std::hex << out[i]
                  << ", expected 0x" << expected[i] << std::dec << std::endl;
      errors++;
    }
  }
  if (errors) {
    std::cout << "fail: " << errors << " mismatches" << std::endl;
    return 1;
  }
  std::cout << "PASS!" << std::endl;
  return 0;
}
