// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Check word order and stream selection for one 1024-bit read and two
// 512-bit reads.

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

constexpr size_t IN0 = 48, IN1 = 16, OUT = IN0 + IN1;

int main() {
  std::vector<int32_t> in0(IN0), in1(IN1), out(OUT, -1);
  for (size_t i = 0; i < IN0; i++)
    in0[i] = int32_t(0x1000 + i);
  for (size_t i = 0; i < IN1; i++)
    in1[i] = int32_t(0x2000 + i);

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"stream_read.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_in0 = xrt::ext::bo{device, IN0 * sizeof(int32_t)};
  xrt::bo bo_in1 = xrt::ext::bo{device, IN1 * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, OUT * sizeof(int32_t)};
  memcpy(bo_in0.map<char *>(), in0.data(), IN0 * sizeof(int32_t));
  memcpy(bo_in1.map<char *>(), in1.data(), IN1 * sizeof(int32_t));
  memcpy(bo_out.map<char *>(), out.data(), OUT * sizeof(int32_t));
  bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Bound the host wait in case a stream read stalls.
  auto run = kernel(bo_in0, bo_in1, bo_out);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "fail: run did not complete within 10s (core likely blocked "
                 "on a stream read)"
              << std::endl;
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(out.data(), bo_out.map<char *>(), OUT * sizeof(int32_t));

  int errors = 0;
  for (size_t i = 0; i < OUT; i++) {
    int32_t want = i < IN0 ? in0[i] : in1[i - IN0];
    if (out[i] != want) {
      if (errors < 16)
        std::cout << "out[" << i << "] = 0x" << std::hex << out[i]
                  << ", expected 0x" << want << std::dec << std::endl;
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
