//===- test.cpp - XRT host for vadd_fp32 ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs the vadd_fp32 example on npu3 hardware through XRT.
// A and B (256 fp32 each) are streamed to the compute tile over two shim
// MM2S channels; C (256 fp32) is drained back over one S2MM channel.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int VECTOR_SIZE = 256;

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "vadd_fp32.elf";

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

  std::vector<float> input_a(VECTOR_SIZE);
  std::vector<float> input_b(VECTOR_SIZE);
  std::vector<float> golden(VECTOR_SIZE);

  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = dis(gen);
    input_b[i] = dis(gen);
    golden[i] = input_a[i] + input_b[i];
  }

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, VECTOR_SIZE * sizeof(float)};
  xrt::bo bo_b = xrt::ext::bo{device, VECTOR_SIZE * sizeof(float)};
  xrt::bo bo_c = xrt::ext::bo{device, VECTOR_SIZE * sizeof(float)};

  std::memcpy(bo_a.map<float *>(), input_a.data(), VECTOR_SIZE * sizeof(float));
  std::memcpy(bo_b.map<float *>(), input_b.data(), VECTOR_SIZE * sizeof(float));
  float *buf_c = bo_c.map<float *>();
  for (int i = 0; i < VECTOR_SIZE; i++)
    buf_c[i] = NAN;

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t errs = 0;
  int32_t max_ulp = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    uint32_t out_bits, ref_bits;
    std::memcpy(&out_bits, &buf_c[i], 4);
    std::memcpy(&ref_bits, &golden[i], 4);
    int32_t ulp_diff = (int32_t)out_bits - (int32_t)ref_bits;
    if (ulp_diff < 0)
      ulp_diff = -ulp_diff;
    if (ulp_diff > max_ulp)
      max_ulp = ulp_diff;

    if (ulp_diff > 0) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got %.8f, expected %.8f\n", i, buf_c[i], golden[i]);
    }
  }

  std::cout << "Max ULP difference: " << max_ulp << "\n";
  if (errs) {
    std::cout << "FAIL (" << errs << " / " << VECTOR_SIZE << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << VECTOR_SIZE << " outputs match host reference)"
            << std::endl;
  return 0;
}
