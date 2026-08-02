//===- test.cpp - XRT host for vmul_int32_fp32 ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs the vmul_int32_fp32 example on npu3 hardware through XRT.
// A (256 int32) and B (256 fp32) are streamed to the compute tile over two
// shim MM2S channels; C (256 fp32) is drained back over one S2MM channel.
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
  const std::string elfFile = (argc > 1) ? argv[1] : "vmul_int32_fp32.elf";

  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dis_a(-1000, 1000);
  std::uniform_real_distribution<float> dis_b(-100.0f, 100.0f);

  std::vector<int32_t> input_a(VECTOR_SIZE);
  std::vector<float> input_b(VECTOR_SIZE);
  std::vector<float> golden(VECTOR_SIZE);

  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = dis_a(gen);
    input_b[i] = dis_b(gen);
    golden[i] = (float)input_a[i] * input_b[i];
  }

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, VECTOR_SIZE * sizeof(int32_t)};
  xrt::bo bo_b = xrt::ext::bo{device, VECTOR_SIZE * sizeof(float)};
  xrt::bo bo_c = xrt::ext::bo{device, VECTOR_SIZE * sizeof(float)};

  std::memcpy(bo_a.map<int32_t *>(), input_a.data(),
              VECTOR_SIZE * sizeof(int32_t));
  std::memcpy(bo_b.map<float *>(), input_b.data(), VECTOR_SIZE * sizeof(float));
  float *buf_c = bo_c.map<float *>();
  for (int i = 0; i < VECTOR_SIZE; i++)
    buf_c[i] = NAN;

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  float max_rel_err = 0.0f;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    float diff = std::fabs(buf_c[i] - golden[i]);
    float ref_abs = std::fabs(golden[i]);
    float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
    if (rel_err > max_rel_err)
      max_rel_err = rel_err;

    bool ok = (ref_abs < 1e-6f) ? (diff < 1e-3f) : (rel_err < 1e-5f);
    if (!ok) {
      if (errors < 16)
        std::cout << "  mismatch @" << i << " got=" << buf_c[i]
                  << " ref=" << golden[i] << " (a=" << input_a[i]
                  << ", b=" << input_b[i] << ", rel_err=" << rel_err << ")\n";
      errors++;
    }
  }

  std::cout << "Max relative error: " << max_rel_err << "\n";
  if (errors) {
    std::cout << "FAIL (" << errors << " / " << VECTOR_SIZE << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << VECTOR_SIZE << " outputs match host reference)"
            << std::endl;
  return 0;
}
