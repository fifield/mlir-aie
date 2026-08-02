//===- test.cpp - XRT host for vmul_int8x8 --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs the vmul_int8x8 example on npu3 hardware through XRT.
// A and B (256 int8 each) are streamed to the compute tile over two shim
// MM2S channels; C (256 int32) is drained back over one S2MM channel.
//
//===----------------------------------------------------------------------===//

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

#include "../cpu_ref/reference_ops.h"

constexpr int VECTOR_SIZE = 256;

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "vmul_int8x8.elf";

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-128, 127);

  std::vector<int8_t> input_a(VECTOR_SIZE);
  std::vector<int8_t> input_b(VECTOR_SIZE);
  std::vector<int32_t> golden(VECTOR_SIZE);

  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = static_cast<int8_t>(dis(gen));
    input_b[i] = static_cast<int8_t>(dis(gen));
  }
  ref_vmul_int8(input_a.data(), input_b.data(), golden.data(), VECTOR_SIZE);

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, VECTOR_SIZE * sizeof(int8_t)};
  xrt::bo bo_b = xrt::ext::bo{device, VECTOR_SIZE * sizeof(int8_t)};
  xrt::bo bo_c = xrt::ext::bo{device, VECTOR_SIZE * sizeof(int32_t)};

  std::memcpy(bo_a.map<int8_t *>(), input_a.data(), VECTOR_SIZE);
  std::memcpy(bo_b.map<int8_t *>(), input_b.data(), VECTOR_SIZE);
  int32_t *buf_c = bo_c.map<int32_t *>();
  for (int i = 0; i < VECTOR_SIZE; i++)
    buf_c[i] = static_cast<int32_t>(0xdeadbeef);

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    if (buf_c[i] != golden[i]) {
      if (errors < 16)
        std::cout << "  mismatch @" << i << " got=" << buf_c[i]
                  << " ref=" << golden[i] << " (a=" << int(input_a[i])
                  << ", b=" << int(input_b[i]) << ")\n";
      errors++;
    }
  }

  if (errors) {
    std::cout << "FAIL (" << errors << " / " << VECTOR_SIZE << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << VECTOR_SIZE << " outputs match host reference)"
            << std::endl;
  return 0;
}
