//===- test.cpp - XRT host for vadd_bf16 ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs the vadd_bf16 example on npu3 hardware through XRT.
// A and B (256 bf16 each, bit-packed into 128 i32 words) are streamed to the
// compute tile over two shim MM2S channels; C (256 bf16, likewise packed) is
// drained back over one S2MM channel.
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

#define VECTOR_SIZE 256

static uint16_t float_to_bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return (uint16_t)(bits >> 16);
}

static float bf16_to_float(uint16_t bf) {
  uint32_t bits = (uint32_t)bf << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "vadd_bf16.elf";

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

  uint16_t input_a[VECTOR_SIZE], input_b[VECTOR_SIZE], golden[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; i++) {
    input_a[i] = float_to_bf16(dis(gen));
    input_b[i] = float_to_bf16(dis(gen));
    float sum = bf16_to_float(input_a[i]) + bf16_to_float(input_b[i]);
    golden[i] = float_to_bf16(sum);
  }

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  const size_t nbytes = VECTOR_SIZE * sizeof(uint16_t); // == 128 * 4

  xrt::bo bo_a = xrt::ext::bo{device, nbytes};
  xrt::bo bo_b = xrt::ext::bo{device, nbytes};
  xrt::bo bo_c = xrt::ext::bo{device, nbytes};

  // Pack the bf16 arrays into the i32 words the kernel/DMA move around.
  std::memcpy(bo_a.map<uint16_t *>(), input_a, nbytes);
  std::memcpy(bo_b.map<uint16_t *>(), input_b, nbytes);
  uint32_t *buf_c_words = bo_c.map<uint32_t *>();
  for (int i = 0; i < VECTOR_SIZE / 2; i++)
    buf_c_words[i] = 0xdeadbeef;

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint16_t output[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE / 2; i++) {
    uint32_t word = buf_c_words[i];
    output[2 * i] = (uint16_t)(word & 0xFFFF);
    output[2 * i + 1] = (uint16_t)(word >> 16);
  }

  uint32_t errs = 0;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    int32_t diff = (int32_t)output[i] - (int32_t)golden[i];
    if (diff < 0)
      diff = -diff;
    if (diff > 1) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ %d: got 0x%04x (%.6f), expected 0x%04x (%.6f)\n", i,
               output[i], bf16_to_float(output[i]), golden[i],
               bf16_to_float(golden[i]));
    }
  }

  if (!errs) {
    std::cout << "PASS (all " << VECTOR_SIZE
              << " outputs match host reference)" << std::endl;
    return 0;
  }
  std::cout << "FAIL (" << errs << " / " << VECTOR_SIZE << " mismatches)"
            << std::endl;
  return 1;
}
