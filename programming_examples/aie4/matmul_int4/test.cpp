//===- test.cpp - XRT host for matmul_int4 --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs the matmul_int4 example on npu3 hardware through XRT.
// C[4x64] += A[4x512] * B[512x64], int4 * int4 -> int32.
//
// A (1024 packed bytes) and B (16384 packed bytes) are streamed to the
// compute tile over two shim MM2S channels; C (256 int32) is drained back
// over one S2MM channel.
//
// Data layout (K_outer-blocked, matching kernel access pattern):
//   A[ko * M * K_INNER + m * K_INNER + k]  — row-major within each block
//   B[ko * N * K_INNER + n * K_INNER + k]  — column-major within each block
// K_INNER=128 for int4 (MMAC 128x64). Elements packed 2 per byte, low
// nibble first.
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

#include "../cpu_ref/int_types.h"

constexpr unsigned M = 4;
constexpr unsigned N = 64;
constexpr unsigned K_INNER = 128;
constexpr unsigned K_OUTER = 4;
constexpr unsigned TOTAL_A = K_OUTER * M * K_INNER;  // 2048 int4
constexpr unsigned TOTAL_B = K_OUTER * N * K_INNER;  // 32768 int4

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "matmul_int4.elf";

  std::vector<int8_t> A_vals(TOTAL_A);
  std::vector<int8_t> B_vals(TOTAL_B);
  std::vector<int32_t> C_ref(M * N);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-7, 7);

  for (unsigned i = 0; i < TOTAL_A; i++)
    A_vals[i] = (int8_t)dis(gen);
  for (unsigned i = 0; i < TOTAL_B; i++)
    B_vals[i] = (int8_t)dis(gen);

  std::vector<uint8_t> A_packed(TOTAL_A / 2);
  std::vector<uint8_t> B_packed(TOTAL_B / 2);

  memset(A_packed.data(), 0, A_packed.size());
  for (unsigned i = 0; i < TOTAL_A; i++)
    int4_pack(A_packed.data(), i, A_vals[i]);
  memset(B_packed.data(), 0, B_packed.size());
  for (unsigned i = 0; i < TOTAL_B; i++)
    int4_pack(B_packed.data(), i, B_vals[i]);

  memset(C_ref.data(), 0, C_ref.size() * sizeof(int32_t));
  for (unsigned ko = 0; ko < K_OUTER; ko++)
    for (unsigned m = 0; m < M; m++)
      for (unsigned n = 0; n < N; n++)
        for (unsigned k = 0; k < K_INNER; k++)
          C_ref[m * N + n] +=
              (int32_t)A_vals[ko * M * K_INNER + m * K_INNER + k] *
              (int32_t)B_vals[ko * N * K_INNER + n * K_INNER + k];

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, A_packed.size()};
  xrt::bo bo_b = xrt::ext::bo{device, B_packed.size()};
  xrt::bo bo_c = xrt::ext::bo{device, M * N * sizeof(int32_t)};

  std::memcpy(bo_a.map<uint8_t *>(), A_packed.data(), A_packed.size());
  std::memcpy(bo_b.map<uint8_t *>(), B_packed.data(), B_packed.size());
  int32_t *C_hw = bo_c.map<int32_t *>();
  for (unsigned i = 0; i < M * N; i++)
    C_hw[i] = (int32_t)0xdeadbeef;

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  unsigned errs = 0;
  for (unsigned i = 0; i < M * N; i++) {
    if (C_hw[i] != C_ref[i]) {
      if (errs < 10)
        std::cout << "  mismatch @ [" << (i / N) << "][" << (i % N)
                  << "]: got " << C_hw[i] << ", expected " << C_ref[i] << "\n";
      errs++;
    }
  }

  if (errs) {
    std::cout << "FAIL (" << errs << " / " << (M * N) << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << (M * N) << " outputs match host reference)"
            << std::endl;
  return 0;
}
