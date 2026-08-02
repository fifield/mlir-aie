//===- test.cpp - XRT host for matmul_fp8 ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs the MMAC fp8(E4M3)*fp8->fp32 matmul example on npu3 hardware via XRT.
// C[4x64] += A[4x256] * B[256x64].
//
// A (1 KB) and B (16 KB) are streamed to the compute tile over two shim
// MM2S channels; C (1 KB of fp32) is drained back over one S2MM channel.
//
// Data layout (K_outer-blocked, matching kernel access pattern):
//   A[ko * M * K_INNER + m * K_INNER + k]  — row-major within each block
//   B[ko * N * K_INNER + n * K_INNER + k]  — column-major within each block
// K_INNER=64 for fp8 (MMAC 64x64).
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

#include "../cpu_ref/fp_types.h"

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "matmul_fp8.elf";

  constexpr unsigned M = 4;
  constexpr unsigned N = 64;
  constexpr unsigned K_INNER = 64;
  constexpr unsigned K_OUTER = 4;

  constexpr unsigned A_ELEMS = K_OUTER * M * K_INNER;
  constexpr unsigned B_ELEMS = K_OUTER * N * K_INNER;

  std::vector<uint8_t> A(A_ELEMS);
  std::vector<uint8_t> B(B_ELEMS);
  std::vector<float> C_ref(M * N, 0.0f);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-4.0f, 4.0f);

  for (unsigned i = 0; i < A_ELEMS; i++)
    A[i] = float_to_fp8_e4m3(dis(gen));
  for (unsigned i = 0; i < B_ELEMS; i++)
    B[i] = float_to_fp8_e4m3(dis(gen));

  for (unsigned ko = 0; ko < K_OUTER; ko++)
    for (unsigned m = 0; m < M; m++)
      for (unsigned n = 0; n < N; n++)
        for (unsigned k = 0; k < K_INNER; k++)
          C_ref[m * N + n] +=
              fp8_e4m3_to_float(A[ko * M * K_INNER + m * K_INNER + k]) *
              fp8_e4m3_to_float(B[ko * N * K_INNER + n * K_INNER + k]);

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, A_ELEMS * sizeof(uint8_t)};
  xrt::bo bo_b = xrt::ext::bo{device, B_ELEMS * sizeof(uint8_t)};
  xrt::bo bo_c = xrt::ext::bo{device, M * N * sizeof(float)};

  std::memcpy(bo_a.map<uint8_t *>(), A.data(), A_ELEMS * sizeof(uint8_t));
  std::memcpy(bo_b.map<uint8_t *>(), B.data(), B_ELEMS * sizeof(uint8_t));
  float *C_hw = bo_c.map<float *>();
  for (unsigned i = 0; i < M * N; i++)
    C_hw[i] = NAN;

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  unsigned errs = 0;
  float max_rel_err = 0.0f;
  for (unsigned i = 0; i < M * N; i++) {
    float diff = std::fabs(C_hw[i] - C_ref[i]);
    float ref_abs = std::fabs(C_ref[i]);
    float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
    if (rel_err > max_rel_err)
      max_rel_err = rel_err;

    if (rel_err > 1e-3f && diff > 1e-3f) {
      errs++;
      if (errs <= 10)
        std::cout << "  mismatch @ [" << (i / N) << "][" << (i % N)
                  << "]: got " << C_hw[i] << ", expected " << C_ref[i]
                  << " (rel=" << rel_err << ")\n";
    }
  }

  std::cout << "Max relative error: " << max_rel_err << "\n";

  if (errs) {
    std::cout << "FAIL (" << errs << " / " << (M * N) << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << (M * N) << " outputs match host reference)"
            << std::endl;
  return 0;
}
