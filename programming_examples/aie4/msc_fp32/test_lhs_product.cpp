//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// a*b - c, the operand order multiply-subtract does not cover.
//
// With a = 1+2^-23, b = 1-2^-23 and c = 1 the three possible answers are
// distinct, so one run says which happened:
//   0        two roundings, no fusion          <- expected
//   -2^-46   fused as a*b - c, one rounding
//   +2^-46   fused as c - a*b, operands swapped
#include <cmath>
#include <cstring>
#include <iostream>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match what the design moves, in elements: dma_bd offsets and lengths
// count elements of the declared type, not bytes.
constexpr int N = 49152;               // 3 cores x 16 tiles x 1024 f32
constexpr size_t BUF = N * sizeof(float);

int main() {
  auto device = xrt::device(0);
  xrt::elf ctx_elf{"msc_lhs_product.elf"};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, BUF};
  xrt::bo bo_b = xrt::ext::bo{device, BUF};
  xrt::bo bo_d = xrt::ext::bo{device, BUF};
  float *a = bo_a.map<float *>();
  float *b = bo_b.map<float *>();
  float *d = bo_d.map<float *>();

  const float av = 1.0f + std::ldexp(1.0f, -23);
  const float bv = 1.0f - std::ldexp(1.0f, -23);
  for (int i = 0; i < N; ++i) { a[i] = av; b[i] = bv; d[i] = -1.0f; }

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_a, bo_b, bo_d).wait2();
  bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  const float eps = std::ldexp(1.0f, -46);
  int sep = 0, fused = 0, swapped = 0, other = 0;
  for (int i = 0; i < N; ++i) {
    if (d[i] == 0.0f) ++sep;
    else if (d[i] == -eps) ++fused;
    else if (d[i] == eps) ++swapped;
    else ++other;
  }
  std::cout << "a*b - c: separate(0)=" << sep << "  fused(-2^-46)=" << fused
            << "  swapped(+2^-46)=" << swapped << "  other=" << other << "\n";
  if (sep == N) { std::cout << "PASS (not fused, operand order preserved)\n"; return 0; }
  if (swapped) std::cout << "FAIL (fused with operands swapped: computed c - a*b)\n";
  else if (fused) std::cout << "FAIL (fused: a*b - c is not a multiply-subtract)\n";
  else std::cout << "FAIL (unexpected value " << d[0] << ")\n";
  return 1;
}
