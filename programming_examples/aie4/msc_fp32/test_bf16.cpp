//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// 2^127 - a*b with a = b = 2^64, exact in bf16. The product 2^128 overflows
// f32: narrowed first it is inf and the answer -inf, kept wide it is -2^127.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match what the design moves, in elements: dma_bd offsets and lengths
// count elements of the declared type, not bytes. Inputs and output share the
// count and differ only in element size, so the design declares the input
// descriptors bf16 and the output f32.
constexpr int N = 49152;               // 3 cores x 16 tiles x 1024

static uint16_t to_bf16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);      // exact for 2^64
}

int main() {
  auto device = xrt::device(0);
  xrt::elf ctx_elf{"msc_bf16.elf"};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, N * sizeof(uint16_t)};
  xrt::bo bo_b = xrt::ext::bo{device, N * sizeof(uint16_t)};
  xrt::bo bo_d = xrt::ext::bo{device, N * sizeof(float)};
  auto *a = bo_a.map<uint16_t *>();
  auto *b = bo_b.map<uint16_t *>();
  auto *d = bo_d.map<float *>();

  const uint16_t v = to_bf16(std::ldexp(1.0f, 64));
  for (int i = 0; i < N; ++i) { a[i] = v; b[i] = v; d[i] = -1.0f; }
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_a, bo_b, bo_d).wait2();
  bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  const float fused = -std::ldexp(1.0f, 127);
  const float separate = -std::numeric_limits<float>::infinity();
  int nf = 0, ns = 0, other = 0;
  for (int i = 0; i < N; ++i) {
    if (d[i] == fused) ++nf; else if (d[i] == separate) ++ns; else ++other;
  }
  std::cout << "bf16: fused(-2^127)=" << nf << "  separate(-inf)=" << ns
            << "  other=" << other << "  first=" << d[0] << "\n";
  if (nf == N) { std::cout << "PASS (product kept wide)\n"; return 0; }
  if (ns == N) { std::cout << "RESULT: product rounded to f32 before the subtract\n"; return 2; }
  std::cout << "FAIL (mixed or unexpected)\n";
  return 1;
}
