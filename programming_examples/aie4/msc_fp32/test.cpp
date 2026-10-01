//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// The device computes 1.0 - a*b in vector<32xf32>. Each case picks a and b so
// that the returned value alone says which behaviour the hardware has.
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
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

namespace {

struct Case {
  const char *name;
  float a, b;
  float expected;      // what the native instruction should return
  const char *why;     // what a different answer would mean
};

bool same(float x, float y) {
  if (std::isnan(x) && std::isnan(y)) return true;
  return std::memcmp(&x, &y, sizeof(float)) == 0;   // distinguishes +0 / -0
}

std::string show(float v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  return buf;
}

} // namespace

int main() {
  const float inf = std::numeric_limits<float>::infinity();

  const Case cases[] = {
      // Exact product 1 - 2^-46 rounds to 1 in f32, so one rounding leaves
      // 2^-46 where two leave 0.
      {"one rounding", 1.0f + std::ldexp(1.0f, -23),
       1.0f - std::ldexp(1.0f, -23), std::ldexp(1.0f, -46),
       "0 would mean the product was rounded before the subtract"},

      // a is subnormal. Section 18.4 clamps subnormal inputs to a signed
      // zero, so the product is 0 and the result is exactly 1. An IEEE
      // machine would return 1 - 2^-23.
      {"subnormal input flushed", std::ldexp(1.0f, -149), std::ldexp(1.0f, 126),
       1.0f, "1-2^-23 would mean subnormal inputs are not flushed"},

      // Ordinary overflow of the product is still an infinity.
      {"infinite product", inf, 1.0f, -inf, "a finite result would be wrong"},

      // Section 18.4 calls this out: subnormal times infinity becomes zero
      // times infinity, which is NaN. Here the zero is explicit.
      {"infinity times zero", inf, 0.0f,
       std::numeric_limits<float>::quiet_NaN(),
       "a finite result would mean 0*inf did not produce NaN"},
  };

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"msc_fp32.elf"};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, BUF};
  xrt::bo bo_b = xrt::ext::bo{device, BUF};
  xrt::bo bo_d = xrt::ext::bo{device, BUF};
  float *a = bo_a.map<float *>();
  float *b = bo_b.map<float *>();
  float *d = bo_d.map<float *>();

  int failures = 0;
  for (const Case &c : cases) {
    for (int i = 0; i < N; ++i) { a[i] = c.a; b[i] = c.b; d[i] = -1.0f; }
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kernel(bo_a, bo_b, bo_d).wait2();
    bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int bad = 0;
    for (int i = 0; i < N; ++i)
      if (!same(d[i], c.expected)) ++bad;

    std::cout << (bad ? "  FAIL  " : "  ok    ") << c.name
              << ": got " << show(d[0]) << ", expected " << show(c.expected);
    if (bad) std::cout << "  (" << bad << "/" << N << " differ; " << c.why << ")";
    std::cout << "\n";
    if (bad) ++failures;
  }

  if (failures) {
    std::cout << "FAIL (" << failures << " of " << std::size(cases)
              << " cases)\n";
    return 1;
  }
  std::cout << "PASS (all " << std::size(cases) << " cases)\n";
  return 0;
}
