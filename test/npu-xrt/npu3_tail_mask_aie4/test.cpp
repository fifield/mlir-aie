// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Masked tail on NPU3: c = a + b over N = 1000 f32 elements (31 full 32-lane
// chunks and an 8-lane tail) and a masked maximum. Guard words around the
// results must keep the sentinel the core wrote before the kernel.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

// Must match tail_mask.py.
constexpr size_t N = 1000;
constexpr size_t FRONT_GUARD = 32;
constexpr size_t MAX_SLOT = 1087;
constexpr size_t UNAL_REGION = 1088;
constexpr size_t UNAL_SRC = 5;
constexpr size_t UNAL_DST = UNAL_REGION + 3;
constexpr size_t OUT_WORDS = UNAL_REGION + 64;
constexpr uint32_t SENTINEL_BITS = 0xDEADBEEFu;
constexpr size_t N_ITERS = 4;

uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
std::string hex32(uint32_t u) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0') << u;
  return os.str();
}

int errors = 0;
void fail(const std::string &msg) {
  if (errors < 40)
    std::cout << "FAIL: " << msg << std::endl;
  errors++;
}

} // namespace

int main() {
  std::vector<float> a(N * N_ITERS), b(N * N_ITERS);
  std::mt19937 rng(20260925);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = dist(rng);
    b[i] = dist(rng);
  }
  // Iteration 1: all of a negative, so a padding lane (0.0) would win an
  // unmasked maximum. Iteration 2: the maximum sits in the 8-lane tail.
  // Iteration 3: the maximum sits in the last full chunk.
  for (size_t i = 0; i < N; i++)
    a[N + i] = -1.0f - std::fabs(a[N + i]);
  a[2 * N + 997] = 500.0f;
  a[3 * N + 990] = 300.0f;

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"tail_mask.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  const size_t in_bytes = N * N_ITERS * 4, out_bytes = OUT_WORDS * N_ITERS * 4;
  xrt::bo bo_a = xrt::ext::bo{device, in_bytes};
  xrt::bo bo_b = xrt::ext::bo{device, in_bytes};
  xrt::bo bo_c = xrt::ext::bo{device, out_bytes};
  std::memcpy(bo_a.map<char *>(), a.data(), in_bytes);
  std::memcpy(bo_b.map<char *>(), b.data(), in_bytes);
  std::memset(bo_c.map<char *>(), 0, out_bytes);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_a, bo_b, bo_c).wait2();
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<float> c(OUT_WORDS * N_ITERS);
  std::memcpy(c.data(), bo_c.map<char *>(), out_bytes);

  for (size_t it = 0; it < N_ITERS; it++) {
    const float *o = &c[it * OUT_WORDS];
    const float *x = &a[it * N], *y = &b[it * N];
    std::string where = "iteration " + std::to_string(it);
    size_t guard_hits = 0;
    for (size_t k = 0; k < OUT_WORDS; k++) {
      bool data = k >= FRONT_GUARD && k < FRONT_GUARD + N;
      bool copied = k >= UNAL_DST && k < UNAL_DST + 32;
      if (copied) {
        float want = x[UNAL_SRC + k - UNAL_DST];
        if (f32_bits(o[k]) != f32_bits(want))
          fail(where + ": unaligned copy out[" + std::to_string(k) + "] = " +
               hex32(f32_bits(o[k])) + ", want " + hex32(f32_bits(want)));
      } else if (data) {
        size_t i = k - FRONT_GUARD;
        float want = x[i] + y[i];
        if (f32_bits(o[k]) != f32_bits(want))
          fail(where + ": c[" + std::to_string(i) + "] = " +
               hex32(f32_bits(o[k])) + ", want " + hex32(f32_bits(want)));
      } else if (k != MAX_SLOT && f32_bits(o[k]) != SENTINEL_BITS) {
        guard_hits++;
        fail(where + ": guard word " + std::to_string(k) +
             " overwritten: " + hex32(f32_bits(o[k])));
      }
    }
    float want_max = x[0];
    for (size_t i = 1; i < N; i++)
      want_max = std::max(want_max, x[i]);
    if (f32_bits(o[MAX_SLOT]) != f32_bits(want_max))
      fail(where + ": masked max = " + std::to_string(o[MAX_SLOT]) + ", want " +
           std::to_string(want_max));
    std::cout << where << ": max=" << o[MAX_SLOT] << " (want " << want_max
              << "), guard words overwritten: " << guard_hits << std::endl;
  }

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " checks failed." << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
