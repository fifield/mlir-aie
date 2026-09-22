// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Explicit shift-17 UPS under the test fixture's expected nonsaturating mode.
// Every output must equal the widened, shifted value modulo 2^32.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int SHIFT = 17; // must match ups_shift.py's SHIFT
constexpr size_t TILE_SIZE = 1024;
constexpr size_t N_CORES = 3;
constexpr size_t DATA_COUNT = TILE_SIZE * N_CORES; // 3072
constexpr size_t IN_BYTES = DATA_COUNT * sizeof(int16_t);
constexpr size_t OUT_BYTES = DATA_COUNT * sizeof(int32_t);

namespace {
// WRAP hypothesis: sign-extend to i32, then a plain 32-bit left shift,
// silently discarding bits shifted past bit 31 (defined via unsigned
// arithmetic to avoid signed-overflow UB in the reference itself).
int32_t wrap_ups(int16_t v, int shift) {
  int32_t ext = int32_t(v); // sign extension
  uint32_t u = uint32_t(ext) << shift;
  return int32_t(u);
}

// SATURATE hypothesis: compute the mathematically exact value in 64 bits,
// then clamp to the i32 range.
int32_t saturate_ups(int16_t v, int shift) {
  int64_t ext = int64_t(v);
  int64_t full = ext * (int64_t{1} << shift);
  if (full > INT32_MAX)
    return INT32_MAX;
  if (full < INT32_MIN)
    return INT32_MIN;
  return int32_t(full);
}
} // namespace

int main() {
  std::vector<int16_t> vec_in(DATA_COUNT, 0);
  std::vector<int32_t> vec_out(DATA_COUNT, 0);

  // Exercise both overflow boundaries and the full signed i16 range.
  std::vector<int16_t> planted16 = {1,      -1,    -2,     16383, 16384, -16384,
                                    -16385, 32767, -32768, 8192,  -8192, 100,
                                    -100,   0,     2,      -3,    4096,  -4096};
  for (size_t i = 0; i < planted16.size(); i++)
    vec_in[i] = planted16[i];

  // Fill the rest with a mix of powers-of-two-ish and small values so most
  // lanes actually exercise the overflow boundary at shift=17 (anything
  // with |v| > 16384 overflows).
  for (size_t i = planted16.size(); i < DATA_COUNT; i++) {
    int16_t mag = int16_t(((i * 2654435761u) % 65536) - 32768);
    vec_in[i] = mag;
  }

  // Set up XRT.
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"ups_shift.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_in = xrt::ext::bo{device, IN_BYTES};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_BYTES};

  memcpy(bo_in.map<char *>(), vec_in.data(), IN_BYTES);
  memset(bo_out.map<char *>(), 0, OUT_BYTES);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_in, bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(vec_out.data(), bo_out.map<char *>(), OUT_BYTES);

  size_t n_wrap = 0, n_sat = 0, n_neither = 0, n_overflow_cases = 0;
  for (size_t i = 0; i < DATA_COUNT; i++) {
    int32_t got = vec_out[i];
    int32_t want_wrap = wrap_ups(vec_in[i], SHIFT);
    int32_t want_sat = saturate_ups(vec_in[i], SHIFT);
    bool overflowed = (want_wrap != want_sat);
    if (overflowed)
      n_overflow_cases++;
    bool m_wrap = (got == want_wrap);
    bool m_sat = (got == want_sat);
    if (m_wrap)
      n_wrap++;
    if (m_sat)
      n_sat++;
    if (!m_wrap && !m_sat)
      n_neither++;
    if (i < planted16.size()) {
      std::cout << "planted[" << i << "] in=" << vec_in[i] << " got=" << got
                << " wrap=" << want_wrap << " saturate=" << want_sat
                << (overflowed ? "  [overflow case]" : "  [in-range]")
                << " match={" << (m_wrap ? "wrap " : "")
                << (m_sat ? "saturate " : "") << "}" << std::endl;
    }
  }

  std::cout << "Totals over " << DATA_COUNT
            << " elements (overflow cases=" << n_overflow_cases
            << "): wrap-matches=" << n_wrap << " saturate-matches=" << n_sat
            << " match-neither=" << n_neither << std::endl;

  if (n_overflow_cases == 0) {
    std::cout << "fail: no planted/random input actually overflowed i32 at "
                 "shift="
              << SHIFT << "; this run proves nothing about saturate-vs-wrap."
              << std::endl;
    return 1;
  }

  if (n_wrap == DATA_COUNT) {
    std::cout << "PASS! 🐬" << std::endl;
    return 0;
  }
  std::cout << "fail: expected wrapping for every element." << std::endl;
  return 1;
}
