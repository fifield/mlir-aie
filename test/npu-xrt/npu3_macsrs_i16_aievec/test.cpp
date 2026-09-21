// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match mac_srs_i16.py.
constexpr int VEC_LANES = 32;
constexpr int TILE_SIZE = 1024;
constexpr int K_STEPS = TILE_SIZE / VEC_LANES; // 32 accumulation iterations
constexpr int SRS_SHIFT = 6;
constexpr int N_TILES = 3 /*cores*/ * 16 /*tiles per core*/;

constexpr size_t IN_COUNT = size_t(N_TILES) * TILE_SIZE;  // 49152
constexpr size_t OUT_COUNT = size_t(N_TILES) * VEC_LANES; // 1536
constexpr size_t IN_BYTES = IN_COUNT * sizeof(int16_t);
constexpr size_t OUT_BYTES = OUT_COUNT * sizeof(int16_t);

// Expected arithmetic for this test:
// 1. Accumulate widened i16 products in i32 without overflow.
// 2. Shift right with floor rounding.
// 3. Narrow to i16 by wrapping modulo 2^16.
// The explicit SRS op reads mutable crRnd/crSat modes; this test expects
// floor rounding without saturation. Compare integer outputs exactly.

/// Divide by 2^s with floor rounding, including for negative inputs.
static int32_t floor_shr(int32_t v, int s) {
  int64_t n = v;
  int64_t d = int64_t(1) << s;
  int64_t q = n / d;
  if ((n % d) != 0 && ((n < 0) != (d < 0)))
    q -= 1;
  return int32_t(q);
}

static int16_t reference_lane(const int16_t *a, const int16_t *b, int lane) {
  int64_t acc = 0;
  for (int k = 0; k < K_STEPS; k++) {
    int idx = k * VEC_LANES + lane;
    acc += int32_t(a[idx]) * int32_t(b[idx]);
  }
  // Bounded by construction; assert rather than silently wrap.
  if (acc > INT32_MAX || acc < INT32_MIN) {
    std::cerr << "test bug: accumulator overflowed i32 (" << acc << ")\n";
    std::exit(2);
  }
  int32_t shifted = floor_shr(int32_t(acc), SRS_SHIFT);
  // Retain the low 16 bits for wrapping narrowing.
  return int16_t(uint16_t(uint32_t(shifted) & 0xffffu));
}

//===----------------------------------------------------------------------===//
// Planted boundary cases
//===----------------------------------------------------------------------===//
//
// Each lane in tile 0 uses at most two products to exercise i16 wrapping
// boundaries and negative rounding. reference_lane computes expected outputs.
struct PlantedLane {
  int16_t a0, b0, a1, b1;
  const char *intent;
};

static const PlantedLane kPlanted[VEC_LANES] = {
    {0, 0, 0, 0, "zero accumulator"},
    // 64 * 32767 shifts to INT16_MAX without wrapping.
    {64, 32767, 0, 0,
     "exactly INT16_MAX, the largest value that does not wrap"},
    // A shifted value of 32768 wraps to INT16_MIN.
    {128, 16384, 0, 0, "one above INT16_MAX: wraps to INT16_MIN"},
    {-128, 16384, 0, 0, "exactly INT16_MIN, does not wrap"},
    {-128, 16384, -64, 1, "one below INT16_MIN: wraps to INT16_MAX"},
    // Negative values around -64 distinguish floor rounding from truncation.
    {-1, 1, 0, 0, "floor: -1 >> 6 == -1, not 0"},
    {-63, 1, 0, 0, "floor: -63 >> 6 == -1, not 0"},
    {-64, 1, 0, 0, "floor: -64 >> 6 == -1"},
    {-65, 1, 0, 0, "floor: -65 >> 6 == -2, not -1"},
    {63, 1, 0, 0, "positive rounds toward zero: 63 >> 6 == 0"},
    {64, 1, 0, 0, "64 >> 6 == 1"},
    {65, 1, 0, 0, "65 >> 6 == 1"},
    // Far outside the i16 range in both directions.
    {3125, 32000, 0, 0, "far above range, deep wrap"},
    {-3125, 32000, 0, 0, "far below range, deep wrap"},
    {64, 32767, -1, 1, "just below the upper edge"},
    {-128, 16384, -1, 1, "just below INT16_MIN, wraps"},
    {1, 1, 0, 0, "smallest positive accumulator"},
    {-64, 32767, -1, 1, "floor lands exactly on INT16_MIN"},
    {64, 32767, 63, 1, "floor lands exactly on INT16_MAX"},
    {64, 32767, 64, 1, "one past INT16_MAX via two terms, wraps"},
    // Products near the i16-factor limits.
    {32767, 32767, 0, 0, "max i16 product, deep wrap"},
    {-32768, 32767, 0, 0, "min i16 product, deep wrap"},
    {100, 100, 0, 0, "10000 >> 6 == 156"},
    // floor(-10000/64) == -157, whereas truncation would give -156.
    {-100, 100, 0, 0, "floor: -10000 >> 6 == -157, not -156"},
    {2, 1, 0, 0, "2 >> 6 == 0"},
    {-2, 1, 0, 0, "floor: -2 >> 6 == -1"},
    {127, 1, 0, 0, "127 >> 6 == 1"},
    {-127, 1, 0, 0, "floor: -127 >> 6 == -2"},
    {128, 1, 0, 0, "128 >> 6 == 2"},
    {-128, 1, 0, 0, "-128 >> 6 == -2"},
    {12345, 1, 0, 0, "12345 >> 6 == 192"},
    {-12345, 1, 0, 0, "floor: -12345 >> 6 == -193"},
};

int main() {
  std::vector<int16_t> vec_a(IN_COUNT), vec_b(IN_COUNT);
  std::vector<int16_t> vec_out(OUT_COUNT, 0);

  // Random inputs in [-512, 511] keep all 32-term sums within i32.
  std::srand(42);
  for (size_t i = 0; i < IN_COUNT; i++) {
    vec_a[i] = int16_t((std::rand() % 1024) - 512);
    vec_b[i] = int16_t((std::rand() % 1024) - 512);
  }

  // Tile 0: the planted boundary cases.
  for (int j = 0; j < VEC_LANES; j++) {
    for (int k = 0; k < K_STEPS; k++) {
      vec_a[k * VEC_LANES + j] = 0;
      vec_b[k * VEC_LANES + j] = 0;
    }
    vec_a[0 * VEC_LANES + j] = kPlanted[j].a0;
    vec_b[0 * VEC_LANES + j] = kPlanted[j].b0;
    vec_a[1 * VEC_LANES + j] = kPlanted[j].a1;
    vec_b[1 * VEC_LANES + j] = kPlanted[j].b1;
  }

  // Build the reference before touching hardware.
  std::vector<int16_t> vec_ref(OUT_COUNT);
  for (int t = 0; t < N_TILES; t++)
    for (int j = 0; j < VEC_LANES; j++)
      vec_ref[size_t(t) * VEC_LANES + j] = reference_lane(
          &vec_a[size_t(t) * TILE_SIZE], &vec_b[size_t(t) * TILE_SIZE], j);

  // Require coverage of wrapping in both directions and negative rounding.
  int n_wrap_hi = 0, n_wrap_lo = 0, n_neg_round = 0;
  for (int t = 0; t < N_TILES; t++) {
    for (int j = 0; j < VEC_LANES; j++) {
      const int16_t *a = &vec_a[size_t(t) * TILE_SIZE];
      const int16_t *b = &vec_b[size_t(t) * TILE_SIZE];
      int64_t acc = 0;
      for (int k = 0; k < K_STEPS; k++) {
        int idx = k * VEC_LANES + j;
        acc += int32_t(a[idx]) * int32_t(b[idx]);
      }
      int32_t shifted = floor_shr(int32_t(acc), SRS_SHIFT);
      if (shifted > 32767)
        n_wrap_hi++;
      if (shifted < -32768)
        n_wrap_lo++;
      // A negative accumulator that is not an exact multiple of 2^SRS_SHIFT:
      // floor and truncate-toward-zero disagree on exactly these.
      if (acc < 0 && (acc % (int64_t(1) << SRS_SHIFT)) != 0)
        n_neg_round++;
    }
  }
  std::cout << "coverage: " << n_wrap_hi << " wrap-high, " << n_wrap_lo
            << " wrap-low, " << n_neg_round << " negative-rounding lanes"
            << std::endl;
  if (n_wrap_hi == 0 || n_wrap_lo == 0 || n_neg_round == 0) {
    std::cout << "fail: boundary coverage lost; the numerical check would no "
                 "longer prove anything about overflow or rounding."
              << std::endl;
    return 1;
  }

  // Set up XRT.
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"mac_srs_i16.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_a = xrt::ext::bo{device, IN_BYTES};
  xrt::bo bo_b = xrt::ext::bo{device, IN_BYTES};
  xrt::bo bo_c = xrt::ext::bo{device, OUT_BYTES};

  memcpy(bo_a.map<char *>(), vec_a.data(), IN_BYTES);
  memcpy(bo_b.map<char *>(), vec_b.data(), IN_BYTES);
  memset(bo_c.map<char *>(), 0, OUT_BYTES);

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(vec_out.data(), bo_c.map<char *>(), OUT_BYTES);

  // Exact comparison.
  int errors = 0;
  for (size_t i = 0; i < OUT_COUNT; i++) {
    if (vec_out[i] != vec_ref[i]) {
      if (errors < 16) {
        int t = int(i / VEC_LANES), j = int(i % VEC_LANES);
        int64_t acc = 0;
        const int16_t *a = &vec_a[size_t(t) * TILE_SIZE];
        const int16_t *b = &vec_b[size_t(t) * TILE_SIZE];
        for (int k = 0; k < K_STEPS; k++)
          acc += int32_t(a[k * VEC_LANES + j]) * int32_t(b[k * VEC_LANES + j]);
        std::cout << "MISMATCH tile " << t << " lane " << j
                  << ": got=" << vec_out[i] << " want=" << vec_ref[i]
                  << " (acc=" << acc << ", acc>>" << SRS_SHIFT << "="
                  << floor_shr(int32_t(acc), SRS_SHIFT) << ")";
        if (t == 0)
          std::cout << "  [planted: " << kPlanted[j].intent << "]";
        std::cout << std::endl;
      }
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "PASS! 🐬" << std::endl;
    return 0;
  }
  std::cout << errors << " / " << OUT_COUNT << " elements mismatched."
            << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
