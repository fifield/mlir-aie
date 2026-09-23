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

// Must match macf32.py.
constexpr int VEC_LANES = 32;
constexpr int TILE_SIZE = 1024;
constexpr int K_STEPS = TILE_SIZE / VEC_LANES; // 32 accumulation iterations
constexpr int N_TILES = 3 /*cores*/ * 16 /*tiles per core*/;

constexpr size_t IN_COUNT = size_t(N_TILES) * TILE_SIZE;  // 49152
constexpr size_t OUT_COUNT = size_t(N_TILES) * VEC_LANES; // 1536
constexpr size_t IN_BYTES = IN_COUNT * sizeof(float);
constexpr size_t OUT_BYTES = OUT_COUNT * sizeof(float);

// Inputs in [-16, 16] keep products and 32-term sums exact in f32.
// The double reference therefore permits exact comparison, but does not
// distinguish fused from separate rounding.

static float reference_lane(const float *a, const float *b, int lane) {
  double acc = 0.0;
  for (int k = 0; k < K_STEPS; k++) {
    int idx = k * VEC_LANES + lane;
    acc += double(a[idx]) * double(b[idx]);
  }
  return float(acc);
}

// Tile 0 uses two selected terms per lane; the remaining terms are zero.
struct PlantedLane {
  float a0, b0, a1, b1;
  const char *intent;
};

static const PlantedLane kPlanted[VEC_LANES] = {
    {0, 0, 0, 0, "zero accumulator"},
    {16, 16, 0, 0, "largest positive product in range: 256"},
    {-16, 16, 0, 0, "largest-magnitude negative product: -256"},
    {16, 16, -16, 16, "two large positive products: 512"},
    {16, 16, -16, -16, "cancels to zero via two large products"},
    {1, 1, 0, 0, "smallest positive accumulator"},
    {-1, 1, 0, 0, "smallest negative accumulator"},
    {8, 8, 8, 8, "two equal products: 128"},
    {-8, 8, 8, -8, "two equal negative products: -128"},
    {12, 3, -5, 7, "mixed sign terms: 36 - 35 = 1"},
    {2, 2, 0, 0, "4"},
    {-2, 2, 0, 0, "-4"},
    {3, 3, 0, 0, "9"},
    {-3, 3, 0, 0, "-9"},
    {4, 4, 0, 0, "16"},
    {-4, 4, 0, 0, "-16"},
    {5, 5, 0, 0, "25"},
    {-5, 5, 0, 0, "-25"},
    {6, 6, 0, 0, "36"},
    {-6, 6, 0, 0, "-36"},
    {7, 7, 0, 0, "49"},
    {-7, 7, 0, 0, "-49"},
    {9, 9, 0, 0, "81"},
    {-9, 9, 0, 0, "-81"},
    {10, 10, 0, 0, "100"},
    {-10, 10, 0, 0, "-100"},
    {11, 11, 0, 0, "121"},
    {-11, 11, 0, 0, "-121"},
    {13, 13, 0, 0, "169"},
    {-13, 13, 0, 0, "-169"},
    {14, 14, 0, 0, "196"},
    {-14, 14, 0, 0, "-196"},
};

int main() {
  std::vector<float> vec_a(IN_COUNT), vec_b(IN_COUNT);
  std::vector<float> vec_out(OUT_COUNT, 0.0f);

  // Initialize random inputs; tile 0 is replaced below.
  std::srand(42);
  for (size_t i = 0; i < IN_COUNT; i++) {
    vec_a[i] = float(int((std::rand() % 33)) - 16);
    vec_b[i] = float(int((std::rand() % 33)) - 16);
  }

  // Tile 0: the planted cases.
  for (int j = 0; j < VEC_LANES; j++) {
    for (int k = 0; k < K_STEPS; k++) {
      vec_a[k * VEC_LANES + j] = 0.0f;
      vec_b[k * VEC_LANES + j] = 0.0f;
    }
    vec_a[0 * VEC_LANES + j] = kPlanted[j].a0;
    vec_b[0 * VEC_LANES + j] = kPlanted[j].b0;
    vec_a[1 * VEC_LANES + j] = kPlanted[j].a1;
    vec_b[1 * VEC_LANES + j] = kPlanted[j].b1;
  }

  // Build the reference before touching hardware.
  std::vector<float> vec_ref(OUT_COUNT);
  for (int t = 0; t < N_TILES; t++)
    for (int j = 0; j < VEC_LANES; j++)
      vec_ref[size_t(t) * VEC_LANES + j] = reference_lane(
          &vec_a[size_t(t) * TILE_SIZE], &vec_b[size_t(t) * TILE_SIZE], j);

  // Check that tile 0 contains positive, negative, and zero results.
  int n_pos = 0, n_neg = 0, n_zero = 0;
  for (int j = 0; j < VEC_LANES; j++) {
    float v = vec_ref[j];
    if (v > 0)
      n_pos++;
    else if (v < 0)
      n_neg++;
    else
      n_zero++;
  }
  std::cout << "coverage: " << n_pos << " positive, " << n_neg
            << " negative, " << n_zero << " zero planted lanes" << std::endl;
  if (n_pos == 0 || n_neg == 0 || n_zero == 0) {
    std::cout << "fail: planted-lane sign coverage lost." << std::endl;
    return 1;
  }

  // Set up XRT.
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"macf32.elf"};
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

  // Compare exactly; all products and sums are representable in f32.
  int errors = 0;
  for (size_t i = 0; i < OUT_COUNT; i++) {
    if (vec_out[i] != vec_ref[i]) {
      if (errors < 16) {
        int t = int(i / VEC_LANES), j = int(i % VEC_LANES);
        std::cout << "MISMATCH tile " << t << " lane " << j
                  << ": got=" << vec_out[i] << " want=" << vec_ref[i];
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
