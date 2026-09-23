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

// Must match macf_bf16.py.
constexpr int VEC_LANES = 32;
constexpr int TILE_SIZE = 1024;
constexpr int K_STEPS = TILE_SIZE / VEC_LANES; // 32 accumulation iterations
constexpr int N_TILES = 3 /*cores*/ * 16 /*tiles per core*/;

// Match the device's single output FIFO: [MUL tile, MAC vector].
constexpr int OUT_MUL_TILE = TILE_SIZE;
constexpr int OUT_MAC_TILE = VEC_LANES;
constexpr int OUT_TILE = OUT_MUL_TILE + OUT_MAC_TILE;

constexpr size_t IN_COUNT = size_t(N_TILES) * TILE_SIZE; // 49152
constexpr size_t OUT_COUNT = size_t(N_TILES) * OUT_TILE; // 50688
constexpr size_t IN_BYTES = IN_COUNT * sizeof(uint16_t); // bf16, 2 bytes/elem
constexpr size_t OUT_BYTES = OUT_COUNT * sizeof(float);

// bf16 occupies the high 16 bits of f32. Truncation is exact for these inputs.

static uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return uint16_t(bits >> 16);
}

static float bf16_to_f32(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Inputs in [-8, 8] are exactly representable in bf16. Products and 32-term
// sums are exact in f32, so comparison with the double reference is exact.
// These inputs test arithmetic and accumulation, not rounding behavior.

static float reference_mul(float a, float b) { return float(double(a) * double(b)); }

static float reference_mac_lane(const float *a, const float *b, int lane) {
  double acc = 0.0;
  for (int k = 0; k < K_STEPS; k++) {
    int idx = k * VEC_LANES + lane;
    acc += double(a[idx]) * double(b[idx]);
  }
  return float(acc);
}

// Seed the first two terms of tile 0 with selected signs and magnitudes.
struct PlantedLane {
  float a0, b0, a1, b1;
  const char *intent;
};

static const PlantedLane kPlanted[VEC_LANES] = {
    {0, 0, 0, 0, "zero accumulator"},
    {8, 8, 0, 0, "largest positive product in range: 64"},
    {-8, 8, 0, 0, "largest-magnitude negative product: -64"},
    {8, 8, -8, 8, "two large positive products: 128"},
    {8, 8, -8, -8, "cancels to zero via two large products"},
    {1, 1, 0, 0, "smallest positive accumulator"},
    {-1, 1, 0, 0, "smallest negative accumulator"},
    {4, 4, 4, 4, "two equal products: 32"},
    {-4, 4, 4, -4, "two equal negative products: -32"},
    {6, 3, -5, 7, "mixed sign terms: 18 - 35 = -17"},
    {2, 2, 0, 0, "4"},
    {-2, 2, 0, 0, "-4"},
    {3, 3, 0, 0, "9"},
    {-3, 3, 0, 0, "-9"},
    {5, 5, 0, 0, "25"},
    {-5, 5, 0, 0, "-25"},
    {6, 6, 0, 0, "36"},
    {-6, 6, 0, 0, "-36"},
    {7, 7, 0, 0, "49"},
    {-7, 7, 0, 0, "-49"},
    {2, 3, 0, 0, "6"},
    {-2, 3, 0, 0, "-6"},
    {3, 4, 0, 0, "12"},
    {-3, 4, 0, 0, "-12"},
    {4, 5, 0, 0, "20"},
    {-4, 5, 0, 0, "-20"},
    {5, 6, 0, 0, "30"},
    {-5, 6, 0, 0, "-30"},
    {6, 7, 0, 0, "42"},
    {-6, 7, 0, 0, "-42"},
    {7, 8, 0, 0, "56"},
    {-7, 8, 0, 0, "-56"},
};

int main() {
  std::vector<uint16_t> vec_a(IN_COUNT), vec_b(IN_COUNT);
  std::vector<float> vec_a_f32(IN_COUNT), vec_b_f32(IN_COUNT);
  std::vector<float> vec_out(OUT_COUNT, 0.0f);

  // Tiles 1..N-1: pseudo-random small integers in [-8, 8].
  std::srand(42);
  for (size_t i = 0; i < IN_COUNT; i++) {
    float a = float(int((std::rand() % 17)) - 8);
    float b = float(int((std::rand() % 17)) - 8);
    vec_a_f32[i] = a;
    vec_b_f32[i] = b;
    vec_a[i] = f32_to_bf16(a);
    vec_b[i] = f32_to_bf16(b);
  }

  // Replace the first two terms of tile 0; later bf16 terms remain random.
  for (int j = 0; j < VEC_LANES; j++) {
    for (int k = 0; k < K_STEPS; k++) {
      vec_a_f32[k * VEC_LANES + j] = 0.0f;
      vec_b_f32[k * VEC_LANES + j] = 0.0f;
    }
    vec_a_f32[0 * VEC_LANES + j] = kPlanted[j].a0;
    vec_b_f32[0 * VEC_LANES + j] = kPlanted[j].b0;
    vec_a_f32[1 * VEC_LANES + j] = kPlanted[j].a1;
    vec_b_f32[1 * VEC_LANES + j] = kPlanted[j].b1;
    vec_a[0 * VEC_LANES + j] = f32_to_bf16(kPlanted[j].a0);
    vec_b[0 * VEC_LANES + j] = f32_to_bf16(kPlanted[j].b0);
    vec_a[1 * VEC_LANES + j] = f32_to_bf16(kPlanted[j].a1);
    vec_b[1 * VEC_LANES + j] = f32_to_bf16(kPlanted[j].b1);
  }

  // Build references from the bf16 values actually sent to the device.
  std::vector<float> vec_a_wide(IN_COUNT), vec_b_wide(IN_COUNT);
  for (size_t i = 0; i < IN_COUNT; i++) {
    vec_a_wide[i] = bf16_to_f32(vec_a[i]);
    vec_b_wide[i] = bf16_to_f32(vec_b[i]);
  }

  std::vector<float> vec_ref(OUT_COUNT);
  for (int t = 0; t < N_TILES; t++) {
    float *out_tile = &vec_ref[size_t(t) * OUT_TILE];
    const float *a_tile = &vec_a_wide[size_t(t) * TILE_SIZE];
    const float *b_tile = &vec_b_wide[size_t(t) * TILE_SIZE];
    for (int i = 0; i < OUT_MUL_TILE; i++)
      out_tile[i] = reference_mul(a_tile[i], b_tile[i]);
    for (int j = 0; j < VEC_LANES; j++)
      out_tile[OUT_MUL_TILE + j] = reference_mac_lane(a_tile, b_tile, j);
  }

  // Coverage self-check on the planted MAC tile (tile 0's MAC segment).
  int n_pos = 0, n_neg = 0, n_zero = 0;
  for (int j = 0; j < VEC_LANES; j++) {
    float v = vec_ref[OUT_MUL_TILE + j];
    if (v > 0)
      n_pos++;
    else if (v < 0)
      n_neg++;
    else
      n_zero++;
  }
  std::cout << "coverage: " << n_pos << " positive, " << n_neg
            << " negative, " << n_zero << " zero planted MAC lanes"
            << std::endl;
  if (n_pos == 0 || n_neg == 0 || n_zero == 0) {
    std::cout << "fail: planted-lane sign coverage lost." << std::endl;
    return 1;
  }

  // Set up XRT.
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"macf_bf16.elf"};
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

  // Exact comparison, both the MUL and MAC segments of every tile.
  int errors = 0;
  for (int t = 0; t < N_TILES; t++) {
    const float *got_tile = &vec_out[size_t(t) * OUT_TILE];
    const float *want_tile = &vec_ref[size_t(t) * OUT_TILE];
    for (int i = 0; i < OUT_MUL_TILE; i++) {
      if (got_tile[i] != want_tile[i]) {
        if (errors < 16)
          std::cout << "MUL MISMATCH tile " << t << " idx " << i
                    << ": got=" << got_tile[i] << " want=" << want_tile[i]
                    << std::endl;
        errors++;
      }
    }
    for (int j = 0; j < VEC_LANES; j++) {
      float got = got_tile[OUT_MUL_TILE + j], want = want_tile[OUT_MUL_TILE + j];
      if (got != want) {
        if (errors < 16) {
          std::cout << "MAC MISMATCH tile " << t << " lane " << j
                    << ": got=" << got << " want=" << want;
          if (t == 0)
            std::cout << "  [planted: " << kPlanted[j].intent << "]";
          std::cout << std::endl;
        }
        errors++;
      }
    }
  }

  if (errors == 0) {
    std::cout << "PASS! 🐬" << std::endl;
    return 0;
  }
  std::cout << errors << " elements mismatched (out of " << OUT_COUNT << ")."
            << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
