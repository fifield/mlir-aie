// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Explicit bf16 conversion under the test fixture's expected floor mode.
// Every output must match floor rounding, including exactly representable
// inputs.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
float bits_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// bf16 stored as the top 16 bits of an f32 bit pattern, widened back to f32
// by shifting left 16 (exact -- bf16's mantissa is a strict prefix of f32's).
float bf16_bits_to_f32(uint16_t bf16_bits) {
  uint32_t u = uint32_t(bf16_bits) << 16;
  return bits_f32(u);
}

// Truncate-toward-zero: drop the low 16 mantissa bits outright. For
// sign-magnitude IEEE floats this rounds the MAGNITUDE down, i.e. toward
// zero, for both signs.
uint16_t trunc_toward_zero(float f) {
  uint32_t u = f32_bits(f);
  return uint16_t(u >> 16);
}

// Floor toward -infinity: truncate-toward-zero, then, if any dropped bits
// were nonzero AND the value is negative, bump the magnitude up by one ULP
// (since increasing the sign-magnitude bit pattern of a negative number
// makes it more negative -- that's floor).
uint16_t floor_toward_neg_inf(float f) {
  uint32_t u = f32_bits(f);
  uint16_t trunc = uint16_t(u >> 16);
  uint32_t dropped = u & 0xffffu;
  bool negative = (u & 0x80000000u) != 0;
  if (negative && dropped != 0)
    trunc = uint16_t(trunc + 1); // magnitude increment -> more negative
  return trunc;
}

// Round-to-nearest-even: standard guard/round/sticky on the low 16 bits of
// the 23-bit f32 mantissa.
uint16_t round_nearest_even(float f) {
  uint32_t u = f32_bits(f);
  uint16_t trunc = uint16_t(u >> 16);
  uint32_t dropped = u & 0xffffu;
  uint32_t halfway = 0x8000u;
  bool round_up;
  if (dropped > halfway) {
    round_up = true;
  } else if (dropped < halfway) {
    round_up = false;
  } else {
    // Exact tie: round to even.
    round_up = (trunc & 1) != 0;
  }
  if (round_up)
    trunc = uint16_t(trunc + 1);
  return trunc;
}

float roundtrip_via(float f, uint16_t (*rounder)(float)) {
  return bf16_bits_to_f32(rounder(f));
}

} // namespace

constexpr size_t TILE_SIZE = 1024;
constexpr size_t N_CORES = 3;
constexpr size_t DATA_COUNT = TILE_SIZE * N_CORES;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(float);

int main() {
  std::vector<float> vec_in(DATA_COUNT), vec_out(DATA_COUNT, NAN);

  // Include ties and both signs to distinguish floor, truncation, and RNE.
  struct Planted {
    uint32_t bits;
    const char *intent;
  };
  std::vector<Planted> planted = {
      {0x3F808001u, "positive, above half ULP"},
      {0xBF804000u, "negative, below half ULP"},
      {0xBF80C000u, "negative, above half ULP"},
      {0x3F818000u, "positive tie, odd kept LSB"},
      {0xBF818000u, "negative tie, odd kept LSB"},
      {0x3F808000u, "positive tie, even kept LSB"},
      {0xBF808000u, "negative tie, even kept LSB"},
      {0x3F800000u, "exact positive"},
      {0xBF800000u, "exact negative"},
      {0x00000000u, "positive zero"},
      {0x80000000u, "negative zero"},
      {0x40490FDBu, "pi"},
      {0xC0490FDBu, "negative pi"},
  };
  for (size_t i = 0; i < planted.size(); i++)
    vec_in[i] = bits_f32(planted[i].bits);

  // Bulk random fill for the remaining elements, spanning a wide dynamic
  // range and both signs.
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  for (size_t i = planted.size(); i < DATA_COUNT; i++)
    vec_in[i] = dist(rng);

  // Set up XRT.
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"bf16_conv_roundtrip.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_in = xrt::ext::bo{device, BUF_SIZE};
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  memcpy(bo_in.map<char *>(), vec_in.data(), BUF_SIZE);
  memcpy(bo_out.map<char *>(), vec_out.data(), BUF_SIZE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_in, bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(vec_out.data(), bo_out.map<char *>(), BUF_SIZE);

  // --- Compare against all three hypotheses -------------------------------
  size_t n_floor = 0, n_trunc = 0, n_rne = 0, n_other = 0;
  for (size_t i = 0; i < DATA_COUNT; i++) {
    float f_floor = roundtrip_via(vec_in[i], floor_toward_neg_inf);
    float f_trunc = roundtrip_via(vec_in[i], trunc_toward_zero);
    float f_rne = roundtrip_via(vec_in[i], round_nearest_even);
    uint32_t got = f32_bits(vec_out[i]);
    bool m_floor = (got == f32_bits(f_floor));
    bool m_trunc = (got == f32_bits(f_trunc));
    bool m_rne = (got == f32_bits(f_rne));
    if (m_floor)
      n_floor++;
    if (m_trunc)
      n_trunc++;
    if (m_rne)
      n_rne++;
    if (!m_floor && !m_trunc && !m_rne)
      n_other++;
    if (i < planted.size()) {
      std::cout << "planted[" << i << "] in=" << vec_in[i]
                << " got=" << vec_out[i] << " floor=" << f_floor
                << " trunc=" << f_trunc << " rne=" << f_rne << " match={"
                << (m_floor ? "floor " : "") << (m_trunc ? "trunc " : "")
                << (m_rne ? "rne " : "") << "}" << "  [" << planted[i].intent
                << "]" << std::endl;
    }
  }

  std::cout << "Totals over " << DATA_COUNT
            << " elements: floor-matches=" << n_floor
            << " trunc-matches=" << n_trunc << " rne-matches=" << n_rne
            << " match-none=" << n_other << std::endl;

  if (n_floor == DATA_COUNT) {
    std::cout << "PASS! 🐬" << std::endl;
    return 0;
  }
  std::cout << "fail: expected floor rounding for every element." << std::endl;
  return 1;
}
