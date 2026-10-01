// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Accuracy and special values of afn 1.0 / x (scalar INV) on NPU3 against a
// double-precision reference, and exactness of strict 1.0 / x (__divsf3).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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

// Must match inv.py.
constexpr size_t TILE = 1024;
constexpr size_t N_TILES = 16;
constexpr size_t NPTS = TILE * N_TILES;

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
std::string hex32(uint32_t u) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0') << u;
  return os.str();
}
// ULP of an f32 value with magnitude |v| (normal range; 2^-149 below).
double f32_ulp(double v) {
  int e;
  std::frexp(std::fabs(v), &e);
  return std::ldexp(1.0, std::max(e - 1, -126) - 23);
}

int errors = 0;
void fail(const std::string &msg) {
  if (errors < 40)
    std::cout << "FAIL: " << msg << std::endl;
  errors++;
}

} // namespace

int main() {
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const float fmax = std::numeric_limits<float>::max();
  const float fmin = std::numeric_limits<float>::min();
  const float denorm = std::numeric_limits<float>::denorm_min();

  std::vector<float> x(NPTS);
  const std::vector<float> specials = {
      qnan,
      -qnan,
      bits_f32(0x7fa00001u),
      inf,
      -inf,
      0.0f,
      -0.0f,
      denorm,
      -denorm,
      bits_f32(0x007fffffu),
      fmin,
      -fmin,
      fmax,
      -fmax,
      bits_f32(0x7e800000u), // 2^126: 1/x = 2^-126, smallest normal
      bits_f32(0x7e800001u), // just above 2^126: 1/x is subnormal
      bits_f32(0x7f000000u), // 2^127
      1.0f,
      -1.0f,
      2.0f,
      0.5f,
      3.0f,
      7.0f,
      10.0f,
      1.5f,
      1.99999988f,
      1.00000012f};
  for (size_t i = 0; i < specials.size(); i++)
    x[i] = specials[i];
  const size_t n_special = specials.size();

  // Sweep evenly over [1, 2), then sample random signs, exponents, and
  // mantissas with both x and 1/x in the normal range.
  const size_t sweep_end = 8192;
  for (size_t i = n_special; i < sweep_end; i++)
    x[i] = bits_f32(
        0x3f800000u |
        uint32_t((i - n_special) * (0x800000ull / (sweep_end - n_special))));
  std::mt19937 rng(20260925);
  std::uniform_int_distribution<uint32_t> exp_dist(1, 252);
  std::uniform_int_distribution<uint32_t> man_dist(0, 0x7fffff);
  std::uniform_int_distribution<uint32_t> sign_dist(0, 1);
  for (size_t i = sweep_end; i < NPTS; i++)
    x[i] = bits_f32((sign_dist(rng) << 31) | (exp_dist(rng) << 23) |
                    man_dist(rng));

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"inv.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  const size_t in_bytes = NPTS * 4, out_bytes = 2 * NPTS * 4;
  xrt::bo bo_in = xrt::ext::bo{device, in_bytes};
  xrt::bo bo_out = xrt::ext::bo{device, out_bytes};
  std::memcpy(bo_in.map<char *>(), x.data(), in_bytes);
  std::memset(bo_out.map<char *>(), 0xff, out_bytes);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_in, bo_out).wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<float> out(2 * NPTS);
  std::memcpy(out.data(), bo_out.map<char *>(), out_bytes);

  std::vector<float> approx(NPTS), exact(NPTS);
  for (size_t t = 0; t < N_TILES; t++)
    for (size_t i = 0; i < TILE; i++) {
      approx[t * TILE + i] = out[t * 2 * TILE + i];
      exact[t * TILE + i] = out[t * 2 * TILE + TILE + i];
    }

  std::cout << "Special x: x | afn 1/x (INV) | strict 1/x (__divsf3)"
            << std::endl;
  for (size_t i = 0; i < n_special; i++)
    std::cout << "  x=" << std::setprecision(9) << x[i] << " ["
              << hex32(f32_bits(x[i])) << "] inv=" << approx[i] << " ["
              << hex32(f32_bits(approx[i])) << "] div=" << exact[i] << " ["
              << hex32(f32_bits(exact[i])) << "]" << std::endl;

  // Strict division must be the correctly rounded IEEE result for every
  // input, including specials (NaN payloads aside).
  size_t strict_mismatch = 0;
  for (size_t i = 0; i < NPTS; i++) {
    float want = 1.0f / x[i];
    bool ok = std::isnan(want) ? std::isnan(exact[i])
                               : f32_bits(exact[i]) == f32_bits(want);
    if (!ok) {
      strict_mismatch++;
      fail("strict 1/" + hex32(f32_bits(x[i])) + " = " +
           hex32(f32_bits(exact[i])) + ", want " + hex32(f32_bits(want)));
    }
  }

  // INV: <= 1 ULP for normal inputs and results. Denormal inputs act as zero,
  // denormal outputs flush to zero, and NaN inputs become quiet NaNs.
  double max_ulp = 0, max_rel = 0, at_ulp = 0;
  size_t n_checked = 0, n_not_rne = 0;
  for (size_t i = 0; i < NPTS; i++) {
    float xi = x[i], got = approx[i];
    uint32_t gb = f32_bits(got);
    if (std::isnan(xi)) {
      uint32_t want = f32_bits(xi) | 0x00400000u;
      if (gb != want)
        fail("INV(NaN " + hex32(f32_bits(xi)) + ") = " + hex32(gb) + ", want " +
             hex32(want));
      continue;
    }
    if (xi == 0 || std::fpclassify(xi) == FP_SUBNORMAL) {
      float want = std::signbit(xi) ? -inf : inf;
      if (got != want)
        fail("INV(" + hex32(f32_bits(xi)) + ") = " + hex32(gb) +
             ", want signed Inf");
      continue;
    }
    if (std::isinf(xi)) {
      if (got != 0 || std::signbit(got) != std::signbit(xi))
        fail("INV(Inf) = " + hex32(gb) + ", want signed zero");
      continue;
    }
    double r = 1.0 / double(xi);
    if (std::fabs(r) < double(fmin)) {
      // Allow either flush to zero or the rounded normal 2^-126 at the edge.
      if (got != 0 && std::fabs(double(got)) != double(fmin))
        fail("INV(" + hex32(f32_bits(xi)) + ") = " + hex32(gb) +
             ", want flushed zero");
      if (got != 0)
        std::cout << "  note: INV(" << hex32(f32_bits(xi))
                  << ") rounded up to the smallest normal" << std::endl;
      continue;
    }
    n_checked++;
    double u = std::fabs(double(got) - r) / f32_ulp(r);
    if (gb != f32_bits(float(r)))
      n_not_rne++;
    if (u > max_ulp) {
      max_ulp = u;
      at_ulp = xi;
    }
    max_rel = std::max(max_rel, std::fabs(double(got) - r) / std::fabs(r));
    if (!(u <= 1.0))
      fail("INV(" + hex32(f32_bits(xi)) + ") = " + hex32(gb) + " error " +
           std::to_string(u) + " ULP");
  }

  std::cout << std::setprecision(6) << "INV over " << n_checked
            << " normal inputs: max_ulp=" << max_ulp << " (at " << at_ulp
            << ") max_rel=" << max_rel << " not_correctly_rounded=" << n_not_rne
            << std::endl;
  std::cout << "strict 1/x mismatches vs IEEE: " << strict_mismatch
            << std::endl;

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " checks failed." << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
