// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// bf16 vector add, sub and fma on NPU3, against a double-precision host
// reference. Every result must be the round-to-nearest-even of the exact
// result. See run.lit.

#include <algorithm>
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

// Must match bf16_arith.py.
constexpr size_t LANES = 32;
constexpr size_t HALF = LANES / 2;
constexpr size_t IN_WORDS = 4 * HALF;
constexpr size_t OUT_WORDS = 4 * HALF;
constexpr size_t CHUNKS = 16 * 64;
constexpr size_t NPTS = CHUNKS * LANES;

double bf16_to_double(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// The float unit reads a subnormal operand as a zero of the same sign.
double daz(uint16_t b) {
  return (b & 0x7f80) == 0 ? ((b & 0x8000) ? -0.0 : 0.0) : bf16_to_double(b);
}

bool is_subnormal(uint16_t b) { return (b & 0x7f80) == 0 && (b & 0x7f); }

bool is_nan(uint16_t b) { return (b & 0x7f80) == 0x7f80 && (b & 0x7f); }

// Every finite non-negative bf16 value in ascending order, with its bits.
std::vector<double> kVals;
std::vector<uint16_t> kBits;

void build_table() {
  for (uint32_t b = 0; b < 0x7f80; b++) {
    kVals.push_back(bf16_to_double(uint16_t(b)));
    kBits.push_back(uint16_t(b));
  }
}

// The two bf16 values that bracket v (equal when v is representable), and
// the one round-to-nearest-even picks. Overflow brackets with infinity.
struct Bracket {
  double lo, hi, rne;
};

Bracket bracket(double v) {
  bool neg = std::signbit(v);
  double m = std::fabs(v);
  const double inf = std::numeric_limits<double>::infinity();
  Bracket r;
  if (m > kVals.back()) {
    double half_ulp = (kVals.back() - kVals[kVals.size() - 2]) / 2;
    r = {kVals.back(), inf, m >= kVals.back() + half_ulp ? inf : kVals.back()};
  } else {
    size_t i = std::lower_bound(kVals.begin(), kVals.end(), m) - kVals.begin();
    if (kVals[i] == m) {
      r = {m, m, m};
    } else {
      double lo = kVals[i - 1], hi = kVals[i];
      double rne = (m - lo < hi - m)    ? lo
                   : (hi - m < m - lo)  ? hi
                   : (kBits[i - 1] & 1) ? hi
                                        : lo;
      r = {lo, hi, rne};
    }
  }
  if (neg)
    r = {-r.hi, -r.lo, -r.rne};
  return r;
}

// x * y + z rounded once. x * y is exact in double, so the only rounding is
// the sum's: TwoSum gives its exact error, and the sum rounded to odd in
// double narrows to bf16 correctly, where the double sum rounded to nearest
// can land on a bf16 midpoint the exact result is not on.
double fma_to_odd(double x, double y, double z) {
  double p = x * y, s = p + z;
  if (!std::isfinite(s))
    return s;
  double bv = s - p, e = (p - (s - bv)) + (z - bv);
  if (e == 0)
    return s;
  uint64_t u;
  std::memcpy(&u, &s, sizeof u);
  if (std::signbit(e) != std::signbit(s))
    u--;
  u |= 1;
  std::memcpy(&s, &u, sizeof s);
  return s;
}

struct Stat {
  const char *name;
  size_t n = 0, rne = 0, flushed = 0, nan_ok = 0;
  std::vector<std::string> bad;

  uint16_t ia = 0, ib = 0, ic = 0;
  // `alt` is a second acceptable exact result, or NaN when there is none.
  void add(double exact, uint16_t got, double alt = NAN) {
    n++;
    if (!std::isnan(alt) && !is_nan(got) &&
        bf16_to_double(got) == bracket(alt).rne) {
      rne++;
      return;
    }
    if (std::isnan(exact)) {
      if (is_nan(got))
        nan_ok++;
      else
        report(exact, got);
      return;
    }
    double g = bf16_to_double(got);
    if (is_nan(got)) {
      report(exact, got);
      return;
    }
    Bracket br = bracket(exact);
    if (g == br.rne) {
      rne++;
    } else if (g == 0 && std::fabs(exact) < 0x1p-126) {
      // A subnormal result flushed to zero.
      flushed++;
    } else {
      report(exact, got);
    }
  }

  void report(double exact, uint16_t got) {
    if (bad.size() < 10) {
      std::ostringstream os;
      os << std::setprecision(10) << "exact " << exact << " got 0x" << std::hex
         << got << " in 0x" << ia << " 0x" << ib << " 0x" << ic;
      bad.push_back(os.str());
    }
  }

  size_t failures() const { return n - rne - flushed - nan_ok; }

  void print() const {
    std::cout << name << ": " << n << " points, " << rne << " RNE, " << flushed
              << " flushed subnormals, " << nan_ok << " NaN, " << failures()
              << " wrong" << std::endl;
    for (auto &s : bad)
      std::cout << "  " << s << std::endl;
  }
};

} // namespace

int main() {
  build_table();
  std::vector<uint16_t> a(NPTS), b(NPTS), c(NPTS);

  // Special values: NaN, infinities, signed zeros, subnormals, the largest
  // finite value, and exact cancellation.
  const uint16_t sp[] = {0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000,
                         0x0001, 0x8001, 0x007f, 0x0080, 0x7f7f, 0xff7f,
                         0x3f80, 0xbf80, 0x4000, 0x3f81, 0x3380, 0x4b80};
  constexpr size_t NSP = sizeof sp / sizeof sp[0];
  size_t i = 0;
  for (size_t x = 0; x < NSP; x++)
    for (size_t y = 0; y < NSP; y++, i++) {
      a[i] = sp[x];
      b[i] = sp[y];
      c[i] = sp[(x + y) % NSP];
    }

  // Double-rounding cases: 1 + 3/256 is exact in f32 and sits between two
  // bf16 values, so floor and nearest-even differ; 1.5 * 1.0078125 - 2^-25
  // rounds in f32 onto a bf16 midpoint the exact result is below.
  a[i] = 0x3f80, b[i] = 0x3c40, c[i] = 0x0000, i++;
  a[i] = 0x3fc0, b[i] = 0x3f81, c[i] = 0xb300, i++;

  std::mt19937 rng(20260930);
  auto finite_bits = [&](std::uniform_int_distribution<int> &exp) {
    uint16_t sign = (rng() & 1) << 15;
    return uint16_t(sign | (exp(rng) << 7) | (rng() & 0x7f));
  };
  std::uniform_int_distribution<int> any_exp(0, 254), near_exp(120, 134);
  const size_t third = (NPTS - i) / 3;
  // Any finite bf16, so most pairs differ widely in exponent.
  for (size_t end = i + third; i < end; i++) {
    a[i] = finite_bits(any_exp);
    b[i] = finite_bits(any_exp);
    c[i] = finite_bits(any_exp);
  }
  // Nearby exponents, where the rounding of the result matters most.
  for (size_t end = i + third; i < end; i++) {
    a[i] = finite_bits(near_exp);
    b[i] = finite_bits(near_exp);
    c[i] = finite_bits(near_exp);
  }
  // Near cancellation: b is within a few ulps of a.
  for (; i < NPTS; i++) {
    a[i] = finite_bits(near_exp);
    b[i] = uint16_t(a[i] + int(rng() % 7) - 3);
    c[i] = uint16_t((a[i] ^ 0x8000) + int(rng() % 7) - 3);
  }

  // Pack into the chunk layout: a | b | c | pad, 16 words each.
  std::vector<uint16_t> in(NPTS / LANES * IN_WORDS * 2, 0);
  for (size_t ch = 0; ch < CHUNKS; ch++)
    for (size_t l = 0; l < LANES; l++) {
      size_t p = ch * LANES + l, base = ch * IN_WORDS * 2;
      in[base + l] = a[p];
      in[base + 2 * HALF + l] = b[p];
      in[base + 4 * HALF + l] = c[p];
    }

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"bf16_arith.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  size_t in_bytes = in.size() * sizeof(uint16_t);
  size_t out_bytes = CHUNKS * OUT_WORDS * sizeof(float);
  xrt::bo bo_in = xrt::ext::bo{device, in_bytes};
  xrt::bo bo_out = xrt::ext::bo{device, out_bytes};
  std::memcpy(bo_in.map<void *>(), in.data(), in_bytes);
  std::memset(bo_out.map<void *>(), 0xff, out_bytes);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_in, bo_out).wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint16_t *out = bo_out.map<const uint16_t *>();

  Stat add{"add"}, sub{"sub"}, fma{"fma"};
  size_t n_daz = 0;
  for (size_t ch = 0; ch < CHUNKS; ch++)
    for (size_t l = 0; l < LANES; l++) {
      size_t p = ch * LANES + l, base = ch * OUT_WORDS * 2;
      double x = daz(a[p]), y = daz(b[p]), z = daz(c[p]);
      n_daz += is_subnormal(a[p]) || is_subnormal(b[p]) || is_subnormal(c[p]);
      add.ia = sub.ia = fma.ia = a[p];
      add.ib = sub.ib = fma.ib = b[p];
      add.ic = sub.ic = fma.ic = c[p];
      add.add(x + y, out[base + l]);
      sub.add(x - y, out[base + 2 * HALF + l]);
      // A product in the f32 subnormal range flushes to zero.
      double prod = x * y;
      double alt = prod != 0 && std::fabs(prod) < 0x1p-126 ? z : double(NAN);
      fma.add(fma_to_odd(x, y, z), out[base + 4 * HALF + l], alt);
    }

  std::cout << n_daz << " of " << NPTS
            << " points have a subnormal operand, read as zero" << std::endl;
  // crRnd as the core saw it, after poison.o's main() set it to floor.
  const float *outf = bo_out.map<const float *>();
  bool rnd_ok = true;
  for (size_t ch = 0; ch < CHUNKS && rnd_ok; ch++)
    if (outf[ch * OUT_WORDS + 3 * HALF] != 12.0f) {
      std::cout << "crRnd in the core is " << outf[ch * OUT_WORDS + 3 * HALF]
                << ", not 12" << std::endl;
      rnd_ok = false;
    }
  add.print();
  sub.print();
  fma.print();
  if (rnd_ok && add.failures() + sub.failures() + fma.failures() == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << "FAIL." << std::endl;
  return 1;
}
