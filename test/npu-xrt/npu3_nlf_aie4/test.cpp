// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// afn math.rsqrt (INVSQRT), arcp+afn arith.divf (a * INV(b)) and strict
// arith.divf on NPU3, against a double-precision host reference. See run.lit
// for the checked bounds and the measured results.

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

// Must match nlf.py.
constexpr size_t LANES = 32;
constexpr size_t IN_WORDS = 3 * LANES;
constexpr size_t OUT_WORDS = 3 * LANES;
constexpr size_t CHUNKS = 4 * 64;
constexpr size_t NPTS = CHUNKS * LANES;
constexpr size_t N_SPECIAL = 64;

// Error bounds in f32 ULPs of the exact result, over normal inputs with a
// normal result. INVSQRT is specified to 1 ULP; a * INV(b) adds the rounding
// of the multiply; the strict division must be correctly rounded.
constexpr double kUlpRsqrt = 1.0;
constexpr double kUlpDivInv = 2.0;
constexpr double kUlpDivStrict = 0.5;

const float kInf = std::numeric_limits<float>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kMax = std::numeric_limits<float>::max();
const float kMin = std::numeric_limits<float>::min();
const float kDenorm = std::numeric_limits<float>::denorm_min();

uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

bool is_subnormal(float f) { return f != 0 && std::fabs(f) < kMin; }

// Error of `got` in ULPs of the f32 value nearest `want`.
double ulp_err(float got, double want) {
  int e;
  std::frexp(want, &e);
  double ulp = std::ldexp(1.0, std::max(e - 24, -149));
  return std::fabs(double(got) - want) / ulp;
}

int errors = 0;
void fail(const std::string &msg) {
  if (errors++ < 30)
    std::cout << "FAIL: " << msg << std::endl;
}

std::string show(float f) {
  std::ostringstream os;
  os << std::setprecision(9) << f << " (0x" << std::hex << bits(f) << ")";
  return os.str();
}

struct Stat {
  const char *name;
  double bound;
  size_t n = 0;
  double max_ulp = 0;
  std::string worst;
  void add(const std::string &in, float got, double want) {
    double u = ulp_err(got, want);
    n++;
    if (u > max_ulp) {
      max_ulp = u;
      worst = in;
    }
    if (u > bound)
      fail(std::string(name) + " " + in + ": got " + show(got) + ", want " +
           std::to_string(want) + ", " + std::to_string(u) + " ulp");
  }
  void print() const {
    std::cout << name << ": " << n << " points, max " << std::setprecision(3)
              << max_ulp << " ulp at " << worst << std::endl;
  }
};

} // namespace

int main() {
  std::vector<float> x(NPTS, 1.0f), a(NPTS, 1.0f), b(NPTS, 1.0f);

  // Special and boundary inputs.
  const std::vector<float> xs = {0.0f,  -0.0f, kDenorm,   -kDenorm, kMin,
                                 -1.0f, -kInf, kInf,      kNaN,     1.0f,
                                 4.0f,  0.25f, kMax,      0x1p126f, 0x1p-126f,
                                 2.0f,  3.0f,  0x1p-149f, 0x1p127f};
  const std::vector<std::pair<float, float>> ds = {
      {0x1p127f, 0x1p127f}, {1.0f, 0x1p127f},  {1.0f, 0x1p126f},
      {6.0f, 3.0f},         {1.0f, 0.0f},      {-1.0f, 0.0f},
      {0.0f, 0.0f},         {1.0f, kInf},      {kInf, 1.0f},
      {kNaN, 1.0f},         {1.0f, kDenorm},   {kDenorm, 1.0f},
      {kMax, 0.5f},         {1.0f, -0.0f},     {1.0f, 3.0f},
      {0x1p-126f, 2.0f},    {1.0f, kMax},      {kMin, 0x1p-3f},
      {1.0f, -0x1p127f},    {7.0f, 0x1p-126f}, {kInf, kInf}};
  for (size_t i = 0; i < xs.size(); i++)
    x[i] = xs[i];
  for (size_t i = 0; i < ds.size(); i++) {
    a[i] = ds[i].first;
    b[i] = ds[i].second;
  }

  std::mt19937 rng(20260930);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const size_t half = N_SPECIAL + (NPTS - N_SPECIAL) / 2;
  for (size_t i = N_SPECIAL; i < NPTS; i++) {
    double f = double(i - N_SPECIAL) / double(NPTS - N_SPECIAL - 1);
    // rsqrt: every normal binade, ascending.
    x[i] = float(std::exp2(-126.0 + 253.99 * f));
    if (i < half) {
      // Quotients across a wide range.
      a[i] = float(std::exp2(-60.0 + 120.0 * unit(rng))) * (rng() & 1 ? -1 : 1);
      b[i] = float(std::exp2(-60.0 + 120.0 * unit(rng)));
    } else {
      // Mantissas in one binade, where the rounding shows.
      a[i] = float(1.0 + unit(rng));
      b[i] = float(1.0 + unit(rng));
    }
  }

  std::vector<float> in(CHUNKS * IN_WORDS);
  for (size_t ch = 0; ch < CHUNKS; ch++)
    for (size_t l = 0; l < LANES; l++) {
      size_t p = ch * LANES + l, base = ch * IN_WORDS;
      in[base + l] = x[p];
      in[base + LANES + l] = a[p];
      in[base + 2 * LANES + l] = b[p];
    }

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"nlf.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  size_t bytes = in.size() * sizeof(float);
  xrt::bo bo_in = xrt::ext::bo{device, bytes};
  xrt::bo bo_out = xrt::ext::bo{device, bytes};
  std::memcpy(bo_in.map<void *>(), in.data(), bytes);
  std::memset(bo_out.map<void *>(), 0xff, bytes);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_in, bo_out).wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float *out = bo_out.map<const float *>();
  auto rs = [&](size_t p) { return out[p / LANES * OUT_WORDS + p % LANES]; };
  auto dinv = [&](size_t p) {
    return out[p / LANES * OUT_WORDS + LANES + p % LANES];
  };
  auto dstr = [&](size_t p) {
    return out[p / LANES * OUT_WORDS + 2 * LANES + p % LANES];
  };

  // --- Special values: printed, and checked where IEEE fixes the answer ----
  std::cout << "rsqrt afn specials:" << std::endl;
  for (size_t i = 0; i < xs.size(); i++)
    std::cout << "  x=" << show(x[i]) << " -> " << show(rs(i)) << std::endl;
  std::cout << "divide specials (arcp+afn | strict):" << std::endl;
  for (size_t i = 0; i < ds.size(); i++)
    std::cout << "  " << show(a[i]) << " / " << show(b[i]) << " -> "
              << show(dinv(i)) << " | " << show(dstr(i)) << std::endl;

  auto expect = [&](const char *what, float in, float got, float want) {
    bool ok = std::isnan(want) ? std::isnan(got) : bits(got) == bits(want);
    if (!ok)
      fail(std::string(what) + " of " + show(in) + ": got " + show(got) +
           ", want " + show(want));
  };
  // INVSQRT: a subnormal input reads as a zero of the same sign.
  const float rs_want[] = {kInf, -kInf, kInf, -kInf, 0x1p63f, kNaN,
                           kNaN, 0.0f,  kNaN, 1.0f,  0.5f,    2.0f};
  for (size_t i = 0; i < sizeof rs_want / sizeof rs_want[0]; i++)
    expect("rsqrt afn", x[i], rs(i), rs_want[i]);
  // a * INV(b): a subnormal operand reads as zero, and the reciprocal of a
  // finite |b| > 2^126 flushes to zero. The strict division is IEEE.
  struct DivWant {
    float inv, strict;
  };
  const DivWant dw[] = {{0.0f, 1.0f}, {0.0f, 0x1p-127f}, {0x1p-126f, 0x1p-126f},
                        {2.0f, 2.0f}, {kInf, kInf},      {-kInf, -kInf},
                        {kNaN, kNaN}, {0.0f, 0.0f},      {kInf, kInf},
                        {kNaN, kNaN}, {kInf, kInf},      {0.0f, kDenorm},
                        {kInf, kInf}, {-kInf, -kInf}};
  for (size_t i = 0; i < sizeof dw / sizeof dw[0]; i++) {
    expect("divide arcp+afn", a[i], dinv(i), dw[i].inv);
    expect("divide strict", a[i], dstr(i), dw[i].strict);
  }

  // --- Accuracy ----------------------------------------------------------
  Stat srs{"rsqrt afn", kUlpRsqrt}, sinv{"divide arcp+afn", kUlpDivInv},
      sstr{"divide strict", kUlpDivStrict};
  for (size_t i = 0; i < NPTS; i++) {
    std::ostringstream in_x, in_ab;
    in_x << "x=" << std::setprecision(9) << x[i];
    in_ab << std::setprecision(9) << a[i] << "/" << b[i];
    if (std::isnormal(x[i]) && x[i] > 0)
      srs.add(in_x.str(), rs(i), 1.0 / std::sqrt(double(x[i])));
    if (std::isnormal(a[i]) && std::isnormal(b[i])) {
      double q = double(a[i]) / double(b[i]);
      if (std::fabs(q) >= kMin && std::fabs(q) <= kMax) {
        sstr.add(in_ab.str(), dstr(i), q);
        if (std::fabs(b[i]) <= 0x1p126f)
          sinv.add(in_ab.str(), dinv(i), q);
      }
    }
  }
  srs.print();
  sinv.print();
  sstr.print();

  // Monotonic over the ascending rsqrt sweep.
  for (size_t i = N_SPECIAL + 1; i < NPTS; i++)
    if (rs(i) > rs(i - 1))
      fail("rsqrt afn increases at x=" + show(x[i]));

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " errors" << std::endl;
  return 1;
}
