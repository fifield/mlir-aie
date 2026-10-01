// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Accuracy and special values of afn math.exp on NPU3 (VEXP2 on x * log2(e)),
// against a double-precision host reference. See run.lit for the measured
// results.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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

// Must match approx_exp.py.
constexpr size_t LANES = 32;
constexpr size_t HALF = LANES / 2;
constexpr size_t IN_WORDS = 2 * LANES;
constexpr size_t OUT_WORDS = 2 * LANES;
constexpr size_t CHUNKS = 8 * 64;
constexpr size_t NPTS = CHUNKS * LANES;

// Point layout.
constexpr size_t N_SPECIAL = 64;
constexpr size_t SWEEP1_END = N_SPECIAL + 12288;
constexpr size_t SWEEP2_END = NPTS;

// Relative error bounds in the normal-result range, from the precision VEXP2
// returns: 10 mantissa bits in its f32 form, 7 in its bf16 form.
constexpr double kRelF32 = 0x1p-10;
constexpr double kRelBf16 = 0x1p-7;

uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}
float bits_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}
float bf16_to_f32(uint16_t b) { return bits_f32(uint32_t(b) << 16); }
uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = f32_bits(f);
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return uint16_t((u >> 16) | 0x40);
  return uint16_t((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

int errors = 0;
void fail(const std::string &msg) {
  if (errors++ < 20)
    std::cout << "FAIL: " << msg << std::endl;
}

struct Stat {
  const char *name;
  double bound;
  size_t n = 0;
  double max_rel = 0, sum_rel = 0, worst_x = 0;
  void add(double x, double got, double want) {
    double rel = (got - want) / want;
    n++;
    sum_rel += rel;
    if (std::fabs(rel) > std::fabs(max_rel)) {
      max_rel = rel;
      worst_x = x;
    }
    if (std::fabs(rel) > bound) {
      std::ostringstream os;
      os << std::setprecision(9) << name << ": x=" << x << " got " << got
         << " want " << want << " rel " << rel;
      fail(os.str());
    }
  }
  void print() const {
    std::cout << std::setprecision(4) << name << ": " << n
              << " points, max |rel| " << std::fabs(max_rel)
              << " at x=" << worst_x << ", mean rel " << (n ? sum_rel / n : 0)
              << std::endl;
  }
};

} // namespace

int main() {
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const float fmax = std::numeric_limits<float>::max();
  const float fmin = std::numeric_limits<float>::min();
  const float denorm = std::numeric_limits<float>::denorm_min();

  // --- Inputs ---------------------------------------------------------------
  std::vector<float> x(NPTS, 0.0f);
  // Special values and the edges of the result range: ln(FLT_MAX) ~ 88.72,
  // ln(FLT_MIN) ~ -87.34, ln(denorm_min) ~ -103.28.
  const std::vector<float> xs = {
      qnan,    -qnan,   inf,     -inf,    0.0f,     -0.0f,  denorm,
      -denorm, 1.0f,    -1.0f,   0.5f,    1e-5f,    -1e-5f, 10.0f,
      -10.0f,  80.0f,   88.0f,   88.7f,   88.72f,   88.73f, 88.75f,
      89.0f,   100.0f,  1000.0f, fmax,    -80.0f,   -87.0f, -87.33f,
      -87.34f, -87.5f,  -90.0f,  -95.0f,  -95.6f,   -95.7f, -100.0f,
      -103.0f, -103.3f, -104.0f, -110.0f, -1000.0f, -fmax};
  for (size_t i = 0; i < xs.size(); i++)
    x[i] = xs[i];
  // Sweep 1: the whole exp domain, ascending, past both edges.
  for (size_t i = N_SPECIAL; i < SWEEP1_END; i++) {
    double f = double(i - N_SPECIAL) / double(SWEEP1_END - N_SPECIAL - 1);
    x[i] = float(-106.0 + 196.0 * f);
  }
  // Sweep 2: near zero, ascending.
  for (size_t i = SWEEP1_END; i < SWEEP2_END; i++) {
    double f = double(i - SWEEP1_END) / double(SWEEP2_END - SWEEP1_END - 1);
    x[i] = float(-2.0 + 4.0 * f);
  }
  std::vector<uint16_t> xb(NPTS);
  for (size_t i = 0; i < NPTS; i++)
    xb[i] = f32_to_bf16_rne(x[i]);

  std::vector<float> in(CHUNKS * IN_WORDS, 0.0f);
  for (size_t ch = 0; ch < CHUNKS; ch++) {
    std::memcpy(&in[ch * IN_WORDS], &x[ch * LANES], LANES * sizeof(float));
    std::memcpy(&in[ch * IN_WORDS + LANES], &xb[ch * LANES],
                LANES * sizeof(uint16_t));
  }

  // --- Run ------------------------------------------------------------------
  auto device = xrt::device(0);
  xrt::elf ctx_elf{"approx_exp.elf"};
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

  std::vector<float> e32(NPTS);
  std::vector<uint16_t> e16(NPTS);
  for (size_t ch = 0; ch < CHUNKS; ch++) {
    std::memcpy(&e32[ch * LANES], &out[ch * OUT_WORDS], LANES * sizeof(float));
    std::memcpy(&e16[ch * LANES], &out[ch * OUT_WORDS + LANES],
                LANES * sizeof(uint16_t));
  }

  // --- Check ----------------------------------------------------------------
  Stat s32{"exp afn f32", kRelF32}, s16{"exp afn bf16", kRelBf16};
  size_t under32 = 0, under16 = 0, over32 = 0, over16 = 0;
  const float bf16_max = bf16_to_f32(0x7f7f);
  auto check = [&](const char *name, double xi, float got, double bound,
                   double top, Stat &st, size_t &under, size_t &over) {
    double want = std::exp(xi);
    std::ostringstream os;
    os << std::setprecision(9) << name << ": x=" << xi << " got " << got;
    if (std::isnan(xi)) {
      if (!std::isnan(got))
        fail(os.str() + ", want NaN");
    } else if (want > top * (1 + bound)) {
      // Past the largest finite result.
      over++;
      if (got != inf)
        fail(os.str() + ", want +Inf");
    } else if (want >= top * (1 - bound)) {
      // At the overflow edge either side is right.
      if (!(got == inf || std::fabs(got - want) <= bound * want))
        fail(os.str() + ", want ~" + std::to_string(want) + " or +Inf");
    } else if (want < fmin) {
      // Underflow: flushing to zero is allowed, going negative or large is not.
      under++;
      if (!(got >= 0 && got <= fmin))
        fail(os.str() + ", want [0, FLT_MIN]");
    } else {
      st.add(xi, got, want);
    }
  };
  for (size_t i = 0; i < NPTS; i++) {
    check("exp afn f32", x[i], e32[i], kRelF32, fmax, s32, under32, over32);
    check("exp afn bf16", bf16_to_f32(xb[i]), bf16_to_f32(e16[i]), kRelBf16,
          bf16_max, s16, under16, over16);
  }
  // exp(+-0) is exactly 1.
  for (size_t i : {4, 5}) {
    if (e32[i] != 1.0f)
      fail("exp afn f32 of zero is not 1");
    if (e16[i] != 0x3f80)
      fail("exp afn bf16 of zero is not 1");
  }
  // Non-decreasing over each ascending sweep.
  auto mono = [&](const char *name, auto val) {
    for (size_t i = N_SPECIAL + 1; i < NPTS; i++)
      if (i != SWEEP1_END && val(i) < val(i - 1))
        fail(std::string(name) + " decreases at x=" + std::to_string(x[i]));
  };
  mono("exp afn f32", [&](size_t i) { return e32[i]; });
  mono("exp afn bf16", [&](size_t i) { return bf16_to_f32(e16[i]); });

  s32.print();
  s16.print();
  std::cout << "f32: " << over32 << " overflow, " << under32
            << " underflow points; bf16: " << over16 << " overflow, " << under16
            << " underflow points" << std::endl;
  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " errors" << std::endl;
  return 1;
}
