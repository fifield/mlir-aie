// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Accuracy and special-value behavior of afn exp2/tanh on NPU3 (VEXP2 and
// VTANH), against a double-precision host reference. See run.lit for the
// measured results.

#include "accuracy.h"

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

using namespace approx_math_accuracy;

namespace {

// Must match approx_math.py.
constexpr size_t LANES = 32;
constexpr size_t HALF = LANES / 2;
constexpr size_t IN_WORDS = 3 * LANES;
constexpr size_t OUT_WORDS = 9 * LANES;
constexpr size_t CHUNKS = 8 * 64;
constexpr size_t NPTS = CHUNKS * LANES;

// Point layout shared by x (and xb) and r.
constexpr size_t N_SPECIAL = 64;
constexpr size_t SWEEP1_BEGIN = N_SPECIAL;
constexpr size_t SWEEP1_END = SWEEP1_BEGIN + 8192;
constexpr size_t SWEEP2_END = SWEEP1_END + 4096;
constexpr size_t SWEEP3_END = NPTS;

// Must match approx_math.py's EXP2_BIAS_F32 / EXP2_BIAS_BF16.
constexpr float kBiasF32 = 895.0f + 0x1p-12f; // 895.000244140625f, exact
constexpr float kBiasBf16 = 895.001f;         // aie4_nlf.h, bit for bit

} // namespace

int main() {
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const float fmax = std::numeric_limits<float>::max();
  const float denorm = std::numeric_limits<float>::denorm_min();

  // --- Inputs ---------------------------------------------------------------
  std::vector<float> x(NPTS, 0.0f), r(NPTS, 895.0f);
  std::vector<uint16_t> xb(NPTS);

  // Special values and inputs near exp2 underflow/overflow thresholds.
  const std::vector<float> xs = {qnan,    -qnan,    bits_f32(0x7f800001u),
                                 inf,     -inf,     0.0f,
                                 -0.0f,   denorm,   -denorm,
                                 1e-30f,  -1e-30f,  1.0f,
                                 -1.0f,   0.5f,     10.0f,
                                 -10.0f,  126.0f,   127.0f,
                                 127.5f,  127.99f,  128.0f,
                                 128.5f,  200.0f,   1e10f,
                                 fmax,    -126.0f,  -127.0f,
                                 -130.0f, -133.0f,  -133.9f,
                                 -134.0f, -134.1f,  -135.0f,
                                 -137.0f, -137.9f,  -138.0f,
                                 -138.1f, -139.0f,  -140.0f,
                                 -150.0f, -1000.0f, -1e10f,
                                 -fmax,   3.0f,     -3.0f,
                                 9.0f,    20.0f,    -20.0f,
                                 0.01f,   -0.01f,   1e-5f,
                                 -1e-5f};
  for (size_t i = 0; i < xs.size(); i++)
    x[i] = xs[i];

  // Special raw VEXP2 inputs b; the region edges are at b - 895.
  const std::vector<float> rs = {qnan,
                                 -qnan,
                                 inf,
                                 -inf,
                                 0.0f,
                                 -0.0f,
                                 1.0f,
                                 -1.0f,
                                 512.0f,
                                 895.0f,
                                 1022.0f,
                                 1022.99951171875f,
                                 1023.0f,
                                 1023.5f,
                                 1024.0f,
                                 2048.0f,
                                 1e30f,
                                 fmax,
                                 -fmax,
                                 761.0f,
                                 761.00048828125f,
                                 760.99951171875f,
                                 757.0f,
                                 757.00048828125f,
                                 756.99951171875f,
                                 756.0f,
                                 755.0f,
                                 700.0f,
                                 769.0f,
                                 768.0f};
  for (size_t i = 0; i < rs.size(); i++)
    r[i] = rs[i];

  // Sweep 1: exp2 domain, ascending, across both underflow and overflow edges.
  // Raw inputs sit exactly on VEXP2's 2^-11 input grid.
  for (size_t i = SWEEP1_BEGIN; i < SWEEP1_END; i++) {
    double f = double(i - SWEEP1_BEGIN) / double(SWEEP1_END - SWEEP1_BEGIN - 1);
    x[i] = float(-141.0 + 272.0 * f);
    double t = std::round((-141.0 + 272.0 * f) * 2048.0) / 2048.0;
    r[i] = float(895.0 + t);
  }
  // Sweep 2: ascending tanh inputs and sorted random raw VEXP2 inputs.
  std::mt19937 rng(20260925);
  std::uniform_real_distribution<float> raw_dist(754.0f, 1026.0f);
  for (size_t i = SWEEP1_END; i < SWEEP2_END; i++) {
    double f = double(i - SWEEP1_END) / double(SWEEP2_END - SWEEP1_END - 1);
    x[i] = float(-10.0 + 20.0 * f);
    r[i] = raw_dist(rng);
  }
  std::sort(r.begin() + SWEEP1_END, r.begin() + SWEEP2_END);
  // Sweep 3: near zero, ascending. Raw inputs cover the normal-result range.
  std::uniform_real_distribution<float> raw_dist2(770.0f, 1022.0f);
  for (size_t i = SWEEP2_END; i < SWEEP3_END; i++) {
    double f = double(i - SWEEP2_END) / double(SWEEP3_END - SWEEP2_END - 1);
    x[i] = float(-0.05 + 0.1 * f);
    r[i] = raw_dist2(rng);
  }
  std::sort(r.begin() + SWEEP2_END, r.begin() + SWEEP3_END);
  for (size_t i = 0; i < NPTS; i++)
    xb[i] = f32_to_bf16_rne(x[i]);

  std::vector<float> in(CHUNKS * IN_WORDS);
  for (size_t c = 0; c < CHUNKS; c++) {
    float *p = &in[c * IN_WORDS];
    std::memcpy(p, &x[c * LANES], LANES * 4);
    std::memcpy(p + LANES, &r[c * LANES], LANES * 4);
    std::memcpy(p + 2 * LANES, &xb[c * LANES], LANES * 2);
  }

  // --- Run ------------------------------------------------------------------
  auto device = xrt::device(0);
  xrt::elf ctx_elf{"approx_math.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  const size_t in_bytes = in.size() * 4, out_bytes = CHUNKS * OUT_WORDS * 4;
  xrt::bo bo_in = xrt::ext::bo{device, in_bytes};
  xrt::bo bo_out = xrt::ext::bo{device, out_bytes};
  std::memcpy(bo_in.map<char *>(), in.data(), in_bytes);
  std::memset(bo_out.map<char *>(), 0xff, out_bytes);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_in, bo_out).wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<float> out(CHUNKS * OUT_WORDS);
  std::memcpy(out.data(), bo_out.map<char *>(), out_bytes);

  // --- Unpack -----------------------------------------------------------
  std::vector<float> e32(NPTS), t32(NPTS), raw20(NPTS), bx(NPTS), bxb(NPTS);
  std::vector<uint16_t> e16(NPTS), t16(NPTS), raw16(NPTS);
  std::vector<float> alt0(NPTS), alt1(NPTS);
  for (size_t c = 0; c < CHUNKS; c++) {
    const float *p = &out[c * OUT_WORDS];
    size_t o = c * LANES;
    std::memcpy(&e32[o], p, LANES * 4);
    std::memcpy(&t32[o], p + LANES, LANES * 4);
    std::memcpy(&raw20[o], p + 2 * LANES, LANES * 4);
    std::memcpy(&bx[o], p + 3 * LANES, LANES * 4);
    std::memcpy(&bxb[o], p + 4 * LANES, LANES * 4);
    std::memcpy(&e16[o], p + 5 * LANES, LANES * 2);
    std::memcpy(&t16[o], p + 5 * LANES + HALF, LANES * 2);
    std::memcpy(&raw16[o], p + 5 * LANES + 2 * HALF, LANES * 2);
    std::memcpy(&alt0[o], p + 7 * LANES, LANES * 4);
    std::memcpy(&alt1[o], p + 8 * LANES, LANES * 4);
  }

  // --- Special-value table (measured, printed for the record) ----------------
  std::cout << "Special x: x | exp2.bf20(x) | exp2.bf16(xb) | tanh f32(x) | "
               "tanh bf16(xb)"
            << std::endl;
  for (size_t i = 0; i < xs.size(); i++)
    std::cout << "  x=" << std::setprecision(9) << x[i] << " ["
              << hex32(f32_bits(x[i])) << "] exp2=" << e32[i] << " ["
              << hex32(f32_bits(e32[i]))
              << "] exp2_bf16=" << bf16_to_f32(e16[i]) << " [" << hex16(e16[i])
              << "] tanh=" << t32[i] << " [" << hex32(f32_bits(t32[i]))
              << "] tanh_bf16=" << bf16_to_f32(t16[i]) << " [" << hex16(t16[i])
              << "]" << std::endl;
  std::cout << "Special raw b: b | biased_exp2 bf20 | biased_exp2 bf16"
            << std::endl;
  for (size_t i = 0; i < rs.size(); i++)
    std::cout << "  b=" << std::setprecision(12) << r[i] << " ["
              << hex32(f32_bits(r[i])) << "] bf20=" << raw20[i] << " ["
              << hex32(f32_bits(raw20[i])) << "] bf16=" << bf16_to_f32(raw16[i])
              << " [" << hex16(raw16[i]) << "]" << std::endl;

  // --- Special-value checks -----------------------------------------------
  for (size_t i = 0; i < NPTS; i++) {
    if (std::isnan(r[i])) {
      if (f32_bits(raw20[i]) != 0x7fc00000u)
        fail("VEXP2.fp32(NaN) = " + hex32(f32_bits(raw20[i])));
      if (raw16[i] != 0x7fc0)
        fail("VEXP2.bf16(NaN) = " + hex16(raw16[i]));
    }
    if (std::isnan(x[i])) {
      if (!std::isnan(e32[i]) || !std::isnan(bf16_to_f32(e16[i])))
        fail("exp2 afn(NaN) is not NaN");
      // VTANH returns 0x7F81 or 0xFF81 depending on the input sign.
      uint16_t want = (f32_bits(x[i]) >> 31) ? 0xff81 : 0x7f81;
      if (t16[i] != want)
        fail("tanh bf16(NaN) = " + hex16(t16[i]) + ", want " + hex16(want));
      if (f32_bits(t32[i]) != (uint32_t(want) << 16))
        fail("tanh f32(NaN) = " + hex32(f32_bits(t32[i])));
    }
    if (std::isinf(x[i])) {
      float want_t = x[i] > 0 ? 1.0f : -1.0f;
      if (t32[i] != want_t || bf16_to_f32(t16[i]) != want_t)
        fail("tanh(+/-Inf) is not +/-1");
    }
  }

  // --- The biased VEXP2 input ----------------------------------------------
  // VADD.f (crVecRnd) rounds the bias add; count which host rounding matches.
  // Recorded runs used loader-provided crVecRnd = 12 (RNE); this test does
  // not initialize it.
  size_t b_rne = 0, b_other = 0, bb_rne = 0, bb_other = 0;
  for (size_t i = 0; i < NPTS; i++) {
    if (!std::isfinite(x[i]))
      continue;
    float want = x[i] + kBiasF32; // host: round to nearest even
    if (f32_bits(bx[i]) == f32_bits(want))
      b_rne++;
    else
      b_other++;
    float want_b = bf16_to_f32(xb[i]) + kBiasBf16;
    if (f32_bits(bxb[i]) == f32_bits(want_b))
      bb_rne++;
    else
      bb_other++;
  }
  std::cout << "x + 895.000244140625f (f32 form): " << b_rne << " match RNE, "
            << b_other << " differ" << std::endl;
  std::cout << "xb + 895.001f (bf16 form): " << bb_rne << " match RNE, "
            << bb_other << " differ" << std::endl;

  Stat raw20_grid, raw20_off, raw16_grid, raw16_off, chain20, chain16;
  for (size_t i = 0; i < NPTS; i++) {
    bool grid = i >= SWEEP1_BEGIN && i < SWEEP1_END;
    check_bf20("VEXP2.fp32(b)", r[i], raw20[i], grid ? raw20_grid : raw20_off);
    check_bf16("VEXP2.bf16(b)", r[i], raw16[i], grid ? raw16_grid : raw16_off);
    check_bf20("exp2 afn f32 via VEXP2 input", bx[i], e32[i], chain20);
    check_bf16("exp2 afn bf16 via VEXP2 input", bxb[i], e16[i], chain16);
  }

  // --- Error of the afn results against exact exp2/tanh ------------------
  // Includes VEXP2's input truncation and the result-form-dependent
  // compensation (kBiasF32 / kBiasBf16).
  Stat e20_norm, e20_sub, e16_norm, th32_trunc, th32_exact, th16;
  size_t tanh_bound_viol = 0;
  // Check exact powers of two at integer inputs in [-137, 127).
  size_t n_int = 0, n_int_bad = 0;
  for (size_t i = 0; i < NPTS; i++) {
    if (!std::isfinite(x[i]))
      continue;
    double xd = x[i], xbd = bf16_to_f32(xb[i]);
    if (xd >= -126 && xd < 127)
      e20_norm.add(xd, e32[i], std::exp2(xd), 10);
    else if (xd >= -137 && xd < -126)
      e20_sub.add(xd, e32[i], std::exp2(xd), 10);
    if (xbd >= -126 && xbd < 127)
      e16_norm.add(xbd, bf16_to_f32(e16[i]), std::exp2(xbd), 7);
    if (xd >= -137 && xd < 127 && xd == std::floor(xd)) {
      n_int++;
      float want_exact = std::exp2f(float(xd));
      if (e32[i] != want_exact) {
        n_int_bad++;
        fail("exp2 afn f32 not exact at integer x=" + std::to_string(xd) +
             ": got " + hex32(f32_bits(e32[i])) + " want " +
             hex32(f32_bits(want_exact)));
      }
    }

    // VTANH truncates its input to bf16 first; the error bound is relative to
    // tanh of the truncated input.
    double obj = std::tanh(double(trunc_bf16(x[i])));
    th32_exact.add(xd, t32[i], std::tanh(xd), 0);
    th32_trunc.add(xd, t32[i], obj, 0);
    double obj_b = std::tanh(xbd);
    th16.add(xbd, bf16_to_f32(t16[i]), obj_b, 0);
    auto within = [](double got, double want) {
      return want == 0 ? got == 0
                       : std::fabs(got - want) < 0.083 * std::fabs(want);
    };
    if (!within(t32[i], obj) || !within(bf16_to_f32(t16[i]), obj_b)) {
      tanh_bound_viol++;
      fail("tanh x=" + std::to_string(xd) + " got " + std::to_string(t32[i]) +
           " / " + std::to_string(bf16_to_f32(t16[i])) + " want " +
           std::to_string(obj));
    }
    if (f32_bits(t32[i]) & 0xffffu)
      fail("tanh f32 result is not a widened bf16: " + hex32(f32_bits(t32[i])));
  }
  // Bound the full lowering against exp2(x), including the bias add.
  // The ULP metric quantizes both results to 10 mantissa bits (see accuracy.h).
  if (e20_norm.max_ulp > 1)
    fail("exp2 afn f32 exceeds 1 fp16 ULP vs exp2(x)");
  if (std::fabs(e20_norm.mean_signed_rel()) > 5e-5)
    fail("exp2 afn f32 mean relative error exceeds 5e-5: " +
         std::to_string(e20_norm.mean_signed_rel()));
  if (n_int_bad != 0)
    fail("exp2 afn f32 inexact at " + std::to_string(n_int_bad) + " of " +
         std::to_string(n_int) + " integers in range");
  std::cout << "  exp2 afn f32 integers in range: " << n_int << ", inexact "
            << n_int_bad << std::endl;
  // End-to-end bf16 error is measured against exp2(xb). Faithful rounding
  // against the instruction's quantized input is checked separately above.
  if (e16_norm.max_ulp > 1)
    fail("exp2 afn bf16 exceeds 1 bf16 ULP vs exp2(xb)");

  // Is VEXP2.bf16 the RNE bf16 rounding of VEXP2.fp32 on the same input?
  size_t bf16_is_rne_of_fp32 = 0, bf16_n = 0;
  for (size_t i = 0; i < NPTS; i++) {
    if (std::isnan(r[i]))
      continue;
    bf16_n++;
    if (raw16[i] == f32_to_bf16_rne(raw20[i]))
      bf16_is_rne_of_fp32++;
  }

  // Other compensations, for comparison (not asserted).
  Stat alt0_norm, alt1_norm;
  for (size_t i = 0; i < NPTS; i++) {
    double xd = x[i];
    if (std::isfinite(xd) && xd >= -126 && xd < 127) {
      alt0_norm.add(xd, alt0[i], std::exp2(xd), 10);
      alt1_norm.add(xd, alt1[i], std::exp2(xd), 10);
    }
  }
  // --- Monotonicity over each ascending sweep ------------------------------
  auto mono = [&](const char *what, auto get) {
    size_t viol = 0;
    for (auto [lo, hi] :
         {std::pair{SWEEP1_BEGIN, SWEEP1_END},
          std::pair{SWEEP1_END, SWEEP2_END}, std::pair{SWEEP2_END, SWEEP3_END}})
      for (size_t i = lo + 1; i < hi; i++)
        if (!(get(i) >= get(i - 1))) {
          if (viol++ < 3)
            fail(std::string(what) + " decreases or is unordered at index " +
                 std::to_string(i));
        }
    std::cout << "  monotonic " << what << ": " << viol
              << " decreases or unordered pairs" << std::endl;
  };
  std::cout << "Monotonicity:" << std::endl;
  mono("exp2 afn f32", [&](size_t i) { return double(e32[i]); });
  mono("exp2 afn bf16", [&](size_t i) { return double(bf16_to_f32(e16[i])); });
  mono("tanh afn f32", [&](size_t i) { return double(t32[i]); });
  mono("tanh afn bf16", [&](size_t i) { return double(bf16_to_f32(t16[i])); });
  mono("VEXP2.fp32 raw", [&](size_t i) { return double(raw20[i]); });
  mono("VEXP2.bf16 raw",
       [&](size_t i) { return double(bf16_to_f32(raw16[i])); });

  std::cout << "VEXP2 contract, objective 2^t (t = floor11(b - 895)):"
            << std::endl;
  raw20_grid.print("fp32, b on 2^-11 grid", "fp16");
  raw20_off.print("fp32, b off grid", "fp16");
  chain20.print("fp32, b = x + 895.000244140625f", "fp16");
  raw16_grid.print("bf16, b on 2^-11 grid", "bf16");
  raw16_off.print("bf16, b off grid", "bf16");
  chain16.print("bf16, b = xb + 895.001f", "bf16");
  std::cout << "  VEXP2.bf16 == RNE(VEXP2.fp32): " << bf16_is_rne_of_fp32
            << " of " << bf16_n << std::endl;
  std::cout << "afn results vs exact double reference:" << std::endl;
  e20_norm.print("exp2 afn f32, -126 <= x < 127", "fp16");
  e20_sub.print("exp2 afn f32, -137 <= x < -126", "fp16");
  e16_norm.print("exp2 afn bf16, -126 <= x < 127", "bf16");
  alt0_norm.print("VEXP2.fp32(x + 895.0), -126 <= x < 127", "fp16");
  // The same bias as the f32 lowering should reproduce e20_norm.
  alt1_norm.print("VEXP2.fp32(x + 895 + 2^-12), -126 <= x < 127", "fp16");
  th32_exact.print("tanh afn f32 vs tanh(x)", nullptr);
  th32_trunc.print("tanh afn f32 vs tanh(trunc_bf16(x))", nullptr);
  th16.print("tanh afn bf16 vs tanh(xb)", nullptr);
  std::cout << "  tanh 8.3% bound violations: " << tanh_bound_viol << std::endl;

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " checks failed." << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
