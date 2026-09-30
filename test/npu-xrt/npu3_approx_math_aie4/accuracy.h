// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared numerical checks for the NPU3 test and its host-only regression.

#ifndef NPU3_APPROX_MATH_ACCURACY_H
#define NPU3_APPROX_MATH_ACCURACY_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace approx_math_accuracy {

inline uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
inline float bits_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
inline float bf16_to_f32(uint16_t b) { return bits_f32(uint32_t(b) << 16); }
inline uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = f32_bits(f);
  if (std::isnan(f))
    return uint16_t((u >> 16) | 0x40);
  uint32_t lsb = (u >> 16) & 1;
  return uint16_t((u + 0x7fff + lsb) >> 16);
}
inline float trunc_bf16(float f) { return bits_f32(f32_bits(f) & 0xffff0000u); }

// Error metric: quantize to `mbits` mantissa bits with f32 exponents
// and gradual underflow below 2^-126, without an overflow limit. nearbyint
// uses the host rounding mode, which must be round-to-nearest-even.
inline int lead_exp(double v) {
  int e;
  std::frexp(v, &e); // v = m * 2^e, m in [0.5, 1)
  return std::max(e - 1, -126);
}
inline double quantize(double v, int mbits) {
  if (v == 0 || !std::isfinite(v))
    return v;
  double g = std::ldexp(1.0, lead_exp(v) - mbits);
  return std::nearbyint(v / g) * g;
}
inline double ulp_of(double q, int mbits) {
  return std::ldexp(1.0, (q == 0 ? -126 : lead_exp(q)) - mbits);
}
// VEXP2 floors (b - 895) to a 2^-11 grid.
inline double vexp2_arg(float b) {
  return std::floor((double(b) - 895.0) * 2048.0) / 2048.0;
}

struct Stat {
  double max_rel = 0, max_abs = 0, max_ulp = 0;
  double at_rel = 0, at_ulp = 0;
  size_t n = 0, nbad = 0, off_by_one = 0;
  double bad_lo = 0, bad_hi = 0;
  // Mean signed relative error detects bias hidden by maximum-error bounds.
  double sum_signed_rel = 0;
  size_t n_signed_rel = 0;
  void bad(double t) {
    bad_lo = nbad ? std::min(bad_lo, t) : t;
    bad_hi = nbad ? std::max(bad_hi, t) : t;
    nbad++;
  }
  void add(double x, double got, double want, int mbits) {
    n++;
    // Callers supply finite references. Infinite errors ensure NaN results
    // cannot bypass aggregate bounds through unordered comparisons.
    if (!std::isfinite(got)) {
      bad(x);
      max_abs = max_rel = max_ulp = std::numeric_limits<double>::infinity();
      at_rel = at_ulp = x;
      sum_signed_rel = std::numeric_limits<double>::infinity();
      n_signed_rel++;
      return;
    }
    double abs_err = std::fabs(got - want);
    max_abs = std::max(max_abs, abs_err);
    if (want != 0 && abs_err / std::fabs(want) > max_rel) {
      max_rel = abs_err / std::fabs(want);
      at_rel = x;
    }
    if (want != 0) {
      sum_signed_rel += (got - want) / std::fabs(want);
      n_signed_rel++;
    }
    if (mbits > 0 && want != 0) {
      double qw = quantize(want, mbits);
      double u = std::fabs(quantize(got, mbits) - qw) / ulp_of(qw, mbits);
      if (u > max_ulp) {
        max_ulp = u;
        at_ulp = x;
      }
    }
  }
  double mean_signed_rel() const {
    return n_signed_rel ? sum_signed_rel / double(n_signed_rel) : 0.0;
  }
  void print(const char *what, const char *ulp_name) const {
    std::cout << std::setprecision(6) << "  " << what << ": n=" << n
              << " max_abs=" << max_abs << " max_rel=" << max_rel << " (at "
              << at_rel << ")" << " mean_rel=" << mean_signed_rel();
    if (ulp_name)
      std::cout << " max_" << ulp_name << "_ulp=" << max_ulp << " (at "
                << at_ulp << ")";
    if (off_by_one)
      std::cout << " not_RNE=" << off_by_one;
    if (nbad)
      std::cout << " VIOLATIONS=" << nbad << " for t in [" << bad_lo << ", "
                << bad_hi << "]";
    std::cout << std::endl;
  }
};

inline int errors = 0;
inline void fail(const std::string &msg) {
  if (errors < 60)
    std::cout << "FAIL: " << msg << std::endl;
  errors++;
}

inline std::string hex32(uint32_t u) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0') << u;
  return os.str();
}
inline std::string hex16(uint16_t u) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(4) << std::setfill('0') << u;
  return os.str();
}

// Check 2^t with t = floor((b - 895) * 2^11) / 2^11. NaN inputs are
// checked separately by the hardware test. Both forms overflow at t >= 128;
// fp32 flushes to zero below -138, bf16 at or below -134.
inline void check_bf20(const char *what, float b, float got, Stat &st) {
  if (std::isnan(b))
    return;
  uint32_t bits = f32_bits(got);
  if (!std::isnan(got) && (bits & 0x7ffu))
    fail(std::string(what) + ": low 11 bits not clear: " + hex32(bits));
  double t = std::isinf(b) ? double(b) : vexp2_arg(b);
  if (t >= 128) {
    if (got != std::numeric_limits<float>::infinity())
      st.bad(t), fail(std::string(what) + ": t=" + std::to_string(t) +
                      " want +Inf, got " + hex32(bits));
  } else if (t < -138) {
    if (got != 0.0f)
      st.bad(t), fail(std::string(what) + ": t=" + std::to_string(t) +
                      " want 0, got " + hex32(bits));
  } else if (!std::isfinite(got)) {
    st.bad(t);
    fail(std::string(what) + ": t=" + std::to_string(t) +
         " want finite result, got " + hex32(f32_bits(got)));
  } else {
    double want = std::exp2(t);
    double qw = quantize(want, 10);
    double u = std::fabs(quantize(got, 10) - qw) / ulp_of(qw, 10);
    st.add(t, got, want, 10);
    if (u > 1.0)
      st.bad(t),
          fail(std::string(what) + ": t=" + std::to_string(t) + " got " +
               hex32(bits) + " error " + std::to_string(u) + " fp16 ULP");
  }
}
inline void check_bf16(const char *what, float b, uint16_t got_bits, Stat &st) {
  if (std::isnan(b))
    return;
  float got = bf16_to_f32(got_bits);
  double t = std::isinf(b) ? double(b) : vexp2_arg(b);
  if (t >= 128) {
    if (got != std::numeric_limits<float>::infinity())
      st.bad(t), fail(std::string(what) + ": t=" + std::to_string(t) +
                      " want +Inf, got " + hex16(got_bits));
  } else if (t <= -134) {
    if (got != 0.0f)
      st.bad(t), fail(std::string(what) + ": t=" + std::to_string(t) +
                      " want 0, got " + hex16(got_bits));
  } else if (!std::isfinite(got)) {
    st.bad(t);
    fail(std::string(what) + ": t=" + std::to_string(t) +
         " want finite result, got " + hex16(got_bits));
  } else {
    double want = std::exp2(t);
    st.add(t, got, want, 7);
    double qw = quantize(want, 7);
    if (double(got) != qw)
      st.off_by_one++;
    // NPU3 does not always return one of the two values bracketing exact 2^t.
    // Enforce its measured bound of <= 1 ULP from RNE(2^t); see run.lit.
    if (std::fabs(double(got) - qw) > ulp_of(qw, 7))
      st.bad(t), fail(std::string(what) + ": t=" + std::to_string(t) + " got " +
                      hex16(got_bits) + " want " + std::to_string(qw));
  }
}

} // namespace approx_math_accuracy

#endif // NPU3_APPROX_MATH_ACCURACY_H
