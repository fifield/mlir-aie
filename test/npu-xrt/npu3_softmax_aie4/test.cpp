// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Row softmax on NPU3 (VMAXR.f, VEXP2.ADDR.f, INV, VMUL.f) against a
// double-precision host softmax.

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

// Must match softmax.py.
constexpr size_t ROWS = 5;
constexpr size_t ROW_LENGTHS[ROWS] = {32, 64, 96, 160, 256};
constexpr size_t SET_ELEMS = 32 + 64 + 96 + 160 + 256;
constexpr size_t STATS = 4;
constexpr size_t OUT_ELEMS = SET_ELEMS + STATS * ROWS;
constexpr size_t N_SETS = 64;

// End to end: |y - ref| <= REL_TOL * ref + ABS_TOL.
// exp2 afn with an f32 result: <= 1 fp16 ULP of exact exp2
// over the normal range, i.e. <= 2^-10 relative (bias 895 + 2^-12).
// Normalization divides by a sum of the same approximations, so y's error is
// at most twice that, plus f32 rounding of the argument, the sums, INV
// (<= 1 ULP) and the product: 2 * 2^-10 + O(2^-16) < 2e-3.
constexpr double REL_TOL = 2e-3;
// Exponentials below 2^-126 are f32-subnormal or flushed; their absolute
// error stays below 2^-126, and y <= e.
const double ABS_TOL = std::ldexp(1.0, -125);

// For normal y, recover the stored exponential e' and check its relative
// error against exp(x - m). The 2^-10 bound for exp2 needs an extra 2^-16:
// t = RN(RN(x - m) * RN(log2 e)) is off by at most 2.5 * 2^-24 * |t|,
// or 1.9e-5 for |t| <= 126, moving 2^t by at most 1.3e-5 relative.
const double EXP_TOL = std::ldexp(1.0, -10) + std::ldexp(1.0, -16);
// Bound the mean signed relative error to detect biased exp2 rounding.
// Truncating uniformly distributed t to a 2^-11 grid has mean relative
// error -ln2 * 2^-12 = -1.7e-4;
// use half that magnitude (8.5e-5) as the limit.
const double MEAN_TOL = std::log(2.0) * std::ldexp(1.0, -13);
// For normal y, |y - e' / sum| <= NORM_TOL * y: INV contributes 2^-23
// relative error and the product rounding adds 2^-24. The (1 + 2^-22)
// factor accounts for expressing the bound relative to y instead of e'/sum.
const double NORM_TOL =
    std::ldexp(1.0, -23) * (1 + std::ldexp(1.0, -22)) + std::ldexp(1.0, -24);

enum Pattern {
  AllEqual,
  Dominant,
  LargePositive,
  LargeNegative,
  Extremes,
  Masked,
  AllNegInf,
  OneFinite,
  NegInfChunk,
  MaxInLastChunk,
  PosInf,
  VaddrTie,
  NumSpecial
};
const char *pattern_name(size_t s) {
  static const char *names[] = {"all equal",
                                "one dominant",
                                "large positive (1e4)",
                                "large negative (-1e4)",
                                "extremes (+/-FLT_MAX)",
                                "masked (-inf)",
                                "all -inf",
                                "one finite, rest -inf",
                                "one chunk all -inf",
                                "max only in last chunk",
                                "contains +inf",
                                "VADDR.f rounding ties"};
  return s < NumSpecial ? names[s] : "random";
}

uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
std::string hex32(uint32_t u) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0') << u;
  return os.str();
}

// Round v to `bits` significant bits, ties to even.
double round_sig(double v, int bits) {
  if (v == 0 || !std::isfinite(v))
    return v;
  int e;
  double m = std::frexp(v, &e);
  return std::ldexp(std::nearbyint(std::ldexp(m, bits)), e - bits);
}

// Right shift of a non-negative integer with round to nearest, ties to even.
uint64_t shift_right_rne(uint64_t m, int sh) {
  if (sh <= 0)
    return m;
  if (sh >= 64)
    return 0;
  uint64_t q = m >> sh, rem = m & ((uint64_t(1) << sh) - 1),
           half = uint64_t(1) << (sh - 1);
  if (rem > half || (rem == half && (q & 1)))
    q++;
  return q;
}

// Model VADDR.f of n lanes plus the scalar seed: each
// 24-bit significand (implicit 1 included) is shifted left by 7, aligned to
// the largest exponent with RNE, and the sum is rounded once (RNE) to f32.
// Subnormal results become zero. Inputs here are non-negative, so the
// significands are rounded as magnitudes.
float vaddr_f(const float *v, size_t n, float seed) {
  std::vector<float> vals(v, v + n);
  vals.push_back(seed);
  bool nan = false, pinf = false, ninf = false;
  for (float f : vals) {
    nan |= std::isnan(f);
    pinf |= f == std::numeric_limits<float>::infinity();
    ninf |= f == -std::numeric_limits<float>::infinity();
  }
  if (nan || (pinf && ninf))
    return std::numeric_limits<float>::quiet_NaN();
  if (pinf || ninf)
    return pinf ? std::numeric_limits<float>::infinity()
                : -std::numeric_limits<float>::infinity();
  auto exponent = [](uint32_t u) {
    int be = (u >> 23) & 0xff;
    return be ? be : 1;
  };
  int max_exp = 1;
  for (float f : vals)
    max_exp = std::max(max_exp, exponent(f32_bits(f)));
  int64_t sum = 0;
  for (float f : vals) {
    uint32_t u = f32_bits(f);
    uint64_t mant = (u & 0x7fffff) | ((u >> 23) & 0xff ? 0x800000 : 0);
    mant = shift_right_rne(mant << 7, max_exp - exponent(u));
    sum += (u >> 31) ? -int64_t(mant) : int64_t(mant);
  }
  // Value = sum * 2^(max_exp - 127 - 23 - 7). Keep 24 significant bits.
  uint64_t mag = sum < 0 ? -sum : sum;
  int scale = max_exp - 157;
  int len = 0;
  while (len < 64 && (mag >> len))
    len++;
  if (len > 24) {
    mag = shift_right_rne(mag, len - 24);
    scale += len - 24;
  }
  double r = std::ldexp(double(mag), scale);
  if (r < std::ldexp(1.0, -126))
    r = 0;
  float out = float(r); // exact, or overflow to +inf
  return sum < 0 ? -out : out;
}

int errors = 0;
void fail(const std::string &msg) {
  if (errors < 40)
    std::cout << "FAIL: " << msg << std::endl;
  errors++;
}

} // namespace

int main() {
  const float pinf = std::numeric_limits<float>::infinity();
  const float ninf = -pinf;
  const float fmax = std::numeric_limits<float>::max();

  std::vector<float> in(SET_ELEMS * N_SETS);
  std::mt19937 rng(20260925);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::uniform_real_distribution<float> unif(0.0f, 1.0f);
  const float scales[] = {0.1f, 1.0f, 3.0f, 10.0f, 30.0f};

  for (size_t s = 0; s < N_SETS; s++) {
    size_t base = s * SET_ELEMS;
    for (size_t r = 0; r < ROWS; r++) {
      size_t n = ROW_LENGTHS[r];
      float *x = &in[base];
      for (size_t i = 0; i < n; i++) {
        switch (s) {
        case AllEqual:
          x[i] = 5.0f;
          break;
        case Dominant:
          x[i] = i == (n * 7) / 13 ? 100.0f : 0.0f;
          break;
        case LargePositive:
          // exp2 of these values overflows without the max subtraction.
          x[i] = 1e4f + 5.0f * gauss(rng);
          break;
        case LargeNegative:
          x[i] = -1e4f + 5.0f * gauss(rng);
          break;
        case Extremes:
          // x - max overflows to -inf for the -FLT_MAX entries.
          x[i] = i % 3 == 0 ? fmax : (i % 3 == 1 ? -fmax : 0.0f);
          break;
        case Masked:
          x[i] = (i % 4 == 1 || unif(rng) < 0.2f) ? ninf : 3.0f * gauss(rng);
          break;
        case AllNegInf:
          x[i] = ninf;
          break;
        case OneFinite:
          x[i] = i == n / 2 + 1 ? 3.0f : ninf;
          break;
        case NegInfChunk:
          x[i] = (n > 32 && i >= 32 && i < 64) || (n == 32 && i < 16)
                     ? ninf
                     : 2.0f * gauss(rng);
          break;
        case MaxInLastChunk:
          // A per-chunk max or sum would normalize earlier chunks wrongly.
          x[i] = gauss(rng) + (i >= n - 32 ? 20.0f : 0.0f);
          break;
        case PosInf:
          // +inf in the last chunk, among finite and -inf entries.
          x[i] = i == n - 5 ? pinf : (i % 7 == 3 ? ninf : gauss(rng));
          break;
        case VaddrTie:
          // exp2 is exact at integer t, so the first chunk's e' is
          // {1, 2^-24, 30 x 2^-31}. Aligned to 1.0 (<< 7), each 2^-31 is
          // exactly half a unit: RNE drops it, and the exact sum 1 + 2^-24 is
          // a tie that RNE rounds to 1.0. Round-half-up gives 1 + 2^-23.
          x[i] = i == 0
                     ? 0.0f
                     : (i == 1 ? float(-24 * std::log(2.0))
                               : (i < 32 ? float(-31 * std::log(2.0)) : ninf));
          break;
        default: {
          float scale = scales[s % 5];
          x[i] = scale * gauss(rng) + float(int(s % 7) - 3);
        }
        }
      }
      base += n;
    }
  }

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"softmax.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  const size_t in_bytes = in.size() * 4, out_bytes = OUT_ELEMS * N_SETS * 4;
  xrt::bo bo_in = xrt::ext::bo{device, in_bytes};
  xrt::bo bo_out = xrt::ext::bo{device, out_bytes};
  std::memcpy(bo_in.map<char *>(), in.data(), in_bytes);
  std::memset(bo_out.map<char *>(), 0xff, out_bytes);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_in, bo_out).wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<float> out(OUT_ELEMS * N_SETS);
  std::memcpy(out.data(), bo_out.map<char *>(), out_bytes);

  // Worst errors per row length; relative output errors use ref >= 2^-100.
  double max_rel[ROWS] = {}, max_abs[ROWS] = {}, max_sum_dev[ROWS] = {},
         max_s_vs_e[ROWS] = {}, max_s_vs_ref[ROWS] = {};
  // Exponential and normalization errors across all rows.
  double max_exp_err = 0, max_norm_err = 0, sum_exp_err = 0;
  size_t n_exp = 0, nrows = 0;

  for (size_t s = 0; s < N_SETS; s++) {
    size_t base = 0;
    for (size_t r = 0; r < ROWS; r++) {
      size_t n = ROW_LENGTHS[r];
      const float *x = &in[s * SET_ELEMS + base];
      const float *y = &out[s * OUT_ELEMS + base];
      const float *st = &out[s * OUT_ELEMS + SET_ELEMS + STATS * r];
      float dev_max = st[0], dev_sum = st[1], dev_recip = st[2];
      std::string where = "set " + std::to_string(s) + " (" + pattern_name(s) +
                          ") n=" + std::to_string(n);
      base += n;
      nrows++;

      float mx = ninf;
      for (size_t i = 0; i < n; i++)
        mx = std::max(mx, x[i]);
      if (f32_bits(dev_max) != f32_bits(mx))
        fail(where + ": device max " + hex32(f32_bits(dev_max)) +
             " != " + hex32(f32_bits(mx)));

      // An infinite maximum makes x - max NaN in at least one lane;
      // the NaN sum then propagates to every output.
      if (mx == ninf || mx == pinf) {
        size_t nan = 0;
        for (size_t i = 0; i < n; i++)
          nan += std::isnan(y[i]);
        std::cout << "  " << where << ": " << nan << "/" << n
                  << " NaN outputs, y[0]=" << hex32(f32_bits(y[0]))
                  << " sum=" << dev_sum << " recip=" << dev_recip << std::endl;
        if (nan != n)
          fail(where + ": NaN in every lane expected (policy)");
        continue;
      }

      std::vector<double> ex(n), ref(n);
      double ref_sum = 0;
      for (size_t i = 0; i < n; i++) {
        ex[i] = x[i] == ninf ? 0.0 : std::exp(double(x[i]) - double(mx));
        ref_sum += ex[i];
      }
      for (size_t i = 0; i < n; i++)
        ref[i] = ex[i] / ref_sum;

      // Reconstruct e' as y / recip rounded to VEXP2's 13 significant bits.
      // For normal y, its 24-bit precision uniquely identifies e' among
      // values spaced at least 2^-13 apart relatively. For every y, check that the
      // reconstructed value reproduces the output after multiplication.
      std::vector<float> e(n);
      double sum_y = 0, sum_e = 0;
      for (size_t i = 0; i < n; i++) {
        e[i] = float(round_sig(double(y[i]) / double(dev_recip), 13));
        if (f32_bits(e[i] * dev_recip) != f32_bits(y[i]))
          fail(where + ": e'[" + std::to_string(i) + "] * recip " +
               hex32(f32_bits(e[i] * dev_recip)) + " != y " +
               hex32(f32_bits(y[i])));
        sum_y += y[i];
        sum_e += e[i];
        if (x[i] == ninf) {
          if (f32_bits(y[i]) != 0)
            fail(where + ": masked position " + std::to_string(i) + " is " +
                 hex32(f32_bits(y[i])) + ", want +0");
          continue;
        }
        double err = std::fabs(double(y[i]) - ref[i]);
        if (!(err <= REL_TOL * ref[i] + ABS_TOL))
          fail(where + ": y[" + std::to_string(i) +
               "]=" + std::to_string(y[i]) + " ref=" + std::to_string(ref[i]));
        max_abs[r] = std::max(max_abs[r], err);
        if (ref[i] >= std::ldexp(1.0, -100))
          max_rel[r] = std::max(max_rel[r], err / ref[i]);

        // Check individual error sources where e' is uniquely recoverable
        // and the reference exponential is normal.
        if (y[i] < std::numeric_limits<float>::min() ||
            ex[i] < std::ldexp(1.0, -126))
          continue;
        double exp_err = (double(e[i]) - ex[i]) / ex[i];
        if (!(std::fabs(exp_err) <= EXP_TOL))
          fail(where + ": e'[" + std::to_string(i) +
               "] rel err vs exp(x - m) " + std::to_string(exp_err));
        max_exp_err = std::max(max_exp_err, std::fabs(exp_err));
        sum_exp_err += exp_err;
        n_exp++;
        double norm_err =
            std::fabs(double(y[i]) - double(e[i]) / double(dev_sum));
        if (!(norm_err <= NORM_TOL * y[i]))
          fail(where + ": y[" + std::to_string(i) + "] vs e'/sum rel err " +
               std::to_string(norm_err / y[i]));
        max_norm_err = std::max(max_norm_err, norm_err / y[i]);
      }

      // The device sum must equal VADDR.f applied chunk by chunk to the
      // recovered e', seeded with the running sum, bit for bit.
      float model = 0.0f;
      for (size_t c = 0; c < n; c += 32)
        model = vaddr_f(&e[c], 32, model);
      if (f32_bits(model) != f32_bits(dev_sum))
        fail(where + ": device sum " + hex32(f32_bits(dev_sum)) +
             " != VADDR.f model " + hex32(f32_bits(model)));
      if (s == VaddrTie)
        std::cout << "  " << where << ": sum " << hex32(f32_bits(dev_sum))
                  << ", model " << hex32(f32_bits(model)) << std::endl;

      max_sum_dev[r] = std::max(max_sum_dev[r], std::fabs(sum_y - 1.0));
      max_s_vs_e[r] =
          std::max(max_s_vs_e[r], std::fabs(double(dev_sum) - sum_e) / sum_e);
      // Device sum vs exact sum of exp(x - max) (whose largest term is 1).
      max_s_vs_ref[r] = std::max(
          max_s_vs_ref[r], std::fabs(double(dev_sum) - ref_sum) / ref_sum);
      // Sum to one: the y are the same approximations divided by their own
      // sum, so only f32 rounding remains: one rounding per 32-lane chunk
      // sum, INV (1 ULP) and one product rounding per element.
      double sum_tol = (double(n / 32) + 2.0) * std::ldexp(1.0, -23);
      if (!(std::fabs(sum_y - 1.0) <= sum_tol))
        fail(where + ": sum(y) - 1 = " + std::to_string(sum_y - 1.0));
    }
  }

  double mean_exp_err = n_exp ? sum_exp_err / double(n_exp) : 0.0;
  if (!(std::fabs(mean_exp_err) <= MEAN_TOL))
    fail("mean signed rel err of e' vs exp(x - m) " +
         std::to_string(mean_exp_err) + " (exp2 biased)");

  std::cout << std::setprecision(4) << "Softmax over " << nrows << " rows ("
            << N_SETS << " sets):" << std::endl;
  std::cout << "  e' vs exp(x - m) over " << n_exp << " elements: max rel err "
            << max_exp_err << " (bound " << EXP_TOL << "), mean signed "
            << mean_exp_err << " (bound " << MEAN_TOL << ")" << std::endl;
  std::cout << "  y vs e'/sum: max rel err " << max_norm_err << " (bound "
            << NORM_TOL << ")" << std::endl;
  std::cout << "  n    max_rel_err  max_abs_err  max|sum-1|  "
               "dev_sum_vs_sum_e  dev_sum_vs_exact"
            << std::endl;
  for (size_t r = 0; r < ROWS; r++)
    std::cout << "  " << std::setw(3) << ROW_LENGTHS[r] << "  " << std::setw(11)
              << max_rel[r] << "  " << std::setw(11) << max_abs[r] << "  "
              << std::setw(10) << max_sum_dev[r] << "  " << std::setw(16)
              << max_s_vs_e[r] << "  " << std::setw(16) << max_s_vs_ref[r]
              << std::endl;

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " checks failed." << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
