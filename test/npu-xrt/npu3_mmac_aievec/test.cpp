// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Host for the design in test/aiecc/Inputs/aie4_mmac_design.py.
// Floating-point inputs, products and sums are exactly representable;
// integer references wrap to i32. Exact comparisons check B staging layout,
// nibble order, staging mode, signedness and zero_acc.
// Targeted modes check bf16 subnormal/rounding behavior, fp8 NaNs and a
// four-tile K loop. These use explicit MMAC expectations where needed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match aie4_mmac_design.py.
constexpr int M = 4;
constexpr int N = 64;

namespace {

uint16_t toBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return uint16_t(bits >> 16); // exact for the values used
}

// OCP E4M3 encoding of an exactly representable value.
uint8_t toE4M3(double v) {
  if (std::isnan(v))
    return 0x7f;
  if (v == 0.0)
    return std::signbit(v) ? 0x80 : 0x00;
  uint8_t sign = v < 0 ? 0x80 : 0x00;
  double a = std::fabs(v);
  int e;
  double frac = std::frexp(a, &e); // a = frac * 2^e, frac in [0.5, 1)
  int exp = e - 1 + 7;             // a = 1.m * 2^(e-1)
  if (exp <= 0) {                  // subnormal: a = m/8 * 2^-6
    int mant = int(std::lround(a / std::ldexp(1.0, -9)));
    return sign | uint8_t(mant);
  }
  int mant = int(std::lround((frac * 2.0 - 1.0) * 8.0));
  return sign | uint8_t(exp << 3) | uint8_t(mant);
}

struct Format {
  std::string name;
  int k;
  bool isFloat;
  bool packed4; // two elements per byte, element 2j in the low nibble
  int storeBytes;
  bool lhsSigned, rhsSigned;
};

Format getFormat(const std::string &name) {
  if (name == "bf16")
    return {name, 32, true, false, 2, true, true};
  if (name == "int8")
    return {name, 64, false, false, 1, false, true}; // X unsigned, Z signed
  if (name == "int4")
    return {name, 128, false, true, 1, true, true};
  if (name == "fp8")
    return {name, 64, true, false, 1, true, true};
  std::cerr << "unknown format " << name << "\n";
  std::exit(2);
}

} // namespace

int main(int argc, const char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage: test.exe <elf> <bf16|int8|int4|fp8> <0|2> "
                 "[signs=XY] [overflow|bf16-edge|fp8-nan|kloop]\n";
    return 2;
  }
  const std::string elfFile = argv[1];
  Format f = getFormat(argv[2]);
  bool overflow = false;
  bool bf16Edge = false, fp8Nan = false, kloop = false;
  for (int i = 4; i < argc; ++i) {
    std::string opt = argv[i];
    if (opt == "overflow") {
      overflow = true;
    } else if (opt == "bf16-edge") {
      bf16Edge = true;
    } else if (opt == "fp8-nan") {
      fp8Nan = true;
    } else if (opt == "kloop") {
      kloop = true;
    } else if (opt.rfind("signs=", 0) == 0 && opt.size() == 8) {
      f.lhsSigned = opt[6] == 's';
      f.rhsSigned = opt[7] == 's';
    } else {
      std::cerr << "unknown option " << opt << "\n";
      return 2;
    }
  }
  const int r3Tile = std::atoi(argv[3]);
  if ((bf16Edge && f.name != "bf16") || (fp8Nan && f.name != "fp8") ||
      (kloop && f.name != "bf16") ||
      int(overflow) + int(bf16Edge) + int(fp8Nan) + int(kloop) > 1) {
    std::cerr << "incompatible test mode\n";
    return 2;
  }
  const int K = f.k;
  const int aTiles = kloop ? 4 : 2, bTiles = kloop ? 4 : 3;
  const int results = kloop ? 1 : 3;
  const int rowStore = f.packed4 ? K / 2 : K; // stored units per row
  const int aUnits = M * rowStore, btUnits = N * rowStore;
  const int in0Units = aTiles * aUnits + bTiles * btUnits;

  std::mt19937 gen(1234);
  // Logical values: A[t][m][k], B[t][k][n], C0[m][n].
  std::vector<double> A(aTiles * M * K), B(bTiles * K * N), C0(M * N);
  auto pick = [&](bool isLhs) -> double {
    if (f.name == "bf16") // halves in [-4, 4]
      return std::uniform_int_distribution<int>(-8, 8)(gen) / 2.0;
    if (f.name == "fp8") { // exact E4M3 values
      static const double vals[] = {0,  0.5,  -0.5,  1,     -1, 1.5, -1.5,
                                    2,  -2,   3,     -3,    4,  -4,  6,
                                    -6, 0.25, -0.25, 0.125, 7,  -7};
      return vals[std::uniform_int_distribution<int>(0, 19)(gen)];
    }
    bool isSigned = isLhs ? f.lhsSigned : f.rhsSigned;
    if (f.name == "int8")
      return isSigned ? std::uniform_int_distribution<int>(-128, 127)(gen)
                      : std::uniform_int_distribution<int>(0, 255)(gen);
    return isSigned ? std::uniform_int_distribution<int>(-8, 7)(gen) // int4
                    : std::uniform_int_distribution<int>(0, 15)(gen);
  };
  for (auto &v : A)
    v = pick(true);
  for (auto &v : B)
    v = pick(false);
  for (auto &v : C0)
    v = f.isFloat ? std::uniform_int_distribution<int>(-100, 100)(gen) / 4.0
                  : std::uniform_int_distribution<int>(-100000, 100000)(gen);

  // Planted values: extremes of each format in tile 0.
  if (f.name == "int8") {
    A[0] = f.lhsSigned ? -128 : 255;
    B[0] = f.rhsSigned ? -128 : 255;
    B[1] = f.rhsSigned ? 127 : 0;
  } else if (f.name == "int4") {
    A[0] = f.lhsSigned ? -8 : 15;
    A[1] = f.lhsSigned ? 7 : 0;
    B[0] = f.rhsSigned ? -8 : 15;
    B[N] = f.rhsSigned ? 7 : 0; // B[0][1][0]: k = 1, n = 0
  }

  // Overflow: X = 255 (unsigned), B = +127 in even columns and -128 in odd
  // columns, and C0 within 1000*m + n of INT32_MAX (even) or INT32_MIN (odd).
  // One tile adds +-64*255*127.. = about +-2.08e6, past the i32 range.
  if (overflow) {
    if (f.name != "int8" || f.lhsSigned || !f.rhsSigned) {
      std::cerr << "overflow mode needs int8 with signs=us\n";
      return 2;
    }
    for (auto &v : A)
      v = 255;
    for (int t = 0; t < 3; ++t)
      for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
          B[(t * K + k) * N + n] = (n % 2 == 0) ? 127 : -128;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n)
        C0[m * N + n] = (n % 2 == 0) ? double(INT32_MAX) - 1000 * m - n
                                     : double(INT32_MIN) + 1000 * m + n;
  }

  if (bf16Edge) {
    std::fill(A.begin(), A.end(), 0.0);
    std::fill(B.begin(), B.end(), 0.0);
    std::fill(C0.begin(), C0.end(), 0.0);
    const double minBf16 = std::ldexp(1.0, -133);
    // Row 0: a subnormal A yields a normal result in even columns and a
    // subnormal result (flushed to zero) in odd columns.
    A[0] = minBf16;
    // Row 1: a subnormal B yields a normal result, proving neither
    // multiplicand is flushed before multiplication.
    A[K + 1] = 128.0;
    // Row 2: even columns add a normal product to a subnormal accumulator.
    // Flushing only the final result would produce 1.5 * 2^-126 instead of
    // 2^-126, so this distinguishes input flushing from result flushing.
    A[2 * K + 2] = 1.0;
    // Row 3: one nonzero product removes reduction-order ambiguity. At
    // 2^24, +1 is a halfway case rounded to even; the next tile adds +2.
    A[3 * K + 3] = A[(M + 3) * K + 3] = 1.0;
    for (int n = 0; n < N; ++n) {
      B[n] = n % 2 == 0 ? 128.0 : 1.0;
      B[N + n] = minBf16;
      B[2 * N + n] = n % 2 == 0 ? std::ldexp(1.0, -126) : 0.0;
      B[3 * N + n] = 1.0;
      B[(K + 3) * N + n] = 2.0;
      C0[2 * N + n] = std::ldexp(1.0, -127);
      C0[3 * N + n] = std::ldexp(1.0, 24);
    }
  }
  if (fp8Nan) {
    // Different rows and positions, including the last element, must
    // poison every output column under emode=all. Other rows stay finite.
    A[0] = A[3 * K - 1] = std::numeric_limits<double>::quiet_NaN();
  }
  if (kloop) {
    // Distinct, exactly representable data in every tile. Each tile's B
    // uses a different scale; mismatching A/B or repeating a tile changes
    // the answer.
    for (int t = 0; t < aTiles; ++t) {
      for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
          A[(t * M + m) * K + k] = ((k + 3 * m + t) % 9 - 4) * (t + 1) / 2.0;
      for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
          B[(t * K + k) * N + n] =
              ((3 * k + n + 2 * t) % 11 - 5) * (t + 1) / 4.0;
    }
  }

  auto aAt = [&](int t, int m, int k) { return A[(t * M + m) * K + k]; };
  auto bAt = [&](int t, int k, int n) { return B[(t * K + k) * N + n]; };

  // Pack in0: all A tiles row-major, then all B^T tiles (B column-major).
  std::vector<uint8_t> in0(size_t(in0Units) * f.storeBytes, 0);
  auto putElem = [&](int unitBase, int idx, double v) {
    if (f.packed4) {
      uint8_t nib = uint8_t(int(v) & 0xF);
      uint8_t &byte = in0[unitBase + idx / 2];
      byte |= (idx % 2 == 0) ? nib : uint8_t(nib << 4);
    } else if (f.name == "bf16") {
      uint16_t h = toBf16(float(v));
      std::memcpy(&in0[size_t(unitBase + idx) * 2], &h, 2);
    } else if (f.name == "fp8") {
      in0[unitBase + idx] = toE4M3(v);
    } else {
      in0[unitBase + idx] = uint8_t(int(v) & 0xFF);
    }
  };
  for (int t = 0; t < aTiles; ++t)
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k)
        putElem(t * aUnits + m * rowStore, k, aAt(t, m, k));
  for (int t = 0; t < bTiles; ++t) {
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k)
        putElem(aTiles * aUnits + t * btUnits + n * rowStore, k, bAt(t, k, n));
  }

  // References: R1 = C0 + A0 B0, R2 = R1 + A1 B1, R3 = A0 B<r3Tile>.
  std::vector<double> ref(results * M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      if (kloop) {
        double sum = C0[m * N + n];
        for (int t = 0; t < aTiles; ++t)
          for (int k = 0; k < K; ++k)
            sum += aAt(t, m, k) * bAt(t, k, n);
        ref[m * N + n] = sum;
        continue;
      }
      double p0 = 0, p1 = 0, p3 = 0;
      for (int k = 0; k < K; ++k) {
        p0 += aAt(0, m, k) * bAt(0, k, n);
        p1 += aAt(1, m, k) * bAt(1, k, n);
        p3 += aAt(0, m, k) * bAt(r3Tile, k, n);
      }
      ref[0 * M * N + m * N + n] = C0[m * N + n] + p0;
      ref[1 * M * N + m * N + n] = C0[m * N + n] + p0 + p1;
      ref[2 * M * N + m * N + n] = p3;
    }
  if (bf16Edge) {
    // Explicit hardware expectations: do not use the unflushed double
    // reference for subnormal results or the halfway rounding case.
    for (int r = 0; r < results; ++r)
      for (int n = 0; n < N; ++n) {
        ref[r * M * N + n] = n % 2 == 0 ? std::ldexp(1.0, -126) : 0.0;
        ref[r * M * N + N + n] = std::ldexp(1.0, -126);
        ref[r * M * N + 2 * N + n] = n % 2 == 0 ? std::ldexp(1.0, -126) : 0.0;
        ref[r * M * N + 3 * N + n] =
            r == 2 ? 1.0 : std::ldexp(1.0, 24) + (r == 1 ? 2.0 : 0.0);
      }
  }

  auto device = xrt::device(0);
  xrt::elf ctxElf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctxElf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo0 = xrt::ext::bo{device, in0.size()};
  xrt::bo bo1 = xrt::ext::bo{device, size_t(M * N) * 4};
  xrt::bo bo2 = xrt::ext::bo{device, size_t(results * M * N) * 4};
  std::memcpy(bo0.map<uint8_t *>(), in0.data(), in0.size());
  if (f.isFloat) {
    float *p = bo1.map<float *>();
    for (int i = 0; i < M * N; ++i)
      p[i] = float(C0[i]);
  } else {
    int32_t *p = bo1.map<int32_t *>();
    for (int i = 0; i < M * N; ++i)
      p[i] = int32_t(C0[i]);
  }
  std::memset(bo2.map<uint8_t *>(), 0xA5, size_t(results * M * N) * 4);
  bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo0, bo1, bo2).wait2();
  bo2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  const char *names[3] = {"R1 (1 tile, acc)", "R2 (2 tiles, C carried)",
                          "R3 (zero_acc)"};
  // Integer references wrap to i32 like arith.addi; count saturating
  // matches separately to tell the two apart when the sum overflows.
  auto wrapI32 = [](double v) {
    return double(int32_t(uint32_t(int64_t(v) & 0xffffffffLL)));
  };
  auto satI32 = [](double v) {
    return std::fmin(std::fmax(v, double(INT32_MIN)), double(INT32_MAX));
  };
  int errors = 0;
  for (int r = 0; r < results; ++r) {
    int bad = 0, satMatches = 0, overflowed = 0;
    for (int i = 0; i < M * N; ++i) {
      double got = f.isFloat ? double(bo2.map<float *>()[r * M * N + i])
                             : double(bo2.map<int32_t *>()[r * M * N + i]);
      double want =
          f.isFloat ? ref[r * M * N + i] : wrapI32(ref[r * M * N + i]);
      if (!f.isFloat) {
        overflowed += satI32(ref[r * M * N + i]) != ref[r * M * N + i];
        satMatches += got == satI32(ref[r * M * N + i]);
      }
      const bool matched = std::isnan(want) ? std::isnan(got) : got == want;
      if (!matched) {
        if (bad < 6)
          std::cout << "  " << (kloop ? "K loop (4 tiles)" : names[r]) << " ["
                    << i / N << "][" << i % N << "]: got " << got
                    << ", expected " << want << "\n";
        ++bad;
      }
    }
    if (kloop)
      std::cout << "K loop (4 tiles)";
    else
      std::cout << f.name << " " << names[r];
    std::cout << ": " << (M * N - bad) << "/" << M * N
              << (bf16Edge || fp8Nan ? " matched\n" : " exact\n");
    if (overflow) {
      // Show the first positive and negative overflow of row 0.
      for (int i = 0; i < 2; ++i) {
        double exact = ref[r * M * N + i];
        std::cout << "  [0][" << i << "] exact " << std::fixed
                  << std::setprecision(0) << exact << ", wrapped "
                  << wrapI32(exact) << ", saturated " << satI32(exact)
                  << ", hardware " << bo2.map<int32_t *>()[r * M * N + i]
                  << "\n";
      }
      std::cout << "  overflowing sums: " << overflowed
                << ", wrapping matches: " << (M * N - bad)
                << ", saturating matches: " << satMatches << "\n";
    }
    errors += bad;
  }
  if (errors) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
