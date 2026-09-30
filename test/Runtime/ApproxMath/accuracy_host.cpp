// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../../npu-xrt/npu3_approx_math_aie4/accuracy.h"

#include <utility>

using namespace approx_math_accuracy;

namespace {
int failures = 0;
void expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << '\n';
    failures++;
  }
}

void expect_bf16(float b, uint16_t bits, bool valid) {
  Stat st;
  errors = 0;
  check_bf16("bf16 regression", b, bits, st);
  expect((errors == 0) == valid, "bf16 b=" + std::to_string(b) +
                                     " got=" + hex16(bits) +
                                     (valid ? " rejected" : " accepted"));
  if (!valid)
    expect(st.nbad != 0, "bf16 failure missing from violation statistics");
}

void expect_bf20(float b, uint32_t bits, bool valid) {
  Stat st;
  errors = 0;
  check_bf20("bf20 regression", b, bits_f32(bits), st);
  expect((errors == 0) == valid, "bf20 b=" + std::to_string(b) +
                                     " got=" + hex32(bits) +
                                     (valid ? " rejected" : " accepted"));
  if (!valid)
    expect(st.nbad != 0, "bf20 failure missing from violation statistics");
}
} // namespace

int main() {
  // Expected validator failures should not obscure regression diagnostics.
  std::ostringstream diagnostics;
  auto *saved = std::cout.rdbuf(diagnostics.rdbuf());

  // Quiet/signaling NaNs of either sign and infinities must fail for finite
  // objectives in both the normal and subnormal result ranges.
  for (uint16_t bits : {0x7fc0, 0xffc0, 0x7f81, 0xff81, 0x7f80, 0xff80}) {
    for (float b : {895.5f, 761.5f}) {
      expect_bf16(b, bits, false);
      expect_bf20(b, uint32_t(bits) << 16, false);
    }
    for (int mbits : {0, 7, 10}) {
      Stat st;
      st.add(0.5, 1.0, 1.0, mbits);
      st.add(1.0, bf16_to_f32(bits), 1.0, mbits);
      st.add(1.5, 1.0, 1.0, mbits);
      expect(st.n == 3 && st.nbad == 1, "invalid sample was not counted");
      expect(st.max_abs > 0 && st.max_rel > 0 && st.max_ulp > 1 &&
                 std::fabs(st.mean_signed_rel()) > 5e-5,
             "invalid sample bypassed an aggregate error bound");
    }
  }

  // Below 1, the measured NPU3 bound uses the spacing at RNE(2^t) = 1.
  // This admits results through one 2^-7 ULP on either side of RNE.
  expect_bf16(894.99951171875f, 0x3f7f, true);
  expect_bf16(894.99951171875f, 0x3f80, true);
  expect_bf16(894.99951171875f, 0x3f7e, true);
  expect_bf16(894.99951171875f, 0x3f81, true);
  expect_bf16(894.99951171875f, 0x3f7d, false);
  expect_bf16(894.99951171875f, 0x3f82, false);

  // The same <= 1 ULP bound applies immediately above the power of two.
  expect_bf16(895.00048828125f, 0x3f80, true);
  expect_bf16(895.00048828125f, 0x3f81, true);
  expect_bf16(895.00048828125f, 0x3f7f, true);
  expect_bf16(895.00048828125f, 0x3f82, false);

  // Exactly representable objectives accept adjacent values within the
  // measured one-ULP bound.
  for (auto [b, bits] : {std::pair{895.0f, 0x3f80}, std::pair{769.0f, 0x0080},
                         std::pair{762.0f, 0x0001}}) {
    expect_bf16(b, bits, true);
    expect_bf20(b, uint32_t(bits) << 16, true);
  }

  // Gradual underflow: zero can bracket a positive objective; the spacing
  // remains 2^-133 on both sides of the smallest normal value.
  expect_bf16(761.00048828125f, 0x0000, true);
  expect_bf16(761.00048828125f, 0x0001, true);
  expect_bf16(761.00048828125f, 0x0002, true);
  expect_bf16(761.00048828125f, 0x0003, false);
  expect_bf16(768.99951171875f, 0x007f, true);
  expect_bf16(768.99951171875f, 0x0080, true);
  expect_bf16(768.99951171875f, 0x007e, false);
  expect_bf16(768.99951171875f, 0x0081, true);
  expect_bf16(768.99951171875f, 0x007d, false);
  expect_bf16(768.99951171875f, 0x0082, false);

  // Below the overflow threshold, finite values within one ULP of the rounded
  // objective are allowed; infinity remains invalid for a finite objective.
  expect_bf16(1022.99951171875f, 0x7f7f, true);
  expect_bf16(1022.99951171875f, 0x7f7e, true);
  expect_bf16(1022.99951171875f, 0x7f7d, false);
  expect_bf16(1022.99951171875f, 0x7f80, false);
  expect_bf16(1023.0f, 0x7f80, true);
  expect_bf16(1023.0f, 0x7fc0, false);
  expect_bf16(761.0f, 0x0000, true);
  expect_bf16(761.0f, 0x7fc0, false);
  expect_bf20(1023.0f, 0x7f800000, true);
  expect_bf20(1023.0f, 0x7fc00000, false);
  expect_bf20(756.0f, 0x00000000, true);
  expect_bf20(756.0f, 0x7fc00000, false);

  Stat exact;
  exact.add(0.0, 0.0, 0.0, 10);
  exact.add(1.0, 1.0, 1.0, 10);
  expect(exact.n == 2 && exact.nbad == 0 && exact.max_abs == 0 &&
             exact.max_rel == 0 && exact.max_ulp == 0 &&
             exact.mean_signed_rel() == 0,
         "exact finite samples should have zero error");

  std::cout.rdbuf(saved);
  if (failures)
    return 1;
  std::cout << "approx_math_accuracy: all checks passed\n";
  return 0;
}
