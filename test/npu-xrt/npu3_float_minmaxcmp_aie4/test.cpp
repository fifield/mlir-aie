// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match float_minmaxcmp_aie4.py.
constexpr size_t VEC_LANES = 16;
constexpr size_t N_OUT_KINDS = 6;
constexpr size_t IN_BYTES = VEC_LANES * sizeof(float);
constexpr size_t OUT_COUNT = VEC_LANES * N_OUT_KINDS;
constexpr size_t OUT_BYTES = OUT_COUNT * sizeof(float);

// Seed NaNs in DMA inputs to avoid NaN literal parsing differences between
// the host LLVM and Peano versions.
static uint32_t qnan_bits(uint32_t payload) {
  // Positive quiet NaN with the payload in the low 22 mantissa bits.
  return 0x7fc00000u | (payload & 0x3fffffu);
}

static float bits_to_f32(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static uint32_t f32_to_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

int main() {
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"float_minmaxcmp_aie4.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_a = xrt::ext::bo{device, IN_BYTES};
  xrt::bo bo_b = xrt::ext::bo{device, IN_BYTES};
  xrt::bo bo_c = xrt::ext::bo{device, OUT_BYTES};

  float A[VEC_LANES];
  float B[VEC_LANES];

  // lane 0: (NaN, x)          -- minnum/maxnum must avoid the NaN.
  A[0] = bits_to_f32(qnan_bits(0x111111));
  B[0] = 5.0f;
  // lane 1: (x, NaN)
  A[1] = 3.0f;
  B[1] = bits_to_f32(qnan_bits(0x222222));
  // lane 2: (NaN, NaN)        -- minnum/maxnum must still produce NaN.
  A[2] = bits_to_f32(qnan_bits(0x333333));
  B[2] = bits_to_f32(qnan_bits(0x444444));
  // lane 3: two distinct finite values.
  A[3] = 2.0f;
  B[3] = 7.0f;
  // lane 4: (+0, -0) -- ordered-cmpf +/-0 behavior (IEEE: +0 == -0).
  A[4] = bits_to_f32(0x00000000u); // +0.0
  B[4] = bits_to_f32(0x80000000u); // -0.0
  // lane 5: (-0, +0) -- reversed operand order.
  A[5] = bits_to_f32(0x80000000u); // -0.0
  B[5] = bits_to_f32(0x00000000u); // +0.0
  // lanes 6..15: finite values covering both operand orderings and equality.
  for (size_t i = 6; i < VEC_LANES; i++) {
    A[i] = float(i) - 10.0f;       // -4 .. 5
    B[i] = float(VEC_LANES - i) - 8.0f; // descending counterpart
  }

  memcpy(bo_a.map<char *>(), A, IN_BYTES);
  memcpy(bo_b.map<char *>(), B, IN_BYTES);
  memset(bo_c.map<char *>(), 0, OUT_BYTES);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  float out[OUT_COUNT];
  memcpy(out, bo_c.map<char *>(), OUT_BYTES);

  // Output layout: 6 kinds x 16 lanes, kind-major.
  float *r_min = out + 0 * VEC_LANES;
  float *r_max = out + 1 * VEC_LANES;
  float *r_oge = out + 2 * VEC_LANES; // 1.0f/0.0f-encoded boolean
  float *r_olt = out + 3 * VEC_LANES; // 1.0f/0.0f-encoded boolean
  float *r_minnum = out + 4 * VEC_LANES;
  float *r_maxnum = out + 5 * VEC_LANES;

  int errors = 0;
  auto expect_nan = [&](float got, const char *what) {
    if (!std::isnan(got)) {
      std::cout << "MISMATCH " << what << ": got=" << got
                 << " (0x" << std::hex << f32_to_bits(got) << std::dec
                 << ") want=NaN" << std::endl;
      errors++;
    } else {
      std::cout << "OK " << what << " = NaN (propagated), as expected"
                 << std::endl;
    }
  };
  // Compare bits to check signed zeros and exact selection of numeric inputs.
  auto expect_bits = [&](float got, float want, const char *what) {
    uint32_t gb = f32_to_bits(got), wb = f32_to_bits(want);
    if (gb != wb) {
      std::cout << "MISMATCH " << what << ": got=" << got << " (0x" << std::hex
                 << gb << ") want=" << want << " (0x" << wb << std::dec << ")"
                 << std::endl;
      errors++;
    } else {
      std::cout << "OK " << what << " = " << got << std::endl;
    }
  };
  auto expect_bool = [&](float got, bool want, const char *what) {
    bool g = (got != 0.0f);
    if (g != want) {
      std::cout << "MISMATCH " << what << ": got=" << g << " want=" << want
                 << std::endl;
      errors++;
    } else {
      std::cout << "OK " << what << " = " << g << std::endl;
    }
  };

  // -- minimumf / maximumf: NaN propagation (IEEE minimum/maximum) --
  expect_nan(r_min[0], "minimumf lane0 (NaN,x)");
  expect_nan(r_min[1], "minimumf lane1 (x,NaN)");
  expect_nan(r_min[2], "minimumf lane2 (NaN,NaN)");
  expect_bits(r_min[3], 2.0f, "minimumf lane3 (x,y) ordinary");
  expect_nan(r_max[0], "maximumf lane0 (NaN,x)");
  expect_nan(r_max[1], "maximumf lane1 (x,NaN)");
  expect_nan(r_max[2], "maximumf lane2 (NaN,NaN)");
  expect_bits(r_max[3], 7.0f, "maximumf lane3 (x,y) ordinary");

  // -- minimumf / maximumf: +/-0 ordering (IEEE minimum: -0 < +0) --
  expect_bits(r_min[4], -0.0f, "minimumf lane4 (+0,-0) -> -0");
  expect_bits(r_max[4], 0.0f, "maximumf lane4 (+0,-0) -> +0");
  expect_bits(r_min[5], -0.0f, "minimumf lane5 (-0,+0) -> -0");
  expect_bits(r_max[5], 0.0f, "maximumf lane5 (-0,+0) -> +0");

  // Ordered comparisons are false whenever either operand is NaN.
  expect_bool(r_oge[0], false, "cmpf oge lane0 (NaN,x)");
  expect_bool(r_oge[1], false, "cmpf oge lane1 (x,NaN)");
  expect_bool(r_oge[2], false, "cmpf oge lane2 (NaN,NaN)");
  expect_bool(r_olt[0], false, "cmpf olt lane0 (NaN,x)");
  expect_bool(r_olt[1], false, "cmpf olt lane1 (x,NaN)");
  expect_bool(r_olt[2], false, "cmpf olt lane2 (NaN,NaN)");
  // ordinary lane: 2 >= 7 is false, 2 < 7 is true.
  expect_bool(r_oge[3], false, "cmpf oge lane3 (2,7)");
  expect_bool(r_olt[3], true, "cmpf olt lane3 (2,7)");
  // Signed zeros compare equal: oge is true and olt is false in both orders.
  expect_bool(r_oge[4], true, "cmpf oge lane4 (+0,-0)");
  expect_bool(r_olt[4], false, "cmpf olt lane4 (+0,-0)");
  expect_bool(r_oge[5], true, "cmpf oge lane5 (-0,+0)");
  expect_bool(r_olt[5], false, "cmpf olt lane5 (-0,+0)");

  // -- minnumf / maxnumf: numeric fallback with one NaN, NaN with two --
  expect_bits(r_minnum[0], 5.0f, "minnumf lane0 (NaN,x) -> x (avoids NaN)");
  expect_bits(r_minnum[1], 3.0f, "minnumf lane1 (x,NaN) -> x (avoids NaN)");
  expect_nan(r_minnum[2], "minnumf lane2 (NaN,NaN) -> NaN");
  expect_bits(r_minnum[3], 2.0f, "minnumf lane3 (x,y) ordinary");
  expect_bits(r_maxnum[0], 5.0f, "maxnumf lane0 (NaN,x) -> x (avoids NaN)");
  expect_bits(r_maxnum[1], 3.0f, "maxnumf lane1 (x,NaN) -> x (avoids NaN)");
  expect_nan(r_maxnum[2], "maxnumf lane2 (NaN,NaN) -> NaN");
  expect_bits(r_maxnum[3], 7.0f, "maxnumf lane3 (x,y) ordinary");

  // -- remaining finite input pairs --
  for (size_t i = 6; i < VEC_LANES; i++) {
    float a = A[i], b = B[i];
    char buf[64];
    snprintf(buf, sizeof(buf), "minimumf lane%zu sweep", i);
    expect_bits(r_min[i], std::fmin(a, b), buf);
    snprintf(buf, sizeof(buf), "maximumf lane%zu sweep", i);
    expect_bits(r_max[i], std::fmax(a, b), buf);
    snprintf(buf, sizeof(buf), "cmpf oge lane%zu sweep", i);
    expect_bool(r_oge[i], a >= b, buf);
    snprintf(buf, sizeof(buf), "cmpf olt lane%zu sweep", i);
    expect_bool(r_olt[i], a < b, buf);
    snprintf(buf, sizeof(buf), "minnumf lane%zu sweep", i);
    expect_bits(r_minnum[i], std::fmin(a, b), buf);
    snprintf(buf, sizeof(buf), "maxnumf lane%zu sweep", i);
    expect_bits(r_maxnum[i], std::fmax(a, b), buf);
  }

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " checks failed." << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
