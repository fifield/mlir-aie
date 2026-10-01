// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Runs entry.elf, clobber.elf, then entry.elf again on NPU3 and checks each
// core's rounding against RNE or floor references:
//   out[0..31]  = in[0..31] + 895.001f  (vector f32 add, crVecRnd)
//   out[32..63] = f32(bf16(bf16(in[32..63]) + 0.005859375)) (scalar bf16 add,
//                 crRnd)
//   out[64..95] = f32(bf16(in[32..63])) (scalar truncation, crRnd)
//   out[96], out[97] = crVecRnd, crRnd read right after core entry
// Usage: test.exe <elf> <rne|floor> [<elf> <rne|floor> ...]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

static uint32_t bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static float from_bits(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
// f32 -> bf16 -> f32 with round-to-nearest-even (finite inputs).
static float bf16_rne(float f) {
  uint32_t u = bits(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return from_bits(u & 0xffff0000u);
}
// f32 -> bf16 -> f32 rounding toward negative infinity.
static float bf16_floor(float f) {
  uint32_t u = bits(f);
  uint32_t t = u & 0xffff0000u;
  if ((u & 0xffffu) && (u >> 31))
    t += 0x10000u;
  return from_bits(t);
}

int main(int argc, char **argv) {
  const int L = 32;
  const float B = 895.001f;
  std::vector<float> in(64), out(128);
  // Add operands: exact ties, negative sums, and plain inexact values.
  for (int k = 0; k < 8; k++)
    in[k] = float(k) * ldexpf(1, -14) + ldexpf(1, -15);
  for (int k = 0; k < 8; k++)
    in[8 + k] =
        -2000.0f - float(k) * ldexpf(1, -12) - ldexpf(1, -13) + 0.1f * k;
  for (int k = 0; k < 16; k++)
    in[16 + k] = 0.1f * float(k + 1) - 0.037f * float(k * k);
  // Truncation operands: bf16 ties and inexact values of both signs.
  for (int k = 0; k < 32; k++) {
    uint32_t u =
        0x3f800000u + (uint32_t(k) << 12) + (k % 3 == 0 ? 0x8000u : 0x1234u);
    if (k & 1)
      u |= 0x80000000u;
    in[32 + k] = from_bits(u);
  }

  auto dev = xrt::device(0);
  int failures = 0;
  for (int a = 1; a + 1 < argc; a += 2) {
    const std::string elf = argv[a];
    const bool want_rne = std::string(argv[a + 1]) == "rne";
    xrt::elf e{elf};
    xrt::hw_context ctx(dev, e);
    auto kernel = xrt::ext::kernel(ctx, "main:sequence");
    xrt::bo bi = xrt::ext::bo{dev, in.size() * 4};
    xrt::bo bo = xrt::ext::bo{dev, out.size() * 4};
    memcpy(bi.map<char *>(), in.data(), in.size() * 4);
    memset(bo.map<char *>(), 0xff, out.size() * 4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kernel(bi, bo).wait2();
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(out.data(), bo.map<char *>(), out.size() * 4);

    int add_inexact = 0, add_ok = 0;
    for (int i = 0; i < L; i++) {
      double v = double(in[i]) + double(B);
      float rne = float(v);
      if (double(rne) == v)
        continue;
      add_inexact++;
      float lo = (double(rne) <= v) ? rne : nextafterf(rne, -INFINITY);
      if (bits(out[i]) == bits(want_rne ? rne : lo))
        add_ok++;
    }
    int cv_inexact = 0, cv_ok = 0;
    for (int i = 0; i < L; i++) {
      float f = in[32 + i];
      if (bf16_rne(f) == bf16_floor(f))
        continue;
      cv_inexact++;
      if (bits(out[64 + i]) == bits(want_rne ? bf16_rne(f) : bf16_floor(f)))
        cv_ok++;
    }
    // bf16 + bf16: exact in f32, then one rounding to bf16 in the mode
    // under test (the operand is the mode-rounded truncation above).
    const float kB = 0.005859375f;
    int badd_inexact = 0, badd_ok = 0;
    for (int i = 0; i < L; i++) {
      float f = in[32 + i];
      float a = want_rne ? bf16_rne(f) : bf16_floor(f);
      float sum = a + kB;
      if (bf16_rne(sum) == bf16_floor(sum))
        continue;
      badd_inexact++;
      if (bits(out[32 + i]) == bits(want_rne ? bf16_rne(sum) : bf16_floor(sum)))
        badd_ok++;
    }
    bool ok = add_ok == add_inexact && cv_ok == cv_inexact &&
              badd_ok == badd_inexact && add_inexact > 0 && cv_inexact > 0 &&
              badd_inexact > 0 && out[96] == 12.0f && out[97] == 12.0f;
    printf("%s (expect %s): crVecRnd=%g crRnd=%g after entry; add %d/%d, "
           "truncf %d/%d, bf16 add %d/%d match -> %s\n",
           elf.c_str(), want_rne ? "RNE" : "floor", out[96], out[97], add_ok,
           add_inexact, cv_ok, cv_inexact, badd_ok, badd_inexact,
           ok ? "ok" : "FAIL");
    failures += !ok;
  }
  if (failures) {
    printf("fail: %d run(s) mismatched\n", failures);
    return 1;
  }
  printf("PASS!\n");
  return 0;
}
