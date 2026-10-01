// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Vector arith.fptosi / arith.fptoui on NPU3 truncate toward zero even when
// the core is entered with crRnd set to nearest even. See run.lit.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {
// Must match fptoint.py.
constexpr size_t LANES = 32;
constexpr size_t IN_WORDS = LANES;
constexpr size_t OUT_WORDS = 3 * LANES;
constexpr size_t CHUNKS = 4 * 16;
constexpr size_t NPTS = CHUNKS * LANES;
} // namespace

int main() {
  std::vector<float> x(NPTS);
  // Halves and other fractions, where nearest even and toward zero differ,
  // on both sides of zero.
  const float sp[] = {0.5f,     -0.5f,     1.5f,  -1.5f,  2.5f,   -2.5f,
                      0.25f,    -0.25f,    0.75f, -0.75f, 0.999f, -0.999f,
                      1e-30f,   -1e-30f,   0.0f,  -0.0f,  7.0f,   -7.0f,
                      1234.75f, -1234.75f, 1e9f,  -1e9f,  3.5f,   -3.5f};
  size_t i = 0;
  for (float v : sp)
    x[i++] = v;
  for (; i < NPTS; i++)
    x[i] = float((double(i) - NPTS / 2.0) * 0.3125 + 0.01);

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"fptoint.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  size_t in_bytes = NPTS * sizeof(float), out_bytes = CHUNKS * OUT_WORDS * 4;
  xrt::bo bo_in = xrt::ext::bo{device, in_bytes};
  xrt::bo bo_out = xrt::ext::bo{device, out_bytes};
  std::memcpy(bo_in.map<void *>(), x.data(), in_bytes);
  std::memset(bo_out.map<void *>(), 0xff, out_bytes);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kernel(bo_in, bo_out).wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const int32_t *out = bo_out.map<const int32_t *>();

  int errors = 0;
  auto fail = [&](const std::string &m) {
    if (errors++ < 20)
      std::cout << "FAIL: " << m << std::endl;
  };
  size_t n_si = 0, n_ui = 0;
  for (size_t p = 0; p < NPTS; p++) {
    size_t base = p / LANES * OUT_WORDS, l = p % LANES;
    double t = std::trunc(double(x[p]));
    int32_t si = out[base + l];
    uint32_t ui = uint32_t(out[base + LANES + l]);
    n_si++;
    if (si != int32_t(t))
      fail("fptosi " + std::to_string(x[p]) + " -> " + std::to_string(si));
    // fptoui is defined only when the truncated value is non-negative.
    if (t >= 0) {
      n_ui++;
      if (ui != uint32_t(t))
        fail("fptoui " + std::to_string(x[p]) + " -> " + std::to_string(ui));
    }
  }
  for (size_t ch = 0; ch < CHUNKS; ch++)
    if (out[ch * OUT_WORDS + 2 * LANES] != 2) {
      fail("crRnd in the core is " +
           std::to_string(out[ch * OUT_WORDS + 2 * LANES]) + ", not 2");
      break;
    }
  std::cout << "fptosi: " << n_si << " points, fptoui: " << n_ui << " points"
            << std::endl;
  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " errors" << std::endl;
  return 1;
}
