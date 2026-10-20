// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host for the stream-B MMAC microbenchmark.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

int main(int argc, const char *argv[]) {
  std::string elf = argv[1];
  unsigned KT = atoi(argv[2]), CALLS = atoi(argv[3]);
  const unsigned MR = 64, N = 64, CW = 64 * 64 + 16;
  std::vector<int8_t> A(KT * MR * 64), B(KT * 4096);
  std::mt19937 g(1);
  std::uniform_int_distribution<int> d(-128, 127);
  for (auto &x : A)
    x = d(g);
  for (auto &x : B)
    x = d(g);
  std::vector<int32_t> ref(MR * N, 0);
  for (unsigned kt = 0; kt < KT; kt++)
    for (unsigned r = 0; r < MR; r++)
      for (unsigned n = 0; n < N; n++)
        for (unsigned k = 0; k < 64; k++)
          ref[r * N + n] += (int32_t)A[(kt * MR + r) * 64 + k] *
                            (int32_t)B[kt * 4096 + n * 64 + k];
  auto dev = xrt::device(0u);
  xrt::elf e{elf};
  xrt::hw_context ctx(dev, e);
  auto kern = xrt::ext::kernel(ctx, "main:sequence");
  xrt::bo ba = xrt::ext::bo{dev, A.size()}, bb = xrt::ext::bo{dev, B.size()},
          bc = xrt::ext::bo{dev, CALLS * CW * 4};
  memcpy(ba.map<int8_t *>(), A.data(), A.size());
  memcpy(bb.map<int8_t *>(), B.data(), B.size());
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  kern(ba, bb, bc).wait2();
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t *C = bc.map<int32_t *>();
  unsigned errs = 0;
  for (unsigned c = 0; c < CALLS; c++) {
    for (unsigned i = 0; i < MR * N; i++)
      if (C[c * CW + i] != ref[i] && errs++ < 5)
        printf("call %u mismatch %u: %d vs %d\n", c, i, C[c * CW + i], ref[i]);
    int32_t *Y = C + c * CW + MR * N;
    unsigned mm = KT * MR;
    printf("call %u: mmac-phase %d cyc (eff %.1f%%), with store %d cyc (eff "
           "%.1f%%)\n",
           c, Y[0], Y[0] ? 100.0 * mm / Y[0] : 0.0, Y[1],
           Y[1] ? 100.0 * mm / Y[1] : 0.0);
  }
  printf("%s (%u errs)\n", errs ? "FAIL" : "PASS", errs);
  return errs != 0;
}
