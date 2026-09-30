// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Check int8 -> int32 widening and signed/mixed-sign MACs across all byte
// values, with pinned boundary cases. See unpack_i8.py for the four outputs.

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

constexpr size_t T = 1024;
constexpr size_t L = 32;
constexpr size_t N_OUT = 4;

int main() {
  // Cover every byte value before pinning boundary cases below.
  std::vector<int8_t> in(T);
  for (size_t i = 0; i < T; i++)
    in[i] = int8_t(uint8_t(i * 7 + 3));
  in[0] = -128;
  in[1] = 127;
  in[2] = -1;
  in[3] = 0;
  // Pin the matching lanes in the mirrored chunk for mixed-sign products.
  in[T - L + 0] = 127;
  in[T - L + 1] = -128;
  in[T - L + 2] = -128;
  in[T - L + 3] = -1;

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"unpack_i8.elf"};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_in = xrt::ext::bo{device, T};
  xrt::bo bo_out = xrt::ext::bo{device, N_OUT * T * sizeof(int32_t)};
  memcpy(bo_in.map<char *>(), in.data(), T);
  std::vector<int32_t> out(N_OUT * T, 0x5a5a5a5a);
  memcpy(bo_out.map<char *>(), out.data(), out.size() * sizeof(int32_t));
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_in, bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(out.data(), bo_out.map<char *>(), out.size() * sizeof(int32_t));

  const char *names[N_OUT] = {"sext", "zext", "mac_s8s8_zacc", "mac_u8s8"};
  int errors[N_OUT] = {0, 0, 0, 0};
  for (size_t i = 0; i < T; i++) {
    const size_t ib = (T - L - (i - i % L)) + i % L; // mirrored chunk
    const int32_t sa = in[i], ua = uint8_t(in[i]), sb = in[ib];
    const int32_t want[N_OUT] = {sa, ua, ua + sa * sa, sa + ua * sb};
    for (size_t q = 0; q < N_OUT; q++) {
      if (out[q * T + i] == want[q])
        continue;
      if (errors[q]++ < 4)
        std::cout << "MISMATCH " << names[q] << " i=" << i
                  << " a=" << int(in[i]) << " b=" << int(in[ib])
                  << " got=" << out[q * T + i] << " want=" << want[q]
                  << std::endl;
    }
  }

  int total = 0;
  for (size_t q = 0; q < N_OUT; q++) {
    std::cout << names[q] << ": " << errors[q] << " errors" << std::endl;
    total += errors[q];
  }
  if (total) {
    std::cout << "FAIL" << std::endl;
    return 1;
  }
  std::cout << "PASS!" << std::endl;
  return 0;
}
