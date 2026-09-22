// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shift-zero UPS must sign-extend i16 inputs exactly, including both extremes.

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

constexpr size_t TILE_SIZE = 1024;
constexpr size_t N_CORES = 3;
constexpr size_t DATA_COUNT = TILE_SIZE * N_CORES; // 3072
constexpr size_t IN_BYTES = DATA_COUNT * sizeof(int16_t);
constexpr size_t OUT_BYTES = DATA_COUNT * sizeof(int32_t);

int main() {
  std::vector<int16_t> vec_in(DATA_COUNT);
  std::vector<int32_t> vec_out(DATA_COUNT, 0);

  // Planted boundary values first.
  std::vector<int16_t> planted = {-32768, 32767, -1, 0, 1, -2, 16384, -16384,
                                   256,    -256,  100, -100};
  for (size_t i = 0; i < planted.size(); i++)
    vec_in[i] = planted[i];

  // Dense sweep across as much of the i16 range as fits, then random fill
  // for the remainder.
  size_t idx = planted.size();
  int32_t sweep_step = 65536 / int32_t(DATA_COUNT - planted.size() + 1);
  if (sweep_step < 1)
    sweep_step = 1;
  int32_t v = -32768;
  for (; idx < DATA_COUNT; idx++) {
    vec_in[idx] = int16_t(v);
    v += sweep_step;
    if (v > 32767)
      v = -32768 + (v - 32768);
  }

  // Set up XRT.
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"ups_i16_i32.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_in = xrt::ext::bo{device, IN_BYTES};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_BYTES};

  memcpy(bo_in.map<char *>(), vec_in.data(), IN_BYTES);
  memset(bo_out.map<char *>(), 0, OUT_BYTES);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_in, bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(vec_out.data(), bo_out.map<char *>(), OUT_BYTES);

  // Exact comparison against plain sign extension.
  int errors = 0;
  for (size_t i = 0; i < DATA_COUNT; i++) {
    int32_t want = int32_t(vec_in[i]); // sign extension via the C++ promotion
    if (vec_out[i] != want) {
      if (errors < 16) {
        std::cout << "MISMATCH at " << i << ": in=" << vec_in[i]
                   << " got=" << vec_out[i] << " want=" << want;
        if (i < planted.size())
          std::cout << "  [planted boundary value]";
        std::cout << std::endl;
      }
      errors++;
    }
  }

  std::cout << (errors == 0 ? "CLAIM 2a RESULT: CONFIRMED -- UPS at shift 0 "
                               "is bit-exact sign extension across the full "
                               "i16 range."
                             : "CLAIM 2a RESULT: REFUTED -- UPS at shift 0 "
                               "diverges from sign extension somewhere in "
                               "range; see mismatches above.")
            << std::endl;

  if (errors == 0) {
    std::cout << "PASS! 🐬" << std::endl;
    return 0;
  }
  std::cout << errors << " / " << DATA_COUNT << " elements mismatched."
            << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
