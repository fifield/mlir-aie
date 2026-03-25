// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

constexpr int DATA_SIZE = 64;
constexpr int ADD_VAL = 41;

int main(int argc, const char *argv[]) {

  // Set up input data
  std::vector<int32_t> vec_in(DATA_SIZE);
  std::vector<int32_t> vec_out(DATA_SIZE);
  for (int i = 0; i < DATA_SIZE; i++) {
    vec_in[i] = i + 1;
    vec_out[i] = 0xdeadbeef;
  }

  // Set up XRT with full ELF flow
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  constexpr size_t BUF_SIZE = DATA_SIZE * sizeof(int32_t);
  xrt::bo bo_in = xrt::ext::bo{device, BUF_SIZE};
  xrt::bo bo_buf = xrt::ext::bo{device, BUF_SIZE};
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  char *buf_in = bo_in.map<char *>();
  char *buf_out = bo_out.map<char *>();

  memcpy(buf_in, vec_in.data(), BUF_SIZE);
  memset(buf_out, 0, BUF_SIZE);

  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Run kernel
  kernel(bo_in, bo_buf, bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(vec_out.data(), buf_out, BUF_SIZE);

  // Validate: each element should be input + ADD_VAL
  int errors = 0;
  for (int i = 0; i < DATA_SIZE; i++) {
    int32_t ref = vec_in[i] + ADD_VAL;
    if (vec_out[i] != ref) {
      std::cout << "Error at " << i << ": " << vec_out[i] << " != " << ref
                << std::endl;
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
