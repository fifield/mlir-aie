// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#define DTYPE float
constexpr size_t DATA_COUNT = 1024 * 3 * 16;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(DTYPE);

int main(int argc, const char *argv[]) {
  // Set up input data
  std::vector<DTYPE> vec_in0(DATA_COUNT);
  std::vector<DTYPE> vec_in1(DATA_COUNT);
  std::vector<DTYPE> vec_out(DATA_COUNT);
  for (int i = 0; i < vec_in0.size(); i++) {
    // Values in [-8, 8) with 1/64 granularity: exactly representable in f32,
    // and their products stay far from any subnormal/overflow edge.
    vec_in0[i] = DTYPE((rand() % 1024) - 512) / 64.0f;
    vec_in1[i] = DTYPE((rand() % 1024) - 512) / 64.0f;
    vec_out[i] = NAN;
  }

  // Set up XRT
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  // The name format here is <kernel_name>:<instance_name> from the config.json
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"vector_vector_mul_f32.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);
  xrt::bo bo_in0 = xrt::ext::bo{device, BUF_SIZE};
  xrt::bo bo_in1 = xrt::ext::bo{device, BUF_SIZE};
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  // Set up kernel run
  char *buf_in0 = bo_in0.map<char *>();
  char *buf_in1 = bo_in1.map<char *>();
  char *buf_out = bo_out.map<char *>();

  memcpy(buf_in0, vec_in0.data(), BUF_SIZE);
  memcpy(buf_in1, vec_in1.data(), BUF_SIZE);
  memcpy(buf_out, vec_out.data(), BUF_SIZE);

  bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Run
  auto t_start = std::chrono::high_resolution_clock::now();
  kernel(bo_in0, bo_in1, bo_out).wait2();
  auto t_stop = std::chrono::high_resolution_clock::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Validate output
  std::vector<DTYPE> vec_ref(DATA_COUNT);
  memcpy(vec_out.data(), buf_out, BUF_SIZE);
  // Reference: plain host-side f32 multiply. The NPU computes the same
  // products via a single `vmul.f` per 32 lanes, so agreement should be
  // near-exact; the tolerance only allows for f32 rounding, not for a
  // different arithmetic mode. A wrong multiplier config word would show up
  // here as grossly wrong values, not as a last-ulp difference.
  constexpr DTYPE REL_TOL = 1e-6f;
  constexpr DTYPE ABS_TOL = 1e-6f;
  int errors = 0;
  for (int i = 0; i < DATA_COUNT; i++) {
    vec_ref[i] = vec_in0[i] * vec_in1[i];
    DTYPE diff = std::fabs(vec_out[i] - vec_ref[i]);
    DTYPE tol = ABS_TOL + REL_TOL * std::fabs(vec_ref[i]);
    if (!(diff <= tol)) {
      if (errors < 16) {
        std::cout << "MISMATCH at " << i << ": in0=" << vec_in0[i]
                  << " in1=" << vec_in1[i] << " out=" << vec_out[i]
                  << " ref=" << vec_ref[i] << " diff=" << diff << std::endl;
      }
      errors++;
    }
  }
  bool outputs_correct = (errors == 0);

  // Report results
  float time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  std::cout << "Elapsed time for all kernel executions: " << std::fixed
            << std::setprecision(0) << std::setw(8) << time << " μs"
            << std::endl;
  if (outputs_correct) {
    std::cout << "PASS! 🐬" << std::endl;
  } else {
    std::cout << errors << " / " << DATA_COUNT << " elements mismatched."
              << std::endl;
    std::cout << "fail." << std::endl;
  }

  return (outputs_correct ? 0 : 1);
}
