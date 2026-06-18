//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int OUT_SIZE = 9 * 2; // 9 values per column (3x3 grid)

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("systolic_cascade");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out0 = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out1 = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out2 = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing instruction buffer.\n";

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_out0, bo_out1, bo_out2);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut0 = bo_out0.map<uint32_t *>();
  uint32_t *bufOut1 = bo_out1.map<uint32_t *>();
  uint32_t *bufOut2 = bo_out2.map<uint32_t *>();

  int errors = 0;

  uint32_t expected[3][9] = {
      {0, 0, 0, 3, 0, 0, 6, 0, 0}, // Column 0: values at indices 0, 3, 6
      {0, 1, 0, 0, 4, 0, 0, 7, 0}, // Column 1: values at indices 1, 4, 7
      {0, 0, 2, 0, 0, 5, 0, 0, 8}  // Column 2: values at indices 2, 5, 8
  };

  // Check phase 0
  if (verbosity >= 1) {
    std::cout << "\nChecking results:\n";
    std::cout << "Column 0:\n";
    for (int i = 0; i < OUT_SIZE / 2; i++) {
      std::cout << "  bufOut0[" << i << "] = " << bufOut0[i] << " (expected "
                << expected[0][i] << ")\n";
    }
    std::cout << "Column 1:\n";
    for (int i = 0; i < OUT_SIZE / 2; i++) {
      std::cout << "  bufOut1[" << i << "] = " << bufOut1[i] << " (expected "
                << expected[1][i] << ")\n";
    }
    std::cout << "Column 2:\n";
    for (int i = 0; i < OUT_SIZE / 2; i++) {
      std::cout << "  bufOut2[" << i << "] = " << bufOut2[i] << " (expected "
                << expected[2][i] << ")\n";
    }
  }

  // Check column 0
  for (int i = 0; i < OUT_SIZE / 2; i++) {
    if (bufOut0[i] != expected[0][i]) {
      std::cout << "Error in column 0 at index " << i << ": " << bufOut0[i]
                << " != " << expected[0][i] << std::endl;
      errors++;
    }
  }

  // Check column 1
  for (int i = 0; i < OUT_SIZE / 2; i++) {
    if (bufOut1[i] != expected[1][i]) {
      std::cout << "Error in column 1 at index " << i << ": " << bufOut1[i]
                << " != " << expected[1][i] << std::endl;
      errors++;
    }
  }

  // Check column 2
  for (int i = 0; i < OUT_SIZE / 2; i++) {
    if (bufOut2[i] != expected[2][i]) {
      std::cout << "Error in column 2 at index " << i << ": " << bufOut2[i]
                << " != " << expected[2][i] << std::endl;
      errors++;
    }
  }

  uint32_t expected_phase1[3][9] = {
      {0, 0, 0, 3, 3, 3, 9, 9, 9},       // Col 0: horizontal accumulated sums
      {0, 1, 1, 4, 11, 14, 23, 39, 48},  // Col 1: horizontal accumulated sums
      {0, 1, 4, 8, 19, 38, 61, 100, 156} // Col 2: horizontal accumulated sums
  };

  // Check Phase 1
  for (int i = 0; i < OUT_SIZE / 2; i++) {
    if (bufOut0[(OUT_SIZE / 2) + i] != expected_phase1[0][i]) {
      std::cout << "Error in column 0 at index " << i << ": "
                << bufOut0[(OUT_SIZE / 2) + i]
                << " != " << expected_phase1[0][i] << std::endl;
      errors++;
    }
  }
  for (int i = 0; i < OUT_SIZE / 2; i++) {
    if (bufOut1[(OUT_SIZE / 2) + i] != expected_phase1[1][i]) {
      std::cout << "Error in column 1 at index " << i << ": "
                << bufOut1[(OUT_SIZE / 2) + i]
                << " != " << expected_phase1[1][i] << std::endl;
      errors++;
    }
  }
  for (int i = 0; i < OUT_SIZE / 2; i++) {
    if (bufOut2[(OUT_SIZE / 2) + i] != expected_phase1[2][i]) {
      std::cout << "Error in column 2 at index " << i << ": "
                << bufOut2[(OUT_SIZE / 2) + i]
                << " != " << expected_phase1[2][i] << std::endl;
      errors++;
    }
  }
  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nFailed with " << errors << " errors.\n\n";
    return 1;
  }
}
