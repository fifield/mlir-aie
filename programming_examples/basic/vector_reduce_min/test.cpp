///===- test.cpp -------------------------------------------000---*- C++
///-*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

#include "test_utils.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED

using INOUT0_DATATYPE = std::int32_t;
using INOUT1_DATATYPE = std::int32_t;
#endif

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Vector Reduce Min Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);
  // AIE4/medusa full-ELF flow: load a single design ELF instead of xclbin+insts.
  options.add_options()("full-elf",
                        "path to full design ELF (AIE4/npu3 flow)",
                        cxxopts::value<std::string>());

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  int INOUT0_VOLUME = 1024;
  int INOUT1_VOLUME = 1;

  size_t INOUT0_SIZE = INOUT0_VOLUME * sizeof(INOUT0_DATATYPE);
  size_t INOUT1_SIZE = INOUT1_VOLUME * sizeof(INOUT1_DATATYPE);

  // TODO Remove trace for now?
  size_t OUT_SIZE = INOUT1_SIZE + trace_size;

  bool use_full_elf = vm.count("full-elf") > 0;

  // ------------------------------------------------------
  // Get device, load the kernel via the appropriate flow
  // ------------------------------------------------------
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string Node = vm["kernel"].as<std::string>();

  // Allocate buffers up front; both flows share these.
  xrt::bo bo_inout0 = xrt::ext::bo{device, INOUT0_SIZE};
  xrt::bo bo_inout1 = xrt::ext::bo{device, INOUT1_SIZE};

  // Set up flow-specific kernel handles.
  std::vector<uint32_t> instr_v;
  xrt::bo bo_instr;
  xrt::kernel kernel_xclbin;
  std::unique_ptr<xrt::ext::kernel> kernel_full_elf;
  std::unique_ptr<xrt::hw_context> hw_ctx;
  std::unique_ptr<xrt::elf> design_elf_obj;

  if (use_full_elf) {
    // AIE4/medusa: single full ELF, no xclbin.
    std::string elf_path = vm["full-elf"].as<std::string>();
    if (verbosity >= 1)
      std::cout << "Loading full ELF: " << elf_path << "\n";
    design_elf_obj = std::make_unique<xrt::elf>(elf_path);
    hw_ctx = std::make_unique<xrt::hw_context>(device, *design_elf_obj);
    if (verbosity >= 1)
      std::cout << "Getting handle to kernel: " << Node << "\n";
    kernel_full_elf = std::make_unique<xrt::ext::kernel>(*hw_ctx, Node);
  } else {
    // npu1/npu2: xclbin + insts ELF.
    instr_v = test_utils::load_instr_binary(vm["instr"].as<std::string>());
    if (verbosity >= 1)
      std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    test_utils::init_xrt_load_kernel(device, kernel_xclbin, verbosity,
                                     vm["xclbin"].as<std::string>(),
                                     vm["kernel"].as<std::string>());

    bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                       XCL_BO_FLAGS_CACHEABLE, kernel_xclbin.group_id(1));

    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize Inout buffer 0
  INOUT0_DATATYPE *bufInOut0 = bo_inout0.map<INOUT0_DATATYPE *>();
  std::int32_t min = (std::int32_t)INT32_MAX;
  const int input_shift = INOUT0_VOLUME / 3;
  for (int i = 0; i < INOUT0_VOLUME; i++) {
    std::int32_t next =
        (i + INOUT0_VOLUME - input_shift) % INOUT0_VOLUME - INOUT0_VOLUME / 2;
    if (next < min)
      min = next;
    bufInOut0[i] = next;
  }

  // Sync buffers to update input buffer values
  bo_inout0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }

    // Run kernel
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::high_resolution_clock::now();
    if (use_full_elf) {
      (*kernel_full_elf)(bo_inout0, bo_inout1).wait2();
    } else {
      unsigned int opcode = 3;
      auto run = kernel_xclbin(opcode, bo_instr, instr_v.size(), bo_inout0,
                               bo_inout1);
      run.wait();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_inout1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    INOUT1_DATATYPE *bufInOut1 = bo_inout1.map<INOUT1_DATATYPE *>();

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // Copy output results and verify they are correct
    if (do_verify) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      if (bufInOut1[0] != min) {
        errors++;
        std::cout << "min is " << min << " calc " << bufInOut1[0] << std::endl;
      }
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // Write trace values if trace_size > 0
    if (trace_size > 0) {
      test_utils::write_out_trace(((char *)bufInOut1) + INOUT1_SIZE, trace_size,
                                  vm["trace_file"].as<std::string>());
    }

    // Accumulate run times
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ------------------------------------------------------
  // Print verification and timing results
  // ------------------------------------------------------

  // TODO - Mac count to guide gflops
  float macs = 0;

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
