//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
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

int main(int argc, const char *argv[]) {
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Vector Vector Mul Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);
  // AIE4/medusa full-ELF flow: load a single design ELF instead of xclbin+insts.
  options.add_options()("full-elf",
                        "path to full design ELF (AIE4/npu3 flow)",
                        cxxopts::value<std::string>());

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  constexpr int IN_SIZE = 256;
  constexpr int OUT_SIZE = 256;

  // ------------------------------------------------------
  // Get device, load the kernel via the appropriate flow
  // ------------------------------------------------------
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string Node = vm["kernel"].as<std::string>();
  bool use_full_elf = vm.count("full-elf") > 0;

  // Allocate buffers up front; both flows share these.
  xrt::bo bo_inA = xrt::ext::bo{device, IN_SIZE * sizeof(int32_t)};
  xrt::bo bo_inB = xrt::ext::bo{device, IN_SIZE * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_SIZE * sizeof(int32_t)};

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint32_t *bufInA = bo_inA.map<uint32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  uint32_t *bufInB = bo_inB.map<uint32_t *>();
  std::vector<uint32_t> srcVecB;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecB.push_back(i);
  memcpy(bufInB, srcVecB.data(), (srcVecB.size() * sizeof(uint32_t)));

  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";

  if (use_full_elf) {
    // AIE4/medusa: single full ELF, no xclbin.
    std::string elf_path = vm["full-elf"].as<std::string>();
    if (verbosity >= 1)
      std::cout << "Loading full ELF: " << elf_path << "\n";
    xrt::elf design_elf(elf_path);
    xrt::hw_context context(device, design_elf);
    if (verbosity >= 1)
      std::cout << "Getting handle to kernel: " << Node << "\n";
    xrt::ext::kernel kernel(context, Node);
    kernel(bo_inA, bo_inB, bo_out).wait2();
  } else {
    // npu1/npu2: xclbin + insts ELF.
    if (verbosity >= 1)
      std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

    if (verbosity >= 1)
      std::cout << "Kernel opcode: " << Node << "\n";

    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                 [Node, verbosity](xrt::xclbin::kernel &k) {
                                   auto name = k.get_name();
                                   if (verbosity >= 1) {
                                     std::cout << "Name: " << name << std::endl;
                                   }
                                   return name.rfind(Node, 0) == 0;
                                 });
    auto kernelName = xkernel.get_name();

    if (verbosity >= 1)
      std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
                << "\n";
    device.register_xclbin(xclbin);

    if (verbosity >= 1)
      std::cout << "Getting hardware context.\n";
    xrt::hw_context context(device, xclbin.get_uuid());

    // Load instruction sequence
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());
    if (verbosity >= 1)
      std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    if (verbosity >= 1)
      std::cout << "Getting handle to kernel:" << kernelName << "\n";
    auto kernel = xrt::kernel(context, kernelName);

    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
    run.wait();
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    if (*(bufOut + i) != *(bufInA + i) * *(bufInB + i)) {
      if (verbosity >= 1) {
        std::cout << "Error in output " << *(bufOut + i)
                  << " != " << *(bufInA + i) << " * " << *(bufInB + i)
                  << std::endl;
      }
      errors++;
    } else if (verbosity >= 1) {
      std::cout << "Correct output " << *(bufOut + i)
                << " == " << *(bufInA + i) * *(bufInB + i) << std::endl;
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
