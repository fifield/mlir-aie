//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <cstdlib>
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
  // Program arguments parsing
  cxxopts::Options options("Passthrough DMAs Test");
  cxxopts::ParseResult vm;

  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())(
      "length,l", "the length of the transfer in int32_t",
      cxxopts::value<int>()->default_value("4096"))(
      "full-elf", "path to full design ELF (AIE4/npu3 flow)",
      cxxopts::value<std::string>());

  try {
    vm = options.parse(argc, argv);

    if (vm.count("help")) {
      std::cout << options.help() << std::endl;
      return 1;
    }

    // Check required options: either (xclbin + instr) or (full-elf), plus kernel.
    bool have_xclbin_flow = vm.count("xclbin") && vm.count("instr");
    bool have_full_elf_flow = vm.count("full-elf") > 0;
    if (!vm.count("kernel") || (!have_xclbin_flow && !have_full_elf_flow)) {
      std::cerr << "Error: Required options missing\n\n";
      std::cerr << "Usage:\n" << options.help() << std::endl;
      return 1;
    }
  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  int verbosity = vm["verbosity"].as<int>();
  bool use_full_elf = vm.count("full-elf") > 0;

  int N = vm["length"].as<int>();
  if ((N % 1024)) {
    std::cerr << "Length must be a multiple of 1024." << std::endl;
    return 1;
  }

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string Node = vm["kernel"].as<std::string>();

  // Allocate buffers up front; both flows share these.
  xrt::bo bo_inA = xrt::ext::bo{device, N * sizeof(int32_t)};
  xrt::bo bo_inB = xrt::ext::bo{device, N * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, N * sizeof(int32_t)};

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  int32_t *bufInA = bo_inA.map<int32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < N; i++)
    srcVecA.push_back(i + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;

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
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());
    if (verbosity >= 1)
      std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

    if (verbosity >= 1)
      std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>()
                << std::endl;
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

    if (verbosity >= 1)
      std::cout << "Kernel opcode: " << Node << std::endl;

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
      std::cout << "Getting hardware context." << std::endl;
    xrt::hw_context context(device, xclbin.get_uuid());

    // get a kernel handle
    if (verbosity >= 1)
      std::cout << "Getting handle to kernel:" << kernelName << std::endl;
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

  for (uint32_t i = 0; i < N; i++) {
    uint32_t ref = (i + 1);
    if (*(bufOut + i) != ref) {
      errors++;
    }
  }

  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
