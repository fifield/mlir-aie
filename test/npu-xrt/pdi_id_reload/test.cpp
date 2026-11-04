// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Alternates between separately compiled AIE4 full ELFs on one hw_context and
// checks that each dispatch ran its own configuration.
//
//   ./test.exe [--order <letters>] [--timeout-ms <ms>] <elf>:<entry>:<k> ...
//
// The first ELF creates the hw_context, the others are add_config()ed to it.
// <order> is a string of letters indexing the ELF list (a = first, b = second,
// ...); step s dispatches kernel "<entry>:seq" of the selected ELF. Each
// design's core computes out[i] = in[i] + k. Every step gets fresh inputs and
// a poisoned output, and every element is checked exactly. Every wait is
// bounded, so a hung dispatch is reported instead of hanging the test.
//
//   ./test.exe --order abab a.elf:main_a:1 b.elf:main_b:2

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

constexpr size_t N = 64;
constexpr int POISON_BYTE = 0xEE;

struct Design {
  std::string elf, entry;
  int32_t k;
};

static void usage(const char *prog) {
  std::cerr << "usage: " << prog
            << " [--order <letters>] [--timeout-ms <ms>]"
               " <elf>:<entry>:<k> [<elf>:<entry>:<k> ...]\n";
}

// Parses "<elf>:<entry>:<k>"; returns false on malformed input.
static bool parseDesign(const std::string &s, Design &d) {
  size_t a = s.find(':'), b = s.rfind(':');
  if (a == std::string::npos || a == b || a == 0 || b == s.size() - 1)
    return false;
  char *end = nullptr;
  std::string k = s.substr(b + 1);
  errno = 0;
  long v = std::strtol(k.c_str(), &end, 10);
  if (k.empty() || *end != '\0' || errno == ERANGE || v < INT32_MIN ||
      v > INT32_MAX)
    return false;
  d = {s.substr(0, a), s.substr(a + 1, b - a - 1), static_cast<int32_t>(v)};
  return true;
}

int main(int argc, char **argv) {
  std::string order = "ab";
  int timeoutMs = 10000;
  std::vector<Design> designs;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "--order" || arg == "--timeout-ms") && i + 1 < argc) {
      std::string val = argv[++i];
      if (arg == "--order") {
        order = val;
      } else {
        timeoutMs = std::atoi(val.c_str());
        if (timeoutMs <= 0) {
          std::cerr << "bad --timeout-ms " << val << "\n";
          return 2;
        }
      }
    } else if (arg.rfind("--", 0) == 0) {
      usage(argv[0]);
      return 2;
    } else {
      Design d;
      if (!parseDesign(arg, d)) {
        std::cerr << "bad design spec '" << arg << "'\n";
        usage(argv[0]);
        return 2;
      }
      designs.push_back(d);
    }
  }
  if (designs.empty() || order.empty()) {
    usage(argv[0]);
    return 2;
  }
  for (char c : order)
    if (c < 'a' || size_t(c - 'a') >= designs.size()) {
      std::cerr << "order letter '" << c << "' has no design\n";
      return 2;
    }

  xrt::device device;
  std::vector<xrt::kernel> kernels;
  xrt::bo boIn, boOut;
  xrt::hw_context context;
  try {
    device = xrt::device(0);
    std::vector<xrt::elf> elfs;
    for (auto &d : designs)
      elfs.emplace_back(d.elf);
    context = xrt::hw_context(device, elfs[0]);
    for (size_t i = 1; i < elfs.size(); i++)
      context.add_config(elfs[i]);
    for (auto &d : designs)
      kernels.push_back(xrt::ext::kernel(context, d.entry + ":seq"));
    boIn = xrt::ext::bo{device, N * sizeof(int32_t)};
    boOut = xrt::ext::bo{device, N * sizeof(int32_t)};
  } catch (const std::exception &e) {
    std::cout << "setup failed: " << e.what() << "\nFAIL" << std::endl;
    return 1;
  }
  int32_t *in = nullptr, *out = nullptr;
  try {
    in = boIn.map<int32_t *>();
    out = boOut.map<int32_t *>();
  } catch (const std::exception &e) {
    std::cout << "setup failed: " << e.what() << "\nFAIL" << std::endl;
    return 1;
  }

  int exact = 0;
  for (size_t step = 0; step < order.size(); step++) {
    const Design &d = designs[order[step] - 'a'];
    for (size_t i = 0; i < N; i++)
      in[i] = int32_t(1000 * (step + 1) + i);
    std::memset(out, POISON_BYTE, N * sizeof(int32_t));

    int errors = 0;
    try {
      boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto run = kernels[order[step] - 'a'](boIn, boOut);
      if (run.wait2(std::chrono::milliseconds(timeoutMs)) ==
          std::cv_status::timeout) {
        std::cout << "step " << step << " (" << d.entry << "): TIMEOUT after "
                  << timeoutMs << " ms\nFAIL" << std::endl;
        return 1;
      }
      boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    } catch (const std::exception &e) {
      std::cout << "step " << step << " (" << d.entry
                << "): dispatch failed: " << e.what() << "\nFAIL" << std::endl;
      return 1;
    }
    for (size_t i = 0; i < N; i++) {
      int32_t want = in[i] + d.k;
      if (out[i] != want) {
        if (errors < 4)
          std::cout << "  " << d.entry << " out[" << i << "] = " << out[i]
                    << " (in + " << (out[i] - in[i]) << "), expected in + "
                    << d.k << std::endl;
        errors++;
      }
    }
    std::cout << "step " << step << " (" << d.entry << ", k=" << d.k
              << "): " << (errors ? "FAIL" : "ok") << std::endl;
    exact += errors == 0;
  }
  std::cout << exact << "/" << order.size() << " runs exact" << std::endl;
  bool ok = exact == int(order.size());
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  return ok ? 0 : 1;
}
