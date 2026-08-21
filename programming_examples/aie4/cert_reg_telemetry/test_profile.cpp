//===- test_profile.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cert_debug_buffer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {
constexpr size_t kOutputWords = 16;
constexpr size_t kDebugWords = 64;
constexpr uint32_t kUcIndex = 0;
constexpr uint32_t kOutputPoison = 0xdeadbeefu;
constexpr uint32_t kDebugPoison = 0xa5a5a5a5u;
constexpr std::array<uint32_t, 4> kIds{101, 102, 103, 104};

struct Stamp {
  uint32_t id;
  uint64_t ticks;
};

class Report {
public:
  void check(const std::string &label, bool condition,
             const std::string &detail) {
    std::cout << std::left << std::setw(32) << label << detail << "  "
              << (condition ? "PASS" : "FAIL") << '\n';
    if (!condition)
      ++errorCount;
  }
  int errors() const { return errorCount; }

private:
  int errorCount = 0;
};
} // namespace

int main(int argc, const char *argv[]) try {
  const std::string elf = argc > 1 ? argv[1] : "cert_reg_profile.elf";

  auto device = xrt::device(0);
  xrt::elf contextElf{elf};
  xrt::hw_context context{device, contextElf};
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  cert_example::DebugBuffer debug{context, kDebugWords, kDebugPoison};
  debug.configure(kUcIndex);

  xrt::bo outputBo = xrt::ext::bo{device, kOutputWords * sizeof(uint32_t)};
  uint32_t *output = outputBo.map<uint32_t *>();
  std::fill_n(output, kOutputWords, kOutputPoison);
  outputBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(outputBo);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cerr << "FAIL: dispatch timed out; the CERT job did not release its "
                 "DMA gate\n";
    return 1;
  }

  outputBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  debug.refresh();

  std::array<Stamp, 4> stamps{};
  for (size_t i = 0; i < stamps.size(); ++i) {
    const size_t word = i * 3;
    stamps[i] = {debug[word], (static_cast<uint64_t>(debug[word + 1]) << 32) |
                                  debug[word + 2]};
  }

  Report report;
  report.check("register arithmetic result", output[1] == 32,
               "obuf[1]=" + std::to_string(output[1]) + " (expected 32)");
  report.check("MMIO RMW result", output[0] == 9,
               "obuf[0]=" + std::to_string(output[0]) + " (expected 9)");
  const bool untouched = std::all_of(output + 2, output + kOutputWords,
                                     [](uint32_t value) { return value == 0; });
  report.check("remaining output", untouched, "obuf[2..15] are zero");

  for (size_t i = 0; i < stamps.size(); ++i)
    report.check("timestamp id " + std::to_string(i), stamps[i].id == kIds[i],
                 "got " + std::to_string(stamps[i].id) + ", expected " +
                     std::to_string(kIds[i]));

  const bool increasing = stamps[0].ticks < stamps[1].ticks &&
                          stamps[1].ticks < stamps[2].ticks &&
                          stamps[2].ticks < stamps[3].ticks;
  report.check("timestamp order", increasing,
               std::to_string(stamps[0].ticks) + " < " +
                   std::to_string(stamps[1].ticks) + " < " +
                   std::to_string(stamps[2].ticks) + " < " +
                   std::to_string(stamps[3].ticks));

  const uint64_t addTicks = stamps[1].ticks - stamps[0].ticks;
  const uint64_t baselineTicks = stamps[2].ticks - stamps[1].ticks;
  const uint64_t mmioTicks = stamps[3].ticks - stamps[2].ticks;
  report.check("measured arithmetic block", addTicks > baselineTicks,
               std::to_string(addTicks) + " ticks vs " +
                   std::to_string(baselineTicks) + " baseline");
  report.check("measured MMIO block", mmioTicks > baselineTicks,
               std::to_string(mmioTicks) + " ticks vs " +
                   std::to_string(baselineTicks) + " baseline");

  const size_t firstExtra = debug.firstChanged(12);
  report.check("no extra debug records", firstExtra == debug.size(),
               firstExtra == debug.size()
                   ? "debug words 12..63 remain poisoned"
                   : "first unexpected word is " + std::to_string(firstExtra));

  std::cout << "profile: 32 ADDs = " << addTicks
            << " ticks; timestamp baseline = " << baselineTicks
            << " ticks; 8 AXI-MM RMWs = " << mmioTicks << " ticks\n";

  if (report.errors()) {
    std::cout << "FAIL (" << report.errors() << " checks)\n";
    return 1;
  }
  std::cout << "PASS: SAVE_TIMESTAMPS profiled both register workloads\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
