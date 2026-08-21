//===- test_snapshot.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cert_debug_buffer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
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

std::string hex32(uint32_t value) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
  return os.str();
}

class Report {
public:
  void word(const std::string &label, uint32_t got, uint32_t expected) {
    std::cout << std::left << std::setw(36) << label << "got " << hex32(got)
              << "  expected " << hex32(expected);
    if (got == expected)
      std::cout << "  PASS\n";
    else {
      std::cout << "  FAIL\n";
      ++errorCount;
    }
  }

  void check(const std::string &label, bool condition,
             const std::string &detail) {
    std::cout << std::left << std::setw(36) << label << detail << "  "
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
  const std::string elf = argc > 1 ? argv[1] : "cert_reg_snapshot.elf";

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

  Report report;
  report.word("obuf[0] seed", output[0], 0x100);
  report.word("obuf[1] pointer-walk result", output[1], 0x123);
  report.word("obuf[2] second result", output[2], 0x223);
  for (size_t i = 3; i < kOutputWords; ++i)
    report.word("obuf[" + std::to_string(i) + "] untouched", output[i], 0);

  std::cout << "--- SAVE_REGISTER records: {unq_id, value} ---\n";
  report.word("record 0 id", debug[0], 201);
  report.word("record 0 value", debug[1], 0x123);
  report.word("record 1 id", debug[2], 202);
  report.word("record 1 value", debug[3], 0x223);

  const size_t firstExtra = debug.firstChanged(4);
  report.check("no extra debug records", firstExtra == debug.size(),
               firstExtra == debug.size()
                   ? "debug words 4..63 remain poisoned"
                   : "first unexpected word is " + std::to_string(firstExtra));

  // The same computed words arrived through two independent paths: normal
  // tile DMA and the firmware's uC debug-record DMA.
  report.check("record/DMA cross-check",
               debug[1] == output[1] && debug[3] == output[2],
               "saved values match obuf[1] and obuf[2]");

  if (report.errors()) {
    std::cout << "FAIL (" << report.errors() << " checks)\n";
    return 1;
  }
  std::cout << "PASS: register computation and SAVE_REGISTER telemetry agree\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
