// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Host side for the AIE4 control-packet demo. Deliberately identical in shape
// to the other npu3 full-ELF examples: the control-packet buffer is created,
// filled from the ELF's `.ctrlpkt-<id>` section, patched and synced entirely
// inside XRT, so the host only ever sees the output buffer.

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr size_t N = 16;
constexpr size_t BUF_SIZE = N * sizeof(uint32_t);

// Must match the `data` array of the aiex.control_packet op in the design.
static const uint32_t kExpected[N] = {4660,  22136, 4661,  22137, 4662,  22138,
                                      4663,  22139, 4664,  22140, 4665,  22141,
                                      4666,  22142, 4667,  22143};

int main(int argc, const char *argv[]) {
  const char *elfName = argc > 1 ? argv[1] : "ctrl_pkt_demo.elf";

  auto device = xrt::device(0);
  xrt::elf ctxElf{elfName};
  xrt::hw_context context(device, ctxElf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo boOut = xrt::ext::bo{device, BUF_SIZE};
  auto *out = boOut.map<uint32_t *>();
  for (size_t i = 0; i < N; i++)
    out[i] = 0xdeadbeef;
  boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, boOut);
  run.start();
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "FAIL: run timed out\n";
    return 1;
  }
  boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (size_t i = 0; i < N; i++) {
    if (out[i] == kExpected[i])
      continue;
    if (errors < 8)
      std::cout << "  [" << i << "] got 0x" << std::hex << out[i]
                << " expected 0x" << kExpected[i] << std::dec << "\n";
    errors++;
  }
  std::cout << (errors ? "FAIL" : "PASS") << " (" << errors << " errors)\n";
  return errors != 0;
}
