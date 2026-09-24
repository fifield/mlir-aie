// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match the planted constants in reduce_aie4.py.
constexpr size_t OUT_COUNT = 12;
constexpr size_t OUT_BYTES = OUT_COUNT * sizeof(int32_t);

static float bits_to_f32(int32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

int main() {
  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"reduce_aie4.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_out = xrt::ext::bo{device, OUT_BYTES};
  memset(bo_out.map<char *>(), 0, OUT_BYTES);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t out[OUT_COUNT];
  memcpy(out, bo_out.map<char *>(), OUT_BYTES);

  int errors = 0;
  auto check_i32 = [&](int idx, int32_t want, const char *what) {
    if (out[idx] != want) {
      std::cout << "MISMATCH [" << idx << "] " << what << ": got=" << out[idx]
                 << " want=" << want << std::endl;
      errors++;
    } else {
      std::cout << "OK [" << idx << "] " << what << " = " << out[idx]
                 << std::endl;
    }
  };
  auto check_f32 = [&](int idx, float want, const char *what) {
    float got = bits_to_f32(out[idx]);
    // Exact comparison distinguishes sequential from native rounding.
    if (got != want) {
      std::cout << "MISMATCH [" << idx << "] " << what << ": got=" << got
                 << " want=" << want << std::endl;
      errors++;
    } else {
      std::cout << "OK [" << idx << "] " << what << " = " << got << std::endl;
    }
  };

  check_i32(0, -5, "source int add, unseeded");
  check_i32(1, -100, "source signed int min");
  check_i32(2, 99, "source signed int max");
  check_i32(3, -2147483644, "source int add, positive overflow wraps");
  check_i32(4, 2147483643, "source int add, negative overflow wraps");
  check_i32(5, 995, "source int add, ordinary seed");
  check_f32(6, 16777216.0f, "source ordered f32 add, sequential rounding");
  check_f32(7, -11.125f, "source f32 minimum");
  check_f32(8, 12.25f, "source f32 maximum");

  check_i32(9, 2147483647, "native VADDR positive saturation");
  check_i32(10, -2147483648, "native VADDR negative saturation");
  check_f32(11, 16777248.0f, "native VADDR.f single-rounded sum");

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " / " << OUT_COUNT << " checks failed." << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
