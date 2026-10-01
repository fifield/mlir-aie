// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Three configurations run in turn on the same three cores. The first and
// third copy the head of each core's weight block to their output; the second
// copies the first one's output. Each configuration's routes differ, so a
// connection left over from an earlier configuration would fork a stream into
// a port that a later one drains.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr size_t VEC = 192;
constexpr size_t PART = 64;
constexpr size_t BLOCK = PART * VEC;
constexpr size_t MAT = VEC * VEC;
constexpr size_t W = 3 * MAT;

static int check(const char *name, const std::vector<uint16_t> &got,
                 const std::vector<uint16_t> &want) {
  int errors = 0;
  for (size_t i = 0; i < got.size(); i++) {
    if (got[i] != want[i]) {
      if (errors < 8)
        std::cout << name << "[" << i << "] = 0x" << std::hex << got[i]
                  << ", expected 0x" << want[i] << std::dec << std::endl;
      errors++;
    }
  }
  return errors;
}

int main() {
  // Normal bf16 values; the blocks the first and third configurations read
  // hold different values.
  std::vector<uint16_t> w(W);
  for (size_t i = 0; i < W; i++)
    w[i] = uint16_t(0x4000 | (i % 4095));

  std::vector<uint16_t> first(VEC), second(VEC), third(VEC);
  for (size_t c = 0; c < 3; c++)
    for (size_t j = 0; j < PART; j++) {
      first[c * PART + j] = w[c * BLOCK + j];
      third[c * PART + j] = w[2 * MAT + c * BLOCK + j];
    }
  second = first;

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"switches.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  auto make_bo = [&](const std::vector<uint16_t> &init) {
    xrt::bo bo = xrt::ext::bo{device, init.size() * sizeof(uint16_t)};
    memcpy(bo.map<char *>(), init.data(), init.size() * sizeof(uint16_t));
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
  };
  std::vector<uint16_t> zeros(VEC, 0);
  xrt::bo bo_x = make_bo(zeros);
  xrt::bo bo_w = make_bo(w);
  xrt::bo bo_first = make_bo(zeros);
  xrt::bo bo_unused = make_bo(zeros);
  xrt::bo bo_second = make_bo(zeros);
  xrt::bo bo_third = make_bo(zeros);

  auto run = kernel(bo_x, bo_w, bo_first, bo_unused, bo_second, bo_third);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "fail: run did not complete within 10s" << std::endl;
    return 1;
  }

  auto read = [&](xrt::bo &bo) {
    std::vector<uint16_t> v(VEC);
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(v.data(), bo.map<char *>(), VEC * sizeof(uint16_t));
    return v;
  };
  int errors = check("first", read(bo_first), first) +
               check("second", read(bo_second), second) +
               check("third", read(bo_third), third);
  if (errors) {
    std::cout << "fail: " << errors << " mismatches" << std::endl;
    return 1;
  }
  std::cout << "PASS!" << std::endl;
  return 0;
}
