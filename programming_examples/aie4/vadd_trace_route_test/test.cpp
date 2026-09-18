//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

static constexpr int NCOLS = 3;
static constexpr int COL_ELEMS = 4096;
static constexpr int TOTAL_ELEMS = NCOLS * COL_ELEMS;
static constexpr int TRACE_BYTES = 8192;

int main(int argc, const char *argv[]) {
  std::string elfFile = (argc > 1) ? argv[1] : "vadd_trace.elf";
  printf("ELF: %s\n", elfFile.c_str());
  fflush(stdout);

  std::vector<int32_t> inA(TOTAL_ELEMS), inB(TOTAL_ELEMS), outC(TOTAL_ELEMS, 0);
  for (int i = 0; i < TOTAL_ELEMS; i++) {
    inA[i] = i;
    inB[i] = i + 1;
  }

  auto device = xrt::device(0);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  size_t dataSize = TOTAL_ELEMS * sizeof(int32_t);
  xrt::bo boInA = xrt::ext::bo{device, dataSize};
  xrt::bo boInB = xrt::ext::bo{device, dataSize};
  xrt::bo boOutC = xrt::ext::bo{device, dataSize};
  xrt::bo boTrace = xrt::ext::bo{device, TRACE_BYTES};

  memcpy(boInA.map<char *>(), inA.data(), dataSize);
  memcpy(boInB.map<char *>(), inB.data(), dataSize);
  memset(boOutC.map<char *>(), 0, dataSize);
  memset(boTrace.map<char *>(), 0, TRACE_BYTES);

  boInA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boInB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boOutC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boTrace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto start = std::chrono::high_resolution_clock::now();
  kernel(boInA, boInB, boOutC, boTrace).wait2();
  auto stop = std::chrono::high_resolution_clock::now();

  boOutC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  boTrace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  const uint32_t *trace = boTrace.map<uint32_t *>();
  bool traceEmpty = true;
  std::ofstream traceFile("trace.bin");
  for (size_t i = 0; i < TRACE_BYTES / sizeof(uint32_t); i++) {
    traceFile << std::setfill('0') << std::setw(8) << std::hex << trace[i]
              << std::endl;
    traceEmpty &= trace[i] == 0;
  }
  traceFile.close();

  auto *output = boOutC.map<int32_t *>();
  int errors = 0;
  for (int i = 0; i < TOTAL_ELEMS; i++) {
    int32_t expected = i + (i + 1);
    if (output[i] != expected) {
      if (errors < 10)
        printf("error @ %d: got %d, expected %d\n", i, output[i], expected);
      errors++;
    }
  }

  float us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                 .count();
  printf("elapsed: %.0f us\n", us);

  if (traceEmpty) {
    fprintf(stderr, "FAIL: trace buffer is empty\n");
    return 2;
  }
  if (errors != 0) {
    printf("FAIL (%d/%d errors)\n", errors, TOTAL_ELEMS);
    return 1;
  }

  printf("PASS (data correct, trace non-empty)\n");
  return 0;
}
