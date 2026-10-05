// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verify row-major C = A * W^T and measure launch time.
// Usage: test.exe <elf> M N K out_bits(8|32) shift iters_per_launch [cols]
//                 [timing] [trace=<bytes>]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Peak int8 MMAC rate of one AIE4 core, in MAC/cycle.
constexpr double CORE_MAC_PER_CYCLE = 4096;

int main(int argc, const char *argv[]) {
  if (argc < 8) {
    fprintf(stderr, "usage: %s elf M N K out_bits shift iters\n", argv[0]);
    return 2;
  }
  const std::string elf = argv[1];
  const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
  const bool out_i8 = atoi(argv[5]) == 8;
  const int shift = atoi(argv[6]);
  const int iters = atoi(argv[7]);
  const int cols = argc > 8 ? atoi(argv[8]) : 3; // 4 cores per column
  if (shift < 0 || shift > 59) {
    fprintf(stderr, "shift must be in [0, 59]\n");
    return 2;
  }
  // Optional modes must match the ELF: timing reads tile cycle stamps;
  // trace=<bytes> adds a trace buffer and writes its hex words to trace.txt.
  bool timing = false;
  size_t trace_bytes = 0;
  for (int i = 9; i < argc; i++) {
    std::string a = argv[i];
    if (a == "timing")
      timing = true;
    else if (a.rfind("trace=", 0) == 0)
      trace_bytes = std::stoul(a.substr(6));
  }
  const double clk_ghz =
      getenv("AIE_CLK_GHZ") ? atof(getenv("AIE_CLK_GHZ")) : 1.2;

  std::vector<int8_t> A((size_t)M * K), W((size_t)N * K);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-128, 127);
  for (auto &x : A)
    x = (int8_t)dis(gen);
  for (auto &x : W)
    x = (int8_t)dis(gen);

  // CPU reference (multithreaded over rows).
  std::vector<int32_t> ref((size_t)M * N);
  {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++)
      th.emplace_back([&, t] {
        for (int m = t; m < M; m += nt)
          for (int n = 0; n < N; n++) {
            const int8_t *a = &A[(size_t)m * K], *w = &W[(size_t)n * K];
            int32_t s = 0;
            for (int k = 0; k < K; k++)
              s += (int32_t)a[k] * (int32_t)w[k];
            ref[(size_t)m * N + n] = s;
          }
      });
    for (auto &x : th)
      x.join();
  }

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elf};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  const size_t c_bytes = (size_t)M * N * (out_i8 ? 1 : 4);
  xrt::bo bo_a = xrt::ext::bo{device, A.size()};
  xrt::bo bo_w = xrt::ext::bo{device, W.size()};
  xrt::bo bo_c = xrt::ext::bo{device, c_bytes};
  memcpy(bo_a.map<int8_t *>(), A.data(), A.size());
  memcpy(bo_w.map<int8_t *>(), W.data(), W.size());
  memset(bo_c.map<uint8_t *>(), 0xa5, c_bytes);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::bo bo_trace;
  if (trace_bytes) {
    bo_trace = xrt::ext::bo{device, trace_bytes};
    memset(bo_trace.map<uint8_t *>(), 0, trace_bytes);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  auto launch = [&]() {
    return trace_bytes ? kernel(bo_a, bo_w, bo_c, bo_trace)
                       : kernel(bo_a, bo_w, bo_c);
  };

  bool timed_out = false;
  // Keep the run alive for timeout diagnostics; _exit avoids aborting it.
  auto run = launch();
  {
    auto state = run.wait(std::chrono::milliseconds(10000));
    if (state != ERT_CMD_STATE_COMPLETED) {
      bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const uint8_t *raw = bo_c.map<uint8_t *>();
      const size_t ob = out_i8 ? 1 : 4;
      printf("TIMEOUT (state %d). 64x64 C tiles written (row-blocks x "
             "col-tiles):\n",
             (int)state);
      for (int mt = 0; mt < M / 64; mt++) {
        printf("  ");
        for (int nt = 0; nt < N / 64; nt++) {
          size_t off = ((size_t)mt * 64 * N + (size_t)nt * 64) * ob;
          printf("%c", raw[off] == 0xa5 && raw[off + 1] == 0xa5 ? '.' : '#');
        }
        printf("\n");
      }
      timed_out = true;
    }
  }
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  if (trace_bytes) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint32_t *w = bo_trace.map<uint32_t *>();
    FILE *f = fopen("trace.txt", "w");
    size_t n = trace_bytes / 4, used = 0;
    for (size_t i = 0; i < n; i++) {
      fprintf(f, "%08x\n", w[i]);
      if (w[i])
        used = i + 1;
    }
    fclose(f);
    printf("trace: %zu of %zu words used -> trace.txt%s\n", used, n,
           used == n ? " (BUFFER FULL: increase TRACE_SIZE)" : "");
  }

  if (timing) {
    const int MB = M / 256, NT = N / (cols * 64);
    const uint8_t *raw = bo_c.map<uint8_t *>();
    const size_t ob = out_i8 ? 1 : 4;
    auto stamp = [&](int c, int r, int mb, int t) {
      size_t row = (size_t)mb * 256 + r * 64, col = (size_t)(c * NT + t) * 64;
      uint32_t v;
      memcpy(&v, raw + (row * N + col) * ob, 4);
      return v;
    };
    printf("per-core tile periods in cycles (ideal ~%d):\n", K + 64);
    for (int c = 0; c < cols; c++)
      for (int r = 0; r < 4; r++) {
        std::vector<double> in, bnd;
        for (int mb = 0; mb < MB; mb++)
          for (int t = 0; t < NT; t++) {
            if (mb == 0 && t == 0)
              continue;
            uint32_t prev =
                t ? stamp(c, r, mb, t - 1) : stamp(c, r, mb - 1, NT - 1);
            (t ? in : bnd)
                .push_back((double)(uint32_t)(stamp(c, r, mb, t) - prev));
          }
        std::sort(in.begin(), in.end());
        std::sort(bnd.begin(), bnd.end());
        uint32_t span = stamp(c, r, MB - 1, NT - 1) - stamp(c, r, 0, 0);
        printf("  core(%d,%d): in-block min %.0f med %.0f max %.0f | "
               "M-block boundary med %.0f max %.0f | span %u cyc = %.1f us\n",
               c, r + 2, in.front(), in[in.size() / 2], in.back(),
               bnd.empty() ? 0 : bnd[bnd.size() / 2],
               bnd.empty() ? 0 : bnd.back(), span, span / (clk_ghz * 1e3));
      }
  }

  // Call only after syncing C from the device.
  auto verify = [&]() {
    size_t errs = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
      int32_t want = ref[i], got;
      if (out_i8) {
        int64_t v = want;
        if (shift > 0)
          v = (v + (1ll << (shift - 1))) >> shift;
        want = (int32_t)std::min<int64_t>(127, std::max<int64_t>(-128, v));
        got = bo_c.map<int8_t *>()[i];
      } else {
        got = bo_c.map<int32_t *>()[i];
      }
      if (got != want && errs++ < 10)
        printf("  mismatch C[%zu][%zu]: got %d expected %d\n", i / N, i % N,
               got, want);
    }
    return errs;
  };
  size_t errs = verify();

  if (timed_out) {
    printf("%s after TIMEOUT (%zu / %zu mismatches)\n",
           errs ? "FAIL" : "data OK", errs, (size_t)M * N);
    fflush(stdout);
    _exit(3);
  }

  // Warm up the NPU clock before measuring launches.
  for (int i = 0; i < 10; i++)
    launch().wait2();
  std::vector<double> us;
  constexpr int timed_launches = 30;
  for (int i = 0; i < timed_launches; i++) {
    if (i == timed_launches - 1) {
      // Exclude poisoning from timing and prevent stale output passing
      // verification.
      memset(bo_c.map<uint8_t *>(), 0xa5, c_bytes);
      bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    launch().wait2();
    auto t1 = std::chrono::high_resolution_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                 iters);
  }
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  errs += verify();
  std::sort(us.begin(), us.end());
  double med = us[us.size() / 2], best = us.front();
  double macs = (double)M * N * K;
  double peak_tops = 2 * CORE_MAC_PER_CYCLE * 4 * cols * clk_ghz * 1e-3;
  printf("GEMM %dx%dx%d int8 -> %s, %d GEMM(s)/launch\n", M, N, K,
         out_i8 ? "int8" : "int32", iters);
  printf("  time/GEMM: median %.1f us, best %.1f us (incl. launch overhead "
         "/ %d)\n",
         med, best, iters);
  printf("  throughput: median %.2f TOPS, best %.2f TOPS  (peak %.1f TOPS @ "
         "%.2f GHz -> %.1f%% / %.1f%%)\n",
         2 * macs / med * 1e-6, 2 * macs / best * 1e-6, peak_tops, clk_ghz,
         100 * 2 * macs / med * 1e-6 / peak_tops,
         100 * 2 * macs / best * 1e-6 / peak_tops);

  if (errs) {
    printf("FAIL (%zu mismatches over 2 checks of %zu outputs)\n", errs,
           (size_t)M * N);
    return 1;
  }
  printf("PASS (all %zu outputs match, first and last timed launch)\n",
         (size_t)M * N);
  return 0;
}
