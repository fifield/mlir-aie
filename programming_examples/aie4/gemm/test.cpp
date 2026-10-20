// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verify row-major C = A * W^T and measure launch time.
// Usage: test.exe <elf> M N K out_bits(8|16|32) shift iters_per_launch [cols]
//                 [dtype=i8|u8|i8u8|u8i8|i4|fp8|bf16] [timing] [trace=<bytes>]
// Integer types are checked bit-exactly: u8 is uint8 x uint8, i8u8 is int8 A x
// uint8 W, u8i8 the reverse, and i4 is signed int4, two elements per byte
// with the low nibble first. u8 narrows to uint8 output, the others to int8.
// fp8 (E4M3) and bf16 inputs produce float output: out_bits 32 is fp32, 16
// bf16, 8 fp8. It is checked against a double reference within a tolerance
// (MMAC fp32 accumulation order and output rounding are not bit-modelled).

#include <algorithm>
#include <chrono>
#include <cmath>
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

#include "../cpu_ref/fp_types.h"

enum class DType { I8, U8, I8U8, U8I8, I4, FP8, BF16 };

int main(int argc, const char *argv[]) {
  if (argc < 8) {
    fprintf(stderr, "usage: %s elf M N K out_bits shift iters\n", argv[0]);
    return 2;
  }
  const std::string elf = argv[1];
  const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
  const int out_bits = atoi(argv[5]);
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
  DType dtype = DType::I8;
  for (int i = 9; i < argc; i++) {
    std::string a = argv[i];
    if (a == "dtype=fp8")
      dtype = DType::FP8;
    else if (a == "dtype=bf16")
      dtype = DType::BF16;
    else if (a == "dtype=u8")
      dtype = DType::U8;
    else if (a == "dtype=i8u8")
      dtype = DType::I8U8;
    else if (a == "dtype=u8i8")
      dtype = DType::U8I8;
    else if (a == "dtype=i4")
      dtype = DType::I4;
    else if (a == "dtype=i8")
      dtype = DType::I8;
    else if (a.rfind("dtype=", 0) == 0) {
      fprintf(stderr, "unknown %s\n", a.c_str());
      return 2;
    } else if (a == "timing")
      timing = true;
    else if (a.rfind("trace=", 0) == 0)
      trace_bytes = std::stoul(a.substr(6));
  }
  const double clk_ghz =
      getenv("AIE_CLK_GHZ") ? atof(getenv("AIE_CLK_GHZ")) : 1.2;

  const bool is_float = dtype == DType::FP8 || dtype == DType::BF16;
  const int elem_bits = dtype == DType::BF16 ? 16 : dtype == DType::I4 ? 4 : 8;
  const bool a_uns = dtype == DType::U8 || dtype == DType::U8I8;
  const bool w_uns = dtype == DType::U8 || dtype == DType::I8U8;
  const bool out_uns = dtype == DType::U8; // unsigned result narrows to uint8
  if (is_float ? (out_bits != 8 && out_bits != 16 && out_bits != 32)
               : (out_bits != 8 && out_bits != 32)) {
    fprintf(stderr, "bad out_bits %d for this dtype\n", out_bits);
    return 2;
  }
  const bool out_i8 = out_bits == 8 && !is_float; // shifted int8 / uint8
  const size_t out_bytes = out_bits / 8;
  // Peak MMAC rate of one AIE4 core, in MAC/cycle: 64x64 int8/fp8, 32x64
  // bf16, 128x64 int4.
  const double core_mac_per_cycle = dtype == DType::BF16 ? 2048
                                    : dtype == DType::I4 ? 8192
                                                         : 4096;

  // A and W as raw bytes; Af/Wf (float) and Ai/Wi (integer) hold the decoded
  // element values.
  std::vector<uint8_t> A((size_t)M * K * elem_bits / 8),
      W((size_t)N * K * elem_bits / 8);
  std::vector<float> Af, Wf;
  std::vector<int32_t> Ai, Wi;
  std::mt19937 gen(42);
  if (is_float) {
    Af.resize((size_t)M * K);
    Wf.resize((size_t)N * K);
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    auto fill = [&](std::vector<uint8_t> &raw, std::vector<float> &val) {
      for (size_t i = 0; i < val.size(); i++) {
        float x = dis(gen);
        if (dtype == DType::FP8) {
          // The MAC input flushes fp8 subnormals to zero; avoid them.
          if (std::fabs(x) < 0.0156251f)
            x = 0.0f;
          uint8_t b = float_to_fp8_e4m3(x);
          raw[i] = b;
          val[i] = fp8_e4m3_to_float(b);
        } else {
          uint16_t b = float_to_bf16(x);
          memcpy(&raw[i * 2], &b, 2);
          val[i] = bf16_to_float(b);
        }
      }
    };
    fill(A, Af);
    fill(W, Wf);
  } else {
    Ai.resize((size_t)M * K);
    Wi.resize((size_t)N * K);
    auto fill = [&](std::vector<uint8_t> &raw, std::vector<int32_t> &val,
                    bool uns) {
      const int lo = dtype == DType::I4 ? -8 : uns ? 0 : -128;
      const int hi = dtype == DType::I4 ? 7 : uns ? 255 : 127;
      std::uniform_int_distribution<int> dis(lo, hi);
      for (size_t i = 0; i < val.size(); i++) {
        int x = dis(gen);
        val[i] = x;
        if (dtype == DType::I4) // two per byte, low nibble first
          raw[i / 2] |= (uint8_t)((x & 0xf) << (4 * (i & 1)));
        else
          raw[i] = (uint8_t)x;
      }
    };
    fill(A, Ai, a_uns);
    fill(W, Wi, w_uns);
  }

  // CPU reference (multithreaded over rows): exact int32 for int8, double
  // for float. `mag` is sum|a*w|, the scale of the float rounding error.
  std::vector<int32_t> ref((size_t)M * N);
  std::vector<double> refd, mag;
  if (is_float) {
    refd.resize((size_t)M * N);
    mag.resize((size_t)M * N);
  }
  {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++)
      th.emplace_back([&, t] {
        for (int m = t; m < M; m += nt)
          for (int n = 0; n < N; n++) {
            if (is_float) {
              const float *a = &Af[(size_t)m * K], *w = &Wf[(size_t)n * K];
              double s = 0, sa = 0;
              for (int k = 0; k < K; k++) {
                double p = (double)a[k] * (double)w[k];
                s += p;
                sa += std::fabs(p);
              }
              refd[(size_t)m * N + n] = s;
              mag[(size_t)m * N + n] = sa;
              continue;
            }
            const int32_t *a = &Ai[(size_t)m * K], *w = &Wi[(size_t)n * K];
            int32_t s = 0;
            for (int k = 0; k < K; k++)
              s += a[k] * w[k];
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

  const size_t c_bytes = (size_t)M * N * out_bytes;
  xrt::bo bo_a = xrt::ext::bo{device, A.size()};
  xrt::bo bo_w = xrt::ext::bo{device, W.size()};
  xrt::bo bo_c = xrt::ext::bo{device, c_bytes};
  memcpy(bo_a.map<uint8_t *>(), A.data(), A.size());
  memcpy(bo_w.map<uint8_t *>(), W.data(), W.size());
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
      const size_t ob = out_bytes;
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
    const size_t ob = out_bytes;
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
  // The float path also tracks the worst error as a fraction of the allowance.
  double worst = 0;
  auto verify = [&]() {
    size_t errs = 0;
    if (is_float) {
      // fp32 accumulation error grows with sum|a*w|; allow 2^-16 of it.
      // Narrow outputs add output rounding: half an ulp of the result
      // relative to the value (worst case just above a power of two); fp8 E4M3
      // also has an absolute floor from its subnormals.
      const double half_ulp = out_bits == 8    ? 1.0 / 16
                              : out_bits == 16 ? 1.0 / 256
                                               : 0;
      const double floor_abs = out_bits == 8 ? 1.0 / 1024 : 0;
      const void *raw = bo_c.map<void *>();
      for (size_t i = 0; i < (size_t)M * N; i++) {
        float got = out_bits == 32 ? ((const float *)raw)[i]
                    : out_bits == 16
                        ? bf16_to_float(((const uint16_t *)raw)[i])
                        : fp8_e4m3_to_float(((const uint8_t *)raw)[i]);
        double err = std::fabs((double)got - refd[i]);
        double allow = 1.0 / 65536 * mag[i] +
                       half_ulp * 1.05 * std::fabs(refd[i]) + floor_abs;
        double rel = err / allow;
        worst = std::max(worst, std::isnan(rel) ? 1e9 : rel);
        if (!(err <= allow) && errs++ < 10)
          printf("  mismatch C[%zu][%zu]: got %g expected %g\n", i / N, i % N,
                 got, refd[i]);
      }
      return errs;
    }
    for (size_t i = 0; i < (size_t)M * N; i++) {
      int32_t want = ref[i], got;
      if (out_i8) {
        int64_t v = want;
        if (shift > 0)
          v = (v + (1ll << (shift - 1))) >> shift;
        if (out_uns) {
          want = (int32_t)std::min<int64_t>(255, std::max<int64_t>(0, v));
          got = bo_c.map<uint8_t *>()[i];
        } else {
          want = (int32_t)std::min<int64_t>(127, std::max<int64_t>(-128, v));
          got = bo_c.map<int8_t *>()[i];
        }
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
  double peak_tops = 2 * core_mac_per_cycle * 4 * cols * clk_ghz * 1e-3;
  const char *in_name = dtype == DType::FP8    ? "fp8"
                        : dtype == DType::BF16 ? "bf16"
                        : dtype == DType::U8   ? "uint8"
                        : dtype == DType::I8U8 ? "int8 x uint8"
                        : dtype == DType::U8I8 ? "uint8 x int8"
                        : dtype == DType::I4   ? "int4"
                                               : "int8";
  printf("GEMM %dx%dx%d %s -> %s, %d GEMM(s)/launch\n", M, N, K, in_name,
         out_i8     ? (out_uns ? "uint8" : "int8")
         : is_float ? (out_bits == 32   ? "fp32"
                       : out_bits == 16 ? "bf16"
                                        : "fp8")
                    : "int32",
         iters);
  printf("  time/GEMM: median %.1f us, best %.1f us (incl. launch overhead "
         "/ %d)\n",
         med, best, iters);
  printf("  throughput: median %.2f TOPS, best %.2f TOPS  (peak %.1f TOPS @ "
         "%.2f GHz -> %.1f%% / %.1f%%)\n",
         2 * macs / med * 1e-6, 2 * macs / best * 1e-6, peak_tops, clk_ghz,
         100 * 2 * macs / med * 1e-6 / peak_tops,
         100 * 2 * macs / best * 1e-6 / peak_tops);

  if (is_float)
    printf("  worst error: %.3g of the allowed tolerance\n", worst);
  if (errs) {
    printf("FAIL (%zu mismatches over 2 checks of %zu outputs)\n", errs,
           (size_t)M * N);
    return 1;
  }
  printf("PASS (all %zu outputs match, first and last timed launch)\n",
         (size_t)M * N);
  return 0;
}
