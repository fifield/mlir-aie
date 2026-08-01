// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// XRT host harness for the cascade-reduced GEMV example.
//   W : per-col [AW_lo blob | W_hi blob], with A's per-row slice
//        embedded at the front of each AW_lo per-core chunk.
//   C : 192 INT32       (output: column-cascade-reduced partials)
//
// WEIGHT LAYOUT (column-major, AIE4 ISA spec, two-stream split w/ A inlined)
// --------------------------------------------------------------------------
// The kernel writes the AIE4 64x64 INT8 staging matrix one v128int8
// chunk at a time, 32 chunks per K-tile.  Per the ISA spec the staging
// matrix is loaded column-major and each v128int8 pair-packs two
// adjacent N columns of 64 K-rows each.
//
// To double weight bandwidth, the per-core 64-KB slab is split into
// two 32-KB halves streamed via two parallel shim MM2S channels:
//   * AW_lo carries N cols 0..31 of every K-tile (plus a 1 KB A slice)
//   * W_hi  carries N cols 32..63 of every K-tile
// Within each stream's per-core "half" buffer:
//   16 K-tiles, each 16 v128 (= 2048 bytes), in K-tile order.
// Within one v128 slot (128 bytes) two adjacent N columns are packed:
//   * bytes 0..63   = col_a (k = 0..63)
//   * bytes 64..127 = col_b (k = 0..63)
//
// The compute tile has only 2 input DMA channels, so the per-row A slice
// (1 KB) is prepended to each core's AW_lo buffer.  The kernel reads A
// from offset 0 and W_lo from offset 1024 inside AW_lo.
//
// Whole-tensor host layout (per col blob):
//   AW_lo blob = 4 rows * (1 KB A_slice + 32 KB W_lo_half) = 135168 B
//   W_hi  blob = 4 rows * 32 KB                            = 131072 B
//   per col    = AW_lo blob || W_hi blob                   = 266240 B
//   tensor     = 3 cols                                    = 798720 B

#include <algorithm>
#include <chrono>
#include <cstdint>
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

// N_COLS (1..3) is set at compile time to match the --cols passed to
// gemv_placed.py.  The weight-stream count (--streams) is internal to the
// device design (shim->memtile partitioning) and does NOT change the host
// data layout, so the host harness does not need to know it.
#ifndef N_COLS
#define N_COLS 3
#endif
static_assert(N_COLS >= 1 && N_COLS <= 3, "N_COLS must be 1, 2, or 3");
constexpr int N_ROWS         = 4;        // compute rows per column
constexpr int TILE_K         = 64;
constexpr int TILE_N         = 64;
// K_TOTAL (contraction length) is set at compile time to match the --ktotal
// passed to gemv_placed.py (and -DK_TOTAL passed to kernel.cc). Must be a
// multiple of N_ROWS*TILE_K (=256).
#ifndef K_TOTAL
#define K_TOTAL 4096
#endif
static_assert(K_TOTAL % (N_ROWS * TILE_K) == 0,
              "K_TOTAL must be a multiple of N_ROWS*TILE_K (=256)");
constexpr int N_TOTAL        = N_COLS * TILE_N;          // 192
constexpr int K_PER_ROW      = K_TOTAL / N_ROWS;         // 1024
constexpr int K_TILES_PER_ROW = K_PER_ROW / TILE_K;      // 16
constexpr int A_SLICE_BYTES        = K_PER_ROW;                       // 1024
constexpr int W_HALF_BYTES         = K_PER_ROW * TILE_N / 2;          // 32768
constexpr int AW_LO_PER_CORE_BYTES = A_SLICE_BYTES + W_HALF_BYTES;    // 33792
constexpr int W_HI_PER_CORE_BYTES  = W_HALF_BYTES;                    // 32768
constexpr int AW_LO_COL_BYTES      = N_ROWS * AW_LO_PER_CORE_BYTES;   // 135168
constexpr int W_HI_COL_BYTES       = N_ROWS * W_HI_PER_CORE_BYTES;    // 131072
constexpr int W_COL_BYTES          = AW_LO_COL_BYTES + W_HI_COL_BYTES;// 266240

constexpr size_t W_BYTES = static_cast<size_t>(N_COLS) * W_COL_BYTES; // 798720 B
constexpr size_t C_BYTES = N_TOTAL * sizeof(int32_t);                 // 768 B

// Convert a logical weight (k, n) into a device-buffer offset.
// Layout per col: [AW_lo blob | W_hi blob].
//   AW_lo per row = [A_slice 1 KB | W_lo half-slab 32 KB] = 33792 B
//   W_hi  per row = W_hi half-slab 32 KB                  = 32768 B
static inline size_t pack_weight_byte(int k, int n) {
  const int c          = n / TILE_N;
  const int n_in_tile  = n % TILE_N;
  const int r          = k / K_PER_ROW;
  const int k_local    = k % K_PER_ROW;
  const int k_tile     = k_local / TILE_K;
  const int k_in_tile  = k_local % TILE_K;

  const int half       = (n_in_tile < 32) ? 0 : 1;
  const int col_in_h   = n_in_tile - 32 * half;       // 0..31
  const int slot       = col_in_h / 2;                // 0..15
  const int col_in_v128 = col_in_h % 2;               // 0 or 1

  const size_t col_blob_off = static_cast<size_t>(c) * W_COL_BYTES;
  const size_t tile_off     = static_cast<size_t>(k_tile) * (16 * 128);  // 2 KB
  const size_t slot_off     = static_cast<size_t>(slot) * 128;
  const size_t in_v128_off  = static_cast<size_t>(col_in_v128) * 64 + k_in_tile;

  if (half == 0) {
    // AW_lo blob: per-row chunk starts with A_slice then W_lo half-slab.
    const size_t aw_row_off = static_cast<size_t>(r) * AW_LO_PER_CORE_BYTES
                              + A_SLICE_BYTES;
    return col_blob_off + aw_row_off + tile_off + slot_off + in_v128_off;
  } else {
    // W_hi blob lives immediately after the AW_lo blob in the col blob.
    const size_t hi_blob_off = AW_LO_COL_BYTES;
    const size_t hi_row_off  = static_cast<size_t>(r) * W_HI_PER_CORE_BYTES;
    return col_blob_off + hi_blob_off + hi_row_off + tile_off + slot_off + in_v128_off;
  }
}

// Offset for the A slice byte at logical position k (which row, which k_in_row).
// A_slice for row r lives at the very start of that row's AW_lo per-core chunk,
// replicated across all N_COLS columns (each col carries its own copy).
static inline size_t pack_a_byte(int c, int k) {
  const int r        = k / K_PER_ROW;
  const int k_in_row = k % K_PER_ROW;
  const size_t col_blob_off = static_cast<size_t>(c) * W_COL_BYTES;
  const size_t aw_row_off   = static_cast<size_t>(r) * AW_LO_PER_CORE_BYTES;
  return col_blob_off + aw_row_off + k_in_row;
}

// Usage: test.exe [elf_file] [iters_per_call] [trace_elems]
//   elf_file       default "gemv_4096_192_cascade_bench.elf"
//   iters_per_call MUST equal the `-i` value the ELF was built with (the
//                  number of GEMV iterations baked into its runtime
//                  sequence): 1 for the default ELF, BENCH_ITERS (100) for
//                  the bench ELF.  This is NOT a runtime knob -- the ELF
//                  always runs exactly its baked-in count per launch.  The
//                  measured per-launch wall time is divided by this value to
//                  report per-iteration throughput, so passing a number that
//                  does NOT match the build just rescales the reported
//                  numbers by a constant and produces meaningless throughput
//                  (e.g. passing 10000 against the 100-iter bench ELF makes
//                  every metric look 100x too fast).  Use `make run_bench`,
//                  which always passes the matching value.
//   trace_elems    if elf name contains "trace", allocate a trace buffer
//                  of this many INT32 elements (default 8192) and write
//                  it to trace.bin after the run.
int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1]
                                         : "gemv_4096_192_cascade_bench.elf";
  const int iters_per_call  = (argc > 2) ? std::atoi(argv[2]) : 1;
  const int trace_elems     = (argc > 3) ? std::atoi(argv[3]) : 8192;
  const bool enable_trace   = elfFile.find("trace") != std::string::npos;

  std::cout << "ELF: " << elfFile
            << "  iters/call=" << iters_per_call
            << "  trace=" << (enable_trace ? "on" : "off") << "\n";
  // ------------------------------------------------------------------
  // Inputs.  Bounded values so INT32 accumulator stays in range:
  //   A bounded [-4, 3]  (max |A| = 4)
  //   W bounded [-2, 2]  (max |W| = 2)
  //   K = 4096           => max |C[n]| <= 4 * 2 * 4096 = 32768
  // ------------------------------------------------------------------
  // Pseudo-random inputs so every (k, n) byte position is exercised
  // distinctly.  Linear-congruential, deterministic across runs.
  auto lcg = [](uint32_t &s) -> uint32_t {
    s = s * 1103515245u + 12345u;
    return s;
  };
  uint32_t s = 0xC0FFEEu;

  std::vector<int8_t> A_logical(K_TOTAL);
  for (int k = 0; k < K_TOTAL; k++)
    A_logical[k] = static_cast<int8_t>((lcg(s) & 0x7) - 4);   // [-4, 3]

  std::vector<int8_t> W_logical(K_TOTAL * N_TOTAL);
  for (int i = 0; i < K_TOTAL * N_TOTAL; i++)
    W_logical[i] = static_cast<int8_t>((lcg(s) % 5) - 2);     // [-2, 2]

  // Host reference: C[n] = sum_k A[k] * W[k, n]
  std::vector<int32_t> C_ref(N_TOTAL, 0);
  for (int n = 0; n < N_TOTAL; n++) {
    int32_t acc = 0;
    for (int k = 0; k < K_TOTAL; k++)
      acc += static_cast<int32_t>(A_logical[k]) *
             static_cast<int32_t>(W_logical[k * N_TOTAL + n]);
    C_ref[n] = acc;
  }

  // ------------------------------------------------------------------
  // Set up XRT.
  // ------------------------------------------------------------------
  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_w = xrt::ext::bo{device, W_BYTES};
  xrt::bo bo_c = xrt::ext::bo{device, C_BYTES};

  int8_t  *buf_w = bo_w.map<int8_t  *>();
  int32_t *buf_c = bo_c.map<int32_t *>();

  // Pack A (replicated per col) and W into the device W tensor.
  std::memset(buf_w, 0, W_BYTES);
  for (int c = 0; c < N_COLS; c++)
    for (int k = 0; k < K_TOTAL; k++)
      buf_w[pack_a_byte(c, k)] = A_logical[k];
  for (int k = 0; k < K_TOTAL; k++)
    for (int n = 0; n < N_TOTAL; n++)
      buf_w[pack_weight_byte(k, n)] = W_logical[k * N_TOTAL + n];

  for (int i = 0; i < N_TOTAL; i++)
    buf_c[i] = static_cast<int32_t>(0xdeadbeef);

  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Optional trace buffer (present only when running the trace elf).
  const size_t trace_bytes = enable_trace
      ? static_cast<size_t>(trace_elems) * sizeof(int32_t)
      : 0;
  xrt::bo bo_trace;
  char *buf_trace = nullptr;
  if (enable_trace) {
    bo_trace = xrt::ext::bo{device, trace_bytes};
    buf_trace = bo_trace.map<char *>();
    std::memset(buf_trace, 0, trace_bytes);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  auto launch = [&]() {
    if (enable_trace) {
      kernel(bo_w, bo_c, bo_trace).wait2();
    } else {
      kernel(bo_w, bo_c).wait2();
    }
  };

  // ------------------------------------------------------------------
  // Correctness pass: one invocation, then verify.
  // ------------------------------------------------------------------
  launch();
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (enable_trace) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::ofstream out("trace.bin", std::ios::binary);
    out.write(buf_trace, trace_bytes);
    std::cout << "trace written to trace.bin (" << trace_bytes
              << " bytes)\n";
  }

  int errors = 0;
  for (int n = 0; n < N_TOTAL; n++) {
    if (buf_c[n] != C_ref[n]) {
      if (errors < 16) {
        std::cout << "  mismatch n=" << std::setw(3) << n
                  << " got=" << std::setw(8) << buf_c[n]
                  << " ref=" << std::setw(8) << C_ref[n] << "\n";
      }
      errors++;
    }
  }
  if (errors != 0) {
    std::cout << "FAIL (" << errors << " / " << N_TOTAL << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << N_TOTAL << " outputs match host reference)"
            << std::endl;

  // ------------------------------------------------------------------
  // Benchmark.  Each kernel launch internally runs `iters_per_call`
  // GEMV iterations (the ITERS the ELF was built with), so we divide
  // the measured wall time by that batch factor to get the per-GEMV
  // latency.  At iters_per_call >> 1 this amortizes the XRT host
  // launch overhead and exposes steady-state device throughput.
  //
  // When tracing is enabled we skip the benchmark loop (the trace
  // buffer would overflow / get overwritten across launches).
  // ------------------------------------------------------------------
  if (enable_trace) return 0;

  constexpr int WARMUP = 5;
  constexpr int OUTER  = 200;

  for (int i = 0; i < WARMUP; i++) launch();

  std::vector<double> us_per_call;
  us_per_call.reserve(OUTER);
  double us_launch_min = 1e30;  // raw per-launch wall time (NOT divided)
  for (int i = 0; i < OUTER; i++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    launch();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_outer = std::chrono::duration<double, std::micro>(t1 - t0).count();
    us_launch_min = std::min(us_launch_min, us_outer);
    us_per_call.push_back(us_outer / iters_per_call);
  }

  std::sort(us_per_call.begin(), us_per_call.end());
  const double us_min    = us_per_call.front();
  const double us_max    = us_per_call.back();
  const double us_median = us_per_call[OUTER / 2];
  double us_mean = 0.0;
  for (double t : us_per_call) us_mean += t;
  us_mean /= OUTER;

  // GEMV math: M=1, K=4096, N=192 INT8.  MACs per call = K * N.
  // Each MAC is 1 mul + 1 add = 2 ops, so OPs = 2 * K * N.
  const double macs_per_call = double(K_TOTAL) * double(N_TOTAL);
  const double ops_per_call  = 2.0 * macs_per_call;

  // Weight-load bandwidth (the BW-dominated path).  Each call streams
  // the full W tensor (with embedded A slices) from DDR through the
  // shim DMAs.
  const double bytes_per_call = double(W_BYTES);

  auto rate = [&](double us) {
    return std::pair<double, double>{
        ops_per_call / us / 1e3,            // GOPS  (ops / us / 1e3)
        bytes_per_call / us / 1e3,          // GB/s  (bytes / us / 1e3)
    };
  };

  auto report = [&](const char *label, double us) {
    auto [gops, gbs] = rate(us);
    std::cout << "  " << std::left << std::setw(8) << label
              << std::right << std::fixed
              << std::setprecision(2) << std::setw(9) << us << " us/iter"
              << std::setprecision(2) << std::setw(9) << gops << " GOPS"
              << std::setprecision(2) << std::setw(9) << gbs  << " GB/s"
              << std::endl;
  };

  std::cout << "\nBenchmark over " << OUTER << " launches x "
            << iters_per_call << " iters/launch (" << WARMUP
            << " warmup), MACs/call = "
            << std::fixed << std::setprecision(0) << macs_per_call
            << ", bytes loaded/call = " << bytes_per_call << ":\n";
  report("min:",    us_min);
  report("median:", us_median);
  report("mean:",   us_mean);
  report("max:",    us_max);

  // ------------------------------------------------------------------
  // Plausibility guard.  The ELF runs a FIXED number of GEMVs per launch
  // (its build-time `-i` value); iters_per_call must match it.  If it does
  // not, the per-iteration numbers above are just the (constant) per-launch
  // time rescaled by a wrong divisor.  Each GEMV must stream W_BYTES from
  // DDR through the shim DMAs, so the implied weight-load bandwidth cannot
  // exceed the device's physical DDR/shim ceiling.  npu3 peaks well under
  // ~100 GB/s on this path (see shim_dma_memtile_bw); anything above this
  // generous ceiling means iters_per_call does not match the ELF.
  constexpr double SANE_PEAK_GBS = 300.0;
  const double best_gbs = bytes_per_call / us_min / 1e3;
  std::cout << "\n  (raw per-launch wall time: " << std::fixed
            << std::setprecision(2) << us_launch_min
            << " us; reported per-iter = this / iters_per_call="
            << iters_per_call << ")\n";
  if (best_gbs > SANE_PEAK_GBS) {
    std::cerr << "\nWARNING: implied " << std::fixed << std::setprecision(0)
              << best_gbs << " GB/s exceeds the physical ceiling (~"
              << SANE_PEAK_GBS << " GB/s).\n"
              << "         iters_per_call=" << iters_per_call
              << " almost certainly does NOT match the ELF's baked-in `-i`"
              << " count.\n"
              << "         The ELF runs a fixed number of GEMVs per launch;"
              << " pass that exact number\n"
              << "         (1 for the default ELF, BENCH_ITERS for the bench"
              << " ELF) or use `make run_bench`.\n"
              << "         Reported per-iteration throughput is invalid.\n";
    return 2;
  }

  return 0;
}
