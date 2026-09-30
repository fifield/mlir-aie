// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// XRT host harness for the cascade-reduced GEMV example.
//   W : per-col [AW_lo blob | W_hi blob], with A's per-row slice
//        embedded at the front of each AW_lo per-core chunk.
//   C : 192 INT32       (output: column-cascade-reduced partials)
//
// GEMV host packing, CPU reference, benchmark, and trace capture.
// See README.md for the device layout and build-time configuration.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

// N_COLS (1..3) is derived by the Makefile as ceil(N/64) from the --n
// passed to gemv_placed.py.  The weight-stream count (--streams) is
// internal to the device design (shim->memtile partitioning) and does NOT
// change the host data layout, so the host harness does not need to know it.
#ifndef N_COLS
#define N_COLS 3
#endif
static_assert(N_COLS >= 1 && N_COLS <= 3, "N_COLS must be 1, 2, or 3");
// N_ROWS (1..4) is set at compile time to match the --rows passed to
// gemv_placed.py.
#ifndef N_ROWS
#define N_ROWS 4
#endif
static_assert(N_ROWS >= 1 && N_ROWS <= 4, "N_ROWS must be 1..4");
constexpr int TILE_K = 64;
constexpr int TILE_N = 64;
// K_TOTAL (contraction length) is set at compile time to match the --ktotal
// passed to gemv_placed.py (and -DK_TOTAL passed to kernel.cc). Must be a
// multiple of N_ROWS*TILE_K.
#ifndef K_TOTAL
#define K_TOTAL 4096
#endif
static_assert(K_TOTAL % (N_ROWS * TILE_K) == 0,
              "K_TOTAL must be a multiple of N_ROWS*TILE_K");
constexpr int N_TOTAL = N_COLS * TILE_N;
// N_VALID is the TRUE requested N (--n); N_TOTAL is padded up to a multiple
// of 64. Defaults to N_TOTAL (no padding).
#ifndef N_VALID
#define N_VALID N_TOTAL
#endif
static_assert(N_VALID > N_TOTAL - TILE_N && N_VALID <= N_TOTAL,
              "N_VALID must be in (N_TOTAL - 64, N_TOTAL]; Makefile N/COLS "
              "arithmetic is inconsistent");
constexpr int K_PER_ROW = K_TOTAL / N_ROWS;
constexpr int K_TILES_PER_ROW = K_PER_ROW / TILE_K;

// K-chunk size, matching kernel.cc's K_CHUNK_TILES_RAW/K_chunk_tiles
// resolution: -DK_CHUNK_TILES_RAW=0 (default) means "whole row."  This only
// affects how AW_lo bytes are interleaved (A_chunk/W_lo_chunk pairs instead
// of one A_slice followed by one W_lo half-slab) -- W_hi and the logical
// (k, n) -> value mapping are unaffected.
#ifndef K_CHUNK_TILES_RAW
#define K_CHUNK_TILES_RAW 0
#endif
constexpr int K_CHUNK_TILES =
    (K_CHUNK_TILES_RAW == 0) ? K_TILES_PER_ROW : K_CHUNK_TILES_RAW;
static_assert(K_TILES_PER_ROW % K_CHUNK_TILES == 0,
              "K_TILES_PER_ROW must be a multiple of K_CHUNK_TILES (KCHUNK)");
constexpr int NUM_CHUNKS = K_TILES_PER_ROW / K_CHUNK_TILES;

constexpr int CHUNK_A_BYTES = K_CHUNK_TILES * TILE_K;
constexpr int CHUNK_WLO_BYTES = K_CHUNK_TILES * TILE_K * TILE_N / 2;
constexpr int AW_LO_CHUNK_BYTES = CHUNK_A_BYTES + CHUNK_WLO_BYTES;
constexpr int W_HI_CHUNK_BYTES = CHUNK_WLO_BYTES;
constexpr int AW_LO_PER_CORE_BYTES = AW_LO_CHUNK_BYTES * NUM_CHUNKS;
constexpr int W_HI_PER_CORE_BYTES = W_HI_CHUNK_BYTES * NUM_CHUNKS;
constexpr int AW_LO_COL_BYTES = N_ROWS * AW_LO_PER_CORE_BYTES;
constexpr int W_HI_COL_BYTES = N_ROWS * W_HI_PER_CORE_BYTES;
constexpr int W_COL_BYTES = AW_LO_COL_BYTES + W_HI_COL_BYTES;

constexpr size_t W_BYTES =
    static_cast<size_t>(N_COLS) * W_COL_BYTES;
constexpr size_t C_BYTES = N_TOTAL * sizeof(int32_t);

// Convert a logical weight (k, n) into a device-buffer offset.
// Layout per col: [AW_lo blob | W_hi blob].
//   AW_lo per row = NUM_CHUNKS * [A_chunk | W_lo_chunk], interleaved chunk
//                   by chunk (not one A_slice followed by one W_lo slab),
//                   so each chunk is a self-contained DMA-able unit -- see
//                   kernel.cc's gemv_chunk_accumulate.
//   W_hi  per row = NUM_CHUNKS * W_hi_chunk (unaffected by chunking, since
//                   W_hi never carried A).
static inline size_t pack_weight_byte(int k, int n) {
  const int c = n / TILE_N;
  const int n_in_tile = n % TILE_N;
  const int r = k / K_PER_ROW;
  const int k_local = k % K_PER_ROW;
  const int k_tile = k_local / TILE_K; // 0..K_TILES_PER_ROW-1
  const int k_in_tile = k_local % TILE_K;
  const int chunk = k_tile / K_CHUNK_TILES;
  const int tile_in_ck = k_tile % K_CHUNK_TILES; // 0..K_CHUNK_TILES-1

  const int half = (n_in_tile < 32) ? 0 : 1;
  const int col_in_h = n_in_tile - 32 * half; // 0..31
  const int slot = col_in_h / 2;              // 0..15
  const int col_in_v128 = col_in_h % 2;       // 0 or 1

  const size_t col_blob_off = static_cast<size_t>(c) * W_COL_BYTES;
  const size_t tile_off = static_cast<size_t>(tile_in_ck) * (16 * 128); // 2 KB
  const size_t slot_off = static_cast<size_t>(slot) * 128;
  const size_t in_v128_off = static_cast<size_t>(col_in_v128) * 64 + k_in_tile;

  if (half == 0) {
    // AW_lo blob: row's chunk `chunk` starts with A_chunk then W_lo_chunk.
    const size_t aw_row_off = static_cast<size_t>(r) * AW_LO_PER_CORE_BYTES +
                              static_cast<size_t>(chunk) * AW_LO_CHUNK_BYTES +
                              CHUNK_A_BYTES;
    return col_blob_off + aw_row_off + tile_off + slot_off + in_v128_off;
  } else {
    // W_hi blob lives immediately after the AW_lo blob in the col blob.
    const size_t hi_blob_off = AW_LO_COL_BYTES;
    const size_t hi_row_off = static_cast<size_t>(r) * W_HI_PER_CORE_BYTES +
                              static_cast<size_t>(chunk) * W_HI_CHUNK_BYTES;
    return col_blob_off + hi_blob_off + hi_row_off + tile_off + slot_off +
           in_v128_off;
  }
}

// Offset for the A slice byte at logical position k (which row, which
// chunk, which tile-in-chunk).  A_chunk for (row r, chunk `chunk`) lives at
// the very start of that chunk's AW_lo sub-buffer, replicated across all
// N_COLS columns (each col carries its own copy).
static inline size_t pack_a_byte(int c, int k) {
  const int r = k / K_PER_ROW;
  const int k_in_row = k % K_PER_ROW;
  const int k_tile = k_in_row / TILE_K;
  const int k_in_tile = k_in_row % TILE_K;
  const int chunk = k_tile / K_CHUNK_TILES;
  const int tile_in_ck = k_tile % K_CHUNK_TILES;
  const size_t col_blob_off = static_cast<size_t>(c) * W_COL_BYTES;
  const size_t aw_row_off = static_cast<size_t>(r) * AW_LO_PER_CORE_BYTES +
                            static_cast<size_t>(chunk) * AW_LO_CHUNK_BYTES;
  const size_t in_chunk_off =
      static_cast<size_t>(tile_in_ck) * TILE_K + k_in_tile;
  return col_blob_off + aw_row_off + in_chunk_off;
}

// Usage: test.exe [--full-range] [elf_file] [iters_per_call] [trace_elems]
//   --full-range   use [-128, 127] inputs and pinned boundary cases.
//   elf_file       default "gemv_4096_192_cascade_bench.elf"
//   iters_per_call must match the ELF's build-time iteration count. It is a
//                  reporting divisor, not a runtime control; use the Makefile
//                  run targets to supply the matching value.
//   trace_elems    if elf name contains "trace", allocate a trace buffer
//                  of this many INT32 elements (default 8192) and write
//                  it to trace.bin after the run.
int main(int argc, const char *argv[]) {
  bool full_range = false;
  std::vector<const char *> args;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--full-range")
      full_range = true;
    else
      args.push_back(argv[i]);
  }
  const std::string elfFile =
      (args.size() > 0) ? args[0] : "gemv_4096_192_cascade_bench.elf";
  const int iters_per_call = (args.size() > 1) ? std::atoi(args[1]) : 1;
  const int trace_elems = (args.size() > 2) ? std::atoi(args[2]) : 8192;
  const bool enable_trace = elfFile.find("trace") != std::string::npos;

  std::cout << "ELF: " << elfFile << "  iters/call=" << iters_per_call
            << "  trace=" << (enable_trace ? "on" : "off")
            << "  inputs=" << (full_range ? "full-range int8" : "bounded")
            << "\n";
  // ------------------------------------------------------------------
  // Inputs.  Bounded values so INT32 accumulator stays in range:
  //   A bounded [-4, 3]  (max |A| = 4)
  //   W bounded [-2, 2]  (max |W| = 2)
  //   max |C[n]| <= 4 * 2 * K_TOTAL
  // With --full-range, A and W are drawn from [-128, 127]:
  //   max |C[n]| <= 128 * 128 * K_TOTAL = 2^26 at K_TOTAL = 4096.
  // Pin boundary cases below to guarantee coverage.
  // ------------------------------------------------------------------
  // Pseudo-random inputs so every (k, n) byte position is exercised
  // distinctly.  Linear-congruential, deterministic across runs.
  auto lcg = [](uint32_t &s) -> uint32_t {
    s = s * 1103515245u + 12345u;
    return s;
  };
  uint32_t s = 0xC0FFEEu;

  static_assert(int64_t(128) * 128 * K_TOTAL <= INT32_MAX,
                "--full-range reference could overflow INT32");
  auto full = [&]() { return static_cast<int8_t>(lcg(s) >> 24); };
  const int8_t extremes[] = {-128, 127, -1, 0};

  std::vector<int8_t> A_logical(K_TOTAL);
  for (int k = 0; k < K_TOTAL; k++)
    A_logical[k] = full_range
                       ? full()
                       : static_cast<int8_t>((lcg(s) & 0x7) - 4); // [-4, 3]

  std::vector<int8_t> W_logical(K_TOTAL * N_TOTAL);
  for (int i = 0; i < K_TOTAL * N_TOTAL; i++)
    W_logical[i] =
        full_range ? full() : static_cast<int8_t>((lcg(s) % 5) - 2); // [-2, 2]

  if (full_range) {
    // Exercise all pairs from {-128, 127, -1, 0}.
    for (int k = 0; k < 4; k++) {
      A_logical[k] = extremes[k];
      for (int n = 0; n < 4; n++)
        W_logical[k * N_TOTAL + n] = extremes[n];
    }
    // Prevent cancellation: columns 4 and 5 use full-scale weights to make
    // every product nonnegative and nonpositive, respectively.
    for (int k = 0; k < K_TOTAL; k++) {
      const bool neg = A_logical[k] < 0;
      W_logical[k * N_TOTAL + 4] = neg ? int8_t(-128) : int8_t(127);
      W_logical[k * N_TOTAL + 5] = neg ? int8_t(127) : int8_t(-128);
    }
  }

  // Host reference: C[n] = sum_k A[k] * W[k, n]
  std::vector<int32_t> C_ref(N_TOTAL, 0);
  for (int n = 0; n < N_TOTAL; n++) {
    int32_t acc = 0;
    for (int k = 0; k < K_TOTAL; k++)
      acc += static_cast<int32_t>(A_logical[k]) *
             static_cast<int32_t>(W_logical[k * N_TOTAL + n]);
    C_ref[n] = acc;
  }
  if (full_range) {
    int64_t max_abs = 0;
    for (int n = 0; n < N_VALID; n++)
      max_abs = std::max(max_abs, std::abs(static_cast<int64_t>(C_ref[n])));
    std::cout << "max |C_ref| = " << max_abs << "\n";
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

  int8_t *buf_w = bo_w.map<int8_t *>();
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
  const size_t trace_bytes =
      enable_trace ? static_cast<size_t>(trace_elems) * sizeof(int32_t) : 0;
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
    // parse.py expects text: one zero-padded 8-digit hex word per line
    // (same as test_utils::write_out_trace), not raw binary.
    std::ofstream out("trace.bin");
    const uint32_t *words = reinterpret_cast<const uint32_t *>(buf_trace);
    for (size_t i = 0; i < trace_bytes / sizeof(uint32_t); i++)
      out << std::setfill('0') << std::setw(8) << std::hex << words[i]
          << std::endl;
    std::cout << "trace written to trace.bin (" << trace_bytes << " bytes)\n";
  }

  // Only the first N_VALID columns are the user-requested output; any
  // trailing columns up to the padded N_TOTAL are device-side padding
  // and are intentionally not checked.
  int errors = 0;
  for (int n = 0; n < N_VALID; n++) {
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
    std::cout << "FAIL (" << errors << " / " << N_VALID << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << N_VALID << " outputs match host reference)"
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
  if (enable_trace)
    return 0;

  constexpr int WARMUP = 5;
  constexpr int OUTER = 200;

  for (int i = 0; i < WARMUP; i++)
    launch();

  std::vector<double> us_per_call;
  us_per_call.reserve(OUTER);
  double us_launch_min = 1e30; // raw per-launch wall time (NOT divided)
  for (int i = 0; i < OUTER; i++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    launch();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_outer =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    us_launch_min = std::min(us_launch_min, us_outer);
    us_per_call.push_back(us_outer / iters_per_call);
  }

  std::sort(us_per_call.begin(), us_per_call.end());
  const double us_min = us_per_call.front();
  const double us_max = us_per_call.back();
  const double us_median = us_per_call[OUTER / 2];
  double us_mean = 0.0;
  for (double t : us_per_call)
    us_mean += t;
  us_mean /= OUTER;

  // GEMV math: MACs per call = K_TOTAL * N_VALID (the
  // useful work the caller asked for; padded columns aren't real output).
  // Each MAC is 1 mul + 1 add = 2 ops, so OPs = 2 * K * N_VALID.
  const double macs_per_call = double(K_TOTAL) * double(N_VALID);
  const double ops_per_call = 2.0 * macs_per_call;

  // Weight-load bandwidth (the BW-dominated path).  Each call streams
  // the full W tensor (with embedded A slices) from DDR through the
  // shim DMAs.
  const double bytes_per_call = double(W_BYTES);

  auto rate = [&](double us) {
    return std::pair<double, double>{
        ops_per_call / us / 1e3,   // GOPS  (ops / us / 1e3)
        bytes_per_call / us / 1e3, // GB/s  (bytes / us / 1e3)
    };
  };

  auto report = [&](const char *label, double us) {
    auto [gops, gbs] = rate(us);
    std::cout << "  " << std::left << std::setw(8) << label << std::right
              << std::fixed << std::setprecision(2) << std::setw(9) << us
              << " us/iter" << std::setprecision(2) << std::setw(9) << gops
              << " GOPS" << std::setprecision(2) << std::setw(9) << gbs
              << " GB/s" << std::endl;
  };

  std::cout << "\nBenchmark over " << OUTER << " launches x " << iters_per_call
            << " iters/launch (" << WARMUP
            << " warmup), MACs/call = " << std::fixed << std::setprecision(0)
            << macs_per_call << ", bytes loaded/call = " << bytes_per_call
            << ":\n";
  report("min:", us_min);
  report("median:", us_median);
  report("mean:", us_mean);
  report("max:", us_max);

  // ------------------------------------------------------------------
  // Reject implausible bandwidth as a check for an incorrect reporting
  // divisor. This heuristic does not prove that iters_per_call matches the ELF.
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
