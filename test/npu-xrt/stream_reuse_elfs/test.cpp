// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Interleaves two separately compiled AIE4 full ELFs on one hardware context.
// Design A (design_a.mlir) copies 1024 x i32 through memtile S2MM 0 / MM2S 0;
// design B (design_b.mlir) takes the same stream through memtile S2MM 2 /
// MM2S 2 and permutes its rows. Both receive the stream on memtile South:0.
// Run r uses A when r is even and B when r is odd, each with its own input and
// a poisoned output; every element of every run is checked.
//
//   --mode seq       start a run, wait for it, start the next (no sleep)
//   --mode runlist   one xrt::runlist holding all N runs of both kernels
//   --mode startall  start all N runs, then wait for each
//
// Every wait is bounded: a hung dispatch is reported as a timeout.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

constexpr int COUNT = 1024;
constexpr int ROW = 32; // B permutes rows of 32 elements
constexpr size_t BYTES = COUNT * sizeof(int32_t);
constexpr int32_t POISON = (int32_t)0xdeadbeef;

static int32_t input_value(int run, int i) { return ((run + 1) << 16) | i; }
static bool is_b(int run) { return run % 2; }

// Design A: out[i] = in[i]. Design B: out row 4 * i + k = in row i + 8 * k.
static int source_index(int run, int idx) {
  if (!is_b(run))
    return idx;
  int row = idx / ROW, col = idx % ROW;
  int i = row / 4, k = row % 4;
  return (i + 8 * k) * ROW + col;
}

static bool check_run(int run, const int32_t *out) {
  int first = -1, bad = 0;
  for (int i = 0; i < COUNT; i++)
    if (out[i] != input_value(run, source_index(run, i))) {
      if (first < 0)
        first = i;
      bad++;
    }
  if (!bad)
    return true;
  std::cout << "run " << run << " (" << (is_b(run) ? "B" : "A") << "): FAIL "
            << bad << "/" << COUNT << " elements wrong, first at index "
            << first << " (got 0x" << std::hex << out[first] << ", expected 0x"
            << input_value(run, source_index(run, first)) << std::dec << ")";
  if (out[first] == POISON)
    std::cout << "; output never written";
  std::cout << "\n";
  return false;
}

[[noreturn]] static void timeout_exit(const std::string &what) {
  // Skip XRT destructors: they can block or throw on a hung context.
  std::cout << "TIMEOUT: " << what << "\nFAIL\n" << std::flush;
  _exit(2);
}

int main(int argc, char **argv) {
  std::string elf_a_name = "design_a.elf", elf_b_name = "design_b.elf";
  std::string mode = "seq";
  int n = 32, timeout_ms = 5000;
  auto parse_int = [](const std::string &name, const char *s, long lo,
                      long hi) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end || v < lo || v > hi) {
      std::cerr << "bad value '" << s << "' for " << name << " (expected " << lo
                << ".." << hi << ")\n";
      std::exit(2);
    }
    return (int)v;
  };
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    bool has_val = i + 1 < argc;
    if (a == "--elf-a" && has_val)
      elf_a_name = argv[++i];
    else if (a == "--elf-b" && has_val)
      elf_b_name = argv[++i];
    else if (a == "--mode" && has_val)
      mode = argv[++i];
    // n is capped so ((run + 1) << 16) stays positive in int32_t.
    else if (a == "--n" && has_val)
      n = parse_int(a, argv[++i], 1, 32767);
    // 0 would mean "block forever" in XRT.
    else if (a == "--timeout-ms" && has_val)
      timeout_ms = parse_int(a, argv[++i], 1, 3600000);
    else {
      std::cerr << "usage: " << argv[0]
                << " [--elf-a f] [--elf-b f] [--mode seq|runlist|startall]"
                   " [--n N] [--timeout-ms T]\n";
      return 2;
    }
  }
  if (mode != "seq" && mode != "runlist" && mode != "startall") {
    std::cerr << "unknown mode " << mode << "\n";
    return 2;
  }
  std::chrono::milliseconds timeout(timeout_ms);

  xrt::device device;
  xrt::hw_context context;
  xrt::kernel kernel_a, kernel_b;
  // Both configurations on one hardware context.
  std::string step = "open device";
  try {
    device = xrt::device(0u);
    step = "load " + elf_a_name;
    xrt::elf elf_a{elf_a_name};
    step = "load " + elf_b_name;
    xrt::elf elf_b{elf_b_name};
    step = "create hw_context from " + elf_a_name;
    context = xrt::hw_context(device, elf_a);
    step = "add_config(" + elf_b_name + ")";
    context.add_config(elf_b);
    step = "get kernel stream_a:sequence";
    kernel_a = xrt::ext::kernel(context, "stream_a:sequence");
    step = "get kernel stream_b:sequence";
    kernel_b = xrt::ext::kernel(context, "stream_b:sequence");
  } catch (const std::exception &e) {
    std::cout << "setup failed at " << step << ": " << e.what() << "\nFAIL\n";
    return 3;
  }

  // Distinct input per run; output prefilled with a poison value.
  std::vector<xrt::bo> bo_in, bo_out;
  for (int r = 0; r < n; r++) {
    bo_in.emplace_back(xrt::ext::bo{device, BYTES});
    bo_out.emplace_back(xrt::ext::bo{device, BYTES});
    auto *in = bo_in[r].map<int32_t *>();
    auto *out = bo_out[r].map<int32_t *>();
    for (int i = 0; i < COUNT; i++) {
      in[i] = input_value(r, i);
      out[i] = POISON;
    }
    bo_in[r].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out[r].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  auto make_run = [&](int r) {
    xrt::run run(is_b(r) ? kernel_b : kernel_a);
    run.set_arg(0, bo_in[r]);
    run.set_arg(1, bo_out[r]);
    return run;
  };
  auto run_name = [&](int r) {
    return "run " + std::to_string(r) + " (" + (is_b(r) ? "B" : "A") + ") of " +
           std::to_string(n);
  };

  try {
    if (mode == "seq") {
      for (int r = 0; r < n; r++) {
        auto run = make_run(r);
        run.start();
        if (run.wait2(timeout) == std::cv_status::timeout)
          timeout_exit(run_name(r) + " did not complete in " +
                       std::to_string(timeout_ms) + " ms (mode seq)");
      }
    } else if (mode == "runlist") {
      std::vector<xrt::run> runs;
      for (int r = 0; r < n; r++)
        runs.push_back(make_run(r));
      xrt::runlist list(context);
      for (auto &run : runs)
        list.add(run);
      list.execute();
      if (list.wait(timeout) == std::cv_status::timeout)
        timeout_exit("runlist of " + std::to_string(n) +
                     " runs did not complete in " + std::to_string(timeout_ms) +
                     " ms");
    } else { // startall
      std::vector<xrt::run> runs;
      for (int r = 0; r < n; r++)
        runs.push_back(make_run(r));
      for (auto &run : runs)
        run.start();
      for (int r = 0; r < n; r++)
        if (runs[r].wait2(timeout) == std::cv_status::timeout)
          timeout_exit(run_name(r) + " did not complete in " +
                       std::to_string(timeout_ms) + " ms (mode startall)");
    }
  } catch (const std::exception &e) {
    std::cout << "EXCEPTION: " << e.what() << "\nFAIL\n" << std::flush;
    _exit(3);
  }

  int bad_runs = 0;
  for (int r = 0; r < n; r++) {
    bo_out[r].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (!check_run(r, bo_out[r].map<int32_t *>()))
      bad_runs++;
  }
  std::cout << "mode " << mode << ": " << n - bad_runs << "/" << n
            << " runs exact (A,B alternating)\n"
            << (bad_runs ? "FAIL" : "PASS") << "\n";
  return bad_runs ? 1 : 0;
}
