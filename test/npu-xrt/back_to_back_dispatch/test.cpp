// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Back-to-back dispatch check for the AIE4 full-ELF design out = in + K
// (1024 x i32, through a memtile). Dispatches the kernel N times with no gap,
// each run with its own input and a poisoned output, then checks every run.
//
//   --mode seq       start a run, wait for it, start the next (no sleep)
//   --mode runlist   one xrt::runlist holding all N runs
//   --mode startall  start all N runs, then wait for each
//   --mode alternate seq, alternating between --elf and --alt-elf on one
//                    hw_context (every dispatch loads the other ELF's
//                    configuration); at most 8 runs
//
// It also reports the time per dispatch: in seq mode the median of runs
// 1..N-1 (run 0 loads the configuration), otherwise the total over N.
// Every wait is bounded: a hung dispatch is reported as a timeout.

#include <algorithm>
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
constexpr size_t BYTES = COUNT * sizeof(int32_t);
constexpr int32_t POISON = (int32_t)0xdeadbeef;
// Alternating ELFs reloads the configuration over live memtile channels on
// every dispatch, displacing output data unless the configuration resets
// the array first; keep it short.
constexpr int MAX_RUNS_ALTERNATE = 8;

using Clock = std::chrono::steady_clock;

static int32_t input_value(int run, int i) { return ((run + 1) << 16) | i; }

// Returns true if the run's output is exact; otherwise prints the first
// mismatch and what the output looks like.
static bool check_run(int run, int k, const int32_t *out) {
  int first = -1, bad = 0;
  for (int i = 0; i < COUNT; i++)
    if (out[i] != input_value(run, i) + k) {
      if (first < 0)
        first = i;
      bad++;
    }
  if (!bad)
    return true;
  std::cout << "run " << run << ": FAIL " << bad << "/" << COUNT
            << " elements wrong, first at index " << first << " (got 0x"
            << std::hex << out[first] << ", expected 0x"
            << input_value(run, first) + k << std::dec << ")";
  if (out[first] == POISON) {
    std::cout << "; output never written\n";
    return false;
  }
  // Which run's input, and which element, the wrong value came from, for
  // either K of the two designs.
  for (int kk : {k, 1, 2}) {
    int32_t v = (int32_t)((uint32_t)out[first] - (uint32_t)kk);
    int src = ((v >> 16) & 0xffff) - 1, elem = v & 0xffff;
    if (src >= 0 && elem < COUNT) {
      std::cout << "; looks like run " << src << " element " << elem << " + "
                << kk;
      break;
    }
  }
  std::cout << "\n";
  return false;
}

[[noreturn]] static void timeout_exit(const std::string &what) {
  // Skip XRT destructors: they can block or throw on a hung context.
  std::cout << "TIMEOUT: " << what << "\nFAIL\n" << std::flush;
  _exit(2);
}

int main(int argc, char **argv) {
  std::string elf_name = "back_to_back_dispatch.elf", mode = "seq";
  std::string kernel_name = "main:sequence";
  std::string alt_elf, alt_kernel = "main_b:sequence";
  int n = 32, timeout_ms = 5000, k = 1, alt_k = 2;
  // Parse a whole-string integer in [lo, hi]; exit with usage error otherwise.
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
    if (a == "--elf" && has_val)
      elf_name = argv[++i];
    else if (a == "--kernel" && has_val)
      kernel_name = argv[++i];
    else if (a == "--k" && has_val)
      k = parse_int(a, argv[++i], 0, 1000);
    else if (a == "--alt-elf" && has_val)
      alt_elf = argv[++i];
    else if (a == "--alt-kernel" && has_val)
      alt_kernel = argv[++i];
    else if (a == "--alt-k" && has_val)
      alt_k = parse_int(a, argv[++i], 0, 1000);
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
                << " [--elf f] [--kernel name] [--k K]"
                   " [--mode seq|runlist|startall|alternate] [--n N]"
                   " [--timeout-ms T] [--alt-elf f] [--alt-kernel name]"
                   " [--alt-k K]\n";
      return 2;
    }
  }
  if (mode != "seq" && mode != "runlist" && mode != "startall" &&
      mode != "alternate") {
    std::cerr << "unknown mode " << mode << "\n";
    return 2;
  }
  bool alternate = mode == "alternate";
  if (alternate && alt_elf.empty()) {
    std::cerr << "--mode alternate needs --alt-elf\n";
    return 2;
  }
  if (alternate && n > MAX_RUNS_ALTERNATE) {
    std::cerr << "refusing --n " << n << " in alternate mode (at most "
              << MAX_RUNS_ALTERNATE << ")\n";
    return 2;
  }
  std::chrono::milliseconds timeout(timeout_ms);

  auto device = xrt::device(0u);
  xrt::elf elf{elf_name};
  xrt::hw_context context(device, elf);
  std::vector<xrt::kernel> kernels;
  std::vector<int> ks{k};
  kernels.push_back(xrt::ext::kernel(context, kernel_name));
  if (alternate) {
    xrt::elf elf2{alt_elf};
    context.add_config(elf2);
    kernels.push_back(xrt::ext::kernel(context, alt_kernel));
    ks.push_back(alt_k);
  }
  auto design_of = [&](int r) { return alternate ? r % 2 : 0; };

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
    xrt::run run(kernels[design_of(r)]);
    run.set_arg(0, bo_in[r]);
    run.set_arg(1, bo_out[r]);
    return run;
  };

  std::vector<double> run_us;
  double total_us = 0;
  try {
    if (mode == "seq" || alternate) {
      for (int r = 0; r < n; r++) {
        auto run = make_run(r);
        auto t0 = Clock::now();
        run.start();
        if (run.wait2(timeout) == std::cv_status::timeout)
          timeout_exit("run " + std::to_string(r) + " of " + std::to_string(n) +
                       " did not complete in " + std::to_string(timeout_ms) +
                       " ms (mode " + mode + ")");
        run_us.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - t0)
                .count());
      }
    } else if (mode == "runlist") {
      std::vector<xrt::run> runs;
      for (int r = 0; r < n; r++)
        runs.push_back(make_run(r));
      xrt::runlist list(context);
      for (auto &run : runs)
        list.add(run);
      auto t0 = Clock::now();
      list.execute();
      if (list.wait(timeout) == std::cv_status::timeout)
        timeout_exit("runlist of " + std::to_string(n) +
                     " runs did not complete in " + std::to_string(timeout_ms) +
                     " ms");
      total_us =
          std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    } else { // startall
      std::vector<xrt::run> runs;
      for (int r = 0; r < n; r++)
        runs.push_back(make_run(r));
      auto t0 = Clock::now();
      for (auto &run : runs)
        run.start();
      for (int r = 0; r < n; r++)
        if (runs[r].wait2(timeout) == std::cv_status::timeout)
          timeout_exit("run " + std::to_string(r) + " of " + std::to_string(n) +
                       " did not complete in " + std::to_string(timeout_ms) +
                       " ms (mode startall)");
      total_us =
          std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }
  } catch (const std::exception &e) {
    std::cout << "EXCEPTION: " << e.what() << "\nFAIL\n" << std::flush;
    _exit(3);
  }

  int bad_runs = 0;
  for (int r = 0; r < n; r++) {
    bo_out[r].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (!check_run(r, ks[design_of(r)], bo_out[r].map<int32_t *>()))
      bad_runs++;
  }
  if (!run_us.empty()) {
    std::cout << "time: run 0 " << (long)run_us[0] << " us";
    if (run_us.size() > 1) {
      std::vector<double> rest(run_us.begin() + 1, run_us.end());
      std::sort(rest.begin(), rest.end());
      std::cout << ", runs 1.." << n - 1 << " median "
                << (long)rest[rest.size() / 2] << " us/dispatch";
    }
    std::cout << "\n";
  } else {
    std::cout << "time: " << (long)(total_us / n) << " us/dispatch over " << n
              << " runs\n";
  }
  std::cout << "mode " << mode << ": " << n - bad_runs << "/" << n
            << " runs exact\n"
            << (bad_runs ? "FAIL" : "PASS") << "\n";
  return bad_runs ? 1 : 0;
}
