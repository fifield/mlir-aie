//===- bench_runner.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// Generic warmup + N-iteration timing loop accepting a std::function<void()>.
// Returns per-iteration nanoseconds plus min/avg/max/p50/p90/p99 stats.
// Based on the timing pattern in runtime_lib/test_lib/xrt_test_wrapper.h:183-244;
// generalized so it works with both xrt::kernel and xrt::ext::kernel paths
// and supports xrt::runlist batching by wrapping the bracketed code in a
// std::function instead of templating on a fixed-arity kernel.
//===----------------------------------------------------------------------===//
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

namespace dispatch_micro {

struct TimingResult {
  std::vector<long long> samples_ns;
  long long min_ns = 0;
  long long max_ns = 0;
  long long avg_ns = 0;
  long long p50_ns = 0;
  long long p90_ns = 0;
  long long p99_ns = 0;
};

// Run `body` `iters` times after `warmup` discarded calls; record per-iter ns.
// `body` is responsible for any required submit + wait inside the timed
// region. Callers can use this for either a single kernel() dispatch or a
// runlist::execute() + wait() pair.
inline TimingResult run_timed(int warmup, int iters,
                              const std::function<void()> &body) {
  for (int w = 0; w < warmup; ++w) body();

  TimingResult r;
  r.samples_ns.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    body();
    auto t1 = std::chrono::steady_clock::now();
    r.samples_ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  }

  std::vector<long long> sorted = r.samples_ns;
  std::sort(sorted.begin(), sorted.end());
  r.min_ns = sorted.front();
  r.max_ns = sorted.back();
  long long sum = 0;
  for (auto v : sorted) sum += v;
  r.avg_ns = sum / static_cast<long long>(sorted.size());
  auto pct = [&](double p) -> long long {
    if (sorted.empty()) return 0;
    size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
    return sorted[idx];
  };
  r.p50_ns = pct(0.50);
  r.p90_ns = pct(0.90);
  r.p99_ns = pct(0.99);
  return r;
}

// Single-shot timer for cold_start: each phase is timed once, no warmup.
struct Phase {
  const char *name;
  long long ns;
};

inline long long time_once_ns(const std::function<void()> &body) {
  auto t0 = std::chrono::steady_clock::now();
  body();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

} // namespace dispatch_micro
