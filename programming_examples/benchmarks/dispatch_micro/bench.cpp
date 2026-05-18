//===- bench.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// Host driver for the dispatch_micro benchmark suite.
//
// Two runtime-family paths share the same timing layer:
//
//   Path X (xclbin family) — mechanism = baseline
//     xrt::xclbin → register_xclbin → xrt::hw_context(device, uuid)
//                → xrt::kernel → kernel(opcode, bo_instr, sz, ...bufs)
//
//   Path E (full-ELF family) — mechanism = load_pdi_fw, load_pdi_expanded
//     xrt::elf{"aie.elf"} → xrt::hw_context(device, elf)
//                        → xrt::ext::kernel(context, "main:sequence")
//                        → run.set_arg / run.start / run.wait2
//
// Metrics:
//   cold_start       once per fresh process; per-phase breakdown
//   warm_reconfig    pre-built context; brackets only the reconfig dispatch
//   pure_dispatch    pre-built; identity-mapped buffers; hot loop
//
// ctrlpkt is not wired up in this initial cut; it has a different artifact
// shape (separate ctrlpkt.bin + ctrlpkt_dma_seq.bin + extra kernel arg slots).
// See README; tracked as v2.
//===----------------------------------------------------------------------===//

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_kernel.h" // xrt::runlist
#include "xrt/experimental/xrt_module.h"

#include "bench_runner.h"
#include "json_writer.h"

namespace fs = std::filesystem;
using namespace dispatch_micro;

namespace {

struct Args {
  std::string build_dir;
  std::string mechanism;   // baseline | load_pdi_fw | load_pdi_expanded
  std::string metric;      // cold_start | warm_reconfig | pure_dispatch
  int warmup = 10;
  int iters = 100;
  int tiles = 1;
  int rows_per_col = 1;
  int bds = 2;
  bool batched = false;
  int batch_size = 16;
  std::string json_out;    // empty = stdout
};

void usage() {
  std::cerr <<
    "Usage: bench --build-dir=<dir> --mechanism=<m> --metric=<met>\n"
    "             [--warmup=N] [--iters=N] [--tiles=N] [--bds=N]\n"
    "             [--batched] [--batch-size=N] [--json-out=<file>]\n"
    "  mechanism: baseline | load_pdi_fw | load_pdi_expanded\n"
    "  metric:    cold_start | warm_reconfig | pure_dispatch\n";
}

bool starts_with(const std::string &s, const char *p) {
  return s.rfind(p, 0) == 0;
}

std::string get_val(const std::string &arg, size_t eq) {
  return arg.substr(eq + 1);
}

bool parse_args(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    auto v = [&]{ return get_val(s, eq); };
    if (starts_with(s, "--build-dir=")) a.build_dir = v();
    else if (starts_with(s, "--mechanism=")) a.mechanism = v();
    else if (starts_with(s, "--metric=")) a.metric = v();
    else if (starts_with(s, "--warmup=")) a.warmup = std::stoi(v());
    else if (starts_with(s, "--iters=")) a.iters = std::stoi(v());
    else if (starts_with(s, "--tiles=")) a.tiles = std::stoi(v());
    else if (starts_with(s, "--rows-per-col=")) a.rows_per_col = std::stoi(v());
    else if (starts_with(s, "--bds=")) a.bds = std::stoi(v());
    else if (s == "--batched") a.batched = true;
    else if (starts_with(s, "--batch-size=")) a.batch_size = std::stoi(v());
    else if (starts_with(s, "--json-out=")) a.json_out = v();
    else if (s == "-h" || s == "--help") { usage(); return false; }
    else { std::cerr << "unknown arg: " << s << "\n"; usage(); return false; }
  }
  if (a.build_dir.empty() || a.mechanism.empty() || a.metric.empty()) {
    usage();
    return false;
  }
  return true;
}

std::vector<uint32_t> load_instr_binary(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  size_t bytes = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint32_t> v(bytes / sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(uint32_t));
  return v;
}

// --- Path X: xclbin + xrt::kernel -----------------------------------------

struct PathX {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr;
  std::vector<xrt::bo> data_bos;
  std::vector<uint32_t> instr_v;
  unsigned int opcode = 3;

  // Phase timings populated by build_cold for the cold_start metric.
  long long load_ns = 0, register_ns = 0, kernel_ns = 0, first_dispatch_ns = 0;

  static std::string find_kernel_name(const xrt::xclbin &x) {
    auto kernels = x.get_kernels();
    for (auto &k : kernels) {
      auto n = k.get_name();
      if (n.rfind("MLIR_AIE", 0) == 0) return n;
    }
    if (kernels.empty()) throw std::runtime_error("no kernels in xclbin");
    return kernels.front().get_name();
  }

  // Pre-load and pre-allocate buffers; used for warm_reconfig / pure_dispatch.
  // No cold-start timing collected here.
  void build_warm(const Args &a, size_t total_bytes) {
    device = xrt::device(0);
    xclbin = xrt::xclbin(a.build_dir + "/aie.xclbin");
    auto name = find_kernel_name(xclbin);
    device.register_xclbin(xclbin);
    context = xrt::hw_context(device, xclbin.get_uuid());
    kernel = xrt::kernel(context, name);

    instr_v = load_instr_binary(a.build_dir + "/insts.bin");
    bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                       XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    std::memcpy(bo_instr.map<void*>(), instr_v.data(),
                instr_v.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Two shared tensor BOs: one in, one out, each big enough for all tiles.
    // Aiecc caps the kernel at 5 BO arg slots (aiecc.cpp:3558), so we cannot
    // pass one BO per tile beyond ~2 tiles; instead, generate.py emits the
    // runtime sequence with two args and per-tile offsets.
    auto bo_in = xrt::bo(device, total_bytes,
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_out = xrt::bo(device, total_bytes,
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    data_bos.push_back(std::move(bo_in));
    data_bos.push_back(std::move(bo_out));
  }

  // Cold-start path: time each phase separately on a fresh state.
  void build_cold(const Args &a, size_t total_bytes) {
    device = xrt::device(0);
    load_ns = time_once_ns([&]{
      xclbin = xrt::xclbin(a.build_dir + "/aie.xclbin");
    });
    auto name = find_kernel_name(xclbin);
    register_ns = time_once_ns([&]{
      device.register_xclbin(xclbin);
      context = xrt::hw_context(device, xclbin.get_uuid());
    });
    kernel_ns = time_once_ns([&]{
      kernel = xrt::kernel(context, name);
    });

    instr_v = load_instr_binary(a.build_dir + "/insts.bin");
    bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                       XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    std::memcpy(bo_instr.map<void*>(), instr_v.data(),
                instr_v.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto bo_in = xrt::bo(device, total_bytes,
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_out = xrt::bo(device, total_bytes,
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    data_bos.push_back(std::move(bo_in));
    data_bos.push_back(std::move(bo_out));

    first_dispatch_ns = time_once_ns([&]{ dispatch_once(); });
  }

  void dispatch_once() {
    // Kernel signature: (opcode, bo_instr, instr_count, bo_t0, bo_t1, ...)
    // Use a runtime-arg-set pattern so we don't have to template on tile count.
    auto run = xrt::run(kernel);
    int idx = 0;
    run.set_arg(idx++, opcode);
    run.set_arg(idx++, bo_instr);
    run.set_arg(idx++, (uint32_t)instr_v.size());
    for (auto &bo : data_bos) run.set_arg(idx++, bo);
    run.start();
    run.wait();
  }

  long long dispatch_batched(int batch) {
    xrt::runlist rl(context);
    std::vector<xrt::run> runs;
    runs.reserve(batch);
    for (int i = 0; i < batch; ++i) {
      xrt::run run(kernel);
      int idx = 0;
      run.set_arg(idx++, opcode);
      run.set_arg(idx++, bo_instr);
      run.set_arg(idx++, (uint32_t)instr_v.size());
      for (auto &bo : data_bos) run.set_arg(idx++, bo);
      rl.add(run);
      runs.push_back(std::move(run));
    }
    auto t0 = std::chrono::steady_clock::now();
    rl.execute();
    rl.wait();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }
};

// --- Path E: full ELF + xrt::ext::kernel ----------------------------------

struct PathE {
  xrt::device device;
  xrt::elf elf;
  xrt::hw_context context;
  // xrt::ext::kernel has no default constructor; hold it in an optional so
  // PathE can be heap-allocated without arguments and built later.
  std::optional<xrt::ext::kernel> kernel;
  xrt::bo bo_inout;
  size_t bytes = 0;

  long long load_ns = 0, register_ns = 0, kernel_ns = 0, first_dispatch_ns = 0;

  void build_warm(const Args &a, size_t total_bytes) {
    bytes = total_bytes;
    device = xrt::device(0);
    elf = xrt::elf(a.build_dir + "/aie.elf");
    context = xrt::hw_context(device, elf);
    kernel.emplace(context, "main:seq");
    bo_inout = xrt::ext::bo(device, bytes);
  }

  void build_cold(const Args &a, size_t total_bytes) {
    bytes = total_bytes;
    device = xrt::device(0);
    load_ns = time_once_ns([&]{
      elf = xrt::elf(a.build_dir + "/aie.elf");
    });
    register_ns = time_once_ns([&]{
      context = xrt::hw_context(device, elf);
    });
    kernel_ns = time_once_ns([&]{
      kernel.emplace(context, "main:seq");
    });
    bo_inout = xrt::ext::bo(device, bytes);
    first_dispatch_ns = time_once_ns([&]{ dispatch_once(); });
  }

  void dispatch_once() {
    auto run = xrt::run(*kernel);
    run.set_arg(0, bo_inout);
    run.start();
    run.wait2();
  }
};

// --- emit JSON ------------------------------------------------------------

void emit_json(const Args &a, const TimingResult *warm, PathX *px, PathE *pe,
               std::ostream &out) {
  JsonWriter w(out);
  w.str("mechanism", a.mechanism);
  w.str("metric", a.metric);
  w.str("build_dir", a.build_dir);
  w.num("tiles", a.tiles);
  w.num("rows_per_col", a.rows_per_col);
  w.num("bds", a.bds);
  w.num("warmup", a.warmup);
  w.num("iters", a.iters);
  w.boolean("batched", a.batched);
  w.num("batch_size", a.batched ? a.batch_size : 1);

  if (warm) {
    std::ostringstream ns;
    ns << "{\"min\":" << warm->min_ns
       << ",\"p50\":" << warm->p50_ns
       << ",\"p90\":" << warm->p90_ns
       << ",\"p99\":" << warm->p99_ns
       << ",\"max\":" << warm->max_ns
       << ",\"avg\":" << warm->avg_ns << "}";
    w.raw("ns", ns.str());
    w.array_ns("ns_samples", warm->samples_ns);
  }
  if (a.metric == "cold_start") {
    std::ostringstream ph;
    long long load = px ? px->load_ns : (pe ? pe->load_ns : 0);
    long long reg  = px ? px->register_ns : (pe ? pe->register_ns : 0);
    long long krn  = px ? px->kernel_ns : (pe ? pe->kernel_ns : 0);
    long long fd   = px ? px->first_dispatch_ns : (pe ? pe->first_dispatch_ns : 0);
    ph << "{\"load_ns\":" << load
       << ",\"register_ns\":" << reg
       << ",\"kernel_ns\":" << krn
       << ",\"first_dispatch_ns\":" << fd << "}";
    w.raw("cold_phases", ph.str());
  }
}

} // namespace

int main(int argc, char **argv) {
  Args a;
  if (!parse_args(argc, argv, a)) return 1;

  // total_bytes = LINE_LEN * bds * tiles * rows_per_col * sizeof(int32) ;
  // LINE_LEN=1024 in generate.py. generate.py packs all (col, row) compute
  // tiles into a single shared in/out buffer (one BO each, since aiecc caps
  // at 5 BO slots).
  constexpr int LINE_LEN = 1024;
  size_t total_bytes =
      static_cast<size_t>(LINE_LEN) * a.bds * a.tiles * a.rows_per_col
      * sizeof(int32_t);

  bool x_family = (a.mechanism == "baseline");
  bool e_family = (a.mechanism == "load_pdi_fw" ||
                   a.mechanism == "load_pdi_expanded");
  if (!x_family && !e_family) {
    std::cerr << "Unsupported mechanism: " << a.mechanism
              << " (ctrlpkt not yet wired up)\n";
    return 2;
  }

  std::ofstream fout;
  std::ostream *out = &std::cout;
  if (!a.json_out.empty()) {
    fout.open(a.json_out, std::ios::app);
    out = &fout;
  }

  if (a.metric == "cold_start") {
    if (x_family) {
      PathX px;
      px.build_cold(a, total_bytes);
      emit_json(a, nullptr, &px, nullptr, *out);
    } else {
      PathE pe;
      pe.build_cold(a, total_bytes);
      emit_json(a, nullptr, nullptr, &pe, *out);
    }
    return 0;
  }

  // warm_reconfig / pure_dispatch — same code path, differ only in what the
  // build artifact already does. The reconfig vs dispatch distinction lives
  // in the txn binary / ELF, not in how the host calls kernel().
  TimingResult tr;
  if (x_family) {
    PathX px;
    px.build_warm(a, total_bytes);
    if (a.batched) {
      tr = run_timed(a.warmup, a.iters,
                     [&]{ (void)px.dispatch_batched(a.batch_size); });
    } else {
      tr = run_timed(a.warmup, a.iters, [&]{ px.dispatch_once(); });
    }
    emit_json(a, &tr, &px, nullptr, *out);
  } else {
    PathE pe;
    pe.build_warm(a, total_bytes);
    tr = run_timed(a.warmup, a.iters, [&]{ pe.dispatch_once(); });
    emit_json(a, &tr, nullptr, &pe, *out);
  }
  return 0;
}
