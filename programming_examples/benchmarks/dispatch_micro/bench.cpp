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
//   Path C (ctrlpkt) — mechanism = ctrlpkt
//     xrt::xclbin{"aie.xclbin"} (skeleton with column-control overlay)
//       + xrt::elf{"aie.elf"} (ctrlpkt-encoded)
//       → device.register_xclbin → xrt::hw_context(device, xclbin.uuid)
//       → xrt::ext::kernel(context, module, name) [three-arg variant]
//       → kernel(opcode=3, 0, 0, bo_in, bo_out) — scalars for instr/size
//
// Metrics:
//   cold_start       once per fresh process; per-phase breakdown
//   warm_reconfig    pre-built context; brackets only the reconfig dispatch
//   pure_dispatch    pre-built; identity-mapped buffers; hot loop
//
// v2 update: ctrlpkt is now wired via Path C above.
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
  int n_configs = 0;
  bool batched = false;
  int batch_size = 16;
  std::string ctrlpkt_strategy = "reuse"; // "reuse" | "fresh_ctx"
  bool vary_args = false;  // for batched: use distinct BOs per run
  std::string json_out;    // empty = stdout
};

void usage() {
  std::cerr <<
    "Usage: bench --build-dir=<dir> --mechanism=<m> --metric=<met>\n"
    "             [--warmup=N] [--iters=N] [--tiles=N] [--bds=N]\n"
    "             [--batched] [--batch-size=N] [--json-out=<file>]\n"
    "  mechanism: baseline | load_pdi_fw | load_pdi_expanded\n"
    "  metric:    cold_start | warm_reconfig | pure_dispatch | ab_toggle | multi_toggle\n"
    "  (ab_toggle: ELF mechanism + AB-mode build dir)\n"
    "  (multi_toggle: ELF mechanism + N-configs build dir; pass --n-configs=N)\n";
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
    else if (starts_with(s, "--n-configs=")) a.n_configs = std::stoi(v());
    else if (s == "--batched") a.batched = true;
    else if (starts_with(s, "--batch-size=")) a.batch_size = std::stoi(v());
    else if (starts_with(s, "--ctrlpkt-strategy=")) a.ctrlpkt_strategy = v();
    else if (s == "--vary-args") a.vary_args = true;
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

  // v2 #6 probe: batched dispatch with *distinct* in/out BO pairs per run.
  // Tests whether the runtime is collapsing identical runs (in which case
  // total batch time should jump dramatically when args differ).
  // Requires allocating `batch` distinct in/out BO pairs ahead of time —
  // call build_warm_vary_args first.
  std::vector<xrt::bo> vary_bo_in, vary_bo_out;
  void build_warm_vary_args(const Args &a, size_t total_bytes, int batch) {
    build_warm(a, total_bytes); // populates everything else
    vary_bo_in.clear();
    vary_bo_out.clear();
    vary_bo_in.reserve(batch);
    vary_bo_out.reserve(batch);
    for (int i = 0; i < batch; ++i) {
      vary_bo_in.emplace_back(device, total_bytes,
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
      vary_bo_out.emplace_back(device, total_bytes,
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    }
  }
  long long dispatch_batched_vary(int batch) {
    xrt::runlist rl(context);
    std::vector<xrt::run> runs;
    runs.reserve(batch);
    for (int i = 0; i < batch; ++i) {
      xrt::run run(kernel);
      run.set_arg(0, opcode);
      run.set_arg(1, bo_instr);
      run.set_arg(2, (uint32_t)instr_v.size());
      run.set_arg(3, vary_bo_in[i]);
      run.set_arg(4, vary_bo_out[i]);
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
  // AB-toggle: second kernel handle bound to the orchestrator's other
  // runtime sequence. Populated only by build_warm_ab().
  std::optional<xrt::ext::kernel> kernel_b;
  // multi_toggle: N kernel handles, one per ab_orch:seq_to_{i}.
  std::vector<xrt::ext::kernel> kernels_n;
  xrt::bo bo_inout;
  xrt::bo bo_out;
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

  // AB-mode warm build. The ELF contains both PDIs (@cfg_a, @cfg_b) plus
  // the @ab_orch device, whose runtime sequences seq_to_a / seq_to_b each
  // issue one npu_load_pdi targeting a distinct PDI. Alternating between
  // the two handles in a hot loop defeats the firmware's PDI cache.
  void build_warm_ab(const Args &a, size_t total_bytes) {
    bytes = total_bytes;
    device = xrt::device(0);
    elf = xrt::elf(a.build_dir + "/aie.elf");
    context = xrt::hw_context(device, elf);
    kernel.emplace(context, "ab_orch:seq_to_a");
    kernel_b.emplace(context, "ab_orch:seq_to_b");
    bo_inout = xrt::ext::bo(device, bytes);
    bo_out = xrt::ext::bo(device, bytes);
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

  // Runlist-batched dispatch for v2 #6 probe. Uses the same single bo_inout
  // for every run — this is the "identical args" case that v1 §3 measured.
  long long dispatch_batched(int batch) {
    xrt::runlist rl(context);
    std::vector<xrt::run> runs;
    runs.reserve(batch);
    for (int i = 0; i < batch; ++i) {
      xrt::run run(*kernel);
      run.set_arg(0, bo_inout);
      rl.add(run);
      runs.push_back(std::move(run));
    }
    auto t0 = std::chrono::steady_clock::now();
    rl.execute();
    rl.wait();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }

  // Same as dispatch_batched but with distinct buffers per run. Tests
  // whether the runtime is collapsing identical runs.
  std::vector<xrt::bo> vary_bos;
  void prep_vary_args(int batch) {
    vary_bos.clear();
    vary_bos.reserve(batch);
    for (int i = 0; i < batch; ++i) {
      vary_bos.emplace_back(xrt::ext::bo(device, bytes));
    }
  }
  long long dispatch_batched_vary(int batch) {
    xrt::runlist rl(context);
    std::vector<xrt::run> runs;
    runs.reserve(batch);
    for (int i = 0; i < batch; ++i) {
      xrt::run run(*kernel);
      run.set_arg(0, vary_bos[i]);
      rl.add(run);
      runs.push_back(std::move(run));
    }
    auto t0 = std::chrono::steady_clock::now();
    rl.execute();
    rl.wait();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }

  void dispatch_to_a() {
    auto run = xrt::run(*kernel);
    run.set_arg(0, bo_inout);
    run.set_arg(1, bo_out);
    run.start();
    run.wait2();
  }

  void dispatch_to_b() {
    auto run = xrt::run(*kernel_b);
    run.set_arg(0, bo_inout);
    run.set_arg(1, bo_out);
    run.start();
    run.wait2();
  }

  // Build N kernel handles bound to ab_orch:seq_to_0..seq_to_{N-1}.
  // Used by --metric=multi_toggle to probe whether a PDI cache exists and
  // how large it is by rotating through N distinct PDIs.
  void build_warm_multi(const Args &a, int n, size_t total_bytes) {
    bytes = total_bytes;
    device = xrt::device(0);
    elf = xrt::elf(a.build_dir + "/aie.elf");
    context = xrt::hw_context(device, elf);
    kernels_n.reserve(n);
    for (int i = 0; i < n; ++i) {
      kernels_n.emplace_back(context, "ab_orch:seq_to_" + std::to_string(i));
    }
    bo_inout = xrt::ext::bo(device, bytes);
    bo_out = xrt::ext::bo(device, bytes);
  }

  void dispatch_to_k(int k) {
    auto run = xrt::run(kernels_n[k]);
    run.set_arg(0, bo_inout);
    run.set_arg(1, bo_out);
    run.start();
    run.wait2();
  }
};

// --- Path C: xclbin overlay + ctrlpkt ELF ---------------------------------

struct PathC {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::elf elf;
  xrt::hw_context context;
  // The ctrlpkt path requires xrt::module to bind the ELF (containing
  // ctrlpkt.bin + ctrlpkt_dma_seq.bin) onto the existing skeleton xclbin's
  // hw_context. xrt::module has a default ctor; xrt::ext::kernel doesn't.
  xrt::module module_;
  std::optional<xrt::ext::kernel> kernel;
  xrt::bo bo_in, bo_out;
  unsigned int opcode = 3;
  size_t bytes = 0;

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

  void build_warm(const Args &a, size_t total_bytes) {
    bytes = total_bytes;
    device = xrt::device(0);
    xclbin = xrt::xclbin(a.build_dir + "/aie.xclbin");
    auto name = find_kernel_name(xclbin);
    device.register_xclbin(xclbin);
    elf = xrt::elf(a.build_dir + "/aie.elf");
    module_ = xrt::module(elf);
    context = xrt::hw_context(device, xclbin.get_uuid());
    kernel.emplace(context, module_, name);
    bo_in = xrt::ext::bo(device, bytes);
    bo_out = xrt::ext::bo(device, bytes);
  }

  void build_cold(const Args &a, size_t total_bytes) {
    bytes = total_bytes;
    device = xrt::device(0);
    load_ns = time_once_ns([&]{
      xclbin = xrt::xclbin(a.build_dir + "/aie.xclbin");
      elf = xrt::elf(a.build_dir + "/aie.elf");
      module_ = xrt::module(elf);
    });
    auto name = find_kernel_name(xclbin);
    register_ns = time_once_ns([&]{
      device.register_xclbin(xclbin);
      context = xrt::hw_context(device, xclbin.get_uuid());
    });
    kernel_ns = time_once_ns([&]{
      kernel.emplace(context, module_, name);
    });
    bo_in = xrt::ext::bo(device, bytes);
    bo_out = xrt::ext::bo(device, bytes);
    first_dispatch_ns = time_once_ns([&]{ dispatch_once(); });
  }

  void dispatch_once() {
    // Per test/npu-xrt/ctrl_packet_reconfig_elf/test.cpp:77 — the ctrlpkt
    // kernel signature still has the (opcode, instr, instr_count, ...) prefix
    // but the instr buffer is unused because the ELF carries the instruction
    // stream; pass 0 / 0 for the scalar slots.
    auto run = xrt::run(*kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, 0);
    run.set_arg(2, 0);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_out);
    run.start();
    run.wait2();
  }

  // Variant of dispatch_once that creates a fresh hw_context per call.
  // Tries to defeat whatever state-machine leakage causes the 2nd-dispatch
  // hang for the ctrlpkt mechanism in v2 #11. Slow per call (~80 ms,
  // dominated by hw_context cost) but lets us get N samples without the
  // firmware getting stuck.
  void dispatch_fresh_context(const Args &a) {
    xrt::hw_context fresh_ctx(device, xclbin.get_uuid());
    xrt::module fresh_mod(elf);
    auto name = find_kernel_name(xclbin);
    xrt::ext::kernel fresh_kernel(fresh_ctx, fresh_mod, name);
    auto run = xrt::run(fresh_kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, 0);
    run.set_arg(2, 0);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_out);
    run.start();
    run.wait2();
  }

  // Lighter-weight reset variants for v2 #11 — try cheaper state resets
  // without recreating hw_context. If any of these work for the
  // 2nd-dispatch hang, the per-dispatch cost would be a more meaningful
  // number than the ~80 ms fresh_context number.
  void dispatch_fresh_kernel() {
    auto name = find_kernel_name(xclbin);
    xrt::ext::kernel fresh_kernel(context, module_, name);
    auto run = xrt::run(fresh_kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, 0);
    run.set_arg(2, 0);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_out);
    run.start();
    run.wait2();
  }

  void dispatch_fresh_module() {
    auto name = find_kernel_name(xclbin);
    xrt::module fresh_mod(elf);
    xrt::ext::kernel fresh_kernel(context, fresh_mod, name);
    auto run = xrt::run(fresh_kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, 0);
    run.set_arg(2, 0);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_out);
    run.start();
    run.wait2();
  }
};

// --- emit JSON ------------------------------------------------------------

void emit_json(const Args &a, const TimingResult *warm,
               PathX *px, PathE *pe, PathC *pc,
               std::ostream &out, const char *direction = nullptr) {
  JsonWriter w(out);
  w.str("mechanism", a.mechanism);
  w.str("metric", a.metric);
  w.str("build_dir", a.build_dir);
  w.num("tiles", a.tiles);
  w.num("rows_per_col", a.rows_per_col);
  w.num("bds", a.bds);
  if (a.n_configs > 0) w.num("n_configs", a.n_configs);
  w.num("warmup", a.warmup);
  w.num("iters", a.iters);
  w.boolean("batched", a.batched);
  w.num("batch_size", a.batched ? a.batch_size : 1);
  if (direction) w.str("direction", direction);

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
    long long load = px ? px->load_ns : (pe ? pe->load_ns : (pc ? pc->load_ns : 0));
    long long reg  = px ? px->register_ns : (pe ? pe->register_ns : (pc ? pc->register_ns : 0));
    long long krn  = px ? px->kernel_ns : (pe ? pe->kernel_ns : (pc ? pc->kernel_ns : 0));
    long long fd   = px ? px->first_dispatch_ns : (pe ? pe->first_dispatch_ns : (pc ? pc->first_dispatch_ns : 0));
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
  bool c_family = (a.mechanism == "ctrlpkt");
  if (!x_family && !e_family && !c_family) {
    std::cerr << "Unsupported mechanism: " << a.mechanism << "\n";
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
      emit_json(a, nullptr, &px, nullptr, nullptr, *out);
    } else if (c_family) {
      PathC pc;
      pc.build_cold(a, total_bytes);
      emit_json(a, nullptr, nullptr, nullptr, &pc, *out);
    } else {
      PathE pe;
      pe.build_cold(a, total_bytes);
      emit_json(a, nullptr, nullptr, &pe, nullptr, *out);
    }
    return 0;
  }

  // AB toggle: ELF family only. Alternates dispatches between two ext::kernel
  // handles (ab_orch:seq_to_a, ab_orch:seq_to_b) so each iteration loads a
  // *different* PDI than the previous one. Defeats the firmware PDI cache.
  // We collect samples per direction (a→b vs b→a) and emit two JSON rows.
  if (a.metric == "ab_toggle") {
    if (!e_family) {
      std::cerr << "ab_toggle requires an ELF-family mechanism (load_pdi_*); "
                << "got " << a.mechanism << "\n";
      return 2;
    }
    PathE pe;
    pe.build_warm_ab(a, total_bytes);

    // Warmup, alternating directions.
    for (int w = 0; w < a.warmup; ++w) {
      if (w & 1) pe.dispatch_to_a(); else pe.dispatch_to_b();
    }
    std::vector<long long> samples_to_b, samples_to_a;
    samples_to_b.reserve(a.iters);
    samples_to_a.reserve(a.iters);
    for (int i = 0; i < a.iters; ++i) {
      // Each loop iteration covers one a→b then one b→a so both directions
      // run the same number of times.
      auto t0 = std::chrono::steady_clock::now();
      pe.dispatch_to_b();
      auto t1 = std::chrono::steady_clock::now();
      pe.dispatch_to_a();
      auto t2 = std::chrono::steady_clock::now();
      samples_to_b.push_back(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      samples_to_a.push_back(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    }

    auto summarize = [](std::vector<long long> &s) {
      TimingResult r;
      r.samples_ns = s;
      auto sorted = s;
      std::sort(sorted.begin(), sorted.end());
      r.min_ns = sorted.front();
      r.max_ns = sorted.back();
      long long sum = 0; for (auto v : sorted) sum += v;
      r.avg_ns = sum / static_cast<long long>(sorted.size());
      auto pct = [&](double p) {
        return sorted[static_cast<size_t>(p * (sorted.size() - 1))];
      };
      r.p50_ns = pct(0.50); r.p90_ns = pct(0.90); r.p99_ns = pct(0.99);
      return r;
    };
    auto tr_b = summarize(samples_to_b);
    auto tr_a = summarize(samples_to_a);
    emit_json(a, &tr_b, nullptr, &pe, nullptr, *out, "a_to_b");
    emit_json(a, &tr_a, nullptr, &pe, nullptr, *out, "b_to_a");
    return 0;
  }

  // multi_toggle: rotate through N distinct PDIs. Distinguishes
  // "firmware caches ≥ K PDIs" from "load_pdi is unconditionally a cheap
  // pointer swap" — if there is a fixed-size cache of size K, per-dispatch
  // latency should jump when N > K. Emits one JSON row per rotation
  // position so we can see if any specific PDI is more expensive than the
  // others (would suggest LRU eviction effects).
  if (a.metric == "multi_toggle") {
    if (!e_family) {
      std::cerr << "multi_toggle requires an ELF-family mechanism; got "
                << a.mechanism << "\n";
      return 2;
    }
    if (a.n_configs < 2) {
      std::cerr << "multi_toggle requires --n-configs=N with N >= 2\n";
      return 2;
    }
    PathE pe;
    pe.build_warm_multi(a, a.n_configs, total_bytes);

    // Warmup: full rotations to settle any cache.
    for (int w = 0; w < a.warmup; ++w) pe.dispatch_to_k(w % a.n_configs);

    // Each "iter" = one full rotation through N PDIs. Record per-position
    // samples so each PDI's latency is tracked separately.
    std::vector<std::vector<long long>> per_slot(a.n_configs);
    for (int k = 0; k < a.n_configs; ++k) per_slot[k].reserve(a.iters);
    for (int i = 0; i < a.iters; ++i) {
      for (int k = 0; k < a.n_configs; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        pe.dispatch_to_k(k);
        auto t1 = std::chrono::steady_clock::now();
        per_slot[k].push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count());
      }
    }

    auto summarize = [](std::vector<long long> &s) {
      TimingResult r;
      r.samples_ns = s;
      auto sorted = s;
      std::sort(sorted.begin(), sorted.end());
      r.min_ns = sorted.front();
      r.max_ns = sorted.back();
      long long sum = 0; for (auto v : sorted) sum += v;
      r.avg_ns = sum / static_cast<long long>(sorted.size());
      auto pct = [&](double p) {
        return sorted[static_cast<size_t>(p * (sorted.size() - 1))];
      };
      r.p50_ns = pct(0.50); r.p90_ns = pct(0.90); r.p99_ns = pct(0.99);
      return r;
    };
    for (int k = 0; k < a.n_configs; ++k) {
      auto tr = summarize(per_slot[k]);
      std::string dir = "slot_" + std::to_string(k);
      emit_json(a, &tr, nullptr, &pe, nullptr, *out, dir.c_str());
    }
    return 0;
  }

  // warm_reconfig / pure_dispatch — same code path, differ only in what the
  // build artifact already does. The reconfig vs dispatch distinction lives
  // in the txn binary / ELF, not in how the host calls kernel().
  TimingResult tr;
  if (x_family) {
    PathX px;
    if (a.batched && a.vary_args) {
      px.build_warm_vary_args(a, total_bytes, a.batch_size);
      tr = run_timed(a.warmup, a.iters,
                     [&]{ (void)px.dispatch_batched_vary(a.batch_size); });
    } else if (a.batched) {
      px.build_warm(a, total_bytes);
      tr = run_timed(a.warmup, a.iters,
                     [&]{ (void)px.dispatch_batched(a.batch_size); });
    } else {
      px.build_warm(a, total_bytes);
      tr = run_timed(a.warmup, a.iters, [&]{ px.dispatch_once(); });
    }
    emit_json(a, &tr, &px, nullptr, nullptr, *out);
  } else if (c_family) {
    PathC pc;
    pc.build_warm(a, total_bytes);
    if (a.ctrlpkt_strategy == "fresh_ctx") {
      tr = run_timed(a.warmup, a.iters,
                     [&]{ pc.dispatch_fresh_context(a); });
    } else if (a.ctrlpkt_strategy == "fresh_kernel") {
      tr = run_timed(a.warmup, a.iters,
                     [&]{ pc.dispatch_fresh_kernel(); });
    } else if (a.ctrlpkt_strategy == "fresh_module") {
      tr = run_timed(a.warmup, a.iters,
                     [&]{ pc.dispatch_fresh_module(); });
    } else {
      tr = run_timed(a.warmup, a.iters, [&]{ pc.dispatch_once(); });
    }
    emit_json(a, &tr, nullptr, nullptr, &pc, *out);
  } else {
    PathE pe;
    pe.build_warm(a, total_bytes);
    if (a.batched && a.vary_args) {
      pe.prep_vary_args(a.batch_size);
      tr = run_timed(a.warmup, a.iters,
                     [&]{ (void)pe.dispatch_batched_vary(a.batch_size); });
    } else if (a.batched) {
      tr = run_timed(a.warmup, a.iters,
                     [&]{ (void)pe.dispatch_batched(a.batch_size); });
    } else {
      tr = run_timed(a.warmup, a.iters, [&]{ pe.dispatch_once(); });
    }
    emit_json(a, &tr, nullptr, &pe, nullptr, *out);
  }
  return 0;
}
