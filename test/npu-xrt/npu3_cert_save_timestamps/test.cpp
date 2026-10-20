//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host side of the `aiex.cert.save_timestamps` hardware test.
//
// Firmware fills the uC debug buffer with 12-byte `{unq_id, ts_high, ts_low}`
// records. This checks three things about them:
//
//   (a) all four unq_ids appear, in program order, starting at word 0;
//   (b) the 64-bit timestamps strictly increase;
//   (c) the gap that brackets 128 `cert.read32`s dwarfs the two adjacent
//       no-work gaps.
//
// (c) is asserted as a *ratio*, never as an absolute cycle count: the tick
// counter is a free-running clock and the absolute values differ every run.
//
// Run with `--no-config` for the negative control: identical, except that
// `config_bo` is never called, and the test then asserts the debug buffer
// comes back entirely poison.
//
//===----------------------------------------------------------------------===//
//
// HOW THE HOST GETS AT THE DEBUG BUFFER  (new pattern -- read this)
//
// The uC debug buffer is not an ordinary `xrt::bo`. It is allocated with a
// dedicated use type and then *registered* with the hardware context, which is
// what makes the driver program the firmware handshake block
// (`lx7_handshake.dbg_buf.{addr_hi, addr_lo, size}`) that
// `copy_dm2mm_for_timestamp` DMAs into. Without the registration step,
// `HOST_DBG_BUF_ADDR == 0` and firmware skips record collection entirely --
// which is exactly what `--no-config` exercises.
//
// Both entry points live in `core/common/api/bo_int.h` in the XRT source tree.
// That header is *internal*: it is not installed under
// /opt/xilinx/xrt/include, so it cannot simply be #included. The symbols
// themselves are exported from the installed libxrt_coreutil.so.2, so the two
// declarations below are reproduced verbatim to link against them.
//
//   $ nm -DC /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 | grep bo_int
//   ... T xrt_core::bo_int::create_bo(xrt::hw_context const&, unsigned long,
//                                     xrt_core::bo_int::use_type)
//   ... T xrt_core::bo_int::config_bo(xrt::bo const&,
//                                     std::map<unsigned int, unsigned long,
//                                       std::less<unsigned int>,
//                                       std::allocator<std::pair<unsigned int
//                                       const,
//                                         unsigned long>>> const&,
//                                     xrt_core::hwctx_handle const*)
//
// This is a deliberate, and fragile, coupling: the declarations participate in
// C++ name mangling, so the *entire* `use_type` enumerator list has to match
// the header (only the numeric value of `uc_debug` actually matters at
// runtime, but the enum's name is mangled into the symbol), and both signatures
// must match exactly. If XRT reorders `use_type`, renames it, or changes either
// signature, this test fails at *link* time with an undefined reference --
// loudly, not silently, which is the saving grace of the approach. If it ever
// does break, re-derive both from
// `<xrt-source>/src/runtime_src/core/common/api/bo_int.h`.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

//===----------------------------------------------------------------------===//
// Verbatim from XRT's internal core/common/api/bo_int.h -- see the note above.
//===----------------------------------------------------------------------===//
namespace xrt_core {
class hwctx_handle;
namespace bo_int {
enum class use_type {
  debug = 1,
  kmd = 2,
  dtrace = 3,
  log = 4,
  debug_queue = 5,
  uc_debug = 6,
  host_only = 7,
  instruction = 8,
  preemption = 9,
  scratch_pad = 10,
  ctrl_scratch_pad = 11,
  pdi = 12,
  ctrlpkt = 13
};
xrt::bo create_bo(const xrt::hw_context &hwctx, size_t sz, use_type type);
void config_bo(const xrt::bo &bo, const std::map<uint32_t, size_t> &buf_sizes,
               const xrt_core::hwctx_handle *ctx_handle);
} // namespace bo_int
} // namespace xrt_core
//===----------------------------------------------------------------------===//

namespace {

// 1 KiB is far more than the 48 bytes of records this design produces; the
// slack is checked for poison, so extra records would be caught.
constexpr size_t kDebugWords = 256;

// Column 0, uC A -- the design's only tiles are in column 0.
constexpr uint32_t kUcIndex = 0;

constexpr size_t kOutWords = 16;

constexpr uint32_t kDebugPoison = 0xa5a5a5a5u;
constexpr uint32_t kOutPoison = 0xdeadbeefu;

// Liveness marker the cert job drops in obuf[0].
constexpr uint32_t kMarker = 0xa5a5a5a5u;
constexpr uint32_t kUntouched = 0u;

// One SAVE_TIMESTAMPS record is {unq_id, ts_high, ts_low}.
constexpr size_t kRecordWords = 3;
constexpr std::array<uint32_t, 4> kIds{101, 102, 103, 104};

// Gap B brackets 128 non-posted AXI-MM reads (cert.read32). The two no-work
// gaps around it are back-to-back opcodes, measured at 260-290 ticks on npu3;
// each read costs ~95 ticks, so gap B is ~12000 ticks and the real ratio is
// around 45x. Requiring only 10x leaves margin for a noisy uC while still
// failing hard if the reads were compiled out.
constexpr uint64_t kMinWorkRatio = 10;

std::string hex32(uint32_t v) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0') << v;
  return os.str();
}

/// Prints one aligned PASS/FAIL line per assertion and tallies the failures.
class Report {
public:
  int errors() const { return errs; }

  void pass(std::string_view what, const std::string &detail) {
    line(what, detail, "PASS");
  }

  void fail(std::string_view what, const std::string &detail) {
    ++errs;
    line(what, detail, "FAIL");
  }

  /// Compares one 32-bit word, distinguishing "still poison, so nothing wrote
  /// here at all" from "written, but wrong".
  void word(std::string_view what, uint32_t got, uint32_t expected,
            uint32_t poison, std::string_view why) {
    std::ostringstream os;
    os << "got " << hex32(got) << "  expected " << hex32(expected);
    if (got == expected)
      return pass(what, os.str());
    os << "  <- "
       << (got == poison ? "unchanged poison, nothing wrote this word: "
                         : "wrong value: ")
       << why;
    fail(what, os.str());
  }

private:
  static constexpr int kLabelWidth = 46;

  void line(std::string_view what, const std::string &detail,
            std::string_view verdict) const {
    std::cout << std::left << std::setw(kLabelWidth) << what << std::right
              << detail << "  " << verdict << '\n';
  }

  int errs = 0;
};

/// The uC debug buffer: an XRT-internal BO plus the poison discipline and the
/// registration step that makes firmware write to it.
class DebugBuffer {
public:
  DebugBuffer(const xrt::hw_context &ctx, size_t words, uint32_t poison)
      : nwords(words), fill(poison),
        bo(xrt_core::bo_int::create_bo(ctx, words * sizeof(uint32_t),
                                       xrt_core::bo_int::use_type::uc_debug)),
        data(bo.map<uint32_t *>()) {
    std::fill_n(data, nwords, fill);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  /// Tells the driver which uC may write which slice of the buffer. Skipping
  /// this leaves HOST_DBG_BUF_ADDR at 0 and firmware collects nothing.
  void configure(uint32_t uc_index) {
    const std::map<uint32_t, size_t> segments{
        {uc_index, nwords * sizeof(uint32_t)}};
    xrt_core::bo_int::config_bo(bo, segments, nullptr);
  }

  void refresh() { bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); }

  size_t size() const { return nwords; }
  uint32_t poison() const { return fill; }
  uint32_t operator[](size_t i) const { return data[i]; }

  /// Index of the first word at or after `first` that is no longer poison, or
  /// size() if the tail is untouched.
  size_t first_changed(size_t first = 0) const {
    auto *it = std::find_if(data + first, data + nwords,
                            [this](uint32_t w) { return w != fill; });
    return static_cast<size_t>(it - data);
  }

  size_t count_changed() const {
    return static_cast<size_t>(std::count_if(
        data, data + nwords, [this](uint32_t w) { return w != fill; }));
  }

private:
  size_t nwords;
  uint32_t fill;
  xrt::bo bo;
  uint32_t *data;
};

struct Stamp {
  uint32_t id;
  uint64_t ticks;
};

struct Options {
  std::string elf{"cert_save_timestamps.elf"};
  bool configure_debug_buffer = true;
};

Options parse(int argc, const char *argv[]) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--no-config")
      opts.configure_debug_buffer = false;
    else
      opts.elf = argv[i];
  }
  return opts;
}

} // namespace

int main(int argc, const char *argv[]) try {
  const Options opts = parse(argc, argv);

  auto device = xrt::device(0);
  xrt::elf ctx_elf{opts.elf};
  xrt::hw_context context{device, ctx_elf};
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  DebugBuffer dbg{context, kDebugWords, kDebugPoison};
  if (opts.configure_debug_buffer) {
    dbg.configure(kUcIndex);
    std::cout << "config_bo(uc_debug, {{" << kUcIndex << ", "
              << kDebugWords * sizeof(uint32_t) << "}}) OK\n";
  } else {
    std::cout << "NEGATIVE CONTROL: config_bo deliberately not called; "
                 "firmware should see HOST_DBG_BUF_ADDR == 0 and collect "
                 "nothing\n";
  }

  xrt::bo bo_out = xrt::ext::bo{device, kOutWords * sizeof(uint32_t)};
  auto *out = bo_out.map<uint32_t *>();
  std::fill_n(out, kOutWords, kOutPoison);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(bo_out);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "FAIL: run did not complete within 10s. The cert job never "
                 "released the gate lock -- either a poll never matched or an "
                 "opcode wedged the microcontroller.\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  dbg.refresh();

  Report report;

  // --- the design ran, regardless of the debug path -------------------------
  report.word("obuf[0] liveness marker", out[0], kMarker, kOutPoison,
              "the cert job never ran to completion");
  for (size_t i = 1; i < kOutWords; ++i)
    report.word("obuf[" + std::to_string(i) + "] untouched", out[i], kUntouched,
                kOutPoison,
                "stray uC write, or the DMA did not cover this word");

  // --- negative control: no config_bo, so nothing may have been written -----
  if (!opts.configure_debug_buffer) {
    const size_t changed = dbg.count_changed();
    if (changed)
      report.fail("negative control",
                  std::to_string(changed) + " of " +
                      std::to_string(dbg.size()) +
                      " debug words changed without config_bo");
    else
      report.pass("negative control",
                  "all " + std::to_string(dbg.size()) + " words still " +
                      hex32(dbg.poison()));
    if (report.errors() == 0) {
      std::cout << "PASS! (negative control: no config_bo, no records)\n";
      return 0;
    }
    std::cout << "fail (" << report.errors() << ").\n";
    return 1;
  }

  // --- decode ---------------------------------------------------------------
  std::vector<Stamp> stamps;
  stamps.reserve(kIds.size());
  for (size_t r = 0; r < kIds.size(); ++r) {
    const size_t w = r * kRecordWords;
    stamps.push_back({dbg[w], (static_cast<uint64_t>(dbg[w + 1]) << 32) |
                                  dbg[w + 2]});
  }

  std::cout << "--- SAVE_TIMESTAMPS records, 12 bytes each: "
               "{unq_id, ts_hi, ts_lo} ---\n";
  for (size_t r = 0; r < stamps.size(); ++r)
    std::cout << "  rec " << r << ": id=" << stamps[r].id
              << " ticks=" << stamps[r].ticks << '\n';

  // (a) ids present, in order, starting at word 0.
  for (size_t r = 0; r < kIds.size(); ++r)
    report.word("rec " + std::to_string(r) + " unq_id", stamps[r].id, kIds[r],
                dbg.poison(),
                "records are missing, out of order, or start somewhere other "
                "than word 0 of the buffer");

  // Nothing beyond the records this design emits.
  const size_t tail = kIds.size() * kRecordWords;
  if (const size_t spurious = dbg.first_changed(tail); spurious < dbg.size())
    report.fail("no spurious records",
                "dbg[" + std::to_string(spurious) +
                    "] = " + hex32(dbg[spurious]) +
                    ", more debug data than this design emits");
  else
    report.pass("no spurious records", "words " + std::to_string(tail) + ".." +
                                           std::to_string(dbg.size() - 1) +
                                           " still " + hex32(dbg.poison()));

  // (b) strictly increasing. A stuck or zero counter fails here.
  const auto not_increasing =
      std::adjacent_find(stamps.begin(), stamps.end(),
                         [](const Stamp &a, const Stamp &b) {
                           return a.ticks >= b.ticks;
                         });
  if (not_increasing == stamps.end())
    report.pass("timestamps strictly increasing",
                std::to_string(stamps.front().ticks) + " .. " +
                    std::to_string(stamps.back().ticks));
  else
    report.fail("timestamps strictly increasing",
                "rec " +
                    std::to_string(not_increasing - stamps.begin()) +
                    " (" + std::to_string(not_increasing->ticks) +
                    ") is not before rec " +
                    std::to_string(not_increasing - stamps.begin() + 1) + " (" +
                    std::to_string(std::next(not_increasing)->ticks) +
                    "); the tick counter is stuck or the records are "
                    "misordered");

  // (c) the workload is measurable, expressed as a ratio.
  const uint64_t gap_a = stamps[1].ticks - stamps[0].ticks; // no work
  const uint64_t gap_b = stamps[2].ticks - stamps[1].ticks; // 128 x read32
  const uint64_t gap_c = stamps[3].ticks - stamps[2].ticks; // no work
  const uint64_t baseline = std::max(gap_a, gap_c);
  std::cout << "gaps: A(101->102)=" << gap_a << "  B(102->103, reads)=" << gap_b
            << "  C(103->104)=" << gap_c << "  ticks\n";

  std::ostringstream ratio;
  ratio << std::fixed << std::setprecision(0)
        << (baseline ? double(gap_b) / double(baseline) : 0.0)
        << "x the larger no-work gap (need >= " << kMinWorkRatio << "x)";
  if (baseline == 0)
    report.fail("workload is measurable",
                "both no-work gaps are 0 ticks; the counter has no usable "
                "resolution");
  else if (gap_b < baseline * kMinWorkRatio)
    report.fail("workload is measurable",
                ratio.str() + " -- the cert.read32s did not consume time, or "
                              "the "
                              "stamps are not where the program says");
  else
    report.pass("workload is measurable", ratio.str());

  if (report.errors() == 0) {
    std::cout << "PASS!\n";
    return 0;
  }
  std::cout << "fail (" << report.errors() << ").\n";
  return 1;
} catch (const std::exception &e) {
  // Covers bo_int::create_bo / config_bo, which throw rather than return an
  // error, and every XRT call around them.
  std::cout << "FAIL: " << e.what() << '\n';
  return 1;
}
