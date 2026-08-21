//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host side of the `aiex.cert.save_register` hardware test.
//
// Two buffers are checked against each other:
//
//   * the uC debug buffer, which firmware fills with 8-byte
//     `{unq_id, value}` SAVE_REGISTER records, and
//   * `obuf`, streamed back over a normal core-DMA -> shim-S2MM path,
//
// so the SAVE_REGISTER records are cross-checked against an independent view
// of the very same memory. Both buffers are poisoned before the run, with
// different poisons, so "nothing was written" is never mistaken for a pass and
// the two failure modes report differently.
//
// Run with `--no-config` for the negative control: everything is identical
// except that `config_bo` is never called, and the test then asserts that the
// debug buffer comes back *entirely* poison. That is what proves the records
// in the normal run really came from the device rather than from anything the
// host happened to leave in the mapping.
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
//                                       std::allocator<std::pair<unsigned int const,
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

// Debug buffer. 1 KiB is far more than the 32 bytes of records this design
// produces; the slack is checked for poison, so a firmware that emitted extra
// or oversized records would be caught.
constexpr size_t kDebugWords = 256;

// The uC the design's job runs on. Column 0, uC A -- the design's only tiles
// are in column 0.
constexpr uint32_t kUcIndex = 0;

constexpr size_t kOutWords = 16;

// Distinct poisons so the two readback paths cannot be confused for one
// another in a failure report.
constexpr uint32_t kDebugPoison = 0xa5a5a5a5u;
constexpr uint32_t kOutPoison = 0xdeadbeefu;

// What the cert job writes, in order.
constexpr uint32_t kValFirst = 0x11112222u;  // 0x600400, then overwritten
constexpr uint32_t kValOther = 0x33334444u;  // 0x600404
constexpr uint32_t kValSecond = 0x55556666u; // 0x600400, final
constexpr uint32_t kValLock = 21u;           // lock 1 value register

// Device configuration zeroes core-tile data memory, so words the job leaves
// alone must read back as 0.
constexpr uint32_t kUntouched = 0u;

// One SAVE_REGISTER record is {unq_id, value}.
constexpr size_t kRecordWords = 2;

struct Expected {
  uint32_t id;
  uint32_t value;
  std::string_view what;
  std::string_view why;
};

constexpr std::array kRecords{
    Expected{201, kValFirst, "0x600400 before overwrite",
             "SAVE_REGISTER of core-tile data memory did not return the word "
             "cert.write32 had just stored there"},
    Expected{202, kValOther, "0x600404",
             "SAVE_REGISTER ignored its address operand, or read the wrong "
             "word"},
    Expected{203, kValSecond, "0x600400 after overwrite",
             "SAVE_REGISTER did not resample memory -- it returned a stale "
             "value for an address it had already saved once"},
    Expected{204, kValLock, "lock 1 value reg, control space",
             "SAVE_REGISTER cannot reach tile control space, only data "
             "memory"},
};

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
    return static_cast<size_t>(
        std::count_if(data, data + nwords,
                      [this](uint32_t w) { return w != fill; }));
  }

private:
  size_t nwords;
  uint32_t fill;
  xrt::bo bo;
  uint32_t *data;
};

struct Options {
  std::string elf{"cert_save_register.elf"};
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
  // Checked first and always: it separates "SAVE_REGISTER produced nothing"
  // from "the job never executed".
  report.word("obuf[0] final contents of 0x600400", out[0], kValSecond,
              kOutPoison, "the cert job never ran, or never overwrote it");
  report.word("obuf[1] contents of 0x600404", out[1], kValOther, kOutPoison,
              "the cert job never ran");
  for (size_t i = 2; i < kOutWords; ++i)
    report.word("obuf[" + std::to_string(i) + "] untouched", out[i], kUntouched,
                kOutPoison,
                "stray uC write, or the DMA did not cover this word");

  // --- negative control: no config_bo, so nothing may have been written -----
  if (!opts.configure_debug_buffer) {
    const size_t changed = dbg.count_changed();
    if (changed)
      report.fail("negative control",
                  std::to_string(changed) + " of " + std::to_string(dbg.size()) +
                      " debug words changed without config_bo");
    else
      report.pass("negative control", "all " + std::to_string(dbg.size()) +
                                          " words still " + hex32(dbg.poison()));
    if (report.errors() == 0) {
      std::cout << "PASS! (negative control: no config_bo, no records)\n";
      return 0;
    }
    std::cout << "fail (" << report.errors() << ").\n";
    return 1;
  }

  // --- the records ----------------------------------------------------------
  std::cout << "--- SAVE_REGISTER records, 8 bytes each: {unq_id, value} ---\n";
  for (size_t r = 0; r < kRecords.size(); ++r)
    std::cout << "  rec " << r << ": id=" << dbg[r * kRecordWords]
              << " value=" << hex32(dbg[r * kRecordWords + 1]) << '\n';

  for (size_t r = 0; r < kRecords.size(); ++r) {
    const Expected &want = kRecords[r];
    const std::string tag = "rec " + std::to_string(r);
    report.word(tag + " unq_id", dbg[r * kRecordWords], want.id, dbg.poison(),
                "records are missing, out of order, or start somewhere other "
                "than word 0 of the buffer");
    report.word(tag + " value (" + std::string(want.what) + ")",
                dbg[r * kRecordWords + 1], want.value, dbg.poison(), want.why);
  }

  // Nothing beyond the records this design emits.
  const size_t tail = kRecords.size() * kRecordWords;
  if (const size_t spurious = dbg.first_changed(tail); spurious < dbg.size())
    report.fail("no spurious records",
                "dbg[" + std::to_string(spurious) + "] = " +
                    hex32(dbg[spurious]) +
                    ", more debug data than this design emits");
  else
    report.pass("no spurious records", "words " + std::to_string(tail) + ".." +
                                           std::to_string(dbg.size() - 1) +
                                           " still " + hex32(dbg.poison()));

  // --- the cross-check that makes this more than a self-consistent dump -----
  // Records 201 and 203 read the same address; 203 and obuf[0] are the same
  // location observed two entirely different ways.
  const uint32_t saved_before = dbg[1];
  const uint32_t saved_after = dbg[5];
  if (saved_before == saved_after)
    report.fail("live sampling",
                "records 201 and 203 read the same address and returned the "
                "same value (" + hex32(saved_before) +
                    "); SAVE_REGISTER is not resampling");
  else if (saved_after != out[0])
    report.fail("live sampling", "record 203 says " + hex32(saved_after) +
                                     " but the DMA says 0x600400 holds " +
                                     hex32(out[0]));
  else
    report.pass("live sampling", "rec 201=" + hex32(saved_before) + " != rec " +
                                     "203=" + hex32(saved_after) +
                                     " == obuf[0]");

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
