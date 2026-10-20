//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host side of the Tier-1 cert register-op hardware test. The buffer is
// pre-filled with a sentinel so a design that never DMAs anything back cannot
// accidentally pass, and the words the cert job does not touch are checked too
// so a stray / off-by-one uC write shows up.
//
// Every expected value below differs from BOTH the host sentinel (0xDEADBEEF)
// and the zero that device configuration leaves in core-tile data memory, so
// no assertion here can be satisfied by a no-op: an op that does nothing
// leaves 0, and a DMA that does nothing leaves 0xDEADBEEF. Both are reported
// with distinct messages ("unchanged sentinel" vs "wrong value").
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

constexpr size_t DATA_COUNT = 16;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(uint32_t);

// Host sentinel: never written by the design, so seeing it come back means the
// DMA did not cover that word.
constexpr uint32_t SENTINEL = 0xdeadbeefu;

// Immediates the cert job pokes into tile(0,2) data memory at 0x600400 /
// 0x600404 via WRITE_32.
constexpr uint32_t VAL0 = 0x11112222u;
constexpr uint32_t VAL1 = 0x33334444u;

// Register round-trips, all landed in obuf by WRITE_32_D.
//   obuf[2]: READ_32 of obuf[0]   -> $r0 -> WRITE_32_D flags 2
//   obuf[3]: READ_32_D of obuf[1] -> $r2 -> WRITE_32_D flags 2
//   obuf[4]: $r1 (obuf[1] pointer) at $r3 -> WRITE_32_D flags 3
//   obuf[5]: immediate at $r4           -> WRITE_32_D flags 1
//   obuf[6]: immediate at immediate     -> WRITE_32_D flags 0
//   obuf[7]: READ_32 of lock 1's value register, which WRITE_32_D (flags 1)
//            had just changed from 5 to 21 -> WRITE_32_D flags 2
//   obuf[8..11]: pointer table (WRITE_32), READ_32 into $r1/$r3/$r4/$r5
constexpr uint32_t READ32_ECHO = VAL0;         // 0x11112222
constexpr uint32_t READ32D_ECHO = VAL1;        // 0x33334444
constexpr uint32_t BOTHREG = 0x00600404u;      // $r1, the obuf[1] pointer
constexpr uint32_t W32D_ADDRREG = 0x55556666u; // flags 1, immediate value
constexpr uint32_t W32D_BOTHIMM = 0x77778888u; // flags 0
constexpr uint32_t LOCK_ECHO = 21u;            // set by WRITE_32_D over a 5
constexpr uint32_t PTR_TABLE[] = {0x00600404u, 0x00600410u, 0x00600414u,
                                  0x006ae010u};
constexpr size_t PTR_TABLE_BASE = 8;
constexpr size_t PTR_TABLE_COUNT = sizeof(PTR_TABLE) / sizeof(PTR_TABLE[0]);

// Device configuration zeroes core-tile data memory before the cert job runs,
// so every word the job leaves alone must read back as 0.
constexpr uint32_t UNTOUCHED = 0x00000000u;

int check(const char *what, uint32_t got, uint32_t expected, const char *why) {
  if (got == expected) {
    printf("%-34s got 0x%08x  expected 0x%08x  PASS\n", what, got, expected);
    return 0;
  }
  const char *how = (got == SENTINEL)
                        ? "unchanged sentinel -- the DMA never covered this "
                          "word, or the uC never got here"
                        : "wrong value";
  printf("%-34s got 0x%08x  expected 0x%08x  FAIL  <- %s: %s\n", what, got,
         expected, how, why);
  return 1;
}

} // namespace

int main(int argc, const char *argv[]) {
  const char *elf_name = (argc > 1) ? argv[1] : "cert_read_write32d.elf";

  auto device = xrt::device(0);
  xrt::elf ctx_elf{elf_name};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  uint32_t *buf_out = bo_out.map<uint32_t *>();
  for (size_t i = 0; i < DATA_COUNT; i++)
    buf_out[i] = SENTINEL;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Bounded wait: a cert job that wedges its microcontroller (or a poll that
  // never matches) must report a clean failure rather than hang the shell.
  auto run = kernel(bo_out);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "FAIL: run did not complete within 10s. The cert job never "
                 "released the gate lock -- either a poll never matched or an "
                 "opcode wedged the microcontroller."
              << std::endl;
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errs = 0;
  errs += check("obuf[0] write32 imm", buf_out[0], VAL0,
                "uC WRITE_32 into tile(0,2) DM 0x600400 did not land");
  errs += check("obuf[1] write32 imm", buf_out[1], VAL1,
                "uC WRITE_32 into tile(0,2) DM 0x600404 did not land");
  errs += check("obuf[2] read32 -> w32d f2", buf_out[2], READ32_ECHO,
                "cert.read32 did not return the word at 0x600400, or "
                "cert.write32_d {value_is_reg} echoed the wrong register");
  errs += check("obuf[3] read32_d -> w32d f2", buf_out[3], READ32D_ECHO,
                "cert.read32_d did not return the word at the address held in "
                "$r1 (0x600404)");
  errs += check("obuf[4] w32d f3 (both reg)", buf_out[4], BOTHREG,
                "cert.write32_d {address_is_reg, value_is_reg} misresolved a "
                "register, or cert.read32 did not load the pointer table");
  errs += check("obuf[5] w32d f1 (addr reg)", buf_out[5], W32D_ADDRREG,
                "cert.write32_d {address_is_reg} did not store the immediate "
                "at the address held in $r4 (0x600414)");
  errs += check("obuf[6] w32d f0 (both imm)", buf_out[6], W32D_BOTHIMM,
                "cert.write32_d with no flags did not behave like WRITE_32");
  errs += check("obuf[7] lock ctrl round-trip", buf_out[7], LOCK_ECHO,
                "cert.write32_d into tile CONTROL space did not take effect "
                "(a 5 here means the earlier plain WRITE_32 stuck), or "
                "cert.read32 did not read the lock value register back");
  for (size_t i = 0; i < PTR_TABLE_COUNT; i++) {
    char label[64];
    snprintf(label, sizeof(label), "obuf[%zu] pointer table",
             PTR_TABLE_BASE + i);
    errs += check(label, buf_out[PTR_TABLE_BASE + i], PTR_TABLE[i],
                  "uC WRITE_32 of a pointer-table entry did not land");
  }
  for (size_t i = PTR_TABLE_BASE + PTR_TABLE_COUNT; i < DATA_COUNT; i++) {
    char label[64];
    snprintf(label, sizeof(label), "obuf[%zu] untouched", i);
    errs += check(label, buf_out[i], UNTOUCHED,
                  "stray uC write, or the DMA did not cover this word");
  }

  // Reaching this point at all means every cert.poll32 / cert.maskpoll32 in
  // the job matched. Unlike earlier revisions of this test, the register ops
  // are no longer merely "seen not to fault": obuf[2..7] carry cert register
  // contents back to the host through cert.write32_d, so the checks above
  // fail if cert.read32 / cert.read32_d produce the wrong value, and none of
  // them can be satisfied by a no-op.
  std::cout << "cert.poll32 / cert.maskpoll32 matched (job completed)\n";
  std::cout << "cert.read32 / cert.read32_d VALUES verified via "
               "cert.write32_d (all 4 flags encodings)\n";

  if (errs == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << "fail (" << errs << ")." << std::endl;
  return 1;
}
