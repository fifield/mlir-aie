//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host side of the Tier-1 `cert.yield` hardware test.
//
// The load-bearing assertion is obuf[0]: the word job A read on the
// instruction immediately after its YIELD. It can only hold job B's value if
// the YIELD really switched to job B, B really ran to completion, and control
// really came back to A.
//
// Every word is classified so a failure says what broke rather than just
// "mismatch":
//
//   0xDEADBEEF  unchanged sentinel  host pre-fill survived: the readback DMA
//                                   never covered this word, or job A never
//                                   released the gate lock
//   0x0BAD0001  poison              job A's own poke survived into its
//                                   post-yield sample: the YIELD did not cede
//                                   control to job B
//   0xA1A1A1A1  job A marker
//   0xB1B1B1B1  job B's real value  written to the rendezvous word
//   0xB2B2B2B2  job B marker
//   0x00000000  zero                nothing wrote here at all
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

// Host pre-fill: never written by the design.
constexpr uint32_t SENTINEL = 0xdeadbeefu;

// Device configuration leaves untouched core-tile data memory at zero.
constexpr uint32_t UNTOUCHED = 0x00000000u;

// Job A pokes this into the rendezvous word before yielding.
constexpr uint32_t POISON = 0x0bad0001u;

constexpr uint32_t MARK_A = 0xa1a1a1a1u; // "job A reached the yield"
constexpr uint32_t MARK_B = 0xb2b2b2b2u; // "job B ran"
constexpr uint32_t REAL_B = 0xb1b1b1b1u; // job B's value for the rendezvous word

// Buffer map, mirroring the header of cert_yield.mlir.
constexpr size_t POST_YIELD = 0; // A's sample of T taken after the YIELD
constexpr size_t A_MARKER = 1;
constexpr size_t B_MARKER = 2;
constexpr size_t RENDEZVOUS = 3; // T itself
constexpr size_t PRE_YIELD = 4;  // A's sample of T taken before the YIELD
constexpr size_t FIRST_UNUSED = 5;

const char *classify(uint32_t got) {
  switch (got) {
  case SENTINEL:
    return "unchanged sentinel -- the readback DMA never covered this word, or "
           "job A never released the gate lock";
  case POISON:
    return "poison 0x0BAD0001 -- job A's own pre-yield poke is still there: "
           "cert.yield did not cede control to job B";
  case MARK_A:
    return "job A marker";
  case MARK_B:
    return "job B marker";
  case REAL_B:
    return "job B's rendezvous value";
  case UNTOUCHED:
    return "zero -- nothing wrote this word at all";
  default:
    return "wrong value -- unexpected word, no op in this design produces it";
  }
}

int errs = 0;
int checked = 0;

void check(const std::string &what, uint32_t got, uint32_t expected,
           const char *why) {
  checked++;
  if (got == expected)
    return;
  errs++;
  printf("%-34s got 0x%08x  expected 0x%08x  FAIL  <- %s: %s\n", what.c_str(),
         got, expected, classify(got), why);
}

std::string idx(const char *base, size_t i) {
  char b[64];
  snprintf(b, sizeof(b), "%s[%zu]", base, i);
  return b;
}

} // namespace

int main(int argc, const char *argv[]) {
  const char *elf_name = (argc > 1) ? argv[1] : "cert_yield.elf";

  auto device = xrt::device(0);
  xrt::elf ctx_elf{elf_name};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  uint32_t *buf_out = bo_out.map<uint32_t *>();
  for (size_t i = 0; i < DATA_COUNT; i++)
    buf_out[i] = SENTINEL;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Bounded wait: a YIELD that never returns control to job A must report a
  // clean failure rather than hang the shell.
  auto run = kernel(bo_out);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "FAIL: run did not complete within 10s. Job A never released "
                 "the gate lock -- control never came back from the "
                 "cert.yield, or an opcode wedged the microcontroller."
              << std::endl;
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  printf("obuf[0] post-yield sample          0x%08x\n", buf_out[POST_YIELD]);
  printf("obuf[1] job A marker               0x%08x\n", buf_out[A_MARKER]);
  printf("obuf[2] job B marker               0x%08x\n", buf_out[B_MARKER]);
  printf("obuf[3] rendezvous word (final)    0x%08x\n", buf_out[RENDEZVOUS]);
  printf("obuf[4] pre-yield sample           0x%08x\n", buf_out[PRE_YIELD]);

  // Liveness first: if job A did not run at all, everything below is noise.
  check("obuf[1] job A ran", buf_out[A_MARKER], MARK_A,
        "cert.write32 from job A before the yield");

  // The poison really was in place when the yield executed.
  check("obuf[4] pre-yield sample is poison", buf_out[PRE_YIELD], POISON,
        "cert.read32 of the rendezvous word taken before cert.yield -- if this "
        "is not the poison, obuf[0] proves nothing");

  // Job B ran at all.
  check("obuf[2] job B ran", buf_out[B_MARKER], MARK_B,
        "cert.write32 from job B: the lexically later job on the same page "
        "never executed");

  // THE assertion: what job A saw the instruction after its YIELD.
  check("obuf[0] post-yield sample", buf_out[POST_YIELD], REAL_B,
        "cert.read32 of the rendezvous word on the instruction right after "
        "cert.yield -- job B's value here is the proof that the yield context "
        "switched to B, B ran to completion, and control returned to A");

  // And the word itself ended up holding job B's value.
  check("obuf[3] rendezvous word", buf_out[RENDEZVOUS], REAL_B,
        "job B's cert.write32 to the rendezvous word");

  for (size_t i = FIRST_UNUSED; i < DATA_COUNT; i++)
    check(idx("obuf", i), buf_out[i], UNTOUCHED,
          "outside every write in this design: a stray uC write happened");

  std::cout << "checked " << checked << " words\n";
  if (errs == 0) {
    std::cout << "cert.yield context switched from job 7 to job 8 and control "
                 "returned to job 7\n"
              << "PASS!" << std::endl;
    return 0;
  }
  std::cout << "fail (" << errs << ")." << std::endl;
  return 1;
}
