//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host side of the Tier-1 cert uC-DMA hardware test. The uC-DMA destination
// ranges live *inside* the buffer that is DMAed back, so the words asserted
// here are the bytes the uC-DMA actually moved.
//
// Every word is classified into one of four outcomes so a failure says what
// broke rather than just "mismatch":
//
//   0xDEADBEEF  unchanged sentinel  the host pre-fill survived: the readback
//                                   DMA never covered this word (or the job
//                                   never released the gate)
//   0x0BAD000N  poison              the cert job's pre-transfer poke survived:
//                                   uC-DMA transfer N never landed here
//   0x00000000  zero                nothing at all wrote here; expected for
//                                   out-of-range words, a bug for in-range
//   anything    wrong value         something transferred, but corrupted /
//               else                misaligned / from the wrong chain
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

constexpr size_t DATA_COUNT = 512;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(uint32_t);

// Host pre-fill: never written by the design.
constexpr uint32_t SENTINEL = 0xdeadbeefu;

// Device configuration leaves untouched core-tile data memory at zero.
constexpr uint32_t UNTOUCHED = 0x00000000u;

// Poison the cert job pokes into the first word of each destination before
// running that transfer.
constexpr uint32_t POISON_BD0 = 0x0bad0001u;
constexpr uint32_t POISON_BD1 = 0x0bad0002u;
constexpr uint32_t POISON_SYNC = 0x0bad0003u;

// Payload contents (must match the memref.globals in cert_ucdma.mlir).
constexpr uint32_t BIG_WORD = 0xa5a5a501u; // @ucdma_big, 256-word splat
constexpr uint32_t TAIL_BASE = 0x11110000u;  // @ucdma_tail, +1..+16
constexpr uint32_t SYNC_BASE = 0x22220000u;  // @ucdma_sync, +1..+16

// Buffer map, mirroring the header of cert_ucdma.mlir.
constexpr size_t ECHO_BD0 = 0;
constexpr size_t ECHO_BD1 = 1;
constexpr size_t ECHO_SYNC = 2;
constexpr size_t BD0_BEGIN = 64, BD0_LEN = 256;   // 0x600500
constexpr size_t BD1_BEGIN = 320, BD1_LEN = 16;   // 0x600900
constexpr size_t SYNC_BEGIN = 384, SYNC_LEN = 16; // 0x600A00

const char *classify(uint32_t got) {
  switch (got) {
  case SENTINEL:
    return "unchanged sentinel -- the readback DMA never covered this word, or "
           "the cert job never released the gate lock";
  case POISON_BD0:
    return "poison 0x0BAD0001 -- @chain_async BD0 never wrote here "
           "(uc_dma_write_des/wait_uc_dma did not complete)";
  case POISON_BD1:
    return "poison 0x0BAD0002 -- @chain_async BD1 never wrote here (next_bd "
           "chain continuation did not happen)";
  case POISON_SYNC:
    return "poison 0x0BAD0003 -- @chain_sync never wrote here "
           "(uc_dma_write_des_sync did not complete)";
  case UNTOUCHED:
    return "zero -- nothing wrote this word at all";
  default:
    return "wrong value -- something transferred here but it is corrupted, "
           "misaligned, or came from the wrong chain";
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
  // Cap the noise from a wholesale failure; the first 24 tell the story.
  if (errs <= 24)
    printf("%-30s got 0x%08x  expected 0x%08x  FAIL  <- %s: %s\n", what.c_str(),
           got, expected, classify(got), why);
  else if (errs == 25)
    printf("... further mismatches suppressed ...\n");
}

std::string idx(const char *base, size_t i) {
  char b[64];
  snprintf(b, sizeof(b), "%s[%zu]", base, i);
  return b;
}

} // namespace

int main(int argc, const char *argv[]) {
  const char *elf_name = (argc > 1) ? argv[1] : "cert_ucdma.elf";

  auto device = xrt::device(0);
  xrt::elf ctx_elf{elf_name};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  uint32_t *buf_out = bo_out.map<uint32_t *>();
  for (size_t i = 0; i < DATA_COUNT; i++)
    buf_out[i] = SENTINEL;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Bounded wait: a uC-DMA that never completes (or a poll that never
  // matches) must report a clean failure rather than hang the shell.
  auto run = kernel(bo_out);
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "FAIL: run did not complete within 10s. The cert job never "
                 "released the gate lock -- a uC-DMA transfer never landed "
                 "(cert.maskpoll32 spinning) or an opcode wedged the "
                 "microcontroller."
              << std::endl;
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // --- the transferred payloads themselves --------------------------------
  // @chain_async BD0: 256-word splat at obuf[64..319].
  for (size_t i = 0; i < BD0_LEN; i++)
    check(idx("obuf", BD0_BEGIN + i), buf_out[BD0_BEGIN + i], BIG_WORD,
          "cert.uc_dma_write_des(@chain_async) BD0 payload");

  // @chain_async BD1: reached only if next_bd=true on BD0 continues the chain.
  for (size_t i = 0; i < BD1_LEN; i++)
    check(idx("obuf", BD1_BEGIN + i), buf_out[BD1_BEGIN + i],
          TAIL_BASE + uint32_t(i + 1),
          "cert.uc_dma_write_des(@chain_async) BD1 payload (chain "
          "continuation via next_bd)");

  // @chain_sync: the standalone blocking op.
  for (size_t i = 0; i < SYNC_LEN; i++)
    check(idx("obuf", SYNC_BEGIN + i), buf_out[SYNC_BEGIN + i],
          SYNC_BASE + uint32_t(i + 1),
          "cert.uc_dma_write_des_sync(@chain_sync) payload");

  // --- uC-visible echoes, sampled right after each wait -------------------
  // These are what the microcontroller itself saw at the tail of each block
  // before any poll ran, so they check ordering, not just final contents.
  check("obuf[0] echo BD0 tail", buf_out[ECHO_BD0], BIG_WORD,
        "cert.read32 of obuf[319] right after cert.wait_uc_dma");
  check("obuf[1] echo BD1 tail", buf_out[ECHO_BD1], TAIL_BASE + 16,
        "cert.read32 of obuf[335] right after cert.wait_uc_dma");
  check("obuf[2] echo sync tail", buf_out[ECHO_SYNC], SYNC_BASE + 16,
        "cert.read32 of obuf[399] right after cert.uc_dma_write_des_sync");

  // --- everything else must be untouched ----------------------------------
  auto in_range = [](size_t i) {
    return i <= ECHO_SYNC || (i >= BD0_BEGIN && i < BD0_BEGIN + BD0_LEN) ||
           (i >= BD1_BEGIN && i < BD1_BEGIN + BD1_LEN) ||
           (i >= SYNC_BEGIN && i < SYNC_BEGIN + SYNC_LEN);
  };
  for (size_t i = 0; i < DATA_COUNT; i++)
    if (!in_range(i))
      check(idx("obuf", i), buf_out[i], UNTOUCHED,
            "outside every uC-DMA destination: a transfer overran its length, "
            "landed at the wrong address, or a stray uC write happened");

  std::cout << "checked " << checked << " words\n";
  if (errs == 0) {
    std::cout
        << "cert.uc_dma_write_des + cert.wait_uc_dma moved 256+16 words "
           "across a 2-BD chain\n"
        << "cert.uc_dma_write_des_sync moved 16 words across a 1-BD chain\n"
        << "next_bd continuation verified (BD1 block landed)\n"
        << "PASS!" << std::endl;
    return 0;
  }
  std::cout << "fail (" << errs << ")." << std::endl;
  return 1;
}
