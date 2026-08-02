//===- test.cpp - XRT host for matmul_int8_sparse -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs matmul_int8_sparse on npu3 hardware through XRT.
// C[4x64] += A[4x128] * B_sparse[128x64] at 4:2 structured sparsity.
//
// Operand convention (AIE4 ISA Spec v1.1, sections 2.7.2.5 and 14.1.5):
//   * A is DENSE: 128 logical K values per row.
//   * B is STORED compressed: 64 slots x 64 columns, column-major, 64 bytes
//     per output column.  Column n's logical K quartet q keeps exactly two of
//     its four positions; those land in stored slots 2*q and 2*q+1.
//   * The sparsity register selects which two, with a 3-bit code PER COLUMN
//     PER QUARTET (64 x 32 x 3 bits = 6 Kbit), laid out as six 1-Kbit row
//     slices of 16 bits per column.
//
// Because the selection is per column, the test uses an independent random
// 4:2 pattern for every (column, quartet) pair and checks a fully general
// golden reference against it.
//
// The compute tile has only two input DMA channels, so A and the sparsity
// image are concatenated into a single AM buffer (A's 512 bytes followed by
// the 768-byte mask) and split again by matmul_hw in kernel.cc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr unsigned M = 4;           // A rows
constexpr unsigned N = 64;          // output columns
constexpr unsigned K_LOGICAL = 128; // dense K positions per A row
constexpr unsigned QUARTETS = K_LOGICAL / 4; // 32
constexpr unsigned K_KEPT = QUARTETS * 2;    // 64 stored slots per column

constexpr size_t A_BYTES = M * K_LOGICAL;            // 512
constexpr size_t MASK_WORDS = 192;                   // 6 Kbit
constexpr size_t MASK_BYTES = MASK_WORDS * sizeof(uint32_t); // 768
constexpr size_t AM_BYTES = A_BYTES + MASK_BYTES;    // 1280
constexpr size_t B_BYTES = K_KEPT * N;               // 4096
constexpr size_t C_BYTES = M * N * sizeof(int32_t);  // 1024

// The six 4:2 quartet selections and their 3-bit encodings (ISA 14.1.5).
// Anything outside 001..101 decodes to x[0],x[1].
struct Selection {
  uint8_t code;
  uint8_t pos[2];
};
static const Selection kSelections[6] = {
    {0b001, {0, 2}}, {0b010, {0, 3}}, {0b011, {1, 2}},
    {0b100, {1, 3}}, {0b101, {2, 3}}, {0b000, {0, 1}},
};

// Byte offset of stored element (slot, column) in the 4096-byte staging
// image.  The staging matrix is column-major: 64 contiguous bytes per column.
static inline size_t stage_byte(unsigned slot, unsigned n) {
  return n * 64 + slot;
}

// Set column n's quartet q to a 3-bit code inside the 6-Kbit register image.
// Column n owns a flat 96-bit field (quartet q at bits 3q..3q+2).  That field
// is sliced into six 16-bit rows; row r is a 1-Kbit plane holding 16 bits for
// each of the 64 columns, packed two columns per 32-bit word.
static void set_sparsity_code(uint32_t *mask, unsigned n, unsigned q,
                              uint8_t code) {
  for (unsigned i = 0; i < 3; i++) {
    if (!((code >> i) & 1))
      continue;
    const unsigned bit = 3 * q + i; // 0..95 within the column's field
    const unsigned row = bit / 16;  // which 1-Kbit row slice
    const unsigned off = bit % 16;  // bit within the column's 16-bit chunk
    const unsigned word = n / 2;    // two columns per 32-bit word
    const unsigned half = n % 2;
    mask[row * 32 + word] |= 1u << (half * 16 + off);
  }
}

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "matmul_int8_sparse.elf";

  // Inputs bounded so the int32 accumulator stays comfortable:
  //   |A| <= 8, |B| <= 8, 64 kept terms => |C| <= 4096.
  const unsigned seed = (argc > 2) ? std::stoul(argv[2]) : 42u;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dis(-8, 8);
  std::uniform_int_distribution<int> sel_dis(0, 5);

  std::vector<int8_t> A_dense(M * K_LOGICAL);
  for (size_t i = 0; i < A_dense.size(); i++)
    A_dense[i] = static_cast<int8_t>(dis(gen));

  // B in stored form: [column][slot].
  std::vector<int8_t> B_stored_logical(N * K_KEPT);
  for (size_t i = 0; i < B_stored_logical.size(); i++)
    B_stored_logical[i] = static_cast<int8_t>(dis(gen));

  // Independent 4:2 selection per (column, quartet).
  std::vector<uint8_t> sel(N * QUARTETS);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < QUARTETS; q++)
      sel[n * QUARTETS + q] = static_cast<uint8_t>(sel_dis(gen));

  std::vector<uint32_t> mask(MASK_WORDS, 0);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < QUARTETS; q++)
      set_sparsity_code(mask.data(), n, q,
                        kSelections[sel[n * QUARTETS + q]].code);

  // Golden: for each column, each quartet contributes its two selected
  // logical activations against the two stored B elements of that quartet.
  std::vector<int32_t> C_ref(M * N, 0);
  for (unsigned m = 0; m < M; m++)
    for (unsigned n = 0; n < N; n++) {
      int32_t acc = 0;
      for (unsigned q = 0; q < QUARTETS; q++) {
        const Selection &s = kSelections[sel[n * QUARTETS + q]];
        for (unsigned i = 0; i < 2; i++) {
          const unsigned k = 4 * q + s.pos[i];
          const unsigned slot = 2 * q + i;
          acc += static_cast<int32_t>(A_dense[m * K_LOGICAL + k]) *
                 static_cast<int32_t>(B_stored_logical[n * K_KEPT + slot]);
        }
      }
      C_ref[m * N + n] = acc;
    }

  // Pack B into the column-major staging image.
  std::vector<int8_t> B_stored(B_BYTES);
  for (unsigned n = 0; n < N; n++)
    for (unsigned slot = 0; slot < K_KEPT; slot++)
      B_stored[stage_byte(slot, n)] = B_stored_logical[n * K_KEPT + slot];

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_am = xrt::ext::bo{device, AM_BYTES};
  xrt::bo bo_b = xrt::ext::bo{device, B_BYTES};
  xrt::bo bo_c = xrt::ext::bo{device, C_BYTES};

  // AM = [A 512 B | sparsity image 768 B]; matmul_hw re-splits it on the core.
  int8_t *buf_am = bo_am.map<int8_t *>();
  std::memcpy(buf_am, A_dense.data(), A_BYTES);
  std::memcpy(buf_am + A_BYTES, mask.data(), MASK_BYTES);

  std::memcpy(bo_b.map<int8_t *>(), B_stored.data(), B_BYTES);

  int32_t *buf_c = bo_c.map<int32_t *>();
  for (unsigned i = 0; i < M * N; i++)
    buf_c[i] = static_cast<int32_t>(0xdeadbeef);

  bo_am.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_am, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (unsigned i = 0; i < M * N; i++) {
    if (buf_c[i] != C_ref[i]) {
      if (errors < 16)
        std::cout << "  mismatch @[" << i / N << "][" << i % N
                  << "] got=" << buf_c[i] << " ref=" << C_ref[i] << "\n";
      errors++;
    }
  }

  if (errors) {
    std::cout << "FAIL (" << errors << " / " << (M * N) << " mismatches)"
              << std::endl;
    return 1;
  }
  std::cout << "PASS (all " << (M * N) << " outputs match host reference)"
            << std::endl;
  return 0;
}
