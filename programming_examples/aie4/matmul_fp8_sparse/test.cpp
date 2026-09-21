//===- test.cpp - XRT host for matmul_fp8_sparse --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs matmul_fp8_sparse on npu3 hardware through XRT.
// C[4x64] += A[4x128] * B_sparse[128x64], fp8 (E4M3) inputs, fp32 output,
// 4:2 structured sparsity.
//
// Operand convention (according to the AIE4 ISA):
//   * A is DENSE: 128 logical K values per row (the activation operand is a
//     v128), so A is 4 x 128 fp8 = 512 bytes.
//   * load_staging() consumes exactly 32 x v128 = 4096 bytes of B, i.e. a
//     64 x 64 STORED matrix, held COLUMN-major: stored slot k of output
//     column n lives at byte offset n * 64 + k.
//   * Sparsity is 4:2 per quartet: each group of 4 logical K positions keeps
//     exactly 2, giving 32 quartets x 2 = 64 stored slots per column.  Column
//     n's quartet q keeps two positions, landing in stored slots 2q and 2q+1.
//   * A 3-bit code PER COLUMN PER QUARTET selects which two, independently
//     for every column, so the sparsity register is 64 x 32 x 3 = 6 Kbit,
//     laid out as six 1-Kbit row slices of 16 bits per column.  The kernel
//     writes it with the two-argument load_staging(B, mask); the one-argument
//     insert_staging_sparsity() only fills one of the six slices.
//
// The test therefore uses an independent random 4:2 pattern for every
// (column, quartet) pair and checks a fully general golden against it.
//
// The compute tile has only two input DMA channels, so A and the 768-byte
// sparsity image are concatenated into a single AM buffer (A's 512 bytes
// followed by the mask) and split again by matmul_hw in kernel.cc.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <stdio.h>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "../cpu_ref/fp_types.h"

constexpr unsigned M = 4;         // rows of A / C
constexpr unsigned N = 64;        // columns of B / C
constexpr unsigned K = 128;       // logical (dense) K positions per A row
constexpr unsigned K_STORED = 64; // stored B rows after 4:2 compression
constexpr unsigned QUARTETS = K / 4;

constexpr unsigned A_BYTES = M * K;              // 512
constexpr unsigned MASK_WORDS = 192;             // 6 Kbit
constexpr unsigned MASK_BYTES = MASK_WORDS * 4;  // 768

// The six 4:2 quartet selections and their 3-bit encodings (ISA).
// Anything outside 001..101 decodes to x[0],x[1].
struct Selection {
  uint8_t code;
  uint8_t pos[2];
};
static const Selection kSelections[6] = {
    {0b001, {0, 2}}, {0b010, {0, 3}}, {0b011, {1, 2}},
    {0b100, {1, 3}}, {0b101, {2, 3}}, {0b000, {0, 1}},
};

// Set column n's quartet q to a 3-bit code inside the 6-Kbit register image.
// Column n owns a flat 96-bit field (quartet q at bits 3q..3q+2).  That field
// is sliced into six 16-bit rows; row r is a 1-Kbit plane holding 16 bits for
// each of the 64 columns, packed two columns per 32-bit word.
static void set_sparsity_code(uint32_t *mask, unsigned n, unsigned q,
                              uint8_t code) {
  for (unsigned i = 0; i < 3; i++) {
    if (!((code >> i) & 1))
      continue;
    const unsigned bit = 3 * q + i;
    const unsigned row = bit / 16;
    const unsigned off = bit % 16;
    mask[row * 32 + n / 2] |= 1u << ((n % 2) * 16 + off);
  }
}
constexpr unsigned AM_BYTES = A_BYTES + MASK_BYTES;
constexpr unsigned B_BYTES = K_STORED * N;      // 4096
constexpr unsigned C_BYTES = M * N * sizeof(float);

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "matmul_fp8_sparse.elf";

  const unsigned seed = (argc > 2) ? std::stoul(argv[2]) : 42u;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(-4.0f, 4.0f);
  std::uniform_int_distribution<int> sel_dis(0, 5);

  // A: dense 4 x 128 fp8 activations, row-major.
  uint8_t A_fp8[M * K];
  for (unsigned i = 0; i < M * K; i++)
    A_fp8[i] = float_to_fp8_e4m3(dis(gen));

  // B: 64 x 64 stored fp8 matrix, COLUMN-major: element (r, n) at n*64 + r.
  uint8_t B_stored[K_STORED * N];
  for (unsigned i = 0; i < K_STORED * N; i++)
    B_stored[i] = float_to_fp8_e4m3(dis(gen));

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
  float C_ref[M * N];
  memset(C_ref, 0, sizeof(C_ref));
  for (unsigned m = 0; m < M; m++)
    for (unsigned n = 0; n < N; n++)
      for (unsigned q = 0; q < QUARTETS; q++) {
        const Selection &sl = kSelections[sel[n * QUARTETS + q]];
        for (unsigned i = 0; i < 2; i++)
          C_ref[m * N + n] +=
              fp8_e4m3_to_float(A_fp8[m * K + 4 * q + sl.pos[i]]) *
              fp8_e4m3_to_float(B_stored[n * K_STORED + 2 * q + i]);
      }

  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_am = xrt::ext::bo{device, AM_BYTES};
  xrt::bo bo_b = xrt::ext::bo{device, B_BYTES};
  xrt::bo bo_c = xrt::ext::bo{device, C_BYTES};

  // AM = [A 512 B | sparsity image 768 B]; matmul_hw re-splits it on the core.
  uint8_t *buf_am = bo_am.map<uint8_t *>();
  std::memcpy(buf_am, A_fp8, A_BYTES);
  std::memcpy(buf_am + A_BYTES, mask.data(), MASK_BYTES);

  std::memcpy(bo_b.map<uint8_t *>(), B_stored, B_BYTES);

  uint32_t *buf_c_words = bo_c.map<uint32_t *>();
  for (unsigned i = 0; i < M * N; i++)
    buf_c_words[i] = 0xdeadbeefu;

  bo_am.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_am, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  float C_hw[M * N];
  std::memcpy(C_hw, bo_c.map<void *>(), C_BYTES);

  uint32_t errs = 0;
  float max_rel_err = 0.0f;
  for (unsigned i = 0; i < M * N; i++) {
    float diff = fabsf(C_hw[i] - C_ref[i]);
    float ref_abs = fabsf(C_ref[i]);
    float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
    if (rel_err > max_rel_err)
      max_rel_err = rel_err;

    if (rel_err > 1e-3f && diff > 1e-3f) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ [%d][%d]: got %.6f, expected %.6f (rel=%.6f)\n", i / N,
               i % N, C_hw[i], C_ref[i], rel_err);
    }
  }

  printf("C[0][0] hw=%.6f ref=%.6f   C[3][63] hw=%.6f ref=%.6f\n", C_hw[0],
         C_ref[0], C_hw[M * N - 1], C_ref[M * N - 1]);
  printf("Max relative error: %.8f\n", max_rel_err);

  if (!errs) {
    printf("PASS (all %u outputs within tolerance)\n", M * N);
    return 0;
  }
  printf("FAIL (%u/%u errors)\n", errs, M * N);
  return 1;
}
