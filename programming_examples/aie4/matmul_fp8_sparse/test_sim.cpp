//===- test_sim.cpp - Simulator harness for matmul_fp8_sparse --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verifies MMAC fp8*fp8 sparse->fp32 on AIE4 by writing directly to tile
// memory.  C[4x64] += A[4x128] * B_sparse[128x64], 4:2 structured sparsity.
//
// The operand layout and the sparsity-register encoding are described in
// README.md (according to the AIE4 ISA); the XRT
// hardware host (test.cpp) uses the same data.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <stdint.h>
#include <stdio.h>

#include "memory_allocator.h"
#include "test_utils.h"

#include "../cpu_ref/fp_types.h"

constexpr unsigned M = 4;
constexpr unsigned N = 64;
constexpr unsigned K = 128;                  // dense logical K per A row
constexpr unsigned QUARTETS = K / 4;         // 32
constexpr unsigned K_STORED = QUARTETS * 2;  // 64 stored slots per column

constexpr unsigned A_BYTES = M * K;          // 512
constexpr unsigned MASK_WORDS = 192;         // 6 Kbit
constexpr unsigned B_BYTES = K_STORED * N;   // 4096

// The six 4:2 quartet selections and their 3-bit encodings (ISA).
struct Selection {
  uint8_t code;
  uint8_t pos[2];
};
static const Selection kSelections[6] = {
    {0b001, {0, 2}}, {0b010, {0, 3}}, {0b011, {1, 2}},
    {0b100, {1, 3}}, {0b101, {2, 3}}, {0b000, {0, 1}},
};

// Set column n's quartet q to a 3-bit code in the 6-Kbit register image.
static void set_sparsity_code(uint32_t *mask, unsigned n, unsigned q,
                              uint8_t code) {
  for (unsigned i = 0; i < 3; i++) {
    if (!((code >> i) & 1))
      continue;
    const unsigned bit = 3 * q + i;
    mask[(bit / 16) * 32 + n / 2] |= 1u << ((n % 2) * 16 + (bit % 16));
  }
}

// Staging image is column-major: 64 contiguous bytes per output column.
static inline unsigned stage_byte(unsigned slot, unsigned n) {
  return n * 64 + slot;
}

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "matmul_fp8_sparse",
      "MMAC fp8*fp8 sparse matmul (4x128 * 128x64, 4:2 structured sparsity)");

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-4.0f, 4.0f);
  std::uniform_int_distribution<int> sel_dis(0, 5);

  // A: dense 4 x 128 fp8 activations.
  std::vector<uint8_t> A_fp8(M * K);
  for (size_t i = 0; i < A_fp8.size(); i++)
    A_fp8[i] = float_to_fp8_e4m3(dis(gen));

  // B in stored form: [column][slot].
  std::vector<uint8_t> B_logical(N * K_STORED);
  for (size_t i = 0; i < B_logical.size(); i++)
    B_logical[i] = float_to_fp8_e4m3(dis(gen));

  // Independent 4:2 selection per (column, quartet).
  std::vector<uint8_t> sel(N * QUARTETS);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < QUARTETS; q++)
      sel[n * QUARTETS + q] = (uint8_t)sel_dis(gen);

  std::vector<uint32_t> mask(MASK_WORDS, 0);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < QUARTETS; q++)
      set_sparsity_code(mask.data(), n, q,
                        kSelections[sel[n * QUARTETS + q]].code);

  std::vector<float> C_ref(M * N, 0.0f);
  for (unsigned m = 0; m < M; m++)
    for (unsigned n = 0; n < N; n++) {
      float acc = 0.0f;
      for (unsigned q = 0; q < QUARTETS; q++) {
        const Selection &sl = kSelections[sel[n * QUARTETS + q]];
        for (unsigned i = 0; i < 2; i++)
          acc += fp8_e4m3_to_float(A_fp8[m * K + 4 * q + sl.pos[i]]) *
                 fp8_e4m3_to_float(B_logical[n * K_STORED + 2 * q + i]);
      }
      C_ref[m * N + n] = acc;
    }

  std::vector<uint8_t> B_stored(B_BYTES);
  for (unsigned n = 0; n < N; n++)
    for (unsigned slot = 0; slot < K_STORED; slot++)
      B_stored[stage_byte(slot, n)] = B_logical[n * K_STORED + slot];

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t col = 0, row = 2;
  uint64_t base = (col << 25) | ((row + 4) << 20);

  // Write A (512 bytes at offset 1024)
  uint32_t *a_words = (uint32_t *)A_fp8.data();
  for (unsigned i = 0; i < A_BYTES / 4; i++)
    mlir_aie_sim_write32(base + 1024 + i * 4, a_words[i]);

  // Write sparsity register image (768 bytes at offset 2048)
  for (unsigned i = 0; i < MASK_WORDS; i++)
    mlir_aie_sim_write32(base + 2048 + i * 4, mask[i]);

  // Write B (4096 bytes at offset 8192)
  uint32_t *b_words = (uint32_t *)B_stored.data();
  for (unsigned i = 0; i < B_BYTES / 4; i++)
    mlir_aie_sim_write32(base + 8192 + i * 4, b_words[i]);

  // Release input lock
  mlir_aie_sim_write32(base + 0x000AE010, 1);

  printf("Waiting for matmul to complete...\n");
  while (mlir_aie_sim_read32(base + 0x000AE030) < 1)
    ;

  // Read C (1024 bytes at offset 65536)
  std::vector<float> C_hw(M * N);
  for (unsigned i = 0; i < M * N; i++) {
    uint32_t bits = mlir_aie_sim_read32(base + 65536 + i * 4);
    memcpy(&C_hw[i], &bits, 4);
  }

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

  printf("Max relative error: %.8f\n", max_rel_err);

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("FAIL: %d/%d errors\n", errs, M * N);
    return -1;
  }
}
