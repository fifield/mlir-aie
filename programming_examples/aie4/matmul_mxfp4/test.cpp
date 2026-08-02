//===- test.cpp - XRT host for matmul_mxfp4 --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Runs the matmul_mxfp4 example on npu3 hardware through XRT.
// The MX-packed A (544 bytes) and B (8704 bytes) operands are streamed to
// the compute tile over two shim MM2S channels; the fp32 C result
// (4x64 = 256 words) is drained back over one S2MM channel.
//
// The MX block packing, the decode-and-reference matmul and the tolerance
// check are taken verbatim from the simulator host (test_sim.cpp).
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
#include "../cpu_ref/mx_types.h"

constexpr unsigned M = 4;
constexpr unsigned N = 64;
constexpr unsigned K = 256;
constexpr unsigned K_OUTER = 4;
constexpr unsigned K_INNER = 64;
constexpr unsigned MX_VEC_BYTES = 68; // 64 packed mantissa + 4 exponents
constexpr unsigned A_MX_BYTES = K_OUTER * (M / 2) * MX_VEC_BYTES; // 544
constexpr unsigned B_MX_BYTES = K_OUTER * 32 * MX_VEC_BYTES;      // 8704

static void pack_mxfp4_vector(const float *elements, uint8_t *packed) {
  unsigned pos = 0;
  for (unsigned b = 0; b < 4; b++) {
    uint8_t packed_mantissas[MXFP4_MANTISSA_BYTES];
    uint8_t shared_exp;
    float_to_mxfp4_block(&elements[b * 32], packed_mantissas, &shared_exp);
    packed[pos++] = shared_exp;
    memcpy(&packed[pos], packed_mantissas, MXFP4_MANTISSA_BYTES);
    pos += MXFP4_MANTISSA_BYTES;
  }
}

static void unpack_mxfp4_vector(const uint8_t *packed, float *elements) {
  unsigned pos = 0;
  for (unsigned b = 0; b < 4; b++) {
    uint8_t shared_exp = packed[pos++];
    mxfp4_block_to_float(&packed[pos], shared_exp, &elements[b * 32]);
    pos += MXFP4_MANTISSA_BYTES;
  }
}

int main(int argc, const char *argv[]) {
  const std::string elfFile = (argc > 1) ? argv[1] : "matmul_mxfp4.elf";

  printf("matmul_mxfp4: MMAC MXFP4 matmul (4x256 * 256x64), fp32 output\n");

  static float A_float[M * K];
  static float B_float[K * N];

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (unsigned i = 0; i < M * K; i++)
    A_float[i] = dis(gen);
  for (unsigned i = 0; i < K * N; i++)
    B_float[i] = dis(gen);

  static uint8_t A_mx[A_MX_BYTES];
  static uint8_t B_mx[B_MX_BYTES];
  memset(A_mx, 0, sizeof(A_mx));
  memset(B_mx, 0, sizeof(B_mx));

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned v = 0; v < M / 2; v++) {
      float vec_elements[128];
      unsigned row0 = 2 * v, row1 = 2 * v + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned j = 0; j < K_INNER; j++) {
        vec_elements[j] = A_float[row0 * K + k_start + j];
        vec_elements[K_INNER + j] = A_float[row1 * K + k_start + j];
      }
      unsigned offset = (k * (M / 2) + v) * MX_VEC_BYTES;
      pack_mxfp4_vector(vec_elements, &A_mx[offset]);
    }
  }

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned j = 0; j < 32; j++) {
      float vec_elements[128];
      unsigned col0 = 2 * j;
      unsigned col1 = 2 * j + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned kk = 0; kk < K_INNER; kk++) {
        vec_elements[kk] = B_float[(k_start + kk) * N + col0];
        vec_elements[K_INNER + kk] = B_float[(k_start + kk) * N + col1];
      }
      unsigned offset = (k * 32 + j) * MX_VEC_BYTES;
      pack_mxfp4_vector(vec_elements, &B_mx[offset]);
    }
  }

  static float A_decoded[M * K], B_decoded[K * N];

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned v = 0; v < M / 2; v++) {
      float vec_elements[128];
      unsigned offset = (k * (M / 2) + v) * MX_VEC_BYTES;
      unpack_mxfp4_vector(&A_mx[offset], vec_elements);
      unsigned row0 = 2 * v, row1 = 2 * v + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned j = 0; j < K_INNER; j++) {
        A_decoded[row0 * K + k_start + j] = vec_elements[j];
        A_decoded[row1 * K + k_start + j] = vec_elements[K_INNER + j];
      }
    }
  }

  for (unsigned k = 0; k < K_OUTER; k++) {
    for (unsigned j = 0; j < 32; j++) {
      float vec_elements[128];
      unsigned offset = (k * 32 + j) * MX_VEC_BYTES;
      unpack_mxfp4_vector(&B_mx[offset], vec_elements);
      unsigned col0 = 2 * j;
      unsigned col1 = 2 * j + 1;
      unsigned k_start = k * K_INNER;
      for (unsigned kk = 0; kk < K_INNER; kk++) {
        B_decoded[(k_start + kk) * N + col0] = vec_elements[kk];
        B_decoded[(k_start + kk) * N + col1] = vec_elements[K_INNER + kk];
      }
    }
  }

  static float C_ref[M * N];
  memset(C_ref, 0, sizeof(C_ref));
  for (unsigned m = 0; m < M; m++)
    for (unsigned n = 0; n < N; n++)
      for (unsigned k = 0; k < K; k++)
        C_ref[m * N + n] += A_decoded[m * K + k] * B_decoded[k * N + n];

  // ---- XRT plumbing -------------------------------------------------------
  auto device = xrt::device(0u);
  xrt::elf ctx_elf{elfFile};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_a = xrt::ext::bo{device, A_MX_BYTES};
  xrt::bo bo_b = xrt::ext::bo{device, B_MX_BYTES};
  xrt::bo bo_c = xrt::ext::bo{device, M * N * sizeof(float)};

  std::memcpy(bo_a.map<uint8_t *>(), A_mx, A_MX_BYTES);
  std::memcpy(bo_b.map<uint8_t *>(), B_mx, B_MX_BYTES);
  uint32_t *buf_c = bo_c.map<uint32_t *>();
  for (unsigned i = 0; i < M * N; i++)
    buf_c[i] = 0xdeadbeefu;

  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_a, bo_b, bo_c).wait2();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  float C_hw[M * N];
  std::memcpy(C_hw, buf_c, sizeof(C_hw));

  // ---- Comparison (same tolerance as the simulator host) ------------------
  uint32_t errs = 0;
  float max_rel_err = 0.0f;
  for (unsigned i = 0; i < M * N; i++) {
    float diff = fabsf(C_hw[i] - C_ref[i]);
    float ref_abs = fabsf(C_ref[i]);
    float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
    if (rel_err > max_rel_err)
      max_rel_err = rel_err;

    if (rel_err > 0.15f && diff > 0.1f) {
      errs++;
      if (errs <= 10)
        printf("ERROR @ [%d][%d]: got %.6f, expected %.6f (rel=%.6f)\n", i / N,
               i % N, C_hw[i], C_ref[i], rel_err);
    }
  }

  printf("Max relative error: %.8f\n", max_rel_err);

  if (!errs) {
    printf("PASS (all %u outputs within tolerance)\n", M * N);
    return 0;
  }
  printf("FAIL (%u/%u errors)\n", errs, M * N);
  return 1;
}
