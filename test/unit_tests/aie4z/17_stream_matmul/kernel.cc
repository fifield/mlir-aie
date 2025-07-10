//===- kernel.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include <stdio.h>

using namespace aie;

inline void get_ss_v128_weight_staging(int stream_id, uint5_t reg_idx,
                                       uint2_t sign) {

  insert_staging(get_ss_v128int8_weight(stream_id), reg_idx, sign);
}

inline __attribute__((always_inline)) void matrix_stream_load(int mul_mode) {

  constexpr unsigned stream_id = 1; // using stream 1 for weights and 0 for acts
  #pragma unroll
  for (size_t reg_idx = 0; reg_idx < 32; reg_idx++) {
    get_ss_v128_weight_staging(stream_id, reg_idx, mul_mode);
  }
}

extern "C" {
void matmul();
}

void matmul() {

  constexpr unsigned int K_outer = 4;
  constexpr unsigned int M = 64;

  constexpr unsigned int M_div_2 = M / 2;
  v64acc32 accA[M_div_2];
  v64acc32 accB[M_div_2];

  #pragma unroll
  for (unsigned l = 0; l < M_div_2; l++) {
    accA[l] = undef_v64acc32();
    accB[l] = undef_v64acc32();
  }

  uint2_t mul_mode = MMAC_INT8;
  bool zero_acc = 1; // Accumulation set to zero in first iteration

  #pragma unroll
  for (size_t reg_idx = 0; reg_idx < 32; reg_idx++) {
    insert_staging(get_ss_v128int8_weight(1), reg_idx, mul_mode);
  }

  for (unsigned int k = 0; k < K_outer - 1; k++)
    chess_prepare_for_pipelining chess_loop_range(3, 3) {

      staging_to_matrix_m64x64int8();

      #pragma unroll
      for (unsigned int i = 0; i < M_div_2; i++) {
        v64int8 input_vector1 = get_ss_v64int8(0);
        v64int8 input_vector2 = get_ss_v64int8(0);

        accA[i] = mac_conf(input_vector1, 1, 1, accA[i], zero_acc);
        accB[i] = mac_conf(input_vector2, 1, 1, accB[i], zero_acc);
      }

      zero_acc = 0; // Accumulation mode set after first iteration

      #pragma unroll
      for (size_t reg_idx = 0; reg_idx < 32; reg_idx++) {
        insert_staging(get_ss_v128int8_weight(1), reg_idx, mul_mode);
      }
    }

  staging_to_matrix_m64x64int8();

  #pragma unroll
  for (unsigned int i = 0; i < M_div_2; i++) {
    v64int8 input_vector1 = get_ss_v64int8(0);
    v64int8 input_vector2 = get_ss_v64int8(0);

    accA[i] = mac_conf(input_vector1, 1, 1, accA[i], zero_acc);
    accB[i] = mac_conf(input_vector2, 1, 1, accB[i], zero_acc);
  }

  // writing results to stream
  #pragma unroll
  for (unsigned i = 0; i < M_div_2; i++) {
    #pragma unroll
    for (unsigned j = 0; j < 2; j++) {
      v32acc32 acc1024b = extract_v32acc32(accA[i], j);
      #pragma unroll
      for (unsigned k = 0; k < 2; k++) {
        v16acc32 acc512b = extract_v16acc32(acc1024b, k);
        v16int32 out512b = v16int32(acc512b);
        put_ms(out512b);
      }
    }

    #pragma unroll
    for (unsigned j = 0; j < 2; j++) {
      v32acc32 acc1024b = extract_v32acc32(accB[i], j);
      #pragma unroll
      for (unsigned k = 0; k < 2; k++) {
        v16acc32 acc512b = extract_v16acc32(acc1024b, k);
        v16int32 out512b = v16int32(acc512b);
        put_ms(out512b);
      }
    }
  }
}
