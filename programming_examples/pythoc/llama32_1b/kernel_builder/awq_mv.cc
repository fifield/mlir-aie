//===- awq_mv.cc - Packed uint4 AWQ matvec, single-buffer ABI -*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Drop-in external AIE matvec for the decode multi-launch GEMV builder.
// Same y[m] = dequant(qweight[m, k/2], params[m, 2*groups]) @ x[k]
// arithmetic as the two-buffer ABI, but consumes one combined uint8 buffer
// per linear so the fused multi-launch decode kernel only uses one weight
// DMA channel pair per GEMV. Each row of the combined buffer is laid out as
// [qweight_bytes (k/2)] [params_bytes (2*groups*sizeof(bf16) = 4*groups)].
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#ifndef AIE_PREPARE_FOR_PIPELINING
#define AIE_PREPARE_FOR_PIPELINING
#endif

#ifndef AIE_LOOP_MIN_ITERATION_COUNT
#define AIE_LOOP_MIN_ITERATION_COUNT(x)
#endif

#ifndef AWQ_MV_GROUP_SIZE
#define AWQ_MV_GROUP_SIZE 128
#endif

#ifndef AWQ_MV_VECTOR_LENGTH
#define AWQ_MV_VECTOR_LENGTH 32
#endif

#ifndef AWQ_MATVEC_FN
#define AWQ_MATVEC_FN awq_matvec_vectorized_u4_bf16
#endif

#ifndef AWQ_LINALG_FILL_FN
#define AWQ_LINALG_FILL_FN awq_linalg_fill_bf16
#endif

#ifndef DIM_M_OUTPUT
#define DIM_M_OUTPUT 8
#endif

// Set to 1 to use the vectorized uint4 -> bf16 unpack inner loop
// (vector_cast<uint4> + to_float<bfloat16>). The scalar fallback uses
// per-nibble extraction and bf16 store-and-load, which on AIE2P scales
// roughly with K and drives 600+ ms per o_gemv_ffn_awq layer.
#ifndef AWQ_MV_VECTORIZE_DEQUANT
#define AWQ_MV_VECTORIZE_DEQUANT 0
#endif

static_assert((AWQ_MV_GROUP_SIZE % 2) == 0,
              "AWQ_MV_GROUP_SIZE must be even for uint4 packing");
static_assert((AWQ_MV_GROUP_SIZE % AWQ_MV_VECTOR_LENGTH) == 0,
              "AWQ_MV_GROUP_SIZE must be divisible by vector length");

template <uint32_t VecLen>
static inline float awq_group_vecdeq(const bfloat16 *__restrict x_group,
                                     const uint8_t *__restrict q_group,
                                     float scale, float zero) {
#if AWQ_MV_VECTORIZE_DEQUANT
  // Vectorized path: load packed bytes, reinterpret as uint4 nibbles
  // (low nibble = even K index, high nibble = odd K index per the AWQ
  // pack layout, matching little-endian vector_cast semantics), convert
  // straight to bf16, then (w - zero) * scale and MAC against x.
  const bfloat16 zero_bf = static_cast<bfloat16>(zero);
  const bfloat16 scale_bf = static_cast<bfloat16>(scale);
  const aie::vector<bfloat16, VecLen> zero_v =
      aie::broadcast<bfloat16, VecLen>(zero_bf);
  const aie::vector<bfloat16, VecLen> scale_v =
      aie::broadcast<bfloat16, VecLen>(scale_bf);

  aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (uint32_t chunk_start = 0; chunk_start < AWQ_MV_GROUP_SIZE;
       chunk_start += VecLen) {
    aie::vector<uint8_t, VecLen / 2> packed =
        aie::load_v<VecLen / 2>(q_group + chunk_start / 2);
    aie::vector<uint4, VecLen> u4 = aie::vector_cast<uint4>(packed);
    aie::vector<bfloat16, VecLen> w_bf16 = aie::to_float<bfloat16>(u4);
    aie::vector<bfloat16, VecLen> wm = aie::sub(w_bf16, zero_v);
    aie::vector<bfloat16, VecLen> ws =
        aie::mul(wm, scale_v).template to_vector<bfloat16>();
    aie::vector<bfloat16, VecLen> x_vec = aie::load_v<VecLen>(x_group + chunk_start);
    acc = aie::mac(acc, ws, x_vec);
  }
  return aie::reduce_add(acc.template to_vector<float>());
#else
  aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (uint32_t chunk_start = 0; chunk_start < AWQ_MV_GROUP_SIZE;
       chunk_start += VecLen) {
    const uint8_t *__restrict q_chunk = q_group + chunk_start / 2;
    aie::vector<bfloat16, VecLen> w_vec = aie::zeros<bfloat16, VecLen>();
    AIE_PREPARE_FOR_PIPELINING
    for (uint32_t pair = 0; pair < VecLen / 2; ++pair) {
      const uint8_t packed = q_chunk[pair];
      const float q_even = static_cast<float>(packed & 0x0F);
      const float q_odd = static_cast<float>((packed >> 4) & 0x0F);
      w_vec.set(static_cast<bfloat16>((q_even - zero) * scale), 2 * pair);
      w_vec.set(static_cast<bfloat16>((q_odd - zero) * scale), 2 * pair + 1);
    }
    aie::vector<bfloat16, VecLen> x_vec =
        aie::load_v<VecLen>(x_group + chunk_start);
    acc = aie::mac(acc, w_vec, x_vec);
  }
  return aie::reduce_add(acc.template to_vector<float>());
#endif
}

extern "C" {

// combined_in: row-major (m, k/2 + 4*groups) uint8 buffer.
// Per row layout: [qweight_bytes (k/2)] [params_bytes (4*groups)],
// where params_bytes is 2*groups bf16 values interpreted as bytes.
void AWQ_MATVEC_FN(uint32_t m, uint32_t k, uint32_t row_offset,
                   const uint8_t *__restrict combined_in,
                   const bfloat16 *__restrict x_in,
                   bfloat16 *__restrict c_out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  c_out += row_offset;

  const uint32_t groups = k / AWQ_MV_GROUP_SIZE;
  const uint32_t packed_per_group = AWQ_MV_GROUP_SIZE / 2;
  const uint32_t packed_per_row = k / 2;
  const uint32_t params_bytes_per_row = 4 * groups; // 2*groups bf16 == 4*groups bytes
  const uint32_t row_stride_bytes = packed_per_row + params_bytes_per_row;

  for (uint32_t row = 0; row < m; ++row) {
    float acc = 0.0f;
    const uint8_t *__restrict row_base = combined_in + row * row_stride_bytes;
    const uint8_t *__restrict q_row = row_base;
    const bfloat16 *__restrict p_row =
        reinterpret_cast<const bfloat16 *>(row_base + packed_per_row);
    for (uint32_t group = 0; group < groups; ++group) {
      const float scale = static_cast<float>(p_row[0]);
      const float zero = static_cast<float>(p_row[1]);
      const bfloat16 *__restrict x_group = x_in + group * AWQ_MV_GROUP_SIZE;
      const uint8_t *__restrict q_group = q_row + group * packed_per_group;
      acc += awq_group_vecdeq<AWQ_MV_VECTOR_LENGTH>(x_group, q_group, scale, zero);
      p_row += 2;
    }
    c_out[row] = static_cast<bfloat16>(acc);
  }
}

void AWQ_LINALG_FILL_FN(bfloat16 zero, bfloat16 *c_out) {
  for (uint32_t i = 0; i < DIM_M_OUTPUT; ++i) {
    c_out[i] = zero;
  }
}

} // extern "C"
