//===- awq_gemv.cc - Packed uint4 AWQ GEMV kernel -*- C++ -*-===========//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Correctness-first external kernel for a dimension-specialized AWQ GEMV:
//   y[m] = sum_k x[k] * ((qweight[m, k/2].nibble - zero[m, group]) * scale[m, group])
//
// qweight is uint8 row-major [M, K/2]. The low nibble holds even K elements and
// the high nibble holds odd K elements. params is bf16 row-major
// [M, 2 * groups] with interleaved [scale, zero] pairs.
//
// The default scalar path remains the correctness baseline. An opt-in
// inline-dequant + BF16 vector-MAC path can be compiled with
// AWQ_GEMV_VECTORIZE_INLINE_DEQUANT=1 for real AWQ groups (g128): each chunk
// dequantizes packed uint4 weights into a small bf16 scratch vector and feeds
// the same aie::mac shape as the optimized BF16 GEMV kernel.
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

#ifndef AWQ_GEMV_K
#define AWQ_GEMV_K 2048
#endif

#ifndef AWQ_GEMV_M
#define AWQ_GEMV_M 2048
#endif

#ifndef AWQ_GEMV_GROUP_SIZE
#define AWQ_GEMV_GROUP_SIZE 128
#endif

#ifndef AWQ_GEMV_VECTORIZE_INLINE_DEQUANT
#define AWQ_GEMV_VECTORIZE_INLINE_DEQUANT 0
#endif

#ifndef AWQ_GEMV_VECTOR_LENGTH
#define AWQ_GEMV_VECTOR_LENGTH 32
#endif

static_assert((AWQ_GEMV_K % 2) == 0, "AWQ_GEMV_K must be even for uint4 packing");
static_assert((AWQ_GEMV_K % AWQ_GEMV_GROUP_SIZE) == 0,
              "AWQ_GEMV_K must be divisible by AWQ_GEMV_GROUP_SIZE");

template <uint32_t VecLen>
static inline void dequant_chunk(const uint8_t *__restrict q_chunk, float scale,
                                 float zero,
                                 bfloat16 *__restrict w_chunk) {
  static_assert((VecLen % 2) == 0, "VecLen must unpack whole uint4 pairs");
  AIE_PREPARE_FOR_PIPELINING
  for (uint32_t pair = 0; pair < VecLen / 2; ++pair) {
    const uint8_t packed = q_chunk[pair];
    const float q_even = static_cast<float>(packed & 0x0F);
    const float q_odd = static_cast<float>((packed >> 4) & 0x0F);
    w_chunk[2 * pair] = static_cast<bfloat16>((q_even - zero) * scale);
    w_chunk[2 * pair + 1] = static_cast<bfloat16>((q_odd - zero) * scale);
  }
}

static inline float awq_gemv_group_scalar(const bfloat16 *__restrict x_group,
                                          const uint8_t *__restrict q_group,
                                          float scale, float zero) {
  constexpr uint32_t PackedPerGroup = AWQ_GEMV_GROUP_SIZE / 2;
  float acc = 0.0f;
  AIE_PREPARE_FOR_PIPELINING
  for (uint32_t pair = 0; pair < PackedPerGroup; ++pair) {
    const uint8_t packed = q_group[pair];
    const float q_even = static_cast<float>(packed & 0x0F);
    const float q_odd = static_cast<float>((packed >> 4) & 0x0F);
    const uint32_t x_idx = 2 * pair;
    acc += static_cast<float>(x_group[x_idx]) * ((q_even - zero) * scale);
    acc += static_cast<float>(x_group[x_idx + 1]) * ((q_odd - zero) * scale);
  }
  return acc;
}

template <uint32_t VecLen>
static inline float awq_gemv_group_vectorized(
    const bfloat16 *__restrict x_group, const uint8_t *__restrict q_group,
    float scale, float zero) {
  static_assert((AWQ_GEMV_GROUP_SIZE % VecLen) == 0,
                "AWQ_GEMV_GROUP_SIZE must be divisible by VecLen");
  aie::accum<accfloat, VecLen> acc = aie::zeros<accfloat, VecLen>();
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (uint32_t chunk_start = 0; chunk_start < AWQ_GEMV_GROUP_SIZE;
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
}

extern "C" {

void awq_gemv_u4_bf16(const bfloat16 *__restrict x,
                      const uint8_t *__restrict qweight,
                      const bfloat16 *__restrict params,
                      bfloat16 *__restrict y) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  constexpr uint32_t Groups = AWQ_GEMV_K / AWQ_GEMV_GROUP_SIZE;
  constexpr uint32_t PackedPerGroup = AWQ_GEMV_GROUP_SIZE / 2;
  constexpr uint32_t PackedPerRow = AWQ_GEMV_K / 2;
  static_assert((AWQ_GEMV_GROUP_SIZE % 2) == 0,
                "AWQ_GEMV_GROUP_SIZE must be even for pairwise unpacking");

  for (uint32_t row = 0; row < AWQ_GEMV_M; ++row) {
    float acc = 0.0f;
    const uint8_t *__restrict q_row = qweight + row * PackedPerRow;
    const bfloat16 *__restrict p_row = params + row * (2 * Groups);

    for (uint32_t group = 0; group < Groups; ++group) {
      const float scale = static_cast<float>(p_row[0]);
      const float zero = static_cast<float>(p_row[1]);
      const bfloat16 *__restrict x_group = x + group * AWQ_GEMV_GROUP_SIZE;
      const uint8_t *__restrict q_group = q_row + group * PackedPerGroup;

      if constexpr (AWQ_GEMV_VECTORIZE_INLINE_DEQUANT &&
                    AWQ_GEMV_GROUP_SIZE >= AWQ_GEMV_VECTOR_LENGTH &&
                    (AWQ_GEMV_GROUP_SIZE % AWQ_GEMV_VECTOR_LENGTH) == 0) {
        constexpr uint32_t VecLen = AWQ_GEMV_VECTOR_LENGTH;
        acc += awq_gemv_group_vectorized<VecLen>(x_group, q_group, scale, zero);
      } else {
        for (uint32_t pair = 0; pair < PackedPerGroup; ++pair) {
          const uint8_t packed = q_group[pair];
          const float q_even = static_cast<float>(packed & 0x0F);
          const float q_odd = static_cast<float>((packed >> 4) & 0x0F);
          const uint32_t x_idx = 2 * pair;
          acc += static_cast<float>(x_group[x_idx]) * ((q_even - zero) * scale);
          acc += static_cast<float>(x_group[x_idx + 1]) * ((q_odd - zero) * scale);
        }
      }

      p_row += 2;
    }

    y[row] = static_cast<bfloat16>(acc);
  }
}

} // extern "C"
