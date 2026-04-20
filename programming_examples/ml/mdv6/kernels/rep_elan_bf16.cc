//===- rep_elan_bf16.cc - Unified MDV6 AIE2P kernels -*-C++-*-===//
//
// Tier 1 consolidation: all kernels needed for a rep_elan layer in one .o.
// Bodies merged verbatim from:
//   - conv/conv_bf16.cc          : conv3x3_fused_packed_bf16, conv1x1_fused_packed_bf16
//   - gemm_conv1x1/gemm_conv1x1_bf16.cc : gemm_conv1x1_fused_packed_bf16, gemm_conv1x1_kblocked_bf16
//   - NEW                        : residual_add_silu_bf16 (scalar; preserves current SiLU math)
//
// Numerics must match pre-consolidation baseline exactly.
// See ~/.claude/plans/make-a-plan-for-jazzy-pebble.md for context.
//
//===----------------------------------------------------------------------===//

#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

// ---------- file-local helpers (from conv/conv_bf16.cc) ----------

inline float fast_sigmoid_conv(float x) {
  return 0.5f + x / (2.0f * (1.0f + (x > 0 ? x : -x)));
}

extern "C" {

// ---------- conv3x3_fused_packed_bf16 (from conv/conv_bf16.cc) ----------

// Vectorized 3x3 conv+BN+SiLU using mmul<4,8,8>.
// Weights must be pre-packed on host as [OC/8, IC/8, 9, 8ic, 8oc].
// Channels must be multiples of 8.
void conv3x3_fused_packed_bf16(bfloat16 *__restrict input,
                                bfloat16 *__restrict packed_weights,
                                bfloat16 *__restrict output,
                                int32_t tile_h,
                                int32_t tile_w,
                                int32_t ic,
                                int32_t oc,
                                int32_t stride,
                                int32_t padding) {
  event0();

  const int patch_w = (tile_w - 1) * stride + 3;
  const int spatial_out = tile_h * tile_w;
  const int wt_size = oc * ic * 9;
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w_ptr = packed_weights + wt_size;
  bfloat16 *bn_b_ptr = bn_w_ptr + oc;

  const int ic_blocks = ic / 8;
  const int oc_blocks = oc / 8;

  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

  for (int oc_blk = 0; oc_blk < oc_blocks; oc_blk++) {
    aie::vector<bfloat16, 8> bn_w_vec = aie::load_v<8>(bn_w_ptr + oc_blk * 8);
    aie::vector<bfloat16, 8> bn_b_vec = aie::load_v<8>(bn_b_ptr + oc_blk * 8);

    const bfloat16 *wt_base = weights + oc_blk * ic_blocks * 9 * 64;

    for (int sp = 0; sp < spatial_out; sp += 4) {
      MMUL acc(aie::zeros<bfloat16, MMUL::size_C>());

      const bfloat16 *wt_ptr = wt_base;

      for (int ic_blk = 0; ic_blk < ic_blocks; ic_blk++) {
        for (int kh = 0; kh < 3; kh++) {
          for (int kw = 0; kw < 3; kw++) {
            aie::vector<bfloat16, MMUL::size_A> A;
            for (int p = 0; p < 4; p++) {
              int out_idx = sp + p;
              if (out_idx < spatial_out) {
                int oh = out_idx / tile_w;
                int ow = out_idx % tile_w;
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;
                A.insert(p, aie::load_v<8>(input + (ih * patch_w + iw) * ic + ic_blk * 8));
              }
            }

            aie::vector<bfloat16, MMUL::size_B> B = aie::load_v<64>(wt_ptr);
            wt_ptr += 64;

            acc.mac(A, B);
          }
        }
      }

      aie::vector<bfloat16, MMUL::size_C> result = acc.template to_vector<bfloat16>();
      for (int p = 0; p < 4; p++) {
        int pidx = sp + p;
        if (pidx < spatial_out) {
          aie::vector<bfloat16, 8> row = result.extract<8>(p);
          aie::accum<accfloat, 8> bn_acc = aie::mul(row, bn_w_vec);
          aie::vector<bfloat16, 8> bn_out = aie::add(bn_acc.to_vector<bfloat16>(), bn_b_vec);
          bfloat16 *out_ptr = output + pidx * oc + oc_blk * 8;
          for (int j = 0; j < 8; j++) {
            float x = (float)bn_out[j];
            float ax = x > 0.0f ? x : -x;
            out_ptr[j] = (bfloat16)(x * (0.5f + x / (2.0f + 2.0f * ax)));
          }
        }
      }
    }
  }

  event1();
}

// ---------- conv1x1_fused_packed_bf16 (from conv/conv_bf16.cc) ----------

void conv1x1_fused_packed_bf16(bfloat16 *input, bfloat16 *packed_weights,
                                bfloat16 *output,
                                int32_t tile_height, int32_t tile_width,
                                int32_t in_channels, int32_t out_channels,
                                int32_t stride_unused, int32_t padding_unused) {
  // Tiled fused conv1x1+BN+SiLU: tile dims are output dims (= input dims for 1x1)
  int wt_size = out_channels * in_channels;
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w = packed_weights + wt_size;
  bfloat16 *bn_b = bn_w + out_channels;

  event0();
  for (int oc = 0; oc < out_channels; oc++) {
    float bw = (float)bn_w[oc];
    float bb = (float)bn_b[oc];
    for (int h = 0; h < tile_height; h++) {
      for (int w = 0; w < tile_width; w++) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
          int idx = (h * tile_width + w) * in_channels + ic;
          sum += (float)input[idx] * (float)weights[oc * in_channels + ic];
        }
        float bn_out = bw * sum + bb;
        float silu_out = bn_out * fast_sigmoid_conv(bn_out);
        output[(h * tile_width + w) * out_channels + oc] = (bfloat16)silu_out;
      }
    }
  }
  event1();
}

// ---------- gemm_conv1x1_fused_packed_bf16 (from gemm_conv1x1/gemm_conv1x1_bf16.cc) ----------

void gemm_conv1x1_fused_packed_bf16(bfloat16 *input, bfloat16 *packed_weights,
                                     bfloat16 *output, int32_t tile_h,
                                     int32_t tile_w, int32_t ic, int32_t oc,
                                     int32_t stride_unused, int32_t padding_unused) {
  event0();

  int32_t tile_m = tile_h * tile_w;
  int wt_size = oc * ic;
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w_ptr = packed_weights + wt_size;
  bfloat16 *bn_b_ptr = bn_w_ptr + oc;

  int ic_blocks = ic / 8;
  int oc_blocks = oc / 8;

  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

  for (int oc_blk = 0; oc_blk < oc_blocks; oc_blk++) {
    aie::vector<bfloat16, 8> bn_w_vec = aie::load_v<8>(bn_w_ptr + oc_blk * 8);
    aie::vector<bfloat16, 8> bn_b_vec = aie::load_v<8>(bn_b_ptr + oc_blk * 8);

    for (int sp = 0; sp < tile_m; sp += 4) {
      MMUL acc(aie::zeros<bfloat16, MMUL::size_C>());

      for (int ic_blk = 0; ic_blk < ic_blocks; ic_blk++) {
        // Load A: 4 pixels x 8 input channels
        aie::vector<bfloat16, MMUL::size_A> A;
        for (int p = 0; p < 4; p++) {
          int pidx = sp + p;
          if (pidx < tile_m) {
            A.insert(p, aie::load_v<8>(input + pidx * ic + ic_blk * 8));
          }
        }

        // Load B: 8ic x 8oc block
        // Weight layout: [ic/8, oc/8, 8, 8] (same as conv_bf16_vec.cc)
        aie::vector<bfloat16, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(weights + (ic_blk * oc_blocks + oc_blk) * 64);

        acc.mac(A, B);
      }

      aie::vector<bfloat16, MMUL::size_C> result = acc.template to_vector<bfloat16>();

      // BN + SiLU
      for (int p = 0; p < 4; p++) {
        int pidx = sp + p;
        if (pidx < tile_m) {
          aie::vector<bfloat16, 8> row = result.extract<8>(p);
          aie::accum<accfloat, 8> bn_acc = aie::mul(row, bn_w_vec);
          aie::vector<bfloat16, 8> bn_out = aie::add(bn_acc.to_vector<bfloat16>(), bn_b_vec);
          bfloat16 *out_ptr = output + pidx * oc + oc_blk * 8;
          for (int j = 0; j < 8; j++) {
            float x = (float)bn_out[j];
            float ax = x > 0.0f ? x : -x;
            out_ptr[j] = (bfloat16)(x * (0.5f + x / (2.0f + 2.0f * ax)));
          }
        }
      }
    }
  }

  event1();
}

// ---------- gemm_conv1x1_kblocked_bf16 (from gemm_conv1x1/gemm_conv1x1_bf16.cc) ----------

// K-blocked GEMM Conv1x1: processes k_block IC channels per call, accumulates
// across multiple calls. Input stays in L1 (tile_m × full_ic), weight chunks
// stream through (k_block × oc). First call zeros output, last applies BN+SiLU.
//
// Weight chunk layout: [k_block/8, oc/8, 8ic, 8oc, bn_w(oc), bn_b(oc)]
// BN params appended to every chunk but only read on last K-block.
void gemm_conv1x1_kblocked_bf16(bfloat16 *input, bfloat16 *wt_chunk,
                                 bfloat16 *output, int32_t tile_m,
                                 int32_t full_ic, int32_t oc,
                                 int32_t k_start, int32_t k_block,
                                 int32_t n_k_blocks) {
  event0();

  int is_first = (k_start == 0);
  int is_last = (k_start + k_block >= full_ic);

  int kb_blocks = k_block / 8;  // IC blocks in this chunk
  int oc_blocks = oc / 8;
  int wt_size = k_block * oc;   // conv weight elements in chunk

  // BN params at end of chunk (only used on last K-block)
  bfloat16 *bn_w_ptr = wt_chunk + wt_size;
  bfloat16 *bn_b_ptr = bn_w_ptr + oc;

  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

  for (int oc_blk = 0; oc_blk < oc_blocks; oc_blk++) {
    for (int sp = 0; sp < tile_m; sp += 4) {
      MMUL acc;

      if (is_first) {
        // First K-block: zero accumulator
        acc = MMUL(aie::zeros<bfloat16, MMUL::size_C>());
      } else {
        // Load partial sums from output buffer
        aie::vector<bfloat16, MMUL::size_C> partial;
        for (int p = 0; p < 4; p++) {
          int pidx = sp + p;
          if (pidx < tile_m) {
            partial.insert(p, aie::load_v<8>(output + pidx * oc + oc_blk * 8));
          }
        }
        acc = MMUL(partial);
      }

      // K-reduction over this chunk's IC channels
      for (int kb = 0; kb < kb_blocks; kb++) {
        // Load A: 4 pixels × 8 IC channels from input
        // Input is tile_m × full_ic; read at IC offset k_start + kb*8
        aie::vector<bfloat16, MMUL::size_A> A;
        for (int p = 0; p < 4; p++) {
          int pidx = sp + p;
          if (pidx < tile_m) {
            A.insert(p, aie::load_v<8>(input + pidx * full_ic + k_start + kb * 8));
          }
        }

        // Load B: [kb, oc_blk] weight block (contiguous 64 elements)
        aie::vector<bfloat16, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(wt_chunk + (kb * oc_blocks + oc_blk) * 64);

        acc.mac(A, B);
      }

      if (is_last) {
        // Last K-block: apply BN + SiLU and write final output
        aie::vector<bfloat16, 8> bn_w_vec = aie::load_v<8>(bn_w_ptr + oc_blk * 8);
        aie::vector<bfloat16, 8> bn_b_vec = aie::load_v<8>(bn_b_ptr + oc_blk * 8);

        aie::vector<bfloat16, MMUL::size_C> result = acc.template to_vector<bfloat16>();
        for (int p = 0; p < 4; p++) {
          int pidx = sp + p;
          if (pidx < tile_m) {
            aie::vector<bfloat16, 8> row = result.extract<8>(p);
            aie::accum<accfloat, 8> bn_acc = aie::mul(row, bn_w_vec);
            aie::vector<bfloat16, 8> bn_out = aie::add(bn_acc.to_vector<bfloat16>(), bn_b_vec);
            bfloat16 *out_ptr = output + pidx * oc + oc_blk * 8;
            for (int j = 0; j < 8; j++) {
              float x = (float)bn_out[j];
              float ax = x > 0.0f ? x : -x;
              out_ptr[j] = (bfloat16)(x * (0.5f + x / (2.0f + 2.0f * ax)));
            }
          }
        }
      } else {
        // Not last: write partial sums (bf16) back to output buffer
        aie::vector<bfloat16, MMUL::size_C> result = acc.template to_vector<bfloat16>();
        for (int p = 0; p < 4; p++) {
          int pidx = sp + p;
          if (pidx < tile_m) {
            aie::vector<bfloat16, 8> row = result.extract<8>(p);
            aie::store_v(output + pidx * oc + oc_blk * 8, row);
          }
        }
      }
    }
  }

  event1();
}

// ---------- residual_add_silu_bf16 (NEW, scalar) ----------
//
// out[i] = silu(current[i] + residual[i])  for i in [0, tile_m * oc)
// SiLU approx: x * (0.5 + x/(2 + 2*|x|))   (matches gemm_conv1x1_bf16.cc:62-64)

void residual_add_silu_bf16(bfloat16 *__restrict current,
                             bfloat16 *__restrict residual,
                             bfloat16 *__restrict out,
                             int32_t tile_m, int32_t oc) {
  event0();
  const int32_t n = tile_m * oc;
  for (int i = 0; i < n; i++) {
    float x = (float)current[i] + (float)residual[i];
    float ax = x > 0.0f ? x : -x;
    out[i] = (bfloat16)(x * (0.5f + x / (2.0f + 2.0f * ax)));
  }
  event1();
}

} // extern "C"
