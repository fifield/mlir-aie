//===- conv_bf16_vec.cc - Vectorized bf16 conv kernels for AIE2P -*-C++-*-===//
//
// Vectorized Conv+BN+SiLU using AIE mmul intrinsics for bfloat16.
// Uses mmul<4,8,8,bfloat16,bfloat16>: 4 output pixels × 8 input channels × 8 output channels
//
// Weight layout (pre-transposed on host):
//   Conv weights: [ic/8][oc/8][8][8] blocks — each 8×8 block stored row-major
//   BN: [fused_weight(OC), fused_bias(OC)] appended after conv weights
//
// Input/Output layout: HWC (channels must be multiple of 8)
//
//===----------------------------------------------------------------------===//

#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

//===========================================================================
// Vectorized Conv1x1 + fused BN + SiLU (v2: contiguous weight loads + vector BN)
//
// Weights must be pre-transposed into [ic/8][oc/8][8ic][8oc] block layout.
// Packed format: [transposed_conv_wts, bn_w(oc), bn_b(oc)]
//===========================================================================
void conv1x1_vec_v2_bf16(bfloat16 *__restrict input,
                          bfloat16 *__restrict packed_weights,
                          bfloat16 *__restrict output,
                          const int32_t tile_h,
                          const int32_t tile_w,
                          const int32_t ic,
                          const int32_t oc,
                          const int32_t stride_unused,
                          const int32_t padding_unused) {
  event0();

  const int spatial = tile_h * tile_w;
  const int wt_size = oc * ic;  // same total, different layout
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w_ptr = packed_weights + wt_size;
  bfloat16 *bn_b_ptr = bn_w_ptr + oc;

  const int ic_blocks = ic / 8;
  const int oc_blocks = oc / 8;

  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

  for (int oc_blk = 0; oc_blk < oc_blocks; oc_blk++) {
    // Load BN params for this oc block (8 values)
    aie::vector<bfloat16, 8> bn_w_vec = aie::load_v<8>(bn_w_ptr + oc_blk * 8);
    aie::vector<bfloat16, 8> bn_b_vec = aie::load_v<8>(bn_b_ptr + oc_blk * 8);

    for (int sp = 0; sp < spatial; sp += 4) {
      MMUL acc(aie::zeros<bfloat16, MMUL::size_C>());

      for (int ic_blk = 0; ic_blk < ic_blocks; ic_blk++) {
        // Load A: 4 pixels × 8 input channels (contiguous in HWC layout)
        aie::vector<bfloat16, MMUL::size_A> A;
        for (int p = 0; p < 4; p++) {
          int pidx = sp + p;
          if (pidx < spatial) {
            A.insert(p, aie::load_v<8>(input + pidx * ic + ic_blk * 8));
          }
        }

        // Load B: 8ic × 8oc block (contiguous in transposed layout)
        // Block at weights[(ic_blk * oc_blocks + oc_blk) * 64]
        aie::vector<bfloat16, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(weights + (ic_blk * oc_blocks + oc_blk) * 64);

        acc.mac(A, B);
      }

      // Extract result as 4×8 = 32 bf16 values
      aie::vector<bfloat16, MMUL::size_C> result = acc.template to_vector<bfloat16>();

      // Apply BN + SiLU per 8-element row, all vectorized
      for (int p = 0; p < 4; p++) {
        int pidx = sp + p;
        if (pidx < spatial) {
          aie::vector<bfloat16, 8> row = result.extract<8>(p);

          // BN: out = w * x + b
          aie::accum<accfloat, 8> bn_acc = aie::mul(row, bn_w_vec);
          aie::vector<bfloat16, 8> bn_out = aie::add(bn_acc.to_vector<bfloat16>(), bn_b_vec);

          // SiLU scalar — AIE2P lacks bf16→float vector conversion for vectorized sigmoid
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

//===========================================================================
// Vectorized Conv3x3 + fused BN + SiLU
//
// Input:  (patch_h × patch_w × ic) bf16 where patch = (tile-1)*stride + 3
// Weights: packed [transposed_conv_wts, bn_w(oc), bn_b(oc)]
//   Conv weights transposed to [ic/8][oc/8][9][8ic_sub][8oc_sub] or simpler:
//   For 3x3: weights are OC×IC×3×3, we process IC in chunks of 8
//   Weight block for ic_blk, oc_blk: weights[(oc_blk*8+j)*ic*9 + (ic_blk*8+i)*9 + kh*3+kw]
//
// For 3x3, we use mmul<4,8,8> accumulating over ic*9 = ic*kernel_area
// Each spatial output position needs 9 input values per input channel
// Strategy: for each output pixel, gather the 3×3 neighborhood per ic chunk,
// then mmul with the weight block.
//
// Simpler approach: treat 3x3 as 9 separate 1x1 convolutions with shifted input,
// accumulating into the same output. This reuses the mmul pattern.
//===========================================================================
void conv3x3_vec_bf16(bfloat16 *__restrict input,
                       bfloat16 *__restrict packed_weights,
                       bfloat16 *__restrict output,
                       const int32_t tile_h,
                       const int32_t tile_w,
                       const int32_t ic,
                       const int32_t oc,
                       const int32_t stride,
                       const int32_t padding) {
  event0();

  const int patch_h = (tile_h - 1) * stride + 3;
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

    for (int sp = 0; sp < spatial_out; sp += 4) {
      MMUL acc(aie::zeros<bfloat16, MMUL::size_C>());

      // For each input channel block
      for (int ic_blk = 0; ic_blk < ic_blocks; ic_blk++) {
        // For each of the 9 kernel positions
        for (int kh = 0; kh < 3; kh++) {
          for (int kw = 0; kw < 3; kw++) {
            // Load A: 4 output pixels, each needs the (kh,kw) shifted input value
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

            // Load B: weight block for this (kh,kw) position
            // weights layout: [oc][ic][3][3], so for oc_blk*8+j, ic_blk*8+i, kh, kw:
            // idx = ((oc_blk*8+j)*ic + (ic_blk*8+i))*9 + kh*3 + kw
            // For mmul B: need 8ic × 8oc with stride
            aie::vector<bfloat16, MMUL::size_B> B;
            for (int i = 0; i < 8; i++) {
              aie::vector<bfloat16, 8> row;
              for (int j = 0; j < 8; j++) {
                int w_idx = ((oc_blk * 8 + j) * ic + (ic_blk * 8 + i)) * 9 + kh * 3 + kw;
                row[j] = weights[w_idx];
              }
              B.insert(i, row);
            }

            acc.mac(A, B);
          }
        }
      }

      // BN + SiLU
      aie::vector<bfloat16, MMUL::size_C> result = acc.template to_vector<bfloat16>();
      for (int p = 0; p < 4; p++) {
        int pidx = sp + p;
        if (pidx < spatial_out) {
          aie::vector<bfloat16, 8> row = result.extract<8>(p);
          aie::accum<accfloat, 8> bn_acc = aie::mul(row, bn_w_vec);
          aie::vector<bfloat16, 8> bn_out = aie::add(bn_acc.to_vector<bfloat16>(), bn_b_vec);
          for (int j = 0; j < 8; j++) {
            float x = (float)bn_out[j];
            float sig = 0.5f + x / (2.0f * (1.0f + (x > 0 ? x : -x)));
            output[pidx * oc + oc_blk * 8 + j] = (bfloat16)(x * sig);
          }
        }
      }
    }
  }

  event1();
}

extern "C" {

void conv1x1_fused_packed_vec_bf16(bfloat16 *input,
                                    bfloat16 *packed_weights,
                                    bfloat16 *output,
                                    int32_t tile_h, int32_t tile_w,
                                    int32_t ic, int32_t oc,
                                    int32_t stride, int32_t padding) {
  conv1x1_vec_v2_bf16(input, packed_weights, output,
                       tile_h, tile_w, ic, oc, stride, padding);
}

void conv3x3_fused_packed_vec_bf16(bfloat16 *input,
                                    bfloat16 *packed_weights,
                                    bfloat16 *output,
                                    int32_t tile_h, int32_t tile_w,
                                    int32_t ic, int32_t oc,
                                    int32_t stride, int32_t padding) {
  conv3x3_vec_bf16(input, packed_weights, output,
                    tile_h, tile_w, ic, oc, stride, padding);
}

} // extern "C"
