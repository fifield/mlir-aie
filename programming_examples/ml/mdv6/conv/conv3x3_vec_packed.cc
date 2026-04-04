//===- conv3x3_vec_packed.cc - Vectorized 3x3 conv with packed weights --*-C++-*-===//
//
// Vectorized Conv3x3 + BN + SiLU using mmul<4,8,8> with pre-packed weights.
//
// Weight layout (pre-packed on host):
//   [OC/8, IC/8, 9, 8ic, 8oc] — each (oc_blk, ic_blk, kpos) has a contiguous
//   64-element block ready for aie::load_v<64>. kpos = kh*3+kw.
//   Packed buffer: [packed_conv_weights, bn_w(OC), bn_b(OC)]
//
// Input/Output layout: HWC (channels must be multiple of 8)
// Input is a spatial patch of size (patch_h × patch_w × IC) where
//   patch_h = (tile_h - 1) * stride + 3
//   patch_w = (tile_w - 1) * stride + 3
//
// Same function signature as the scalar conv3x3_fused_packed_bf16 —
// drop-in replacement via different .o file.
//
//===----------------------------------------------------------------------===//

#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

void conv3x3_fused_packed_bf16(bfloat16 *__restrict input,
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
  const int wt_size = oc * ic * 9;  // total packed conv weight elements
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w_ptr = packed_weights + wt_size;
  bfloat16 *bn_b_ptr = bn_w_ptr + oc;

  const int ic_blocks = ic / 8;
  const int oc_blocks = oc / 8;

  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

  for (int oc_blk = 0; oc_blk < oc_blocks; oc_blk++) {
    // BN params for this output channel block
    aie::vector<bfloat16, 8> bn_w_vec = aie::load_v<8>(bn_w_ptr + oc_blk * 8);
    aie::vector<bfloat16, 8> bn_b_vec = aie::load_v<8>(bn_b_ptr + oc_blk * 8);

    // Weight base for this oc_blk: [ic_blocks * 9] contiguous 64-element blocks
    const bfloat16 *wt_base = weights + oc_blk * ic_blocks * 9 * 64;

    for (int sp = 0; sp < spatial_out; sp += 4) {
      MMUL acc(aie::zeros<bfloat16, MMUL::size_C>());

      // Pointer walks contiguously through weight blocks
      const bfloat16 *wt_ptr = wt_base;

      // K-reduction: over input channels and 9 kernel positions
      for (int ic_blk = 0; ic_blk < ic_blocks; ic_blk++) {
        for (int kh = 0; kh < 3; kh++) {
          for (int kw = 0; kw < 3; kw++) {
            // Load A: 4 output pixels, each shifted by (kh, kw)
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

            // Load B: contiguous 64-element block [8ic, 8oc]
            aie::vector<bfloat16, MMUL::size_B> B = aie::load_v<64>(wt_ptr);
            wt_ptr += 64;

            acc.mac(A, B);
          }
        }
      }

      // Extract result + BN + SiLU
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

// 1x1 conv — same as gemm_conv1x1_bf16.cc
void conv1x1_fused_packed_bf16(bfloat16 *__restrict input,
                                bfloat16 *__restrict packed_weights,
                                bfloat16 *__restrict output,
                                const int32_t tile_h,
                                const int32_t tile_w,
                                const int32_t ic,
                                const int32_t oc,
                                const int32_t stride_unused,
                                const int32_t padding_unused) {
  event0();

  const int tile_m = tile_h * tile_w;
  const int wt_size = oc * ic;
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w_ptr = packed_weights + wt_size;
  bfloat16 *bn_b_ptr = bn_w_ptr + oc;

  const int ic_blocks = ic / 8;
  const int oc_blocks = oc / 8;

  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

  for (int oc_blk = 0; oc_blk < oc_blocks; oc_blk++) {
    aie::vector<bfloat16, 8> bn_w_vec = aie::load_v<8>(bn_w_ptr + oc_blk * 8);
    aie::vector<bfloat16, 8> bn_b_vec = aie::load_v<8>(bn_b_ptr + oc_blk * 8);

    for (int sp = 0; sp < tile_m; sp += 4) {
      MMUL acc(aie::zeros<bfloat16, MMUL::size_C>());

      for (int ic_blk = 0; ic_blk < ic_blocks; ic_blk++) {
        aie::vector<bfloat16, MMUL::size_A> A;
        for (int p = 0; p < 4; p++) {
          int pidx = sp + p;
          if (pidx < tile_m) {
            A.insert(p, aie::load_v<8>(input + pidx * ic + ic_blk * 8));
          }
        }
        aie::vector<bfloat16, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(weights + (ic_blk * oc_blocks + oc_blk) * 64);
        acc.mac(A, B);
      }

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
    }
  }

  event1();
}

} // extern "C"
