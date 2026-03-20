//===- conv_bf16_vec.cc - Vectorized bf16 conv kernels for AIE2P -*-C++-*-===//
//
// Vectorized Conv+BN+SiLU using AIE mmul intrinsics for bfloat16.
// Uses mmul<4,8,8,bfloat16,bfloat16>: 4 output pixels × 8 input channels × 8 output channels
//
// Data layout:
//   Input:  HWC (height, width, channels) - channels must be multiple of 8
//   Output: HWC
//   Weights: OC×IC for 1x1, OC×IC×3×3 for 3x3 (to be tiled for mmul)
//   BN: [fused_weight(OC), fused_bias(OC)] appended after conv weights
//
//===----------------------------------------------------------------------===//

#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

// Fast sigmoid for SiLU: sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|))
inline float fast_sigmoid(float x) {
  return 0.5f + x / (2.0f * (1.0f + (x > 0 ? x : -x)));
}

//===========================================================================
// Vectorized Conv1x1 + fused BN + SiLU
//
// Input:  (tile_h × tile_w × ic) bf16, ic must be multiple of 8
// Weights: packed [conv_wts(oc×ic), bn_w(oc), bn_b(oc)] bf16
// Output: (tile_h × tile_w × oc) bf16, oc must be multiple of 8
//
// Uses mmul<4,8,8>: processes 4 spatial pixels, 8 input channels, 8 output channels
// at a time. Accumulates over ic in chunks of 8.
//===========================================================================
void conv1x1_vec_bf16(bfloat16 *__restrict input,
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
  const int wt_size = oc * ic;
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w = packed_weights + wt_size;
  bfloat16 *bn_b = bn_w + oc;

  // mmul<4,8,8>: A is 4×8 (4 pixels, 8 input channels), B is 8×8 (8 ic, 8 oc)
  // C is 4×8 (4 pixels, 8 output channels)
  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

  // Process 4 spatial positions at a time, 8 output channels at a time
  for (int oc_blk = 0; oc_blk < oc; oc_blk += 8) {
    // Load BN params for this oc block
    aie::vector<bfloat16, 8> bn_w_vec = aie::load_v<8>(bn_w + oc_blk);
    aie::vector<bfloat16, 8> bn_b_vec = aie::load_v<8>(bn_b + oc_blk);

    for (int sp = 0; sp < spatial; sp += 4) {
      // Initialize accumulator to zero
      MMUL acc(aie::zeros<bfloat16, MMUL::size_C>());

      // Accumulate over input channels in chunks of 8
      for (int ic_blk = 0; ic_blk < ic; ic_blk += 8) {
        // Load 4 pixels × 8 input channels
        // Input layout is HWC: pixel[sp+i] has channels at input[(sp+i)*ic + ic_blk]
        aie::vector<bfloat16, MMUL::size_A> A;
        for (int p = 0; p < 4; p++) {
          int pixel_idx = sp + p;
          if (pixel_idx < spatial) {
            auto chunk = aie::load_v<8>(input + pixel_idx * ic + ic_blk);
            A.insert(p, chunk);
          }
        }

        // Load 8×8 weight block: weights[oc_blk:oc_blk+8, ic_blk:ic_blk+8]
        // Weight layout: weights[oc * ic], stored as oc-major
        // For mmul, B should be ic(8) × oc(8)
        // But weights are stored as [oc][ic], so we need to load transposed
        // Actually, mmul<4,8,8> expects A(4×8) × B(8×8) = C(4×8)
        // A rows = spatial pixels, A cols = input channels
        // B rows = input channels, B cols = output channels
        // So B[ic_blk:ic_blk+8, oc_blk:oc_blk+8] maps to weights with stride
        aie::vector<bfloat16, MMUL::size_B> B;
        for (int i = 0; i < 8; i++) {
          // Row i of B = ic_blk+i, cols = oc_blk..oc_blk+7
          // weights[(oc_blk+j) * ic + (ic_blk+i)] for j=0..7
          // We need to gather 8 values with stride=ic
          aie::vector<bfloat16, 8> row;
          for (int j = 0; j < 8; j++) {
            int w_idx = (oc_blk + j) * ic + (ic_blk + i);
            row[j] = weights[w_idx];
          }
          B.insert(i, row);
        }

        acc.mac(A, B);
      }

      // Extract result, apply BN + SiLU, store
      aie::vector<bfloat16, MMUL::size_C> result = acc.template to_vector<bfloat16>();
      for (int p = 0; p < 4; p++) {
        int pixel_idx = sp + p;
        if (pixel_idx < spatial) {
          for (int j = 0; j < 8; j++) {
            float val = (float)result[p * 8 + j];
            float bw = (float)bn_w_vec[j];
            float bb = (float)bn_b_vec[j];
            float bn_out = bw * val + bb;
            float silu_out = bn_out * fast_sigmoid(bn_out);
            output[pixel_idx * oc + oc_blk + j] = (bfloat16)silu_out;
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
  conv1x1_vec_bf16(input, packed_weights, output,
                    tile_h, tile_w, ic, oc, stride, padding);
}

} // extern "C"
