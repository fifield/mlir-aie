//===- gemm_conv1x1_bf16.cc - GEMM-based 1x1 conv for AIE2P ------*-C++-*-===//
//
// Vectorized Conv1x1 + BN + SiLU using aie::mmul<8,8,8> with 2x2 register
// blocking. Maps 1x1 convolution directly to matrix multiply:
//
//   Output[M, N] = Input[M, K] x Weights[K, N]
//   where M = tile_h * tile_w (spatial), K = ic, N = oc
//
// Uses 8x8x8 bf16 mmul (BFP16 on AIE2P hardware) for full 512-multiplier
// throughput. 2x2 register blocking gives 2x load reuse:
//   A0 reused for C00 and C01, B0 reused for C00 and C10.
//
// Input layout:  HWC (row-major [M, K]) -- no pre-blocking needed
// Weight layout: [oc/8, ic/8, 8, 8] bf16 (pre-packed offline)
//   Packed buffer: [conv_weights, bn_w(oc), bn_b(oc)]
// Output layout: HWC (row-major [M, N])
//
// Constraints:
//   tile_h * tile_w must be divisible by 16
//   ic must be divisible by 8
//   oc must be divisible by 16
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

// --- Pure GEMM kernel (no BN/SiLU) ---
//
// C[M,N] += A[M,K] * B[K,N]
//
// A in HWC order: A[m,k] at input[m * ic + k]
// B pre-packed:   B[n_blk, k_blk, 8, 8] at weights[(n_blk * K/8 + k_blk) * 64]
// C in HWC order: C[m,n] at output[m * oc + n]
//
template <int TILE_M, int IC, int OC>
static inline void gemm_bf16_8x8x8(const bfloat16 *__restrict input,
                                    const bfloat16 *__restrict weights,
                                    bfloat16 *__restrict output) {

  constexpr int M_BLOCKS = TILE_M / 8;
  constexpr int K_BLOCKS = IC / 8;
  constexpr int N_BLOCKS = OC / 8;

  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16>;

  // 2x2 register blocking over M and N
  for (int m = 0; m < M_BLOCKS; m += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      for (int n = 0; n < N_BLOCKS; n += 2) {

        // Initialize 4 accumulators to zero
        MMUL C00(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C01(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C10(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C11(aie::zeros<bfloat16, MMUL::size_C>());

        // Weight pointers for the two N-blocks (K is contiguous within each)
        const bfloat16 *pB0 = weights + n * K_BLOCKS * 64;
        const bfloat16 *pB1 = weights + (n + 1) * K_BLOCKS * 64;

        // K-reduction loop
        for (int k = 0; k < K_BLOCKS; k++)
          chess_prepare_for_pipelining chess_loop_range(2, ) {

            // Load A0: 8 pixels [m*8..m*8+7], 8 channels [k*8..k*8+7]
            // Gather from HWC: 8 loads of 8 contiguous channels
            aie::vector<bfloat16, MMUL::size_A> A0;
            chess_unroll_loop() for (int p = 0; p < 8; p++) {
              A0.insert(p, aie::load_v<8>(input + (m * 8 + p) * IC + k * 8));
            }

            // Load A1: next 8 pixels [(m+1)*8..(m+1)*8+7]
            aie::vector<bfloat16, MMUL::size_A> A1;
            chess_unroll_loop() for (int p = 0; p < 8; p++) {
              A1.insert(p,
                        aie::load_v<8>(input + ((m + 1) * 8 + p) * IC + k * 8));
            }

            // Load B0, B1: contiguous 64-element blocks from pre-packed weights
            aie::vector<bfloat16, MMUL::size_B> B0 = aie::load_v<64>(pB0);
            pB0 += 64;
            aie::vector<bfloat16, MMUL::size_B> B1 = aie::load_v<64>(pB1);
            pB1 += 64;

            // 4 MACs with 2x data reuse
            C00.mac(A0, B0);
            C01.mac(A0, B1); // A0 reused
            C10.mac(A1, B0); // B0 reused
            C11.mac(A1, B1);
          }

        // Store results in HWC order
        // C00: pixels m*8..m*8+7, channels n*8..n*8+7
        aie::vector<bfloat16, MMUL::size_C> r00 =
            C00.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r01 =
            C01.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r10 =
            C10.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r11 =
            C11.template to_vector<bfloat16>();

        // Scatter-store: each 8-element row goes to a different pixel's channel
        // slice. r[p] = 8 output channels for pixel p.
        chess_unroll_loop() for (int p = 0; p < 8; p++) {
          aie::store_v(output + (m * 8 + p) * OC + n * 8, r00.extract<8>(p));
          aie::store_v(output + (m * 8 + p) * OC + (n + 1) * 8,
                       r01.extract<8>(p));
          aie::store_v(output + ((m + 1) * 8 + p) * OC + n * 8,
                       r10.extract<8>(p));
          aie::store_v(output + ((m + 1) * 8 + p) * OC + (n + 1) * 8,
                       r11.extract<8>(p));
        }
      }
    }
}

// --- GEMM Conv1x1 + fused BN + SiLU ---
//
// Same GEMM core as above but with fused batch norm and SiLU activation
// applied per output-channel block after the matmul completes.
//
// Packed weight buffer: [conv_weights(oc*ic), bn_w(oc), bn_b(oc)]
//   where conv_weights are in [oc/8, ic/8, 8, 8] blocked layout
//
void gemm_conv1x1_fused_bf16(bfloat16 *__restrict input,
                              bfloat16 *__restrict packed_weights,
                              bfloat16 *__restrict output,
                              const int32_t tile_m, const int32_t ic,
                              const int32_t oc) {
  event0();

  const int wt_size = oc * ic;
  bfloat16 *weights = packed_weights;
  bfloat16 *bn_w_ptr = packed_weights + wt_size;
  bfloat16 *bn_b_ptr = bn_w_ptr + oc;

  const int m_blocks = tile_m / 8;
  const int k_blocks = ic / 8;
  const int n_blocks = oc / 8;

  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16>;

  // 2x2 register blocking
  for (int m = 0; m < m_blocks; m += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      for (int n = 0; n < n_blocks; n += 2) {

        MMUL C00(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C01(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C10(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C11(aie::zeros<bfloat16, MMUL::size_C>());

        const bfloat16 *pB0 = weights + n * k_blocks * 64;
        const bfloat16 *pB1 = weights + (n + 1) * k_blocks * 64;

        for (int k = 0; k < k_blocks; k++)
          chess_prepare_for_pipelining chess_loop_range(2, ) {

            aie::vector<bfloat16, MMUL::size_A> A0;
            chess_unroll_loop() for (int p = 0; p < 8; p++) {
              A0.insert(p, aie::load_v<8>(input + (m * 8 + p) * ic + k * 8));
            }
            aie::vector<bfloat16, MMUL::size_A> A1;
            chess_unroll_loop() for (int p = 0; p < 8; p++) {
              A1.insert(p,
                        aie::load_v<8>(input + ((m + 1) * 8 + p) * ic + k * 8));
            }

            aie::vector<bfloat16, MMUL::size_B> B0 = aie::load_v<64>(pB0);
            pB0 += 64;
            aie::vector<bfloat16, MMUL::size_B> B1 = aie::load_v<64>(pB1);
            pB1 += 64;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }

        // Extract + BN + SiLU
        aie::vector<bfloat16, MMUL::size_C> r00 =
            C00.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r01 =
            C01.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r10 =
            C10.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r11 =
            C11.template to_vector<bfloat16>();

        // BN params for the two output-channel blocks
        aie::vector<bfloat16, 8> bn_w0 = aie::load_v<8>(bn_w_ptr + n * 8);
        aie::vector<bfloat16, 8> bn_b0 = aie::load_v<8>(bn_b_ptr + n * 8);
        aie::vector<bfloat16, 8> bn_w1 =
            aie::load_v<8>(bn_w_ptr + (n + 1) * 8);
        aie::vector<bfloat16, 8> bn_b1 =
            aie::load_v<8>(bn_b_ptr + (n + 1) * 8);

// Apply BN + SiLU to each 8-element row and scatter-store
#define BN_SILU_STORE(result, bn_w, bn_b, pixel_base, chan_base)               \
  chess_unroll_loop() for (int p = 0; p < 8; p++) {                            \
    aie::vector<bfloat16, 8> row = result.extract<8>(p);                       \
    aie::accum<accfloat, 8> bn_acc = aie::mul(row, bn_w);                     \
    aie::vector<bfloat16, 8> bn_out =                                          \
        aie::add(bn_acc.to_vector<bfloat16>(), bn_b);                          \
    bfloat16 *out_ptr = output + (pixel_base + p) * oc + chan_base;            \
    for (int j = 0; j < 8; j++) {                                              \
      float x = (float)bn_out[j];                                              \
      float ax = x > 0.0f ? x : -x;                                           \
      out_ptr[j] = (bfloat16)(x * (0.5f + x / (2.0f + 2.0f * ax)));          \
    }                                                                           \
  }

        BN_SILU_STORE(r00, bn_w0, bn_b0, m * 8, n * 8);
        BN_SILU_STORE(r01, bn_w1, bn_b1, m * 8, (n + 1) * 8);
        BN_SILU_STORE(r10, bn_w0, bn_b0, (m + 1) * 8, n * 8);
        BN_SILU_STORE(r11, bn_w1, bn_b1, (m + 1) * 8, (n + 1) * 8);

#undef BN_SILU_STORE
      }
    }

  event1();
}

// --- Dynamic-size wrapper ---
//
// Called from IRON with runtime tile_m, ic, oc.
// Since aie::mmul shapes are compile-time, we dispatch to the dynamic version.
//
void gemm_conv1x1_dynamic_bf16(bfloat16 *__restrict input,
                                bfloat16 *__restrict packed_weights,
                                bfloat16 *__restrict output,
                                const int32_t tile_m, const int32_t ic,
                                const int32_t oc) {
  gemm_conv1x1_fused_bf16(input, packed_weights, output, tile_m, ic, oc);
}

extern "C" {

// Entry point for IRON Kernel() — matches the existing fused packed interface
// but with GEMM-optimized weight layout [oc/8, ic/8, 8, 8] instead of
// [ic/8, oc/8, 8, 8].
//
// Args: input, packed_weights, output, tile_h, tile_w, ic, oc, stride(unused),
// padding(unused)
void gemm_conv1x1_fused_packed_bf16(bfloat16 *input, bfloat16 *packed_weights,
                                     bfloat16 *output, int32_t tile_h,
                                     int32_t tile_w, int32_t ic, int32_t oc,
                                     int32_t stride, int32_t padding) {
  int32_t tile_m = tile_h * tile_w;
  gemm_conv1x1_fused_bf16(input, packed_weights, output, tile_m, ic, oc);
}

// Pure GEMM entry point (no BN/SiLU) — for testing and non-fused layers
void gemm_conv1x1_bf16(bfloat16 *input, bfloat16 *weights, bfloat16 *output,
                        int32_t tile_h, int32_t tile_w, int32_t ic,
                        int32_t oc, int32_t stride, int32_t padding) {
  event0();

  int32_t tile_m = tile_h * tile_w;
  int m_blocks = tile_m / 8;
  int k_blocks = ic / 8;
  int n_blocks = oc / 8;

  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16>;

  for (int m = 0; m < m_blocks; m += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      for (int n = 0; n < n_blocks; n += 2) {

        MMUL C00(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C01(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C10(aie::zeros<bfloat16, MMUL::size_C>());
        MMUL C11(aie::zeros<bfloat16, MMUL::size_C>());

        const bfloat16 *pB0 = weights + n * k_blocks * 64;
        const bfloat16 *pB1 = weights + (n + 1) * k_blocks * 64;

        for (int k = 0; k < k_blocks; k++)
          chess_prepare_for_pipelining chess_loop_range(2, ) {
            aie::vector<bfloat16, MMUL::size_A> A0;
            chess_unroll_loop() for (int p = 0; p < 8; p++) {
              A0.insert(p, aie::load_v<8>(input + (m * 8 + p) * ic + k * 8));
            }
            aie::vector<bfloat16, MMUL::size_A> A1;
            chess_unroll_loop() for (int p = 0; p < 8; p++) {
              A1.insert(p,
                        aie::load_v<8>(input + ((m + 1) * 8 + p) * ic + k * 8));
            }
            aie::vector<bfloat16, MMUL::size_B> B0 = aie::load_v<64>(pB0);
            pB0 += 64;
            aie::vector<bfloat16, MMUL::size_B> B1 = aie::load_v<64>(pB1);
            pB1 += 64;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }

        // Store in HWC order
        aie::vector<bfloat16, MMUL::size_C> r00 =
            C00.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r01 =
            C01.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r10 =
            C10.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> r11 =
            C11.template to_vector<bfloat16>();

        chess_unroll_loop() for (int p = 0; p < 8; p++) {
          aie::store_v(output + (m * 8 + p) * oc + n * 8, r00.extract<8>(p));
          aie::store_v(output + (m * 8 + p) * oc + (n + 1) * 8,
                       r01.extract<8>(p));
          aie::store_v(output + ((m + 1) * 8 + p) * oc + n * 8,
                       r10.extract<8>(p));
          aie::store_v(output + ((m + 1) * 8 + p) * oc + (n + 1) * 8,
                       r11.extract<8>(p));
        }
      }
    }

  event1();
}

} // extern "C"
