// M1 independent C++ leaves. Same tiled ABI, independently expressed
// arithmetic. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void laya_gemm(const bfloat16 *__restrict a,
                          const bfloat16 *__restrict b, float *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int m = 0; m < 4; ++m)
    for (int n = 0; n < 8; ++n) {
      int off = (m * 8 + n) * 64;
      aie::accum<accfloat, 64> initial;
      initial.from_vector(aie::load_v<64>(c + off));
      aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat> acc(initial);
      for (int k = 0; k < 4; ++k)
        acc.mac(aie::load_v<64>(a + (m * 4 + k) * 64),
                aie::load_v<64>(b + (k * 8 + n) * 64));
      aie::store_v(c + off, acc.to_vector<float>());
    }
}

static inline aie::vector<float, 16> widen(const bfloat16 *x) {
  aie::accum<accfloat, 16> a(aie::load_v<16>(x));
  return a.to_vector<float>();
}

// Adapted from aie_kernels/aie2p/layer_norm.cc's centered FP32 affine path.
extern "C" void laya_layer_norm(const bfloat16 *__restrict x,
                                float *__restrict y) {
  aie::vector<float, 16> sum = aie::zeros<float, 16>();
  for (int i = 0; i < 1024; i += 16)
    sum = aie::add(sum, widen(x + i));
  float mean = aie::reduce_add(sum) / 1024.0f;
  auto mv = aie::broadcast<float, 16>(mean);
  aie::vector<float, 16> var = aie::zeros<float, 16>();
  for (int i = 0; i < 1024; i += 16) {
    auto d = aie::sub(widen(x + i), mv);
    aie::vector<float, 16> square = aie::mul(d, d);
    var = aie::add(var, square);
  }
  float inv = 1.0f / ::_sqrtf(aie::reduce_add(var) / 1024.0f + 1e-5f);
  auto iv = aie::broadcast<float, 16>(inv);
  for (int i = 0; i < 1024; i += 16) {
    auto d = aie::sub(widen(x + i), mv);
    aie::vector<float, 16> normalized = aie::mul(d, iv);
    aie::vector<float, 16> scaled = aie::mul(normalized, widen(x + 1024 + i));
    aie::store_v(y + i, aie::add(scaled, widen(x + 2048 + i)));
  }
}

// Adapted from rope_halfsplit: retain products and their sum in FP32.
extern "C" void laya_rope(const bfloat16 *__restrict x, float *__restrict y) {
  auto a = aie::load_v<32>(x), b = aie::load_v<32>(x + 32);
  auto c = aie::load_v<32>(x + 64), s = aie::load_v<32>(x + 96);
  auto lo = aie::sub(aie::mul(a, c), aie::mul(b, s));
  auto hi = aie::add(aie::mul(a, s), aie::mul(b, c));
  aie::store_v(y, lo.to_vector<float>());
  aie::store_v(y + 32, hi.to_vector<float>());
}
