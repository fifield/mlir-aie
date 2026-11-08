#include <aie_api/aie.hpp>
#ifndef MATRIX_NBLOCK
#define MATRIX_NBLOCK 4
#endif
using Mat = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8, accfloat>;
static __attribute__((always_inline)) inline Mat initial(float *c) {
  aie::accum<accfloat, 64> x;
  x.from_vector(aie::load_v<64>(c));
  return Mat(x);
}
static __attribute__((always_inline)) inline void
update(Mat &acc, const aie::block_vector<bfp16ebs8, 64> &a, const bfloat16 *b) {
  aie::accum<accfloat, 64> bacc(aie::transpose(aie::load_v<64>(b), 8, 8));
  auto bv = bacc.to_vector<bfp16ebs8>();
  acc.mac(a, aie::op_transpose(bv));
}
extern "C" void laya_gemm_native(const bfloat16 *__restrict a,
                                 const bfloat16 *__restrict b,
                                 float *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int m = 0; m < 4; ++m)
    for (int n = 0; n < 8; n += MATRIX_NBLOCK) {
      int off = (m * 8 + n) * 64;
      Mat c0 = initial(c + off), c1 = initial(c + off + 64);
#if MATRIX_NBLOCK == 4
      Mat c2 = initial(c + off + 128), c3 = initial(c + off + 192);
#endif
#pragma clang loop unroll(disable)
      for (int k = 0; k < 4; ++k) {
        aie::accum<accfloat, 64> aa(aie::load_v<64>(a + (m * 4 + k) * 64));
        auto av = aa.to_vector<bfp16ebs8>();
        const bfloat16 *bp = b + (k * 8 + n) * 64;
        update(c0, av, bp);
        update(c1, av, bp + 64);
#if MATRIX_NBLOCK == 4
        update(c2, av, bp + 128);
        update(c3, av, bp + 192);
#endif
      }
      aie::store_v(c + off, c0.to_vector<float>());
      aie::store_v(c + off + 64, c1.to_vector<float>());
#if MATRIX_NBLOCK == 4
      aie::store_v(c + off + 128, c2.to_vector<float>());
      aie::store_v(c + off + 192, c3.to_vector<float>());
#endif
    }
}
