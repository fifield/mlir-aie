// Fast dense leaf (M30): each converted B tile is reused across two M tiles.
// Every output keeps the n2 leaf's K order and conversions, so results are
// bitwise identical to mmul_reuse.cc with MATRIX_NBLOCK=2.
#include <aie_api/aie.hpp>
using Mat = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8, accfloat>;
static __attribute__((always_inline)) inline Mat initial(float *c) {
  aie::accum<accfloat, 64> x;
  x.from_vector(aie::load_v<64>(c));
  return Mat(x);
}
extern "C" void laya_gemm_native(const bfloat16 *__restrict a,
                                 const bfloat16 *__restrict b,
                                 float *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int m = 0; m < 4; m += 2)
    for (int n = 0; n < 8; n += 2) {
      int off = (m * 8 + n) * 64;
      Mat c00 = initial(c + off), c01 = initial(c + off + 64);
      Mat c10 = initial(c + off + 512), c11 = initial(c + off + 576);
#pragma clang loop unroll(disable)
      for (int k = 0; k < 4; ++k) {
        aie::accum<accfloat, 64> aa0(aie::load_v<64>(a + (m * 4 + k) * 64));
        aie::accum<accfloat, 64> aa1(
            aie::load_v<64>(a + ((m + 1) * 4 + k) * 64));
        auto av0 = aa0.to_vector<bfp16ebs8>();
        auto av1 = aa1.to_vector<bfp16ebs8>();
        const bfloat16 *bp = b + (k * 8 + n) * 64;
        aie::accum<accfloat, 64> bb0(aie::transpose(aie::load_v<64>(bp), 8, 8));
        aie::accum<accfloat, 64> bb1(
            aie::transpose(aie::load_v<64>(bp + 64), 8, 8));
        auto bv0 = bb0.to_vector<bfp16ebs8>();
        auto bv1 = bb1.to_vector<bfp16ebs8>();
        c00.mac(av0, aie::op_transpose(bv0));
        c01.mac(av0, aie::op_transpose(bv1));
        c10.mac(av1, aie::op_transpose(bv0));
        c11.mac(av1, aie::op_transpose(bv1));
      }
      aie::store_v(c + off, c00.to_vector<float>());
      aie::store_v(c + off + 64, c01.to_vector<float>());
      aie::store_v(c + off + 512, c10.to_vector<float>());
      aie::store_v(c + off + 576, c11.to_vector<float>());
    }
}
