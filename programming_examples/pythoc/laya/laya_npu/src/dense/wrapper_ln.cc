
#include <stdint.h>
#ifndef M9A_HOST
#include <aie_api/aie.hpp>
#endif
extern "C" void residual(const uint16_t *src, uint16_t *dst);
extern "C" void laya_layer_norm(const uint16_t *x, uint32_t *y);
extern "C" void cast_bias(const uint32_t *src, uint16_t *dst);
extern "C" void m8b_crrnd_set(void);
#ifdef M9A_HOST
#include <string.h>
static void m9a_copy(uint32_t *d, const uint32_t *s, int words) {
  memcpy(d, s, (size_t)words * 4);
}
static void m9a_zero(uint32_t *d, int words) {
  memset(d, 0, (size_t)words * 4);
}
#define M9A_NOUNROLL
#else
#define M9A_NOUNROLL _Pragma("clang loop unroll(disable)")
static void m9a_copy(uint32_t *__restrict d, const uint32_t *__restrict s,
                     int words) {
  M9A_NOUNROLL for (int i = 0; i < words; i += 16)
      aie::store_v(d + i, aie::load_v<16>(s + i));
}
static void m9a_zero(uint32_t *d, int words) {
  M9A_NOUNROLL for (int i = 0; i < words; i += 16)
      aie::store_v(d + i, aie::zeros<uint32_t, 16>());
}
#endif
static inline void m9a_run8(uint16_t *__restrict d,
                            const uint16_t *__restrict s) {
#ifdef M9A_HOST
  memcpy(d, s, 16);
#else
  aie::store_v(d, aie::load_v<8>(s));
#endif
}
static inline int m9a_blk(int i, int j, int w) {
  return ((i >> 3) * (w >> 3) + (j >> 3)) * 64 + (i & 7) * 8 + (j & 7);
}
/* A object (32 x 32, blocked [4][4][8][8] by the memtile) -> 1,024 u16 linear
 */
extern "C" void m9a_unblock_a(const uint16_t *obj, uint16_t *dst, int stride) {
  M9A_NOUNROLL for (int i = 0; i < 32; ++i) {
    M9A_NOUNROLL for (int t = 0; t < 4; ++t)
        m9a_run8(dst + stride * i + 8 * t, obj + m9a_blk(i, 8 * t, 32));
  }
}
/* One LNr row, M8b's calls in M8b's order. parts = [R.A row | R.B row] (2 x
   1,024 u16, linear); tmpl = [x | gamma | beta] (gamma|beta already staged);
   stage = 512 words; ln32 = 1,024 FP32; norm = 1,024 u16 (may alias parts: dead
   after the residuals). */
extern "C" void m9a_lnr_row(const uint16_t *parts, uint16_t *tmpl,
                            uint32_t *stage, uint32_t *ln32, uint16_t *norm) {
  uint16_t *s16 = (uint16_t *)stage;
  M9A_NOUNROLL for (int c = 0; c < 4; ++c) {
    m9a_copy((uint32_t *)s16, (const uint32_t *)(parts + 256 * c),
             128); /* [A_c | B_c]: M7a's gather object */
    m9a_copy((uint32_t *)(s16 + 256),
             (const uint32_t *)(parts + 1024 + 256 * c), 128);
    residual(s16, tmpl + 256 * c); /* hidden, straight into the x slot */
  }
#ifndef M9A_NOSET
  m8b_crrnd_set();
#endif
  laya_layer_norm(tmpl, ln32);
  m9a_zero(stage + 256, 256);
  M9A_NOUNROLL for (int c = 0; c < 4; ++c) {
    m9a_copy(stage, ln32 + 256 * c, 256);
    cast_bias(stage, norm + 256 * c);
  }
}
/* [hidden | norm] -> the output object as logical rows 0..15 | 16..31 of a 32 x
 * 64 tile, blocked [4][8][8][8] */
extern "C" void m9a_ln_out(const uint16_t *hidden, const uint16_t *norm,
                           uint16_t *out) {
  M9A_NOUNROLL for (int i = 0; i < 32; ++i) {
    const uint16_t *src = i < 16 ? hidden + 64 * i : norm + 64 * (i - 16);
    M9A_NOUNROLL for (int t = 0; t < 8; ++t)
        m9a_run8(out + m9a_blk(i, 8 * t, 64), src + 8 * t);
  }
}
