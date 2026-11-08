
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void cast_bias(const uint32_t *src, uint16_t *dst);
extern "C" void rope(const uint16_t *src, float *dst);
#ifndef M8D_M
#error M8D_M
#endif
#define M8D_CHUNKS (M8D_M * 64 / 256)
alignas(64) static uint32_t m8d_cast_stage[512];
alignas(64) static uint16_t m8d_rope_stage[128];
alignas(64) static uint16_t m8d_cast_out[256];

#define M8D_NOUNROLL _Pragma("clang loop unroll(disable)")
static void m8d_copy(uint32_t *__restrict d, const uint32_t *__restrict s,
                     int words) {
  M8D_NOUNROLL for (int i = 0; i < words; i += 16)
      aie::store_v(d + i, aie::load_v<16>(s + i));
}
static void m8d_zero(uint32_t *d, int words) {
  M8D_NOUNROLL for (int i = 0; i < words; i += 16)
      aie::store_v(d + i, aie::zeros<uint32_t, 16>());
}
/* 8 BF16 = 4 words; every offset used is a multiple of 8 elements (16 bytes).
 * Integer moves only (R4). */
static inline void m8d_run8(uint16_t *__restrict d,
                            const uint16_t *__restrict s) {
  aie::store_v(d, aie::load_v<8>(s));
}
/* blocked [m/8][w/8][8][8] position of (row i, column j) for a tile of width w
 */
static inline int m8d_blk(int i, int j, int w) {
  return ((i >> 3) * (w >> 3) + (j >> 3)) * 64 + (i & 7) * 8 + (j & 7);
}
/* 8-element runs of one row: dst[d0 + 8t] <- src[blk(i, s0 + 8t)] (gather) for
 * t < runs */
static void m8d_gather(uint16_t *dst, const uint16_t *src, int i, int s0, int w,
                       int runs) {
  M8D_NOUNROLL for (int t = 0; t < runs; ++t)
      m8d_run8(dst + 8 * t, src + m8d_blk(i, s0 + 8 * t, w));
}
static void m8d_linear(uint16_t *dst, const uint16_t *src, int runs) {
  M8D_NOUNROLL for (int t = 0; t < runs; ++t)
      m8d_run8(dst + 8 * t, src + 8 * t);
}

/* M8b's cast of a blocked FP32 C tile into the same blocked order in BF16
 * (accepted cast_bias, +0.0 bias). */
static void m8d_cast_tile(const uint32_t *c, uint16_t *dst) {
  m8d_zero(m8d_cast_stage + 256, 256);
  M8D_NOUNROLL for (int ch = 0; ch < M8D_CHUNKS; ++ch) {
    m8d_copy(m8d_cast_stage, c + 256 * ch, 256);
    cast_bias(m8d_cast_stage, dst + 256 * ch);
  }
}

/* Table object (m x 32 BF16 A object, blocked by the memtile as [m/8][4][8][8])
 * -> table [m][64] = cos | sin. */
extern "C" void m8d_table_cos(const uint16_t *obj, uint16_t *table) {
  M8D_NOUNROLL for (int i = 0; i < M8D_M; ++i)
      m8d_gather(table + i * 64, obj, i, 0, 32, 4);
}
extern "C" void m8d_table_sin(const uint16_t *obj, uint16_t *table) {
  M8D_NOUNROLL for (int i = 0; i < M8D_M; ++i)
      m8d_gather(table + i * 64 + 32, obj, i, 0, 32, 4);
}

extern "C" void m8d_cast_epilogue(const uint32_t *c, uint16_t *out) {
  m8d_cast_tile(c, out);
}

extern "C" void m8d_rope_epilogue(const uint32_t *c, const uint16_t *table,
                                  uint16_t *x16, uint16_t *out) {
  m8d_cast_tile(c, x16); /* q or k before RoPE: M8b's qkvC bits, blocked */
#ifdef M8D_MUT_ROPE_SKIPPED
  m8d_copy((uint32_t *)out, (const uint32_t *)x16, M8D_M * 32);
  return;
#endif
  M8D_NOUNROLL for (int g = 0; g < M8D_M / 4; ++g) {
    m8d_zero(m8d_cast_stage + 256,
             256); /* A-8: the +0.0 bias half, before every cast call */
    M8D_NOUNROLL for (int r = 0; r < 4; ++r) {
      int i = 4 * g + r;
#ifdef M8D_MUT_PAIR_SWAPPED
      m8d_gather(m8d_rope_stage, x16, i, 32, 64, 4);
      m8d_gather(m8d_rope_stage + 32, x16, i, 0, 64,
                 4); /* [x[32:64] | x[0:32]] */
#else
      m8d_gather(m8d_rope_stage, x16, i, 0, 64, 8); /* [x[0:32] | x[32:64]] */
#endif
#ifdef M8D_MUT_COS_SIN_SWAPPED
      m8d_linear(m8d_rope_stage + 64, table + i * 64 + 32, 4);
      m8d_linear(m8d_rope_stage + 96, table + i * 64, 4);
#else
      m8d_linear(m8d_rope_stage + 64, table + i * 64, 8); /* [cos | sin] */
#endif
      rope(m8d_rope_stage, (float *)(m8d_cast_stage + 64 * r));
    }
    cast_bias(m8d_cast_stage, m8d_cast_out); /* [4 rope outputs | +0.0]: the
                                                accepted cast_{n} object */
    M8D_NOUNROLL for (int r = 0; r < 4; ++r) {
      int i = 4 * g + r;
      M8D_NOUNROLL for (int t = 0; t < 8; ++t)
          m8d_run8(out + m8d_blk(i, 8 * t, 64), m8d_cast_out + r * 64 + 8 * t);
    }
  }
}
