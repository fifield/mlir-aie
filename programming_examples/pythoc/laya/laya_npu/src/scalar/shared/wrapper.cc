
#include <stdint.h>
#ifndef M8B_HOST
#include <aie_api/aie.hpp>
#endif
extern "C" void residual(const uint16_t *src, uint16_t *dst);
extern "C" void laya_layer_norm(const uint16_t *x, uint32_t *y);
extern "C" void cast_bias(const uint32_t *src, uint16_t *dst);
extern "C" void m8b_crrnd_set(void);
#ifdef M8B_HOST
#include "host_stubs.h"
#endif
alignas(64) static uint32_t m8b_ln32[1024];
alignas(64) static uint32_t m8b_stage[512];
#ifdef M8B_MUT_SWAP_AB
alignas(64) static uint16_t m8b_swap[512];
#endif

static inline void m8b_copy(uint32_t *__restrict d,
                            const uint32_t *__restrict s, int words) {
#ifdef M8B_HOST
  host_copy(d, s, words);
#else
  for (int i = 0; i < words; i += 16)
    aie::store_v(d + i, aie::load_v<16>(s + i));
#endif
}
static inline void m8b_zero(uint32_t *d, int words) {
#ifdef M8B_HOST
  host_zero(d, words);
#else
  for (int i = 0; i < words; i += 16)
    aie::store_v(d + i, aie::zeros<uint32_t, 16>());
#endif
}

#ifdef M8B_MUT_CAST_SKIPPED
static inline void m8b_convert_nobias(
    const uint32_t *src,
    uint16_t *dst) { /* mutant only: vconv without the +0 bias add */
#ifdef M8B_HOST
  for (int i = 0; i < 256; ++i)
    dst[i] = (uint16_t)(src[i] >> 16);
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < 256; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>((const float *)src + i);
    aie::accum<accfloat, 16> acc;
    acc.from_vector(v);
    aie::store_v((bfloat16 *)dst + i, acc.template to_vector<bfloat16>());
  }
#endif
}
#endif
/* LayerNorm of the row in tmpl's x slot -> 4 cast objects -> norm (1,024 BF16).
 */
static inline void m8b_norm(uint16_t *tmpl, uint16_t *norm) {
#if defined(M8B_MUT_GAMMA_BETA)
  m8b_copy(m8b_ln32, (const uint32_t *)(tmpl + 1024),
           512); /* gamma <-> beta swapped in L1 */
  m8b_copy((uint32_t *)(tmpl + 1024), (const uint32_t *)(tmpl + 2048), 512);
  m8b_copy((uint32_t *)(tmpl + 2048), m8b_ln32, 512);
#endif
#if defined(M8B_MUT_CXX_WRITE)
  aie::set_rounding(
      aie::rounding_mode::floor); /* C++ write in place of the thunk */
#elif !defined(M8B_NOSET)
  m8b_crrnd_set();
#endif
  laya_layer_norm(tmpl, m8b_ln32);
#if defined(M8B_MUT_CAST_SKIPPED)
  for (int c = 0; c < 4; ++c)
    m8b_convert_nobias(m8b_ln32 + 256 * c,
                       norm + 256 * c); /* conversion without the +0 bias add */
#else
  m8b_zero(m8b_stage + 256, 256);
  for (int c = 0; c < 4; ++c) {
    m8b_copy(m8b_stage, m8b_ln32 + 256 * c, 256);
    cast_bias(m8b_stage, norm + 256 * c);
  }
#endif
}

#ifndef M8B_LNF
extern "C" void m8b_lnr_row(const uint16_t *res, uint16_t *tmpl,
                            uint16_t *hidden, uint16_t *norm) {
  for (int c = 0; c < 4; ++c) {
#if defined(M8B_MUT_RES_MISALIGNED)
    residual(res + 512 * c + 8, hidden + 256 * c); /* 16 B off the object */
#elif defined(M8B_MUT_HIDDEN_ROTATED)
    residual(res + 512 * c,
             hidden + 256 * ((c + 1) & 3)); /* chunks rotated in the row */
#elif defined(M8B_MUT_SWAP_AB)
    m8b_copy((uint32_t *)m8b_swap, (const uint32_t *)(res + 512 * c + 256),
             128); /* [B | A] instead of [A | B] */
    m8b_copy((uint32_t *)(m8b_swap + 256), (const uint32_t *)(res + 512 * c),
             128);
    residual(m8b_swap, hidden + 256 * c);
#else
    residual(res + 512 * c, hidden + 256 * c);
#endif
  }
  m8b_copy((uint32_t *)tmpl, (const uint32_t *)hidden, 512);
  m8b_norm(tmpl, norm);
#if defined(M8B_TAP)
  m8b_copy((uint32_t *)hidden, m8b_ln32,
           1024); /* gate-only: FP32 row to out0 (2,048 u16) */
#endif
}
#else
#if defined(M8B_TAP)
alignas(64) static uint16_t m8b_tap_norm[1024];
#endif
extern "C" void m8b_lnf_row(const uint16_t *x, uint16_t *tmpl, uint16_t *norm) {
  m8b_copy((uint32_t *)tmpl, (const uint32_t *)x, 512);
#if defined(M8B_TAP)
  m8b_norm(tmpl, m8b_tap_norm); /* production path, normalized row kept in L1 */
  m8b_copy((uint32_t *)norm, m8b_ln32,
           1024); /* gate-only: FP32 row to out0 (2,048 u16) */
#else
  m8b_norm(tmpl, norm);
#endif
}
#endif
