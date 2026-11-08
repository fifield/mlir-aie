
#include <stdint.h>
#ifndef M8B_HOST
#include <aie_api/aie.hpp>
#endif
extern "C" void cast_bias(const uint32_t *src, uint16_t *dst);
#if defined(M8B_GELU) && !defined(M8B_GELU_FAST)
extern "C" void gelu_observable(uint16_t *src, uint16_t *dst);
#endif
#ifdef M8B_HOST
#include "host_stubs.h"
#endif
#ifndef M8B_CHUNKS
#error M8B_CHUNKS
#endif
alignas(64) static uint32_t m8b_cast_stage[512];
#ifdef M8B_GELU
alignas(64) static uint16_t m8b_up16[256 * M8B_CHUNKS];
alignas(64) static uint16_t m8b_gelu_out[512];
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
/* Blocked FP32 C tile (M8B_CHUNKS x 256 words) -> the same blocked order in
 * BF16 through the accepted cast_bias. */
__attribute__((noinline)) static void m8b_cast_tile(const uint32_t *c,
                                                    uint16_t *dst) {
  m8b_zero(m8b_cast_stage + 256, 256);
  for (int ch = 0; ch < M8B_CHUNKS; ++ch) {
#ifdef M8B_MUT_CHUNK_TWICE
    m8b_copy(m8b_cast_stage, c + 256 * (ch == 1 ? 0 : ch),
             256); /* chunk 0 cast twice, chunk 1 never */
#else
    m8b_copy(m8b_cast_stage, c + 256 * ch, 256);
#endif
#ifdef M8B_MUT_CAST_SKIPPED
    m8b_convert_nobias(m8b_cast_stage, dst + 256 * ch);
#else
    cast_bias(m8b_cast_stage, dst + 256 * ch);
#endif
  }
}

#ifndef M8B_GELU
extern "C" void m8b_cast_epilogue(const uint32_t *c, uint16_t *out) {
  m8b_cast_tile(c, out);
}
#else
extern "C" void m8b_gate_epilogue(const uint32_t *c, uint16_t *gate16) {
  m8b_cast_tile(c, gate16);
}
#ifdef M8B_GELU_FAST
/* Branchless gelu_observable for one 256-element chunk, bit-identical for all
 * 65536 BF16 inputs (exhaustive on host and device). Same LUT (stage[512:],
 * 3074 entries: magnitudes 15104..16640, positive then negative), same
 * out-of-range integer rules, same gated product vmul.f -> vst.conv (crRnd 12).
 * 1. per lane: idx = clamp(mag, 15104, 16640) - 15104 (+1537 if negative);
 *    woff = byte offset of the 32-bit LUT word holding entry idx.
 * 2. scalar: one aligned 32-bit LUT load per element into word[].
 * 3. per lane: the half by idx & 1; mag < 256: srs(mag, 1) under crRnd 12
 *    (= round half to even) | sign; mag < 15104: (mag - 128) | sign;
 *    mag > 16640: x if positive else sign; then times up. */
static void m8b_gelu_chunk(const uint16_t *__restrict g,
                           const uint16_t *__restrict u,
                           const uint32_t *__restrict lut32,
                           uint16_t *__restrict woff,
                           uint32_t *__restrict word, uint16_t *__restrict y) {
  for (int i = 0; i < 256; i += 32) {
    aie::vector<int16, 32> x = aie::load_v<32>((const int16 *)g + i);
    aie::vector<int16, 32> mag = aie::bit_and((int16)0x7fff, x);
    aie::mask<32> neg = aie::lt(x, (int16)0);
    aie::vector<int16, 32> idx = aie::sub(
        aie::min(aie::max(mag, (int16)15104), (int16)16640), (int16)15104);
    idx = aie::select(idx, aie::add(idx, (int16)1537), neg);
    aie::vector<int16, 32> e = aie::bit_and((int16)0x7ffe, idx);
    aie::store_v((int16 *)woff + i, aie::add(e, e));
  }
  const char *base = (const char *)lut32;
  const uint16_t *__restrict wp = woff;
  uint32_t *__restrict dp = word;
  for (int i = 0; i < 256; i += 8) {
    uint16_t o0 = wp[0], o1 = wp[1], o2 = wp[2], o3 = wp[3];
    uint16_t o4 = wp[4], o5 = wp[5], o6 = wp[6], o7 = wp[7];
    dp[0] = *(const uint32_t *)(base + o0);
    dp[1] = *(const uint32_t *)(base + o1);
    dp[2] = *(const uint32_t *)(base + o2);
    dp[3] = *(const uint32_t *)(base + o3);
    dp[4] = *(const uint32_t *)(base + o4);
    dp[5] = *(const uint32_t *)(base + o5);
    dp[6] = *(const uint32_t *)(base + o6);
    dp[7] = *(const uint32_t *)(base + o7);
    wp += 8;
    dp += 8;
  }
  for (int i = 0; i < 256; i += 32) {
    aie::vector<int16, 32> x = aie::load_v<32>((const int16 *)g + i);
    aie::vector<int16, 32> mag = aie::bit_and((int16)0x7fff, x);
    aie::vector<int16, 32> sign = aie::bit_and((int16)0x8000, x);
    aie::mask<32> neg = aie::lt(x, (int16)0);
    aie::vector<int16, 32> idx = aie::sub(
        aie::min(aie::max(mag, (int16)15104), (int16)16640), (int16)15104);
    idx = aie::select(idx, aie::add(idx, (int16)1537), neg);
    aie::mask<32> odd = aie::neq(aie::bit_and((int16)1, idx), (int16)0);
    aie::vector<int16, 64> w = aie::load_v<64>((const int16 *)(word + i));
    aie::vector<int16, 32> r =
        aie::select(aie::filter_even(w), aie::filter_odd(w), odd);
    aie::accum<acc32, 32> a;
    a.from_vector(mag, 0);
    aie::vector<int16, 32> half = a.template to_vector<int16>(1);
    aie::vector<int16, 32> small =
        aie::select(aie::bit_or(aie::sub(mag, (int16)128), sign),
                    aie::bit_or(half, sign), aie::lt(mag, (int16)256));
    r = aie::select(r, small, aie::lt(mag, (int16)15104));
    r = aie::select(r, aie::select(x, sign, neg), aie::lt((int16)16640, mag));
    /* gelu_observable's own intrinsics: bf16 mul (conf 60), accfloat -> bf16 */
    v32bfloat16 gv = (v32bfloat16)r.template cast_to<bfloat16>();
    v32bfloat16 uv = (v32bfloat16)aie::load_v<32>((const bfloat16 *)u + i);
    *(v32bfloat16 *)(y + i) = __builtin_aie2p_v32accfloat_to_v32bf16(
        __builtin_aie2p_I512_I512_ACC1024_bf_mul_conf(gv, uv, 60));
  }
}
extern "C" void m8b_pair_epilogue(const uint32_t *c, const uint16_t *gate16,
                                  uint16_t *stage, uint16_t *out) {
  m8b_cast_tile(c, m8b_up16);
  aie::set_rounding(aie::rounding_mode::conv_even); /* gelu_observable's 12 */
  const uint32_t *lut32 = (const uint32_t *)(stage + 512);
  for (int ch = 0; ch < M8B_CHUNKS; ++ch)
    m8b_gelu_chunk(gate16 + 256 * ch, m8b_up16 + 256 * ch, lut32, stage,
                   (uint32_t *)m8b_gelu_out, out + 256 * ch);
}
#else
extern "C" void m8b_pair_epilogue(const uint32_t *c, const uint16_t *gate16,
                                  uint16_t *stage, uint16_t *out) {
  m8b_cast_tile(c, m8b_up16);
  for (int ch = 0; ch < M8B_CHUNKS; ++ch) {
#ifdef M8B_MUT_GATEUP_SWAPPED
    m8b_copy((uint32_t *)stage, (const uint32_t *)(m8b_up16 + 256 * ch),
             128); /* [up | gate] */
    m8b_copy((uint32_t *)(stage + 256), (const uint32_t *)(gate16 + 256 * ch),
             128);
#else
    m8b_copy((uint32_t *)stage, (const uint32_t *)(gate16 + 256 * ch), 128);
    m8b_copy((uint32_t *)(stage + 256), (const uint32_t *)(m8b_up16 + 256 * ch),
             128);
#endif
    gelu_observable(stage, m8b_gelu_out);
#ifdef M8B_MUT_GATED_FROM_GELU
    m8b_copy((uint32_t *)(out + 256 * ch), (const uint32_t *)m8b_gelu_out,
             128); /* the GELU half */
#else
    m8b_copy((uint32_t *)(out + 256 * ch),
             (const uint32_t *)(m8b_gelu_out + 256), 128);
#endif
  }
}
#endif
#endif
