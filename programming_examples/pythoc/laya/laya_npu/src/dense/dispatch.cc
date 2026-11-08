
#include <stdint.h>
extern "C" {
void m8b_gate_epilogue(const uint32_t *c, uint16_t *gate16);
void m8b_pair_epilogue(const uint32_t *c, const uint16_t *gate16,
                       uint16_t *stage, uint16_t *out);
void m8d_rope_epilogue(const uint32_t *c, const uint16_t *table, uint16_t *x16,
                       uint16_t *out);
void m8d_table_cos(const uint16_t *obj, uint16_t *table);
void m8d_table_sin(const uint16_t *obj, uint16_t *table);
void m9a_unblock_a(const uint16_t *obj, uint16_t *dst, int stride);
void m9a_lnr_row(const uint16_t *parts, uint16_t *tmpl, uint32_t *stage,
                 uint32_t *ln32, uint16_t *norm);
void m9a_ln_out(const uint16_t *hidden, const uint16_t *norm, uint16_t *out);
}
/* Header: nine 32-bit fields (u16 pairs, high half 0) at core words 0..8 = DDR
   (row 2i/8, column 2i%8) of the 32 x 32 object: [magic, op, units, ntab,
   ntile, npass, kt, xor(fields 0..6, npre), npre]. Only op 1, 2, 3, 5 is valid
   (A-2). An invalid header zeroes every trip count; the host never binds one
   (m9a.contract, before every dispatch). 32-bit loads only (R4). */
extern "C" void m9a_header(const uint16_t *obj, int32_t *h) {
  const int32_t *w = (const int32_t *)obj;
  int32_t x = w[8];
  for (int i = 0; i < 7; ++i)
    x = x ^ w[i];
  int32_t op = w[1];
  int32_t ok = (int32_t)(w[0] == 0x9A05) & (int32_t)(w[7] == x) &
               (int32_t)((op == 1) | (op == 2) | (op == 3) | (op == 5));
  int32_t m = -ok;
  h[0] = w[2] & m;
  h[1] = w[3] & m;
  h[2] = w[4] & m;
  h[3] = w[5] & m;
  h[4] = w[6] & m;
  h[5] = op & m;
  h[6] = ok;
  h[7] = w[8] & m;
}
/* A-only objects. col < 0: the section prelude (LN: gamma, beta -> the
   template's gamma|beta slots in the C buffer); otherwise per unit: qkvR
   tables; LN: keep this column's row (objects 2 col, 2 col + 1), release the
   others. */
extern "C" void m9a_table(const uint16_t *obj, uint16_t *table, uint32_t *c,
                          const int32_t *h, int32_t t, int32_t col) {
  if (col < 0) {
    if (h[5] == 5)
      m9a_unblock_a(obj, (uint16_t *)c + 1024 + 1024 * t, 32);
    return;
  }
  if (h[5] == 3) {
    m9a_unblock_a(obj, table + 32 * (t & 1), 64);
  } else if (h[5] == 5 && (t >> 1) == col)
    m9a_unblock_a(obj, table + 1024 * (t & 1), 32);
}
extern "C" void m9a_pass(const uint32_t *c, uint16_t *gate16, const int32_t *h,
                         int32_t p) {
  if (h[5] == 2 && p == 0)
    m8b_gate_epilogue(c, gate16);
}
extern "C" void m9a_final(uint32_t *c, uint16_t *table, uint16_t *x16,
                          const uint16_t *gate16, uint16_t *stage,
                          uint16_t *out, const int32_t *h, int32_t j) {
  int op = h[5];
  if (op == 1)
    m8b_gate_epilogue(c, out); /* M8b's cast tile (= m8b_cast_epilogue) */
  else if (op == 2)
    m8b_pair_epilogue(c, gate16, stage, out);
  else if (op == 3) {
    if (j < 4)
      m8d_rope_epilogue(c, table, x16, out);
    else
      m8b_gate_epilogue(c, out);
  } else if (op == 5) {
    uint16_t *tmpl = (uint16_t *)c;
    m9a_lnr_row(table, tmpl, c + 1536, (uint32_t *)x16,
                table); /* normalized overwrites the dead R parts */
    m9a_ln_out(tmpl, table, out);
  }
}
