// Exact opaque-bit phase/stripe diagnostic; no SPP arithmetic.
#include <stdint.h>
extern "C" void phase_alias_tag(uint16_t *arena, int32_t length, int32_t mask) {
  for (int i = 0; i < length; ++i) arena[i] ^= static_cast<uint16_t>(mask);
}
