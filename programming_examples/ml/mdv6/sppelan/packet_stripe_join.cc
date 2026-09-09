// Opaque-bit diagnostic only; not SPP projection arithmetic.
#include <stdint.h>
extern "C" void packet_stripe_join_tag(uint16_t *payload, int32_t worker) {
  const uint16_t mask = static_cast<uint16_t>(0x1111 * (worker + 1));
  for (int i = 0; i < 256; ++i) payload[i] ^= mask;
}
