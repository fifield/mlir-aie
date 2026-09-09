// Opaque-bit diagnostic only; no SPP arithmetic.
#include <stdint.h>
extern "C" void packet_gather_tag(uint16_t *planes, int32_t worker) {
  const uint16_t mask = static_cast<uint16_t>(0x1111 * (worker + 1));
  for (int i = 0; i < 12800; ++i) planes[i] ^= mask;
}
