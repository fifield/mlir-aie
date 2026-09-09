// Opaque-bit diagnostic only: prove each worker processes its own payload.
#include <stdint.h>
extern "C" void packet_aggregate_tag(uint16_t *planes, int32_t worker) {
  const uint16_t mask = static_cast<uint16_t>(0x1111 * (worker + 1));
  for (int i = 0; i < 12800; ++i) planes[i] ^= mask;
}
