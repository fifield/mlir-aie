// Preserve validated production KB128 arithmetic and pool/rounding contracts.
#include "phase_a_shard.cc"
extern "C" void phase_a_gather_project(bfloat16 *input, bfloat16 *weights,
                                        bfloat16 *packet, int32_t stripe) {
  phase_a_project_stripe(input, weights, packet, stripe);
}
extern "C" void phase_a_gather_pool(bfloat16 *packet) {
  phase_a_pool_planes(packet, reinterpret_cast<uint16_t *>(packet + 12800));
}
