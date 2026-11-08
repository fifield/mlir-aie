#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void attention_max16(const float *scores, float *result, int valid) {
  float maximum = -3.402823466e38f;
  if (valid == 16) {
    auto values = aie::load_v<16>(scores);
    auto magnitude = aie::bit_and(0x7fffffffu, values.cast_to<uint32_t>());
    // Original ordered comparisons for zeros, subnormals, infinities and NaNs.
    bool special = !aie::lt(magnitude, 0x00800000u).empty() ||
                   !aie::ge(magnitude, 0x7f800000u).empty();
    if (!special) {
      *result = aie::reduce_max(aie::max(values, maximum));
      return;
    }
  }
  for (int i = 0; i < valid; ++i)
    if (scores[i] > maximum)
      maximum = scores[i];
  *result = maximum;
}
