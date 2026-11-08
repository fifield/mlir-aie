
#include <aie_api/aie.hpp>
#include <stdint.h>
/* ReLU on 256 BF16 bit patterns as int16: max(x, 0) keeps x when the sign bit
 * is clear and gives +0 otherwise (all 65,536 patterns). */
extern "C" void m13b_relu256(uint16_t *p) {
  _Pragma("clang loop unroll(disable)") for (int i = 0; i < 256; i += 32) {
    aie::vector<int16_t, 32> v = aie::load_v<32>((const int16_t *)p + i);
    aie::store_v((int16_t *)p + i, aie::max(v, aie::zeros<int16_t, 32>()));
  }
}
