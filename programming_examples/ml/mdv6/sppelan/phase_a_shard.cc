// One full-spatial SPP9 channel shard. Reuse production GEMM arithmetic;
// no new convolution approximation or rounding-mode setting is introduced.
#include "../kernels/rep_elan_bf16.cc"

namespace {
constexpr int kPixels = 400;
constexpr int kChannels = 8;
constexpr int kPlane = kPixels * kChannels;
constexpr int kChunkStride = 1056;
static_assert((kChunkStride * sizeof(bfloat16)) %
                  aie::vector_ldst_align_v<bfloat16, 64> == 0);

void pool_planes(bfloat16 *planes, uint16_t *metadata) {
  for (int level = 1; level < 4; ++level) {
    const bfloat16 *input = planes + (level - 1) * kPlane;
    bfloat16 *output = planes + level * kPlane;
    for (int y = 0; y < 20; ++y) {
      for (int x = 0; x < 20; ++x) {
        for (int channel = 0; channel < kChannels; ++channel) {
          float maximum = -__builtin_inff();
          for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
              const int iy = y + dy;
              const int ix = x + dx;
              if (iy >= 0 && iy < 20 && ix >= 0 && ix < 20) {
                const float value = (float)input[(iy * 20 + ix) * kChannels + channel];
                if (value > maximum) maximum = value;
              }
            }
          }
          output[(y * 20 + x) * kChannels + channel] = (bfloat16)maximum;
        }
      }
    }
  }
  for (int i = 0; i < 32; ++i) metadata[i] = 0;
  metadata[0] = static_cast<uint16_t>(get_rnd());
}
} // namespace

extern "C" void phase_a_project_stripe(bfloat16 *input, bfloat16 *weights,
                                         bfloat16 *planes, int32_t stripe) {
  bfloat16 *output = planes + stripe * 16 * kChannels;
  // Match deployed SPP conv1 KB128 reduction, including its bf16 partial
  // write after K0..127 and BN/SiLU only after K128..255.
  gemm_conv1x1_kblocked_bf16(input, weights, output,
                            16, 256, 8, 0, 128, 2);
  gemm_conv1x1_kblocked_bf16(input, weights + kChunkStride, output,
                            16, 256, 8, 128, 128, 2);
}

extern "C" void phase_a_pool_planes(bfloat16 *planes, uint16_t *metadata) {
  pool_planes(planes, metadata);
}

extern "C" void phase_a_pool_only(bfloat16 *input, bfloat16 *planes,
                                   uint16_t *metadata) {
  for (int i = 0; i < kPlane; ++i) planes[i] = input[i];
  pool_planes(planes, metadata);
}
