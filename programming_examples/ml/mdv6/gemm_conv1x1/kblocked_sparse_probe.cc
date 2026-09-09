// Diagnostic wrapper only: compile/link the actual production kernel so fixes
// are exercised here too. Output exposes first-K partial and final-K result.
#include "../kernels/rep_elan_bf16.cc"

// Keep dynamic extraction in a distinct externally visible codegen context.
// Inlining/CSE with adjacent literal traces can incidentally avoid the defect.
extern "C" __attribute__((noinline)) void store_rows_dynamic(
    aie::vector<bfloat16, 32> result, bfloat16 *output, int rows) {
#pragma clang loop unroll(disable)
  for (int row = 0; row < rows; ++row)
    aie::store_v(output + row * 8, result.extract<8>(row));
}

extern "C" void kblocked_sparse_probe(bfloat16 *input, bfloat16 *weights,
                                       bfloat16 *output) {
  gemm_conv1x1_kblocked_bf16(input, weights, output, 4, 16, 8, 0, 8, 2);
  // Retain an observable first-K partial before importing it into the next MAC.
  for (int i = 0; i < 32; ++i) output[32 + i] = output[i];
  // AIE2P load_v<64> requires 64-byte alignment. A compact 80-bf16 chunk
  // would place chunk 1 at byte 160 (misaligned): pad each to 96 bf16.
  // Production KB16/OC128 chunks already satisfy this requirement.
  static_assert((96 * sizeof(bfloat16)) % aie::vector_ldst_align_v<bfloat16, 64> == 0);
  gemm_conv1x1_kblocked_bf16(input, weights + 96, output + 32,
                            4, 16, 8, 8, 8, 2);
  aie::vector<bfloat16, 32> a;
  for (int row = 0; row < 4; ++row)
    a.insert(row, aie::load_v<8>(input + row * 16));
  aie::store_v(output + 64, a);
  aie::vector<bfloat16, 32> transposed = ::shuffle(a, a, T16_4x8);
  aie::store_v(output + 96, transposed);
  // Mirror the installed API's emulated MMUL broadcast stage explicitly.
  auto trace_column = [&]<int Col>() {
    aie::vector<bfloat16, 32> repeated = ::shuffle(
        ::extract_v4bfloat16_broadcast_to_v32bfloat16(transposed, Col), T16_8x4);
    aie::store_v(output + 128 + Col * 32, repeated);
  };
  trace_column.template operator()<0>();
  trace_column.template operator()<1>();
  trace_column.template operator()<2>();
  trace_column.template operator()<3>();
  trace_column.template operator()<4>();
  trace_column.template operator()<5>();
  trace_column.template operator()<6>();
  trace_column.template operator()<7>();
  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;
  MMUL imported(aie::load_v<32>(output));
  aie::store_v(output + 384, imported.to_vector<bfloat16>());
  aie::vector<bfloat16, 32> raw_broadcast =
      ::extract_v4bfloat16_broadcast_to_v32bfloat16(transposed, 0);
  aie::store_v(output + 416, raw_broadcast);
  auto b = aie::load_v<64>(weights);
  MMUL direct(aie::zeros<bfloat16, 32>());
  direct.mac(a, b);
  auto result = direct.to_vector<bfloat16>();
  aie::store_v(output + 448, result);
  store_rows_dynamic(result, output + 480, 4);
  aie::store_v(reinterpret_cast<float *>(output + 512),
               direct.to_accum().to_vector<float>());
  MMUL product;
  product.mul(a, b);
  aie::store_v(output + 576, product.to_vector<bfloat16>());
  auto native_product = ::mul_4x8_8x8_bf16(a, b);
  aie::store_v(output + 608, native_product.to_vector<bfloat16>());
  // Constant extract indices avoid the compiler's dynamic-extraction defect.
  // Keep the dynamic version above as a separately reported reproducer.
  aie::store_v(output + 640, result.extract<8>(0));
  aie::store_v(output + 648, result.extract<8>(1));
  aie::store_v(output + 656, result.extract<8>(2));
  aie::store_v(output + 664, result.extract<8>(3));
  // Observe, never change, the rounding mode used by production conversion.
  // Metadata is raw uint16, not a bf16 numerical value.
  auto metadata = reinterpret_cast<uint16_t *>(output + 672);
  for (int i = 0; i < 32; ++i) metadata[i] = 0;
  metadata[0] = static_cast<uint16_t>(get_rnd());
}
