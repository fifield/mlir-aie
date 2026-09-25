// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Check the packed output from shuffle_aie4.py against scalar permutations.

#include <cstdint>
#include <cstring>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Must match OUT_LANES and the output layout in shuffle_aie4.py.
constexpr size_t OUT_LANES = 96;
constexpr size_t OUT_BYTES = OUT_LANES * sizeof(int32_t);

constexpr int VEC_LANES_I32 = 16;
constexpr int VEC_LANES_I8 = 64;

static int32_t kT32_4x4[VEC_LANES_I32];
static int8_t kT8_8x8[VEC_LANES_I8];
static int32_t kZipA[VEC_LANES_I32];
static int32_t kZipB[VEC_LANES_I32];

// Match the device inputs defined in shuffle_aie4.py.
static void init_golden_inputs() {
  for (int i = 0; i < VEC_LANES_I32; ++i)
    kT32_4x4[i] = 1000 + i;
  for (int i = 0; i < VEC_LANES_I8; ++i)
    kT8_8x8[i] = static_cast<int8_t>(1 + i);
  for (int i = 0; i < VEC_LANES_I32; ++i) {
    kZipA[i] = 2000 + i;
    kZipB[i] = 3000 + i;
  }
}

// Row-major transpose: out[r, c] = in[c, r].
static void golden_transpose_4x4(const int32_t in[16], int32_t out[16]) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      out[r * 4 + c] = in[c * 4 + r];
}

static void golden_transpose_8x8(const int8_t in[64], int8_t out[64]) {
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c)
      out[r * 8 + c] = in[c * 8 + r];
}

// Interleave eight lanes from each input, starting at base (0 or 8).
static void golden_zip(const int32_t a[16], const int32_t b[16], int base,
                        int32_t out[16]) {
  for (int i = 0; i < 8; ++i) {
    out[2 * i] = a[base + i];
    out[2 * i + 1] = b[base + i];
  }
}

int main() {
  init_golden_inputs();

  int32_t exp_4x4[VEC_LANES_I32];
  golden_transpose_4x4(kT32_4x4, exp_4x4);
  int32_t exp_4x4_rt[VEC_LANES_I32];
  golden_transpose_4x4(exp_4x4, exp_4x4_rt);

  int8_t exp_8x8[VEC_LANES_I8];
  golden_transpose_8x8(kT8_8x8, exp_8x8);
  int8_t exp_8x8_rt[VEC_LANES_I8];
  golden_transpose_8x8(exp_8x8, exp_8x8_rt);

  int32_t exp_8x8_i32[VEC_LANES_I32];
  int32_t exp_8x8_rt_i32[VEC_LANES_I32];
  std::memcpy(exp_8x8_i32, exp_8x8, sizeof(exp_8x8_i32));
  std::memcpy(exp_8x8_rt_i32, exp_8x8_rt, sizeof(exp_8x8_rt_i32));

  int32_t exp_zip_lo[VEC_LANES_I32];
  int32_t exp_zip_hi[VEC_LANES_I32];
  golden_zip(kZipA, kZipB, 0, exp_zip_lo);
  golden_zip(kZipA, kZipB, 8, exp_zip_hi);

  // Check the reference model: transposing twice must restore the input.
  if (std::memcmp(exp_4x4_rt, kT32_4x4, sizeof(kT32_4x4)) != 0) {
    std::cout << "golden model bug: 4x4 double-transpose != identity\n";
    return 1;
  }
  if (std::memcmp(exp_8x8_rt, kT8_8x8, sizeof(kT8_8x8)) != 0) {
    std::cout << "golden model bug: 8x8 double-transpose != identity\n";
    return 1;
  }

  auto device = xrt::device(0);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"shuffle_aie4.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_out = xrt::ext::bo{device, OUT_BYTES};
  memset(bo_out.map<char *>(), 0, OUT_BYTES);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  kernel(bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t out[OUT_LANES];
  memcpy(out, bo_out.map<char *>(), OUT_BYTES);

  int errors = 0;
  auto check_block = [&](int offset, const int32_t *want, int count,
                          const char *what) {
    int block_errors = 0;
    for (int i = 0; i < count; ++i) {
      if (out[offset + i] != want[i]) {
        std::cout << "MISMATCH [" << what << "][" << i
                   << "] got=" << out[offset + i] << " want=" << want[i]
                   << std::endl;
        block_errors++;
      }
    }
    if (block_errors == 0)
      std::cout << "OK [" << what << "] all " << count << " lanes match"
                 << std::endl;
    errors += block_errors;
  };

  check_block(0, exp_4x4, VEC_LANES_I32, "t32_4x4");
  check_block(16, exp_4x4_rt, VEC_LANES_I32, "t32_4x4 round-trip == input");
  check_block(32, exp_8x8_i32, VEC_LANES_I32, "t8_8x8 (bitcast i32 words)");
  check_block(48, exp_8x8_rt_i32, VEC_LANES_I32,
              "t8_8x8 round-trip == input (bitcast i32 words)");
  check_block(64, exp_zip_lo, VEC_LANES_I32, "t32_2x16_lo");
  check_block(80, exp_zip_hi, VEC_LANES_I32, "t32_2x16_hi");

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " lane mismatches." << std::endl;
  std::cout << "fail." << std::endl;
  return 1;
}
