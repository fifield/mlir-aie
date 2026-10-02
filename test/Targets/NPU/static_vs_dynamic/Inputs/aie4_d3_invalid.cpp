// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include GEN_HDR
#include <cstdint>
#include <cstdio>
#include <initializer_list>

int main() {
  // Oversized i32-scale strides previously wrapped to zero before the guard.
  // i64 values must also be checked before narrowing to the register word.
  const int64_t invalidStrides[] = {0, -1, 4194304, 134217728,
                                    (int64_t(1) << 32) + 4};
  for (int64_t stride : invalidStrides)
    if (generate_txn_main_stride_dynamic(stride)) {
      std::fprintf(stderr, "accepted invalid MDS stride %lld\n",
                   (long long)stride);
      return 1;
    }
  const int64_t invalidSizes[] = {0, -1, 4096, (int64_t(1) << 32) + 3};
  for (int64_t size : invalidSizes)
    if (generate_txn_main_size_dynamic(size)) {
      std::fprintf(stderr, "accepted invalid MDS size %lld\n", (long long)size);
      return 1;
    }
  for (int32_t length : {0, -1, 47, 134217729})
    if (generate_txn_main_length_dynamic(length)) {
      std::fprintf(stderr, "accepted invalid MDS length %d\n", length);
      return 1;
    }
  if (generate_txn_main_wide_length(INT32_MAX) ||
      generate_txn_main_wide_stride((int64_t(1) << 31) + 4)) {
    std::fprintf(stderr, "accepted overflowing wide-element BD\n");
    return 1;
  }
  if (!generate_txn_main_wide_length(48) || !generate_txn_main_wide_stride(4)) {
    std::fprintf(stderr, "rejected valid wide-element BD\n");
    return 1;
  }
  return 0;
}
