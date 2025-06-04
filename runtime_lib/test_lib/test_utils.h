//===- test_utils.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the generic host code

#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

#include "cxxopts.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#if defined(__STDCPP_BFLOAT16_T__)
#include <stdfloat>
#endif
#include <ctime>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

namespace xrt {
class device;
class kernel;
class xclbin;
} // namespace xrt

namespace test_utils {

void check_arg_file_exists(const cxxopts::ParseResult &result,
                           std::string name);

void add_default_options(cxxopts::Options &options);

void parse_options(int argc, const char *argv[], cxxopts::Options &options,
                   cxxopts::ParseResult &result);

std::vector<uint32_t> load_instr_sequence(std::string instr_path);
std::vector<uint32_t> load_instr_binary(std::string instr_path);
std::vector<uint8_t> load_binary(std::string binary_path);

// Resolve the full name of the first kernel in `xclbin` whose name starts with
// `name_prefix`, which is what a host's -k argument carries (the packaged name
// is decorated, e.g. -k MLIR_AIE matches "MLIR_AIE:{...}"), so xrt::xclbin's
// own exact-match get_kernel() does not serve. Throws with the available
// kernel names listed if nothing matches.
std::string get_kernel_name(const xrt::xclbin &xclbin,
                            const std::string &name_prefix, int verbosity = 0);

void init_xrt_load_kernel(xrt::device &device, xrt::kernel &kernel,
                          int verbosity, std::string xclbinFileName,
                          std::string kernelNameInXclbin);

static inline std::int16_t random_int16_t(int32_t range = 0x10000) {
  return (std::int16_t)rand() % range;
}

static inline std::int32_t random_int32_t(int32_t range = 0x10000) {
  return (std::int32_t)rand() % range;
}

// The Linux toolchain has std::bfloat16_t. MSVC does not.
//
// Use this host-side helper for bfloat16 XRT buffers and reference checks.
// Device code should use the AIE bfloat16 types and APIs.
#if defined(__STDCPP_BFLOAT16_T__)
using bfloat16_t = std::bfloat16_t;

static inline bfloat16_t bfloat16_from_float(float value) {
  return bfloat16_t(value);
}

static inline bfloat16_t bfloat16_from_bits(std::uint16_t bits) {
  bfloat16_t value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline float bfloat16_to_float(bfloat16_t value) {
  return static_cast<float>(value);
}
#else
using bfloat16_t = std::uint16_t;

static inline bfloat16_t bfloat16_from_bits(std::uint16_t bits) { return bits; }

static inline float bfloat16_to_float(bfloat16_t bits) {
  const std::uint32_t expanded_bits = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &expanded_bits, sizeof(value));
  return value;
}

static inline bfloat16_t bfloat16_from_float(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  // Round to nearest-even instead of truncating.
  const std::uint32_t lsb = (bits >> 16) & 1U;
  const std::uint32_t rounding_bias = 0x7FFFU + lsb;
  return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}
#endif

static inline bfloat16_t random_bfloat16_t(bfloat16_t scale, bfloat16_t bias) {
  const float scale_value = bfloat16_to_float(scale);
  const float bias_value = bfloat16_to_float(bias);
  return bfloat16_from_float((scale_value * (float)rand() / (float)(RAND_MAX)) +
                             bias_value);
}

bool nearly_equal(float a, float b, float epsilon = 128 * FLT_EPSILON,
                  float abs_th = FLT_MIN);

static inline bool nearly_equal_bfloat16(bfloat16_t a, bfloat16_t b,
                                         float epsilon = 128 * FLT_EPSILON,
                                         float abs_th = FLT_MIN) {
  return nearly_equal(bfloat16_to_float(a), bfloat16_to_float(b), epsilon,
                      abs_th);
}

static inline bfloat16_t bfloat16_add(bfloat16_t lhs, bfloat16_t rhs) {
  return bfloat16_from_float(bfloat16_to_float(lhs) + bfloat16_to_float(rhs));
}

static inline bfloat16_t bfloat16_mul(bfloat16_t lhs, bfloat16_t rhs) {
  return bfloat16_from_float(bfloat16_to_float(lhs) * bfloat16_to_float(rhs));
}

static inline bfloat16_t bfloat16_div(bfloat16_t lhs, bfloat16_t rhs) {
  return bfloat16_from_float(bfloat16_to_float(lhs) / bfloat16_to_float(rhs));
}

static inline bfloat16_t bfloat16_tanh(bfloat16_t value) {
  return bfloat16_from_float(std::tanh(bfloat16_to_float(value)));
}

template <typename T>
void print_matrix(const std::vector<T> matrix, int n_cols,
                  int n_printable_rows = 10, int n_printable_cols = 10,
                  std::ostream &ostream = std::cout,
                  const char col_sep[] = "  ", const char elide_sym[] = " ... ",
                  int w = -1) {
  assert(matrix.size() % n_cols == 0);

  auto maxima = std::minmax_element(matrix.begin(), matrix.end());
  T max_val = std::max(*maxima.first, std::abs(*maxima.second));
  size_t n_digits = log10(max_val);
  if (w == -1) {
    w = n_digits;
  }
  int n_rows = matrix.size() / n_cols;

  n_printable_rows = std::min(n_rows, n_printable_rows);
  n_printable_cols = std::min(n_cols, n_printable_cols);

  const bool elide_rows = n_printable_rows < n_rows;
  const bool elide_cols = n_printable_cols < n_cols;

  if (elide_rows || elide_cols) {
    w = std::max((int)w, (int)strlen(elide_sym));
  }

  w += 3; // for decimal point and two decimal digits
  ostream << std::fixed << std::setprecision(2);

#define print_row(what)                                                        \
  for (int col = 0; col < n_printable_cols / 2; col++) {                       \
    ostream << std::right << std::setw(w) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }                                                                            \
  if (elide_cols) {                                                            \
    ostream << std::setw(0) << elide_sym;                                      \
  }                                                                            \
  for (int col = n_printable_cols / 2 + 1; col < n_printable_cols; col++) {    \
    ostream << std::right << std::setw(w) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }

  for (int row = 0; row < n_printable_rows / 2; row++) {
    print_row(matrix[row * n_rows + col]);
    ostream << std::endl;
  }
  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int row = n_printable_rows / 2 + 1; row < n_printable_rows; row++) {
    print_row(matrix[row * n_rows + col]);
    ostream << std::endl;
  }

#undef print_row
}

void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path);

void print_test_banner(const char *, const char *);

void wait_on_lock(uint64_t, uint32_t);

// Print utilities
void print_dma_status_register_fields(const char *, uint32_t);
void print_shim_tile_status(uint64_t);
void print_core_tile_status(uint64_t, uint64_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t);

void wait_for_cycles(uint64_t, uint64_t, uint32_t);

// Write transaction binary in simulation
void write_transaction_binary(unsigned char *data, uint64_t nbytes);

} // namespace test_utils

/* Tensor and Matrix processing utilities
These include CPU reference implementations for tensor operators
and tensor equality tests
*/

namespace matrix_utils {

void pack_int8_to_uint32(uint32_t *sim_mem, const int8_t *input,
                         size_t num_bytes);

template <typename T>
bool is_tensor_equal(T *AIEOutput, T *cpuOutput, unsigned int numElems) {

  for (unsigned int i = 0; i < numElems; i++) {
    if (AIEOutput[i] != cpuOutput[i]) {
      std::cout << "NOTE: mismatch at index " << i << " with values "
                << AIEOutput[i] << " : " << cpuOutput[i] << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T>
void transpose_cpu(T *rowMajorInput, T *transposeResult, unsigned in_rows,
                   unsigned in_cols) {

  // Cols and rows are flipped for the transpose result matrix
  for (unsigned int i = 0; i < in_cols; i++) {
    for (unsigned int j = 0; j < in_rows; j++) {
      unsigned int outIndex = i * in_rows + j; // in_rows is cols for output
      unsigned int inIndex = j * in_cols + i;
      transposeResult[outIndex] = rowMajorInput[inIndex];
    }
  }
}

template <typename T>
void print_matrix(T *matrix, unsigned int rows, unsigned int cols) {

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      unsigned int offset = i * cols + j;
      std::cout << int(matrix[offset]) << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void matmulCPU(T *inputMatrix, T *weightsMatrix, int32_t *outputArray,
               unsigned int M, unsigned int N, unsigned int K,
               unsigned int K_outer) {

  for (unsigned int k_outer = 0; k_outer < K_outer; k_outer++) {
    unsigned int A_K_stride = k_outer * M * K;
    unsigned int B_K_stride = k_outer * N * K;
    for (unsigned int i = 0; i < M; i++) {
      for (unsigned int j = 0; j < N; j++) {
        int sum = 0;
        for (unsigned int k = 0; k < K; k++) {
          unsigned int A_index = A_K_stride + (i * K) + k;
          unsigned int B_index = B_K_stride + (j * K) + k;
          int A_val = int(inputMatrix[A_index]);
          int B_val = int(weightsMatrix[B_index]);
          sum += A_val * B_val;
        }

        unsigned int C_index = i * N + j;
        outputArray[C_index] = outputArray[C_index] + sum;
      }
    }
  }
}

template <typename T> void init_matrix_to_zeros(T *matrix, size_t num_elems) {

  for (size_t i = 0; i < num_elems; i++) {
    matrix[i] = static_cast<T>(0);
  }
}

template <typename T> void init_matrix_to_ones(T *matrix, size_t num_elems) {

  for (size_t i = 0; i < num_elems; i++) {
    matrix[i] = static_cast<T>(2);
  }
}

template <typename T>
void init_matrix_with_val(T *matrix, size_t num_elems, T val) {

  for (size_t i = 0; i < num_elems; i++) {
    matrix[i] = static_cast<T>(val);
  }
}

template <typename T>
void init_matrix_random(T *matrix, size_t num_elems, T min_val, T max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());

  if constexpr (std::is_integral<T>::value) {
    std::uniform_int_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < num_elems; ++i) {
      matrix[i] = dist(gen);
    }
  } else if constexpr (std::is_floating_point<T>::value) {
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < num_elems; ++i) {
      matrix[i] = dist(gen);
    }
  }
}

enum class InitType { Ones, Random };

template <typename T>
void init_matrix(T *matrix, size_t num_elems, T min, T max, InitType type) {
  switch (type) {
  case InitType::Ones:
    matrix_utils::init_matrix_to_ones(matrix, num_elems);
    break;
  case InitType::Random:
    matrix_utils::init_matrix_random(matrix, num_elems, min, max);
    break;
  }
}

} // namespace matrix_utils

#endif // _TEST_UTILS_H_
