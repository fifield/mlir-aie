//===- test.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"
#include "test_utils.h"
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

using namespace std;

using matrix_utils::InitType;

template <typename T> struct InitTestCase {
  T min;
  T max;
  InitType type;
};

int main(int argc, char **argv) {

  unsigned int M = 4;
  unsigned int tile_N = 64;
  unsigned int tile_K = 64;
  unsigned int K_outer = 4;

  size_t act_size = K_outer * M * tile_K;
  size_t wghts_size = K_outer * tile_K * tile_N;
  size_t act_size_i32 = act_size / 4;
  size_t wghts_size_i32 = wghts_size / 4;
  size_t out_size_i32 = M * tile_N; // INT32 output - K_outer is reduced

  int8_t *A_in = (int8_t *)malloc(act_size * sizeof(int8_t));
  int8_t *B_in = (int8_t *)malloc(wghts_size * sizeof(int8_t));
  uint32_t *A_sim = (uint32_t *)malloc(act_size_i32 * sizeof(uint32_t));
  uint32_t *B_sim = (uint32_t *)malloc(wghts_size_i32 * sizeof(uint32_t));
  int32_t *C = (int32_t *)malloc(out_size_i32 * sizeof(int32_t));
  int32_t *C_ref = (int32_t *)malloc(out_size_i32 * sizeof(int32_t));

  int8_t rnd_min = -127;
  int8_t rnd_max = 127;

  // Each pair of functions defines a test case.
  // First function of the pair initializes A
  // Second function initializes B
  std::vector<std::pair<InitTestCase<int8_t>, InitTestCase<int8_t>>> testCases =
      {{{-127, 127, InitType::Random}, {-127, 127, InitType::Random}},
       {{0, 0, InitType::Ones}, {0, 0, InitType::Ones}}};

  aie_libxaie_ctx_t xaie;

  uint32_t *A_ptr_in;
  uint32_t *B_ptr_in;
  uint32_t *C_ptr_out;

  ext_mem_model_t A_handle;
  ext_mem_model_t B_handle;
  ext_mem_model_t C_handle;

  A_ptr_in = (uint32_t *)mlir_aie_mem_alloc(&xaie, A_handle, act_size_i32);
  B_ptr_in = (uint32_t *)mlir_aie_mem_alloc(&xaie, B_handle, wghts_size_i32);
  C_ptr_out = (uint32_t *)mlir_aie_mem_alloc(&xaie, C_handle, out_size_i32);

  std::cout << "** Loading transaction binary" << std::endl;
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  unsigned total_tests = testCases.size();
  unsigned passed = 0;

  for (size_t i = 0; i < testCases.size(); i++) {

    const auto &[A_test, B_test] = testCases[i];

    init_matrix(A_in, act_size, A_test.min, A_test.max, A_test.type);
    init_matrix(B_in, wghts_size, B_test.min, B_test.max, B_test.type);
    std::cout << "Done: Init_Matrix \n";

    matrix_utils::pack_int8_to_uint32(A_sim, A_in, act_size);
    matrix_utils::pack_int8_to_uint32(B_sim, B_in, wghts_size);

    uint64_t tile_col = 0;
    uint64_t tile_row = 2;

    uint64_t shim_col = 0;
    uint64_t shim_row = 0;

    for (unsigned j = 0; j < act_size_i32; j++)
      A_ptr_in[j] = A_sim[j];
    mlir_aie_sync_mem_dev(A_handle);

    for (unsigned j = 0; j < wghts_size_i32; j++)
      B_ptr_in[j] = B_sim[j];
    mlir_aie_sync_mem_dev(B_handle);

    std::cout << "Done: mlir-aie-sync. \n";

    uint64_t tile_lock_1 =
        ((tile_col << 25) | ((tile_row + 4) << 20)) + 0x000AE010;
    uint64_t tile_lock_3 =
        ((tile_col << 25) | ((tile_row + 4) << 20)) + 0x000AE030;

    std::cout << "Assign tile read buffer lock \n";
    mlir_aie_sim_write32(tile_lock_1, 1);

    // NOTE: shim tile lock calc different from core tiles
    uint64_t shim_a_read_lock =
        ((shim_col << 25) | (shim_row << 20)) + 0x00010010;
    uint64_t shim_b_read_lock =
        ((shim_col << 25) | (shim_row << 20)) + 0x00010030;

    std::cout << "Assign Shim A-Read Lock \n";
    mlir_aie_sim_write32(shim_a_read_lock, 1);

    std::cout << "Assign Shim B-Read Lock \n";
    mlir_aie_sim_write32(shim_b_read_lock, 1);

    uint64_t shim_c_write_lock =
        ((shim_col << 25) | (shim_row << 20)) + 0x00010060;

    std::cout << "Wait for completion \n";
    while (1) {
      uint32_t lock3 = mlir_aie_sim_read32(shim_c_write_lock);
      if (lock3 >= 1)
        break;
    }
    std::cout << "Done running\n";

    // sync data from device to host memory
    mlir_aie_sync_mem_cpu(C_handle);

    std::cout << "- Clear Shim output lock for next test iteration\n";
    mlir_aie_sim_write32(shim_c_write_lock, 0);

    for (int i = 0; i < out_size_i32; i++) {
      C[i] = C_ptr_out[i];
    }

    // test_utils::print_shim_tile_status(0);

    std::cout << std::endl << " AIE Output C " << std::endl;
    matrix_utils::print_matrix(C, M * tile_N, 1);

    matrix_utils::init_matrix_to_zeros(C_ref, out_size_i32);
    matrix_utils::matmulCPU(A_in, B_in, C_ref, M, tile_N, tile_K, K_outer);

    std::cout << std::endl << " CPU Ref Output C " << std::endl;
    matrix_utils::print_matrix(C_ref, M * tile_N, 1);

    if (matrix_utils::is_tensor_equal(C, C_ref, out_size_i32)) {
      cout << "--- Test Pass --- \n";
      passed += 1;
    } else {
      cout << "Test Fail. CPU Results Do NOT Match... \n";
    }
  }

  cout << "\n-Total Tests = " << total_tests << " Total Passed = " << passed
       << std::endl;

  free(A_in);
  free(B_in);
  free(A_sim);
  free(B_sim);
  free(C);
  free(C_ref);

  if (passed == total_tests) {
    cout << "PASS \n";
    return 0;
  } else {
    cout << "NOTE: FAIL \n";
    return -1;
  }
}
