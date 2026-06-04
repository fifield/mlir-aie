//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "memory_allocator.h"
#include "test_utils.h"

#define VERBOSE_TEST 0

int main(int argc, char *argv[]) {

  test_utils::print_test_banner("21_shim_dma_4d_static",
                                "Four dimensions, one stream.");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_in;
  ext_mem_model_t handle_out;

  uint64_t col = 0;
  uint64_t row;

  ////////////////////////////////////////////////////
  // SETUP EXTERNAL MEMORY

  // Write an input pattern to DDR
  uint32_t *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle_in, 256);
  for (int i = 0; i < 256; i++)
    ddr_ptr_in[i] = 0xdeadbeef + i;
  mlir_aie_sync_mem_dev(handle_in);

  // Write an output pattern to DDR which will get overwritten
  uint32_t *ddr_ptr_out = mlir_aie_mem_alloc(&xaie, handle_out, 256);
  for (int i = 0; i < 256; i++)
    ddr_ptr_out[i] = 0xbadbad00 + i;
  mlir_aie_sync_mem_dev(handle_out);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 512, 4, 640, 4);
  }

  ////////////////////////////////////////////////////
  // CONFIGURATION

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  ////////////////////////////////////////////////////
  // TEST START

  // At this point, we need to toggle the shimtile lock to indicate that data
  // has been written and can be read by the DMA This is the
  // shim_tile_input_read_lock which the MM2S channel0 0 is blocked on
  // aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)

  row = 0;
  uint64_t shim_tile_lock_1 = ((col << 25) | (row << 20)) + 0x00010010;
  // There's 1 tile of data there now, go for it
  mlir_aie_sim_write32(shim_tile_lock_1, 1);

  if (VERBOSE_TEST) {
    printf("Test just started, possibly not completed yet");
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  ////////////////////////////////////////////////////
  // CHECKING FOR TEST COMPLETION

  uint64_t shim_tile_lock_3 = (col << 25) | 0x00010030;
  test_utils::wait_on_lock(shim_tile_lock_3, 1);

  // When your lock logic goes wrong, this code below can help
  // printf("Going to wait for 10,000 cycles\n");
  // test_utils::wait_for_cycles(0, 2, 10000);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  // EXECUTION COMPLETE

  uint32_t errs = 0;

  // Lets test the input
  uint32_t *ddr_input = mlir_aie_sync_mem_cpu(handle_in);
  for (int i = 0; i < 256; i++) {
    if (ddr_input[i] != 0xdeadbeef + i) {
      errs++;
      printf(" IN: error @ %d : %x\n", i, ddr_input[i]);
    } else if (VERBOSE_TEST)
      printf(" IN:   yes @ %d : %x\n", i, ddr_input[i]);
  }

  // Test the output. The shim MM2S read the input with the 4D (D0..D3) scatter
  // defined in aie4.mlir, the core added 1, and the result was streamed out
  // contiguously. So ddr_output[k] == ddr_input[addr(k)] + 1, where k follows
  // the nested D3,D2,D1,D0 loop order and addr(k) = 64*i3 + 4*i2 + 16*i1 + i0.
  uint32_t *ddr_output = mlir_aie_sync_mem_cpu(handle_out);
  int k = 0;
  for (int i3 = 0; i3 < 4; i3++)            // d3, stride 64
    for (int i2 = 0; i2 < 4; i2++)          // d2, stride 4
      for (int i1 = 0; i1 < 4; i1++)        // d1, stride 16
        for (int i0 = 0; i0 < 4; i0++) {    // d0, stride 1
          int addr = 64 * i3 + 4 * i2 + 16 * i1 + i0;
          uint32_t ref = 0xdeadbeef + addr + 1;
          if (ddr_output[k] != ref) {
            errs++;
            printf("OUT: error @ %d : %x (expected %x)\n", k, ddr_output[k],
                   ref);
          } else if (VERBOSE_TEST)
            printf("OUT:   yes @ %d : %x\n", k, ddr_output[k]);
          k++;
        }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", 512 - errs, 512);
    return -1;
  }
}