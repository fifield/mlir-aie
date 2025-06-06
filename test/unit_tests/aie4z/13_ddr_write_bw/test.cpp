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

  // Parse command line arguments for columns and channels
  int opt;
  uint32_t num_cols = 1;
  uint32_t num_channels = 1;

#if defined(__MESIM__)
  while ((opt = getopt(argc, argv, "c:n:")) != -1) {
    switch (opt) {
    case 'c':
      num_cols = std::stoi(optarg);
      break;
    case 'n':
      num_channels = std::stoi(optarg);
      break;
    default:
      std::cerr << "Usage: " << argv[0] << " [-c num_cols] [-n num_channels]\n";
      return -1;
    }
  }
  if (num_cols > 3) {
    std::cout << "ERROR: num_cols must be <= 3\n";
    return -1;
  }
  if (num_channels > 8) {
    std::cout << "ERROR: num_channels must be <= 8\n";
    return -1;
  }
  std::cout << "num_cols: " << num_cols << ", num_channels: " << num_channels
            << "\n";
#endif

  test_utils::print_test_banner("13_ddr_write_bw",
                                "I wrote The Stand in 3 seconds");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_out;

  uint64_t row;

  ////////////////////////////////////////////////////
  // SETUP EXTERNAL MEMORY

  // Write an input pattern to DDR, it'll get overwritten with zeroes
  uint32_t *ddr_ptr_out = mlir_aie_mem_alloc(&xaie, handle_out, 65536);
  for (int i = 0; i < 65536; i++)
    ddr_ptr_out[i] = i;
  mlir_aie_sync_mem_dev(handle_out);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 512, 4, 640, 4);
  }

  ////////////////////////////////////////////////////
  // CONFIGURATION

  for (uint32_t col = 0; col < num_cols; col++) {
    for (row = 0; row < num_channels; row++) {
      uint64_t tile_data_mem = ((col << 25) | ((row + 2 + 4) << 20));
      for (int i = 0; i < 2048; i++) {
        mlir_aie_sim_write32(tile_data_mem + (i * 4), 0);
        mlir_aie_sim_write32(tile_data_mem + (i * 4) + 32768, 0);
      }
    }
  }
  row = 2;

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  // SET UP PERFORMANCE COUNTERS
  // First, the locks we want to look at
  for (uint32_t col = 0; col < num_cols; col++) {
    uint64_t locks_event_selection_A_0 = (col << 25) | 0x00010100;
    uint64_t locks_event_selection_A_1 = (col << 25) | 0x00010104;
    mlir_aie_sim_write32(locks_event_selection_A_0, 0x00000000);
    mlir_aie_sim_write32(locks_event_selection_A_1, 0x00010000);

    // Second set the perf counter start and stop
    // http://cervino-doc/aie4/r0p41.1/tile_links/xregdb_aie4_shim_tile_doc.html#pl_module___Performance_Start_Stop_0_1

    uint32_t stop_start_stop_start = 0x00000000;
    uint32_t start = 129; // LockSel0_acq_ge, we will use lock 0
    uint32_t stop = 134;  // LockSel0_release, we will use lock 1
    stop_start_stop_start |= (stop << 8) | (start << 0);
    mlir_aie_sim_write32((uint64_t)((col << 25) | 0x00004000),
                         stop_start_stop_start);
  }

  ////////////////////////////////////////////////////
  // TEST START

  // At this point, we need to toggle the shimtile lock to indicate that data
  // has been written and can be read by the DMA This is the
  // shim_tile_input_read_lock which the MM2S channel0 0 is blocked on
  // aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)

  uint32_t before_perf_counter_0 = mlir_aie_sim_read32(0x00004020);
  printf("Before Perf counter 0: %d\n", before_perf_counter_0);

  row = 0;
  for (uint32_t col = 0; col < num_cols; col++) {
    // There's 1 tile of data there now, go for it
    uint64_t shim_tile_lock_base = (col << 25) + 0x00010000;
    for (int i = 0; i < num_channels; i++) {
      // write every other lock, so 0x20 apart.
      mlir_aie_sim_write32(shim_tile_lock_base + i * 0x20, 1);
    }
  }

  if (VERBOSE_TEST) {
    printf("Test just started, possibly not completed yet");
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  ////////////////////////////////////////////////////
  // CHECKING FOR TEST COMPLETION
  for (uint32_t col = 0; col < num_cols; col++) {
    // There's 1 tile of data there now, go for it
    uint64_t shim_tile_lock_base = (col << 25) + 0x00010010;
    for (int i = 0; i < num_channels; i++) {
      // write every other lock, so 0x20 apart.
      printf("Going to wait for col %d lock %d to increment\n", col, 2 * i + 1);
      test_utils::wait_on_lock(shim_tile_lock_base + i * 0x20, 1);
    }
  }

  // When your lock logic goes wrong, this code below can help
  // printf("Going to wait for 10,000 cycles\n");
  // test_utils::wait_for_cycles(0, 2, 10000);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  // EXECUTION COMPLETE

  for (uint32_t col = 0; col < num_cols; col++) {
    uint32_t perf_counter_0 = mlir_aie_sim_read32((col << 25) | 0x00004020);
    printf("Column %d Perf counter 0: %d\n", col, perf_counter_0);

    uint32_t ideal = 65536 * 32 / 512;
    float efficiency = 100.0 * (float)ideal / (float)perf_counter_0;
    // This tells us the cycles, what's the bandwidth?

    uint32_t data_volume = 65536 * 4; // bytes
    printf("Transfered %d bytes\n", data_volume);
    float bytes_per_cycle = (float)data_volume / (float)perf_counter_0;
    float GB_per_sec = bytes_per_cycle; //  Assume 1 GHz clock

    printf("Achieved bandwidth: %f GB/s, %2.f%% efficient\n", GB_per_sec,
           efficiency);
  }
  uint32_t errs = 0;

  // if (efficiency < 80.0) {
  //   printf("FAIL: Efficiency of %.2f%%is less than 80%%\n", efficiency);
  //   errs++;
  // }

  // Lets make sure the output is zeros ...
  uint32_t *ddr_output = mlir_aie_sync_mem_cpu(handle_out);
  for (int i = 0; i < 65536; i++) {
    if (ddr_output[i] != 0) {
      errs++;
      printf(" OUT: error @ %d : %x\n", i, ddr_output[i]);
    } else if (VERBOSE_TEST)
      printf(" OUT:   yes @ %d : %x\n", i, ddr_output[i]);
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", 65536 - errs, 65536);
    return -1;
  }
}