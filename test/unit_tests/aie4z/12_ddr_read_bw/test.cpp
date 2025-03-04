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
  if (num_channels > 4) {
    std::cout << "ERROR: num_channels must be <= 4\n";
    return -1;
  }
  std::cout << "num_cols: " << num_cols << ", num_channels: " << num_channels
            << "\n";
#endif

  test_utils::print_test_banner("12_ddr_read_bw",
                                "I read War and Peace in 2 minutes");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_in;
  ext_mem_model_t handle_out;

  ////////////////////////////////////////////////////
  // SETUP EXTERNAL MEMORY

  // Write an input pattern to DDR
  int *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle_in, 65536);
  for (int i = 0; i < 65536; i++)
    ddr_ptr_in[i] = i;
  mlir_aie_sync_mem_dev(handle_in);

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

  // SET UP PERFORMANCE COUNTERS
  // First, the locks we want to look at
  for (uint32_t col = 0; col < num_cols; col++) {
    uint64_t dma_event_channel_selection_A = (col << 25) | 0x000185C0;
    uint64_t locks_event_selection_A_0 = (col << 25) | 0x00010100;
    uint64_t locks_event_selection_A_1 = (col << 25) | 0x00010104;
    uint64_t locks_event_selection_A_2 = (col << 25) | 0x00010108;
    uint64_t locks_event_selection_A_3 = (col << 25) | 0x0001010C;
    mlir_aie_sim_write32(dma_event_channel_selection_A, 0x03020100);
    mlir_aie_sim_write32(locks_event_selection_A_0, 0x00010000);
    mlir_aie_sim_write32(locks_event_selection_A_1, 0x00030000);
    mlir_aie_sim_write32(locks_event_selection_A_2, 0x00050000);
    mlir_aie_sim_write32(locks_event_selection_A_3, 0x00070000);

    // Second set the perf counter start and stop
    // http://cervino-doc/aie4/r0p41.1/tile_links/xregdb_aie4_shim_tile_doc.html#pl_module___Performance_Start_Stop_0_1

    uint32_t stop_start_stop_start = 0x00000000;
    uint32_t start = 128; // LockSel0_acq_eq, we will use lock 1
    uint32_t stop = 54;   // DMA_MM2S_Sel0_finished_BD
    stop_start_stop_start |= (stop << 8) | (start << 0);
    start = 132; // LockSel1_acq_eq, we will use lock 3
    stop = 55;   // DMA_MM2S_Sel1_finished_BD
    stop_start_stop_start |= (stop << 24) | (start << 16);
    mlir_aie_sim_write32((uint64_t)((col << 25) | 0x00004000),
                         stop_start_stop_start);

    stop_start_stop_start = 0x00000000;
    start = 136; // LockSel2_acq_eq, we will use lock 5
    stop = 56;   // DMA_MM2S_Sel2_finished_BD
    stop_start_stop_start |= (stop << 8) | (start << 0);
    start = 140; // LockSel3_acq_eq, we will use lock 7
    stop = 57;   // DMA_MM2S_Sel3_finished_BD
    stop_start_stop_start |= (stop << 24) | (start << 16);
    mlir_aie_sim_write32((uint64_t)((col << 25) | 0x00004004),
                         stop_start_stop_start);
  }

  ////////////////////////////////////////////////////
  // TEST START

  // At this point, we need to toggle the shimtile lock to indicate that data
  // has been written and can be read by the DMA This is the
  // shim_tile_input_read_lock which the MM2S channel0 0 is blocked on
  // aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)
  for (uint32_t col = 0; col < num_cols; col++) {
    uint32_t before_perf_counter_0 =
        mlir_aie_sim_read32((col << 25) | 0x00004020);
    uint32_t before_perf_counter_1 =
        mlir_aie_sim_read32((col << 25) | 0x00004024);
    uint32_t before_perf_counter_2 =
        mlir_aie_sim_read32((col << 25) | 0x00004028);
    uint32_t before_perf_counter_3 =
        mlir_aie_sim_read32((col << 25) | 0x0000402C);
    printf("Before Perf counter 0 for column %u: %d\n", col,
           before_perf_counter_0);
    if (num_channels > 1)
      printf("Before Perf counter 1 for column %u: %d\n", col,
             before_perf_counter_1);
    if (num_channels > 2)
      printf("Before Perf counter 2 for column %u: %d\n", col,
             before_perf_counter_2);
    if (num_channels > 3)
      printf("Before Perf counter 3 for column %u: %d\n", col,
             before_perf_counter_3);

    // There's 1 tile of data there now, go for it
    uint64_t shim_tile_lock_1 = (col << 25) + 0x00010010;
    uint64_t shim_tile_lock_3 = (col << 25) + 0x00010030;
    uint64_t shim_tile_lock_5 = (col << 25) + 0x00010050;
    uint64_t shim_tile_lock_7 = (col << 25) + 0x00010070;
    mlir_aie_sim_write32(shim_tile_lock_1, 1);
    if (num_channels == 1)
      continue;
    mlir_aie_sim_write32(shim_tile_lock_3, 1);
    if (num_channels == 2)
      continue;
    mlir_aie_sim_write32(shim_tile_lock_5, 1);
    if (num_channels == 3)
      continue;
    mlir_aie_sim_write32(shim_tile_lock_7, 1);
  }

  if (VERBOSE_TEST) {
    printf("Test just started, possibly not completed yet");
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  ////////////////////////////////////////////////////
  // CHECKING FOR TEST COMPLETION

  for (uint32_t col = 0; col < num_cols; col++) {
    uint64_t shim_tile_lock_0 = (col << 25) | 0x00010000;
    uint64_t shim_tile_lock_2 = (col << 25) | 0x00010020;
    uint64_t shim_tile_lock_4 = (col << 25) | 0x00010040;
    uint64_t shim_tile_lock_6 = (col << 25) | 0x00010060;
    printf("Going to wait for col %u lock 0 to increment\n", col);
    test_utils::wait_on_lock(shim_tile_lock_0, 2);
    if (num_channels == 1)
      continue;
    printf("Going to wait for col %u lock 2 to increment\n", col);
    test_utils::wait_on_lock(shim_tile_lock_2, 2);
    if (num_channels == 2)
      continue;
    printf("Going to wait for col %u lock 4 to increment\n", col);
    test_utils::wait_on_lock(shim_tile_lock_4, 2);
    if (num_channels == 3)
      continue;
    printf("Going to wait for col %u lock 6 to increment\n", col);
    test_utils::wait_on_lock(shim_tile_lock_6, 2);
  }

  // When your lock logic goes wrong, this code below can help
  // printf("Going to wait for 10,000 cycles\n");
  // test_utils::wait_for_cycles(0, 2, 10000);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  uint32_t errs = 0;

  // EXECUTION COMPLETE
  for (uint32_t col = 0; col < num_cols; col++) {
    uint32_t perf_counter_0 = mlir_aie_sim_read32((col << 25) | 0x00004020);
    uint32_t perf_counter_1 = mlir_aie_sim_read32((col << 25) | 0x00004024);
    uint32_t perf_counter_2 = mlir_aie_sim_read32((col << 25) | 0x00004028);
    uint32_t perf_counter_3 = mlir_aie_sim_read32((col << 25) | 0x0000402C);
    printf("Perf counter 0 for column %u: %d\n", col, perf_counter_0);
    if (num_channels > 1)
      printf("Perf counter 1 for column %u: %d\n", col, perf_counter_1);
    if (num_channels > 2)
      printf("Perf counter 2 for column %u: %d\n", col, perf_counter_2);
    if (num_channels > 3)
      printf("Perf counter 3 for column %u: %d\n", col, perf_counter_3);

    uint32_t ideal = 65536 * 32 / 512;
    uint32_t data_volume = 65536 * 4; // bytes
    printf("Transferred %d bytes per channel for column %u\n", data_volume,
           col);

    float efficiency = 100.0 * (float)ideal / (float)perf_counter_0;
    float bytes_per_cycle = (float)data_volume / (float)perf_counter_0;
    float GB_per_sec = bytes_per_cycle; // Assume 1 GHz clock
    printf("Achieved bandwidth for column %u channel 0: %f GB/s, %.2f%% "
           "efficient\n",
           col, GB_per_sec, efficiency);
    if (num_channels == 1)
      continue;

    efficiency = 100.0 * (float)ideal / (float)perf_counter_1;
    bytes_per_cycle = (float)data_volume / (float)perf_counter_1;
    GB_per_sec = bytes_per_cycle; // Assume 1 GHz clock
    printf("Achieved bandwidth for column %u channel 1: %f GB/s, %.2f%% "
           "efficient\n",
           col, GB_per_sec, efficiency);
    if (num_channels == 2)
      continue;

    efficiency = 100.0 * (float)ideal / (float)perf_counter_2;
    bytes_per_cycle = (float)data_volume / (float)perf_counter_2;
    GB_per_sec = bytes_per_cycle; // Assume 1 GHz clock
    printf("Achieved bandwidth for column %u channel 2: %f GB/s, %.2f%% "
           "efficient\n",
           col, GB_per_sec, efficiency);
    if (num_channels == 3)
      continue;

    efficiency = 100.0 * (float)ideal / (float)perf_counter_3;
    bytes_per_cycle = (float)data_volume / (float)perf_counter_3;
    GB_per_sec = bytes_per_cycle; // Assume 1 GHz clock
    printf("Achieved bandwidth for column %u channel 3: %f GB/s, %.2f%% "
           "efficient\n",
           col, GB_per_sec, efficiency);

    // if (efficiency < 80.0) {
    //   printf("FAIL: Efficiency of %.2f%% for column %u is less than 80%%\n",
    //   efficiency, col); errs++;
    // }
  }

  // Lets make sure the input is still there ...
  mlir_aie_sync_mem_cpu(handle_in);
  uint32_t *ddr_input = reinterpret_cast<uint32_t *>(handle_in.virtualAddr);
  for (int i = 0; i < 65536; i++) {
    if (ddr_input[i] != i) {
      errs++;
      printf(" IN: error @ %d : %x\n", i, ddr_input[i]);
    } else if (VERBOSE_TEST)
      printf(" IN:   yes @ %d : %x\n", i, ddr_input[i]);
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", 65536 - errs, 65536);
    return -1;
  }
}
