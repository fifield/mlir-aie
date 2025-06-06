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

  test_utils::print_test_banner("13_ddr_write_bw",
                                "I wrote The Stand in 3 seconds");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_out;

  uint64_t col = 0;
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

  row = 2;
  uint64_t tile_data_mem = ((col << 25) | ((row + 4) << 20));
  for (int i = 0; i < 2048; i++) {
    mlir_aie_sim_write32(tile_data_mem + (i * 4), 0);
    mlir_aie_sim_write32(tile_data_mem + (i * 4) + 32768, 0);
  }

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  // SET UP PERFORMANCE COUNTERS
  // First, the locks we want to look at
  uint64_t locks_event_selection_A_0 = (col << 25) | 0x00010100;
  uint64_t locks_event_selection_A_1 = (col << 25) | 0x00010104;
  mlir_aie_sim_write32(locks_event_selection_A_0, 0x00000000);
  mlir_aie_sim_write32(locks_event_selection_A_1, 0x00010000);

  // Second set the perf counter start and stop
  // http://cervino-doc/aie4/r0p41.1/tile_links/xregdb_aie4_shim_tile_doc.html#pl_module___Performance_Start_Stop_0_1

  uint32_t stop_start_stop_start = 0x00000000;
  uint32_t start = 128; // LockSel0_acq_eq, we will use lock 0
  uint32_t stop = 134;  // LockSel _release, we will use lock 1
  stop_start_stop_start |= (stop << 8) | (start << 0);
  mlir_aie_sim_write32((uint64_t)(0x00004000), stop_start_stop_start);

  ////////////////////////////////////////////////////
  // TEST START

  // At this point, we need to toggle the shimtile lock to indicate that data
  // has been written and can be read by the DMA This is the
  // shim_tile_input_read_lock which the MM2S channel0 0 is blocked on
  // aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)

  uint32_t before_perf_counter_0 = mlir_aie_sim_read32(0x00004020);
  printf("Before Perf counter 0: %d\n", before_perf_counter_0);

  row = 0;
  uint64_t shim_tile_lock_0 = ((col << 25) | (row << 20)) + 0x00010000;
  // There's 1 tile of data there now, go for it
  mlir_aie_sim_write32(shim_tile_lock_0, 1);

  if (VERBOSE_TEST) {
    printf("Test just started, possibly not completed yet");
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  ////////////////////////////////////////////////////
  // CHECKING FOR TEST COMPLETION

  printf("Going to wait for lock 1 to increment\n");

  uint64_t shim_tile_lock_1 = (col << 25) | 0x00010010;
  test_utils::wait_on_lock(shim_tile_lock_1, 1);

  // When your lock logic goes wrong, this code below can help
  // printf("Going to wait for 10,000 cycles\n");
  // test_utils::wait_for_cycles(0, 2, 10000);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0);
    test_utils::print_core_tile_status(0, 2, 4, 2048, 8, 2560, 8);
  }

  // EXECUTION COMPLETE

  uint32_t perf_counter_0 = mlir_aie_sim_read32(0x00004020);
  printf("Perf counter 0: %d\n", perf_counter_0);

  uint32_t ideal = 65536 * 32 / 512;
  float efficiency = 100.0 * (float)ideal / (float)perf_counter_0;
  // This tells us the cycles, what's the bandwidth?

  uint32_t data_volume = 65536 * 4; // bytes
  printf("Transfered %d bytes\n", data_volume);
  float bytes_per_cycle = (float)data_volume / (float)perf_counter_0;
  float GB_per_sec = bytes_per_cycle; //  Assume 1 GHz clock

  printf("Achieved bandwidth: %f GB/s, %2.f%% efficient\n", GB_per_sec,
         efficiency);

  uint32_t errs = 0;

  if (efficiency < 80.0) {
    printf("FAIL: Efficiency of %.2f%%is less than 80%%\n", efficiency);
    errs++;
  }

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