// test.cpp

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

extern "C" {

typedef unsigned int uint;
void NpuWrite32(uint64_t Addr, uint32_t Data);
uint32_t NpuRead32(uint64_t Addr);
void NpuWriteTransactionBinary(unsigned char *data, size_t nbytes);
}

#define VERBOSE_TEST 0

int main(int argc, char *argv[]) {

  test_utils::print_test_banner("11_ddr_loopback","When you see yourself, are you far away?");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_in;
  ext_mem_model_t handle_out;

  uint64_t col = 0;
  uint64_t row;

  ////////////////////////////////////////////////////
  // SETUP EXTERNAL MEMORY

  // Write an input pattern to DDR
  int *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle_in, 256);
  for (int i = 0; i < 256; i++)
    ddr_ptr_in[i] = 0xdeadbeef + i;
  mlir_aie_sync_mem_dev(handle_in);

  // Write an output pattern to DDR which will get overwritten
  int *ddr_ptr_out = mlir_aie_mem_alloc(&xaie, handle_out, 256);
  for (int i = 0; i < 256; i++)
    ddr_ptr_out[i] = 0xbadbad00 + i;
  mlir_aie_sync_mem_dev(handle_out);


  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0, NpuRead32);
    test_utils::print_core_tile_status(0,2, 4, 512, 4, 640, 4, NpuRead32);
  }

  ////////////////////////////////////////////////////
  // CONFIGURATION

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  NpuWriteTransactionBinary(reinterpret_cast<uint8_t *>(data.data()),
                            data.size() * sizeof(uint32_t));


  ////////////////////////////////////////////////////
  // TEST START

  // At this point, we need to toggle the shimtile lock to indicate that data has been written
  // and can be read by the DMA
  // This is the shim_tile_input_read_lock which the MM2S channel0 0 is blocked on
  // aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)

  row = 0;
  uint64_t shim_tile_lock_1 = ((col << 25) | (row  << 20)) + 0x00010010;
  NpuWrite32(shim_tile_lock_1, 1);  // There's 1 tile of data there now, go for it


  if (VERBOSE_TEST) {
    printf("Test just started, possibly not completed yet");
    test_utils::print_shim_tile_status(0, NpuRead32);
    test_utils::print_core_tile_status(0,2, 4, 2048, 8, 2560, 8, NpuRead32);
  }

  ////////////////////////////////////////////////////
  // CHECKING FOR TEST COMPLETION

  uint64_t shim_tile_lock_3 = (col << 25) | 0x00010030;
  test_utils::wait_on_lock(shim_tile_lock_3, 1, NpuRead32);

  // When your lock logic goes wrong, this code below can help
  //printf("Going to wait for 10,000 cycles\n");
  //test_utils::wait_for_cycles(0, 2, 10000, NpuRead32);

  if (VERBOSE_TEST) {
    test_utils::print_shim_tile_status(0, NpuRead32);
    test_utils::print_core_tile_status(0,2, 4, 2048, 8, 2560, 8, NpuRead32);
  }

  // EXECUTION COMPLETE

  uint32_t errs = 0;

  // Lets test the input 
  mlir_aie_sync_mem_cpu(handle_in);
  uint32_t *ddr_input = reinterpret_cast<uint32_t *>(handle_in.virtualAddr);
  for (int i = 0; i < 256; i++) {
    if (ddr_input[i] != 0xdeadbeef + i) {
      errs++;
      printf(" IN: error @ %d : %x\n", i, ddr_input[i]);
    }
    else
      if (VERBOSE_TEST)
        printf(" IN:   yes @ %d : %x\n", i, ddr_input[i]);
  }


  // Lets test the output 
  mlir_aie_sync_mem_cpu(handle_out);
  uint32_t *ddr_output = reinterpret_cast<uint32_t *>(handle_out.virtualAddr);
  for (int i = 0; i < 256; i++) {
    if (ddr_output[i] != 0xdeadbeef + i + 1) {
      errs++;
      printf("OUT: error @ %d : %x\n", i, ddr_output[i]);
    }
    else
      if (VERBOSE_TEST)
        printf("OUT:   yes @ %d : %x\n", i, ddr_output[i]);
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", 512 - errs, 512);
    return -1;
  }
}