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
void NpuWrite32(uint64_t Addr, uint Data);
uint NpuRead32(uint64_t Addr);
void NpuWriteTransactionBinary(unsigned char *data, size_t nbytes);
}

int main(int argc, char *argv[]) {
  printf("Hello, world.\n");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle;
  int *ddr_ptr_in = mlir_aie_mem_alloc(&xaie, handle, 256 * sizeof(int));
  for (int i = 0; i < 256; i++)
    ddr_ptr_in[i] = 0xdeadbeef + i;
  mlir_aie_sync_mem_dev(handle);

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  NpuWriteTransactionBinary(reinterpret_cast<uint8_t *>(data.data()),
                            data.size() * sizeof(uint32_t));

  uint64_t col = 0;
  uint64_t row = 2;
  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  int timeout = 0;
  while (1) {
    uint32_t l = NpuRead32(tile_lock_1);
    if (l >= 1)
      break;
  }

  uint32_t output[256];
  uint64_t tile_data_mem = ((col << 25) | ((row + 4) << 20));
  int errs = 0;
  for (int i = 0; i < 256; i++) {
    output[i] = NpuRead32(tile_data_mem + (4 * i));
    if (output[i] != 0xdeadbeef + i) {
      errs++;
      printf("error @ %d : %x\n", i, output[i]);
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail (%d/%d).\n", 256 - errs, 256);
    return -1;
  }
}