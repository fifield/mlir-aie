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

extern "C" {

typedef unsigned int uint;
void NpuWrite32(uint64_t Addr, uint Data);
uint NpuRead32(uint64_t Addr);
void NpuWriteTransactionBinary(unsigned char *data, size_t nbytes);

}

static std::vector<uint8_t> load_instr_binary(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  if (!instr_file.is_open()) {
    std::cerr << "Error: Could not open file " << instr_path << "\n";
    return {};
  }
  // read size of file, reserve space in  instr_v, then read the file into
  // instr_v
  instr_file.seekg(0, instr_file.end);
  int size = instr_file.tellg();
  instr_file.seekg(0, instr_file.beg);
  std::vector<uint8_t> instr_v(size);
  instr_file.read(reinterpret_cast<char *>(instr_v.data()), size);
  return instr_v;
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
  std::vector<uint8_t> data = load_instr_binary(txn_path);
  NpuWriteTransactionBinary(data.data(), data.size());

  uint64_t col = 0;
  uint64_t row = 2;
  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  int timeout = 0;
  while (1) {
    uint32_t l = NpuRead32(tile_lock_1);
    if (l == 1)
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