// test.cpp

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>

#include <stdint.h>
#include <stdio.h>

extern "C" {

typedef unsigned int uint;
void NpuWrite32(uint64_t Addr, uint Data);
uint NpuRead32(uint64_t Addr);

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

static void write_transaction_binary(std::vector<uint8_t> &data) {
  assert(data.size() >= 12);

  uint32_t major = data[0];
  uint32_t minor = data[1];
  uint32_t num_cols = data[4];

  uint32_t num_ops, txn_size;
  std::memcpy(&num_ops, &data[8], 4);
  std::memcpy(&txn_size, &data[12], 4);

  // std::cout << "Major: " << major << "\n";
  // std::cout << "Minor: " << minor << "\n";
  // std::cout << "DevGen: " << data[2] << "\n";
  // std::cout << "NumRows: " << data[3] << "\n";
  // std::cout << "NumCols: " << num_cols << "\n";
  // std::cout << "NumMemTileRows: " << data[5] << "\n";
  // std::cout << "NumOps: " << num_ops << "\n";
  // std::cout << "TxnSize: " << txn_size << " bytes\n";

  uint32_t i = 16;

  uint64_t addr = 0;
  uint32_t value = 0;
  uint32_t size = 0;
  uint32_t mask = 0;
  const uint32_t *data_ptr = nullptr;

  if (major == 1 && minor == 0) {
    while (i < data.size()) {
      uint8_t opc = data[i];
      if (opc == 0x00) {
        // std::cout << "opcode: WRITE (0x00)\n";
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        i += 12;
        NpuWrite32(addr, value);
      } else if (opc == 0x01) {
        // std::cout << "opcode: BLOCKWRITE (0x01)\n";
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&size, &data[i + 8], 4);
        data_ptr = reinterpret_cast<const uint32_t*>(data.data() + i + 12);
        i += size;
        size = size - 12;
        for (int j = 0; j < size / 4; j++)
          NpuWrite32(addr+(4 * j), data_ptr[j]);
      }
      else if (opc == 0x03) {
        // std::cout << "opcode: MASKWRITE (0x03)\n";
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        std::memcpy(&mask, &data[i + 12], 4);
        uint32_t r = NpuRead32(addr);
        uint32_t w = (r & ~mask) | (value & mask);
        NpuWrite32(addr, w);
        i += 16;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  printf("Hello, world.\n");

  std::string txn_path = "transaction.bin";
  std::vector<uint8_t> data = load_instr_binary(txn_path);
  write_transaction_binary(data);

  return 0;
}