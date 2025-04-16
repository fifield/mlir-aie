//===- app.cpp --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <stdint.h>
#include <unistd.h>

#define XAIE_BASE_ADDR 0x20000000000

extern int main(int argc, char *argv[]);

extern "C" {

// 'run_main' is the entry point to C/C++ from Python called from app.py.
// The main job of run_main is to call the user's main(), which has been
// renamed to 'ps_main' by aiecc.
int run_main(int argc, char *argv[]);

uint32_t mesim_Read32(uint64_t addr);
void mesim_Write32(uint64_t addr, uint32_t Data);

void mesim_ReadGM(uint64_t address, void *data, uint64_t size);
void mesim_WriteGM(uint64_t address, void *data, uint64_t size);

void set_read32_callback(uint32_t (*read32_cb)(uint64_t addr));
void set_write32_callback(void (*write32_cb)(uint64_t addr, uint32_t data));

void set_readGM_callback(void (*readGM_cb)(uint64_t addr, uint64_t size,
                                           char *filename));
void set_writeGM_callback(void (*readGM_cb)(uint64_t addr, uint64_t size,
                                            char *filename));
}

// global callback pointers set by set_read32_callback/set_write32_callback
static uint32_t (*__read32_cb)(uint64_t addr) = nullptr;
static void (*__write32_cb)(uint64_t addr, uint32_t data) = nullptr;
static void (*__readGM_cb)(uint64_t addr, uint64_t size,
                           char *filename) = nullptr;
static void (*__writeGM_cb)(uint64_t addr, uint64_t size,
                            char *filename) = nullptr;

void set_read32_callback(uint32_t (*read32_cb)(uint64_t addr)) {
  __read32_cb = read32_cb;
}

void set_write32_callback(void (*write32_cb)(uint64_t addr, uint32_t data)) {
  __write32_cb = write32_cb;
}

void set_readGM_callback(void (*readGM_cb)(uint64_t addr, uint64_t size,
                                           char *filename)) {
  __readGM_cb = readGM_cb;
}

void set_writeGM_callback(void (*writeGM_cb)(uint64_t addr, uint64_t size,
                                             char *filename)) {
  __writeGM_cb = writeGM_cb;
}

// Write 32bits to aie array
void mesim_Write32(uint64_t addr, uint32_t data) {
  assert(__write32_cb);
  __write32_cb(XAIE_BASE_ADDR + addr, data);
}

// Read 32bits from aie array
uint32_t mesim_Read32(uint64_t addr) {
  assert(__read32_cb);
  return __read32_cb(XAIE_BASE_ADDR + addr);
}

void mesim_WriteGM(uint64_t address, void *data, uint64_t size) {
  char filename[] = "/tmp/mesim_WriteGMXXXXXX";
  int fd = mkstemp(filename);
  if (fd == -1) {
    std::cerr << "ERROR: Could not create temporary file\n";
    assert(0);
  }
  close(fd);
  std::ofstream outputFile(filename);
  unsigned char *mem_ptr = reinterpret_cast<unsigned char *>(data);
  for (uint32_t i = 0; i < size; i++) {
    outputFile << std::setfill('0') << std::setw(2) << std::hex
               << (uint32_t)mem_ptr[i] << " ";
    if ((i + 1) % 16 == 0)
      outputFile << std::endl;
  }
  outputFile.close();
  __writeGM_cb(address, size, filename);
  unlink(filename);
}

void mesim_ReadGM(uint64_t address, void *data, uint64_t size) {
  char filename[] = "/tmp/mesim_ReadGMXXXXXX";
  int fd = mkstemp(filename);
  if (fd == -1) {
    std::cerr << "ERROR: Could not create temporary file\n";
    assert(0);
  }
  close(fd);
  __readGM_cb(address, size, filename);
  std::ifstream inputFilename(filename);
  unsigned char *mem_ptr = reinterpret_cast<unsigned char *>(data);

  std::ifstream inputFile(filename);
  if (!inputFile.good()) {
    std::cout << "ERROR: Could not open '" << filename << "'\n";
    assert(0);
  }

  uint32_t count = 0;
  uint16_t buffer = 0;
  uint32_t current_offset = 0;
  while (count < size && !inputFile.eof()) {
    std::string line;
    std::getline(inputFile, line);
    std::istringstream iss(line);
    if (line.empty() == false) {
      while (iss >> std::hex >> buffer) {
        mem_ptr[current_offset] = buffer & 0xff;
        current_offset = current_offset + 1;
      }
      count += 16;
    }
  }
  assert((count >= size) && "File has less bytes than specified in size field "
                            "of fill_external_memory");
  inputFile.close();
  unlink(filename);
}

// Run user main()
int run_main(int argc, char *argv[]) { return ps_main(argc, argv); }