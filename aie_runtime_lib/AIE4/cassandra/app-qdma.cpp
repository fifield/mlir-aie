//===- app.cpp --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <stdint.h>
#include <unistd.h>

#define XAIE_BASE_ADDR 0x30000000000
#define CASSANDRA_RAM_BASE 0x800000000

extern int main(int argc, char *argv[]);

static int cassandra_fd = -1;

extern "C" {

// 'run_main' is the entry point to C/C++ from Python called from app.py.
// The main job of run_main is to call the user's main(), which has been
// renamed to 'ps_main' by aiecc.
int run_main(int argc, char *argv[]);

uint32_t mesim_Read32(uint64_t addr);
void mesim_Write32(uint64_t addr, uint32_t Data);

uint32_t qdma_Read32(uint64_t addr);
void qdma_Write32(uint64_t addr, uint32_t Data);

void mesim_ReadGM(uint64_t address, void *data, uint64_t size);
void mesim_WriteGM(uint64_t address, void *data, uint64_t size);

void acquire_cassandra_handle(char *cassandra_path);
void release_cassandra_handle();
}

// Write 32bits to aie array
void mesim_Write32(uint64_t addr, uint32_t data) {
  assert(cassandra_fd != -1 && "Invalid Cassandra File Descriptor");
  off_t offset = addr + XAIE_BASE_ADDR;
  // printf("Writing to %lx!\n", addr);
  off_t rc = lseek(cassandra_fd, offset, SEEK_SET);
  assert(rc == offset && "Failed QDMA Seek");
  write(cassandra_fd, &data, 4);
}

// Read 32bits from aie array
uint32_t mesim_Read32(uint64_t addr) {
  uint32_t val;
  assert(cassandra_fd != -1 && "Invalid Cassandra File Descriptor");
  off_t offset = addr + XAIE_BASE_ADDR;
  off_t rc = lseek(cassandra_fd, offset, SEEK_SET);
  assert(rc == offset && "Failed QDMA Seek");
  read(cassandra_fd, &val, 4);
  // printf("Reading %x from %lx!\n", val, addr);
  return val;
}

// Write 32bits to qdma without offsets
void qdma_Write32(uint64_t addr, uint32_t data) {
  assert(cassandra_fd != -1 && "Invalid Cassandra File Descriptor");
  off_t offset = addr;
  off_t rc = lseek(cassandra_fd, offset, SEEK_SET);
  assert(rc == offset && "Failed QDMA Seek");
  write(cassandra_fd, &data, 4);
}

// Read 32bits from without offsets
uint32_t qdma_Read32(uint64_t addr) {
  uint32_t val;
  assert(cassandra_fd != -1 && "Invalid Cassandra File Descriptor");
  off_t offset = addr;
  off_t rc = lseek(cassandra_fd, offset, SEEK_SET);
  assert(rc == offset && "Failed QDMA Seek");
  read(cassandra_fd, &val, 4);
  return val;
}

void mesim_WriteGM(uint64_t address, void *data, uint64_t size) {
  assert(cassandra_fd != -1 && "Invalid Cassandra File Descriptor");
  off_t offset = address + 0x800000000; // + CASSANDRA_RAM_BASE;
  printf("Writing to %lx\n", offset);
  off_t rc = lseek(cassandra_fd, offset, SEEK_SET);
  assert(rc == offset && "Failed QDMA Seek");
  write(cassandra_fd, data, size);
  // size_t dat_offset =
  // while(size) {
  //   uint64_t tx_size = size > 1024 ? 1024 : size;

  // }
}

void mesim_ReadGM(uint64_t address, void *data, uint64_t size) {
  assert(cassandra_fd != -1 && "Invalid Cassandra File Descriptor");
  off_t offset = address + 0x800000000; // + CASSANDRA_RAM_BASE;
  off_t rc = lseek(cassandra_fd, offset, SEEK_SET);
  assert(rc == offset && "Failed QDMA Seek");
  read(cassandra_fd, data, size);
  // while(size) {
  //   uint64_t tx_size = size > 1024 ? 1024 : size;

  // }
}

void acquire_cassandra_handle(char *cassandra_path) {
  printf("Acquiring FPGA %s\n", cassandra_path);
  cassandra_fd = open(cassandra_path, O_RDWR);
}

void release_cassandra_handle() {
  if (cassandra_fd >= 0) {
    close(cassandra_fd);
  }
}

// Run user main()
int run_main(int argc, char *argv[]) { return ps_main(argc, argv); }