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
#include <fcntl.h>

#include <stdint.h>
#include <unistd.h>

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdint.h>

#define XAIE_BASE_ADDR 0x30000000000
#define CASSANDRA_RAM_BASE 0x800000000

// #define HOST "localhost"
// #define PORT 3335
// #define BUFFER_SIZE 16384

int socket_fd = -1;

extern int main(int argc, char *argv[]);

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

void cold_reset_npu();

void acquire_cassandra_handle(char* server_path, int port);
void release_cassandra_handle();
}

// Write 32bits to aie array
void mesim_Write32(uint64_t addr, uint32_t data) {
  // printf("MESIM Write32 at %lu, %u\n", addr, data);
  uint8_t msg[13];  // 1 + 8 + 4 bytes
  uint8_t ret_code;

  addr += XAIE_BASE_ADDR;
  // Pack message: cmd=1, addr (8 bytes), data (4 bytes)
  msg[0] = 1;
  memcpy(&msg[1], &addr, 8);
  memcpy(&msg[9], &data, 4);

  // Send message
  assert(send(socket_fd, msg, sizeof(msg), 0) >= 0 && "Failed to Write");
  recv(socket_fd, &ret_code, 1, 0);
}

// Read 32bits from aie array
uint32_t mesim_Read32(uint64_t addr) {
  // printf("MESIM Read32 at %lu\n", addr);
  uint32_t val;
  uint8_t msg[9];  // 1 + 8 bytes
  uint8_t ret_code;

  addr += XAIE_BASE_ADDR;
  // Pack message: cmd=0, addr (8 bytes)
  msg[0] = 0;
  memcpy(&msg[1], &addr, 8);

  // Send message
  assert (send(socket_fd, msg, sizeof(msg), 0) >= 0 && "Error Reading Value");
  recv(socket_fd, &ret_code, 1, 0);
  recv(socket_fd, &val, 8, 0);
  return val;
}

// Write 32bits to qdma without offsets
void qdma_Write32(uint64_t addr, uint32_t data) {

  uint8_t msg[13];  // 1 + 8 + 4 bytes
  uint8_t ret_code;

  // Pack message: cmd=1, addr (8 bytes), data (4 bytes)
  msg[0] = 1;
  memcpy(&msg[1], &addr, 8);
  memcpy(&msg[9], &data, 4);

  // Send message
  assert(send(socket_fd, msg, sizeof(msg), 0) >= 0 && "Failed to Write");
  recv(socket_fd, &ret_code, 1, 0);
 }

// Read 32bits from without offsets
uint32_t qdma_Read32(uint64_t addr) {
  uint32_t val;
  uint8_t msg[9];  // 1 + 8 bytes
  uint8_t ret_code;

  // Pack message: cmd=0, addr (8 bytes)
  msg[0] = 0;
  memcpy(&msg[1], &addr, 8);

  // Send message
  assert (send(socket_fd, msg, sizeof(msg), 0) >= 0 && "Error Reading Value");
  recv(socket_fd, &ret_code, 1, 0);
  recv(socket_fd, &val, 8, 0);
  return val;
}

void mesim_WriteGM(uint64_t address, void *data, uint64_t size) {
  address += 0x800000000; // + CASSANDRA_RAM_BASE;
  printf("Writing to %lx\n", address);

  uint8_t header[13];  // 1 + 8 + 4 bytes
  uint8_t reply;

  // Pack header: cmd=3, addr (8 bytes), size (4 bytes)
  header[0] = 3;
  memcpy(&header[1], &address, 8);
  memcpy(&header[9], &size, 4);
  send(socket_fd, header, sizeof(header), 0);
  send(socket_fd, data, size, 0);
  recv(socket_fd, &reply, 1, 0);
}

void mesim_ReadGM(uint64_t address, void *data, uint64_t size) {
  uint8_t msg[13];  // 1 + 8 + 4 bytes
  uint8_t reply;
  uint32_t recieved_bytes = 0;
  int n;

  address += 0x800000000; // + CASSANDRA_RAM_BASE;
  printf("Reading from %lx\n", address);

  // Pack message: cmd=2, addr (8 bytes), size (4 bytes)
  msg[0] = 2;
  memcpy(&msg[1], &address, 8);
  memcpy(&msg[9], &size, 4);

  // Send message
  send(socket_fd, msg, sizeof(msg), 0);
  recv(socket_fd, &reply, 1, 0);

  while (recieved_bytes < size) {
    recieved_bytes += recv(socket_fd, (uint8_t*) data+recieved_bytes, size-recieved_bytes, 0);
  }
}

void cold_reset_npu() {
  uint8_t msg = 255;
  uint8_t reply;
  send(socket_fd, &msg, sizeof(msg), 0);
  recv(socket_fd, &reply, 1, 0);
}

void acquire_cassandra_handle(char* server_path, int port) {
  struct sockaddr_in server_addr;
  struct hostent *server;

  printf("Connecting to Cassandra Server %s:%d\n", server_path, port);
  socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  server = gethostbyname(server_path);

  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
  server_addr.sin_port = htons(port);
  if (connect(socket_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
    printf("Error connecting to server!");
    close(socket_fd);
    assert(0 && "Failed to connect to server, aborting");
  }
}

void release_cassandra_handle() {
  if (socket_fd >= 0) {
    close(socket_fd);
  }
}

// Run user main()
int run_main(int argc, char *argv[]) { return ps_main(argc, argv); }