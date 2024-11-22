//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "test_utils.h"

extern "C" {

typedef unsigned int uint;
void NpuWrite32(uint64_t Addr, uint Data);
uint NpuRead32(uint64_t Addr);
void NpuWriteTransactionBinary(unsigned char *data, size_t nbytes);

}

float fp8_to_fp32(uint8_t fp8) {
  uint8_t sign = (fp8 & 0x80) >> 7;
  uint8_t exponent = (fp8 & 0x7F) >> 3;
  uint8_t mantissa = (fp8 & 0x07);

  if (exponent == 0) {
    if (mantissa == 0) {
      return 0.0;
    } else {
      return (sign == 0) ? 0.5 * (1 << mantissa) : -0.5 * (1 << mantissa);
    }
  } 
  else {
    return (sign == 0) ? (1 << (exponent - 7)) * (1 + (mantissa / 8.0)) : -1 * (1 << (exponent - 7)) * (1 + (mantissa / 8.0));
  } 
}

bool is_roughly_equal(float a, uint8_t fp8, float tolerance) {
  float fp8_as_fp32 = fp8_to_fp32(fp8);

  const float epsilon = 0.0001; // avoid div by zero

  float abs_diff = fabsf(a - fp8_as_fp32);
  float max_magnitude = fmaxf(fabsf(a), fabsf(fp8_as_fp32));
  float relative_diff = abs_diff / fmaxf(epsilon, max_magnitude);

  return relative_diff < tolerance;

}

int main(int argc, char *argv[]) {
  std::cout << "test start.\n";

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  NpuWriteTransactionBinary(reinterpret_cast<uint8_t *>(data.data()),
                            data.size() * sizeof(uint32_t));

  uint64_t col = 1;
  uint64_t row = 3;

  uint64_t tile_data_mem = ((col << 25) | ((row + 4) << 20)) + 1024 + 1024;
  uint64_t output_tile_data_mem = ((col << 25) | ((row + 4) << 20)) + 32768 + 1024;

  // We will start from a known seed as this is regression test
  srand(42);

  for (int i = 0; i < 256; i++) {
    // Lets make a random float between -100 and 100
    int random = rand() % 200;
    float f = (float )random - 100.0;
    int as_int = *(int *)&f;
    NpuWrite32(tile_data_mem + (i * 4), as_int);
  }

  uint64_t tile_lock_3 = ((col << 25) | ((row + 4) << 20)) + 0x000AE030;
  uint64_t tile_lock_5 = ((col << 25) | ((row + 4) << 20)) + 0x000AE050;

  std::cout << "Read buffer lock\n";
  uint32_t lock3 = NpuRead32(tile_lock_3);
  if (lock3 != 0) {
    std::cout << "Unexpected initial lock value: " << lock3 << "\n";
    return -1;
  }

  std::cout << "Assign buffer lock\n";
  NpuWrite32(tile_lock_3, 1);

  std::cout << "Wait for completion\n";
  while (1) {
    uint32_t lock5 = NpuRead32(tile_lock_5);
    if (lock5 >= 1)
      break;
  }
  std::cout << "Done running\n";

  bool pass = true;

  for (int i = 0; i < 256; i++) {
    int32_t input = NpuRead32(tile_data_mem + (i * 4));

    float f = *(float *)&input;
    int32_t output = NpuRead32(output_tile_data_mem + ((i/4) * 4));

    // The fp8 numbers are in S.E4.M3 format, S is the sign bit, E is the exponent and M is the mantissa
    // So the bit pattern is S E E E E M M M

    uint8_t expected = (uint8_t)(output >> (i % 4) * 8);
    uint8_t sign = (expected & 0x80) >> 7;
    uint8_t exponent = (expected & 0x7F) >> 3;
    uint8_t mantissa = (expected & 0x07);

    bool roughly_equal = is_roughly_equal(f, expected, 0.1);

    if (!roughly_equal) {
      pass = false;
    } 

    /*
    // Should gate this by a verbose flag
    std::cout << "Sample " << i;
    std::cout << " roughly_equal " << roughly_equal;
    std::cout << " input " << std::hex << input << std::dec; 
    std::cout << " output " << std::hex << output << std::dec ;

    // print the fp8 number as S.E.M with numbers
    std::cout << " Sign " << static_cast<int>(sign);
    std::cout << " Exponent " << static_cast<int>(exponent);
    std::cout << " Mantissa " << static_cast<int>(mantissa);
    std::cout << " as floats " << f << " -> " << fp8_to_fp32(expected);
    std::cout << "\n";
    */
  }  

  if (pass) {
    std::cout << "PASS!\n";
    return 0;
  } else {
    std::cout << "FAIL!\n";
    return -1;
  }
}
