//===- batchnorm_silu_bf16.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

//*****************************************************************************
// BatchNorm + SiLU - bfloat16
// Fused implementation for MDV6
// 
// BatchNorm: y = weight * x + bias (per channel)
// SiLU: y = x * sigmoid(x) ≈ x * (0.5 * tanh(0.5 * x) + 0.5)
//
// Input: (H, W, C) in bfloat16
// BN params: (2*C,) in bfloat16 - first C elements are weight, next C are bias
// Output: (H, W, C) in bfloat16
//*****************************************************************************
void batchnorm_silu_bf16_scalar(bfloat16 *input, 
                                bfloat16 *bn_params,
                                bfloat16 *output,
                                const int32_t height,
                                const int32_t width,
                                const int32_t channels) {
  event0();

  const int spatial_size = height * width;
  bfloat16 *bn_weight = bn_params;
  bfloat16 *bn_bias = bn_params + channels;
  
  // Process each spatial location
  for (int hw = 0; hw < spatial_size; hw++) {
    for (int c = 0; c < channels; c++) {
      int idx = hw * channels + c;
      
      // Load input
      float x = (float)input[idx];
      
      // Apply BatchNorm: y = weight * x + bias
      float bn_w = (float)bn_weight[c];
      float bn_b = (float)bn_bias[c];
      float bn_out = bn_w * x + bn_b;
      
      // Apply SiLU: x * sigmoid(x)
      // Use simple sigmoid approximation: sigmoid(x) ≈ 0.5 + 0.25*x for small x
      // For better accuracy, use: sigmoid(x) ≈ x / (1 + |x|) (fast sigmoid)
      float abs_x = (bn_out >= 0) ? bn_out : -bn_out;
      float sigmoid_approx = bn_out / (1.0f + abs_x);
      sigmoid_approx = 0.5f * (sigmoid_approx + 1.0f);  // Shift to [0, 1]
      float silu_out = bn_out * sigmoid_approx;
      
      // Store output
      output[idx] = (bfloat16)silu_out;
    }
  }

  event1();
}

//*****************************************************************************
// BatchNorm only - bfloat16
// For cases where we don't want SiLU activation
//*****************************************************************************
void batchnorm_bf16_scalar(bfloat16 *input,
                           bfloat16 *bn_weight,
                           bfloat16 *bn_bias,
                           bfloat16 *output,
                           const int32_t height,
                           const int32_t width,
                           const int32_t channels) {
  event0();

  const int spatial_size = height * width;
  
  for (int hw = 0; hw < spatial_size; hw++) {
    for (int c = 0; c < channels; c++) {
      int idx = hw * channels + c;
      
      float x = (float)input[idx];
      float bn_w = (float)bn_weight[c];
      float bn_b = (float)bn_bias[c];
      float bn_out = bn_w * x + bn_b;
      
      output[idx] = (bfloat16)bn_out;
    }
  }

  event1();
}

//*****************************************************************************
// SiLU only - bfloat16
// For standalone SiLU activation
//*****************************************************************************
void silu_bf16_scalar(bfloat16 *input,
                      bfloat16 *output,
                      const int32_t size) {
  event0();

  for (int i = 0; i < size; i++) {
    float x = (float)input[i];
    
    // SiLU: x * sigmoid(x) using fast sigmoid approximation
    float abs_x = (x >= 0) ? x : -x;
    float sigmoid_approx = x / (1.0f + abs_x);
    sigmoid_approx = 0.5f * (sigmoid_approx + 1.0f);
    float silu_out = x * sigmoid_approx;
    
    output[i] = (bfloat16)silu_out;
  }

  event1();
}

extern "C" {

void batchnorm_silu_bf16(bfloat16 *input,
                         bfloat16 *bn_params,
                         bfloat16 *output,
                         int32_t height,
                         int32_t width,
                         int32_t channels) {
  batchnorm_silu_bf16_scalar(input, bn_params, output, height, width, channels);
}

void batchnorm_bf16(bfloat16 *input,
                    bfloat16 *bn_params,
                    bfloat16 *output,
                    int32_t height,
                    int32_t width,
                    int32_t channels) {
  // Split params into weight and bias
  bfloat16 *bn_weight = bn_params;
  bfloat16 *bn_bias = bn_params + channels;
  batchnorm_bf16_scalar(input, bn_weight, bn_bias, output, height, width, channels);
}

void silu_bf16(bfloat16 *input,
               bfloat16 *output,
               int32_t size) {
  silu_bf16_scalar(input, output, size);
}

} // extern "C"
