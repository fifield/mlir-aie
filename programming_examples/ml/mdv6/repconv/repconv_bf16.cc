//===- repconv_bf16.cc -------------------------------------------*- C++ -*-===//
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
// Fast square root approximation using Newton-Raphson
//*****************************************************************************
inline float fast_sqrt(float x) {
  if (x <= 0.0f) return 0.0f;
  
  // Initial guess using bit manipulation
  union { float f; uint32_t i; } conv;
  conv.f = x;
  conv.i = 0x1fbd1df5 + (conv.i >> 1);
  float y = conv.f;
  
  // One iteration of Newton-Raphson: y = 0.5 * (y + x/y)
  y = 0.5f * (y + x / y);
  
  return y;
}

//*****************************************************************************
// Fast sigmoid approximation for SiLU activation
//*****************************************************************************
inline float fast_sigmoid(float x) {
  // Fast approximation: sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|))
  return 0.5f + x / (2.0f * (1.0f + (x > 0 ? x : -x)));
}

inline float silu(float x) {
  // SiLU(x) = x * sigmoid(x)
  return x * fast_sigmoid(x);
}

//*****************************************************************************
// RepConv - bfloat16
// Reparameterizable convolution combining 3×3 and 1×1 convolutions
//
// Architecture:
//   Input → Conv3x3 + BN (no activation) ─┐
//                                          ├→ Add → SiLU → Output
//   Input → Conv1x1 + BN (no activation) ─┘
//
// This implementation uses sequential execution:
// 1. Compute Conv3x3 + BN → temp1
// 2. Compute Conv1x1 + BN → temp2
// 3. Add temp1 + temp2 → temp3
// 4. Apply SiLU → output
//
// Weights layout: [conv3x3_weights, bn3x3_params, conv1x1_weights, bn1x1_params]
// BN params: [weight, bias, mean, var] for each conv
//*****************************************************************************
void repconv_bf16_scalar(bfloat16 *input,
                         bfloat16 *weights_and_bn,
                         bfloat16 *output,
                         bfloat16 *temp1,  // Temp buffer for conv3x3+bn output
                         bfloat16 *temp2,  // Temp buffer for conv1x1+bn output
                         const int32_t height,
                         const int32_t width,
                         const int32_t in_channels,
                         const int32_t out_channels,
                         const int32_t stride,
                         const int32_t padding) {
  event0();

  const float bn_eps = 1e-3f;
  const int output_height = (height + 2 * padding - 3) / stride + 1;
  const int output_width = (width + 2 * padding - 3) / stride + 1;
  
  // Extract weight pointers
  const int conv3x3_weight_size = out_channels * in_channels * 3 * 3;
  const int bn_param_size = 4 * out_channels;
  const int conv1x1_weight_size = out_channels * in_channels * 1 * 1;
  
  bfloat16 *conv3x3_weights = weights_and_bn;
  bfloat16 *bn3x3_weight = conv3x3_weights + conv3x3_weight_size;
  bfloat16 *bn3x3_bias = bn3x3_weight + out_channels;
  bfloat16 *bn3x3_mean = bn3x3_bias + out_channels;
  bfloat16 *bn3x3_var = bn3x3_mean + out_channels;
  
  bfloat16 *conv1x1_weights = bn3x3_var + out_channels;
  bfloat16 *bn1x1_weight = conv1x1_weights + conv1x1_weight_size;
  bfloat16 *bn1x1_bias = bn1x1_weight + out_channels;
  bfloat16 *bn1x1_mean = bn1x1_bias + out_channels;
  bfloat16 *bn1x1_var = bn1x1_mean + out_channels;

  // Stage 1: Conv3x3 + BN
  for (int oc = 0; oc < out_channels; oc++) {
    // Extract BN parameters for this channel
    float gamma = (float)bn3x3_weight[oc];
    float beta = (float)bn3x3_bias[oc];
    float mean = (float)bn3x3_mean[oc];
    float var = (float)bn3x3_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        float sum = 0.0f;
        
        // 3×3 convolution
        for (int ic = 0; ic < in_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = oh * stride + kh - padding;
              int iw = ow * stride + kw - padding;
              
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = (ih * width + iw) * in_channels + ic;
                int weight_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;
                sum += (float)input[input_idx] * (float)conv3x3_weights[weight_idx];
              }
            }
          }
        }
        
        // Apply BatchNorm (no activation)
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        
        int temp_idx = (oh * output_width + ow) * out_channels + oc;
        temp1[temp_idx] = (bfloat16)bn_out;
      }
    }
  }

  // Stage 2: Conv1x1 + BN
  for (int oc = 0; oc < out_channels; oc++) {
    // Extract BN parameters for this channel
    float gamma = (float)bn1x1_weight[oc];
    float beta = (float)bn1x1_bias[oc];
    float mean = (float)bn1x1_mean[oc];
    float var = (float)bn1x1_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        float sum = 0.0f;
        
        // 1×1 convolution
        for (int ic = 0; ic < in_channels; ic++) {
          int ih = oh * stride;
          int iw = ow * stride;
          
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int input_idx = (ih * width + iw) * in_channels + ic;
            int weight_idx = oc * in_channels + ic;
            sum += (float)input[input_idx] * (float)conv1x1_weights[weight_idx];
          }
        }
        
        // Apply BatchNorm (no activation)
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        
        int temp_idx = (oh * output_width + ow) * out_channels + oc;
        temp2[temp_idx] = (bfloat16)bn_out;
      }
    }
  }

  // Stage 3: Add + SiLU
  const int output_size = output_height * output_width * out_channels;
  for (int i = 0; i < output_size; i++) {
    float sum = (float)temp1[i] + (float)temp2[i];
    float activated = silu(sum);
    output[i] = (bfloat16)activated;
  }

  event1();
}

extern "C" {

void repconv_bf16(bfloat16 *input,
                  bfloat16 *weights_and_bn,
                  bfloat16 *output,
                  bfloat16 *temp1,
                  bfloat16 *temp2,
                  int32_t height,
                  int32_t width,
                  int32_t in_channels,
                  int32_t out_channels,
                  int32_t stride,
                  int32_t padding) {
  repconv_bf16_scalar(input, weights_and_bn, output, temp1, temp2,
                      height, width, in_channels, out_channels, stride, padding);
}

} // extern "C"
