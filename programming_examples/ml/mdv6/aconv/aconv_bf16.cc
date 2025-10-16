//===- aconv_bf16.cc ----------------------------------------------*- C++ -*-===//
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
// AvgPool2d - bfloat16
// Average pooling with configurable kernel size and stride
//
// Input: (H, W, C) in bfloat16
// Output: (H_out, W_out, C) in bfloat16
// where H_out = (H + 2*padding - kernel_size) / stride + 1
//*****************************************************************************
void avgpool2d_bf16_scalar(bfloat16 *input,
                           bfloat16 *output,
                           const int32_t input_height,
                           const int32_t input_width,
                           const int32_t channels,
                           const int32_t kernel_size,
                           const int32_t stride,
                           const int32_t padding) {
  event0();

  const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  const float pool_size = (float)(kernel_size * kernel_size);

  // For each output position
  for (int oh = 0; oh < output_height; oh++) {
    for (int ow = 0; ow < output_width; ow++) {
      for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        int count = 0;
        
        // Average over kernel window
        for (int kh = 0; kh < kernel_size; kh++) {
          for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride + kh - padding;
            int iw = ow * stride + kw - padding;
            
            // Check bounds (zero padding)
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
              int input_idx = (ih * input_width + iw) * channels + c;
              sum += (float)input[input_idx];
              count++;
            }
          }
        }
        
        int output_idx = (oh * output_width + ow) * channels + c;
        output[output_idx] = (bfloat16)(sum / (float)count);
      }
    }
  }

  event1();
}

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
  // This avoids exp() which may not be available
  return 0.5f + x / (2.0f * (1.0f + (x > 0 ? x : -x)));
}

inline float silu(float x) {
  // SiLU(x) = x * sigmoid(x)
  return x * fast_sigmoid(x);
}

//*****************************************************************************
// AConv - bfloat16
// Fused AvgPool2d + Conv3x3 + BatchNorm + SiLU
// This is the complete AConv layer from MDV6
//
// Stage 1: AvgPool2d (kernel=2, stride=1, padding=0)
// Stage 2: Conv3x3 (stride=2, padding=1)
// Stage 3: BatchNorm
// Stage 4: SiLU activation
//
// Weights layout: [conv_weights, bn_weight, bn_bias, bn_mean, bn_var]
//*****************************************************************************
void aconv_bf16_scalar(bfloat16 *input,
                       bfloat16 *weights_and_bn,
                       bfloat16 *output,
                       bfloat16 *temp_buffer,  // Intermediate buffer for pooled output
                       const int32_t input_height,
                       const int32_t input_width,
                       const int32_t input_channels,
                       const int32_t output_channels) {
  event0();

  // Extract pointers to different parts of weights_and_bn
  const int weight_size = output_channels * input_channels * 3 * 3;
  bfloat16 *weights = weights_and_bn;
  bfloat16 *bn_weight = weights_and_bn + weight_size;
  bfloat16 *bn_bias = bn_weight + output_channels;
  bfloat16 *bn_mean = bn_bias + output_channels;
  bfloat16 *bn_var = bn_mean + output_channels;
  const float bn_eps = 1e-3f;

  // Stage 1: AvgPool2d (2×2, stride=1, padding=0)
  const int pooled_height = input_height - 1;  // (H + 0 - 2) / 1 + 1 = H - 1
  const int pooled_width = input_width - 1;
  
  for (int oh = 0; oh < pooled_height; oh++) {
    for (int ow = 0; ow < pooled_width; ow++) {
      for (int c = 0; c < input_channels; c++) {
        float sum = 0.0f;
        
        // 2×2 average pooling
        for (int kh = 0; kh < 2; kh++) {
          for (int kw = 0; kw < 2; kw++) {
            int ih = oh + kh;
            int iw = ow + kw;
            int input_idx = (ih * input_width + iw) * input_channels + c;
            sum += (float)input[input_idx];
          }
        }
        
        int temp_idx = (oh * pooled_width + ow) * input_channels + c;
        temp_buffer[temp_idx] = (bfloat16)(sum / 4.0f);
      }
    }
  }

  // Stage 2: Conv3x3 (stride=2, padding=1) + BatchNorm + SiLU
  const int conv_output_height = (pooled_height + 2 - 3) / 2 + 1;
  const int conv_output_width = (pooled_width + 2 - 3) / 2 + 1;

  for (int oc = 0; oc < output_channels; oc++) {
    // Extract BN parameters for this channel
    float gamma = (float)bn_weight[oc];
    float beta = (float)bn_bias[oc];
    float mean = (float)bn_mean[oc];
    float var = (float)bn_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int oh = 0; oh < conv_output_height; oh++) {
      for (int ow = 0; ow < conv_output_width; ow++) {
        float sum = 0.0f;
        
        // 3×3 convolution with stride=2, padding=1
        for (int ic = 0; ic < input_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = oh * 2 + kh - 1;  // stride=2, padding=1
              int iw = ow * 2 + kw - 1;
              
              if (ih >= 0 && ih < pooled_height && iw >= 0 && iw < pooled_width) {
                int temp_idx = (ih * pooled_width + iw) * input_channels + ic;
                int weight_idx = ((oc * input_channels + ic) * 3 + kh) * 3 + kw;
                
                sum += (float)temp_buffer[temp_idx] * (float)weights[weight_idx];
              }
            }
          }
        }
        
        // Apply BatchNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        
        // Apply SiLU activation: y = x * sigmoid(x)
        float activated = silu(bn_out);
        
        int output_idx = (oh * conv_output_width + ow) * output_channels + oc;
        output[output_idx] = (bfloat16)activated;
      }
    }
  }

  event1();
}

extern "C" {

void avgpool2d_bf16(bfloat16 *input,
                    bfloat16 *output,
                    int32_t input_height,
                    int32_t input_width,
                    int32_t channels,
                    int32_t kernel_size,
                    int32_t stride,
                    int32_t padding) {
  avgpool2d_bf16_scalar(input, output, input_height, input_width, channels,
                        kernel_size, stride, padding);
}

void aconv_bf16(bfloat16 *input,
                bfloat16 *weights,
                bfloat16 *output,
                bfloat16 *temp_buffer,
                int32_t input_height,
                int32_t input_width,
                int32_t input_channels,
                int32_t output_channels) {
  aconv_bf16_scalar(input, weights, output, temp_buffer,
                    input_height, input_width, input_channels, output_channels);
}

} // extern "C"
