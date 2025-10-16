//===- bottleneck_bf16.cc ----------------------------------------*- C++ -*-===//
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
// Bottleneck - bfloat16
// Bottleneck block with optional residual connection
//
// Architecture:
//   Input → RepConv → Conv+BN+SiLU ─┐
//                                    ├→ Add → Output (if residual)
//   Input ──────────────────────────┘
//
// RepConv = Conv3x3+BN + Conv1x1+BN → Add → SiLU
// 
// Full pipeline:
//   1. RepConv stage:
//      - Conv3x3+BN → temp1
//      - Conv1x1+BN → temp2
//      - Add+SiLU → temp3
//   2. Conv+BN+SiLU stage:
//      - Conv3x3+BN+SiLU on temp3 → temp4
//   3. Residual (if enabled):
//      - Add input + temp4 → output
//
// Weights layout: [repconv_weights, conv2_weights]
// where repconv_weights = [conv3x3_wts, bn3x3, conv1x1_wts, bn1x1]
// and conv2_weights = [conv_wts, bn]
//*****************************************************************************
void bottleneck_bf16_scalar(bfloat16 *input,
                            bfloat16 *weights_and_bn,
                            bfloat16 *output,
                            bfloat16 *input_copy,  // Copy of input for residual
                            bfloat16 *temp1,       // Conv3x3+BN output
                            bfloat16 *temp2,       // Conv1x1+BN output
                            bfloat16 *temp3,       // RepConv output
                            bfloat16 *temp4,       // Final conv output
                            const int32_t height,
                            const int32_t width,
                            const int32_t in_channels,
                            const int32_t out_channels,
                            const int32_t stride,
                            const int32_t padding,
                            const int32_t residual) {
  event0();

  const float bn_eps = 1e-3f;
  const int output_height = (height + 2 * padding - 3) / stride + 1;
  const int output_width = (width + 2 * padding - 3) / stride + 1;
  const int input_size = height * width * in_channels;
  const int output_size = output_height * output_width * out_channels;
  
  // Copy input for residual connection (if needed)
  if (residual) {
    for (int i = 0; i < input_size; i++) {
      input_copy[i] = input[i];
    }
  }
  
  // Extract weight pointers
  // RepConv weights
  const int conv3x3_weight_size = out_channels * in_channels * 3 * 3;
  const int conv1x1_weight_size = out_channels * in_channels * 1 * 1;
  const int bn_param_size = 4 * out_channels;
  
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
  
  // Conv2 weights (final conv)
  bfloat16 *conv2_weights = bn1x1_var + out_channels;
  const int conv2_weight_size = out_channels * out_channels * 3 * 3;
  bfloat16 *bn2_weight = conv2_weights + conv2_weight_size;
  bfloat16 *bn2_bias = bn2_weight + out_channels;
  bfloat16 *bn2_mean = bn2_bias + out_channels;
  bfloat16 *bn2_var = bn2_mean + out_channels;

  //===========================================================================
  // Stage 1: RepConv (Conv3x3+BN + Conv1x1+BN → Add → SiLU)
  //===========================================================================
  
  // Stage 1a: Conv3x3 + BN
  for (int oc = 0; oc < out_channels; oc++) {
    float gamma = (float)bn3x3_weight[oc];
    float beta = (float)bn3x3_bias[oc];
    float mean = (float)bn3x3_mean[oc];
    float var = (float)bn3x3_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        float sum = 0.0f;
        
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
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (oh * output_width + ow) * out_channels + oc;
        temp1[temp_idx] = (bfloat16)bn_out;
      }
    }
  }

  // Stage 1b: Conv1x1 + BN
  for (int oc = 0; oc < out_channels; oc++) {
    float gamma = (float)bn1x1_weight[oc];
    float beta = (float)bn1x1_bias[oc];
    float mean = (float)bn1x1_mean[oc];
    float var = (float)bn1x1_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
          int ih = oh * stride;
          int iw = ow * stride;
          
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int input_idx = (ih * width + iw) * in_channels + ic;
            int weight_idx = oc * in_channels + ic;
            sum += (float)input[input_idx] * (float)conv1x1_weights[weight_idx];
          }
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (oh * output_width + ow) * out_channels + oc;
        temp2[temp_idx] = (bfloat16)bn_out;
      }
    }
  }

  // Stage 1c: Add + SiLU (complete RepConv)
  for (int i = 0; i < output_size; i++) {
    float sum = (float)temp1[i] + (float)temp2[i];
    temp3[i] = (bfloat16)silu(sum);
  }

  //===========================================================================
  // Stage 2: Conv+BN+SiLU on RepConv output
  //===========================================================================
  
  for (int oc = 0; oc < out_channels; oc++) {
    float gamma = (float)bn2_weight[oc];
    float beta = (float)bn2_bias[oc];
    float mean = (float)bn2_mean[oc];
    float var = (float)bn2_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        float sum = 0.0f;
        
        // 3×3 convolution on temp3 (RepConv output)
        for (int ic = 0; ic < out_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = oh * stride + kh - padding;
              int iw = ow * stride + kw - padding;
              
              if (ih >= 0 && ih < output_height && iw >= 0 && iw < output_width) {
                int temp_idx = (ih * output_width + iw) * out_channels + ic;
                int weight_idx = ((oc * out_channels + ic) * 3 + kh) * 3 + kw;
                sum += (float)temp3[temp_idx] * (float)conv2_weights[weight_idx];
              }
            }
          }
        }
        
        // Apply BatchNorm + SiLU
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        float activated = silu(bn_out);
        
        int temp_idx = (oh * output_width + ow) * out_channels + oc;
        temp4[temp_idx] = (bfloat16)activated;
      }
    }
  }

  //===========================================================================
  // Stage 3: Residual connection (if enabled and dimensions match)
  //===========================================================================
  
  if (residual && in_channels == out_channels && 
      height == output_height && width == output_width) {
    // Add residual: output = input + temp4
    for (int i = 0; i < output_size; i++) {
      output[i] = (bfloat16)((float)input_copy[i] + (float)temp4[i]);
    }
  } else {
    // No residual: output = temp4
    for (int i = 0; i < output_size; i++) {
      output[i] = temp4[i];
    }
  }

  event1();
}

extern "C" {

void bottleneck_bf16(bfloat16 *input,
                     bfloat16 *weights_and_bn,
                     bfloat16 *output,
                     bfloat16 *input_copy,
                     bfloat16 *temp1,
                     bfloat16 *temp2,
                     bfloat16 *temp3,
                     bfloat16 *temp4,
                     int32_t height,
                     int32_t width,
                     int32_t in_channels,
                     int32_t out_channels,
                     int32_t stride,
                     int32_t padding,
                     int32_t residual) {
  bottleneck_bf16_scalar(input, weights_and_bn, output, input_copy,
                         temp1, temp2, temp3, temp4,
                         height, width, in_channels, out_channels,
                         stride, padding, residual);
}

} // extern "C"
