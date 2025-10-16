//===- elan_bf16.cc ----------------------------------------------*- C++ -*-===//
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
  
  union { float f; uint32_t i; } conv;
  conv.f = x;
  conv.i = 0x1fbd1df5 + (conv.i >> 1);
  float y = conv.f;
  
  y = 0.5f * (y + x / y);
  
  return y;
}

//*****************************************************************************
// Fast sigmoid approximation for SiLU activation
//*****************************************************************************
inline float fast_sigmoid(float x) {
  return 0.5f + x / (2.0f * (1.0f + (x > 0 ? x : -x)));
}

inline float silu(float x) {
  return x * fast_sigmoid(x);
}

//*****************************************************************************
// Conv1x1 + BatchNorm + SiLU helper function
//*****************************************************************************
inline void conv1x1_bn_silu(bfloat16 *input,
                            bfloat16 *weights,
                            bfloat16 *bn_weight,
                            bfloat16 *bn_bias,
                            bfloat16 *bn_mean,
                            bfloat16 *bn_var,
                            bfloat16 *output,
                            const int32_t height,
                            const int32_t width,
                            const int32_t in_channels,
                            const int32_t out_channels) {
  const float bn_eps = 1e-3f;
  
  for (int oc = 0; oc < out_channels; oc++) {
    float gamma = (float)bn_weight[oc];
    float beta = (float)bn_bias[oc];
    float mean = (float)bn_mean[oc];
    float var = (float)bn_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
          int input_idx = (h * width + w) * in_channels + ic;
          int weight_idx = oc * in_channels + ic;
          sum += (float)input[input_idx] * (float)weights[weight_idx];
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        float activated = silu(bn_out);
        
        int output_idx = (h * width + w) * out_channels + oc;
        output[output_idx] = (bfloat16)activated;
      }
    }
  }
}

//*****************************************************************************
// Conv3x3 + BatchNorm + SiLU helper function
//*****************************************************************************
inline void conv3x3_bn_silu(bfloat16 *input,
                            bfloat16 *weights,
                            bfloat16 *bn_weight,
                            bfloat16 *bn_bias,
                            bfloat16 *bn_mean,
                            bfloat16 *bn_var,
                            bfloat16 *output,
                            const int32_t height,
                            const int32_t width,
                            const int32_t in_channels,
                            const int32_t out_channels,
                            const int32_t padding) {
  const float bn_eps = 1e-3f;
  
  for (int oc = 0; oc < out_channels; oc++) {
    float gamma = (float)bn_weight[oc];
    float beta = (float)bn_bias[oc];
    float mean = (float)bn_mean[oc];
    float var = (float)bn_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = h + kh - padding;
              int iw = w + kw - padding;
              
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = (ih * width + iw) * in_channels + ic;
                int weight_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;
                sum += (float)input[input_idx] * (float)weights[weight_idx];
              }
            }
          }
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        float activated = silu(bn_out);
        
        int output_idx = (h * width + w) * out_channels + oc;
        output[output_idx] = (bfloat16)activated;
      }
    }
  }
}

//*****************************************************************************
// 4-way channel concatenation in HWC format
//*****************************************************************************
inline void concat_4way_channels(bfloat16 *x1,
                                 bfloat16 *x2,
                                 bfloat16 *x3,
                                 bfloat16 *x4,
                                 bfloat16 *output,
                                 const int32_t height,
                                 const int32_t width,
                                 const int32_t c1,
                                 const int32_t c2,
                                 const int32_t c3,
                                 const int32_t c4) {
  const int total_channels = c1 + c2 + c3 + c4;
  
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int spatial_idx = h * width + w;
      int out_offset = spatial_idx * total_channels;
      
      // Copy x1 channels
      for (int c = 0; c < c1; c++) {
        output[out_offset + c] = x1[spatial_idx * c1 + c];
      }
      
      // Copy x2 channels
      for (int c = 0; c < c2; c++) {
        output[out_offset + c1 + c] = x2[spatial_idx * c2 + c];
      }
      
      // Copy x3 channels
      for (int c = 0; c < c3; c++) {
        output[out_offset + c1 + c2 + c] = x3[spatial_idx * c3 + c];
      }
      
      // Copy x4 channels
      for (int c = 0; c < c4; c++) {
        output[out_offset + c1 + c2 + c3 + c] = x4[spatial_idx * c4 + c];
      }
    }
  }
}

//*****************************************************************************
// ELAN - bfloat16
// ELAN-1 (Efficient Layer Aggregation Network) structure
//
// Architecture:
//   Input → Conv1 (1×1) → split into [x1, x2]
//                              ↓
//                          x2 → Conv2 (3×3) → x3
//                                  ↓
//                              x3 → Conv3 (3×3) → x4
//                                  ↓
//                    Concat [x1, x2, x3, x4] (4-way)
//                                  ↓
//                          Conv4 (1×1) → Output
//
// Multi-scale feature aggregation through sequential processing
//
// Default parameters:
// - process_channels = part_channels // 2
// - All convs have BN + SiLU
//
// Weights layout: [conv1_wts+bn, conv2_wts+bn, conv3_wts+bn, conv4_wts+bn]
//*****************************************************************************
void elan_bf16_scalar(bfloat16 *input,
                      bfloat16 *weights_and_bn,
                      bfloat16 *output,
                      bfloat16 *conv1_output,  // Contains x1 and x2 (split)
                      bfloat16 *x3,            // Conv2 output
                      bfloat16 *x4,            // Conv3 output
                      bfloat16 *concat_buffer, // 4-way concatenation
                      const int32_t height,
                      const int32_t width,
                      const int32_t in_channels,
                      const int32_t out_channels,
                      const int32_t part_channels,
                      const int32_t process_channels) {
  event0();

  const int half_part = part_channels / 2;
  const int concat_channels = part_channels + 2 * process_channels;
  
  // Extract weight pointers
  // Conv1 (1×1): in_channels → part_channels
  const int conv1_weight_size = part_channels * in_channels;
  bfloat16 *conv1_weights = weights_and_bn;
  bfloat16 *conv1_bn_weight = conv1_weights + conv1_weight_size;
  bfloat16 *conv1_bn_bias = conv1_bn_weight + part_channels;
  bfloat16 *conv1_bn_mean = conv1_bn_bias + part_channels;
  bfloat16 *conv1_bn_var = conv1_bn_mean + part_channels;
  
  // Conv2 (3×3): half_part → process_channels
  const int conv2_weight_size = process_channels * half_part * 3 * 3;
  bfloat16 *conv2_weights = conv1_bn_var + part_channels;
  bfloat16 *conv2_bn_weight = conv2_weights + conv2_weight_size;
  bfloat16 *conv2_bn_bias = conv2_bn_weight + process_channels;
  bfloat16 *conv2_bn_mean = conv2_bn_bias + process_channels;
  bfloat16 *conv2_bn_var = conv2_bn_mean + process_channels;
  
  // Conv3 (3×3): process_channels → process_channels
  const int conv3_weight_size = process_channels * process_channels * 3 * 3;
  bfloat16 *conv3_weights = conv2_bn_var + process_channels;
  bfloat16 *conv3_bn_weight = conv3_weights + conv3_weight_size;
  bfloat16 *conv3_bn_bias = conv3_bn_weight + process_channels;
  bfloat16 *conv3_bn_mean = conv3_bn_bias + process_channels;
  bfloat16 *conv3_bn_var = conv3_bn_mean + process_channels;
  
  // Conv4 (1×1): concat_channels → out_channels
  const int conv4_weight_size = out_channels * concat_channels;
  bfloat16 *conv4_weights = conv3_bn_var + process_channels;
  bfloat16 *conv4_bn_weight = conv4_weights + conv4_weight_size;
  bfloat16 *conv4_bn_bias = conv4_bn_weight + out_channels;
  bfloat16 *conv4_bn_mean = conv4_bn_bias + out_channels;
  bfloat16 *conv4_bn_var = conv4_bn_mean + out_channels;

  //===========================================================================
  // Stage 1: Conv1 (1×1) + BN + SiLU
  //===========================================================================
  conv1x1_bn_silu(input, conv1_weights, conv1_bn_weight, conv1_bn_bias,
                  conv1_bn_mean, conv1_bn_var, conv1_output,
                  height, width, in_channels, part_channels);

  //===========================================================================
  // Stage 2: Split conv1_output into x1 and x2 (pointer arithmetic, no copy)
  //===========================================================================
  bfloat16 *x1 = conv1_output;
  bfloat16 *x2 = conv1_output + (height * width * half_part);

  //===========================================================================
  // Stage 3: Conv2 (3×3) + BN + SiLU on x2
  //===========================================================================
  conv3x3_bn_silu(x2, conv2_weights, conv2_bn_weight, conv2_bn_bias,
                  conv2_bn_mean, conv2_bn_var, x3,
                  height, width, half_part, process_channels, 1);

  //===========================================================================
  // Stage 4: Conv3 (3×3) + BN + SiLU on x3
  //===========================================================================
  conv3x3_bn_silu(x3, conv3_weights, conv3_bn_weight, conv3_bn_bias,
                  conv3_bn_mean, conv3_bn_var, x4,
                  height, width, process_channels, process_channels, 1);

  //===========================================================================
  // Stage 5: 4-way Concatenation [x1, x2, x3, x4]
  //===========================================================================
  concat_4way_channels(x1, x2, x3, x4, concat_buffer,
                       height, width, half_part, half_part, 
                       process_channels, process_channels);

  //===========================================================================
  // Stage 6: Conv4 (1×1) + BN + SiLU
  //===========================================================================
  conv1x1_bn_silu(concat_buffer, conv4_weights, conv4_bn_weight, conv4_bn_bias,
                  conv4_bn_mean, conv4_bn_var, output,
                  height, width, concat_channels, out_channels);

  event1();
}

extern "C" {

void elan_bf16(bfloat16 *input,
               bfloat16 *weights_and_bn,
               bfloat16 *output,
               bfloat16 *conv1_output,
               bfloat16 *x3,
               bfloat16 *x4,
               bfloat16 *concat_buffer,
               int32_t height,
               int32_t width,
               int32_t in_channels,
               int32_t out_channels,
               int32_t part_channels,
               int32_t process_channels) {
  elan_bf16_scalar(input, weights_and_bn, output,
                   conv1_output, x3, x4, concat_buffer,
                   height, width, in_channels, out_channels,
                   part_channels, process_channels);
}

} // extern "C"
