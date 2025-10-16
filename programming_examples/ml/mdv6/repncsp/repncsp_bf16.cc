//===- repncsp_bf16.cc -------------------------------------------*- C++ -*-===//
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
// Optimized for 1×1 convolutions (no spatial kernel)
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
        
        // 1×1 convolution (point-wise)
        for (int ic = 0; ic < in_channels; ic++) {
          int input_idx = (h * width + w) * in_channels + ic;
          int weight_idx = oc * in_channels + ic;
          sum += (float)input[input_idx] * (float)weights[weight_idx];
        }
        
        // Apply BatchNorm + SiLU
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        float activated = silu(bn_out);
        
        int output_idx = (h * width + w) * out_channels + oc;
        output[output_idx] = (bfloat16)activated;
      }
    }
  }
}

//*****************************************************************************
// Channel concatenation in HWC format
// Concatenates x1 and x2 along channel dimension
//*****************************************************************************
inline void concat_channels(bfloat16 *x1,
                            bfloat16 *x2,
                            bfloat16 *output,
                            const int32_t height,
                            const int32_t width,
                            const int32_t channels1,
                            const int32_t channels2) {
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int spatial_idx = h * width + w;
      
      // Copy x1 channels
      for (int c = 0; c < channels1; c++) {
        int src_idx = spatial_idx * channels1 + c;
        int dst_idx = spatial_idx * (channels1 + channels2) + c;
        output[dst_idx] = x1[src_idx];
      }
      
      // Copy x2 channels
      for (int c = 0; c < channels2; c++) {
        int src_idx = spatial_idx * channels2 + c;
        int dst_idx = spatial_idx * (channels1 + channels2) + channels1 + c;
        output[dst_idx] = x2[src_idx];
      }
    }
  }
}

//*****************************************************************************
// RepNCSP - bfloat16
// RepNCSP block with RepConv bottlenecks (CSP architecture)
//
// Architecture:
//   Input → Conv1 → Bottleneck(s) → x1 ─┐
//                                        ├→ Concat → Conv3 → Output
//   Input → Conv2 ──────────────────→ x2 ─┘
//
// CSP (Cross Stage Partial) splits processing into two paths:
// - Path 1: Deep processing with bottlenecks
// - Path 2: Shallow bypass
// - Merge: Concatenate and project to output channels
//
// Default parameters:
// - kernel_size=1 (all 1×1 convs)
// - csp_expand=0.5 (neck_channels = out_channels / 2)
// - repeat_num=1 (single bottleneck)
//
// Weights layout: [conv1_wts+bn, bottleneck_wts, conv2_wts+bn, conv3_wts+bn]
//*****************************************************************************
void repncsp_bf16_scalar(bfloat16 *input,
                         bfloat16 *weights_and_bn,
                         bfloat16 *output,
                         bfloat16 *x1_conv1,      // Conv1 output
                         bfloat16 *x1_bottleneck, // Bottleneck output
                         bfloat16 *x2_conv2,      // Conv2 output
                         bfloat16 *concat_buffer, // Concatenated features
                         // Bottleneck internal buffers (reuse bottleneck logic)
                         bfloat16 *bn_input_copy,
                         bfloat16 *bn_temp1,
                         bfloat16 *bn_temp2,
                         bfloat16 *bn_temp3,
                         bfloat16 *bn_temp4,
                         const int32_t height,
                         const int32_t width,
                         const int32_t in_channels,
                         const int32_t out_channels,
                         const float csp_expand) {
  event0();

  const int neck_channels = (int)(out_channels * csp_expand);
  const int total_concat_channels = 2 * neck_channels;
  
  // Extract weight pointers
  // Conv1 (1×1): in_channels → neck_channels
  const int conv1_weight_size = neck_channels * in_channels;
  bfloat16 *conv1_weights = weights_and_bn;
  bfloat16 *conv1_bn_weight = conv1_weights + conv1_weight_size;
  bfloat16 *conv1_bn_bias = conv1_bn_weight + neck_channels;
  bfloat16 *conv1_bn_mean = conv1_bn_bias + neck_channels;
  bfloat16 *conv1_bn_var = conv1_bn_mean + neck_channels;
  
  // Bottleneck weights (neck_channels → neck_channels)
  // Layout: [repconv_weights, conv2_weights]
  bfloat16 *bottleneck_weights = conv1_bn_var + neck_channels;
  
  // Calculate bottleneck weight size
  const int bn_conv3x3_size = neck_channels * neck_channels * 3 * 3;
  const int bn_conv1x1_size = neck_channels * neck_channels * 1 * 1;
  const int bn_bn_params = 4 * neck_channels;
  const int bn_conv2_size = neck_channels * neck_channels * 3 * 3;
  const int bottleneck_weight_size = bn_conv3x3_size + bn_bn_params + 
                                      bn_conv1x1_size + bn_bn_params +
                                      bn_conv2_size + bn_bn_params;
  
  // Conv2 (1×1): in_channels → neck_channels
  bfloat16 *conv2_weights = bottleneck_weights + bottleneck_weight_size;
  bfloat16 *conv2_bn_weight = conv2_weights + conv1_weight_size;
  bfloat16 *conv2_bn_bias = conv2_bn_weight + neck_channels;
  bfloat16 *conv2_bn_mean = conv2_bn_bias + neck_channels;
  bfloat16 *conv2_bn_var = conv2_bn_mean + neck_channels;
  
  // Conv3 (1×1): total_concat_channels → out_channels
  const int conv3_weight_size = out_channels * total_concat_channels;
  bfloat16 *conv3_weights = conv2_bn_var + neck_channels;
  bfloat16 *conv3_bn_weight = conv3_weights + conv3_weight_size;
  bfloat16 *conv3_bn_bias = conv3_bn_weight + out_channels;
  bfloat16 *conv3_bn_mean = conv3_bn_bias + out_channels;
  bfloat16 *conv3_bn_var = conv3_bn_mean + out_channels;

  //===========================================================================
  // Stage 1: Conv1 (1×1) + BN + SiLU
  //===========================================================================
  conv1x1_bn_silu(input, conv1_weights, conv1_bn_weight, conv1_bn_bias,
                  conv1_bn_mean, conv1_bn_var, x1_conv1,
                  height, width, in_channels, neck_channels);

  //===========================================================================
  // Stage 2: Bottleneck (inline the bottleneck logic)
  // For simplicity with repeat_num=1, we inline one bottleneck
  // Bottleneck = RepConv → Conv+BN+SiLU → residual
  //===========================================================================
  
  // Extract bottleneck weight pointers
  const float bn_eps = 1e-3f;
  bfloat16 *bn_conv3x3_weights = bottleneck_weights;
  bfloat16 *bn_bn3x3_weight = bn_conv3x3_weights + bn_conv3x3_size;
  bfloat16 *bn_bn3x3_bias = bn_bn3x3_weight + neck_channels;
  bfloat16 *bn_bn3x3_mean = bn_bn3x3_bias + neck_channels;
  bfloat16 *bn_bn3x3_var = bn_bn3x3_mean + neck_channels;
  
  bfloat16 *bn_conv1x1_weights = bn_bn3x3_var + neck_channels;
  bfloat16 *bn_bn1x1_weight = bn_conv1x1_weights + bn_conv1x1_size;
  bfloat16 *bn_bn1x1_bias = bn_bn1x1_weight + neck_channels;
  bfloat16 *bn_bn1x1_mean = bn_bn1x1_bias + neck_channels;
  bfloat16 *bn_bn1x1_var = bn_bn1x1_mean + neck_channels;
  
  bfloat16 *bn_conv2_weights = bn_bn1x1_var + neck_channels;
  bfloat16 *bn_bn2_weight = bn_conv2_weights + bn_conv2_size;
  bfloat16 *bn_bn2_bias = bn_bn2_weight + neck_channels;
  bfloat16 *bn_bn2_mean = bn_bn2_bias + neck_channels;
  bfloat16 *bn_bn2_var = bn_bn2_mean + neck_channels;
  
  // Copy input for bottleneck residual (neck_channels == neck_channels, so residual active)
  const int neck_size = height * width * neck_channels;
  for (int i = 0; i < neck_size; i++) {
    bn_input_copy[i] = x1_conv1[i];
  }
  
  // Bottleneck Stage 1a: Conv3x3 + BN (RepConv branch 1)
  for (int oc = 0; oc < neck_channels; oc++) {
    float gamma = (float)bn_bn3x3_weight[oc];
    float beta = (float)bn_bn3x3_bias[oc];
    float mean = (float)bn_bn3x3_mean[oc];
    float var = (float)bn_bn3x3_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < neck_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = h + kh - 1;  // padding=1
              int iw = w + kw - 1;
              
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = (ih * width + iw) * neck_channels + ic;
                int weight_idx = ((oc * neck_channels + ic) * 3 + kh) * 3 + kw;
                sum += (float)x1_conv1[input_idx] * (float)bn_conv3x3_weights[weight_idx];
              }
            }
          }
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (h * width + w) * neck_channels + oc;
        bn_temp1[temp_idx] = (bfloat16)bn_out;
      }
    }
  }
  
  // Bottleneck Stage 1b: Conv1x1 + BN (RepConv branch 2)
  for (int oc = 0; oc < neck_channels; oc++) {
    float gamma = (float)bn_bn1x1_weight[oc];
    float beta = (float)bn_bn1x1_bias[oc];
    float mean = (float)bn_bn1x1_mean[oc];
    float var = (float)bn_bn1x1_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < neck_channels; ic++) {
          int input_idx = (h * width + w) * neck_channels + ic;
          int weight_idx = oc * neck_channels + ic;
          sum += (float)x1_conv1[input_idx] * (float)bn_conv1x1_weights[weight_idx];
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (h * width + w) * neck_channels + oc;
        bn_temp2[temp_idx] = (bfloat16)bn_out;
      }
    }
  }
  
  // Bottleneck Stage 1c: Add + SiLU (complete RepConv)
  for (int i = 0; i < neck_size; i++) {
    float sum = (float)bn_temp1[i] + (float)bn_temp2[i];
    bn_temp3[i] = (bfloat16)silu(sum);
  }
  
  // Bottleneck Stage 2: Conv3x3 + BN + SiLU
  for (int oc = 0; oc < neck_channels; oc++) {
    float gamma = (float)bn_bn2_weight[oc];
    float beta = (float)bn_bn2_bias[oc];
    float mean = (float)bn_bn2_mean[oc];
    float var = (float)bn_bn2_var[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < neck_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = h + kh - 1;  // padding=1
              int iw = w + kw - 1;
              
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int temp_idx = (ih * width + iw) * neck_channels + ic;
                int weight_idx = ((oc * neck_channels + ic) * 3 + kh) * 3 + kw;
                sum += (float)bn_temp3[temp_idx] * (float)bn_conv2_weights[weight_idx];
              }
            }
          }
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        float activated = silu(bn_out);
        int temp_idx = (h * width + w) * neck_channels + oc;
        bn_temp4[temp_idx] = (bfloat16)activated;
      }
    }
  }
  
  // Bottleneck Stage 3: Residual (neck_channels == neck_channels, so active)
  for (int i = 0; i < neck_size; i++) {
    x1_bottleneck[i] = (bfloat16)((float)bn_input_copy[i] + (float)bn_temp4[i]);
  }

  //===========================================================================
  // Stage 3: Conv2 (1×1) + BN + SiLU (bypass path)
  //===========================================================================
  conv1x1_bn_silu(input, conv2_weights, conv2_bn_weight, conv2_bn_bias,
                  conv2_bn_mean, conv2_bn_var, x2_conv2,
                  height, width, in_channels, neck_channels);

  //===========================================================================
  // Stage 4: Concatenation [x1_bottleneck, x2_conv2]
  //===========================================================================
  concat_channels(x1_bottleneck, x2_conv2, concat_buffer,
                  height, width, neck_channels, neck_channels);

  //===========================================================================
  // Stage 5: Conv3 (1×1) + BN + SiLU
  //===========================================================================
  conv1x1_bn_silu(concat_buffer, conv3_weights, conv3_bn_weight, conv3_bn_bias,
                  conv3_bn_mean, conv3_bn_var, output,
                  height, width, total_concat_channels, out_channels);

  event1();
}

extern "C" {

void repncsp_bf16(bfloat16 *input,
                  bfloat16 *weights_and_bn,
                  bfloat16 *output,
                  bfloat16 *x1_conv1,
                  bfloat16 *x1_bottleneck,
                  bfloat16 *x2_conv2,
                  bfloat16 *concat_buffer,
                  bfloat16 *bn_input_copy,
                  bfloat16 *bn_temp1,
                  bfloat16 *bn_temp2,
                  bfloat16 *bn_temp3,
                  bfloat16 *bn_temp4,
                  int32_t height,
                  int32_t width,
                  int32_t in_channels,
                  int32_t out_channels,
                  float csp_expand) {
  repncsp_bf16_scalar(input, weights_and_bn, output,
                      x1_conv1, x1_bottleneck, x2_conv2, concat_buffer,
                      bn_input_copy, bn_temp1, bn_temp2, bn_temp3, bn_temp4,
                      height, width, in_channels, out_channels, csp_expand);
}

} // extern "C"
