//===- repncsp_elan_bf16.cc --------------------------------------*- C++ -*-===//
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
// 2-way channel concatenation in HWC format
//*****************************************************************************
inline void concat_2way_channels(bfloat16 *x1,
                                 bfloat16 *x2,
                                 bfloat16 *output,
                                 const int32_t height,
                                 const int32_t width,
                                 const int32_t c1,
                                 const int32_t c2) {
  const int total_channels = c1 + c2;
  
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
// RepNCSPELAN - bfloat16
// RepNCSPELAN block combining RepNCSP with ELAN structure
//
// Architecture:
//   Input → Conv1 (1×1) → split into [x1, x2]
//                              ↓
//                          x2 → RepNCSP → Conv3x3 → x3
//                                              ↓
//                                          x3 → RepNCSP → Conv3x3 → x4
//                              ↓              ↓              ↓
//                    Concat [x1, x2, x3, x4] (4-way)
//                              ↓
//                          Conv4 (1×1) → Output
//
// Most complex layer in MDV6 with nested RepNCSP blocks
//
// Default parameters:
// - process_channels = part_channels // 2
// - csp_expand = 0.5 (for RepNCSP blocks)
// - All convs have BN + SiLU
//
// Weights layout: [conv1, repncsp1, conv3x3_1, repncsp2, conv3x3_2, conv4]
//*****************************************************************************
void repncsp_elan_bf16_scalar(
    bfloat16 *input,
    bfloat16 *weights_and_bn,
    bfloat16 *output,
    // Main stage buffers
    bfloat16 *conv1_output,      // Conv1 output (contains x1 and x2)
    bfloat16 *x3_repncsp_out,    // RepNCSP #1 output
    bfloat16 *x3_conv_out,       // Conv3x3 #1 output (final x3)
    bfloat16 *x4_repncsp_out,    // RepNCSP #2 output
    bfloat16 *x4_conv_out,       // Conv3x3 #2 output (final x4)
    bfloat16 *concat_buffer,     // 4-way concatenation
    // RepNCSP #1 internal buffers
    bfloat16 *rn1_conv1_out,
    bfloat16 *rn1_bottleneck_out,
    bfloat16 *rn1_conv2_out,
    bfloat16 *rn1_concat,
    bfloat16 *rn1_bn_input_copy,
    bfloat16 *rn1_bn_temp1,
    bfloat16 *rn1_bn_temp2,
    bfloat16 *rn1_bn_temp3,
    bfloat16 *rn1_bn_temp4,
    // RepNCSP #2 internal buffers
    bfloat16 *rn2_conv1_out,
    bfloat16 *rn2_bottleneck_out,
    bfloat16 *rn2_conv2_out,
    bfloat16 *rn2_concat,
    bfloat16 *rn2_bn_input_copy,
    bfloat16 *rn2_bn_temp1,
    bfloat16 *rn2_bn_temp2,
    bfloat16 *rn2_bn_temp3,
    bfloat16 *rn2_bn_temp4,
    // Dimensions
    const int32_t height,
    const int32_t width,
    const int32_t in_channels,
    const int32_t out_channels,
    const int32_t part_channels,
    const int32_t process_channels) {
  
  event0();

  const float bn_eps = 1e-3f;
  const int half_part = part_channels / 2;
  const int concat_channels = part_channels + 2 * process_channels;
  
  // RepNCSP parameters (csp_expand = 0.5)
  const int rn1_neck = process_channels / 2;  // For RepNCSP #1
  const int rn2_neck = process_channels / 2;  // For RepNCSP #2
  
  //===========================================================================
  // Extract weight pointers - Complex nested structure
  //===========================================================================
  
  // Conv1 (1×1): in_channels → part_channels
  const int conv1_weight_size = part_channels * in_channels;
  bfloat16 *conv1_weights = weights_and_bn;
  bfloat16 *conv1_bn_weight = conv1_weights + conv1_weight_size;
  bfloat16 *conv1_bn_bias = conv1_bn_weight + part_channels;
  bfloat16 *conv1_bn_mean = conv1_bn_bias + part_channels;
  bfloat16 *conv1_bn_var = conv1_bn_mean + part_channels;
  
  // RepNCSP #1 weights (half_part → process_channels)
  bfloat16 *rn1_weights = conv1_bn_var + part_channels;
  
  // RepNCSP #1 internal structure:
  // - Conv1 (1×1): half_part → rn1_neck
  const int rn1_conv1_wsize = rn1_neck * half_part;
  bfloat16 *rn1_conv1_w = rn1_weights;
  bfloat16 *rn1_conv1_bn_w = rn1_conv1_w + rn1_conv1_wsize;
  bfloat16 *rn1_conv1_bn_b = rn1_conv1_bn_w + rn1_neck;
  bfloat16 *rn1_conv1_bn_m = rn1_conv1_bn_b + rn1_neck;
  bfloat16 *rn1_conv1_bn_v = rn1_conv1_bn_m + rn1_neck;
  
  // - Bottleneck (rn1_neck → rn1_neck)
  //   - RepConv: Conv3x3+BN, Conv1x1+BN
  const int rn1_bn_conv3x3_wsize = rn1_neck * rn1_neck * 9;
  bfloat16 *rn1_bn_conv3x3_w = rn1_conv1_bn_v + rn1_neck;
  bfloat16 *rn1_bn_bn3x3_w = rn1_bn_conv3x3_w + rn1_bn_conv3x3_wsize;
  bfloat16 *rn1_bn_bn3x3_b = rn1_bn_bn3x3_w + rn1_neck;
  bfloat16 *rn1_bn_bn3x3_m = rn1_bn_bn3x3_b + rn1_neck;
  bfloat16 *rn1_bn_bn3x3_v = rn1_bn_bn3x3_m + rn1_neck;
  
  const int rn1_bn_conv1x1_wsize = rn1_neck * rn1_neck;
  bfloat16 *rn1_bn_conv1x1_w = rn1_bn_bn3x3_v + rn1_neck;
  bfloat16 *rn1_bn_bn1x1_w = rn1_bn_conv1x1_w + rn1_bn_conv1x1_wsize;
  bfloat16 *rn1_bn_bn1x1_b = rn1_bn_bn1x1_w + rn1_neck;
  bfloat16 *rn1_bn_bn1x1_m = rn1_bn_bn1x1_b + rn1_neck;
  bfloat16 *rn1_bn_bn1x1_v = rn1_bn_bn1x1_m + rn1_neck;
  
  //   - Conv2 (3×3): rn1_neck → rn1_neck
  const int rn1_bn_conv2_wsize = rn1_neck * rn1_neck * 9;
  bfloat16 *rn1_bn_conv2_w = rn1_bn_bn1x1_v + rn1_neck;
  bfloat16 *rn1_bn_bn2_w = rn1_bn_conv2_w + rn1_bn_conv2_wsize;
  bfloat16 *rn1_bn_bn2_b = rn1_bn_bn2_w + rn1_neck;
  bfloat16 *rn1_bn_bn2_m = rn1_bn_bn2_b + rn1_neck;
  bfloat16 *rn1_bn_bn2_v = rn1_bn_bn2_m + rn1_neck;
  
  // - Conv2 (1×1): half_part → rn1_neck (bypass)
  const int rn1_conv2_wsize = rn1_neck * half_part;
  bfloat16 *rn1_conv2_w = rn1_bn_bn2_v + rn1_neck;
  bfloat16 *rn1_conv2_bn_w = rn1_conv2_w + rn1_conv2_wsize;
  bfloat16 *rn1_conv2_bn_b = rn1_conv2_bn_w + rn1_neck;
  bfloat16 *rn1_conv2_bn_m = rn1_conv2_bn_b + rn1_neck;
  bfloat16 *rn1_conv2_bn_v = rn1_conv2_bn_m + rn1_neck;
  
  // - Conv3 (1×1): 2*rn1_neck → process_channels (merge)
  const int rn1_conv3_wsize = process_channels * 2 * rn1_neck;
  bfloat16 *rn1_conv3_w = rn1_conv2_bn_v + rn1_neck;
  bfloat16 *rn1_conv3_bn_w = rn1_conv3_w + rn1_conv3_wsize;
  bfloat16 *rn1_conv3_bn_b = rn1_conv3_bn_w + process_channels;
  bfloat16 *rn1_conv3_bn_m = rn1_conv3_bn_b + process_channels;
  bfloat16 *rn1_conv3_bn_v = rn1_conv3_bn_m + process_channels;
  
  // Conv3x3 #1 (3×3): process_channels → process_channels
  const int conv3x3_1_wsize = process_channels * process_channels * 9;
  bfloat16 *conv3x3_1_w = rn1_conv3_bn_v + process_channels;
  bfloat16 *conv3x3_1_bn_w = conv3x3_1_w + conv3x3_1_wsize;
  bfloat16 *conv3x3_1_bn_b = conv3x3_1_bn_w + process_channels;
  bfloat16 *conv3x3_1_bn_m = conv3x3_1_bn_b + process_channels;
  bfloat16 *conv3x3_1_bn_v = conv3x3_1_bn_m + process_channels;
  
  // RepNCSP #2 weights (process_channels → process_channels)
  bfloat16 *rn2_weights = conv3x3_1_bn_v + process_channels;
  
  // RepNCSP #2 internal structure (same as #1 but different input size)
  // - Conv1 (1×1): process_channels → rn2_neck
  const int rn2_conv1_wsize = rn2_neck * process_channels;
  bfloat16 *rn2_conv1_w = rn2_weights;
  bfloat16 *rn2_conv1_bn_w = rn2_conv1_w + rn2_conv1_wsize;
  bfloat16 *rn2_conv1_bn_b = rn2_conv1_bn_w + rn2_neck;
  bfloat16 *rn2_conv1_bn_m = rn2_conv1_bn_b + rn2_neck;
  bfloat16 *rn2_conv1_bn_v = rn2_conv1_bn_m + rn2_neck;
  
  // - Bottleneck (rn2_neck → rn2_neck)
  const int rn2_bn_conv3x3_wsize = rn2_neck * rn2_neck * 9;
  bfloat16 *rn2_bn_conv3x3_w = rn2_conv1_bn_v + rn2_neck;
  bfloat16 *rn2_bn_bn3x3_w = rn2_bn_conv3x3_w + rn2_bn_conv3x3_wsize;
  bfloat16 *rn2_bn_bn3x3_b = rn2_bn_bn3x3_w + rn2_neck;
  bfloat16 *rn2_bn_bn3x3_m = rn2_bn_bn3x3_b + rn2_neck;
  bfloat16 *rn2_bn_bn3x3_v = rn2_bn_bn3x3_m + rn2_neck;
  
  const int rn2_bn_conv1x1_wsize = rn2_neck * rn2_neck;
  bfloat16 *rn2_bn_conv1x1_w = rn2_bn_bn3x3_v + rn2_neck;
  bfloat16 *rn2_bn_bn1x1_w = rn2_bn_conv1x1_w + rn2_bn_conv1x1_wsize;
  bfloat16 *rn2_bn_bn1x1_b = rn2_bn_bn1x1_w + rn2_neck;
  bfloat16 *rn2_bn_bn1x1_m = rn2_bn_bn1x1_b + rn2_neck;
  bfloat16 *rn2_bn_bn1x1_v = rn2_bn_bn1x1_m + rn2_neck;
  
  const int rn2_bn_conv2_wsize = rn2_neck * rn2_neck * 9;
  bfloat16 *rn2_bn_conv2_w = rn2_bn_bn1x1_v + rn2_neck;
  bfloat16 *rn2_bn_bn2_w = rn2_bn_conv2_w + rn2_bn_conv2_wsize;
  bfloat16 *rn2_bn_bn2_b = rn2_bn_bn2_w + rn2_neck;
  bfloat16 *rn2_bn_bn2_m = rn2_bn_bn2_b + rn2_neck;
  bfloat16 *rn2_bn_bn2_v = rn2_bn_bn2_m + rn2_neck;
  
  // - Conv2 (1×1): process_channels → rn2_neck (bypass)
  const int rn2_conv2_wsize = rn2_neck * process_channels;
  bfloat16 *rn2_conv2_w = rn2_bn_bn2_v + rn2_neck;
  bfloat16 *rn2_conv2_bn_w = rn2_conv2_w + rn2_conv2_wsize;
  bfloat16 *rn2_conv2_bn_b = rn2_conv2_bn_w + rn2_neck;
  bfloat16 *rn2_conv2_bn_m = rn2_conv2_bn_b + rn2_neck;
  bfloat16 *rn2_conv2_bn_v = rn2_conv2_bn_m + rn2_neck;
  
  // - Conv3 (1×1): 2*rn2_neck → process_channels (merge)
  const int rn2_conv3_wsize = process_channels * 2 * rn2_neck;
  bfloat16 *rn2_conv3_w = rn2_conv2_bn_v + rn2_neck;
  bfloat16 *rn2_conv3_bn_w = rn2_conv3_w + rn2_conv3_wsize;
  bfloat16 *rn2_conv3_bn_b = rn2_conv3_bn_w + process_channels;
  bfloat16 *rn2_conv3_bn_m = rn2_conv3_bn_b + process_channels;
  bfloat16 *rn2_conv3_bn_v = rn2_conv3_bn_m + process_channels;
  
  // Conv3x3 #2 (3×3): process_channels → process_channels
  const int conv3x3_2_wsize = process_channels * process_channels * 9;
  bfloat16 *conv3x3_2_w = rn2_conv3_bn_v + process_channels;
  bfloat16 *conv3x3_2_bn_w = conv3x3_2_w + conv3x3_2_wsize;
  bfloat16 *conv3x3_2_bn_b = conv3x3_2_bn_w + process_channels;
  bfloat16 *conv3x3_2_bn_m = conv3x3_2_bn_b + process_channels;
  bfloat16 *conv3x3_2_bn_v = conv3x3_2_bn_m + process_channels;
  
  // Conv4 (1×1): concat_channels → out_channels
  const int conv4_wsize = out_channels * concat_channels;
  bfloat16 *conv4_w = conv3x3_2_bn_v + process_channels;
  bfloat16 *conv4_bn_w = conv4_w + conv4_wsize;
  bfloat16 *conv4_bn_b = conv4_bn_w + out_channels;
  bfloat16 *conv4_bn_m = conv4_bn_b + out_channels;
  bfloat16 *conv4_bn_v = conv4_bn_m + out_channels;

  //===========================================================================
  // Stage 1: Conv1 (1×1) + BN + SiLU → split into [x1, x2]
  //===========================================================================
  conv1x1_bn_silu(input, conv1_weights, conv1_bn_weight, conv1_bn_bias,
                  conv1_bn_mean, conv1_bn_var, conv1_output,
                  height, width, in_channels, part_channels);
  
  // Split conv1_output into x1 and x2 (pointer arithmetic)
  bfloat16 *x1 = conv1_output;
  bfloat16 *x2 = conv1_output + (height * width * half_part);

  //===========================================================================
  // Stage 2: RepNCSP #1 (x2 → x3_repncsp_out)
  // Input: x2 (half_part channels)
  // Output: x3_repncsp_out (process_channels)
  //===========================================================================
  
  // RepNCSP #1 - Conv1 (1×1)
  conv1x1_bn_silu(x2, rn1_conv1_w, rn1_conv1_bn_w, rn1_conv1_bn_b,
                  rn1_conv1_bn_m, rn1_conv1_bn_v, rn1_conv1_out,
                  height, width, half_part, rn1_neck);
  
  // RepNCSP #1 - Bottleneck (rn1_neck → rn1_neck)
  const int rn1_neck_size = height * width * rn1_neck;
  
  // Copy input for residual
  for (int i = 0; i < rn1_neck_size; i++) {
    rn1_bn_input_copy[i] = rn1_conv1_out[i];
  }
  
  // Bottleneck - RepConv branch 1: Conv3x3 + BN
  for (int oc = 0; oc < rn1_neck; oc++) {
    float gamma = (float)rn1_bn_bn3x3_w[oc];
    float beta = (float)rn1_bn_bn3x3_b[oc];
    float mean = (float)rn1_bn_bn3x3_m[oc];
    float var = (float)rn1_bn_bn3x3_v[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < rn1_neck; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = h + kh - 1;
              int iw = w + kw - 1;
              
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = (ih * width + iw) * rn1_neck + ic;
                int weight_idx = ((oc * rn1_neck + ic) * 3 + kh) * 3 + kw;
                sum += (float)rn1_conv1_out[input_idx] * (float)rn1_bn_conv3x3_w[weight_idx];
              }
            }
          }
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (h * width + w) * rn1_neck + oc;
        rn1_bn_temp1[temp_idx] = (bfloat16)bn_out;
      }
    }
  }
  
  // Bottleneck - RepConv branch 2: Conv1x1 + BN
  for (int oc = 0; oc < rn1_neck; oc++) {
    float gamma = (float)rn1_bn_bn1x1_w[oc];
    float beta = (float)rn1_bn_bn1x1_b[oc];
    float mean = (float)rn1_bn_bn1x1_m[oc];
    float var = (float)rn1_bn_bn1x1_v[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < rn1_neck; ic++) {
          int input_idx = (h * width + w) * rn1_neck + ic;
          int weight_idx = oc * rn1_neck + ic;
          sum += (float)rn1_conv1_out[input_idx] * (float)rn1_bn_conv1x1_w[weight_idx];
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (h * width + w) * rn1_neck + oc;
        rn1_bn_temp2[temp_idx] = (bfloat16)bn_out;
      }
    }
  }
  
  // Bottleneck - Add + SiLU (complete RepConv)
  for (int i = 0; i < rn1_neck_size; i++) {
    float sum = (float)rn1_bn_temp1[i] + (float)rn1_bn_temp2[i];
    rn1_bn_temp3[i] = (bfloat16)silu(sum);
  }
  
  // Bottleneck - Conv2 (3×3) + BN + SiLU
  conv3x3_bn_silu(rn1_bn_temp3, rn1_bn_conv2_w, rn1_bn_bn2_w, rn1_bn_bn2_b,
                  rn1_bn_bn2_m, rn1_bn_bn2_v, rn1_bn_temp4,
                  height, width, rn1_neck, rn1_neck, 1);
  
  // Bottleneck - Residual add
  for (int i = 0; i < rn1_neck_size; i++) {
    rn1_bottleneck_out[i] = (bfloat16)((float)rn1_bn_input_copy[i] + (float)rn1_bn_temp4[i]);
  }
  
  // RepNCSP #1 - Conv2 (1×1) bypass path
  conv1x1_bn_silu(x2, rn1_conv2_w, rn1_conv2_bn_w, rn1_conv2_bn_b,
                  rn1_conv2_bn_m, rn1_conv2_bn_v, rn1_conv2_out,
                  height, width, half_part, rn1_neck);
  
  // RepNCSP #1 - Concatenate [bottleneck_out, conv2_out]
  concat_2way_channels(rn1_bottleneck_out, rn1_conv2_out, rn1_concat,
                       height, width, rn1_neck, rn1_neck);
  
  // RepNCSP #1 - Conv3 (1×1) merge
  conv1x1_bn_silu(rn1_concat, rn1_conv3_w, rn1_conv3_bn_w, rn1_conv3_bn_b,
                  rn1_conv3_bn_m, rn1_conv3_bn_v, x3_repncsp_out,
                  height, width, 2 * rn1_neck, process_channels);

  //===========================================================================
  // Stage 3: Conv3x3 #1 (x3_repncsp_out → x3_conv_out)
  //===========================================================================
  conv3x3_bn_silu(x3_repncsp_out, conv3x3_1_w, conv3x3_1_bn_w, conv3x3_1_bn_b,
                  conv3x3_1_bn_m, conv3x3_1_bn_v, x3_conv_out,
                  height, width, process_channels, process_channels, 1);

  //===========================================================================
  // Stage 4: RepNCSP #2 (x3_conv_out → x4_repncsp_out)
  // Input: x3_conv_out (process_channels)
  // Output: x4_repncsp_out (process_channels)
  //===========================================================================
  
  // RepNCSP #2 - Conv1 (1×1)
  conv1x1_bn_silu(x3_conv_out, rn2_conv1_w, rn2_conv1_bn_w, rn2_conv1_bn_b,
                  rn2_conv1_bn_m, rn2_conv1_bn_v, rn2_conv1_out,
                  height, width, process_channels, rn2_neck);
  
  // RepNCSP #2 - Bottleneck (rn2_neck → rn2_neck)
  const int rn2_neck_size = height * width * rn2_neck;
  
  // Copy input for residual
  for (int i = 0; i < rn2_neck_size; i++) {
    rn2_bn_input_copy[i] = rn2_conv1_out[i];
  }
  
  // Bottleneck - RepConv branch 1: Conv3x3 + BN
  for (int oc = 0; oc < rn2_neck; oc++) {
    float gamma = (float)rn2_bn_bn3x3_w[oc];
    float beta = (float)rn2_bn_bn3x3_b[oc];
    float mean = (float)rn2_bn_bn3x3_m[oc];
    float var = (float)rn2_bn_bn3x3_v[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < rn2_neck; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = h + kh - 1;
              int iw = w + kw - 1;
              
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = (ih * width + iw) * rn2_neck + ic;
                int weight_idx = ((oc * rn2_neck + ic) * 3 + kh) * 3 + kw;
                sum += (float)rn2_conv1_out[input_idx] * (float)rn2_bn_conv3x3_w[weight_idx];
              }
            }
          }
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (h * width + w) * rn2_neck + oc;
        rn2_bn_temp1[temp_idx] = (bfloat16)bn_out;
      }
    }
  }
  
  // Bottleneck - RepConv branch 2: Conv1x1 + BN
  for (int oc = 0; oc < rn2_neck; oc++) {
    float gamma = (float)rn2_bn_bn1x1_w[oc];
    float beta = (float)rn2_bn_bn1x1_b[oc];
    float mean = (float)rn2_bn_bn1x1_m[oc];
    float var = (float)rn2_bn_bn1x1_v[oc];
    float inv_std = 1.0f / fast_sqrt(var + bn_eps);
    
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < rn2_neck; ic++) {
          int input_idx = (h * width + w) * rn2_neck + ic;
          int weight_idx = oc * rn2_neck + ic;
          sum += (float)rn2_conv1_out[input_idx] * (float)rn2_bn_conv1x1_w[weight_idx];
        }
        
        float bn_out = gamma * (sum - mean) * inv_std + beta;
        int temp_idx = (h * width + w) * rn2_neck + oc;
        rn2_bn_temp2[temp_idx] = (bfloat16)bn_out;
      }
    }
  }
  
  // Bottleneck - Add + SiLU (complete RepConv)
  for (int i = 0; i < rn2_neck_size; i++) {
    float sum = (float)rn2_bn_temp1[i] + (float)rn2_bn_temp2[i];
    rn2_bn_temp3[i] = (bfloat16)silu(sum);
  }
  
  // Bottleneck - Conv2 (3×3) + BN + SiLU
  conv3x3_bn_silu(rn2_bn_temp3, rn2_bn_conv2_w, rn2_bn_bn2_w, rn2_bn_bn2_b,
                  rn2_bn_bn2_m, rn2_bn_bn2_v, rn2_bn_temp4,
                  height, width, rn2_neck, rn2_neck, 1);
  
  // Bottleneck - Residual add
  for (int i = 0; i < rn2_neck_size; i++) {
    rn2_bottleneck_out[i] = (bfloat16)((float)rn2_bn_input_copy[i] + (float)rn2_bn_temp4[i]);
  }
  
  // RepNCSP #2 - Conv2 (1×1) bypass path
  conv1x1_bn_silu(x3_conv_out, rn2_conv2_w, rn2_conv2_bn_w, rn2_conv2_bn_b,
                  rn2_conv2_bn_m, rn2_conv2_bn_v, rn2_conv2_out,
                  height, width, process_channels, rn2_neck);
  
  // RepNCSP #2 - Concatenate [bottleneck_out, conv2_out]
  concat_2way_channels(rn2_bottleneck_out, rn2_conv2_out, rn2_concat,
                       height, width, rn2_neck, rn2_neck);
  
  // RepNCSP #2 - Conv3 (1×1) merge
  conv1x1_bn_silu(rn2_concat, rn2_conv3_w, rn2_conv3_bn_w, rn2_conv3_bn_b,
                  rn2_conv3_bn_m, rn2_conv3_bn_v, x4_repncsp_out,
                  height, width, 2 * rn2_neck, process_channels);

  //===========================================================================
  // Stage 5: Conv3x3 #2 (x4_repncsp_out → x4_conv_out)
  //===========================================================================
  conv3x3_bn_silu(x4_repncsp_out, conv3x3_2_w, conv3x3_2_bn_w, conv3x3_2_bn_b,
                  conv3x3_2_bn_m, conv3x3_2_bn_v, x4_conv_out,
                  height, width, process_channels, process_channels, 1);

  //===========================================================================
  // Stage 6: 4-way Concatenation [x1, x2, x3_conv_out, x4_conv_out]
  //===========================================================================
  concat_4way_channels(x1, x2, x3_conv_out, x4_conv_out, concat_buffer,
                       height, width, half_part, half_part, 
                       process_channels, process_channels);

  //===========================================================================
  // Stage 7: Conv4 (1×1) + BN + SiLU
  //===========================================================================
  conv1x1_bn_silu(concat_buffer, conv4_w, conv4_bn_w, conv4_bn_b,
                  conv4_bn_m, conv4_bn_v, output,
                  height, width, concat_channels, out_channels);

  event1();
}

extern "C" {

void repncsp_elan_bf16(
    bfloat16 *input,
    bfloat16 *weights_and_bn,
    bfloat16 *output,
    bfloat16 *conv1_output,
    bfloat16 *x3_repncsp_out,
    bfloat16 *x3_conv_out,
    bfloat16 *x4_repncsp_out,
    bfloat16 *x4_conv_out,
    bfloat16 *concat_buffer,
    bfloat16 *rn1_conv1_out,
    bfloat16 *rn1_bottleneck_out,
    bfloat16 *rn1_conv2_out,
    bfloat16 *rn1_concat,
    bfloat16 *rn1_bn_input_copy,
    bfloat16 *rn1_bn_temp1,
    bfloat16 *rn1_bn_temp2,
    bfloat16 *rn1_bn_temp3,
    bfloat16 *rn1_bn_temp4,
    bfloat16 *rn2_conv1_out,
    bfloat16 *rn2_bottleneck_out,
    bfloat16 *rn2_conv2_out,
    bfloat16 *rn2_concat,
    bfloat16 *rn2_bn_input_copy,
    bfloat16 *rn2_bn_temp1,
    bfloat16 *rn2_bn_temp2,
    bfloat16 *rn2_bn_temp3,
    bfloat16 *rn2_bn_temp4,
    int32_t height,
    int32_t width,
    int32_t in_channels,
    int32_t out_channels,
    int32_t part_channels,
    int32_t process_channels) {
  
  repncsp_elan_bf16_scalar(
      input, weights_and_bn, output,
      conv1_output, x3_repncsp_out, x3_conv_out, x4_repncsp_out, x4_conv_out,
      concat_buffer,
      rn1_conv1_out, rn1_bottleneck_out, rn1_conv2_out, rn1_concat,
      rn1_bn_input_copy, rn1_bn_temp1, rn1_bn_temp2, rn1_bn_temp3, rn1_bn_temp4,
      rn2_conv1_out, rn2_bottleneck_out, rn2_conv2_out, rn2_concat,
      rn2_bn_input_copy, rn2_bn_temp1, rn2_bn_temp2, rn2_bn_temp3, rn2_bn_temp4,
      height, width, in_channels, out_channels, part_channels, process_channels);
}

} // extern "C"
