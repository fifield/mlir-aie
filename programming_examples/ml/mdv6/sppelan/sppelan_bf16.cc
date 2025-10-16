//===- sppelan_bf16.cc -------------------------------------------*- C++ -*-===//
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
// MaxPool2d operation
// Performs max pooling with specified kernel size, stride, and padding
//*****************************************************************************
inline void maxpool2d(bfloat16 *input,
                      bfloat16 *output,
                      const int32_t height,
                      const int32_t width,
                      const int32_t channels,
                      const int32_t kernel_size,
                      const int32_t stride,
                      const int32_t padding) {
  // Calculate output dimensions
  const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
  const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
  
  for (int c = 0; c < channels; c++) {
    for (int oh = 0; oh < out_height; oh++) {
      for (int ow = 0; ow < out_width; ow++) {
        float max_val = -3.4e38f;  // Very negative value for bfloat16 range
        
        // Scan the kernel window
        for (int kh = 0; kh < kernel_size; kh++) {
          for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride + kh - padding;
            int iw = ow * stride + kw - padding;
            
            // Check bounds
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
              int input_idx = (ih * width + iw) * channels + c;
              float val = (float)input[input_idx];
              if (val > max_val) {
                max_val = val;
              }
            }
          }
        }
        
        int output_idx = (oh * out_width + ow) * channels + c;
        output[output_idx] = (bfloat16)max_val;
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
// SPPELAN - bfloat16
// Spatial Pyramid Pooling ELAN structure
//
// Architecture:
//   Input → Conv1 (1×1) → features[0]
//               ↓
//           MaxPool(5×5, s=1, p=2) → features[1]
//               ↓
//           MaxPool(5×5, s=1, p=2) → features[2]
//               ↓
//           MaxPool(5×5, s=1, p=2) → features[3]
//               ↓
//       Concat [f0, f1, f2, f3] (4-way)
//               ↓
//           Conv5 (1×1) → Output
//
// Spatial pyramid pooling captures multi-scale features through
// successive max pooling operations
//
// Default parameters:
// - kernel_size=5, stride=1, padding=2 (maintains spatial dimensions)
// - All convs have BN + SiLU
//
// Weights layout: [conv1_wts+bn, conv5_wts+bn]
//*****************************************************************************
void sppelan_bf16_scalar(bfloat16 *input,
                         bfloat16 *weights_and_bn,
                         bfloat16 *output,
                         bfloat16 *conv1_output,   // Conv1 output (f0)
                         bfloat16 *pool1_output,   // First MaxPool output (f1)
                         bfloat16 *pool2_output,   // Second MaxPool output (f2)
                         bfloat16 *pool3_output,   // Third MaxPool output (f3)
                         bfloat16 *concat_buffer,  // 4-way concatenation
                         const int32_t height,
                         const int32_t width,
                         const int32_t in_channels,
                         const int32_t out_channels,
                         const int32_t neck_channels,
                         const int32_t kernel_size,
                         const int32_t stride,
                         const int32_t padding) {
  event0();

  const int concat_channels = 4 * neck_channels;
  
  // Extract weight pointers
  // Conv1 (1×1): in_channels → neck_channels
  const int conv1_weight_size = neck_channels * in_channels;
  bfloat16 *conv1_weights = weights_and_bn;
  bfloat16 *conv1_bn_weight = conv1_weights + conv1_weight_size;
  bfloat16 *conv1_bn_bias = conv1_bn_weight + neck_channels;
  bfloat16 *conv1_bn_mean = conv1_bn_bias + neck_channels;
  bfloat16 *conv1_bn_var = conv1_bn_mean + neck_channels;
  
  // Conv5 (1×1): concat_channels → out_channels
  const int conv5_weight_size = out_channels * concat_channels;
  bfloat16 *conv5_weights = conv1_bn_var + neck_channels;
  bfloat16 *conv5_bn_weight = conv5_weights + conv5_weight_size;
  bfloat16 *conv5_bn_bias = conv5_bn_weight + out_channels;
  bfloat16 *conv5_bn_mean = conv5_bn_bias + out_channels;
  bfloat16 *conv5_bn_var = conv5_bn_mean + out_channels;

  //===========================================================================
  // Stage 1: Conv1 (1×1) + BN + SiLU → features[0]
  //===========================================================================
  conv1x1_bn_silu(input, conv1_weights, conv1_bn_weight, conv1_bn_bias,
                  conv1_bn_mean, conv1_bn_var, conv1_output,
                  height, width, in_channels, neck_channels);

  //===========================================================================
  // Stage 2: MaxPool(5×5, s=1, p=2) on conv1_output → features[1]
  //===========================================================================
  maxpool2d(conv1_output, pool1_output,
            height, width, neck_channels,
            kernel_size, stride, padding);

  //===========================================================================
  // Stage 3: MaxPool(5×5, s=1, p=2) on pool1_output → features[2]
  //===========================================================================
  maxpool2d(pool1_output, pool2_output,
            height, width, neck_channels,
            kernel_size, stride, padding);

  //===========================================================================
  // Stage 4: MaxPool(5×5, s=1, p=2) on pool2_output → features[3]
  //===========================================================================
  maxpool2d(pool2_output, pool3_output,
            height, width, neck_channels,
            kernel_size, stride, padding);

  //===========================================================================
  // Stage 5: 4-way Concatenation [f0, f1, f2, f3]
  //===========================================================================
  concat_4way_channels(conv1_output, pool1_output, pool2_output, pool3_output,
                       concat_buffer,
                       height, width, neck_channels, neck_channels,
                       neck_channels, neck_channels);

  //===========================================================================
  // Stage 6: Conv5 (1×1) + BN + SiLU
  //===========================================================================
  conv1x1_bn_silu(concat_buffer, conv5_weights, conv5_bn_weight, conv5_bn_bias,
                  conv5_bn_mean, conv5_bn_var, output,
                  height, width, concat_channels, out_channels);

  event1();
}

extern "C" {

void sppelan_bf16(bfloat16 *input,
                  bfloat16 *weights_and_bn,
                  bfloat16 *output,
                  bfloat16 *conv1_output,
                  bfloat16 *pool1_output,
                  bfloat16 *pool2_output,
                  bfloat16 *pool3_output,
                  bfloat16 *concat_buffer,
                  int32_t height,
                  int32_t width,
                  int32_t in_channels,
                  int32_t out_channels,
                  int32_t neck_channels,
                  int32_t kernel_size,
                  int32_t stride,
                  int32_t padding) {
  sppelan_bf16_scalar(input, weights_and_bn, output,
                      conv1_output, pool1_output, pool2_output, pool3_output,
                      concat_buffer,
                      height, width, in_channels, out_channels, neck_channels,
                      kernel_size, stride, padding);
}

} // extern "C"
