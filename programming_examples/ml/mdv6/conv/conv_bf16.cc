//===- conv_bf16.cc -----------------------------------------------*- C++ -*-===//
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
// Conv2D 3x3 stride 1 - bfloat16
// Simple implementation for MDV6 Conv layer
// Input: (H, W, C_in) in bfloat16
// Weights: (C_out, C_in, 3, 3) in bfloat16
// Output: (H_out, W_out, C_out) in bfloat16
//
// This is a basic implementation that processes one output pixel at a time.
// For better performance, we can vectorize and tile later.
//*****************************************************************************
void conv3x3_bf16_scalar(bfloat16 *input, bfloat16 *weights, bfloat16 *output,
                         const int32_t input_height, const int32_t input_width,
                         const int32_t input_channels, const int32_t output_channels,
                         const int32_t stride, const int32_t padding) {
  event0();

  const int output_height = (input_height + 2 * padding - 3) / stride + 1;
  const int output_width = (input_width + 2 * padding - 3) / stride + 1;

  // For each output channel
  for (int oc = 0; oc < output_channels; oc++) {
    // For each output position
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        float sum = 0.0f;
        
        // Convolve with 3x3 kernel
        for (int ic = 0; ic < input_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = oh * stride + kh - padding;
              int iw = ow * stride + kw - padding;
              
              // Check bounds (zero padding)
              if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = (ih * input_width + iw) * input_channels + ic;
                int weight_idx = ((oc * input_channels + ic) * 3 + kh) * 3 + kw;
                
                float in_val = (float)input[input_idx];
                float w_val = (float)weights[weight_idx];
                sum += in_val * w_val;
              }
            }
          }
        }
        
        int output_idx = (oh * output_width + ow) * output_channels + oc;
        output[output_idx] = (bfloat16)sum;
      }
    }
  }

  event1();
}

//*****************************************************************************
// Conv2D 1x1 stride 1 - bfloat16 (pointwise convolution)
// Used in many MDV6 layers
// This is essentially a matrix multiplication per spatial location
//*****************************************************************************
void conv1x1_bf16_scalar(bfloat16 *input, bfloat16 *weights, bfloat16 *output,
                         const int32_t input_height, const int32_t input_width,
                         const int32_t input_channels, const int32_t output_channels) {
  event0();

  // For each spatial location
  for (int h = 0; h < input_height; h++) {
    for (int w = 0; w < input_width; w++) {
      // For each output channel
      for (int oc = 0; oc < output_channels; oc++) {
        float sum = 0.0f;
        
        // Dot product over input channels
        for (int ic = 0; ic < input_channels; ic++) {
          int input_idx = (h * input_width + w) * input_channels + ic;
          int weight_idx = oc * input_channels + ic;
          
          float in_val = (float)input[input_idx];
          float w_val = (float)weights[weight_idx];
          sum += in_val * w_val;
        }
        
        int output_idx = (h * input_width + w) * output_channels + oc;
        output[output_idx] = (bfloat16)sum;
      }
    }
  }

  event1();
}

//*****************************************************************************
// Fused Conv + BatchNorm + SiLU activation
// This combines the Conv operation with BatchNorm and SiLU to reduce memory traffic
//*****************************************************************************
void conv3x3_bn_silu_bf16(bfloat16 *input, bfloat16 *weights, 
                          bfloat16 *bn_weight, bfloat16 *bn_bias,
                          bfloat16 *output,
                          const int32_t input_height, const int32_t input_width,
                          const int32_t input_channels, const int32_t output_channels,
                          const int32_t stride, const int32_t padding) {
  event0();

  const int output_height = (input_height + 2 * padding - 3) / stride + 1;
  const int output_width = (input_width + 2 * padding - 3) / stride + 1;

  // For each output channel
  for (int oc = 0; oc < output_channels; oc++) {
    // For each output position
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        float sum = 0.0f;
        
        // Convolve with 3x3 kernel
        for (int ic = 0; ic < input_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = oh * stride + kh - padding;
              int iw = ow * stride + kw - padding;
              
              if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = (ih * input_width + iw) * input_channels + ic;
                int weight_idx = ((oc * input_channels + ic) * 3 + kh) * 3 + kw;
                
                float in_val = (float)input[input_idx];
                float w_val = (float)weights[weight_idx];
                sum += in_val * w_val;
              }
            }
          }
        }
        
        // Apply BatchNorm: y = weight * x + bias
        float bn_w = (float)bn_weight[oc];
        float bn_b = (float)bn_bias[oc];
        float bn_out = bn_w * sum + bn_b;
        
        // Apply SiLU: x * sigmoid(x) ≈ x * (0.5 * tanh(0.5 * x) + 0.5)
        float half_x = 0.5f * bn_out;
        float tanh_val = tanh((double)half_x);  // Use standard tanh
        float sigmoid_approx = 0.5f * (tanh_val + 1.0f);
        float silu_out = bn_out * sigmoid_approx;
        
        int output_idx = (oh * output_width + ow) * output_channels + oc;
        output[output_idx] = (bfloat16)silu_out;
      }
    }
  }

  event1();
}

//*****************************************************************************
// Conv3x3 Tiled for Conv0 - bfloat16
// Processes a spatial tile (tile_h x tile_w) with output channel blocking
// Input: (tile_h+2*padding, tile_w+2*padding, input_channels) - includes halo
// Weights: (output_channels_block, input_channels, 3, 3)
// Output: (tile_h, tile_w, output_channels_block)
//*****************************************************************************
void conv3x3_tile_blocked_bf16(bfloat16 *input_patch, bfloat16 *weights,
                               bfloat16 *output_tile,
                               const int32_t tile_height, const int32_t tile_width,
                               const int32_t input_channels,
                               const int32_t output_channels_block,
                               const int32_t stride, const int32_t padding) {
  event0();

  const int patch_height = tile_height + 2 * padding;
  const int patch_width = tile_width + 2 * padding;

  // For each output channel in the block
  for (int oc = 0; oc < output_channels_block; oc++) {
    // For each output position in tile
    for (int oh = 0; oh < tile_height; oh++) {
      for (int ow = 0; ow < tile_width; ow++) {
        float sum = 0.0f;
        
        // Convolve with 3x3 kernel
        for (int ic = 0; ic < input_channels; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              // Patch already includes padding border, so add padding offset
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              
              int input_idx = (ih * patch_width + iw) * input_channels + ic;
              int weight_idx = ((oc * input_channels + ic) * 3 + kh) * 3 + kw;
              
              float in_val = (float)input_patch[input_idx];
              float w_val = (float)weights[weight_idx];
              sum += in_val * w_val;
            }
          }
        }
        
        int output_idx = (oh * tile_width + ow) * output_channels_block + oc;
        output_tile[output_idx] = (bfloat16)sum;
      }
    }
  }

  event1();
}

//*****************************************************************************
// Conv3x3 Partial Accumulation for Conv1 - float32 accumulation
// Processes one input channel block and accumulates to float32 buffer
// Input: (tile_h+2, tile_w+2, input_channels_block)
// Weights: (output_channels_block, input_channels_block, 3, 3)
// Accum: (tile_h, tile_w, output_channels_block) - float32, updated in place
//*****************************************************************************
void conv3x3_partial_accum_bf16(bfloat16 *input_patch, bfloat16 *weights,
                                float *accum_out,
                                const int32_t tile_height, const int32_t tile_width,
                                const int32_t input_channels_block,
                                const int32_t output_channels_block,
                                const int32_t stride, const int32_t padding) {
  event0();

  const int patch_height = tile_height + 2 * padding;
  const int patch_width = tile_width + 2 * padding;

  // For each output channel in the block
  for (int oc = 0; oc < output_channels_block; oc++) {
    // For each output position in tile
    for (int oh = 0; oh < tile_height; oh++) {
      for (int ow = 0; ow < tile_width; ow++) {
        float sum = 0.0f;
        
        // Accumulate over this input channel block
        for (int ic = 0; ic < input_channels_block; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              
              int input_idx = (ih * patch_width + iw) * input_channels_block + ic;
              int weight_idx = ((oc * input_channels_block + ic) * 3 + kh) * 3 + kw;
              
              float in_val = (float)input_patch[input_idx];
              float w_val = (float)weights[weight_idx];
              sum += in_val * w_val;
            }
          }
        }
        
        // Accumulate to existing buffer (allows multiple input channel blocks)
        int output_idx = (oh * tile_width + ow) * output_channels_block + oc;
        accum_out[output_idx] += sum;
      }
    }
  }

  event1();
}

//*****************************************************************************
// Convert float32 accumulation buffer to bfloat16 output
//*****************************************************************************
void accum_to_bf16(float *accum, bfloat16 *output, const int32_t size) {
  event0();
  
  for (int i = 0; i < size; i++) {
    output[i] = (bfloat16)accum[i];
  }
  
  event1();
}

extern "C" {

void conv3x3_bf16(bfloat16 *input, bfloat16 *weights, bfloat16 *output,
                  int32_t input_height, int32_t input_width,
                  int32_t input_channels, int32_t output_channels,
                  int32_t stride, int32_t padding) {
  conv3x3_bf16_scalar(input, weights, output, input_height, input_width,
                      input_channels, output_channels, stride, padding);
}

void conv1x1_bf16(bfloat16 *input, bfloat16 *weights, bfloat16 *output,
                  int32_t input_height, int32_t input_width,
                  int32_t input_channels, int32_t output_channels) {
  conv1x1_bf16_scalar(input, weights, output, input_height, input_width,
                      input_channels, output_channels);
}

void conv3x3_fused_bf16(bfloat16 *input, bfloat16 *weights,
                        bfloat16 *bn_weight, bfloat16 *bn_bias,
                        bfloat16 *output,
                        int32_t input_height, int32_t input_width,
                        int32_t input_channels, int32_t output_channels,
                        int32_t stride, int32_t padding) {
  conv3x3_bn_silu_bf16(input, weights, bn_weight, bn_bias, output,
                       input_height, input_width, input_channels, output_channels,
                       stride, padding);
}

void conv3x3_tiled_bf16(bfloat16 *input_patch, bfloat16 *weights,
                        bfloat16 *output_tile,
                        int32_t tile_height, int32_t tile_width,
                        int32_t input_channels, int32_t output_channels_block,
                        int32_t stride, int32_t padding) {
  conv3x3_tile_blocked_bf16(input_patch, weights, output_tile,
                            tile_height, tile_width, input_channels,
                            output_channels_block, stride, padding);
}

void conv3x3_partial_bf16(bfloat16 *input_patch, bfloat16 *weights,
                          float *accum_out,
                          int32_t tile_height, int32_t tile_width,
                          int32_t input_channels_block,
                          int32_t output_channels_block,
                          int32_t stride, int32_t padding) {
  conv3x3_partial_accum_bf16(input_patch, weights, accum_out,
                             tile_height, tile_width, input_channels_block,
                             output_channels_block, stride, padding);
}

void convert_accum_bf16(float *accum, bfloat16 *output, int32_t size) {
  accum_to_bf16(accum, output, size);
}

} // extern "C"
