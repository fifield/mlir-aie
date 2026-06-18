//===- kernel.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

extern "C" {

void systolic_kernel(int32_t *buf, int32_t size, int32_t id, int32_t is_top,
                     int32_t is_bottom, int32_t phase) {
  if (phase == 0) {
    // Phase 0: Initialization phase - propagate core IDs through cascade
    // Each core adds its ID to the cascade value when it reaches its index
    // position.
    for (int i = 0; i < size; i++) {
      // Get value from cascade (0 if top row)
      int32_t cascade_val = 0;
      if (!is_top) {
        v32int16 v32 = get_scd_v32int16();
        cascade_val = ext_elem(v32, 0);
      }

      // Add cascade value to buffer at current index
      if (id == i)
        cascade_val = cascade_val + id;
      buf[i] = cascade_val;

      // Put to cascade (if not bottom row)
      if (!is_bottom) {
        v32int16 v32;
        v32 = upd_elem(v32, 0, (short)(cascade_val));
        put_mcd(v32);
      }
    }
  }

  else if (phase == 1) {
    // Phase 1: Accumulation phase - compute running sum of buffer and cascade
    // values
    uint32_t running = 0;
    for (int i = 0; i < size; i++) {
      int32_t cascade_val = 0;
      int32_t buf_val = buf[i];
      // Get value from cascade (0 if top row)
      if (!is_top) {
        v32int16 v32 = get_scd_v32int16();
        cascade_val = ext_elem(v32, 0);
      }

      // Add cascade value to buffer at current index
      running = running + buf_val + cascade_val;
      buf[i] = running;

      // Put to cascade (if not bottom row)
      if (!is_bottom) {
        v32int16 v32;
        v32 = upd_elem(v32, 0, (short)(running));
        put_mcd(v32);
      }
    }
  }
}
} // extern "C"
