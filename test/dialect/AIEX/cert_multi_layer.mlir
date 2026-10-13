//===- cert_multi_layer.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

module {
  aie.device(npu3) {
    // Data declarations (uC DMA chains and BD data)
    memref.global "private" constant @shim00_bd0 : memref<9xi32> = dense<[1, 0, 0, 0, 0, 0, 0, 0x02000000, 0]>
    memref.global "private" constant @mem11_bd0 : memref<8xi32> = dense<[1, 0x00020000, 0, 0, 0, 0, 0, 0x80000000]>
    
    // CHECK: aiex.cert.uc_dma_chain @uc_dma_bd0
    aiex.cert.uc_dma_chain @uc_dma_bd0 {
      aiex.cert.uc_dma_bd @shim00_bd0, 0x00009000, 9, true
      aiex.cert.uc_dma_bd @mem11_bd0, 0x021A0000, 8, false
    }

    // Main control code with pages
    // CHECK: aiex.cert.attach_to_group(0)
    aiex.cert.attach_to_group(0) {
    
      // Page 0: LOAD_PDI job
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(0)
        aiex.cert.job(0) {
          // CHECK: aiex.cert.load_pdi(0, @pdi0)
          aiex.cert.load_pdi(0, @pdi0)
        }
      }
      
      // Page 1: Main job
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(1)
        aiex.cert.job(1) {
          aiex.cert.apply_offset_57(@shim00_bd0, 1, 0xFFFF)
          aiex.cert.uc_dma_write_des_sync(@uc_dma_bd0)
          aiex.cert.write32(0x009314, 0x80000000)  // enqueue bd0 0_0 to mm2s0
          aiex.cert.write32(0x21A0604, 0x80000000) // enqueue bd0 1_1 to s2mm0
          aiex.cert.wait_tcts(0, 0, 1)  // TILE_0_0, SHIM_MM2S_0
          aiex.cert.wait_tcts(1, 1, 1)  // TILE_1_1, MEM_S2MM_0
        }
      }
      
      // Page 2: Preemption point
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(2)
        aiex.cert.job(2) {
          // CHECK: aiex.cert.preempt(0, @save0, @restore0)
          aiex.cert.preempt(0, @save0, @restore0)
        }
      }
      
      // Page 3: Post-barrier work
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(3)
        aiex.cert.job(3) {
          aiex.cert.remote_barrier(rb3, 0x3)
          aiex.cert.apply_offset_57(@shim00_bd0, 1, 0xFFFF)
          aiex.cert.uc_dma_write_des_sync(@uc_dma_bd0)
        }
      }
    }

    // Named control code sections (correspond to label: ... .endl label in asm)
    
    // PDI section (pdi0: .include multi_pdi0.asm .endl pdi0)
    // CHECK: aiex.cert.section @pdi0
    aiex.cert.section @pdi0 {
      // PDI can contain multiple pages
      aiex.cert.page {
        aiex.cert.job(30) {
          aiex.cert.uc_dma_write_des_sync(@uc_dma_bd0)
        }
      }
    }

    // Save section for preemption (save0: .include multi_save0.asm .endl save0)
    // CHECK: aiex.cert.section @save0
    aiex.cert.section @save0 {
      aiex.cert.page {
        aiex.cert.job(40) {
          aiex.cert.write32(0x21A0000, 0x0)  // Save memtile state
        }
      }
    }

    // Restore section for preemption (restore0: .include multi_restore0.asm .endl restore0)
    // CHECK: aiex.cert.section @restore0
    aiex.cert.section @restore0 {
      aiex.cert.page {
        aiex.cert.job(41) {
          aiex.cert.write32(0x21A0000, 0x1)  // Restore memtile state
        }
      }
    }
  }
}
