//===- cert_verify_ucdma_handle.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// G-uchandle: a cert.wait_uc_dma handle must be a job-private local register
// r0..r7 whose reaching definition in the same job is a cert.uc_dma_write_des.
// This is a reaching-definition check, not merely "a producer exists
// somewhere earlier": an intervening write clobbers the handle. The op
// itself rejects r8..r23 as a handle, since the firmware accepts only
// r0..r7.

// RUN: aie-opt %s --aie-cert-verify --split-input-file --verify-diagnostics

// Positive: straight-line producer/consumer.
aie.device(npu2) {
  memref.global "private" constant @d : memref<8xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @d, 0, 8, false
  }
  aiex.cert.job(0) {
    aiex.cert.uc_dma_write_des(3, @chain)
    aiex.cert.wait_uc_dma(3)
  }
}

// -----

// Positive: a write to a DIFFERENT register in between does not clobber the
// handle -- clobber tracking is per-register.
aie.device(npu2) {
  memref.global "private" constant @d : memref<8xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @d, 0, 8, false
  }
  aiex.cert.job(0) {
    aiex.cert.uc_dma_write_des(3, @chain)
    aiex.cert.read32(4, 0x02100000)
    aiex.cert.wait_uc_dma(3)
  }
}

// -----

// Positive: cert.write32_d with address_is_reg only READS the
// register named by `address` ("The write address is in this register",
// isa-spec.yaml); it does not define/clobber it. No diagnostic.
aie.device(npu2) {
  memref.global "private" constant @d : memref<8xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @d, 0, 8, false
  }
  aiex.cert.job(0) {
    aiex.cert.uc_dma_write_des(3, @chain)
    aiex.cert.write32_d(3, 5) {address_is_reg}
    aiex.cert.wait_uc_dma(3)
  }
}

// -----

// Negative: no producer at all.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{cert.wait_uc_dma waits on r3, which no earlier cert.uc_dma_write_des in this job writes}}
    aiex.cert.wait_uc_dma(3)
  }
}

// -----

// Negative: the handle is clobbered by an intervening cert.read32_d, whose
// second operand is its destination register.
aie.device(npu2) {
  memref.global "private" constant @d : memref<8xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @d, 0, 8, false
  }
  aiex.cert.job(0) {
    aiex.cert.uc_dma_write_des(3, @chain)
    aiex.cert.read32_d(4, 3)
    // expected-error@+1 {{cert.wait_uc_dma waits on r3, but its last write is a 'aiex.cert.read32_d', which replaced the handle}}
    aiex.cert.wait_uc_dma(3)
  }
}

// -----

// Negative: the handle is clobbered by an intervening cert.read32.
aie.device(npu2) {
  memref.global "private" constant @d : memref<8xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @d, 0, 8, false
  }
  aiex.cert.job(0) {
    aiex.cert.uc_dma_write_des(3, @chain)
    aiex.cert.read32(3, 0x02100000)
    // expected-error@+1 {{cert.wait_uc_dma waits on r3, but its last write is a 'aiex.cert.read32', which replaced the handle}}
    aiex.cert.wait_uc_dma(3)
  }
}

// -----

// Negative: cross-job. The wait_handle rule is per-job, so a producer on a
// different job does not reach.
aie.device(npu2) {
  memref.global "private" constant @d : memref<8xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @d, 0, 8, false
  }
  aiex.cert.job(0) {
    aiex.cert.uc_dma_write_des(3, @chain)
  }
  aiex.cert.job(1) {
    // expected-error@+1 {{cert.wait_uc_dma waits on r3, which no earlier cert.uc_dma_write_des in this job writes}}
    aiex.cert.wait_uc_dma(3)
  }
}
