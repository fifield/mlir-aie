// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-assign-lock-ids --aie-assign-bd-ids --aie-materialize-runtime-sequences --aie-lower-dma-channel-reset --aie-dma-to-npu --aie-lower-set-lock --aie-npu-to-cert %s | FileCheck %s --implicit-check-not=aiex.npu --implicit-check-not=aie.objectfifo_rearm_binding

// Re-arm both the memtile producer and compute-tile consumer. Every channel
// must be reset before locks are initialized, and both queues must be pushed
// only after the locks are ready. This also exercises AIE4 physical row mapping.
module {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @weights (%mem, {%core}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      // CHECK: aiex.cert.job
      // CHECK-NEXT: aiex.cert.maskwrite32(1089120, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089120, 2, 0)
      // CHECK-NEXT: aiex.cert.maskwrite32(6999552, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(6999552, 2, 0)
      // CHECK-NEXT: aiex.cert.write32(1114112, 1)
      // CHECK-NEXT: aiex.cert.write32(1114128, 0)
      // CHECK-NEXT: aiex.cert.write32(7004160, 1)
      // CHECK-NEXT: aiex.cert.write32(7004176, 0)
      // CHECK-NEXT: aiex.cert.write32(1089124, 0)
      // CHECK-NEXT: aiex.cert.write32(6999556, 0)
      aiex.dma_channel_reset_for(@weights)
    }
  }
}
