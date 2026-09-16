//===- lower_event.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-standard-lowering --split-input-file %s | FileCheck %s

// CHECK-LABEL: test_aie1
// CHECK: call @llvm.aie.event0()
// CHECK: call @llvm.aie.event1()
module @test_aie1 {
 aie.device(xcvc1902) {
  %tile11 = aie.tile(1, 1)
  %core11 = aie.core(%tile11) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}

// -----

// CHECK-LABEL: test_aie2
// CHECK: call @llvm.aie2.event(%{{.*}})
// CHECK: call @llvm.aie2.event(%{{.*}})
module @test_aie2 {
 aie.device(npu1) {
  %tile02 = aie.tile(0, 2)
  %core02 = aie.core(%tile02) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}

// -----

// CHECK-LABEL: test_aie2p
// CHECK: call @llvm.aie2p.event(%{{.*}})
// CHECK: call @llvm.aie2p.event(%{{.*}})
module @test_aie2p {
 aie.device(npu2) {
  %tile02 = aie.tile(0, 2)
  %core02 = aie.core(%tile02) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}

// -----

// aie.event is not supported on AIE2ps; this must be a diagnostic, not an
// abort. (2, 3) is a core tile on xcve3858, unlike (0, 2).
module @test_aie2ps {
 aie.device(xcve3858) {
  %tile23 = aie.tile(2, 3)
  %core23 = aie.core(%tile23) {
    // expected-error@+2 {{'aie.event' op aie.event is not yet supported on the AIE2PS}}
    // expected-error@+1 {{failed to legalize operation 'aie.event' that was explicitly marked illegal}}
    aie.event(0)
    aie.end
  }
 }
}

// -----

// AIE4 event lowering is not yet implemented; this must be a clear
// named-architecture diagnostic, not a blank "Could not find the intrinsic
// function " message.
module @test_aie4 {
 aie.device(npu3) {
  %tile02 = aie.tile(0, 2)
  %core02 = aie.core(%tile02) {
    // expected-error@+2 {{aie.event is not yet supported on the AIE4 architecture}}
    // expected-error@+1 {{failed to legalize operation 'aie.event' that was explicitly marked illegal}}
    aie.event(0)
    aie.end
  }
 }
}
