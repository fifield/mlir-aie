// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-narrow-core-loops --split-input-file %s | FileCheck %s

// Only the bounded inner loop is narrowed. The outer sentinel loop and the
// index used by memref operations retain their original types.
// CHECK-LABEL: module @aie4
// CHECK:       arith.constant 9223372036854775807 : index
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[LB:.*]] = arith.constant 0 : i32
// CHECK:         %[[UB:.*]] = arith.constant 1024 : i32
// CHECK:         %[[STEP:.*]] = arith.constant 16 : i32
// CHECK:         scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] : i32 {
// CHECK:           arith.index_cast %[[IV]] : i32 to index
// CHECK:       scf.for %{{.*}} = %c0 to %c2147483647 step %c2 {
module @aie4 {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "buf"} : memref<1024xi32>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c1024 = arith.constant 1024 : index
      %cintmax = arith.constant 2147483647 : index
      %cmax = arith.constant 9223372036854775807 : index
      %x = arith.constant 1 : i32
      scf.for %outer = %c0 to %cmax step %c1 {
        scf.for %i = %c0 to %c1024 step %c16 {
          %v = memref.load %buf[%i] : memref<1024xi32>
          memref.store %v, %buf[%i] : memref<1024xi32>
        }
      }
      // 0 to INT32_MAX step 2 increments past INT32_MAX and stays at index.
      scf.for %i = %c0 to %cintmax step %c2 {
        memref.store %x, %buf[%c0] : memref<1024xi32>
      }
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @aie2
// CHECK-NOT:   scf.for {{.*}} : i32
module @aie2 {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %i = %c0 to %c16 step %c1 {
      }
      aie.end
    }
  }
}
