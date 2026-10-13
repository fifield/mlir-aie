//===- tile_mapped_intrinsics.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck %s

// Core outlining re-creates the body; ops it creates must be legal, so the
// LLVM and XLLVM ops the Python TM helpers emit need explicit legality.

// CHECK-LABEL: func.func @core_0_2()
// CHECK: llvm.inttoptr
// CHECK-NEXT: xllvm.intr.aie4.sched.barrier
// CHECK-NEXT: xllvm.intr.aie4.read.tm
// CHECK-NEXT: xllvm.intr.aie4.write.tm
// CHECK-NEXT: xllvm.intr.aie4.sched.barrier
module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      %a = llvm.mlir.constant(299104 : i32) : i32
      %p = llvm.inttoptr %a : i32 to !llvm.ptr
      xllvm.intr.aie4.sched.barrier
      %v = xllvm.intr.aie4.read.tm %p : !llvm.ptr
      xllvm.intr.aie4.write.tm %v, %p : !llvm.ptr
      xllvm.intr.aie4.sched.barrier
      aie.end
    }
  }
}
