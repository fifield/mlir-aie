// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize --cse | aie-translate --mlir-to-llvmir | FileCheck %s
// Repeated writes and unused reads are observable. CSE/DCE must keep them,
// along with each explicit scheduling boundary. Check the actual LLVM ABI.

// CHECK-LABEL: define i32 @aie2_tm(
// CHECK: call void @llvm.aie2.write.tm(i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2.sched.barrier()
// CHECK-NEXT: call i32 @llvm.aie2.read.tm(ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2.sched.barrier()
// CHECK-NEXT: call void @llvm.aie2.write.tm(i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2.sched.barrier()
// CHECK-NEXT: {{.*}}call i32 @llvm.aie2.read.tm(ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2.sched.barrier()
llvm.func @aie2_tm(%value: i32, %ptr: !llvm.ptr) -> i32 {
  xllvm.intr.aie2.write.tm %value, %ptr : !llvm.ptr
  xllvm.intr.aie2.sched.barrier
  %unused = xllvm.intr.aie2.read.tm %ptr : !llvm.ptr
  xllvm.intr.aie2.sched.barrier
  xllvm.intr.aie2.write.tm %value, %ptr : !llvm.ptr
  xllvm.intr.aie2.sched.barrier
  %result = xllvm.intr.aie2.read.tm %ptr : !llvm.ptr
  xllvm.intr.aie2.sched.barrier
  llvm.return %result : i32
}

// CHECK-LABEL: define i32 @aie2p_tm(
// CHECK: call void @llvm.aie2p.write.tm(i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2p.sched.barrier()
// CHECK-NEXT: call i32 @llvm.aie2p.read.tm(ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2p.sched.barrier()
// CHECK-NEXT: call void @llvm.aie2p.write.tm(i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2p.sched.barrier()
// CHECK-NEXT: {{.*}}call i32 @llvm.aie2p.read.tm(ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie2p.sched.barrier()
llvm.func @aie2p_tm(%value: i32, %ptr: !llvm.ptr) -> i32 {
  xllvm.intr.aie2p.write.tm %value, %ptr : !llvm.ptr
  xllvm.intr.aie2p.sched.barrier
  %unused = xllvm.intr.aie2p.read.tm %ptr : !llvm.ptr
  xllvm.intr.aie2p.sched.barrier
  xllvm.intr.aie2p.write.tm %value, %ptr : !llvm.ptr
  xllvm.intr.aie2p.sched.barrier
  %result = xllvm.intr.aie2p.read.tm %ptr : !llvm.ptr
  xllvm.intr.aie2p.sched.barrier
  llvm.return %result : i32
}

// CHECK-LABEL: define i32 @aie4_tm(
// CHECK: call void @llvm.aie4.write.tm(i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie4.sched.barrier()
// CHECK-NEXT: call i32 @llvm.aie4.read.tm(ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie4.sched.barrier()
// CHECK-NEXT: call void @llvm.aie4.write.tm(i32 %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie4.sched.barrier()
// CHECK-NEXT: {{.*}}call i32 @llvm.aie4.read.tm(ptr %{{.*}})
// CHECK-NEXT: call void @llvm.aie4.sched.barrier()
llvm.func @aie4_tm(%value: i32, %ptr: !llvm.ptr) -> i32 {
  xllvm.intr.aie4.write.tm %value, %ptr : !llvm.ptr
  xllvm.intr.aie4.sched.barrier
  %unused = xllvm.intr.aie4.read.tm %ptr : !llvm.ptr
  xllvm.intr.aie4.sched.barrier
  xllvm.intr.aie4.write.tm %value, %ptr : !llvm.ptr
  xllvm.intr.aie4.sched.barrier
  %result = xllvm.intr.aie4.read.tm %ptr : !llvm.ptr
  xllvm.intr.aie4.sched.barrier
  llvm.return %result : i32
}
