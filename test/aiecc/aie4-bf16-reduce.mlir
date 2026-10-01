// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
// RUN: FileCheck %s --check-prefix=ASM --input-file=%t.dis
// RUN: FileCheck %s --check-prefix=NOCALL --input-file=%t.dis

// bf16 maximum/minimum reductions compile to the native widening, the f32
// reduce, and a scalar narrowing. Both reductions read one widened vector.

// IR-LABEL: define void @core_0_2()
// IR: %[[W0:[0-9]+]] = {{.*}}call <32 x float> @llvm.aie4.v32bf16.to.v32accfloat(
// IR: %[[M:[0-9]+]] = {{.*}}call float @llvm.aie4.max.reduce.FP1024(<32 x float> %[[W0]]
// IR: fptrunc float %[[M]] to bfloat
// IR: %[[N:[0-9]+]] = {{.*}}call float @llvm.aie4.min.reduce.FP1024(<32 x float> %[[W0]]
// IR: fptrunc float %[[N]] to bfloat
// IR-NOT: @llvm.vector.reduce

// ASM-DAG: vmaxr.f
// ASM-DAG: vminr.f

// NOCALL-NOT: R_AIE{{.*(llvm|reduce)}}

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<32xbf16>
    %out = aie.buffer(%t) {sym_name = "out"} : memref<2xbf16>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %v = vector.load %a[%c0] : memref<32xbf16>, vector<32xbf16>
      %mx = vector.reduction <maximumf>, %v : vector<32xbf16> into bf16
      memref.store %mx, %out[%c0] : memref<2xbf16>
      %mn = vector.reduction <minimumf>, %v : vector<32xbf16> into bf16
      memref.store %mn, %out[%c1] : memref<2xbf16>
      aie.end
    }
  }
}
