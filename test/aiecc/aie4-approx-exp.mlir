// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
// RUN: FileCheck %s --check-prefix=ASM --input-file=%t.dis
// RUN: FileCheck %s --check-prefix=NOCALL --input-file=%t.dis

// afn exp (f32 and bf16 results, and a split 64-lane vector) compiles to an
// f32 multiply by log2(e), the VEXP2 bias add and native VEXP2.

// IR-LABEL: define void @core_0_2()
// IR: %[[T0:[0-9]+]] = {{.*}}call <32 x float> @llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf(<32 x float> %{{[0-9]+}}, <32 x float> splat (float 0x3FF7154760000000), i32 27)
// IR: %[[B0:[0-9]+]] = fadd <32 x float> %[[T0]], splat (float 0x408BF80080000000)
// IR: call <32 x float> @llvm.aie4.exp2.bf20(<32 x float> %[[B0]])
// IR: %[[W:[0-9]+]] = {{.*}}call <32 x float> @llvm.aie4.v32bf16.to.v32accfloat(
// IR: %[[T1:[0-9]+]] = {{.*}}call <32 x float> @llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf(<32 x float> %[[W]], <32 x float> splat (float 0x3FF7154760000000), i32 27)
// IR: %[[B1:[0-9]+]] = fadd <32 x float> %[[T1]], splat (float 0x408BF80200000000)
// IR: call <32 x bfloat> @llvm.aie4.exp2(<32 x float> %[[B1]])
// IR-NOT: @llvm.exp

// Two VEXP2.fp32 for the 64-lane source plus one for the 32-lane source.
// RUN: grep -c "vexp2.fp32" %t.dis | FileCheck %s --check-prefix=NFP32
// RUN: grep -c "vexp2.bf16" %t.dis | FileCheck %s --check-prefix=NBF16
// NFP32: {{^}}3{{$}}
// NBF16: {{^}}1{{$}}
// ASM-DAG: vmul.f
// ASM-DAG: vadd.f

// NOCALL-NOT: R_AIE{{.*(exp|llvm)}}

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %xf = aie.buffer(%t) {sym_name = "xf"} : memref<32xf32>
    %xb = aie.buffer(%t) {sym_name = "xb"} : memref<32xbf16>
    %xw = aie.buffer(%t) {sym_name = "xw"} : memref<64xf32>
    %e_f32 = aie.buffer(%t) {sym_name = "e_f32"} : memref<32xf32>
    %e_bf16 = aie.buffer(%t) {sym_name = "e_bf16"} : memref<32xbf16>
    %e_wide = aie.buffer(%t) {sym_name = "e_wide"} : memref<64xf32>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %f = vector.load %xf[%c0] : memref<32xf32>, vector<32xf32>
      %b = vector.load %xb[%c0] : memref<32xbf16>, vector<32xbf16>
      %w = vector.load %xw[%c0] : memref<64xf32>, vector<64xf32>
      %e0 = math.exp %f fastmath<afn> : vector<32xf32>
      vector.store %e0, %e_f32[%c0] : memref<32xf32>, vector<32xf32>
      %e1 = math.exp %b fastmath<afn> : vector<32xbf16>
      vector.store %e1, %e_bf16[%c0] : memref<32xbf16>, vector<32xbf16>
      %e2 = math.exp %w fastmath<afn> : vector<64xf32>
      vector.store %e2, %e_wide[%c0] : memref<64xf32>, vector<64xf32>
      aie.end
    }
  }
}
