// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
// RUN: FileCheck %s --check-prefix=ASM --input-file=%t.dis
// RUN: FileCheck %s --check-prefix=NOCALL --input-file=%t.dis

// afn exp2 (f32 and bf16 results, including a split 64-lane vector) and afn
// tanh (f32 and bf16 results) compile to native VEXP2/VTANH. llc accepts an
// unknown llvm.aie4.* name as an external call (the core link then fails on
// the undefined symbol), so check the mnemonics and relocations.
//
// Wider sources can fail Peano load selection (e.g. vector<128xf32> or
// vector<96xbf16>), so this end-to-end test stops at 64 lanes.

// Check flag-free bias adds: 895 + 2^-12 for f32, vendor 895.001 for bf16.
// IR-LABEL: define void @core_0_2()
// IR: %[[B0:[0-9]+]] = fadd <32 x float> %{{[0-9]+}}, splat (float 0x408BF80080000000)
// IR: call <32 x float> @llvm.aie4.exp2.bf20(<32 x float> %[[B0]])
// IR: %[[B1:[0-9]+]] = fadd <32 x float> %{{[0-9]+}}, splat (float 0x408BF80200000000)
// IR: call <32 x bfloat> @llvm.aie4.exp2(<32 x float> %[[B1]])
// IR-DAG: call <32 x bfloat> @llvm.aie4.tanh(
// IR-DAG: call <32 x bfloat> @llvm.aie4.tanh(
// IR-NOT: @llvm.exp2
// IR-NOT: @llvm.tanh

// Two VEXP2.fp32 for the 64-lane source plus one for the 32-lane source.
// RUN: grep -c "vexp2.fp32" %t.dis | FileCheck %s --check-prefix=NFP32
// RUN: grep -c "vexp2.bf16" %t.dis | FileCheck %s --check-prefix=NBF16
// RUN: grep -c "vtanh" %t.dis | FileCheck %s --check-prefix=NTANH
// NFP32: {{^}}3{{$}}
// NBF16: {{^}}1{{$}}
// NTANH: {{^}}2{{$}}
// ASM-DAG: vexp2.fp32
// ASM-DAG: vexp2.bf16
// ASM-DAG: vtanh
// ASM-DAG: vadd.f

// No relocation names an exp2/tanh routine or an llvm.* intrinsic.
// NOCALL-NOT: R_AIE{{.*(exp|tanh|llvm)}}

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %xf = aie.buffer(%t) {sym_name = "xf"} : memref<32xf32>
    %xb = aie.buffer(%t) {sym_name = "xb"} : memref<32xbf16>
    %xw = aie.buffer(%t) {sym_name = "xw"} : memref<64xf32>
    %e_f32 = aie.buffer(%t) {sym_name = "e_f32"} : memref<32xf32>
    %e_bf16 = aie.buffer(%t) {sym_name = "e_bf16"} : memref<32xbf16>
    %e_wide = aie.buffer(%t) {sym_name = "e_wide"} : memref<64xf32>
    %t_f32 = aie.buffer(%t) {sym_name = "t_f32"} : memref<32xf32>
    %t_bf16 = aie.buffer(%t) {sym_name = "t_bf16"} : memref<32xbf16>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %f = vector.load %xf[%c0] : memref<32xf32>, vector<32xf32>
      %b = vector.load %xb[%c0] : memref<32xbf16>, vector<32xbf16>
      %w = vector.load %xw[%c0] : memref<64xf32>, vector<64xf32>
      %e0 = math.exp2 %f fastmath<afn> : vector<32xf32>
      vector.store %e0, %e_f32[%c0] : memref<32xf32>, vector<32xf32>
      %e1 = math.exp2 %b fastmath<afn> : vector<32xbf16>
      vector.store %e1, %e_bf16[%c0] : memref<32xbf16>, vector<32xbf16>
      %e2 = math.exp2 %w fastmath<afn> : vector<64xf32>
      vector.store %e2, %e_wide[%c0] : memref<64xf32>, vector<64xf32>
      %t0 = math.tanh %f fastmath<afn> : vector<32xf32>
      vector.store %t0, %t_f32[%c0] : memref<32xf32>, vector<32xf32>
      %t1 = math.tanh %b fastmath<afn> : vector<32xbf16>
      vector.store %t1, %t_bf16[%c0] : memref<32xbf16>, vector<32xbf16>
      aie.end
    }
  }
}
