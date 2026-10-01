// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: FileCheck %s --check-prefix=NOBF --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
// RUN: FileCheck %s --check-prefix=ASM --input-file=%t.dis
// RUN: FileCheck %s --check-prefix=NOCALL --input-file=%t.dis

// bf16 add, sub and fma compile to native conversions around the f32 unit.
// The core sets crRnd to nearest even at entry, since the narrowings read it.

// IR-LABEL: define void @core_0_2()
// IR: call void @llvm.aie4.set.ctrl.reg(i32 46, i32 12)
// IR-COUNT-2: call <32 x float> @llvm.aie4.v32bf16.to.v32accfloat(
// IR: %[[S:[0-9]+]] = fadd <32 x float>
// IR: call <32 x bfloat> @llvm.aie4.v32accfloat.to.v32bfloat16(<32 x float> %[[S]])
// IR: %[[D:[0-9]+]] = fsub <32 x float>
// IR: call <32 x bfloat> @llvm.aie4.v32accfloat.to.v32bfloat16(<32 x float> %[[D]])
// fma: exact product, sum, TwoSum error, round to odd, one narrowing.
// IR: %[[P:[0-9]+]] = {{.*}}call <32 x float> @llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf(
// IR: %[[FS:[0-9]+]] = fadd <32 x float> %[[P]],
// IR-COUNT-4: fsub <32 x float>
// IR: fadd <32 x float>
// IR: or <32 x i32> %{{[0-9]+}}, splat (i32 1)
// IR: call <32 x bfloat> @llvm.aie4.v32accfloat.to.v32bfloat16(

// No bf16 arithmetic, fptrunc or fma is left for Peano to legalize.
// NOBF-NOT: f{{add|sub|mul}} <32 x bfloat>
// NOBF-NOT: fptrunc
// NOBF-NOT: @llvm.fma
// NOBF-NOT: mac.ewise

// ASM-DAG: vadd.f
// ASM-DAG: vsub.f
// ASM-DAG: vmul.f
// ASM-DAG: conv.fp32.bf16
// ASM-DAG: conv.bf16.fp32

// NOCALL-NOT: R_AIE{{.*(llvm|fma|trunc)}}

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<32xbf16>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xbf16>
    %c = aie.buffer(%t) {sym_name = "c"} : memref<32xbf16>
    %s = aie.buffer(%t) {sym_name = "s"} : memref<32xbf16>
    %d = aie.buffer(%t) {sym_name = "d"} : memref<32xbf16>
    %f = aie.buffer(%t) {sym_name = "f"} : memref<32xbf16>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %va = vector.load %a[%c0] : memref<32xbf16>, vector<32xbf16>
      %vb = vector.load %b[%c0] : memref<32xbf16>, vector<32xbf16>
      %vc = vector.load %c[%c0] : memref<32xbf16>, vector<32xbf16>
      %sum = arith.addf %va, %vb : vector<32xbf16>
      vector.store %sum, %s[%c0] : memref<32xbf16>, vector<32xbf16>
      %dif = arith.subf %va, %vb : vector<32xbf16>
      vector.store %dif, %d[%c0] : memref<32xbf16>, vector<32xbf16>
      %fma = vector.fma %va, %vb, %vc : vector<32xbf16>
      vector.store %fma, %f[%c0] : memref<32xbf16>, vector<32xbf16>
      aie.end
    }
  }
}
