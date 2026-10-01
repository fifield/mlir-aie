// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
// RUN: FileCheck %s --check-prefix=ASM --input-file=%t.dis

// Compile the masked form emitted by linalg vectorization for a 1000-element
// add. Full chunks use vector accesses; the 8-element tail uses scalar accesses.
// No unsupported llvm.masked.* intrinsic should remain.

// IR-LABEL: define void @core_0_2()
// IR-DAG: load <32 x float>, ptr %{{.*}}, align 128
// IR-DAG: load float, ptr %{{.*}}, align 4
// IR-DAG: fadd <32 x float>
// IR-DAG: store <32 x float> %{{.*}}, ptr %{{.*}}, align 128
// IR-DAG: store float %{{.*}}, ptr %{{.*}}, align 4
// IR-NOT: llvm.masked

// ASM-DAG: vadd.f{{[[:space:]]+}}y{{[0-9]+}}
// ASM-DAG: vst{{[[:space:]]+}}y{{[0-9]+}}, [p{{[0-9]+}}, dj{{[0-9]+}}]
// ASM-DAG: {{[[:space:]]}}st{{[[:space:]]+}}r{{[0-9]+}}, [p{{[0-9]+}}, dj{{[0-9]+}}]
// ASM-DAG: {{[[:space:]]}}lda{{[[:space:]]+}}r{{[0-9]+}}, [p{{[0-9]+}}, dj{{[0-9]+}}]
// ASM-NOT: R_AIE{{.*}}{{llvm|masked}}

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<1000xf32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<1000xf32>
    %c = aie.buffer(%t) {sym_name = "c"} : memref<1000xf32>
    %core = aie.core(%t) {
      %0 = ub.poison : f32
      %c0 = arith.constant 0 : index
      %c1000 = arith.constant 1000 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0 to %c1000 step %c32 {
        %n = affine.min affine_map<(d0) -> (-d0 + 1000, 32)>(%i)
        %m = vector.create_mask %n : vector<32xi1>
        %x = vector.mask %m { vector.transfer_read %a[%i], %0 {in_bounds = [true]} : memref<1000xf32>, vector<32xf32> } : vector<32xi1> -> vector<32xf32>
        %y = vector.mask %m { vector.transfer_read %b[%i], %0 {in_bounds = [true]} : memref<1000xf32>, vector<32xf32> } : vector<32xi1> -> vector<32xf32>
        %s = arith.addf %x, %y : vector<32xf32>
        vector.mask %m { vector.transfer_write %s, %c[%i] {in_bounds = [true]} : vector<32xf32>, memref<1000xf32> } : vector<32xi1>
      }
      aie.end
    }
  }
}
