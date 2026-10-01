// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// D19 entry writes reach the core ELF. The core function itself writes
// crVecRnd and crRnd (movx ..., #0xc = RNE); Peano's MIR gives every reader
// an implicit use of the register, so the scheduler keeps each write in time
// for its readers. Hardware proof: npu-xrt/npu3_core_entry_modes.
//
// The core also exercises the generic crRnd readers that need RNE: a vector
// f32 add, a scalar bf16 add, and an f32 -> bf16 truncation spelled with
// to_nearest_even. That attribute is dropped once crRnd = 12 is guaranteed;
// kept, it lowers to llvm.experimental.constrained.fptrunc and llc aborts.
//
// REQUIRES: peano
// RUN: %aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: llvm-objdump -dr --no-show-raw-insn %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s
// RUN: %aiecc --unified --get-core-elfs --tmpdir %t.uprj --output-dir %t.uout %s
// RUN: llvm-objdump -dr --no-show-raw-insn %t.uout/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// The core function is aliased to main in the ELF.
// CHECK-LABEL: <main>:
// CHECK-DAG:   movx crvecrnd, #0xc
// CHECK-DAG:   movx crrnd, #0xc
// CHECK-DAG:   vadd.f
// CHECK-DAG:   vconv.bf16.fp32

module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_a = aie.buffer(%tile_0_2) {sym_name = "buf_a"} : memref<32xf32>
    %buf_c = aie.buffer(%tile_0_2) {sym_name = "buf_c"} : memref<32xf32>
    %buf_h = aie.buffer(%tile_0_2) {sym_name = "buf_h"} : memref<32xbf16>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %pad = arith.constant 0.0 : f32
      %a = vector.transfer_read %buf_a[%c0], %pad {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      %s = arith.addf %a, %a : vector<32xf32>
      vector.transfer_write %s, %buf_c[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
      %x = memref.load %buf_c[%c0] : memref<32xf32>
      %h = arith.truncf %x to_nearest_even : f32 to bf16
      memref.store %h, %buf_h[%c0] : memref<32xbf16>
      %p = memref.load %buf_h[%c0] : memref<32xbf16>
      %q = memref.load %buf_h[%c1] : memref<32xbf16>
      %r = arith.addf %p, %q : bf16
      memref.store %r, %buf_h[%c2] : memref<32xbf16>
      aie.end
    }
  }
}
