// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano

// --device-name selects the AIE4 full-ELF entry device, but the devices that
// entry loads still run in its program: their cores have to be compiled, or
// the entry's control code configures them without code. Compile the same
// design with the entry named by --device-name and by the "main" default, and
// check the loaded device's configuration section comes out the same.
// The run's sequence name @a must not pull the unrelated device @a into the
// explicitly selected entry's core compilation.

// RUN: rm -rf aie4_full_elf_entry_loaded_core_prj && mkdir -p aie4_full_elf_entry_loaded_core_prj
// RUN: sed 's/@ENTRY/@top/' %s > aie4_full_elf_entry_loaded_core_prj/top.mlir
// RUN: sed 's/@ENTRY/@main/' %s > aie4_full_elf_entry_loaded_core_prj/main.mlir
// RUN: %aiecc --get-full-elf --dump-intermediates --device-name=top --tmpdir=aie4_full_elf_entry_loaded_core_prj/top --output-dir=aie4_full_elf_entry_loaded_core_prj/top aie4_full_elf_entry_loaded_core_prj/top.mlir
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_entry_loaded_core_prj/main --output-dir=aie4_full_elf_entry_loaded_core_prj/main aie4_full_elf_entry_loaded_core_prj/main.mlir
// RUN: FileCheck %s --check-prefix=ELF --input-file=aie4_full_elf_entry_loaded_core_prj/top/physical_with_elfs.mlir
// RUN: diff aie4_full_elf_entry_loaded_core_prj/top/cfga.asm aie4_full_elf_entry_loaded_core_prj/main/cfga.asm
// RUN: FileCheck %s --check-prefix=JSON --input-file=aie4_full_elf_entry_loaded_core_prj/top/full_elf_config.json

// ELF-NOT: elfs_a_core_0_2.elf
// ELF: elf_file = "{{.*}}elfs_cfga_core_0_2.elf"
// ELF-NOT: elfs_a_core_0_2.elf

// JSON: "id": "entry"
// JSON: "name": "top"

module {
  aie.device(npu3) @ENTRY {
    aie.runtime_sequence @entry(%x: memref<16xi32>) {
      aiex.configure @cfga {
        aiex.run @a(%x) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu3) @cfga {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "cbuf"} : memref<16xi32>
    aie.core(%tile) {
      %c7 = arith.constant 7 : i32
      %c0 = arith.constant 0 : index
      memref.store %c7, %buf[%c0] : memref<16xi32>
      aie.end
    }
    aie.runtime_sequence @a(%p: memref<16xi32>) {
      %addr = arith.constant 393224 : i32
      %value = arith.constant 3333 : i32
      aiex.npu.write32(%addr, %value) : i32, i32
    }
  }
  aie.device(npu3) @a {
    %tile = aie.tile(0, 2)
    aie.core(%tile) {
      aie.end
    }
  }
}
