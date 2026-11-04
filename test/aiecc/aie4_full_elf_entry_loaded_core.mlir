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
// RUN: diff aie4_full_elf_entry_loaded_core_prj/top/cfga__site0.asm aie4_full_elf_entry_loaded_core_prj/main/cfga__site0.asm
// RUN: FileCheck %s --check-prefix=JSON --input-file=aie4_full_elf_entry_loaded_core_prj/top/full_elf_config.json

// The CERT firmware skips a LOAD_PDI whose id matches the last one it loaded,
// across every ELF on a hardware context, so the id is derived from the
// section's content: the two compilations name their
// entry differently but load the same @cfga configuration, and so the same id.
// RUN: grep LOAD_PDI aie4_full_elf_entry_loaded_core_prj/top/full_elf_cert.mlir.asm > aie4_full_elf_entry_loaded_core_prj/top.load_pdi
// RUN: grep LOAD_PDI aie4_full_elf_entry_loaded_core_prj/main/full_elf_cert.mlir.asm > aie4_full_elf_entry_loaded_core_prj/main.load_pdi
// RUN: FileCheck %s --check-prefix=PDI --input-file=aie4_full_elf_entry_loaded_core_prj/top.load_pdi
// RUN: diff aie4_full_elf_entry_loaded_core_prj/top.load_pdi aie4_full_elf_entry_loaded_core_prj/main.load_pdi

// A configuration that differs (here, in one lock's initial value) gets a
// different id even under the same device names; otherwise loading one ELF's
// @cfg after the other's would be skipped. These designs have no cores.
// RUN: sed -n 's|^// NOCORE: ||p' %s | sed 's/LOCKINIT/1/' > aie4_full_elf_entry_loaded_core_prj/nocore1.mlir
// RUN: sed -n 's|^// NOCORE: ||p' %s | sed 's/LOCKINIT/2/' > aie4_full_elf_entry_loaded_core_prj/nocore2.mlir
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_entry_loaded_core_prj/nocore1 --output-dir=aie4_full_elf_entry_loaded_core_prj/nocore1 aie4_full_elf_entry_loaded_core_prj/nocore1.mlir
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_entry_loaded_core_prj/nocore2 --output-dir=aie4_full_elf_entry_loaded_core_prj/nocore2 aie4_full_elf_entry_loaded_core_prj/nocore2.mlir
// RUN: grep LOAD_PDI aie4_full_elf_entry_loaded_core_prj/nocore1/full_elf_cert.mlir.asm > aie4_full_elf_entry_loaded_core_prj/nocore1.load_pdi
// RUN: grep LOAD_PDI aie4_full_elf_entry_loaded_core_prj/nocore2/full_elf_cert.mlir.asm > aie4_full_elf_entry_loaded_core_prj/nocore2.load_pdi
// RUN: FileCheck %s --check-prefix=NOCORE-PDI --input-file=aie4_full_elf_entry_loaded_core_prj/nocore1.load_pdi
// RUN: FileCheck %s --check-prefix=NOCORE-PDI --input-file=aie4_full_elf_entry_loaded_core_prj/nocore2.load_pdi
// RUN: not diff aie4_full_elf_entry_loaded_core_prj/nocore1.load_pdi aie4_full_elf_entry_loaded_core_prj/nocore2.load_pdi

// PDI: LOAD_PDI {{ *}}{{[0-9]+}}, @cfga
// NOCORE-PDI: LOAD_PDI {{ *}}{{[0-9]+}}, @cfg

// NOCORE: module {
// NOCORE:   aie.device(npu3) @main {
// NOCORE:     aie.runtime_sequence @entry(%x: memref<16xi32>) {
// NOCORE:       aiex.configure @cfg {
// NOCORE:         aiex.run @s(%x) : (memref<16xi32>)
// NOCORE:       }
// NOCORE:     }
// NOCORE:   }
// NOCORE:   aie.device(npu3) @cfg {
// NOCORE:     %tile = aie.tile(0, 2)
// NOCORE:     %lock = aie.lock(%tile, 0) {init = LOCKINIT : i32}
// NOCORE:     aie.runtime_sequence @s(%p: memref<16xi32>) {
// NOCORE:       %addr = arith.constant 393224 : i32
// NOCORE:       %value = arith.constant 3333 : i32
// NOCORE:       aiex.npu.write32(%addr, %value) : i32, i32
// NOCORE:     }
// NOCORE:   }
// NOCORE: }

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
