// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A loaded device may have a runtime sequence of its own named `configure`.
// The generated configuration then takes the name `configure0` and is tagged
// aie.generated_configure; that tag, not the name, selects what becomes the
// device's cert section. The section holds the lock initialization (memtile
// (0, 1) lock 0 at 0x00110000, init 3), and the user's `configure` (marker
// 0x5EC0DE01 written to 0x00110010) runs where the entry's sequence calls it.

// RUN: rm -rf aie4_user_configure_sequence_prj && mkdir -p aie4_user_configure_sequence_prj
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_user_configure_sequence_prj --output-dir=aie4_user_configure_sequence_prj %s
// RUN: FileCheck %s --check-prefix=SEC --input-file=aie4_user_configure_sequence_prj/cfg__site0.asm
// RUN: FileCheck %s --check-prefix=ENTRY --input-file=aie4_user_configure_sequence_prj/full_elf_cert.mlir.asm

// SEC-NOT: 0x5ec0de01
// SEC: WRITE_32 0x00110000, 0x00000003
// SEC-NOT: 0x5ec0de01

// ENTRY: LOAD_PDI {{[0-9]+}}, @cfg
// ENTRY: WRITE_32 0x00110010, 0x5ec0de01

module {
  aie.device(npu3_1col) @main {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @seq(%a: memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @configure(%a) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu3_1col) @cfg {
    %m = aie.tile(0, 1)
    %l = aie.lock(%m, 0) {init = 3 : i32}
    aie.runtime_sequence @configure(%a: memref<16xi32>) {
      %addr = arith.constant 0x00110010 : i32
      %val = arith.constant 0x5EC0DE01 : i32
      aiex.npu.write32(%addr, %val) : i32, i32
    }
  }
}
