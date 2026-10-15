// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The AIE4 full-ELF control code is the program of one entry device:
// --device-name, or "main" by default. Every runtime sequence of the entry
// device is a job of that program, unless --sequence-name keeps just one.
// full_elf_config.json must describe exactly that program -- one kernel, for
// the entry device, with one instance per job.

// RUN: rm -rf aie4_full_elf_entry_prj && mkdir -p aie4_full_elf_entry_prj
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_entry_prj/all --output-dir=aie4_full_elf_entry_prj/all %s
// RUN: FileCheck %s --check-prefix=ALL-ASM --input-file=aie4_full_elf_entry_prj/all/full_elf_cert.mlir.asm
// RUN: FileCheck %s --check-prefix=ALL-JSON --input-file=aie4_full_elf_entry_prj/all/full_elf_config.json

// RUN: %aiecc --get-full-elf --dump-intermediates --sequence-name=first --tmpdir=aie4_full_elf_entry_prj/seq --output-dir=aie4_full_elf_entry_prj/seq %s
// RUN: FileCheck %s --check-prefix=SEQ-ASM --input-file=aie4_full_elf_entry_prj/seq/full_elf_cert.mlir.asm
// RUN: FileCheck %s --check-prefix=SEQ-JSON --input-file=aie4_full_elf_entry_prj/seq/full_elf_config.json

// RUN: %aiecc --get-full-elf --dump-intermediates --device-name=other --tmpdir=aie4_full_elf_entry_prj/dev --output-dir=aie4_full_elf_entry_prj/dev %s
// RUN: FileCheck %s --check-prefix=DEV-ASM --input-file=aie4_full_elf_entry_prj/dev/full_elf_cert.mlir.asm
// RUN: FileCheck %s --check-prefix=DEV-JSON --input-file=aie4_full_elf_entry_prj/dev/full_elf_config.json

// RUN: not %aiecc --get-full-elf --sequence-name=missing --tmpdir=aie4_full_elf_entry_prj/bad-seq --output-dir=aie4_full_elf_entry_prj/bad-seq %s 2>&1 | FileCheck %s --check-prefix=BAD-SEQ
// RUN: not %aiecc --get-full-elf --device-name=missing --tmpdir=aie4_full_elf_entry_prj/bad-dev --output-dir=aie4_full_elf_entry_prj/bad-dev %s 2>&1 | FileCheck %s --check-prefix=BAD-DEV

// ALL-ASM: WRITE_32 {{.*}}0x11111111
// ALL-ASM: WRITE_32 {{.*}}0x22222222
// ALL-ASM-NOT: 0x33333333
// ALL-JSON-NOT: "other"
// ALL-JSON: "id": "first"
// ALL-JSON: "id": "second"
// ALL-JSON: "name": "main"
// ALL-JSON-NOT: "other"

// SEQ-ASM: WRITE_32 {{.*}}0x11111111
// SEQ-ASM-NOT: 0x22222222
// SEQ-ASM-NOT: 0x33333333
// SEQ-JSON-NOT: "second"
// SEQ-JSON-NOT: "other"
// SEQ-JSON: "id": "first"
// SEQ-JSON-NOT: "second"
// SEQ-JSON: "name": "main"
// SEQ-JSON-NOT: "other"

// DEV-ASM-NOT: 0x11111111
// DEV-ASM-NOT: 0x22222222
// DEV-ASM: WRITE_32 {{.*}}0x33333333
// DEV-JSON-NOT: "main"
// DEV-JSON-NOT: "first"
// DEV-JSON: "id": "third"
// DEV-JSON: "name": "other"
// DEV-JSON-NOT: "main"

// BAD-SEQ: no runtime sequence matching --sequence-name "missing" in device "main"
// BAD-DEV: no device found matching --device-name "missing"

module {
  aie.device(npu3) @main {
    aie.runtime_sequence @first() {
      %addr = arith.constant 393216 : i32
      %value = arith.constant 286331153 : i32
      aiex.npu.write32(%addr, %value) : i32, i32
    }
    aie.runtime_sequence @second() {
      %addr = arith.constant 393220 : i32
      %value = arith.constant 572662306 : i32
      aiex.npu.write32(%addr, %value) : i32, i32
    }
  }
  aie.device(npu3) @other {
    aie.runtime_sequence @third() {
      %addr = arith.constant 393224 : i32
      %value = arith.constant 858993459 : i32
      aiex.npu.write32(%addr, %value) : i32, i32
    }
  }
}
