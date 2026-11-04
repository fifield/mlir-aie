//===- per_site_sections_assemble.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A -> B -> A. aiebu rejects two LOAD_PDIs of one label in a column, so
// aie-cert-per-site-sections gives each site its own section, label and file.
// The two copies of @a are the same file content and keep one PDI id; the
// result assembles both as plain aie4 control code and as the aie4_config
// full ELF that aiecc builds.

// REQUIRES: aiebu_asm
// RUN: rm -rf per_site_sections_assemble_prj && mkdir -p per_site_sections_assemble_prj
// RUN: aie-opt -cert-legalize-pages -aie-cert-per-site-sections %s | aie-translate -aie-cert-to-asm --cert-section-dir=per_site_sections_assemble_prj -o per_site_sections_assemble_prj/ctrl.asm
// RUN: FileCheck %s --input-file=per_site_sections_assemble_prj/ctrl.asm
// RUN: diff per_site_sections_assemble_prj/a__site0.asm per_site_sections_assemble_prj/a__site1.asm
// RUN: echo '{"xrt-kernels":[{"name":"main","arguments":[{"name":0,"type":"char *","offset":"0x0"}],"instance":[{"id":"seq","ctrl_code_file":"ctrl.asm"}]}]}' > per_site_sections_assemble_prj/config.json
// RUN: cd per_site_sections_assemble_prj && aiebu-asm -t aie4 -c ctrl.asm -L . -o ctrl.elf && aiebu-asm -t aie4_config -j config.json -o config.elf

// CHECK: a__site0:
// CHECK-NEXT: .include a__site0.asm
// CHECK: a__site1:
// CHECK-NEXT: .include a__site1.asm
// CHECK: b__site0:
// CHECK-NEXT: .include b__site0.asm
// CHECK: LOAD_PDI 11, @a__site0
// CHECK: LOAD_PDI 12, @b__site0
// CHECK: LOAD_PDI 11, @a__site1

module {
  aie.device(npu3) {
    aiex.cert.job(1) {
      aiex.cert.load_pdi(11, @a)
    }
    aiex.cert.job(2) {
      aiex.cert.load_pdi(12, @b)
    }
    aiex.cert.job(3) {
      aiex.cert.load_pdi(11, @a)
    }
    aiex.cert.section @a {
      aiex.cert.page {
        aiex.cert.job(5) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
    aiex.cert.section @b {
      aiex.cert.page {
        aiex.cert.job(6) {
          aiex.cert.write32(4096, 2)
        }
      }
    }
  }
}
