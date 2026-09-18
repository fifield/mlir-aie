//===- cert_to_asm_section_owns_chains.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// A cert.section's body is written to its own <section>.asm and pulled back in
// with `.include`. An assembler resolves a label only within the file that
// defines it, so a uC-DMA chain named from a section's text has to be defined
// in that same file. It used to be emitted into the main assembly's data
// region instead, and `aiebu-asm` rejected the result with
//   Error: Label not found 1:chain_8
// on every module with more than one device.
//
// Every section gets its OWN COPY of every chain it names, plus the blockwrite
// data those BDs address. Copying rather than moving is what makes a shared
// chain work: the label is file-scoped, so a copy per file is not a duplicate
// definition, and both referrers can resolve it. The main assembly keeps its
// copy only when something outside a section names the chain too.
//
// A chain is named by the synchronous AND the asynchronous write-des op; only
// counting the synchronous one would move a chain out from under an async
// reference in the main stream.
//
// RUN: rm -rf %t.d && mkdir -p %t.d && cd %t.d
// RUN: aie-translate -aie-cert-to-asm %s -o %t.d/main.asm
// RUN: FileCheck %s --check-prefix=MAIN < %t.d/main.asm
// RUN: FileCheck %s --check-prefix=SOLE < %t.d/sole_owner.asm
// RUN: FileCheck %s --check-prefix=SHARER < %t.d/sharer_a.asm
// RUN: FileCheck %s --check-prefix=MAINREF < %t.d/main.asm
// RUN: FileCheck %s --check-prefix=NOCHAIN < %t.d/no_chains.asm

// @only_here is named by @sole_owner and by nobody else, so it leaves the main
// assembly entirely -- as does the blockwrite data its BD addresses.
// MAIN-NOT: only_here:
// MAIN-NOT: @moved_data

// @shared_chain is named by two sections and by nothing outside one, so BOTH
// sections carry it and the main assembly drops it.
// MAIN-NOT: shared_chain:

// The sole owner carries the chain and the data, after its own EOF so that
// both are in its data region.
// SOLE: uC_DMA_WRITE_DES_SYNC  @only_here
// SOLE: EOF
// SOLE: only_here:
// SOLE: UC_DMA_BD {{.*}}@moved_data
// SOLE: moved_data:

// A sharer both references AND defines the shared chain, so its file resolves
// on its own. Before this it referenced a label defined only in main, and
// assembly failed with "Label not found".
// SHARER: uC_DMA_WRITE_DES_SYNC  @shared_chain
// SHARER: shared_chain:
// SHARER: kept_data:

// @also_named_by_main is named by one section AND by the device's own job --
// and by the ASYNCHRONOUS write-des op, which the ownership analysis has to
// count as well. Main keeps its definition, or that job loses its target.
// MAINREF-DAG: also_named_by_main:
// MAINREF-DAG: UC_DMA_WRITE_DES 

// A section that names no chain at all still gets a well-formed file: text,
// its EOF, and no stray chain header.
// NOCHAIN: START_JOB 4
// NOCHAIN: EOF
// NOCHAIN-NOT: data (chains)

module {
  aie.device(npu3) {
    memref.global "private" constant @moved_data : memref<2xi32> = dense<[1, 2]>
    memref.global "private" constant @kept_data : memref<2xi32> = dense<[3, 4]>

    aiex.cert.uc_dma_chain @only_here {
      aiex.cert.uc_dma_bd @moved_data, 0x600400, 2, false
    }
    aiex.cert.uc_dma_chain @shared_chain {
      aiex.cert.uc_dma_bd @kept_data, 0x600500, 2, false
    }
    aiex.cert.uc_dma_chain @also_named_by_main {
      aiex.cert.uc_dma_bd @kept_data, 0x600600, 2, false
    }

    aiex.cert.section @sole_owner {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.uc_dma_write_des_sync(@only_here)
        }
      }
    }

    aiex.cert.section @sharer_a {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.uc_dma_write_des_sync(@shared_chain)
        }
      }
    }

    aiex.cert.section @sharer_b {
      aiex.cert.page {
        aiex.cert.job(3) {
          aiex.cert.uc_dma_write_des_sync(@shared_chain)
        }
      }
    }

    // Sole section referrer, but the main stream names it too -- must not move.
    aiex.cert.section @would_own_but_for_main {
      aiex.cert.page {
        aiex.cert.job(5) {
          aiex.cert.uc_dma_write_des_sync(@also_named_by_main)
        }
      }
    }

    // A section with no chains at all.
    aiex.cert.section @no_chains {
      aiex.cert.page {
        aiex.cert.job(4) {
          aiex.cert.write32(0x5000, 5)
        }
      }
    }

    aiex.cert.page {
      aiex.cert.job(9) {
        // Asynchronous form on purpose: the analysis must count it too.
        aiex.cert.uc_dma_write_des(3, @also_named_by_main)
      }
    }
  }
}
